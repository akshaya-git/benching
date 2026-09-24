#!/usr/bin/env python3
"""
Apple Silicon LLM Benchmark — backend (stdlib only, no pip installs).

Serves index.html, manages model framework lifecycle, runs harness tests,
streams live activity, and serves generated outputs (e.g. HTML Tetris pages).

Usage:  python3 server.py   →  open http://localhost:7090

Configuration lives in config.json (seeded from config.example.json on first
run): host/port, per-framework model + start command, harness paths. See
docs/CONFIG.md.
"""

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import discovery

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "outputs")
LOG_DIR = os.path.join(ROOT, "logs")
WORK_DIR = os.path.join(ROOT, "work")
RUNS_DIR = os.path.join(ROOT, "runs")
for _d in (OUTPUT_DIR, LOG_DIR, WORK_DIR, RUNS_DIR):
    os.makedirs(_d, exist_ok=True)

CONFIG_PATH = os.path.join(ROOT, "config.json")
CONFIG_EXAMPLE = os.path.join(ROOT, "config.example.json")

# ----------------------------------------------------------------------------
# Framework definitions — DEFAULTS. Override per machine in config.json:
# model, port, start_cmd, context window, weight size, notes. The benchmark
# waits for each server's /v1/models endpoint to answer before running tests,
# and shuts the process down (SIGTERM then SIGKILL) before the next framework.
# start_cmd entries may use {model} / {model_path} placeholders ({model_path}
# resolves to the local HF-cache snapshot path when the model is cached).
# ----------------------------------------------------------------------------
DEFAULT_FRAMEWORKS = {
    "omlx": {
        "name": "OMLX",
        "model_source": "hf",
        "notes": "Model, context (261000), max-tokens (32768) and reasoning=low are applied PER MODEL from ~/.omlx/model_settings.json (entry: mlx-community--Qwen3.8-27B-8bit) at request time — the server starts bare, discovers models from the HF cache, and each request's model id selects its settings entry. Same base weights as MLX-VLM/MLX-Serve. MTP speculative decoding is ON via the embedded mlx-vlm engine (vlm_mtp_enabled=true, vlm_mtp_draft_model=mlx-community--Qwen3.8-27B-MTP-8bit) — the same draft arrangement MLX-VLM uses, user-confirmed working; the plain mtp_enabled/draft_model keys were inert in A/B (18.2 vs 18.1 TGS). The Jundot oQ8e build was retired after QA 25 on LONG tasks — see run 20260923-110534.",
        "port": 7001,
        "model": "mlx-community--Qwen3.8-27B-8bit",
        "model_gb": 28, "ctx_tokens": 261000,
        "start_cmd": ["omlx", "serve", "--port", "7001"],
    },
    "mtplx": {
        "name": "MTPLX",
        # model_source "mtplx": models live in ~/.mtplx/models (a different
        # format than HF/MLX), so the model picker shows only those for MTPLX.
        "model_source": "mtplx",
        "notes": "All parameters are CLI flags applied at server start (this command). --reasoning-effort low applies to every request; the served model id is normalized (e.g. mtplx-qwen38-27b-optimized-quality) and auto-adopted. Models are loaded from ~/.mtplx/models by repo id ({repo}); pick one in the Model panel.",
        "port": 7002,
        "model": "mtplx-qwen38-27b-optimized-quality",
        # repo: the HF repo behind the normalized served id; also the id used
        # for the CLI --model flag and the ~/.mtplx/models directory.
        "repo": "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality",
        "model_gb": 28, "ctx_tokens": 261000,
        "start_cmd": ["mtplx", "serve",
                      "--model", "{repo}",
                      "--context-window", "261000",
                      "--max-tokens", "32768",
                      "--reasoning-effort", "low",
                      "--port", "7002"],
    },
    # No MTP support in mlx-lm, so this runs the plain 8-bit conversion of the
    # same base model — a useful "reference runtime" row rather than a like-
    # for-like MTP comparison. Context length follows the model config; there
    # is no server-side context flag in mlx-lm. --max-tokens here is the
    # server-side cap (its default of 8192 would truncate thinking models).
    "mlxlm": {
        "name": "MLX-VLM",
        "model_source": "hf",
        "notes": "All parameters are CLI flags applied at server start (this command): MTP speculative decoding via the Qwen3.8-MTP-8bit draft, max-tokens 32768. Context follows the model config (no flag on mlx_vlm; serves 262144). Per-request metrics come from its JSON /metrics. Uses python3 from PATH — point it at the interpreter that has mlx_vlm installed if yours differs.",
        "port": 7003,
        "model": "mlx-community/Qwen3.8-27B-8bit",
        "model_gb": 28, "ctx_tokens": 261000,
        # MTP speculative decoding via the Qwen3.8 MTP draft — comparable (or
        # faster) decode vs the other frameworks. No ctx flag on this server;
        # context follows the model config (262144).
        "start_cmd": ["python3",
                      "-m", "mlx_vlm.server",
                      "--model", "mlx-community/Qwen3.8-27B-8bit",
                      "--draft-model", "mlx-community/Qwen3.8-27B-MTP-8bit",
                      "--draft-kind", "mtp",
                      "--max-tokens", "32768",
                      "--port", "7003"],
    },
    # mlx-serve: vLLM-compatible server with the richest metrics surface of
    # all (Prometheus /metrics: TTFT/prefill/decode histograms, prefix cache,
    # speculative-decode counters). Preferred instance is GUI-started with
    # MTP (the benchmark reuses whatever is healthy on :7004 and adopts its
    # REAL context length from /v1/models); this start_cmd is the cold-start
    # fallback with PLD (on by default).
    "mlxserve": {
        "name": "MLX-Serve",
        "model_source": "hf",
        "notes": "--max-tokens/--reasoning-budget are request defaults set at start; --ctx-size is OVERRIDDEN by ~/.mlx-serve/model-settings.json (ctx_size, set to 261000) — the file wins. --metrics enables the Prometheus surface (on by default in the GUI, not the CLI). PLD speculative decoding is on by default.",
        "port": 7004,
        "model": "mlx-community/Qwen3.8-27B-8bit",
        "model_gb": 28, "ctx_tokens": 261000,
        # --metrics: Prometheus surface (on by default in the GUI, not CLI).
        # Served ctx follows ~/.mlx-serve/model-settings.json (ctx_size),
        # overriding --ctx-size — the benchmark adopts the served value.
        "start_cmd": ["mlx-serve", "--model", "mlx-community/Qwen3.8-27B-8bit",
                      "--serve", "--host", "127.0.0.1", "--metrics",
                      "--port", "7004", "--ctx-size", "261000",
                      "--max-tokens", "32768", "--reasoning-budget", "1024"],
    },
}

DEFAULT_CONFIG = {
    "host": "127.0.0.1",
    "port": 7090,
    "route_via_proxy": False,
    "pi_thinking": "",
    "hart_path": os.path.join(ROOT, "hart", "hart.py"),
    "frameworks": DEFAULT_FRAMEWORKS,
}


def _merge_config(base, override):
    """Shallow-merge override onto base; frameworks merged per-framework."""
    out = dict(base)
    for k, v in (override or {}).items():
        if k == "frameworks" and isinstance(v, dict):
            fws = dict(out.get("frameworks") or {})
            for fw, fwc in v.items():
                if fw in fws and isinstance(fwc, dict):
                    fws[fw] = {**fws[fw], **fwc}
                else:
                    fws[fw] = fwc
            out["frameworks"] = fws
        else:
            out[k] = v
    return out


def load_config():
    """Load config.json over the built-in defaults. On first run, seed
    config.json from config.example.json so users can discover and edit it."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy of defaults
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                cfg = _merge_config(cfg, json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠ config.json unreadable ({e}) — using built-in defaults",
                  flush=True)
    else:
        try:
            src = CONFIG_EXAMPLE if os.path.isfile(CONFIG_EXAMPLE) else None
            if src:
                with open(src) as f:
                    with open(CONFIG_PATH, "w") as out:
                        out.write(f.read())
            else:
                with open(CONFIG_PATH, "w") as f:
                    json.dump(DEFAULT_CONFIG, f, indent=2)
        except OSError:
            pass
    return cfg


def save_config():
    """Persist the live config (e.g. after a model selection) to config.json."""
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(CONFIG, f, indent=2)
        return True
    except OSError as e:
        log(f"could not save config.json: {e}", level="err")
        return False


CONFIG = load_config()
FRAMEWORKS = CONFIG["frameworks"]


def resolve_start_cmd(fw_cfg):
    """Expand {model} / {repo} / {model_path} placeholders in a framework's
    start_cmd.
      {model}      → the served/request model id (cfg["model"])
      {repo}       → the HF repo id (cfg["repo"] or cfg["model"]); MTPLX's CLI
                     --model flag needs the repo, not the normalized served id
      {model_path} → the local HF-cache snapshot path when the repo is cached
                     (some CLIs need a real path), else the model id"""
    repo = fw_cfg.get("repo") or fw_cfg.get("model", "")
    model_path = discovery.snapshot_dir(repo)
    cmd = []
    for part in fw_cfg.get("start_cmd", []):
        if part == "{model}":
            cmd.append(fw_cfg["model"])
        elif part == "{repo}":
            cmd.append(repo)
        elif part == "{model_path}":
            cmd.append(model_path or fw_cfg["model"])
        else:
            cmd.append(part)
    return cmd

# 5 benchmark tasks — prompts that produce runnable/inspectable output.
TASKS = [
    {"id": "tetris", "name": "HTML Tetris game", "artifact": "tetris.html",
     "prompt": "Create a complete, playable Tetris game in a single HTML file with embedded CSS and JavaScript. Include score, level, next-piece preview, and keyboard controls. Return only the full HTML file in one ```html code block."},
    {"id": "snake", "name": "HTML Snake game", "artifact": "snake.html",
     "prompt": "Create a complete, playable Snake game in a single HTML file with embedded CSS and JavaScript. Include score display, increasing speed, and arrow-key controls. Return only the full HTML file in one ```html code block."},
    {"id": "pong", "name": "HTML Pong vs AI", "artifact": "pong.html",
     "prompt": "Create a complete Pong game in a single HTML file where the player plays against a simple AI paddle. Include score display and mouse/keyboard control. Return only the full HTML file in one ```html code block."},
    {"id": "todo", "name": "Todo app (localStorage)", "artifact": "todo.html",
     "prompt": "Create a complete Todo list web app in a single HTML file with embedded CSS and JavaScript. Support adding, completing, deleting items and persisting to localStorage. Return only the full HTML file in one ```html code block."},
    {"id": "fib", "name": "Explain + code: Fibonacci",
     "prompt": "Explain memoized Fibonacci in plain English, then give a working Python implementation with a quick test snippet. Keep it under 300 words of explanation."},
    # ---- long-horizon tasks (60-120 min class; industry benchmark families) ----
    {"id": "chip8", "name": "LONG · CHIP-8 emulator (industry core)", "long": True,
     "artifact": "chip8.html",
     "prompt": "Build a complete CHIP-8 emulator in a single HTML file (canvas display, keyboard mapping to the 16-key hex keypad, beeper via Web Audio). Implement ALL 35 standard opcodes (00E0, 00EE, 1NNN, 2NNN, 3XNN, 4XNN, 5XY0, 6XNN, 7XNN, 8XY0-8XY7, 8XYE, 9XY0, ANNN, BNNN, CXNN, DXYN, EX9E, EXA1, FX07, FX0A, FX15, FX18, FX1E, FX29, FX33, FX55, FX65), 60Hz delay/sound timers, correct sprite drawing with XOR and VF clipping. Include a built-in self-test panel that verifies opcode implementations and a small embedded test ROM that draws a pattern and reports pass/fail per opcode group on screen."},
    {"id": "raytracer", "name": "LONG · Ray tracer (Ray Tracing in One Weekend)", "long": True,
     "artifact": "raytracer.html",
     "prompt": "Build a ray tracer in a single HTML file following the 'Ray Tracing in One Weekend' curriculum: vec3 math library, ray-sphere intersection, camera with anti-aliasing, diffuse (Lambertian) materials with recursive scatter, metal materials with fuzz, dielectric glass with Schlick approximation, and defocus blur (thin lens). Render progressively to a canvas (start low-res, refine), with a scene of the classic three large spheres (ground, center diffuse, left metal, right glass) plus ~30 small random spheres. Include a progress bar, render-time display, adjustable samples-per-pixel, and a built-in unit-test panel that verifies vec3 operations and intersection math with pass/fail output."},
    {"id": "spreadsheet", "name": "LONG · Spreadsheet with formula engine", "long": True,
     "artifact": "spreadsheet.html",
     "prompt": "Build a working spreadsheet in a single HTML file: 26 columns x 50 rows, click-to-select, arrow-key navigation, in-cell editing, and a real formula engine supporting =A1+B2 cell references and ranges (A1:B5), arithmetic with correct operator precedence, parentheses, and the functions SUM, AVERAGE, MIN, MAX, COUNT, IF, ROUND, ABS. Recompute dependents on edit via a dependency graph with cycle detection (show #CYCLE!), and persist the sheet to localStorage. Include a built-in test panel that runs at least 15 formula test cases (references, ranges, nesting, cycles) and prints pass/fail per case."},
    {"id": "markdown", "name": "LONG · Markdown compiler + spec tests", "long": True,
     "artifact": "markdown.html",
     "prompt": "Build a Markdown compiler in a single HTML file: a two-pane editor (raw markdown left, live rendered HTML right) with a CommonMark-subset engine written from scratch (no libraries): ATX headings, paragraphs, bold/italic, inline code, fenced code blocks with language labels, links, images, unordered/ordered lists with nesting, blockquotes, horizontal rules, and tables. Include a built-in spec test panel with at least 20 test cases (input to expected HTML) covering edge cases like nested lists, unclosed fences, and emphasis inside code; run them on load and show pass/fail counts. Persist editor content to localStorage."},
    {"id": "conduit", "name": "LONG · RealWorld Conduit blog (SPA)", "long": True,
     "artifact": "index.html",
     "prompt": "Build the RealWorld 'Conduit' blogging platform as a single-file SPA (the industry-standard RealWorld demo app spec): hash-based routing with pages Home (global feed with pagination), Article (markdown-rendered body, tags), Sign in / Sign up (with validation errors), Editor (create/edit articles with tag input), Settings, and Profile (my articles / favorited articles). Persist users, articles, favorites, and comments to localStorage acting as the backend, seeded with 5 sample articles and 2 users. Include favoriting, commenting, and authenticated navigation state. Everything client-side, no server calls."},
]

RAWPLUS_MAX_ROUNDS = 4  # 4 x per-chunk cap; context 261k holds prompt+output

HARNESS_LABELS = {
    "raw": "Raw (api)",
    "rawplus": "raw+",
    "pi": "pi",
    "opencode": "opencode",
    "goose": "Goose",
    "hart": "hart",
}

# hart harness location — bundled in this repo under hart/ (so a fresh clone
# works out of the box); override via "hart_path" in config.json. Relative
# paths resolve against the repo root; ~ is expanded.
_hart_cfg = CONFIG.get("hart_path", os.path.join(ROOT, "hart", "hart.py"))
HART_PATH = os.path.expanduser(
    _hart_cfg if os.path.isabs(_hart_cfg) else os.path.join(ROOT, _hart_cfg))

# Routing for agent harnesses (pi/opencode/goose). True = via the local
# measurement proxy, which adds per-request PP/TGS/TTFT to their rows.
# False = straight to the framework, byte-identical to an independent harness
# run (agent rows then report TPS/QA only — raw keeps full metrics either way).
ROUTE_VIA_PROXY = bool(CONFIG.get("route_via_proxy", False))

# pi thinking-level suffix for --model (pi supports :off…:xhigh). Empty =
# inherit the server's reasoning setting like every other harness, so all
# cells share one uniform budget.
PI_THINKING = CONFIG.get("pi_thinking", "")


def agent_base_url(fw):
    port = PROXY_PORT if ROUTE_VIA_PROXY else FRAMEWORKS[fw]["port"]
    return f"http://127.0.0.1:{port}/v1"


def prompt_for_harness(prompt, artifact, harness):
    """Agent harnesses must deliver files, not chat text — the raw prompt's
    “return only the HTML in one code block” instruction makes them answer
    inline and stall instead of using their write tools."""
    if harness in ("raw", "rawplus") or not artifact:
        return prompt
    # Drop any sentence telling the model to answer inline in a code block —
    # exact canonical phrasing first, then any edited variant.
    base = re.sub(r"Return only the full HTML file in one ```html code block\.\s*$",
                  "", prompt)
    base = re.sub(r"[^.!?\n]*```html code block[^.!?\n]*[.!\n]?\s*", "", base).strip()
    return (base + f"\n\nDo not print the file contents in chat. Write the complete, "
            f"self-contained file to '{artifact}' in the current working directory; "
            f"it must run by simply opening it in a browser.")

# ----------------------------------------------------------------------------
# Shared state
# ----------------------------------------------------------------------------
STATE = {
    "running": False,
    "current_step": "",
    "run_started": None,
    "framework_status": {fw: "down" for fw in FRAMEWORKS},
    "results": [],
}
ACTIVITY = []          # list of {ts, msg, fw, harness, level}
ACTIVITY_MAX = 1500
RUN_FLAG = threading.Event()
LOCK = threading.Lock()
PROCS = {}             # fw -> subprocess.Popen
FW_LOGS = {}           # fw -> open log file for the framework's stdout
CURRENT = {"proc": None, "sock": None}

# verbose per-cell tail: every harness output line + synthetic freshness
# heartbeats, served to the UI for second-level stuck-vs-working decisions
CELL_TAIL = {"lines": [], "active": False, "last_line_ts": 0.0, "started": 0.0,
             "cell": ""}
TAIL_MAX = 80


def tail_reset(cell):
    with LOCK:
        CELL_TAIL.update({"lines": [], "active": True, "last_line_ts": time.time(),
                          "started": time.time(), "cell": cell})


def tail_add(line):
    with LOCK:
        CELL_TAIL["lines"].append({"ts": time.time(), "text": line[:220]})
        CELL_TAIL["last_line_ts"] = time.time()
        if len(CELL_TAIL["lines"]) > TAIL_MAX:
            del CELL_TAIL["lines"][: len(CELL_TAIL["lines"]) - TAIL_MAX]


def tail_snapshot(n=20):
    with LOCK:
        return {"active": CELL_TAIL["active"], "cell": CELL_TAIL["cell"],
                "lines": CELL_TAIL["lines"][-n:],
                "since_line": round(time.time() - CELL_TAIL["last_line_ts"], 1)
                if CELL_TAIL["active"] else None,
                "elapsed": round(time.time() - CELL_TAIL["started"], 0)
                if CELL_TAIL["active"] else None}   # in-flight work, killable by Stop
CUR_LOCK = threading.Lock()


class RunStopped(Exception):
    """Raised inside a benchmark worker when the user pressed Stop."""


def request_stop():
    """Stop the run: clear the flag AND actively interrupt whatever cell is
    in flight (kill the CLI process group, shut the streaming socket) so the
    worker threads wake immediately instead of at their next checkpoint."""
    RUN_FLAG.clear()
    with CUR_LOCK:
        proc, sock = CURRENT["proc"], CURRENT["sock"]
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    if sock:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass


def log(msg, fw=None, harness=None, level=""):
    with LOCK:
        ACTIVITY.append({"ts": time.time(), "msg": msg, "fw": fw,
                         "harness": harness, "level": level})
        if len(ACTIVITY) > ACTIVITY_MAX:
            del ACTIVITY[: len(ACTIVITY) - ACTIVITY_MAX]
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{fw or 'sys'}] {msg}"
    print(line, flush=True)
    try:
        with open(os.path.join(LOG_DIR, "bench.log"), "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ----------------------------------------------------------------------------
# System helpers
# ----------------------------------------------------------------------------
_TOTAL_MEM = None
_RAM_CACHE = {"t": 0.0, "free": 0, "total": 0}


def _total_mem():
    """Total RAM in bytes — computed once (macOS sysctl / Linux meminfo)."""
    global _TOTAL_MEM
    if _TOTAL_MEM is None:
        _TOTAL_MEM = 0
        try:
            if sys.platform == "darwin":
                _TOTAL_MEM = int(subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=5).stdout.strip())
            elif os.path.isfile("/proc/meminfo"):
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            _TOTAL_MEM = int(line.split()[1]) * 1024
                            break
        except Exception:
            _TOTAL_MEM = 0
    return _TOTAL_MEM


def free_ram():
    """Return (free_bytes, total_bytes). macOS: vm_stat; Linux: /proc/meminfo.
    Cached for 3s — the UI polls /api/state every ~1.2s and this used to
    spawn two subprocesses per poll."""
    now = time.time()
    if _RAM_CACHE["total"] and now - _RAM_CACHE["t"] < 3.0:
        return _RAM_CACHE["free"], _RAM_CACHE["total"]
    total = _total_mem()
    free = 0
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["vm_stat"], capture_output=True,
                                 text=True, timeout=5).stdout
            ps = int(re.search(r"page size of (\d+) bytes", out).group(1))
            free = (int(re.search(r"Pages free:\s+(\d+)", out).group(1))
                    + int(re.search(r"Pages speculative:\s+(\d+)", out).group(1))
                    + int(re.search(r"Pages inactive:\s+(\d+)", out).group(1))) * ps
        elif os.path.isfile("/proc/meminfo"):
            with open("/proc/meminfo") as f:
                mi = {}
                for line in f:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        mi[k] = int(v.split()[0]) * 1024
            free = mi.get("MemAvailable", mi.get("MemFree", 0))
    except Exception:
        free = 0
    _RAM_CACHE.update({"t": now, "free": free, "total": total})
    return free, total


def port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def framework_healthy(fw):
    """Framework is up when its OpenAI-compatible /v1/models answers."""
    cfg = FRAMEWORKS[fw]
    if not port_open(cfg["port"]):
        return False
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{cfg['port']}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def _get_json(url, timeout=4):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _prom_scrape(text):
    """Parse Prometheus text exposition into {name: value}. Labels are
    stripped and summed across series (all metrics we read are counters or
    histogram _sum/_count, for which summing is correct); _bucket lines are
    dropped."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        name, val = parts
        if "{" in name:
            name = name.split("{", 1)[0]
        if name.endswith("_bucket"):
            continue
        try:
            out[name] = out.get(name, 0.0) + float(val)
        except ValueError:
            pass
    return out


def server_snapshot(fw):
    """Server-side metric snapshot. OMLX: totals + split prefill/generation
    seconds. MTPLX: per-request 'recent' records. MLX-LM: none (client-side
    only). These give authoritative PP/TGS for ANY harness — including pi,
    whose non-streaming calls hide timing from clients."""
    cfg = FRAMEWORKS[fw]
    if fw == "omlx":
        s = _get_json(f"http://127.0.0.1:{cfg['port']}/admin/api/stats")
        u = _get_json(f"http://127.0.0.1:{cfg['port']}/admin/api/usage")
        if s is None:
            return None
        t = (u or {}).get("totals", {})
        return {"kind": "omlx",
                "prompt": s.get("total_prompt_tokens") or 0,
                "completion": s.get("total_completion_tokens") or 0,
                "pre_s": t.get("prefill_seconds") or 0.0,
                "gen_s": t.get("generation_seconds") or 0.0,
                "cache_eff": s.get("cache_efficiency")}
    if fw == "mlxserve":
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{cfg['port']}/metrics", timeout=4) as r:
                m = _prom_scrape(r.read().decode("utf-8", "replace"))
        except Exception:
            return None
        if "vllm:prompt_tokens_total" not in m:
            return None
        return {"kind": "mlxserve",
                "prompt": m.get("vllm:prompt_tokens_total", 0),
                "completion": m.get("vllm:generation_tokens_total", 0),
                "cache": m.get("mlx_serve:prefix_cache_tokens_total", 0),
                "pre_s": m.get("vllm:request_prefill_time_seconds_sum", 0.0),
                "gen_s": m.get("vllm:request_decode_time_seconds_sum", 0.0),
                "ttft_s": m.get("vllm:time_to_first_token_seconds_sum", 0.0),
                "ttft_n": m.get("vllm:time_to_first_token_seconds_count", 0)}
    if fw == "mlxlm":  # mlx_vlm.server: JSON /metrics with per-request records
        m = _get_json(f"http://127.0.0.1:{cfg['port']}/metrics")
        if m is None:
            return None
        s = m.get("summary") or {}
        return {"kind": "mlxvlm",
                "prompt": s.get("prompt_tokens_total") or 0,
                "completion": s.get("completion_tokens_total") or 0,
                "recent": m.get("recent") or []}
    if fw == "mtplx":
        m = _get_json(f"http://127.0.0.1:{cfg['port']}/metrics")
        if m is None:
            return None
        return {"kind": "mtplx", "recent": m.get("recent") or [],
                "latest": m.get("latest")}
    return None


def server_cell_delta(fw, before, after):
    """Per-cell server-side metrics from a before/after snapshot diff."""
    if not before or not after or before.get("kind") != after.get("kind"):
        return {}
    out = {}
    if before["kind"] == "omlx":
        dp = after["prompt"] - before["prompt"]
        dc = after["completion"] - before["completion"]
        dpre = max(after["pre_s"] - before["pre_s"], 0.0)
        dgen = max(after["gen_s"] - before["gen_s"], 0.0)
        # usage counters flush every ~5s — a diff window that small yields
        # garbage rates (observed 134k tok/s). Require a real timing window
        # and clamp to physically plausible ranges.
        if dp > 0 and dpre >= 2.0:
            pp = dp / dpre
            if pp <= 10000:
                out["server_pp"] = round(pp, 1)
        if dc > 0 and dgen >= 2.0:
            tgs = dc / dgen
            if tgs <= 500:
                out["server_tgs"] = round(tgs, 1)
        out["server_prompt_tokens"] = dp
        out["server_completion_tokens"] = dc
    elif before["kind"] == "mlxserve":
        dp = after["prompt"] - before["prompt"]
        dc = after["completion"] - before["completion"]
        dpre = max(after["pre_s"] - before["pre_s"], 0.0)
        dgen = max(after["gen_s"] - before["gen_s"], 0.0)
        dttft = after["ttft_s"] - before["ttft_s"]
        dtn = after["ttft_n"] - before["ttft_n"]
        if dp > 0 and dpre > 0.01:
            out["server_pp"] = round(dp / dpre, 1)
        if dc > 0 and dgen > 0.01:
            out["server_tgs"] = round(dc / dgen, 1)
        if dtn > 0 and dttft > 0:
            out["server_ttft_avg"] = round(dttft / dtn, 3)
        out["server_prompt_tokens"] = dp
        out["server_completion_tokens"] = dc
        out["server_cached_tokens"] = after["cache"] - before["cache"]
    elif before["kind"] == "mlxvlm":
        n0 = {json.dumps(r, sort_keys=True) for r in before["recent"]}
        fresh = [r for r in after["recent"]
                 if json.dumps(r, sort_keys=True) not in n0]
        pps = [r["prefill_tok_s"] for r in fresh
               if isinstance(r.get("prefill_tok_s"), (int, float))]
        tgss = [r["decode_tok_s"] for r in fresh
                if isinstance(r.get("decode_tok_s"), (int, float))]
        ttfts = [r["ttft_s"] for r in fresh
                 if isinstance(r.get("ttft_s"), (int, float))]
        if pps:
            out["server_pp"] = round(sum(pps) / len(pps), 1)
        if tgss:
            out["server_tgs"] = round(sum(tgss) / len(tgss), 1)
        if ttfts:
            out["server_ttft_avg"] = round(sum(ttfts) / len(ttfts), 3)
        out["server_prompt_tokens"] = after["prompt"] - before["prompt"]
        out["server_completion_tokens"] = after["completion"] - before["completion"]
        out["server_requests"] = len(fresh)
    elif before["kind"] == "mtplx":
        n0 = {json.dumps(r, sort_keys=True) for r in before["recent"]}
        fresh = [r for r in after["recent"]
                 if json.dumps(r, sort_keys=True) not in n0]
        out["server_requests"] = len(fresh)
        pps, tgss, tts = [], [], []
        for r in fresh:
            for pk in ("prefill_tok_s", "prompt_tps", "prefill_tps", "pp_tps"):
                v = r.get(pk)
                if isinstance(v, (int, float)) and 0.5 < v < 10000:
                    pps.append(v); break
            for gk in ("decode_tok_s", "generation_tps", "gen_tps", "tgs"):
                v = r.get(gk)
                if isinstance(v, (int, float)) and 0.5 < v < 500:
                    tgss.append(v); break
            if isinstance(r.get("ttft_s"), (int, float)):
                tts.append(r["ttft_s"])
        if tts:
            out["server_ttft_avg"] = round(sum(tts) / len(tts), 3)
        if pps:
            out["server_pp"] = round(sum(pps) / len(pps), 1)
        if tgss:
            out["server_tgs"] = round(sum(tgss) / len(tgss), 1)
    return out


def log_device_info(fw):
    """OMLX device-info: chip context for reports (one line, once per start)."""
    cfg = FRAMEWORKS[fw]
    d = _get_json(f"http://127.0.0.1:{cfg['port']}/admin/api/device-info")
    if d:
        log(f"device: {d.get('chip_name')} {d.get('chip_variant')}, "
            f"{d.get('memory_gb')} GB, {d.get('gpu_cores')} GPU cores",
            fw=cfg["name"], level="ok")


def adopt_served_meta(fw):
    """Adopt runtime metadata from the served model entry (mlx-serve exposes
    context_length per model — GUI instances may differ from config)."""
    cfg = FRAMEWORKS[fw]
    if fw != "mlxserve":
        return
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{cfg['port']}/v1/models", timeout=5) as r:
            for entry in json.loads(r.read()).get("data", []):
                if entry.get("id") == cfg["model"] or entry.get("loaded"):
                    cl = entry.get("context_length")
                    if cl and cl != cfg.get("ctx_tokens"):
                        log(f"{cfg['name']} served context length {cl} "
                            f"(config had {cfg.get('ctx_tokens')}) — adopting",
                            fw=fw)
                        cfg["ctx_tokens"] = cl
                    break
    except Exception:
        pass


_FW_STATUS_TS = {"t": 0.0}


def refresh_fw_status_idle(max_age=4.0):
    """Re-check framework liveness for the UI while idle. The status map is
    otherwise a stale snapshot (set at startup / during runs), so externally
    killed servers kept showing green forever. TTL-cached: the UI polls
    /api/state every ~1.2s; dead ports refuse instantly, live ones are a
    cheap TCP connect (no HTTP GET — the run path uses framework_healthy)."""
    if time.time() - _FW_STATUS_TS["t"] < max_age:
        return
    _FW_STATUS_TS["t"] = time.time()
    for fw in FRAMEWORKS:
        if STATE["framework_status"].get(fw) != "starting":
            set_fw_status(fw, "up" if port_open(FRAMEWORKS[fw]["port"]) else "down")


def set_fw_status(fw, status):
    with LOCK:
        STATE["framework_status"][fw] = status


# ----------------------------------------------------------------------------
# Measurement proxy — agent harnesses (pi/opencode) point here instead of at
# the framework directly. Traffic is relayed untouched while per-request
# timing (TTFT / prefill / decode, from the SSE stream) is recorded, giving
# them the same PP/TGS metrics raw gets. Runs are sequential, so a single
# stats accumulator suffices.
# ----------------------------------------------------------------------------
PROXY_PORT = 7010
PROXY_STATE = {"target": None, "fw": None, "model": None}
PROXY_LOCK = threading.Lock()
PROXY_STATS = {}   # per-run aggregates (reset before each harness dispatch)
PROXY_TOTALS = {}  # cumulative since last manual clear (Proxy Inspector)
PROXY_LOG = []     # per-request records for reporting, capped
PROXY_LOG_MAX = 500
PROXY_ZERO = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
              "ttft_sum": 0.0, "decode_sum": 0.0, "wall_sum": 0.0, "length_hits": 0}
PROXY_STATS.update(PROXY_ZERO)
PROXY_TOTALS.update(PROXY_ZERO)


def proxy_reset():
    with PROXY_LOCK:
        PROXY_STATS.clear()
        PROXY_STATS.update({"requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
                            "ttft_sum": 0.0, "decode_sum": 0.0, "wall_sum": 0.0})


def proxy_read():
    with PROXY_LOCK:
        return dict(PROXY_STATS)


def _proxy_record(usage, t0, first_tok, last_tok, model=None, path="", stream=False, status=200,
                  finish=None):
    now = time.time()
    rec = {"ts": now, "path": path, "model": model, "stream": stream, "status": status,
           "finish": finish,
           "prompt_tokens": (usage or {}).get("prompt_tokens") or 0,
           "completion_tokens": (usage or {}).get("completion_tokens") or 0,
           "ttft": round(first_tok - t0, 3) if first_tok else None,
           "decode": round(last_tok - first_tok, 3)
                     if (first_tok and last_tok and last_tok > first_tok) else None,
           "wall": round(now - t0, 3)}
    with PROXY_LOCK:
        for store in (PROXY_STATS, PROXY_TOTALS):  # per-run + since-clear
            store["requests"] += 1
            store["prompt_tokens"] += rec["prompt_tokens"]
            store["completion_tokens"] += rec["completion_tokens"]
            if finish == "length":
                store["length_hits"] += 1
            if first_tok:
                store["ttft_sum"] += first_tok - t0
            if first_tok and last_tok and last_tok > first_tok:
                store["decode_sum"] += last_tok - first_tok
            store["wall_sum"] += now - t0
        PROXY_LOG.append(rec)
        if len(PROXY_LOG) > PROXY_LOG_MAX:
            del PROXY_LOG[: len(PROXY_LOG) - PROXY_LOG_MAX]


class ProxyHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _relay_error(self, e):
        try:
            payload = e.read()
        except Exception:
            payload = b"{}"
        self.send_response(e.code)
        ctype = e.headers.get("Content-Type", "application/json") if e.headers else "application/json"
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _relay(self):
        target = PROXY_STATE["target"]
        if not target:
            self.send_error(503, "no framework under test")
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() in ("content-type", "authorization", "accept")}
        t0 = time.time()
        try:
            payload = json.loads(body) if body else None
        except json.JSONDecodeError:
            payload = None
        model = payload.get("model") if isinstance(payload, dict) else None
        stream = bool(isinstance(payload, dict) and payload.get("stream"))
        # Ask streaming endpoints to include token usage; some clients omit it.
        injected = False
        if isinstance(payload, dict) and payload.get("stream") and "stream_options" not in payload:
            payload["stream_options"] = {"include_usage": True}
            body = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
            injected = True
        req = urllib.request.Request(target + self.path, data=body or None,
                                     headers=headers, method=self.command)
        try:
            resp = urllib.request.urlopen(req, timeout=1800)
        except urllib.error.HTTPError as e:
            if injected and e.code == 400:  # server rejects stream_options — retry plain
                payload.pop("stream_options", None)
                body = json.dumps(payload).encode()
                try:
                    resp = urllib.request.urlopen(urllib.request.Request(
                        target + self.path, data=body, headers=headers,
                        method=self.command), timeout=1800)
                except urllib.error.HTTPError as e2:
                    _proxy_record(None, t0, None, None, model, self.path, stream, e2.code)
                    self._relay_error(e2)
                    return
                except Exception as e2:
                    _proxy_record(None, t0, None, None, model, self.path, stream, 502)
                    self.send_error(502, str(e2))
                    return
            else:
                _proxy_record(None, t0, None, None, model, self.path, stream, e.code)
                self._relay_error(e)
                return
        except Exception:
            _proxy_record(None, t0, None, None, model, self.path, stream, 502)
            self.send_error(502, "relay failed")
            return

        ct = resp.headers.get("Content-Type", "")
        if "event-stream" in ct:
            self.send_response(resp.status)
            self.send_header("Content-Type", ct)
            self.end_headers()
            first = last = None
            usage = None
            finish = None
            try:
                for raw in resp:
                    self.wfile.write(raw)
                    self.wfile.flush()
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    d = line[5:].strip()
                    if d == "[DONE]":
                        break  # some servers keep-alive after DONE — stop reading
                    if not d:
                        continue
                    try:
                        chunk = json.loads(d)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                        last = time.time()
                    for ch in chunk.get("choices") or []:
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
                        delta = ch.get("delta") or {}
                        if delta.get("content") or delta.get("reasoning_content") or ch.get("text"):
                            now = time.time()
                            if first is None:
                                first = now
                            last = now
            except Exception:
                pass  # client disconnected mid-stream; keep what we measured
            _proxy_record(usage, t0, first, last, model, self.path, True, resp.status, finish)
        else:
            data = resp.read()
            usage = None
            finish = None
            try:
                j = json.loads(data)
                usage = j.get("usage")
                ch = (j.get("choices") or [{}])[0]
                finish = ch.get("finish_reason")
            except Exception:
                pass
            self.send_response(resp.status)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            _proxy_record(usage, t0, None, None, model, self.path, False, resp.status, finish)

    do_GET = _relay
    do_POST = _relay


def start_proxy():
    srv = ThreadingHTTPServer(("127.0.0.1", PROXY_PORT), ProxyHandler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()


# ----------------------------------------------------------------------------
# Framework lifecycle
# ----------------------------------------------------------------------------
def _adopt_served_model_id(fw):
    """Align cfg['model'] with the id the server actually serves. Frameworks
    like OMLX serve cache-style ids (org--name) while the config/picker use
    repo-style (org/name); match by normalized key so requests resolve. Falls
    back to adopting the single served id, else warns."""
    cfg = FRAMEWORKS[fw]
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{cfg['port']}/v1/models", timeout=5) as r:
            ids = [m.get("id") for m in json.loads(r.read()).get("data", [])]
    except Exception:
        return
    if not ids or cfg["model"] in ids:
        return
    match = next((i for i in ids
                  if discovery.normalize_key(i) == discovery.normalize_key(cfg["model"])),
                 None)
    if match:
        log(f"{cfg['name']} serves “{cfg['model']}” as “{match}” — adopting",
            fw=fw, level="ok")
        cfg["model"] = match
    elif len(ids) == 1:
        log(f"{cfg['name']} serves model id “{ids[0]}” — adopting", fw=fw, level="ok")
        cfg["model"] = ids[0]
    else:
        log(f"⚠ configured model “{cfg['model']}” not in {cfg['name']}'s list {ids} "
            f"— requests may fail", fw=fw, level="err")


def start_framework(fw):
    cfg = FRAMEWORKS[fw]
    if framework_healthy(fw):
        log(f"{cfg['name']} already running on port {cfg['port']} — reusing", fw=fw)
        _adopt_served_model_id(fw)
        adopt_served_meta(fw)
        set_fw_status(fw, "up")
        return True
    set_fw_status(fw, "starting")
    cmd = resolve_start_cmd(cfg)
    log(f"starting {cfg['name']}: {' '.join(cmd)}", fw=fw)
    # Server stdout/stderr goes to logs/<fw>-<ts>.log — a failed start is
    # diagnosable instead of silently discarded.
    try:
        logf = open(os.path.join(LOG_DIR, f"{fw}-{time.strftime('%Y%m%d-%H%M%S')}.log"), "w")
    except OSError:
        logf = subprocess.DEVNULL
    try:
        proc = subprocess.Popen(cmd, stdout=logf,
                                stderr=subprocess.STDOUT, start_new_session=True)
    except FileNotFoundError:
        log(f"{cfg['name']} CLI not found — check start_cmd in config.json", fw=fw, level="err")
        set_fw_status(fw, "down")
        return False
    with LOCK:
        PROCS[fw] = proc
        FW_LOGS[fw] = logf
    deadline = time.time() + 300  # allow up to 5 min for model load
    while time.time() < deadline and proc.poll() is None:
        if not RUN_FLAG.is_set():
            stop_framework(fw)
            return False
        if framework_healthy(fw):
            # Adopt the model id the server actually reports (e.g. OMLX serves
            # cache-style org--name, MTPLX a normalized id) and, for mlx-serve,
            # the served context length so ctx-fill % reflects the real window.
            _adopt_served_model_id(fw)
            adopt_served_meta(fw)
            set_fw_status(fw, "up")
            log(f"{cfg['name']} healthy on port {cfg['port']} ({cfg['model']})", fw=fw, level="ok")
            return True
        time.sleep(2)
    log(f"{cfg['name']} failed to become healthy", fw=fw, level="err")
    stop_framework(fw)
    return False


def stop_framework(fw):
    cfg = FRAMEWORKS[fw]
    PROXY_STATE.update({"target": None, "fw": None, "model": None})
    set_fw_status(fw, "down")
    with LOCK:
        proc = PROCS.pop(fw, None)
        logf = FW_LOGS.pop(fw, None)
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        log(f"{cfg['name']} shut down", fw=fw)
    elif port_open(cfg["port"]):
        # Server we didn't spawn (was already running) — leave it alone.
        log(f"{cfg['name']} was pre-existing on port {cfg['port']} — leaving it running", fw=fw)
    if logf:
        try:
            logf.close()
        except OSError:
            pass


# ----------------------------------------------------------------------------
# Harness execution
# ----------------------------------------------------------------------------
def _chat_request(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=7200)


def call_chat(fw, prompt, settings, messages=None):
    """Streaming OpenAI-compatible chat completion. Time-to-first-token gives
    prompt-processing (prefill) speed; the remaining wall time gives token
    generation (decode) speed. Returns (text, metrics_dict). `messages`
    overrides the default single-user-message prompt (used by raw+
    continuations)."""
    cfg = FRAMEWORKS[fw]
    url = f"http://127.0.0.1:{cfg['port']}/v1/chat/completions"
    body = {
        "model": cfg["model"],
        "messages": messages or [{"role": "user", "content": prompt}],
        "temperature": settings.get("temperature", 0.7),
        "top_p": settings.get("top_p", 0.95),
        "max_tokens": settings.get("max_tokens", 32768),
        "stream": True,
    }
    try:
        resp = _chat_request(url, dict(body, stream_options={"include_usage": True}))
    except urllib.error.HTTPError as e:
        if e.code == 400:  # server doesn't know stream_options — retry plain
            resp = _chat_request(url, body)
        else:
            raise
    if not RUN_FLAG.is_set():
        resp.close()
        raise RunStopped()

    # Register the underlying socket so Stop can shut it and wake this read.
    sock = None
    try:
        sock = resp.fp.raw._sock
        with CUR_LOCK:
            CURRENT["sock"] = sock
    except (AttributeError, OSError):
        pass

    t0 = time.time()
    ttft = None
    parts = []
    usage = None
    finish = None
    try:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    if ttft is None:
                        ttft = time.time() - t0
                    parts.append(delta["content"])
                elif delta.get("reasoning_content"):
                    # Thinking models stream reasoning first — prefill is done
                    # when the first token of ANY kind arrives.
                    if ttft is None:
                        ttft = time.time() - t0
                if choices[0].get("finish_reason"):
                    finish = choices[0]["finish_reason"]
    except (OSError, urllib.error.URLError) as e:
        with CUR_LOCK:
            CURRENT["sock"] = None
        if not RUN_FLAG.is_set():
            raise RunStopped() from e
        raise
    finally:
        with CUR_LOCK:
            CURRENT["sock"] = None
        resp.close()
    # Stop manifests as a clean EOF on some platforms (shutdown → EOF), so a
    # ended stream with a cleared flag is a stop, not a short response.
    if not RUN_FLAG.is_set():
        raise RunStopped()

    total = time.time() - t0
    text = "".join(parts)
    ptok = (usage or {}).get("prompt_tokens") or len(prompt.split())
    ctok = (usage or {}).get("completion_tokens") or len(text.split())
    gen_time = max(total - (ttft or 0), 1e-6)
    metrics = {
        "pp": round(ptok / ttft, 1) if ttft else None,       # prefill tok/s
        "tgs": round(ctok / gen_time, 1) if ctok else None,  # decode tok/s
        "tps": round(ctok / total, 1) if ctok and total else None,  # overall
        "ttft": round(ttft, 3) if ttft else None,
        "prompt_tokens": ptok,
        "wall": round(total, 3),  # total call wall time (raw+ decode math)
    }
    return text, ctok, total, finish == "length", metrics


def plausible_html(s):
    """Sanity check that a candidate string is really a self-contained HTML
    artifact (not prose, not a random snippet, not a truncated sentence)."""
    s = (s or "").strip()
    return (len(s) > 120 and "<" in s and
            re.search(r"<(html|body|head|div|canvas|script|main|section)\b", s, re.I))


def _trim_to_html_close(s):
    """Cut anything after the last </html> so outputs are html and only html."""
    i = s.rfind("</html>")
    return s[: i + 7] if i != -1 else s


def extract_html(text):
    """Pull the first plausible HTML artifact out of a model response.
    Strategies, in order: ```html-labeled fences, bare fences containing a
    full document, raw <!DOCTYPE …> blobs, and bare <html>…</html> blocks.
    Every candidate must pass plausible_html; otherwise the next is tried."""
    candidates = []
    for m in re.finditer(r"```(?:html|htm|x-html|html4strict)\s*\n(.*?)```", text, re.S | re.I):
        candidates.append(m.group(1))
    for m in re.finditer(r"```\w*\s*\n(.*?)```", text, re.S):
        candidates.append(m.group(1))
    m = re.search(r"(<!DOCTYPE html.*)", text, re.S | re.I)
    if m:
        candidates.append(m.group(1))
    m = re.search(r"(<html[\s>].*</html>)", text, re.S | re.I)
    if m:
        candidates.append(m.group(1))

    for cand in candidates:
        cand = cand.strip()
        # Labeled/bare fences may carry prose before the document itself.
        m = re.search(r"(<!DOCTYPE html.*)", cand, re.S | re.I)
        if m:
            cand = m.group(1)
        else:
            m = re.search(r"(<html[\s>].*)", cand, re.S | re.I)
            if m:
                cand = m.group(1)
        cand = _trim_to_html_close(cand)
        if plausible_html(cand):
            return cand
    return None


CLI_TIMEOUT = 7200  # long-horizon tasks need up to 120 min per cell

PI_CONFIG_DIR = os.path.join(ROOT, "harness-configs", "pi")
OC_CONFIG_ROOT = os.path.join(ROOT, "harness-configs", "opencode")


def newest_html(workdir, since):
    """Newest .html file an agent harness created during its run (agents
    like opencode/pi write artifacts to disk instead of answering in chat)."""
    best = None
    for dirpath, dirnames, filenames in os.walk(workdir):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            if not fn.endswith((".html", ".htm")):
                continue
            p = os.path.join(dirpath, fn)
            try:
                mt = os.path.getmtime(p)
            except OSError:
                continue
            if mt >= since and (best is None or mt > best[0]):
                best = (mt, p)
    # Prefer the newest plausible artifact; ignore stray/stub html files.
    if best:
        try:
            with open(best[1], encoding="utf-8", errors="replace") as f:
                if plausible_html(f.read(20000)):
                    return best[1]
        except OSError:
            return None
    return None


NODE_BIN = shutil.which("node")


def js_syntax_ok(js_source):
    """Run `node --check` on the page's inline JS. Returns (ok, error)."""
    if not NODE_BIN or not js_source.strip():
        return None, None  # node unavailable / no JS to check
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(js_source)
        path = f.name
    try:
        p = subprocess.run([NODE_BIN, "--check", path], capture_output=True, text=True, timeout=30)
        if p.returncode == 0:
            return True, None
        err = (p.stderr or "").strip().splitlines()
        return False, err[0][:200] if err else "syntax error"
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    finally:
        os.unlink(path)


def qa_artifact(path):
    """Quality gate for a generated HTML artifact. functionality% is a
    weighted checklist (structure, interactivity, completeness) plus a hard
    node --check on the inline JS; quality is a code-hygiene score.
    Below 90% functionality = flagged not usable."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            src = f.read()
    except OSError:
        return None

    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", src, re.S | re.I)
    js = "\n;\n".join(scripts)
    # interactivity may live in <script> blocks OR inline handler attributes;
    # plain (non-canvas) pages are scored without game-oriented checks —
    # same profile-aware calibration as the hart harness, so scores are
    # comparable across all five harnesses
    has_inline_handlers = bool(re.search(r"\son[a-z]+\s*=\s*[\"']", src, re.I))
    interactive = "<canvas" in src.lower()

    checks = [
        ("doctype", bool(re.search(r"<!DOCTYPE html", src, re.I)), 5),
        ("closed document", "</html>" in src.lower(), 10),
        ("has javascript", len(js.strip()) > 50 or has_inline_handlers, 15),
        ("event handlers", bool(re.search(r"addEventListener|on(keydown|click|mouse|touch|input)",
                                          js, re.I)) or has_inline_handlers, 15),
        ("run loop", bool(re.search(r"requestAnimationFrame|setInterval|setTimeout", js)), 15),
        ("dom/canvas use", "<canvas" in src.lower() or bool(re.search(r"getElementById|querySelector", js)), 10),
        ("braces balanced", js.count("{") == js.count("}") and js.count("(") == js.count(")"), 10),
        ("substantial", len(src) > 500, 10),
        ("no placeholders", "TODO" not in src and "lorem ipsum" not in src.lower(), 5),
    ]
    if not interactive:  # plain pages scored on applicable checks only
        checks = [c for c in checks
                  if c[0] not in ("run loop", "dom/canvas use", "substantial")]
    notes = [name for name, ok, _ in checks if not ok]
    func = round(100 * sum(w for _, ok, w in checks if ok) / sum(w for _, _, w in checks))

    syn_ok, syn_err = js_syntax_ok(js)
    if syn_ok is False:
        # A syntax error means the page categorically cannot work.
        func = min(func, 25)
        notes.append(f"JS syntax error: {syn_err}")
    elif syn_ok:
        notes.append("JS syntax OK")

    funcs = len(re.findall(r"\bfunction\b|=>", js))
    qchecks = [
        ("decomposed", funcs >= 3, 40),
        ("no eval/doc.write", not re.search(r"\beval\s*\(|document\.write", js), 30),
        ("modern decls", bool(re.search(r"\b(const|let)\b", js)), 30),
    ]
    qual = round(100 * sum(w for _, ok, w in qchecks if ok) / sum(w for _, _, w in qchecks))

    return {"qa_func": func, "qa_qual": qual, "qa_notes": "; ".join(notes[:6]) or "all checks passed",
            "usable": func >= 90}


def prep_pi(fw):
    """Point pi at this framework via an ISOLATED config dir (PI_CODING_AGENT_DIR)
    so the user's global ~/.pi/agent/models.json is never touched — safe for
    concurrent pi use and for multi-user deployments. Returns (cmd, env)."""
    cfg = FRAMEWORKS[fw]
    os.makedirs(PI_CONFIG_DIR, exist_ok=True)
    with open(os.path.join(PI_CONFIG_DIR, "models.json"), "w") as f:
        json.dump({"providers": {"bench": {
            "baseUrl": agent_base_url(fw),
            "api": "openai-completions",
            "apiKey": "bench",
            "models": [{"id": cfg["model"]}],
        }}}, f, indent=2)
    cmd = ["pi", "--print", "--provider", "bench",
           "--model", cfg["model"] + PI_THINKING, "--no-session"]
    env = dict(os.environ, PI_CODING_AGENT_DIR=PI_CONFIG_DIR)
    return cmd, env


def prep_opencode(fw):
    """Generate an isolated opencode config pointing at this framework.
    Returns the XDG_CONFIG_HOME dir to run opencode from (verified working:
    opencode sends the configured model id through unchanged, even with
    slashes in the id)."""
    cfg = FRAMEWORKS[fw]
    cfgdir = os.path.join(OC_CONFIG_ROOT, fw, "opencode")
    os.makedirs(cfgdir, exist_ok=True)
    with open(os.path.join(cfgdir, "opencode.json"), "w") as f:
        json.dump({
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "bench": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": f"Bench {cfg['name']}",
                    "options": {"baseURL": agent_base_url(fw),
                                "apiKey": "bench"},
                    "models": {cfg["model"]: {"name": cfg["model"]}},
                }
            },
        }, f, indent=2)
    return os.path.join(OC_CONFIG_ROOT, fw)


GOOSE_CONFIG_ROOT = os.path.join(ROOT, "harness-configs", "goose")


def prep_goose(fw):
    """Generate an isolated goose config (XDG_CONFIG_HOME isolation, verified:
    goose's openai provider honors OPENAI_BASE_URL from the process env and
    the model from providers.openai.model). Lean extension set — developer
    only — so runs stay comparable with the other agent harnesses.
    Returns (cmd, env)."""
    cfg = FRAMEWORKS[fw]
    cfgdir = os.path.join(GOOSE_CONFIG_ROOT, fw, "goose")
    os.makedirs(cfgdir, exist_ok=True)
    with open(os.path.join(cfgdir, "config.yaml"), "w") as f:
        f.write(f"""\
GOOSE_TELEMETRY_ENABLED: false
active_provider: openai
providers:
  openai:
    enabled: true
    model: {cfg['model']}
    configured: true
OPENAI_BASE_URL: {agent_base_url(fw)}
OPENAI_API_KEY: bench
extensions:
  developer:
    enabled: true
    type: builtin
    name: developer
  summon:
    enabled: false
    type: platform
    name: summon
  chatrecall:
    enabled: false
    type: platform
    name: chatrecall
""")
    return os.path.join(GOOSE_CONFIG_ROOT, fw)


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_leading_fence(chunk):
    m = re.match(r"^\s*```(?:html|htm|xml|js|javascript|json)?\s*\n?", chunk)
    return chunk[m.end():] if m else chunk


def _seam_overlap(buf, chunk, window=600, min_overlap=8):
    """Longest suffix of buf that equals a prefix of chunk (model repeats)."""
    tail, head = buf[-window:], chunk[:window]
    for size in range(min(len(tail), len(head)), min_overlap, -1):
        if tail.endswith(head[:size]):
            return size
    return 0


CONTINUATION_PREAMBLE = (
    "\n\n---\nBEGINNING OF PREVIOUS OUTPUT (cut off mid-generation):\n"
    "{tail}\n---END OF PREVIOUS OUTPUT (cut mid-content).\n\n"
    "CONTINUATION: continue the file EXACTLY at the cut point — begin with "
    "the very next character. Do NOT repeat any prior content, do NOT "
    "re-open a code fence, do NOT summarize or re-introduce. Output only the "
    "remaining file content through to the very end."
    "\n```html\n")


def rawplus_generate(fw, prompt, settings):
    """raw+ : one-shot generation with automatic continuation. The model's
    output already streams; the per-call max_tokens ceiling is a GENERATION
    limit, so when finish_reason=length, raw+ re-issues the request with the
    accumulated tail and continues the SAME artifact — streaming-buffer
    generation across calls. Per-chunk cap unchanged (uniform settings);
    total budget = RAWPLUS_MAX_ROUNDS x cap. Returns (text, tokens, wall,
    truncated, metrics, rounds)."""
    t0 = time.time()
    buffer = ""
    total_ctok = 0
    total_ptok = 0
    decode_s = 0.0
    first_metrics = {}
    rounds = 0
    truncated = True
    for rnd in range(RAWPLUS_MAX_ROUNDS):
        rounds = rnd + 1
        if rnd == 0:
            msgs = None
        else:
            tail = buffer[-3000:]
            msgs = [{"role": "user",
                     "content": prompt + CONTINUATION_PREAMBLE.format(tail=tail)}]
        text, ntok, gen, trunc, met = call_chat(fw, prompt, settings, messages=msgs)
        chunk = text
        cut = 0
        if rnd > 0:
            chunk = _strip_leading_fence(chunk)
            cut = _seam_overlap(buffer, chunk)
            if cut:
                chunk = chunk[cut:]
        buffer += chunk
        total_ctok += ntok
        total_ptok += met.get("prompt_tokens") or 0
        decode_s += max((met.get("wall") or 0) - (met.get("ttft") or 0), 0.01)
        if rnd == 0:
            first_metrics = met
        log(f"raw+ round {rounds}: +{ntok} tok"
            + (f", seam −{cut} chars" if cut else "")
            + (", finish=length → continuing" if trunc else ", finish=stop ✓"),
            fw=FRAMEWORKS[fw]["name"], harness="raw+")
        tail_add(f"[raw+] round {rounds} · +{ntok} tok"
                 + (f" · seam −{cut}" if cut else "")
                 + (" · continuing…" if trunc else " · complete ✓"))
        if not trunc:
            truncated = False
            break
    wall = time.time() - t0
    metrics = {"pp": first_metrics.get("pp"), "ttft": first_metrics.get("ttft"),
               "tgs": round(total_ctok / decode_s, 1) if decode_s > 0 else None,
               "prompt_tokens": total_ptok, "continuations": rounds - 1}
    return buffer, total_ctok, wall, truncated, metrics, rounds


def run_cli_abortable(cmd, env, cwd, timeout, line_cb=None):
    """Run a harness CLI so that Stop works mid-flight AND its output streams
    live: each stdout line (stderr merged) is forwarded to line_cb as it is
    produced — the harness narration lands in the activity feed in real time
    instead of being swallowed by a captured pipe. The process is polled in
    short slices; on stop (or timeout) its whole process group is killed.
    Returns (out, err, returncode, stopped, timed_out)."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=env, cwd=cwd, start_new_session=True,
                         bufsize=1)
    with CUR_LOCK:
        CURRENT["proc"] = p

    import queue as _queue
    lines_q = _queue.Queue()

    def _reader():
        try:
            for line in p.stdout:
                lines_q.put(line)
        except (OSError, ValueError):
            pass
        finally:
            lines_q.put(None)

    threading.Thread(target=_reader, daemon=True).start()

    def _tail_hb():
        while True:
            time.sleep(5)
            if p.poll() is not None:
                return
            with LOCK:
                since = round(time.time() - CELL_TAIL["last_line_ts"], 0)
                elapsed = round(time.time() - CELL_TAIL["started"], 0)
            tail_add(f"⏳ {elapsed}s elapsed · last output {since}s ago · "
                     f"rc pending")

    if CELL_TAIL.get("active"):
        threading.Thread(target=_tail_hb, daemon=True).start()

    deadline = time.time() + timeout
    stopped = timed_out = False
    out_lines = []
    while True:
        try:
            line = lines_q.get(timeout=2)
        except _queue.Empty:
            line = "__poll__"
        if line is None:
            p.wait()
            break
        if line != "__poll__":
            out_lines.append(line)
            if line_cb:
                try:
                    clean = ANSI_RE.sub("", line).rstrip()
                    if clean:
                        line_cb(clean)
                except Exception:
                    pass
            continue
        if p.poll() is not None:
            # drain anything the reader has left, then finish
            while True:
                try:
                    line = lines_q.get(timeout=0.5)
                except _queue.Empty:
                    line = None
                if line is None:
                    break
                out_lines.append(line)
                if line_cb:
                    clean = ANSI_RE.sub("", line).rstrip()
                    if clean:
                        line_cb(clean)
            p.wait()
            break
        if not RUN_FLAG.is_set():
            stopped = True
        elif time.time() >= deadline:
            timed_out = True
        else:
            continue
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            p.kill()
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        break
    with CUR_LOCK:
        CURRENT["proc"] = None
    with LOCK:
        CELL_TAIL["active"] = False
    if not timed_out and not RUN_FLAG.is_set():
        stopped = True
    return "".join(out_lines), "", p.returncode, stopped, timed_out


def run_harness(fw, harness, task_name, prompt, settings):
    """Run one harness test against the framework. pi and opencode are routed
    via generated per-framework configs; their stderr is captured and shown
    in the activity log so failures are diagnosable."""
    cfg = FRAMEWORKS[fw]
    label = HARNESS_LABELS.get(harness, harness)
    truncated = False
    artifact = None
    metrics = {}

    if harness in ("raw", "rawplus"):
        # Raw is a single one-shot request with no agent to budget context —
        # give it its own (higher) token cap so thinking models can finish.
        # raw+ adds automatic continuation when the generation cap is hit.
        s = dict(settings)
        if settings.get("max_tokens_raw"):
            s["max_tokens"] = settings["max_tokens_raw"]
        tail_reset(f"{cfg['name']} / {label}")
        tail_add(f"[{label}] single streaming request · "
                 f"max_tokens {s.get('max_tokens')}")
        import threading as _th
        _raw_hb_stop = _th.Event()
        def _raw_hb():
            while not _raw_hb_stop.wait(5):
                with LOCK:
                    el = round(time.time() - CELL_TAIL["started"], 0)
                tail_add(f"⏳ {el}s elapsed · streaming generation in flight")
        _th.Thread(target=_raw_hb, daemon=True).start()
        _raw_ok = False
        try:
            if harness == "rawplus":
                text, ntok, gen, truncated, metrics, _rnds = \
                    rawplus_generate(fw, prompt, s)
            else:
                text, ntok, gen, truncated, metrics = call_chat(fw, prompt, s)
            _raw_ok = True
        finally:
            _raw_hb_stop.set()
            if _raw_ok:
                tail_add(f"[{label}] stream complete · {ntok} tokens · "
                         f"truncated={truncated}")
            with LOCK:
                CELL_TAIL["active"] = False
    else:
        if harness == "pi":
            cmd, env = prep_pi(fw)
            cmd = cmd + [prompt]
        elif harness == "opencode":
            # --dir: opencode resolves its project dir itself and ignores the
            # subprocess cwd, so file artifacts must be routed explicitly.
            workdir = os.path.join(WORK_DIR, f"{fw}-{harness}-{uuid.uuid4().hex[:8]}")
            os.makedirs(workdir, exist_ok=True)
            cmd = ["opencode", "run", "--dir", workdir,
                   "--model", f"bench/{cfg['model']}", prompt]
            env = dict(os.environ, XDG_CONFIG_HOME=prep_opencode(fw))
        elif harness == "goose":
            xdg = prep_goose(fw)
            cmd = ["goose", "run", "-t", prompt, "--no-session"]
            env = dict(os.environ, XDG_CONFIG_HOME=xdg, OPENAI_API_KEY="bench",
                       OPENAI_BASE_URL=agent_base_url(fw))
        elif harness == "hart":
            if not os.path.isfile(HART_PATH):
                return {"status": "error", "latency": 0, "tokens": 0, "tps": 0.0,
                        "error": f"hart harness not found at {HART_PATH}"}
            workdir = os.path.join(WORK_DIR, f"{fw}-{harness}-{uuid.uuid4().hex[:8]}")
            os.makedirs(workdir, exist_ok=True)
            # Cell bounds: standard tasks 2×1500s; long-horizon tasks
            # 3×2400s (~2h ceiling, matching CLI_TIMEOUT). hart's internal
            # defaults (50 epochs) are for standalone use only.
            if settings.get("long_task"):
                t_budget, epochs = "2400", "3"
            else:
                t_budget, epochs = "1500", "2"
            cmd = [sys.executable, HART_PATH,
                   "--base-url", agent_base_url(fw),
                   "--model", cfg["model"],
                   "--workdir", workdir,
                   # uniform model settings: 32k per-call cap for every task
                   "--per-call-tokens", "32768",
                   "--time-budget", t_budget, "--epochs", epochs,
                   prompt]
            env = dict(os.environ)
        else:
            return {"status": "error", "latency": 0, "tokens": 0, "tps": 0.0,
                    "error": f"unknown harness {harness}"}

        if not shutil.which(cmd[0]):
            log(f"{label} CLI not found — using raw API fallback", fw=cfg["name"], harness=label)
            text, ntok, gen, truncated, metrics = call_chat(fw, prompt, settings)
        else:
            log(f"dispatching via CLI: {cmd[0]}", fw=cfg["name"], harness=label)
            if harness not in ("opencode", "hart"):  # these set their own workdir
                workdir = os.path.join(WORK_DIR, f"{fw}-{harness}-{uuid.uuid4().hex[:8]}")
                os.makedirs(workdir, exist_ok=True)
            proxy_reset()
            t0 = time.time()

            tail_reset(f"{cfg['name']} / {label}")

            def _tail_line(line):
                # every line lands in the verbose UI tail; hart narration
                # additionally flows into the activity feed
                tail_add(f"[{label}] {line}")
                if harness == "hart" and (
                        line.startswith(("[", "AGEDIN", "resum", "[epoch"))
                        or line.startswith(("⚠", "⏳"))):
                    log(line[:160], fw=cfg["name"], harness=label)

            out, errtext, rc, stopped, timed_out = run_cli_abortable(
                cmd, env, workdir, CLI_TIMEOUT, line_cb=_tail_line)
            gen = time.time() - t0
            if stopped:
                log(f"{label} stopped by user after {gen:.0f}s",
                    fw=cfg["name"], harness=label, level="err")
                raise RunStopped()
            text = out
            # Model-side timing measured by the proxy across all the
            # agent's calls (the CLIs don't expose timing internals).
            st = proxy_read()
            if st["requests"]:
                metrics = {"prompt_tokens": st["prompt_tokens"],
                           "ttft": round(st["ttft_sum"] / st["requests"], 3)}
                if st["ttft_sum"] > 0:
                    metrics["pp"] = round(st["prompt_tokens"] / st["ttft_sum"], 1)
                if st["decode_sum"] > 0 and st["completion_tokens"] > 0:
                    metrics["tgs"] = round(st["completion_tokens"] / st["decode_sum"], 1)
                log(f"model traffic: {st['requests']} call(s), "
                    f"{st['prompt_tokens']}→{st['completion_tokens']} tok"
                    + (f", {st['length_hits']} hit the harness token cap"
                       if st.get("length_hits") else ""),
                    fw=cfg["name"], harness=label)
            if st.get("length_hits"):
                truncated = True
                log(f"⚠ {st['length_hits']} model call(s) ended finish=length — the HARNESS's "
                    f"per-call token cap (not the framework) cut the answer; thinking models "
                    f"spend the budget on reasoning first",
                    fw=cfg["name"], harness=label, level="err")
            # Agent harnesses deliver results as files; collect the newest
            # HTML they created and count its content as generated tokens.
            artifact = newest_html(workdir, t0)
            if artifact:
                with open(artifact, encoding="utf-8", errors="replace") as f:
                    ntok = len(text.split()) + len(f.read().split())
                log(f"agent wrote artifact → {os.path.basename(artifact)}",
                    fw=cfg["name"], harness=label)
            else:
                ntok = len(text.split())
            if timed_out:
                extra = (f"; model traffic so far: {st['requests']} call(s), "
                         f"{st['completion_tokens']} tok") if st["requests"] else ""
                if artifact:
                    # The killed process may still have delivered its artifact.
                    with open(artifact, encoding="utf-8", errors="replace") as f:
                        ntok = len(f.read().split())
                    truncated = True
                    log(f"{label} timed out after {CLI_TIMEOUT}s but the artifact "
                        f"was delivered — keeping it{extra}",
                        fw=cfg["name"], harness=label, level="err")
                else:
                    log(f"{label} timed out after {CLI_TIMEOUT}s{extra}",
                        fw=cfg["name"], harness=label, level="err")
                    return {"status": "error", "latency": round(time.time() - t0, 2),
                            "tokens": 0, "tps": 0.0,
                            "error": f"timeout after {CLI_TIMEOUT}s{extra}"}
            elif rc != 0:
                err = (errtext or out or "no output").strip()[-300:]
                if artifact:
                    # The agent finished the task (file delivered) but a
                    # later call failed — keep the artifact, don't fail.
                    log(f"{label} exited {rc} after delivering the "
                        f"artifact — keeping it ({err[:120]})",
                        fw=cfg["name"], harness=label, level="err")
                else:
                    log(f"{label} exited {rc}: {err}",
                        fw=cfg["name"], harness=label, level="err")
                    return {"status": "error", "latency": round(gen, 2), "tokens": 0,
                            "tps": 0.0, "error": err}

    # An agent may deliver the artifact as a file and print nothing to stdout
    # — that is a success, not an empty response. Only fail when there is
    # neither text nor a collected artifact.
    if not text.strip() and not artifact:
        return {"status": "error", "latency": gen, "tokens": 0, "tps": 0.0,
                "error": "empty response"}

    # hart reports accurate totals on its last stdout line — prefer them
    # over word-count estimates (applies after artifact/ntok fallback logic).
    if harness == "hart":
        m = re.search(r"HART_RESULT (\{.*\})", text, re.S)
        if m:
            try:
                ar = json.loads(m.group(1))
                if ar.get("tokens"):
                    ntok = ar["tokens"]
                metrics.update({"calls": ar.get("calls"),
                                "hart_status": ar.get("status")})
            except json.JSONDecodeError:
                pass

    result = {"status": "done", "latency": round(gen, 2), "tokens": ntok,
              "tps": round(ntok / gen, 1) if gen > 0 else 0.0,
              "truncated": truncated, **metrics}

    # Every successful run gets an Output button: the agent's written HTML
    # file if one was collected, else extracted HTML, else the response text.
    oid = uuid.uuid4().hex[:12]
    if artifact:
        with open(artifact, encoding="utf-8", errors="replace") as f:
            content = f.read()
        fname = f"{oid}.html"
    else:
        html = extract_html(text)
        if html:
            content, fname = html, f"{oid}.html"
        else:
            content, fname = text, f"{oid}.txt"
    ctype = "text/html" if fname.endswith(".html") else "text/plain"
    with open(os.path.join(OUTPUT_DIR, fname), "w") as f:
        f.write(content)
    result["output_url"] = f"/output/{fname}"

    if fname.endswith(".html"):
        qa = qa_artifact(os.path.join(OUTPUT_DIR, fname))
        if qa:
            result.update(qa)
            log(f"QA: functionality {qa['qa_func']}%, quality {qa['qa_qual']}"
                + ("" if qa["usable"] else " — ⚠ BELOW 90% USABILITY THRESHOLD"),
                fw=FRAMEWORKS[fw]["name"], harness=HARNESS_LABELS.get(harness, harness),
                level="ok" if qa["usable"] else "err")

    result["_text"] = text[:5000]  # kept internally, not sent wholesale to UI
    return result


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def _validate_run(req):
    """Validate a /api/run body. Raises ValueError with a user-readable
    message on the first problem — keeps bad input from 500-ing the server."""
    if not isinstance(req, dict):
        raise ValueError("body must be a JSON object")
    harnesses = req.get("harnesses")
    if not isinstance(harnesses, list) or not harnesses:
        raise ValueError("harnesses: non-empty list required")
    bad = [h for h in harnesses if h not in HARNESS_LABELS]
    if bad:
        raise ValueError(f"unknown harness(es): {', '.join(map(str, bad))}")
    if not isinstance(req.get("task_id"), str) or not req.get("task_id"):
        raise ValueError("task_id: non-empty string required")
    if not isinstance(req.get("prompt"), str) or not req["prompt"].strip():
        raise ValueError("prompt: non-empty string required")
    settings = req.get("settings") or {}
    if not isinstance(settings, dict):
        raise ValueError("settings: object required")
    for k in ("temperature", "top_p", "max_tokens", "max_tokens_raw", "repeats"):
        if k in settings and settings[k] is not None \
                and not isinstance(settings[k], (int, float)):
            raise ValueError(f"settings.{k}: number required")
    if "frameworks" in req and req["frameworks"] is not None \
            and not isinstance(req["frameworks"], list):
        raise ValueError("frameworks: list required")


def run_benchmark(req):
    _validate_run(req)
    harnesses = req["harnesses"]
    # optional framework subset (e.g. compare just two); unknown ids ignored,
    # empty/missing → all
    frameworks = [fw for fw in (req.get("frameworks") or list(FRAMEWORKS))
                  if fw in FRAMEWORKS] or list(FRAMEWORKS)
    task_id = req["task_id"]
    prompt = req["prompt"]
    settings = req.get("settings", {})
    try:
        repeats = max(1, min(5, int(settings.get("repeats", 1))))
    except (TypeError, ValueError):
        repeats = 1
    task_name = next((t["name"] for t in TASKS if t["id"] == task_id), task_id)
    artifact = next((t.get("artifact") for t in TASKS if t["id"] == task_id), None)
    if any(t.get("long") for t in TASKS if t["id"] == task_id):
        settings = dict(settings, long_task=True)

    with LOCK:
        STATE["running"] = True
        STATE["results"] = []
        STATE["run_started"] = time.time()
    RUN_FLAG.set()
    shutil.rmtree(WORK_DIR, ignore_errors=True)  # scratch dirs from prior runs

    try:
        for fw in frameworks:
            if not RUN_FLAG.is_set():
                log("benchmark stopped by user", level="err")
                break
            cfg = FRAMEWORKS[fw]
            with LOCK:
                STATE["current_step"] = f"starting {cfg['name']}…"
            if not start_framework(fw):
                continue
            PROXY_STATE.update({"target": f"http://127.0.0.1:{cfg['port']}",
                                "fw": cfg["name"], "model": cfg["model"]})
            log_device_info(fw)

            # Warmup so first-request overhead (graph compile, cache fill)
            # doesn't skew the first measured row.
            log("warmup request (not measured)…", fw=cfg["name"])
            try:
                call_chat(fw, "Say OK.", {"temperature": 0.1, "max_tokens": 16})
            except RunStopped:
                raise
            except Exception as e:
                log(f"warmup failed (continuing): {e}", fw=cfg["name"], level="err")

            for rep in range(repeats):
                for h in harnesses:
                    if not RUN_FLAG.is_set():
                        break
                    label = HARNESS_LABELS.get(h, h)
                    row = {"framework": fw, "model": cfg["model"], "harness": label,
                           "task": task_name + (f" #{rep + 1}" if repeats > 1 else ""),
                           "status": "running", "latency": None, "tokens": None,
                           "tps": None, "output_url": None}
                    with LOCK:
                        STATE["results"].append(row)
                        STATE["current_step"] = f"{cfg['name']} / {label} / {task_name}"
                    log(f"running task “{task_name}”", fw=cfg["name"], harness=label)
                    snap_before = server_snapshot(fw)
                    try:
                        r = run_harness(fw, h, task_name,
                                        prompt_for_harness(prompt, artifact, h), settings)
                        row.update({k: v for k, v in r.items() if k != "_text"})
                        row["_text"] = r.get("_text", "")
                        snap_after = server_snapshot(fw)
                        srv = server_cell_delta(fw, snap_before, snap_after)
                        if fw == "omlx" and snap_after and \
                                not srv.get("server_tgs") and not srv.get("server_pp"):
                            # usage counters flush every ~5s — the diff window
                            # may have caught a flush boundary; settle + retry
                            time.sleep(6)
                            srv = server_cell_delta(fw, snap_before,
                                                    server_snapshot(fw))
                        if srv:
                            row.update(srv)
                        # context fill % and effective-bandwidth estimate
                        if row.get("prompt_tokens") and cfg.get("ctx_tokens"):
                            row["ctx_fill_pct"] = round(
                                100.0 * row["prompt_tokens"] / cfg["ctx_tokens"], 1)
                        tgs_source = row.get("server_tgs") or row.get("tgs")
                        if tgs_source and cfg.get("model_gb"):
                            row["est_gbps"] = round(
                                tgs_source * cfg["model_gb"] / 1024, 1)
                        if srv or row.get("ctx_fill_pct") is not None:
                            log(f"cell metrics: server pp={srv.get('server_pp')} "
                                f"tgs={srv.get('server_tgs')}"
                                + (f" | ctx fill {row.get('ctx_fill_pct')}%"
                                   if row.get("ctx_fill_pct") is not None else "")
                                + (f" | ~{row.get('est_gbps')} GB/s est"
                                   if row.get("est_gbps") else ""),
                                fw=cfg["name"], harness=label)
                        log(f"done in {row['latency']}s @ {row['tps']} tok/s"
                            + (" — ⚠ truncated at max_tokens, raise the cap and rerun"
                               if row.get("truncated") else ""),
                            fw=cfg["name"], harness=label,
                            level="ok" if row["status"] == "done" else "err")
                    except RunStopped:
                        row.update({"status": "error", "error": "stopped by user"})
                        log(f"{label} stopped by user", fw=cfg["name"],
                            harness=label, level="err")
                        raise
                    except Exception as e:
                        row.update({"status": "error", "error": str(e)[:300]})
                        log(f"failed: {e}", fw=cfg["name"], harness=label, level="err")

            # framework's tests complete → shut it down before next framework
            with LOCK:
                STATE["current_step"] = f"stopping {cfg['name']}…"
            stop_framework(fw)
    finally:
        for fw in list(PROCS):
            stop_framework(fw)
        with LOCK:
            STATE["running"] = False
            STATE["current_step"] = ""
            STATE["run_started"] = None
            rows = [{k: v for k, v in r.items() if k != "_text"} for r in STATE["results"]]
        # Persist run history so the community can compare across runs/days.
        if any(r["status"] == "done" for r in rows):
            try:
                os.makedirs(RUNS_DIR, exist_ok=True)
                fname = os.path.join(RUNS_DIR, time.strftime("%Y%m%d-%H%M%S") + ".json")
                with open(fname, "w") as f:
                    json.dump({"ts": time.time(), "task_id": task_id,
                               "task_name": task_name, "harnesses": harnesses,
                               # frameworks_run = the subset actually run this
                               # time; frameworks = full config snapshot (the
                               # UI's history view reads this one).
                               "frameworks_run": frameworks,
                               "settings": settings,
                               "frameworks": {k: {"port": v["port"], "model": v["model"]}
                                              for k, v in FRAMEWORKS.items()},
                               "results": rows}, f, indent=2)
                log(f"run history saved → {fname}", level="ok")
            except OSError as e:
                log(f"could not save run history: {e}", level="err")
        log("benchmark complete", level="ok")


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _safe(self, fn):
        """Run a route handler; bad input → 400, anything else → 500 JSON
        (never an HTML traceback page)."""
        try:
            fn()
        except ValueError as e:
            self._json({"ok": False, "error": str(e)[:200]}, 400)
        except BrokenPipeError:
            pass
        except Exception as e:
            log(f"HTTP {self.command} {self.path} failed: {e}", level="err")
            try:
                self._json({"ok": False, "error": f"internal error: {str(e)[:150]}"}, 500)
            except Exception:
                pass

    def do_GET(self):
        self._safe(self._route_get)

    def _route_get(self):
        if self.path == "/" or self.path == "/index.html":
            with open(os.path.join(ROOT, "index.html"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Cache-Control", "no-cache, must-revalidate")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/health":
            with LOCK:
                self._json({"ok": True, "running": STATE["running"],
                            "ts": time.time(),
                            "frameworks": dict(STATE["framework_status"])})
        elif self.path == "/api/tasks":
            self._json(TASKS)
        elif self.path == "/api/harnesses":
            # ordered harness id -> display label (UI renders from this)
            self._json(HARNESS_LABELS)
        elif self.path == "/api/state":
            refresh_fw_status_idle()
            free, total = free_ram()
            with LOCK:
                snap = {**STATE, "results": [{k: v for k, v in r.items() if k != "_text"}
                                             for r in STATE["results"]]}
            snap["ram"] = {"free": free, "total": total}
            snap["cell_tail"] = tail_snapshot()
            self._json(snap)
        elif self.path.startswith("/api/activity?since="):
            try:
                since = float(self.path.split("since=")[1])
            except (IndexError, ValueError):
                raise ValueError("since= must be a number")
            with LOCK:
                lines = [l for l in ACTIVITY if l["ts"] > since]
            self._json(lines)
        elif self.path == "/api/runs_history":
            import glob as _g
            runs = []
            for f in sorted(_g.glob(os.path.join(RUNS_DIR, "*.json")),
                            reverse=True):
                try:
                    with open(f) as fh:
                        d = json.load(fh)
                    rows = [{k: v for k, v in r.items() if k != "_text"}
                            for r in d.get("results", [])]
                    runs.append({"file": os.path.basename(f),
                                 "ts": d.get("ts"), "task": d.get("task_name"),
                                 "frameworks": {k: v.get("model")
                                                for k, v in (d.get("frameworks")
                                                             or {}).items()},
                                 "harnesses": d.get("harnesses"),
                                 "rows": rows})
                except (json.JSONDecodeError, OSError):
                    pass
            self._json(runs)
        elif self.path == "/api/tail":
            self._json(tail_snapshot(n=TAIL_MAX))
        elif self.path == "/api/frameworks":
            with LOCK:
                self._json({fw: {"name": c["name"], "port": c["port"],
                                 "model": c["model"],
                                 "repo": c.get("repo"),
                                 "ctx_tokens": c.get("ctx_tokens"),
                                 "model_gb": c.get("model_gb"),
                                 "start_cmd": " ".join(resolve_start_cmd(c)),
                                 "notes": c.get("notes", ""),
                                 "status": STATE["framework_status"].get(fw)}
                            for fw, c in FRAMEWORKS.items()})
        elif self.path.startswith("/api/models/discover"):
            # ?fw=<id> limits the "current" marker to one framework; the
            # candidate list is always the unified set (all sources), each
            # tagged with the frameworks it can serve. The UI greys out the
            # rest for the selected framework.
            qfw = self.path.split("fw=")[1].split("&")[0] if "fw=" in self.path else None
            fw = qfw if qfw in FRAMEWORKS else None
            free, _total = free_ram()
            free_gb = round(free / 1073741824, 1)
            candidates = discovery.all_candidates(free_gb, FRAMEWORKS)
            if fw:
                cur_key = discovery.normalize_key(
                    FRAMEWORKS[fw].get("repo") or FRAMEWORKS[fw]["model"])
                for c in candidates:
                    c["current"] = (discovery.normalize_key(c["id"]) == cur_key)
                out = {fw: {
                    "current": FRAMEWORKS[fw].get("repo") or FRAMEWORKS[fw]["model"],
                    "ctx_tokens": FRAMEWORKS[fw].get("ctx_tokens"),
                    "free_ram_gb": free_gb,
                    "candidates": candidates,
                }}
            else:
                out = {
                    "free_ram_gb": free_gb,
                    "candidates": candidates,
                    "frameworks": {f: {
                        "current": FRAMEWORKS[f].get("repo") or FRAMEWORKS[f]["model"],
                        "ctx_tokens": FRAMEWORKS[f].get("ctx_tokens"),
                    } for f in FRAMEWORKS},
                }
            self._json(out)
        elif self.path == "/api/proxy":
            with PROXY_LOCK:
                data = {"port": PROXY_PORT, **PROXY_STATE,
                        "totals": dict(PROXY_TOTALS), "log": list(PROXY_LOG)}
            self._json(data)
        elif self.path.startswith("/output/"):
            name = os.path.basename(self.path.split("?")[0])
            fp = os.path.join(OUTPUT_DIR, name)
            if os.path.isfile(fp):
                with open(fp, "rb") as f:
                    body = f.read()
                ctype = ("text/html" if name.endswith(".html") else "text/plain") + "; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def do_POST(self):
        self._safe(self._route_post)

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            raise ValueError("bad Content-Length")
        if length > 1_000_000:  # prompts are small; reject anything fat
            raise ValueError("body too large")
        raw = self.rfile.read(length) if length else b""
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError:
            raise ValueError("body must be valid JSON")

    def _route_post(self):
        req = self._read_body()
        if self.path == "/api/run":
            _validate_run(req)  # 400 on bad input before spawning a thread
            with LOCK:
                already = STATE["running"]
            if already:
                self._json({"ok": False, "error": "already running"}, 409)
                return
            threading.Thread(target=run_benchmark, args=(req,), daemon=True).start()
            self._json({"ok": True})
        elif self.path == "/api/stop":
            request_stop()
            log("stop requested — killing in-flight work", level="err")
            self._json({"ok": True})
        elif self.path == "/api/models/select":
            fw = req.get("fw") if isinstance(req, dict) else None
            model = req.get("model") if isinstance(req, dict) else None
            if fw not in FRAMEWORKS:
                raise ValueError(f"unknown framework: {fw}")
            if not isinstance(model, str) or not model.strip():
                raise ValueError("model: non-empty string required")
            cfg = FRAMEWORKS[fw]
            model = model.strip()
            cfg["model"] = model
            # MTPLX: the CLI --model flag loads by repo id from ~/.mtplx/models,
            # but the server serves a normalized id (auto-adopted on start). Track
            # the repo separately so start_cmd's {repo} placeholder resolves right.
            if cfg.get("model_source") == "mtplx":
                cfg["repo"] = model
            for key, cast in (("ctx_tokens", int), ("model_gb", float)):
                if isinstance(req.get(key), (int, float)) and req[key] > 0:
                    cfg[key] = cast(req[key])
            if not save_config():
                raise ValueError("model updated in memory but config.json save failed")
            log(f"model for {cfg['name']} set to {cfg['model']}", fw=fw, level="ok")
            self._json({"ok": True, "model": cfg["model"]})
        elif self.path == "/api/proxy/clear":
            with PROXY_LOCK:
                PROXY_LOG.clear()
                PROXY_TOTALS.clear()
                PROXY_TOTALS.update(PROXY_ZERO)
            self._json({"ok": True})
        else:
            self.send_error(404)

    def do_DELETE(self):
        self._safe(self._route_delete)

    def _route_delete(self):
        # /api/runs/<YYYYmmdd-HHMMSS>.json — permanently remove one saved run:
        # its JSON file plus any /output artifacts referenced only by it.
        m = re.fullmatch(r"/api/runs/(\d{8}-\d{6}\.json)", self.path)
        if not m:
            self._json({"ok": False, "error": "bad request"}, 400)
            return
        name = m.group(1)
        fp = os.path.join(RUNS_DIR, name)
        if not os.path.isfile(fp):
            self._json({"ok": False, "error": "no such run"}, 404)
            return
        refs = set()
        try:
            with open(fp) as fh:
                d = json.load(fh)
            for r in d.get("results", []):
                u = r.get("output_url")
                if u and "/output/" in u:
                    refs.add(os.path.basename(u.split("?")[0]))
        except (json.JSONDecodeError, OSError):
            pass
        os.remove(fp)
        # keep any output file another surviving run still points at
        import glob as _g
        for f in _g.glob(os.path.join(RUNS_DIR, "*.json")):
            if os.path.basename(f) == name:
                continue
            try:
                with open(f) as fh:
                    other = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            for r in other.get("results", []):
                u = r.get("output_url")
                if u and "/output/" in u:
                    refs.discard(os.path.basename(u.split("?")[0]))
        removed = 0
        for n in sorted(refs):
            op = os.path.join(OUTPUT_DIR, n)
            if os.path.isfile(op):
                try:
                    os.remove(op)
                    removed += 1
                except OSError:
                    pass
        log(f"deleted run {name}"
            + (f" + {removed} orphaned output file(s)" if removed else ""),
            level="ok")
        self._json({"ok": True, "deleted_outputs": removed})


def _shutdown(signum, _frame):
    log(f"received signal {signum} — shutting down frameworks", level="err")
    for fw in list(PROCS):
        stop_framework(fw)
    sys.exit(0)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(CONFIG.get("port", 7090))
    host = CONFIG.get("host", "127.0.0.1")
    if ROUTE_VIA_PROXY:
        start_proxy()
    # reflect real status of any servers already listening
    for fw in FRAMEWORKS:
        set_fw_status(fw, "up" if framework_healthy(fw) else "down")
    signal.signal(signal.SIGTERM, _shutdown)
    log("benchmark server ready"
        + (f" (measurement proxy on :{PROXY_PORT})" if ROUTE_VIA_PROXY else ""))
    if host in ("0.0.0.0", "::"):
        log("⚠ listening on ALL interfaces — anyone on your network can start "
            "runs and delete data. Use host 127.0.0.1 unless you know why.",
            level="err")
    print(f"\n  ➜ Open http://{'localhost' if host in ('127.0.0.1', '::1') else host}:{port}\n")
    try:
        ThreadingHTTPServer((host, port), Handler).serve_forever()
    except OSError as e:
        print(f"\n  ✗ Could not bind {host}:{port} — {e}\n"
              f"    Is another benching server (or a framework) already using it?\n"
              f"    Change 'port' in config.json or pass a different port: python3 server.py <port>",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        for fw in list(PROCS):
            stop_framework(fw)
