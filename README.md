# benching

**Local LLM benchmarking for Apple Silicon.** Compare how the same model performs
across the major local inference frameworks (OMLX, MTPLX, MLX-VLM, MLX-Serve) and
across harnesses (raw API, raw+ streaming, and the agent CLIs pi / opencode / goose /
hart) — with real throughput (TPS/TGS/PP), latency, and a quality gate (QA) on the
actual artifact each cell produces.

Everything runs **locally** against models already on your machine. No cloud, no
accounts, no telemetry. The backend is **pure Python standard library** — there is
nothing to `pip install`.

---

## What it measures

For every **framework × harness** cell on a chosen task, benching records:

| Metric | Meaning |
|--------|---------|
| **TPS** | Overall throughput — completion tokens / total wall time |
| **TGS** | Decode speed — tokens generated per second (server-measured where the framework exposes it) |
| **PP** | Prefill speed — prompt tokens processed per second |
| **Latency** | End-to-end wall time for the cell |
| **Tokens** | Completion tokens produced |
| **QA** | Automated quality score (functionality / quality) of the produced artifact, with a 90% "usable" threshold |
| **Output** | The actual artifact (HTML/text), saved and viewable |

The dashboard shows a live activity feed, a per-cell streaming tail, a comparison
chart (TPS/TGS/PP per framework across harnesses), and a "winning combo" (best usable
result, soonest). Every completed run is saved to `runs/*.json` for later comparison.

---

## Prerequisites

- **Apple Silicon Mac** (M1 or later) with a recent **Python 3.10+** on your `PATH`.
- The **framework CLIs** you want to benchmark (install only what you need):
  - `omlx` — OMLX
  - `mtplx` — MTPLX
  - `mlx-serve` — MLX-Serve
  - a Python interpreter with `mlx_vlm` installed — MLX-VLM
- **Agent harness CLIs** (only for the agent rows): `pi`, `opencode`, and `goose` are
  community tools you install yourself. The **`hart`** agentic harness is **bundled** in
  this repo under `hart/` — it works out of the box, no install needed.
- The **models** you want to test, already on your machine:
  - OMLX / MLX-VLM / MLX-Serve read from your Hugging Face cache
    (`~/.cache/huggingface/hub`).
  - MTPLX reads from its own store (`~/.mtplx/models`).

  benching auto-discovers both, tags each model with the frameworks it can serve, and
  tells you what fits in your free RAM.

`install.sh` checks all of the above and tells you exactly what's missing.

---

## Quick start

```bash
# 1. Get the code
git clone https://github.com/akshaya-git/benching.git
cd benching

# 2. Check prerequisites + seed config.json
./install.sh

# 3. Edit config.json to match your machine (models, ports, start commands)
$EDITOR config.json

# 4. Start the server
python3 server.py            # or: make run
#    ➜ Open http://localhost:7090
```

Then in the browser: pick your frameworks and harnesses, choose a task (or edit the
prompt), and hit **▶ Run Benchmark**.

> **First run tip:** the *Model per Framework* panel lists every model on your machine
> (HF cache + MTPLX store) with a live compatibility verdict (**ready / tight /
> too-large / unknown**) based on your current free RAM. Models a framework can't serve
> are greyed out (e.g. MTPLX-only models for OMLX/MLX-VLM/MLX-Serve). Pick a model and
> it's written straight into `config.json`.

---

## Configuration

All machine-specific settings live in **`config.json`** (seeded from
`config.example.json` on first run). The important knobs:

| Key | What it does |
|-----|--------------|
| `host` / `port` | Where the web UI listens. Keep `host: 127.0.0.1` unless you know why you'd expose it. |
| `frameworks.<id>.model` | The model id that framework serves / should serve. |
| `frameworks.<id>.model_source` | Where the framework loads models from: `hf` (HF cache) or `mtplx` (`~/.mtplx/models`). Controls which models the picker offers for that framework. |
| `frameworks.<id>.start_cmd` | The exact command to cold-start that framework's server. Supports `{model}`, `{repo}`, and `{model_path}` placeholders. |
| `frameworks.<id>.port` | The port that framework's server listens on. |
| `frameworks.<id>.ctx_tokens` | Configured context window (used for compatibility scoring + ctx-fill %). |
| `frameworks.<id>.model_gb` | Approx weight size in GB (used for RAM-fit scoring). |
| `route_via_proxy` | Route agent harnesses through the local measurement proxy (adds per-request PP/TGS/TTFT to agent rows). |
| `pi_thinking` | pi thinking-level suffix for `--model` (e.g. `:high`). Empty = inherit server default. |
| `hart_path` | Path to the `hart` agentic harness script. Defaults to the bundled `./hart/hart.py`. |

See **[CONFIG.md](CONFIG.md)** for the full reference and **[ARCHITECTURE.md](ARCHITECTURE.md)**
for how it all fits together.

---

## How a run works

1. For each selected **framework** (in order):
   - Start its server (or reuse one already healthy on its port).
   - Wait for its `/v1/models` endpoint to answer.
   - For each selected **harness**, run the task and record the cell.
   - Shut the server down (SIGTERM → SIGKILL) before moving on.
2. Results stream to the dashboard live; the full run is saved to `runs/`.

Frameworks are run **one at a time** so they never compete for RAM/GPU. You can
**Stop** a run at any time — in-flight work is killed and partial results are kept.

---

## Project layout

```
benching/
├── server.py            # the whole backend (stdlib only): HTTP API, orchestration, metrics
├── discovery.py         # model discovery + RAM/context compatibility scoring
├── index.html           # the dashboard (single-file frontend, no build step)
├── config.example.json  # the config template (copied to config.json on first run)
├── install.sh           # prerequisite check + config seeding
├── Makefile             # setup / run / check / clean
├── requirements.txt     # (empty — stdlib only)
├── hart/                # the bundled `hart` agentic harness (hart.py + docs)
├── examples/            # sample artifacts (e.g. a generated pong.html)
├── runs/                # saved runs (gitignored)
├── outputs/             # produced artifacts (gitignored)
├── logs/                # bench.log + per-framework server logs (gitignored)
├── work/                # scratch dirs for agent file artifacts (gitignored)
└── harness-configs/     # per-run harness configs, regenerated each run (gitignored)
```

---

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — full system diagram (Mermaid) + data flow.
- **[CONFIG.md](CONFIG.md)** — every config key, explained.
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** — common failures and fixes.
- **[ANALYSIS.md](ANALYSIS.md)** — the code review: bugs found & fixed, design notes.

---

## Requirements & license

- Python 3.10+ (standard library only — no dependencies).
- MIT License. See [LICENSE](LICENSE).
