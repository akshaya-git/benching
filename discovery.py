#!/usr/bin/env python3
"""
Model discovery & compatibility scoring for benching.

Two discovery sources per framework:
  1. served — the framework's live /v1/models (authoritative while it runs)
  2. local  — repos already in the Hugging Face cache (serve-ready, no download)

Every candidate is scored for compatibility with the benchmark:
  RAM fit (weights + KV-cache headroom), context window vs the configured
  window, local availability, and MTP draft availability for speculative
  decoding. The UI uses this to let users pick a model per framework.

Stdlib only.
"""

import glob
import json
import os
import urllib.request

HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
_GB = 1073741824
# Weights + KV cache + engine overhead: 25% headroom over raw weight size.
RAM_HEADROOM = 1.25


def _get_json(url, timeout=4):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def cache_dirname(model_id):
    """Resolve an HF-cache directory for a model id. Handles both id styles:
    repo ids ("org/name") and cache-style ids ("org--name" as served by OMLX).
    Returns the directory path or None."""
    candidates = []
    if "/" in model_id:
        candidates.append("models--" + model_id.replace("/", "--"))
    candidates.append("models--" + model_id)
    for c in candidates:
        d = os.path.join(HF_CACHE, c)
        if os.path.isdir(d):
            return d
    return None


def snapshot_dir(model_id):
    """Latest snapshot dir inside the cache entry (CLI --model flags need a
    real path; repo names only work as request ids)."""
    d = cache_dirname(model_id)
    if not d:
        return None
    hits = sorted(glob.glob(os.path.join(d, "snapshots", "*")))
    return hits[-1].rstrip("/") if hits else None


def _snapshot_size_gb(cache_dir):
    """Largest snapshot's total size in GB (models may have several)."""
    best = 0
    for snap in glob.glob(os.path.join(cache_dir, "snapshots", "*")):
        total = 0
        for dirpath, _dirnames, filenames in os.walk(snap):
            for fn in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, fn))
                except OSError:
                    pass
        best = max(best, total)
    return round(best / _GB, 1) if best else None


def _cached_context(cache_dir):
    """Context window from the cached config.json, when the model declares one.
    Checks the top level and the nested text_config (multimodal models nest
    the LM config there)."""
    for p in sorted(glob.glob(os.path.join(cache_dir, "snapshots", "*", "config.json"))):
        try:
            with open(p) as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for node in (cfg, cfg.get("text_config") or {}):
            if not isinstance(node, dict):
                continue
            for key in ("max_position_embeddings", "seq_len", "model_max_length"):
                v = node.get(key)
                if isinstance(v, int) and v > 0:
                    return v
    return None


def normalize_key(model_id):
    """Canonical dedup key for a model id, in either style. Repo ids
    ("org/name") and cache-style ids ("org--name", as OMLX serves them) both
    map to the HF-cache directory name, so the same model de-dups across the
    served and local sources."""
    if "/" in model_id:
        return "models--" + model_id.replace("/", "--")
    return "models--" + model_id


def discover_served(port, timeout=4):
    """Live model list from a running OpenAI-compatible server."""
    data = _get_json(f"http://127.0.0.1:{port}/v1/models", timeout)
    if not data:
        return []
    out = []
    for m in data.get("data", []):
        mid = m.get("id")
        if not mid:
            continue
        out.append({
            "id": mid,
            "context_length": m.get("context_length"),
            "served": True,
            "in_cache": cache_dirname(mid) is not None,
        })
    return out


def discover_local():
    """All repos in the local HF cache, with size + context where known."""
    out = []
    if not os.path.isdir(HF_CACHE):
        return out
    for entry in sorted(os.listdir(HF_CACHE)):
        if not entry.startswith("models--"):
            continue
        d = os.path.join(HF_CACHE, entry)
        repo = entry[len("models--"):].replace("--", "/")
        out.append({
            "id": repo,
            "served": False,
            "in_cache": True,
            "size_gb": _snapshot_size_gb(d),
            "context_length": _cached_context(d),
        })
    return out


def mtp_draft_for(model_id, local_ids):
    """Heuristic: a cached MTP draft matching the base model (speculative
    decoding). Returns the draft repo id or None."""
    base = model_id.split("/")[-1]
    for lid in local_ids:
        short = lid.split("/")[-1]
        if short != base and short.startswith(base) and "mtp" in short.lower():
            return lid
    return None


def compatibility(cand, free_ram_gb, ctx_tokens):
    """Score one candidate model. Returns {verdict, reasons[]}.
    verdict: ready | tight | too-large | unknown
      ready     — fits RAM, context OK, locally available
      tight     — fits, but declared context < configured window
      too-large — weights + KV headroom exceed free RAM
      unknown   — not locally cached and not currently served (download needed)
    """
    reasons = []
    verdict = "ready"
    size = cand.get("size_gb")
    if size and free_ram_gb:
        need = size * RAM_HEADROOM
        if need > free_ram_gb:
            verdict = "too-large"
            reasons.append(f"~{size} GB weights (+KV headroom) > {free_ram_gb:.0f} GB free")
    ctx = cand.get("context_length")
    if ctx and ctx_tokens and ctx < ctx_tokens:
        if verdict == "ready":
            verdict = "tight"
        reasons.append(f"context {ctx:,} < configured {ctx_tokens:,}")
    if not cand.get("in_cache") and not cand.get("served"):
        if verdict == "ready":
            verdict = "unknown"
        reasons.append("not in local cache — framework must download on start")
    return {"verdict": verdict, "reasons": reasons}


def candidates_for(fw_cfg, free_ram_gb):
    """Merged, de-duplicated candidate list for one framework, each with a
    compatibility verdict. Served models first (they are what the running
    server can actually answer with right now)."""
    port = fw_cfg.get("port")
    ctx_tokens = fw_cfg.get("ctx_tokens")
    local = discover_local()
    local_ids = [m["id"] for m in local]
    by_key = {}
    for m in local:
        by_key[normalize_key(m["id"])] = dict(m)
    if port:
        for m in discover_served(port):
            k = normalize_key(m["id"])
            e = by_key.get(k, dict(m))
            # merge served facts but keep the repo-style id for display
            for key, val in m.items():
                if val is not None and key != "id":
                    e[key] = val
            by_key[k] = e
    # Also surface the configured model even if neither source lists it
    # (e.g. MTPLX serves a normalized id whose repo is cached under another name).
    cur = fw_cfg.get("model")
    if cur and normalize_key(cur) not in by_key:
        repo = fw_cfg.get("repo") or cur
        d = cache_dirname(repo)
        by_key[normalize_key(cur)] = {
            "id": cur, "served": False,
            "in_cache": d is not None,
            "size_gb": _snapshot_size_gb(d) if d else fw_cfg.get("model_gb"),
            "context_length": _cached_context(d) if d else None,
        }
    out = []
    for m in by_key.values():
        c = compatibility(m, free_ram_gb, ctx_tokens)
        m["compat"] = c
        m["mtp_draft"] = mtp_draft_for(m["id"], local_ids)
        out.append(m)
    # served first, then ready, then the rest; stable by id
    rank = {"ready": 0, "tight": 1, "unknown": 2, "too-large": 3}
    out.sort(key=lambda m: (0 if m.get("served") else 1,
                            rank.get(m["compat"]["verdict"], 9), m["id"]))
    return out
