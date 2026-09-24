# benching — Configuration reference

All configuration lives in **`config.json`** at the project root. On first run it is
seeded from **`config.example.json`**. The server merges `config.json` over built-in
defaults, so you only need to specify the keys you care about.

> **Live editing:** the dashboard's *Model per Framework* panel writes `model` (and
> `ctx_tokens`) straight back to `config.json` via `POST /api/models/select`. Other
> keys are edited by hand, then restart the server.

## Top-level keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `host` | string | `"127.0.0.1"` | Interface the web UI binds to. **Keep `127.0.0.1`** — binding `0.0.0.0` exposes run control and run deletion to your whole network (the server warns loudly if you do). |
| `port` | int | `7090` | Port for the web UI. Override per-invocation with `python3 server.py <port>`. |
| `route_via_proxy` | bool | `false` | Route agent harnesses (pi/opencode/goose/hart) through the local measurement proxy so their rows get per-request PP/TGS/TTFT. `false` = agents connect straight to the framework (byte-identical to an independent harness run; agent rows then report TPS/QA only). |
| `pi_thinking` | string | `""` | pi thinking-level suffix appended to `--model` (e.g. `":high"`). Empty = inherit the server's reasoning setting so all cells share one uniform budget. |
| `hart_path` | string | `"~/Documents/hart/hart.py"` | Path to the `hart` agentic harness script. |
| `frameworks` | object | (see below) | One entry per framework. Merged per-framework over the defaults. |

## `frameworks.<id>` keys

Each framework has a stable id (e.g. `omlx`, `mtplx`, `mlxlm`, `mlxserve`). You can
**add** new frameworks or **remove** ones you don't use — the UI renders whatever is
here.

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `name` | string | yes | Display name (e.g. `"OMLX"`). |
| `model_source` | string | no | Where the framework loads models from: `"hf"` (the Hugging Face cache, the default) or `"mtplx"` (`~/.mtplx/models`). This controls which models the *Model per Framework* picker offers for that framework — MTPLX models are only offered to `model_source: "mtplx"` frameworks and vice-versa. |
| `port` | int | yes | Port the framework's inference server listens on. benching waits for `/v1/models` here before running, and reuses a healthy server already on this port. |
| `model` | string | yes | The model id the framework serves / should serve. This is the id used in requests. (Some frameworks normalize it — e.g. MTPLX serves `mtplx-qwen38-27b-optimized-quality`.) |
| `repo` | string | no | The Hugging Face repo behind a normalized served id (used for cache mapping in discovery and for the `{repo}` placeholder). Set this when `model` is a normalized id that differs from the repo name. For MTPLX this is also the id used to load the model from `~/.mtplx/models`. |
| `start_cmd` | string[] | yes | The exact command to cold-start the framework's server. May use the placeholders `{model}` (→ `model`) and `{model_path}` (→ the local HF-cache snapshot path when cached, else `model`). |
| `ctx_tokens` | int | no | Configured context window. Used for compatibility scoring and the ctx-fill % metric. |
| `model_gb` | number | no | Approximate weight size in GB. Used for RAM-fit scoring and the effective-bandwidth estimate. |
| `notes` | string | no | Free-text shown in the framework info modal — document *when* each parameter applies (per-request vs at-start). |

### Placeholders in `start_cmd`

- `{model}` → the value of `model` (the served/request id).
- `{repo}` → the value of `repo` (or `model` if `repo` is unset). Use this for CLIs whose
  `--model` flag needs the HF repo id rather than the normalized served id — this is how
  MTPLX loads a model from `~/.mtplx/models`.
- `{model_path}` → the resolved local HF-cache snapshot directory (a real path), when
  the model (or its `repo`) is cached; otherwise falls back to `model`. Use this for
  CLIs whose `--model` flag needs a real path rather than a repo id.

### Example (from `config.example.json`)

```json
"omlx": {
  "name": "OMLX",
  "model_source": "hf",
  "port": 7001,
  "model": "mlx-community--Qwen3.8-27B-8bit",
  "model_gb": 28,
  "ctx_tokens": 261000,
  "start_cmd": ["omlx", "serve", "--port", "7001"]
}
```

MTPLX loads from its own store (`~/.mtplx/models`) by repo id, so it uses
`model_source: "mtplx"` and the `{repo}` placeholder:

```json
"mtplx": {
  "name": "MTPLX",
  "model_source": "mtplx",
  "port": 7002,
  "model": "mtplx-qwen38-27b-optimized-quality",
  "repo": "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality",
  "model_gb": 28,
  "ctx_tokens": 261000,
  "start_cmd": ["mtplx", "serve", "--model", "{repo}",
                "--context-window", "261000", "--max-tokens", "32768",
                "--reasoning-effort", "low", "--port", "7002"]
}
```

## Adding a new framework

1. Add an entry under `frameworks` with a new id.
2. Set `name`, `port`, `model`, and `start_cmd`.
3. Restart the server. The new framework appears in the top strip, the selection grid,
   and the model picker automatically.

## Removing a framework

Delete its entry under `frameworks` and restart. (Its port is then free for anything
else.)

## What is *not* in config.json

- **Tasks / prompts** — defined in `server.py` (`TASKS`) and editable live in the UI.
- **Harness list** — defined in `server.py` (`HARNESS_LABELS`); the UI renders it from
  `/api/harnesses`.
- **Run history** — written to `runs/*.json`, not config.
