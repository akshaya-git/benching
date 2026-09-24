# benching — Troubleshooting

Quick fixes for the common failure modes. If something's wrong, check **`logs/bench.log`**
first — every framework start/stop, cell result, and error is timestamped there.

## The server won't start

**"Could not bind 127.0.0.1:7090"**
Another benching server (or a framework) is already using the port.
- Stop the other server, or
- Use a different port: `python3 server.py 7100` (or change `port` in `config.json`).

**"config.json unreadable"**
Your `config.json` has a JSON syntax error. The server falls back to built-in defaults
so it still starts — fix the JSON (a trailing comma is the usual culprit) and restart.

## A framework shows red (down) or never comes up

- **CLI not found** — the log says `<name> CLI not found`. Install the CLI or fix
  `start_cmd` in `config.json`. `./install.sh` lists what's missing.
- **Server crashed on start** — the framework's own stdout/stderr is captured to
  **`logs/<fw>-<timestamp>.log`**. Open the newest one for that framework; it has the
  real error (OOM, bad flag, model not found, port in use).
- **Model not in cache** — if the model isn't in `~/.cache/huggingface/hub`, the
  framework may try to download it (slow) or fail. Use the *Model per Framework* panel
  to pick a model that's already cached (verdict **ready**).
- **Port already in use** — if something else is on the framework's port, benching
  *reuses* it if it answers `/v1/models` healthily, otherwise the start fails. Free the
  port or change the framework's `port`.

## A cell errors out

- **"empty response"** — the model returned nothing and produced no artifact. Usually a
  model/server mismatch or a context overflow. Check the framework's log.
- **"stopped by user"** — you hit Stop; expected.
- **Truncated (⚠)** — the response hit `max_tokens`. Raise *Max tokens* (agents) or
  *Raw max tokens* and rerun, or use **raw+** which auto-continues.
- **QA below 90%** — the artifact was produced but scored low. Open the **Output** link
  to see what it actually made; this is a real capability datum, not a bug.

## The model picker shows everything as "too-large"

Your **free RAM** is low (another app or a running framework is using it). The verdict
is `size_gb × 1.25 > free RAM`. Close other apps / stop running frameworks, or pick a
smaller model. Total RAM is shown in the top strip; the verdict uses *free* RAM.

## Agent harness (pi / opencode / goose / hart) rows look empty or fail

- **CLI not installed** — install it (`./install.sh` shows the command).
- **`hart` not found** — set `hart_path` in `config.json` to the real path of `hart.py`.
- **Isolated config** — agents run with an isolated config dir under `harness-configs/`.
  If you changed how an agent authenticates, note that benching uses a throwaway key
  (`bench`) and points the agent at `http://127.0.0.1:<port>/v1`.
- **pi thinking** — if pi rows behave differently, check `pi_thinking` in `config.json`.

## Metrics look wrong or missing

- **PP/TGS show "–" for a cell** — that framework doesn't expose server-side counters
  for that harness, and it wasn't routed through the proxy. Set `route_via_proxy: true`
  (and restart) to get per-request PP/TGS/TTFT on agent rows.
- **raw+ TGS looks off** — raw+ sums decode time across continuation rounds; a high
  continuation count (see the `continuations` field) means the task exceeded one
  `max_tokens` budget.

## The UI looks stale / broken

- **Hard-refresh** (Cmd+Shift+R) — the frontend is a single file; a stale cache is the
  usual cause.
- **Restart the backend** if an endpoint 404s (the endpoint is newer than your running
  server).
- The UI polls `/api/state` every 1.2s; if the server is restarting, the poll silently
  retries — give it a moment.

## Reset everything

```bash
make clean        # removes runs/, outputs/, logs/, work/, harness-configs/*
rm config.json    # (optional) force re-seed from config.example.json on next start
```
