# agedIn — Architecture & Capabilities (current state)

This document describes agedIn **as built** (v0.1.x, 2026-09-21). The original
frozen design lives in [DESIGN.md](DESIGN.md); the change log of the review
driven improvements is in [IMPROVEMENTS.md](IMPROVEMENTS.md); live validation
evidence is in [VALIDATION.md](VALIDATION.md).

**One file, zero dependencies:** `agedIn.py` (~2 000 lines, Python 3.9+
stdlib only). A pure client — listens on no port, starts no servers; it dials
any OpenAI-compatible backend (`/v1/chat/completions`).

---

## 1. Pipeline architecture

Sequential state-machine over a shared **blackboard** (`<workdir>/.agedIn/state.json`,
written atomically after every transition — the checkpoint *is* the
architecture). Five stages; the original Integrator is merged into Summary.

```
connection → planning → building ⇄ validating → summary
                          ↑ repair cycles / re-plan / fix-loop bounce back here
```

| Stage | What it does now |
|---|---|
| **connection** | `/v1/models` probe with served-id adoption; **capability probe** (strict-JSON call — calibrates tokenizer ratio, baselines TTFT/TGS, warns early if the model can't emit JSON); **cold-model warm-in** when probe TTFT > 5 s (27 GB weight loads) |
| **planning** | Decomposes into *as many steps as the task genuinely needs* — neutral step count, each step carries **acceptance criteria**; plan self-check (outputs present, **duplicate-output steps merged** — the #1 churn source, detail ≤ 500 chars); template fallback plans; `--plan-file` loads an external plan (JSON or markdown checklist) |
| **building** | **Oneshot fast path** for single-step/single-artifact tasks (one call writes the complete file — pi-parity speed); otherwise batched JSON-action loop (write_file / read_file / list_files / run_shell / done); per-target read-result retention (no re-read churn); **step-completion gate** — `done` is accepted only when the step's outputs pass deterministic checks; loop ladder (nudge → re-plan → repair); budget escalation warnings (55/75/90 %) + final rescue-delivery call |
| **validating** | Deterministic gate first (**profile-aware QA**: canvas artifacts get the full game checklist, plain pages are scored on applicable checks; `node --check` on inline JS; `py_compile` for Python) → model review judged against the goal **and the plan's acceptance criteria** → **self-heal**: the Validator's executable write_file patches are applied and re-checked with zero extra model calls → **regression guard** (a fix that lowers QA is reverted) → progress-aware fix-loop (identical failure signature two rounds in a row stops the loop) |
| **summary** | Merged Integrator: one call produces integration gaps + project README + final report (deterministic fast path on clean oneshot builds); always ends stdout with the machine line `AGEDIN_RESULT {…}` |

**Protocol:** all model interaction uses one prompted **JSON-action envelope**
(no native tool-calling — the portability layer across OMLX/MTPLX/MLX-LM).
Parsing is **graduated and lenient**: strict JSON → `strict=False` (models
embed literal newlines inside strings — the #1 repair-loop trigger, 11→1
repairs when fixed) → fenced → balanced-span extraction, with a repair-retry
that feeds the exact error back to the model.

## 2. Resilience & autonomy

- **Checkpoints everywhere**: atomic state writes + append-only
  `events.jsonl` journal + per-call `metrics.jsonl`. SIGTERM/Ctrl-C → state
  saved, exit 130, `--resume` continues any agent from any stage.
- **Epochs (default ON)**: when any per-epoch budget is exhausted, the run
  checkpoints, hard-trims context, resets budgets, and **continues in a fresh
  epoch** — unattended days-long runs work like pi. Default 50 epochs × 7 200 s.
  A **no-progress guard** (two consecutive epochs with zero workspace change)
  stops pathological spinning.
- **Repair cycles** (3 default): builder-level failures (loops, protocol
  breakdown, no-progress, step-budget exhaustion) become *diagnosed retries*
  — the exact error plus a "take a different approach" directive — instead of
  a dead run.
- **Budgets are safety valves, not the plan**: `--max-steps/--time-budget/
  --epochs/--repair-rounds` exist for hard walls; the default behavior is
  fastest-path-first, then take whatever time quality requires.
- **Compaction** at 60 % of context: observation history summarized via a
  cheap model call, files always re-read from disk (never accumulated),
  per-run token-ratio calibration from real `prompt_tokens` feedback.

## 3. Observability (no black boxes)

- **Live narration**: stage descriptions, the full plan with acceptance
  criteria, every model call (tokens in/out, tok/s, TTFT), every tool call
  with ok/ERR status, `[next] step N` previews.
- **In-flight heartbeat** (always on, stderr, 20 s): `⏳ model call in
  flight — 62s, 1 840 tok streamed` — stuck vs. generating is instantly
  distinguishable (crucial on throttled backends).
- **`--verbose`**: full request and response bodies per call, pi-style.
- **`AGEDIN_RESULT {…}`**: machine-parseable final line (status, calls,
  tokens, tps, QA, artifacts, workdir) — what the benchmark consumes.
- `agedIn runs`: registry of all runs with status.

## 4. Differentiated capabilities vs other harnesses

| Capability | raw API | pi | opencode | Goose | **agedIn** |
|---|---|---|---|---|---|
| Agentic file/shell tools | — | ✅ | ✅ | ✅ | ✅ (jailed: cwd, deny-patterns, 60 s) |
| Deterministic quality gate (QA %, usable threshold) | — | — | — | — | ✅ **profile-aware, node --check** |
| Plan with per-step acceptance criteria | — | — | — | — | ✅ |
| Step completion gated on output quality | — | — | — | — | ✅ |
| Self-healing validation (executable patches, zero extra calls) | — | — | — | — | ✅ |
| Repair cycles on failure (diagnose → different approach) | — | partial | partial | partial | ✅ bounded + journaled |
| Checkpoint/resume any agent mid-run | — | sessions | sessions | sessions | ✅ **blackboard, any stage, hand-editable** |
| Auto-continuation past budgets (epochs) | — | ✅ (compacts/resumes) | ✅ | ✅ | ✅ default-on, no-progress-guarded |
| Per-call SSE metrics (TTFT/PP/TGS/TPS, usage) | caller-side | — | — | — | ✅ **every call, model-side** |
| Live narration + heartbeat + wire dumps | — | ✅ transcript | ✅ | partial | ✅ all three |
| Framework presets (OMLX/MTPLX/MLX-LM served ids) | — | config | config | config | ✅ one flag, id auto-adoption |
| Two-turn repo workflow (analyze → FIXPLAN.md → `--plan-file` execute) | — | manual | manual | manual | ✅ first-class |
| Zero dependencies, single file | n/a | npm | npm | binary | ✅ stdlib Python |

Honest trade-off: agedIn spends **more total tokens and calls than pi** on
equivalent tasks (plan + validation + report ceremony) — that is the cost of
its verification guarantees. On quality-gated comparisons it reaches pi-parity
speed on single-artifact tasks (oneshot path: 4 calls vs pi's 4, 201–288 s)
and exceeds pi on tasks where validation/repair actually matters.

## 5. Workflows

```bash
# single artifact (fast path)
agedIn --framework omlx "Write tetris.html: a playable Tetris game"

# multi-file project
agedIn --framework mtplx --workdir site "Build a 3-page static site: index.html, styles.css, app.js"

# turn 1: repo analysis (read-only) → analysis/ + FIXPLAN.md
agedIn --framework mtplx --workdir /path/to/repo --max-steps 100 \
  "Perform a full codebase analysis… write a prioritized FIXPLAN.md. Do NOT modify sources."

# turn 2: execute the plan
agedIn --framework mtplx --workdir /path/to/repo --plan-file FIXPLAN.md \
  --task "Execute the P0 items from FIXPLAN.md"

# long-horizon (defaults handle it; epochs auto-continue)
caffeinate -dims agedIn --framework omlx --workdir big "…CHIP-8 emulator…"
```

## 6. Benchmark integration (5th harness)

`benchtest/server.py` launches agedIn per cell with `--base-url` (direct to
the framework under test), `--model` (served id), a scratch `--workdir`, and
cell bounds (standard 2×1 500 s; LONG tasks 3×2 400 s). Results: tokens and
call counts parsed from `AGEDIN_RESULT`; artifacts collected via the same
`newest_html` pipeline as the other agent harnesses; QA scored by the
benchmark's own profile-aware gate so **all five harnesses are compared on
identical terms**.

## 7. Known limits

- Sequential pipeline (state-independent agents, not concurrent).
- Quality of semantic review is bounded by the backend model; deterministic
  gates are the floor, not the ceiling.
- Backend throughput swings (measured 14–60 tok/s on identical configs across
  days) dominate wall-time comparisons — normalize before comparing runs.
- No native tool-calling, no GUI/serve mode, no cross-run project memory
  (all deliberate; see DESIGN.md non-goals).
