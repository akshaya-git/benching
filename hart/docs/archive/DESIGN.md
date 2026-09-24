> **Status note (2026-09-21):** this is the *original frozen design* (v1.0).
> The implementation has since evolved — Integrator merged into Summary,
> oneshot fast path, step-completion gates with acceptance criteria, repair
> cycles, epochs (auto-continuation), live narration/heartbeat/verbose wire
> dumps, lenient JSON parsing, `--plan-file` two-turn workflow, and benchmark
> integration as the fifth harness. **Current-state architecture and
> capabilities: [ARCHITECTURE.md](ARCHITECTURE.md).**

# agedIn — Design Document

**A fully agentic, multi-agent harness for local Apple-Silicon LLM backends**
Version 1.0 · 2026-09-19 · status: design frozen for MVP build
Prepared for: the agedIn project owner. Companion ecosystem: `/Users/darthvader/Documents/benchtest`.

---

## 1. Executive summary

agedIn is a single-command, zero-dependency CLI that takes a task ("build a
Tetris game in one HTML file"), talks to any OpenAI-compatible local backend
(OMLX, MTPLX, MLX-LM — and anything else speaking `/v1/chat/completions`), and
runs it to a **validated, integrated, documented deliverable** through a
pipeline of six independent agents: **Connection → Planning → Builder →
Validator → Integrator → Summary**.

Three properties drive every design decision:

1. **Resumable by construction** — the entire pipeline state lives in one
   on-disk blackboard (`state.json`); any agent can pick up any run where it
   stopped. Checkpointing is not a feature layered on top; it *is* the
   architecture.
2. **Portable across weak runtimes** — agedIn never uses native OpenAI
   tool-calling. All three target frameworks have inconsistent or missing
   `tools` support (observed: MTPLX emits `finish_reason=tool_calls` that thin
   clients never execute; mlx-lm's tool support is partial). Instead, every
   model interaction uses a **prompted JSON-action protocol** that works on any
   chat-completions endpoint.
3. **Measured like a benchmark citizen** — every model call is timed from the
   SSE stream (TTFT, prefill tok/s, decode tok/s, overall tok/s, token usage)
   and every run emits metrics shaped to drop straight into the benchtest
   `runs/*.json` schema, so agedIn becomes the ecosystem's fifth harness with
   zero schema work.

**MVP boundary** (buildable in one focused session): Connection + Planning +
Builder + Validator + Summary over the blackboard, with checkpoints, one
compaction trigger, loop hashing, and metrics. **Deferred**: Integrator's
holistic cross-plan verification pass, TUI/live progress view, an optional
`--serve` browser view, `runs --prune` cleanup, remote backends, project
memory across runs.

### Goals
- G1: `agedIn.py --task "…"` produces a working artifact + report, hands-off.
- G2: Any agent resumes any interrupted/failed run from disk, no special flags
  beyond `--resume`.
- G3: Works identically against OMLX :7001, MTPLX :7002, MLX-LM :7003 and any
  other `/v1/chat/completions` endpoint.
- G4: Full per-call streaming metrics (PP/TGS/TPS/TTFT/usage) + per-agent
  timing, compatible with benchtest history.
- G5: Zero pip installs; one Python file; macOS-first.

### Non-goals (v1)
- No parallel/multi-agent concurrency (pipeline is sequential by design — see §2).
- No GUI/TUI (plain stdout; `--quiet` supported).
- No cloud providers, auth, or multi-tenancy.
- No Windows/Linux packaging (Python portability comes free later).
- No training/fine-tuning, no RAG/vector stores.

---

## 2. Architecture

### 2.1 Composition — decision: **sequential state-machine with a shared blackboard**

Three candidates were weighed:

| Option | How it works | Why not / why yes |
|---|---|---|
| **A. Peer-to-peer message bus** | Agents run as independent loops exchanging messages over a queue; no central owner. | Best "independence" optics, but checkpointing means serializing an in-flight mailbox; debugging is hard; ordering failures (two agents mid-write) corrupt state. Overkill for a 6-stage pipeline. Rejected for MVP. |
| **B. Orchestrator with private agent memory** | Central conductor holds all state; agents are called like functions but keep private scratch. | Common, but private memory breaks "any agent picks up where it stopped" — resume requires replaying private context. Rejected. |
| **C. Sequential state-machine + shared blackboard** *(chosen)* | One `RunState` document is the *only* source of truth. A thin orchestrator loop walks agents in order; each agent is a pure-ish step: `run(state) -> state` (mutating blackboard + emitting events). Agent "independence" is achieved because each agent reads *everything* it needs from the blackboard and writes *everything* it produced back — no hidden memory. | Resume = load JSON, skip agents marked `done`, continue. Trivially debuggable (`cat state.json`), deterministic, and matches how the owner's benchmark already thinks (one result row per pipeline). |

Independence caveat, stated honestly: agents are independent in *state*, not in
*concurrency*. v1 runs them sequentially because later stages consume earlier
stages' outputs (Validator needs the Builder's files; Summary needs
everything). The blackboard keeps them swappable/replaceable — a different
model can drive each agent (per-agent model override arrives post-MVP; the
Connection module already supports it, see §3.1).

### 2.2 The blackboard: `state.json`

One JSON document per run, in `<workdir>/.agedIn/state.json`:

```json
{
  "version": 1,
  "run_id": "20260919-091500-a1b2",
  "goal": "Write hello.html: a single HTML page with an <h1>hello</h1> …",
  "workdir": "/abs/path/to/workdir",
  "backend": {"base_url": "http://127.0.0.1:7001/v1", "model": "scottlowry--Qwen3.8-27B-oQ6e-mtp", "api_key": "…"},
  "budget": {"max_steps": 40, "ctx_tokens": 262144, "per_call_max_tokens": 8192},
  "pipeline": {
    "connection": {"status": "done", "attempts": 1, "server_probe": {"healthy": true, "models": ["…"]}},
    "planning":   {"status": "done", "attempts": 1},
    "building":   {"status": "running", "attempts": 2, "step_index": 4},
    "validating": {"status": "pending", "attempts": 0},
    "integrating":{"status": "pending", "attempts": 0},
    "summary":    {"status": "pending", "attempts": 0}
  },
  "plan": {
    "steps": [
      {"id": 1, "title": "Create hello.html", "kind": "code",
       "detail": "Single HTML file, h1 + button + alert, inline JS", "done": false,
       "outputs": ["hello.html"]}
    ],
    "created_by": "planning", "revisions": 0
  },
  "context": [
    {"role": "system", "content": "…agedIn system prompt…"},
    {"role": "user", "content": "…goal + plan + recent events…"}
  ],
  "compactions": [{"at_event": 57, "tokens_before": 182000, "summary_ref": "c-01"}],
  "loop_guard": {"hashes": {"building:write_file:hello.html<md5>": 3}},
  "artifacts": ["hello.html"],
  "result": null
}
```

Rules that make this work:
- **Write-through**: every agent mutation is followed by an atomic
  `state.json` write (write temp + `os.replace`) *before* the next model call.
  A crash mid-call loses nothing but the call itself.
- **Append-only audit**: everything that happens is also appended to
  `.agedIn/events.jsonl` (`{ts, agent, event, detail}`) — the audit log is
  never rewritten; the blackboard is.
- **Context reconstruction** (§4.2) rebuilds each agent's model context from
  the blackboard + journal, never from a private in-memory transcript. This is
  what makes "any agent resumes anywhere" true.

### 2.3 Message contract: the **JSON-action protocol**

The Builder (and any agent that needs the model to *do* something) exchanges a
strict envelope with the model — no native `tools`:

**System prompt (excerpt, Builder):** "You are the Builder agent of agedIn.
Reply with exactly one JSON object, no prose, no markdown fences:

```json
{"think": "≤200 tokens of private reasoning",
 "action": "write_file | run_shell | request_info | done",
 "args": {"path": "hello.html", "content": "…"} ,
 "step_id": 4,
 "notes": "one line for the blackboard"}
```
"

**Action set (MVP):**

| action | args | effect on blackboard |
|---|---|---|
| `write_file` | `path`, `content` | writes file under workdir, records artifact |
| `run_shell` | `cmd` (≤200 chars), `timeout_s ≤ 60` | runs via `subprocess`, captures `rc/stdout[-2000]/stderr[-2000]` into the event journal + observation |
| `done` | `step_id` | marks plan step done, advances `step_index` |
| `request_info` | `question` | (post-MVP, interactive mode; MVP auto-answers "proceed with best judgment") |

**Protocol enforcement** (the reliability core):
1. Model output is passed through `extract_json()` — tries strict parse, then
   first `{…}` block, then fence-stripped parse (same graduated strategy the
   benchtest HTML extractor uses, which is field-proven on these models).
2. Unparseable / unknown action / missing required args → **repair retry**:
   the observation `{"error": "protocol", "hint": "<what was wrong>"}` is
   appended and the same call retried (max 2 repairs, then the step fails into
   the recovery path §4.4).
3. `run_shell` hardening: command is rejected if it matches a deny-pattern
   (`rm -rf ~`, `sudo`, `curl … | sh`); runs with `cwd=workdir`, 60 s timeout,
   output truncated. The model never gets an interactive shell.

The Planner and Validator use the same envelope with different verbs
(`plan` → `{"steps": [...]}`; `validate` → `{"verdict": "pass|fail",
"failures": [...], "fix_actions": [...]}`), so there is exactly **one**
protocol to harden, not six.

### 2.4 Component layout (single file, `agedIn.py`)

```
agedIn.py            # everything; ~1.2–1.6k LOC target
├── args / main()            CLI entry, run dir bootstrap
├── Blackboard               load/save/atomic-write state.json, events.jsonl
├── Connection (module)      health probe, /v1/models adopt, chat_stream()
│   └── chat_stream()        urllib SSE POST → yields chunks; per-call Metrics
├── MetricsRecorder          per-call + per-agent metrics; metrics.jsonl; result row
├── Protocol                 build agent prompts, extract_json(), repair loop
├── Agents: plan / build / validate / integrate / summarize
│   └── each: (state) -> state, honoring status/attempts, budget, loop guard
├── Resilience               compaction, loop hashing, budgets, recovery
├── QACheck (port)           node --check + weighted checks, from benchtest
└── Orchestrator loop        walk pipeline, resume, final result row
```

Deliberately **no** classes for agents — each agent is one top-level function
(`agent_plan(state)`, `agent_build(state)`, …) so the morning-build session can
implement and test them in isolation against a recorded fake server.

---

## 3. Agent specifications

All agents receive the blackboard; all emit journal events; all model calls go
through Connection with `MetricsRecorder` attached. "Budget" = hard stop that
fails the agent into recovery (§4.4) rather than hanging.

### 3.1 Connection (module, not a model-caller)
- **Responsibility**: the only code that talks HTTP. Owns `chat_stream(url,
  payload) -> (text, usage, finish_reason, metrics)` and `probe()`.
- **Behaviors**: on startup probes `/v1/models`; if the configured model id is
  absent it **adopts the served id** when exactly one is listed (benchtest
  convention; MTPLX normalizes `Youssofal/…` → `mtplx-qwen38-27b-…`), else
  warns and proceeds with the configured id. Streams with `stream_options:
  {"include_usage": True}` and silently retries without it on HTTP 400
  (mlx-lm). Time-to-first-token counts any delta kind (`content` **or**
  `reasoning_content`) — thinking models stream reasoning first (Qwen3.8 does).
- **Failure modes**: connection refused → actionable error ("is the framework
  up? benchtest starts them"); stream dies mid-read → call marked `aborted`,
  retried once by the calling agent; socket killed by external Stop → `RunStopped`
  surfaces as a clean run abort (same pattern as benchtest).

### 3.2 Planning
- **Input**: goal, workdir listing (files ≤ 50), backend caps.
- **Calls**: 1 model call (`max_tokens` 4k). System prompt: "Decompose into
  3–7 concrete steps; each step must produce a verifiable output (file / test
  result). Reply JSON: `{\"steps\":[{\"id\":1,\"title\":…,\"kind\":…,
  \"detail\":…,\"outputs\":[…]}]}`".
- **Output**: `state.plan` (schema above). Validation: 1–7 steps, ids unique;
  else one repair retry, then fallback plan `[{id:1, title: goal, kind: code}]`
  (degraded but never blocked).
- **Failure modes**: model returns prose → repair loop; returns 10+ steps →
  truncate to 7 with a journal note.

### 3.3 Builder / Executor
- **Input**: plan, current step, observations of previous actions (last 5 in
  context; older compacted).
- **Calls**: one call per action, looping until `done` for the step or budgets
  (`max_steps` per run, 8 actions per step). File writes are the *deliverable*
  path; `run_shell` is for quick checks (syntax, list files) not builds/tests —
  tests belong to the Validator.
- **Failure modes**: protocol violations (→ repair), identical-action loop
  (→ §4.4), file outside workdir (rejected), huge `content` (> 200 KB → reject,
  ask model to split).

### 3.4 Validator
- **Input**: artifacts list, plan, goal.
- **Behavior**: two phases. (a) **Deterministic checks, no model**: for each
  `.html` artifact run the QA gate ported from benchtest (`node --check` on
  inline JS + weighted checklist: doctype, closed doc, JS present, event
  handlers, run loop, canvas/DOM use, balanced braces, size, no placeholders;
  < 90% ⇒ not usable) — the exact checks, weights, and thresholds the owner
  already trusts. For code tasks: run `python -m py_compile` / `node --check`
  per file, plus any `test_*` files found. (b) **Model review**: one call with
  the QA results + artifact heads (≤ 200 lines each): verdict JSON
  `pass|fail` + concrete `fix_actions`.
- **Output**: `state.pipeline.validating` = `done(pass)` or `fail` with
  `fix_actions` appended to the blackboard. On fail with attempts < 2: control
  returns to **Builder** scoped to the fix actions (a bounded fix-loop: max 2
  validate→build round-trips, then Integrator is entered anyway with failures
  documented — the owner prefers honest failure over infinite polishing).
- **Failure modes**: node missing (QA degrades to checklist-only, noted);
  timeout on checks (60 s, counted as failure with evidence).

### 3.5 Integrator
- **Input**: all artifacts, plan, validation verdicts.
- **Calls**: 1–2 calls. (a) Holistic review: "the goal was X; the parts are …;
  is the whole coherent and complete? list integration gaps as JSON".
  (b) Writes/updates `README.md` in the workdir (model-drafted, describing
  what was built and how to run it) — via the same write_file protocol.
- **MVP note**: the holistic check is advisory in MVP (it annotates the
  summary; it does not gate). **Deferred**: closing integration gaps
  automatically with a build round-trip.

### 3.6 Summary
- **Input**: everything above.
- **Calls**: 1 call. Produces the human report printed to stdout (and saved to
  `REPORT.md`): what was built, QA scores, metrics table, how to run it,
  failures honestly listed. **The last line of stdout is always the machine
  line** `AGEDIN_RESULT {…}` — a compact JSON the benchmark (or scripts) can
  parse without scraping prose.

---

## 4. State & resilience

### 4.1 Checkpoints

There is exactly **one** checkpoint mechanism: the blackboard.

- **When written**: after every agent transition, every model call completion,
  every file write, and every plan mutation — atomic `os.replace` of a temp
  file. Frequency is high because the document is small (typically < 200 KB).
- **Restore protocol**: `agedIn.py --resume [run_id]` (default: newest run dir
  under `<workdir>/.agedIn/`). Load `state.json`; the orchestrator enters the
  first pipeline stage whose `status != "done"` and continues. Journal events
  with `ts > state.saved_at` (a crash between journal append and state write)
  are replayed idempotently: journal entries carry `agent` + `event`; replay
  only re-records observations that the state already reflects (dedup by
  event hash).
- **Independent pickup**: because agents read only the blackboard, "Builder
  died at step 4" resumes as "Planning: done → Building: step 4, attempts 2" —
  no replay of Planning's model calls. A user may also hand-edit
  `state.json` (documented, versioned) to force re-planning.

### 4.2 Context compaction

- **Trigger**: before each model call, estimate prompt tokens (chars/3.6
  heuristic, corrected by the API's `prompt_tokens` feedback) against
  `budget.ctx_tokens`; when the projected prompt exceeds **60%** of budget,
  compact; hard-fail the call if it would exceed **90%** after compaction.
- **What is dropped/kept** (per pipeline stage):
  - Planner context is always tiny (goal + listing) — never compacted.
  - Builder context keeps: system prompt, goal, **full plan**, last 5
    observations verbatim, one-line rollups of older observations, and the
    current file being edited is **re-supplied in full** (the model must see
    what it is editing — it is re-read from disk, never from history).
  - When compacting: a single model call summarizes the dropped observations
    into `state.compactions[]` ("did what, learned what, failed what"), and
    those one-liners replace them.
- **MVP simplification**: compaction is **observation-history-only**. File
  contents are re-read from disk on demand (so they cost prompt tokens per
  call but never accumulate in history). This sidesteps the classic
  "summarize my own code badly" failure entirely.
- **Deferred**: file-content chunked editing for files > context budget
  (MVP rejects > 200 KB writes anyway).

### 4.3 Repetitive-loop detection

- **Hash**: `sha1(agent | action | normalized-args)` where normalization
  lowercases paths and ignores whitespace-only content diffs. Stored in
  `state.loop_guard.hashes` with counts.
- **Thresholds**: identical action **3×** ⇒ first inject a system nudge
  ("you already did X; the result was Y; choose a different approach"),
  **4×** ⇒ force a re-plan (Planning is re-entered with a failure report),
  **5×** ⇒ fail the run with `loop_detected` (honest stop).
- **No-progress guard**: after each Builder step, hash the workspace
  (`stat` + content-hash of artifacts); if unchanged across 2 consecutive
  actions that claimed writes, treat as a loop hit.
- **Budgets as backstops**: `max_steps` (default 40), per-call `max_tokens`
  (default 8 192), per-call socket timeout 1 800 s, whole-run wall clock
  (`--time-budget`, default 3 600 s). Everything is configurable on the CLI.

### 4.4 Run lifecycle: abandoning, restarting, concurrent invocations

Real usage (observed in the ecosystem) includes canceling runs that look wrong
or slow and immediately starting a different task. Semantics:

- **Cancel** (Ctrl-C / SIGTERM / parent Stop): agedIn traps the signal, writes
  state atomically, exits 130. Killing the process closes its streaming
  sockets — the framework stops decoding immediately; no orphan GPU work.
- **Abandoned runs are frozen, not deleted.** Each run owns its workdir
  (`agedIn-run-<ts>/`); nothing is shared between runs, so an abandoned run
  blocks nothing. Resume later with `--resume <run_id>` (plain `--resume`
  takes the newest unfinished run).
- **New task after abandoning** → new workdir, fresh state, no interference;
  prior artifacts stay in the prior directory.
- **Same task, modified request** → default is a new run. Reusing the same
  `--workdir` requires `--fresh`, which *archives* the prior state to
  `state-<ts>.json.bak` (never silently deletes).
- **Hygiene**: `agedIn.py runs` lists runs (id, task head, status, age,
  artifact count, workdir path); `agedIn.py runs --prune <id|older-than>` is
  the explicit cleanup path. MVP ships `runs` (list only); prune is post-MVP
  (`rm -rf` works fine until then).
- **Concurrent invocations** are allowed only across *different* workdirs
  (each is fully self-contained). Same workdir twice = the second invocation
  exits with an error pointing at the lock marker (`.agedIn/LOCK`, stale-lock
  detection by pid liveness).

### 4.5 Failure recovery ladder

1. **Protocol violation** → repair prompt (2×) → step fail.
2. **Step fail** → re-attempt step (attempts ≤ 2) with failure observation.
3. **Agent fail** (e.g., Planner unparseable after repairs) → degrade
   (fallback plan) or skip-then-document (Integrator) per agent.
4. **Run fail** → blackboard records `result.status="error"` + reason;
   `--resume` always available. Nothing is ever silently swallowed — every
   degradation appends a journal event and lands in the Summary.

---

## 5. Metrics & instrumentation

### 5.1 Per-call capture (Connection)

Identical technique to benchtest `call_chat` (field-proven against all three
frameworks): streaming request; **TTFT** = first delta of any kind
(`content`/`reasoning_content`); decode window = first→last token; **PP** =
prompt_tokens / TTFT; **TGS** = completion_tokens / decode window; **TPS** =
completion_tokens / wall; usage from the trailing `usage` chunk
(`stream_options` with graceful 400-retry). `finish_reason` recorded
(`length` ⇒ `truncated` flag, reported, never hidden).

Each call appends to `.agedIn/metrics.jsonl`:

```json
{"ts": 1789812000, "agent": "building", "call": 7,
 "ttft": 0.42, "pp": 183.5, "tgs": 47.2, "tps": 39.1,
 "prompt_tokens": 412, "completion_tokens": 980, "wall": 25.1,
 "finish": "stop", "model": "scottlowry--Qwen3.8-27B-oQ6e-mtp"}
```

Per-agent rollups (calls, tokens, wall, errors) and tool-call counts live in
the final `agedIn-metrics.json`.

### 5.2 benchtest compatibility

Final machine line / metrics file carry a result row matching the ecosystem's
`runs/*.json` row exactly (verified against `runs/20260918-110404.json`):
`framework, model, harness:"agedIn", task, status, latency, tokens, tps,
output_url, qa_func, qa_qual, usable, truncated, ttft, tgs, pp,
prompt_tokens, error`. `output_url` is emitted as a `file://`-style relative
path when running standalone (benchtest replaces it with `/output/…` when it
hosts the artifact, as it already does for agent CLIs).

### 5.3 Integration as the 5th harness (designate, don't build here)

`server.py` gains one `HARNESS_CLIS`-style entry launching
`agedIn.py --base-url http://127.0.0.1:<port>/v1 --model <cfg model> --task …
--print --quiet --workdir <scratch>` — direct-to-framework routing like the
other agents (no proxy; ROUTE_VIA_PROXY path stays available). stdout's last
line `AGEDIN_RESULT {…}` gives benchtest a clean parse; artifacts land in the
scratch workdir where benchtest's existing `newest_html` collector finds them.
Estimated integration effort: **~30 lines in benchtest + this CLI contract**
(post-MVP, owner-led).

---

## 6. CLI / UX

```
agedIn.py --base-url URL --model ID [--api-key K] (--task "…" | --task-file F)
          [--print]              # non-interactive, exit on completion (default true)
          [--workdir D]          # default: ./agedIn-run-<ts>
          [--resume [RUN_ID]]    # default: newest interrupted run in workdir
          [--max-steps 40] [--ctx-tokens 262144] [--per-call-tokens 8192]
          [--time-budget 3600] [--effort low|medium|high]   # → reasoning-effort hints
          [--metrics-out FILE] [--quiet] [--fresh]           # ignore checkpoints
```

- stdout is human-readable progress (`[planning] ✓ plan: 4 steps`,
  `[building] step 2/4 → write_file hello.html (3.1 kB)`), ending with
  Summary + the `AGEDIN_RESULT {…}` machine line. `--quiet` prints only the
  machine line.

**Interface model (explicit):** the terminal *is* the product — same
interface class as pi / opencode / goose, which is what the target community
already uses. agedIn is a **pure client**: it dials out to the framework's
OpenAI-compatible port and **listens on no port itself** — nothing runs in
the background, nothing conflicts with the benchmark, nothing to secure.
Humans and benchtest consume it identically (subprocess + stdout); it exposes
no HTTP surface of its own. A browser view, if ever demanded, is an optional
`--serve` mode wrapping the same pipeline (deferred, post-MVP).
- Exit codes: 0 pass · 1 task failed · 2 protocol/infra error · 130 aborted.
- Honesty rules carried over from the ecosystem: truncated outputs flagged ⚠,
  QA < 90% reported as not-usable, failures never dressed up as success.

---

## 7. Technology choice

**Chosen: Python 3.9+ standard library only, one file (`agedIn.py`).**

| Option | Verdict | Reasoning |
|---|---|---|
| **Python stdlib-only** | **chosen** | Matches the ecosystem's proven pattern (benchtest does streaming SSE, HTTP servers, process control stdlib-only). `urllib` + `subprocess` + `json` + `hashlib` cover 100% of needs; SSE parsing is ~30 lines. Zero-install is a real adoption feature for the local-LLM community and makes the benchtest integration one `which` check away. `asyncio` deliberately avoided — one request at a time, threads unnecessary. |
| Python + deps (httpx/pydantic) | rejected for MVP | Nicer ergonomics, but installs friction + version drift for zero capability gain at this scale. Revisit if a plugin ecosystem emerges. |
| Node/TypeScript | rejected | Great streaming story, but splits the ecosystem's language base (benchtest is Python; QA gate shells to `node` only for syntax checks) and adds runtime-management burden. |
| Swift | rejected | Native macOS polish the project doesn't need; kills cross-platform future; slowest iteration. |

Packaging (post-MVP): keep single-file as the primary distribution
(`curl`-able), plus an optional Homebrew formula. A `.app` wrapper is explicitly
**out of scope** until the CLI is community-proven.

---

## 8. Quality strategy

- **Deterministic checks outrank model opinions**: the Validator's QA gate
  (ported from benchtest: `node --check` on inline JS, weighted functionality
  checklist, 90% usability threshold, `py_compile`/`node --check` for code
  files) runs before any model review. The model reviews *with* those results
  in hand; it cannot overrule a hard syntax failure (capped at 25% QA, same as
  benchtest).
- **Bounded fix-loop**: validate→build→validate at most twice, then ship with
  documented failures. Chasing 100% with a 27B local model is how harnesses
  burn 40 minutes producing nothing (observed repeatedly in the ecosystem).
- **Protocol repair loop** (§2.3) is the single highest-leverage quality
  mechanism: it converts the most common local-model failure mode (sloppy
  JSON) from a crash into a self-correcting retry.
- **Honest artifacts**: truncation flags, QA scores, and failure lists flow
  into Summary verbatim; the harness never marks a run `pass` with a
  not-usable artifact.
- **Self-review pass** (deferred): a second Validator model call with fresh
  eyes on the integrated whole. MVP ships the Integrator's advisory review.

---

## 9. Risks & mitigations

| # | Risk | Mitigation |
|---|---|---|
| 1 | Frameworks differ in SSE details (usage chunk, keep-alives, `finish_reason`) | Connection reuses benchtest's battle-tested parser: `[DONE]` break, graceful `stream_options` 400-retry, usage-optional, keep-alive tolerant |
| 2 | Model ignores the JSON-action protocol (esp. small/quantized backends) | Strict envelope + `extract_json()` graduated parse + repair retries + fallback plans; protocol is the only one to harden |
| 3 | MTPLX serves a normalized model id ≠ configured | Connection adopts `/v1/models` id when unambiguous (benchtest pattern) |
| 4 | Thinking models eat budgets before answering (observed: 16k reasoning, 30-min stalls) | Server-side `reasoning_effort low` documented as recommended backend config; per-call `max_tokens`; TTFT counts reasoning so metrics stay honest |
| 5 | Runaway loops burn hours (observed: 60-min opencode cells) | Loop hashing (3/4/5 ladder), no-progress workspace guard, step/run budgets, wall-clock kill |
| 6 | Context overflow mid-build on long artifacts | Files re-read from disk per call (never accumulated), observation compaction at 60% budget, 200 KB per-write cap |
| 7 | Crash/corruption of state.json mid-write | Atomic `os.replace` + journal replay; worst case = redo current call |
| 8 | `run_shell` abused by model | Deny-patterns, cwd jail, 60 s timeout, output caps, no interactive commands |
| 9 | Owner's Stop button kills benchmark but not agedIn subprocess | agedIn handles SIGTERM: writes state, exits 130 — resume works (benchtest already kills process groups) |
| 10 | Scope creep before MVP (TUI, plugins, memory) | Non-goals list §1; MVP markings in this doc are the contract for the morning build |

---

## 10. Implementation plan & level of effort

Assumptions: **1 experienced Python developer, full-time**, frameworks already
installed and their quirks known (they are — documented above), no context
switching. Calendar time ≈ LOE × 1.5 if evenings/weekends only.

| Milestone | Content | Days |
|---|---|---|
| M0 | This design frozen | 0 (done) |
| M1 | Core skeleton: CLI, blackboard (atomic writes), Connection (probe, adopt, streaming + metrics), JSON-action protocol + repair, Planner, Builder with `write_file`/`run_shell`, Summary → **one task runs end-to-end on OMLX** | **3–4** |
| M2 | Validator (QA-gate port, checks + model review) + bounded fix-loop + Integrator + honest REPORT | **2** |
| M3 | Resilience: checkpoint/resume polish, compaction, loop ladder, budgets/time-kill, SIGTERM safety | **2–3** |
| M4 | Three-framework validation tour (OMLX/MTPLX/MLX-LM) + fixes for each backend's quirks; metrics/result-row parity with `runs/*.json` | **1–2** |
| M5 | benchtest integration as 5th harness (CLI contract, scratch workdir, `AGEDIN_RESULT` parse) | **1** |
| M6 | Community release: README, samples, license, versioned single-file drop, demo GIF/script | **1–2** |
| **Total** | | **10–14 working days** (~2–3 calendar weeks part-time) |

Morning-session prototype (the 9 AM run) maps to **M1 + smoke slices of
M2/M4** — roughly 40% of total effort, deliberately front-loaded so the
risky parts (protocol hardening, streaming metrics) exist on day one.

---

## 11. Open questions (owner)

1. Per-agent model mixing (e.g., Planner on the big model, Builder on the fast
   one) — worth a `--planner-model` flag post-MVP?
2. Interactive mode (`request_info` → stdin) — needed for HITL tasks, or is
   autonomous-only fine for v1?
3. Should the Validator's QA gate import benchtest's checks live (DRY but
   couples projects) or keep the ported copy (duplicated but standalone)?
   Default: ported copy.
4. Memory across runs (project profile: "this codebase uses pytest, prefers
   dark UI") — post-MVP file or deliberate non-feature?
5. Windows/Linux: free from stdlib Python — test-and-document now or later?

---

## Assumptions (decisions made without asking)

- Sequential pipeline; "independent agents" = state-independent & resumable,
  not concurrently running (§2.1).
- No native OpenAI tool-calling anywhere; JSON-action protocol instead.
- OMLX/MTPLX/MLX-LM started externally (by the owner or benchtest); agedIn
  probes and errors helpfully rather than managing server lifecycles (unlike
  benchtest, which must).
- Artifacts are files in `--workdir`; no in-memory deliverables.
- Single user, localhost backends, no auth beyond optional `--api-key`.
- English-only prompts; workdir sizes < a few hundred files.
- `node` is available for QA syntax checks (it is, on this machine); QA
  degrades gracefully without it.
