# agedIn — Design Review & Improvement Recommendations

Review of `DESIGN.md` v1.0 (2026-09-19). Focus: performance and output quality.

---

## Overall Assessment

**Strengths:**
- The "checkpointing *is* the architecture" framing is excellent — it eliminates an entire class of bugs
- The JSON-action protocol is the right call for weak local runtimes
- Honest scoping (MVP boundary, non-goals, deferred items) makes this buildable
- The failure recovery ladder (§4.5) is well-ordered
- Single-file, stdlib-only is the right adoption strategy for this community

**Core tension:** The design optimizes heavily for *resilience* (resume, repair, loop detection, compaction) but relatively less for *throughput* and *output quality*. For a 27B model on Apple Silicon where each call is 10–60s, the difference between 12 calls and 18 calls is the difference between a 5-minute and an 8-minute run.

---

## Performance Improvements

### 1. Reduce model call count (highest leverage)

The current design makes these calls in a typical run:
- Planning: 1
- Builder: N (one per action, up to 8/step × 7 steps = 56 worst case)
- Validator: 1 (model review) + deterministic checks (free)
- Integrator: 1–2
- Summary: 1
- Compaction: 0–K (unbounded)

**Suggestions:**

- **Batch Builder actions.** Instead of one call per action, let the model emit a *sequence* of actions in a single response:
  ```json
  {"actions": [
    {"action": "write_file", "args": {"path": "index.html", "content": "..."}},
    {"action": "write_file", "args": {"path": "style.css", "content": "..."}},
    {"action": "run_shell", "args": {"cmd": "ls -la"}}
  ], "step_id": 2}
  ```
  This is safe because the actions within a step are independent (they're all "produce the files for this step"). You execute them sequentially but the model only thinks once. This could cut Builder calls by 40–60% for multi-file steps. The repair loop still works per-batch.

- **Merge Integrator + Summary.** The Integrator's holistic review and the Summary are both "look at everything and write prose." One call can do both: produce the integration assessment *and* the final report. Saves a call, and the model has the full context in one shot.

- **Skip Planning for trivial tasks.** Add a heuristic: if the goal is < 80 chars and mentions a single file type, skip Planning and go straight to Builder with a synthetic single-step plan. This is the "hello.html" fast path. Saves 1 call + the latency of a 4k-token generation.

- **Amortize compaction.** Instead of a dedicated compaction call at 60% budget, do *incremental summarization at step boundaries*. After each Builder step completes, append a one-line summary of that step's observations to a running `step_rollups[]` array. This is free (it's just truncation + the model's own `notes` field from the action envelope). The expensive "summarize dropped observations" call becomes rare — only needed if a single step generates > 5 observations.

### 2. Connection-level optimizations

- **Keep-alive / connection reuse.** `urllib` opens a new TCP connection per call. For a pipeline making 10–20 calls, that's 10–20 × ~50ms of handshake overhead on localhost (negligible) but more importantly, some frameworks (MTPLX) have slow connection setup. Use `http.client.HTTPConnection` with a persistent connection, or at minimum set `Connection: keep-alive` and reuse the socket.

- **Prefill the context.** The system prompt + goal + plan are the same across all Builder calls. If the framework supports prompt caching (some do via `cache_control` or similar), mark the stable prefix. If not, at least structure the prompt so the stable prefix is *byte-identical* across calls (some frameworks do prefix matching for KV cache reuse). This means: system prompt → goal → plan → [variable: observations + current step]. Never reorder.

- **Parallel deterministic checks.** The Validator's `node --check` / `py_compile` calls are independent per file. Run them with `concurrent.futures.ThreadPoolExecutor` (even in stdlib Python, this is fine for I/O-bound subprocess calls). For a 5-file project, this saves ~2–4s.

### 3. Context window efficiency

- **The chars/3.6 heuristic is too rough.** For Qwen3.8's tokenizer, English text is closer to chars/3.2, but code (which is what the Builder produces) is closer to chars/2.8. Since you're correcting with `prompt_tokens` feedback anyway, consider a *per-call-type* calibration: after the first call, you know the actual ratio for that content type. Store it in state and use it for subsequent estimates. This makes the 60% compaction trigger more accurate and avoids both premature compaction (wasted call) and late compaction (truncation).

- **Observation window: 5 is too small for debugging.** When the Builder is stuck (which is when you most need context), 5 observations might not include the relevant failure. Consider: keep the last 3 *verbatim* + a structured summary of the previous 10 (one line each: `step 3, action 2: write_file hello.html → 2.1kB, rc=0`). This gives the model both detail and history without the token cost.

- **File re-supply strategy.** "The current file being edited is re-supplied in full" is correct for correctness but expensive for large files. Add a threshold: if the file is < 200 lines, supply in full. If > 200 lines, supply the first 50 + last 50 lines + a line-count note, and let the model request specific ranges via a `read_file` action (new action, cheap: `{"action": "read_file", "args": {"path": "…", "start": 100, "end": 150}}`). This keeps the protocol simple but avoids blowing context on a 500-line file when the model only needs to change line 230.

---

## Quality Improvements

### 4. Protocol hardening

The current enforcement is: parse → check action name → check required args. Add:

- **Schema validation per action.** Define the expected arg types explicitly:
  ```python
  ACTION_SCHEMAS = {
      "write_file": {"path": str, "content": str},
      "run_shell": {"cmd": str, "timeout_s": int},
      "done": {"step_id": int},
  }
  ```
  Type mismatch → repair hint: `"args.path must be a string, got int"`. This catches a common failure mode where models emit `"path": 123` or `"content": null`.

- **Content sanity checks before execution.** Before `write_file` actually writes:
  - If `content` is empty or < 10 chars → reject with hint
  - If `content` looks like a *description* of the file rather than the file itself (heuristic: starts with "This file contains" or "The file should") → reject with hint "provide the actual file content, not a description"
  - If `path` contains `..` or absolute path → reject (you have this, but make the error message specific)

- **Idempotency for file writes.** If the model writes the same path twice within a step, the second write silently overwrites. Add a journal event and a nudge: "you already wrote hello.html in this step; are you intentionally overwriting?" (auto-continue after the nudge, but log it). This catches a common model behavior of "writing a file, then writing it again with a slightly different version" which wastes a call.

### 5. Plan quality

- **Add a `depends_on` field to plan steps.** Even if execution is linear, knowing dependencies helps the Builder understand *why* step 3 comes before step 4, and helps the Validator check that prerequisites were met. Schema:
  ```json
  {"id": 3, "title": "Add event handlers", "depends_on": [1, 2], ...}
  ```

- **Plan self-check.** After the Planner produces steps, add a *deterministic* validation (no model call):
  - Every step has at least one `outputs` entry
  - No two steps produce the same output file (conflict)
  - Step count is 1–7
  - If a step's `kind` is "test" but no prior step produces the file being tested → flag as suspicious (don't block, just note)

- **The fallback plan is too coarse.** `[{id:1, title: goal, kind: code}]` means the Builder gets the entire goal as a single step with 8 actions. For a "build a Tetris game" goal, this will fail. Better fallback: a *template* plan based on task type detection:
  ```python
  if "html" in goal.lower():
      fallback = [
          {"id": 1, "title": "Create HTML structure", "outputs": ["index.html"]},
          {"id": 2, "title": "Add styling", "outputs": ["index.html"]},
          {"id": 3, "title": "Add interactivity", "outputs": ["index.html"]},
      ]
  ```

### 6. Validator improvements

- **Add a "smoke test" action.** For HTML artifacts, the strongest validation is: does it actually render? Add an optional `run_shell` check: `open -a "Google Chrome" --args --headless --dump-dom <file>` (macOS) or use `node -e "require('puppeteer')…"` if available. Even simpler: check that the HTML has a `<body>` with non-empty content and no unclosed tags (you have the balanced-braces check, extend to HTML tags).

- **Model review prompt should include the goal verbatim.** The model needs to judge "does this accomplish what was asked?" not just "is this well-formed?" Make the prompt: "The goal was: {goal}. The artifacts are: {heads}. The QA gate scored: {scores}. Does this accomplish the goal? If not, what specific changes would make it work?"

- **Fix actions should be executable.** The current `fix_actions` is vague. Constrain it to the same action vocabulary as the Builder:
  ```json
  {"fix_actions": [
    {"action": "write_file", "args": {"path": "hello.html", "content": "…full corrected content…"}},
    {"action": "run_shell", "args": {"cmd": "node --check hello.html"}}
  ]}
  ```
  This means the Builder can execute them directly without re-interpreting. The Validator is essentially producing a *patch plan*.

### 7. Loop detection refinement

- **Include step_id in the hash.** Currently: `sha1(agent | action | normalized-args)`. If the model writes `hello.html` in step 2 and then (correctly) writes `hello.html` again in step 5 with different content, the hash collides. Use `sha1(agent | step_id | action | normalized-args)`.

- **The "no-progress" guard is good but add a "quality regression" guard.** If the Validator scores 85% on round 1 and 72% on round 2 (after a fix), the fix made things *worse*. Detect this and revert to the round-1 artifact (keep a copy before each fix attempt). This is cheap: just `cp artifact artifact.bak` before the fix loop.

### 8. Observability & debugging

- **Add a `--verbose` flag** that prints the full model input/output for each call to stderr (stdout stays clean for the machine line). This is the #1 debugging need when the protocol repair loop fires.

- **Add a `--dry-run` flag** that runs Planning + shows the plan + exits. No Builder, no file writes. Useful for: "let me see what it *would* do before committing 10 minutes of GPU time."

- **Structured event types in events.jsonl.** Currently `{ts, agent, event, detail}`. Add a `type` field with an enum: `call_start`, `call_end`, `action_exec`, `action_result`, `repair`, `compaction`, `loop_nudge`, `budget_warning`, `state_write`. This makes the journal queryable: `grep '"type":"repair"' events.jsonl` shows all protocol failures.

---

## Architectural Suggestions

### 9. Add a "Model Capability Probe" (pre-pipeline)

Before Planning, make one small call (max_tokens=100) with a strict JSON prompt:
```
Reply with exactly: {"ok": true, "count": 7}
```
This tells you:
- The model *can* produce valid JSON (if not, you know the repair loop will fire constantly — warn the user)
- The actual TTFT/TPS for this model (calibrates your token estimation)
- The model's actual output format (some models add a trailing newline, some don't)

Cost: ~2–5s. Benefit: avoids a 10-minute run that fails at step 3 because the model can't do JSON.

### 10. Consider a "confidence" signal from the model

Add an optional field to the action envelope:
```json
{"think": "…", "action": "write_file", "args": {...}, "confidence": 0.8}
```
If the model reports low confidence (< 0.5) on a `write_file`, the harness can:
- Log a warning
- Trigger an immediate re-read of the file after writing (verify it's what was intended)
- In the Validator, flag low-confidence artifacts for extra scrutiny

This is cheap (one float) and gives you a signal you don't have otherwise. Models are often "confident" about wrong outputs; asking them to self-assess is a known calibration technique.

### 11. State file size management

You note state.json is "typically < 200 KB" but the `context` array (full conversation history) will grow. For a 40-step run with observations, this could hit 1–2 MB. The atomic write is still fast, but:

- **Don't store the full `context` array in state.json.** Store only the *parameters* needed to reconstruct it (goal, plan, step_index, last_N_observations, compaction refs). The context is *derived*, not stored. This keeps state.json < 50 KB always and makes the "context reconstruction" in §4.2 the *only* path (not a fallback).

- **The `plan.steps[].detail` field** can grow if the model writes verbose details. Cap it at 500 chars in the schema validation.

---

## Minor / Quick Wins

| # | Change | Why |
|---|--------|-----|
| 12 | Add `--seed` flag (passed to the backend if supported) | Reproducibility for benchmarking |
| 13 | `run_shell` output: include `rc` in the observation, not just stdout/stderr | The model needs to know if the command *failed* |
| 14 | Add a `list_files` action (cheaper than `run_shell: ls -la`) | Models often waste a shell call just to list files; a dedicated action is faster and safer |
| 15 | The `AGEDIN_RESULT` line should include `run_id` and `workdir` | Makes post-hoc analysis easier without parsing the report |
| 16 | Add `--max-fix-rounds N` (default 2) to the CLI | The "bounded fix-loop" is a quality knob the user should control |
| 17 | Journal events should include `tokens_in`/`tokens_out` for model calls | Makes it easy to compute per-agent token cost from the journal alone |
| 18 | The `--effort` flag should also control `max_tokens` (low=4k, medium=8k, high=16k) | Currently it only sends a "reasoning-effort hint" which many backends ignore; tying it to max_tokens makes it effective everywhere |

---

## What I'd Cut or Defer

- **The Integrator as a separate agent** (MVP). Its value is marginal over a good Summary prompt. Merge it into Summary for MVP; split it out post-MVP when you have data on whether the holistic review actually catches things the Validator missed.

- **`request_info` action** (already deferred, but I'd remove it from the MVP action set entirely). Even as "auto-answers proceed with best judgment," it adds a code path that never fires in MVP. Remove from the protocol; add post-MVP.

- **The `--serve` browser view** (already deferred, but I'd remove it from the document entirely). It's a non-goal that keeps getting mentioned, which creates scope-creep pressure. If it's not in the MVP or the next milestone, it shouldn't be in the design doc.

---

## Summary of Highest-Impact Changes (if you only do 5)

1. **Batch Builder actions** (one call → multiple file writes) — cuts call count 40%+
2. **Model Capability Probe** before the pipeline — avoids 10-minute failures
3. **Merge Integrator into Summary** — saves a call, simplifies MVP
4. **Fix actions as executable patches** (same action vocabulary) — makes the fix-loop actually work
5. **Don't store `context` in state.json** — keeps state small, makes reconstruction the only path

These five changes together would cut a typical run from ~15–20 model calls to ~8–12, which on a 27B Apple Silicon backend is the difference between a 6-minute and a 10-minute run. And they all fit within the existing architecture without changing the blackboard pattern or the protocol.
