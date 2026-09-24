# agedIn — Improvement Pass (adoption of recoimprov.md)

Source: `recoimprov.md` (external design review). This file records what was
adopted, what was deferred, and measured impact. Applied: 2026-09-19.

## Adopted (implemented + validated)

| # | Change | Impact measured / expected |
|---|---|---|
| 1 | **Batched Builder actions** — model emits all of a step's actions in one reply; single-action form still accepted | hello.html run: **11 → 6 model calls (−45%)**; batching value grows with file count |
| 2 | **Capability probe** (connection): tiny strict-JSON call calibrates tokenizer ratio, measures baseline TTFT/TGS, warns early if the model can't emit JSON | prevents late-discovered protocol failures; costs ~2 s |
| 3 | **Merged Integrator into Summary** — one call produces gaps + README + report (was 2 calls / 2 stages) | −1 call per run, −1 stage |
| 4 | **Executable validator fixes** — fix_actions with full corrected file content are applied directly, then re-checked; a fix round with **zero extra model calls**; self-heal recorded in the verdict | kills the most expensive churn path |
| 5 | **Quality-regression guard** — artifacts snapshotted before fixes; if QA drops after a fix, the fix is reverted | no more "fix made it worse" outcomes |
| 6 | **Step-scoped loop hashing** — writing the same file in different steps no longer collides | correctness fix |
| 7 | **Action schema validation** — typed arg checks with targeted repair hints (`args.path must be a non-empty string`) | catches `path: 123`-class failures |
| 8 | **Write sanity + double-write nudge** — rejects empty/description-style content; warns on same-file rewrites within a step | catches "wrote a description instead of the file" |
| 9 | **`read_file` + `list_files` actions; big-file head/tail supply (>200 lines)** — models stop wasting shell calls and never blow context on large files | context efficiency |
| 10 | **Plan self-check + template fallback** — outputs present, no duplicate outputs, detail capped at 500 chars; degraded planning now yields a task-type template, not one giant step | recovery quality |
| 11 | **Tokenizer-ratio calibration** — per-run EMA from real `prompt_tokens` feedback drives the 60% compaction trigger (was fixed chars/3.6) | accurate compaction timing |
| 12 | **`--effort` now controls per-call tokens** (low 4k / medium 8k / high 16k) — effective on backends that ignore reasoning-effort hints | makes the flag real |
| 13 | **`--verbose`** (wire I/O → stderr), **`--dry-run`** (plan only), **`--max-fix-rounds N`**, **`--seed`** | operability + reproducibility |
| 14 | **Parallel deterministic checks** (ThreadPool over per-file QA/py_compile) | ~2–4 s on multi-file projects |
| 15 | **Structured event types** (`type` field: repair / loop_nudge / self_heal / regression / verdict…), **free step-boundary rollups** from action notes, **rc in shell observations**, **`AGEDIN_RESULT` now includes `workdir`** | observability |

## Deferred (with rationale)

- **Persistent keep-alive connections** — localhost handshake cost is noise
  vs 10–60 s generations; connection lifecycle would complicate the abort path.
  Revisit if remote backends matter.
- **Confidence self-assessment field** — models are poorly calibrated on their
  own outputs; the QA gate already measures actual quality.
- **Headless-browser smoke test** — heavy dependency against the zero-dep
  goal; the deterministic gate + model review cover the common failures.
- **Trivial-task planning skip** — planning is 1 call (~5 s) and batching
  already cut the rest; the heuristic risks misrouting real tasks.
- **`request_info`** — never existed in the implementation; confirms the
  review's "remove entirely" recommendation.

## Validation after the improvement pass

hello.html task on all three frameworks (see console transcripts from
2026-09-19 20:51–20:56): **done / QA 100 / usable on OMLX, MTPLX, MLX-LM.**
Call count on OMLX hello: 11 → **6** (connection+probe, planning, 2×builder,
validating, summary). Wall time on this trivial task is comparable (~45–78 s
across frameworks); the call-count reduction is the durable win — fewer
protocol failure points, less queue overhead, and the gap widens on
multi-file tasks where batching eliminates per-file calls.
