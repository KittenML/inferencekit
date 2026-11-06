# state.json, the run id, the evidence root, resume

## Start and resume

- **Run id:** `run_id` from the intake if it has one; otherwise `derived.run_id` from the last dry check of the intake *file* before `--run-dir`. It contains the time, so a later dry check prints a different id: reuse the one you copied everywhere.
- **Fresh run:** `--run-dir` writes a redacted intake copy, `intake.normalized.json`, `state.json`, `experiments.jsonl` and `QUESTIONS.md`. It only creates them, never overwrites.
- **Evidence root, one per run: `runs/<run-id>/evidence/`,** as `INFERENCEKIT_RUNS_DIR` (absolute) for every inferencekit call, here and on the GPU host ([remote-host.md](remote-host.md)). Shells forget exports: prefix each call or pass `inferencekit --runs-dir <abs> <command>`. Without it, records land in `./runs/ev_*`, outside the sync, and citations (`evidence show`, `--gpu-bw`, `ledger trailers`, `review build`) break. MCP tools see only the MCP server's start environment: without the variable there, use the CLI.
- **Resume:** `intake resume` checks pinned policy and cited evidence. Use `--file <original>` if `state.intake_source` moved; `--now <UTC>` fixes budget time. Set `evidence_root`; integrity is not proof. Never redo a `done` stage whose evidence exists. Redacted/changed intakes are refused: changes need a new run id; never edit the copy, `intake_sha256` or targets ([intake.md](intake.md)).

## state.json

Created by `intake validate --run-dir`; update it after every step (write a temporary file, `mv` it over). A resumed agent continues from it alone: keep `next_action` current.

```json
{"state_version": 1, "run_id": "...", "intake_sha256": "...", "current_stage": "3_candidates",
 "stages": {"3_candidates": {"status": "in_progress", "started": "...Z", "ended": null, "time_box_h": 1.4,
                             "evidence": ["ev_..."], "notes": ["sglang: NOT SUPPORTED (see candidates.sglang)"]}},
 "budget": {"host_hours_cap": 8.0, "usd_cap": 6.0, "price_usd_h": 0.6, "run_started": "...Z",
            "meter": [{"start": "...Z", "end": null, "price_usd_h": 0.6}]},
 "model": {"id": "...", "revision": "...", "revision_resolved": "<40-hex>"},
 "candidates": {"vllm": {"status": "supported", "version": "x.y.z", "label": "vllm-tuned", "error": null,
                         "resolved_config": {}, "evidence": ["ev_..."]}},
 "best": {"single_user": "vllm-tuned", "capacity": null},
 "decisions": [{"at": "...Z", "stage": "6_stop_point", "question": "...", "choice": "...", "why": "...",
                "asked_user": false}],
 "next_action": "stage 4: sweep vllm cudagraph mode on bucket medium"}
```

- **Stage status values:** `pending`, `in_progress`, `done`, `skipped` (reported as NOT RUN, with the reason), `over_box` (the time box expired; the unfinished work is NOT RUN).
- **Entering a stage:**
  - set `current_stage`, `stages.<k>.status = "in_progress"` and `stages.<k>.started = now` (UTC, `...Z`). `intake budget` measures the time box from `started`.
  - When a stage ends early, add its unused hours to the next stage's `time_box_h`; a skipped stage adds its whole box (a task-named baseline: stages 3-5 go to stage 7). Never take time from `8_report`.
- **Meter (the GPU cost clock, running while the host exists, busy or not):** at first contact append a segment (`start` = when it was rented if known, else now); set `end` when your GPU work is over.
- **Candidate status values:** `candidate`, `supported`, `NOT SUPPORTED`, `excluded` (with the reason: constraint, precision, or modality).

## experiments.jsonl (one line per attempt)

```json
{"at": "...Z", "stage": "4_tuning", "engine": "vllm", "label": "vllm-cg-full", "change": {"cudagraph_mode": "FULL"},
 "resolved": {"cudagraph_mode": "FULL", "source": "startup log: 'compilation_config ... cudagraph_mode'"},
 "in_effect": true, "bucket": "medium", "metric": "decode_tok_s", "value": 123.4, "ci95": [122.9, 123.8],
 "vs": "vllm-recipe", "verdict": "improved", "evidence": ["ev_..."], "kept": true, "note": ""}
```

- **Failures are rows too:** `"value": null, "error": "..."`.
- **Proposals from an autotuner** are rows with `"value": null` and `"note": "proposed by <tool>"` ([tuning.md](tuning.md)).
- **Capacity rows** use `"metric": "n_star"` and put the batch-cap check in `note`: the resolved cap, `mean_inflight`, and running/waiting at N\* ([capacity.md](capacity.md), section 3).

## Best so far: screened, checked or verified (one timestamped row per new best)

Each check is one `ikstep job` ([remote-host.md](remote-host.md), "One command per step"): start it, work GPU-free, wait once. A brief that asks for more verification wins.
- **Baseline**, once per run and reused: all gates, greedy `reference` and `reference-r2` trajectories (the A/A floor), all 1,319 of GSM8K, `quality sampling`, and an A/A screen (the screen noise).
- **Screened** = passed its fast correctness check and the screen (step replays, or `ikstep job screen`).
- **Checked** = passed the light check, at the first candidate that screens >= 1.2x and the first >= 1.5x of the baseline (or the brief's target), never otherwise: `bench rounds --rounds 2` with `--limit-per-bucket 8` (32 prompts); `gates run --gates caching --gates token_count --gates rare_tokens`; quality trajectory and score with `--limit 8` (the first 8 prompts, all very short: long-prompt drift shows only in the full proof); GSM8K `--limit 100 --concurrency 0`; `quality sampling` only if sampling code changed.
- **Verified** = passed the full proof, once, for the final candidate and any commit you hand over (with a deadline, started at the report reserve minus its length): `bench rounds --rounds 3` on all 64 dev prompts (a hard cap; with two engines the record notes the unbalanced order), read with the pre-registered decision rule (effect e); all gates; quality trajectory and score on all 64 inside the floor (or the lossy budget); all 1,319 of GSM8K `--paired-with` the baseline's; `quality sampling`.
- **Each new best** gets an `experiments.jsonl` row (`"verified": true` or `false`), a `state.best_history` entry (`{"at", "label", "commit", "metric", "value", "ci95", "ratio_to_baseline", "sampling", "checks", "evidence"}`, `checks` = `screened`, `checked` or `verified`) and a row in REPORT.md's "Best speed over time" table (hours since `--run-dir`, commit, speed [CI95 when verified], ratio, screened, checked or verified). Move a hand-over tag only to a checked or verified build. Screened and checked rows are progress, not claims: no verdict, labelled as such.


Preserve intake’s `state.budget.spend_cap`, `deadline`, `stage_timeboxes` and `report_reserve_h` on resume. Uncapped spending stays null; false timeboxes ignore stale stage boxes; zero reserve disables early freeze. Validate/package continuously until the deadline.

## Decisions and policy changes

Record the opening choices and authorization source/time/scope in `state.decisions`; a new run does not require `PLAN.md`. Honor legacy recorded approvals on resume.

A change to quality permission needs a new pinned intake/run. Link the previous run and candidate commits; retain its failure reasons and evidence. Numerical-only rejects can enter the new queue, with fresh policy-appropriate screens and final proof. Do not rewrite previous grades. Keep current hypothesis, relevant sources, pending patch, device owner and next action compact so resumed workers continue from evidence.
