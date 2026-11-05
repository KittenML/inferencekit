# Stop point (stage 6)

## Is a target met?

A target is met only if the **conservative end of its 95% CI** meets it, **in every bucket** listed in `goals.targets.buckets`:

| Target | Better | Met when |
|---|---|---|
| `decode_tok_s` | higher | CI lower bound ≥ target in each bucket |
| `ttft_ms` | lower | CI upper bound ≤ target in each bucket |
| `decode_ceiling_fraction` | higher | (CI lower bound of decode tok/s) ÷ (that bucket's ceiling tok/s) ≥ target |
| `n_star` | higher | N* ≥ target (N* already requires the SLO at that level and every lower one) |
| `usd_per_1m` | lower | cost computed from the CI's conservative speed bound ≤ target |
| `ttfa_ms`, `rtf` | lower | as `ttft_ms`, from a modality client. Without the harness the result is UNPROVEN. |

**Preconditions:** every gate passed, quality inside the noise floor, an allowed precision; a row missing one cannot meet a target, however fast. A screened result that meets a target is a milestone: checked after the light check, counted only after the full proof ([state.md](state.md)).

**Verdicts per target:**

| Verdict | When |
|---|---|
| **PASS** | Met as above. |
| **FAIL** | Measured, and the point estimate misses it. |
| **UNPROVEN** | The point estimate meets it but the CI bound does not, or a precondition is missing (e.g. no ceiling for a fraction target). |
| **NOT RUN** | Never measured (no supported engine, the time box expired, or the target does not apply to the modality). |

## Ranked table (show it at every stop point)

**Lead `single_user`:**

| Rank | Engine / label | Precision | GPUs | Bucket | Decode tok/s [95% CI] | TTFT ms | % of ceiling | Quality vs floor | Gates | $/1M | N* (load) |
|---|---|---|---|---|---|---|---|---|---|---|---|

**Lead `capacity`:** the table leads with N\*, goodput and $/1M tokens at N\*, then the single-user speed ([capacity.md](capacity.md), section 6). Order by N\*, then goodput.

- **Order:** the lead goal's metric, in the target bucket (or the mean rank across buckets).
- **Separate tracks:** equal-precision rows first (plus allowed numerics changes, labelled: [fidelity.md](fidelity.md)), then each other precision track in its own block. Rows on a different GPU count also get their own block ([parallelism.md](parallelism.md)).
- **Absent engines:** NOT SUPPORTED engines are listed under the table, with their errors.

## Decision

1. **Targets set, all PASS:** with `after_target: stop`, go to stage 8. With `keep_climbing` (the default for unattended, benchmark or comparison and fixed-budget runs), continue as in 2 until the report reserve; before each iteration, log in `decisions` the next lever, its headroom and why. New bests go into the best-speed log, screened or verified ([state.md](state.md)).
2. **A target FAIL or UNPROVEN:** compute the headroom and the cost of continuing.
   - **Headroom** (below): the hard bound decides reachability; the achievable product only orders the work.
   - **Resolution:** the smallest effect the benchmark resolves, from the CI widths, typically 1–3%. A gap larger than the hard bound is unreachable on this hardware within the allowed fidelity: **FAIL**, reason "above the hard bound (efficiency × bytes × speculation)". Allowed but unmeasured speculation is unknown, so it never makes a target unreachable: measure it first. The verdict needs measured `ceiling bw` and `roofline` records, never an estimated bandwidth, and it never stops a run with a stated deadline or `keep_climbing`.
   - **Budget:** `inferencekit intake budget runs/<run-id>`. A stage-7 iteration is a change, its fast check and a screen; keep room for the next milestone's checks and the final full proof (`bench rounds`, gates, quality).
     - Budget for at least one iteration, and the target within the hard bound (`keep_climbing`: or any lever's headroom above the resolution): continue to stage 7. This is automatic, and logged in `decisions`.
     - Otherwise, with a stated deadline: keep climbing (next lever, byte cuts, a re-measured ceiling) until the deadline minus the reserve. Without one, or with only the reserve left: stage 8, saying why.
3. **No target, interactive:** show the table, the headroom and the remaining budget. Ask "stop here, or continue to custom engine work for up to N hours?" Record the answer word for word in `decisions`.
4. **No target, non_interactive:** continue while budget remains, keeping the report reserve. Log the choice in `QUESTIONS.md`.

Stage 7 follows the route (below). Re-enter stage 6 after each kept change: a screen that meets a target makes a milestone.

## Headroom: three estimates (stage 2, recomputed after every byte cut)

The ceiling moves: exact byte cuts lower bytes per token; speculation spreads a weight pass over several tokens. All derived:

| # | Estimate | Formula | Counts when |
|---|---|---|---|
| 1 | Efficiency | ceiling tok/s ÷ tuned engine tok/s (the stated baseline's until stage 4; batch 1; capacity: [capacity.md](capacity.md), section 4) | always |
| 2 | Bytes | stored bytes per token ÷ `floor_bytes_per_token` ([ceilings.md](ceilings.md), "Exact byte floor") | an exact cut applies |
| 3 | Speculation | tokens per verify step ÷ verify cost in plain steps ([speculative.md](speculative.md)) | allowed, low concurrency |

- **Hard bound** = the product of those that count, efficiency taken to 1.0 of the moved ceiling, speculation at its measured estimate (unknown until measured). It decides reachability and whether stage 7 has headroom.
- **Achievable** = the same with efficiency capped at about 0.85–0.9 of the ceiling (inferred). **Working target** = tuned speed × the largest achievable product. It orders the work; it never replaces the intake's targets.
- **Byte cuts run alongside kernel work from the start** when estimate 2 exceeds the resolution. After each kept cut, recompute bytes per token, the ceiling and all three: above 100% of the old ceiling is then expected, above the new one is a bug.

## Route: build or patch (stage 0; caps recorded at stage 7 entry)

- **`single_user`, and a single-purpose engine is acceptable** (this model, this GPU, the [custom-engine contract](candidates.md#custom-engine-contract)): **build** a new engine from the [engine skeleton](../../optimizing-kernels/skeleton/README.md). A general server pays per-step costs (scheduler, batching, paging) a batch-1 engine does not need (inferred). The build route enters stage 7 right after stage 5 (when the task names the baseline, right after timing it: no engine survey or tuning); the tuned incumbent (or the task's baseline) stays the baseline (its speculative row: [speculative.md](speculative.md)), and the new engine pulls rungs 1, 3 and 5 itself.
- **`capacity`, or a general-purpose server required:** **patch** the best engine; the scheduler and batching are the work.
- `route: build|patch` in `constraints` overrides. Record the route and both caps either way (patch ≈ 1 ÷ step-time share the incumbent keeps after the planned patches, from nsys; build = achievable combined headroom); switch only when measured caps clearly favour the other, and log why.

## The lever ladder (before stage 7)

Levers in order of expected value for **this** workload; on the patch route custom kernels come last (the order is our inference from gpt-fast, the vLLM optimisation docs, Anyscale on continuous batching and Manus on KV-cache hit rate). Walk it top to bottom whenever stage 6 continues, from the first rung not done (stage 4 covers rungs 1 and 2). Each rung: one `state.decisions` line (applies? the workload reason, derived headroom, result or why skipped), its own track (a quality gate when lossy), one row in REPORT.md's "Lever ladder" table (rung, applies?, headroom, result, verdict).

| # | Lever | Applies when | Headroom (derived) and how to run it |
|---|---|---|---|
| 1 | **Overhead:** CUDA graphs covering every batch size reached, compile, async scheduling | always | nsys: GPU idle gaps between kernels in a decode step. Headroom ≈ step time ÷ summed kernel time on the critical path. |
| 2 | **Batching and scheduler:** batch cap, batched tokens, chunked prefill, KV memory | lead `capacity` | The batch-cap check in [capacity.md](capacity.md), section 3: cap-bound or KV-bound means headroom. |
| 3 | **Precision and KV dtype** (exact byte cuts run from the start, not here) | only for formats `fidelity` and `precision_allowed` / `kv_cache_dtype_allowed` allow | Bytes ratio: `inferencekit roofline --method config --dtype <fmt>` for a dense model (the default reads the real dtypes from the headers and ignores `--dtype`), or `scripts/bytes_per_token.py` on a checkpoint in that format. Separate track; quality gate. Never outside what the intake allows ([non-interactive.md](non-interactive.md)). |
| 4 | **Prefix reuse** | prefix share high (chat with a long system prompt, agents, RAG): ahead of rung 5 | **NOT RUN for now**: there is no shared-prefix prompt set. Record the inferred benefit (prefill skipped ≈ prefix share × ISL tokens per request) as an opportunity. **Never use `--no-cache-salt` in a timed run**: it replays identical prompts and measures exact-repeat cache hits. |
| 5 | **Speculative decoding / MTP** | low concurrency: single-user lead, or N\* well below B_crit ([capacity.md](capacity.md), section 4); unless forbidden (`speculation: off`) | Estimate 3 ([speculative.md](speculative.md)): kept only if faster, and exact. |
| 6 | **Parallelism** | `hardware.count` > 1 | [parallelism.md](parallelism.md). With count = 1: NOT RUN. |
| 7 | **Custom kernels, engine patches or a new engine** (stage 7) | patch route: headroom left after rungs 1–6; build route: right after stage 5 or a task-named baseline | The stage-7 gate below. |

- **Custom engine work (stage 7) needs headroom** above the resolution, by the hard bound. Its efficiency part, 1 / (best fraction of ceiling), is a **batch-1** number. With a capacity lead it says nothing about N\*: use the per-level MBU in [capacity.md](capacity.md), section 4. MBU near 1 at N\* means kernels cannot help much; low MBU with idle gaps points at overhead (rung 1), not kernels.

## Capacity goal
A missed N* target walks the ladder above, then stage 7 with capacity work: scheduler and batching patches, KV memory, kernels at the batch sizes load mode reaches (not the single-user decode kernels). Every capacity commit carries `Perf-Capacity-N`, with its SLO, and `Perf-Goodput`.

## Summary (top of REPORT.md, before every table)

Use the shared [report format and final reply](../../reporting-perf-claims/references/wording-guide.md). Its plain-English summary and detailed tables must support the same claims and verdicts:
- **What you got:** kept engine/configuration and outcome in Result.
- **How much faster vs the baseline:** named baseline, before/after values, ratio/CI/conservative bound, missed targets and other retained options ([tuning.md](tuning.md), step 4).
- **Still correct?:** passed, failed or inconclusive checks and accepted trade-offs, visible in Result.
- **How to run it:** working directory, install/launch/example request commands; links to real outputs.
- **What it cost:** in Details, agent time (from `--run-dir` to report), tokens from the client's usage readout (else NOT RUN, never estimated), cost at API list price (input/output/cache separately; model and price date), and GPU time (GPU-hours and $ from intake budget).

## The verdict table (REPORT.md)

In Details, use one row per claim. The targets come from the intake. Use descriptive evidence links; keep raw ids and numeric classifications here, not in the opening prose.

| Claim | Verdict | Value (class) | Evidence | Note |
|---|---|---|---|---|
| Target: decode ≥ 0.70 of ceiling (short, medium) | PASS / FAIL / UNPROVEN / NOT RUN | 0.73 [0.72, 0.74] (derived) | ev_…, ev_… | ceiling = measured bw / bytes per token |
| Engine X supported at nvfp4 | NOT SUPPORTED | — | experiments.jsonl line 7 | error: … |
| Best config passes gates | PASS | — | ev_… | caching, length, exactness |
| Quality inside noise floor | PASS | agreement 99.1% vs floor 98.9% (observed) | ev_… | |
| A re-run of the manifest on a clean host reproduces | NOT RUN | — | — | done by whoever checks the run, not by this agent |
| N* at the SLO is the true capacity (grid extended until the SLO failed, ≥100 requests per level) | PASS / UNPROVEN | N* = 24 (observed) | ev_… | first failing level 32; batch cap 256 vs 26 in flight |
| Warm-prefix (shared-prefix) track | NOT RUN | — | — | no shared-prefix prompt set yet; prefix share 60% (workload) |
| Multi-GPU (TP/PP/DP/EP) | NOT RUN | — | — | hardware.count = 1 |
| Best speed over time | — | one row per new best, screened or verified ([state.md](state.md)) | ev_… | |
| 3 replicas serve the 4 rps peak at the SLO | UNPROVEN / NOT RUN | 3 = ceil(4 rps ÷ 1.6 rps goodput per replica) (derived) | ev_… | UNPROVEN: replicas behind a router were not measured (not supported yet); NOT RUN without `goals.traffic.arrival_rate_rps` |
