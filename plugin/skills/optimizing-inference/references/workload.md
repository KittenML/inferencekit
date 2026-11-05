# Workload and SLO (stage 1b) and the deadline

Experienced inference engineers start from the workload, not the engine: prompt and answer lengths, arrivals and burstiness, requests in flight, the shared share of each prompt, and the latency each user must get (the SLO). This page maps the answers to the intake's fields. **The schema is fixed: never invent a field;** the rest goes into one `constraints` entry, which the report quotes.

## When

- **Stage 0, drafting the intake:** reuse [starting answers](starting-questions.md), derive from supplied examples/logs, and ask only missing details that affect the goal; they set `goals.lead`, `goals.slo`, `goals.traffic`, `buckets`, `workload.max_tokens`, `workload.concurrency` and `budget.wall_clock_h`, settled before `--run-dir` (a changed intake is a new run).
- **Stage 1b, after `intake validate --run-dir`:** record the workload, what you derived and every gap in `state.decisions`, and start `REPORT.md`'s gap table (below). Nothing is measured here.
- **An intake you cannot change** (non_interactive, or handed over as-is): derive the rest as below; each missing number goes to `QUESTIONS.md` with the value assumed, and into the gap table.

## Technical mapping (derive first; translate necessary questions into ordinary language)

| # | Ask | Intake field | If unknown |
|---|---|---|---|
| 1 | **SLO:** the p90 or p95 time to first token (TTFT) and time per output token (TPOT) a user must get, and the failures allowed | `goals.slo.ttft_p95_ms`, `tpot_p95_ms`, `max_fail` | Interactive: ask. There is no default SLO. You may offer a starting point for a chat UI, e.g. p95 TTFT ≤ 1,000 ms and TPOT ≤ 50 ms (≥ 20 tok/s, faster than reading speed) (inferred). It is never the person's SLO until they say yes. |
| 2 | **Input length (ISL)** p50 and p90, in tokens, counting the **whole** prompt: system prompt and any shared prefix included | `buckets` (below), plus the constraint | Estimate from the product (a chat turn, a RAG prompt with k chunks, an agent step). Label it "estimate". |
| 3 | **Output length (OSL)** p50 and p90 | `workload.max_tokens` (below), plus the constraint | 256 (the default), labelled "default, not your workload". |
| 4 | **Peak requests per second** | `goals.traffic.arrival_rate_rps` | No replica plan: stage 8 marks it NOT RUN. |
| 5 | **Burstiness and session shape** | `goals.traffic.burstiness` (e.g. "poisson", "3x spikes at 09:00"), `goals.traffic.session_len` (e.g. "1 turn", "8 turns") | "poisson", "1 turn". |
| 6 | **Concurrency:** typical and peak requests in flight | `goals.lead` and `workload.concurrency` (below), plus the constraint | Derive it with Little's law: in flight ≈ RPS × mean end-to-end seconds. |
| 7 | **Shared-prefix share:** how much of a typical prompt is identical across requests (system prompt, tool schemas, few-shot examples), as shared-prefix tokens ÷ ISL p50. The prefix is part of ISL, so it can never exceed it | the constraint | 0, so no prefix reuse is claimed. |
| 8 | **Deadline:** "by 09:00" or "within 6 hours" | `budget.wall_clock_h` (below), plus the clock time in the constraint | The budget's other hour caps only. |
| 9 | **Engine scope:** is a single-purpose engine (this model and GPU, the [server contract](candidates.md#custom-engine-contract)) acceptable? | `route: build\|patch` | From the goal ([stop-point.md](stop-point.md), "Route"). |
| 10 | **Fidelity:** exact outputs, or a quality loss within a budget? | `fidelity`, `quality.budget` | `lossless` ([fidelity.md](fidelity.md)). |
| 11 | **Sampling** users send; speculative decoding acceptable? | `workload:` constraint; `speculation: off\|auto` | `generation_config.json` defaults; `auto`. |

Record the answers as one `constraints` entry that starts with `workload:`, and name the source of the numbers (real traffic logs, a trace, a load-test plan, or an estimate). For example:

```yaml
constraints:
  - "workload: ISL p50 900 / p90 3000 tok, whole prompt; OSL p50 200 / p90 600 tok; peak 4 rps; concurrency typical 12, peak 30; prefix share 60% (a 550-token system prompt inside the 900); SLO as given: p90 TTFT <= 800 ms, p90 TPOT <= 40 ms; deadline 09:00 Europe/London 2026-10-15; source: production logs, last 7 days"
```

## Deriving the intake fields

**Lead goal (`goals.lead`).**
- **One user at a time** (a local assistant, a single-tenant tool at batch 1): `single_user`, ranked by single-user decode tok/s and TTFT.
- **More than one request in flight at peak:** `capacity`, judged by goodput at the SLO: **N\*** (the highest concurrency at which the SLO holds there and at every lower level) and **goodput** (requests per second that met the SLO). Validate derives `modes: [single, load]`, so single-user numbers are still reported next to N\*. It needs `goals.slo.ttft_p95_ms` (refused without); add `tpot_p95_ms`, or N\* holds only the first-token SLO.

**SLO percentiles.** The harness judges p95 only: a p90 SLO goes into the p95 fields (stricter, so N\* is conservative), and the report says "SLO given at p90, judged at p95". Never loosen an SLO to fit the harness.

**Buckets (input lengths).** The prompt sets have four fixed buckets: `very_short` 1–5 words, `short` ~20–50 tokens, `medium` ~300 tokens and `long` 2k–4k tokens.
- ISL is the whole prompt, shared prefix included (a 550-token system prompt inside a 900-token p50 prompt still means 900; add a separately given prefix and rest).
- Take the buckets nearest ISL p50 and ISL p90 (one when they are the same).
- Load mode's `dev-load` set holds `short` and `medium` prompts only. When the lead is capacity, also include whichever of those two is nearest ISL p50.
- ISL p90 above 4k tokens: `long` is the closest bucket, and contexts above 4k are NOT RUN (there is no bucket for them). In long-context and agent workloads (around 100 input tokens per output token), prefill and the chunked-prefill budget dominate. Say so in the gap table.
- ISL p50 above `medium` (~300 tokens) with a capacity lead: the load test (`short` and `medium` only) under-represents prefill and the KV cache per request, so N\*, goodput and the replica plan are optimistic ([capacity.md](capacity.md), section 7). Say so in the gap table.
- `goals.targets.buckets` is the bucket nearest ISL p50, unless the person names others.

**Output length (`workload.max_tokens`).** Every request generates exactly this many tokens, for every engine.
- `single_user` lead: OSL p50, rounded to a round number (128, 256, 512, ...).
- `capacity` lead: OSL p50 as well, unless OSL p90 > 2 × p50. Then use p90: longer outputs keep more KV cache in flight, so N\* comes out conservative. Say which one you used.
- A longer output lengthens every timed step. The run-time estimate ([harness-commands.md](harness-commands.md), section 2) scales with it.

**Load grid (`workload.concurrency`).** Use doubling levels from 1 to at least 2 × the peak concurrency. For example, typical 12 and peak 30 give `[1, 2, 4, 8, 16, 32, 64]`. This grid is a floor: [capacity.md](capacity.md) extends it upward until the SLO fails.

**Traffic (`goals.traffic`).**
- `arrival_rate_rps` is the **peak** rate: stage 8 sizes replicas for it.
- `burstiness` and `session_len` are free text. The load generator sends Poisson arrivals, so a burstier real workload is a gap.

**Prefix share.**
- The primary runs keep prefix caching off and every prompt unique (hard rule 9; the caching gate checks it).
- A high prefix share (chat with a long system prompt, agents, RAG with a shared preamble) makes prefix reuse an early rung of the lever ladder ([stop-point.md](stop-point.md)). It is **NOT RUN** yet, because there is no shared-prefix prompt set yet.
- **Never use `bench serve --no-cache-salt` in a timed run or as a "warm prefix" measurement:** it resends identical prompts, so it measures exact-repeat cache hits (the reward hack the caching gate catches).

**Multi-GPU.** `hardware.count` > 1 adds parallel configurations as candidates ([parallelism.md](parallelism.md)). With `count: 1`, multi-GPU is NOT RUN.

## The deadline (`budget.wall_clock_h`)

- **Given in hours** ("within 6 hours"): `wall_clock_h: 6`.
- **Given as a clock time** ("by 9am", "by 18:00 UTC"):
  1. Pin the time zone (ask, else the person's stated location, else UTC, said so); a time with no date means its next occurrence.
  2. `wall_clock_h` = hours from your `intake validate --run-dir` to the deadline, rounded **down** to 0.25 h, computed when you write the intake (after the go), so the approval wait does not count.
  3. Write the clock time and timezone in `budget.deadline` and the decision record.
  4. The deadline is the commitment, not the hour count: a cut under about 10% caused only by the approval wait needs no new question; a larger one changes what fits (show what moves to NOT RUN and ask again, [plan.md](plan.md) step 4). It never grows.
- **It is one of the hour caps;** when it binds (`binding: wall_clock_h`), every stage's time box is a share of it, and the 10% report reserve means a report exists at the deadline.
- **A short deadline** (below about 3 h; inferred from typical install, compile and rounds times): plan 2 engines and at most one capacity entry (the build route skips the survey and goes straight to stage 7). Say in the plan (or `state.decisions`) what the deadline makes NOT RUN.
- **Every stage:** `inferencekit intake budget runs/<run-id>`, obeyed; on `stage_over`, finish the current measurement, mark the rest NOT RUN, move on.

## The gap table (`REPORT.md`, section "Workload: real vs measured")

One row per dimension. A row where the run did not match the real workload is a limitation, never a hidden assumption.

| Dimension | Real workload (source) | What was measured | Effect on the numbers |
|---|---|---|---|
| Input length | p50 900 / p90 3,000 tok (logs) | single-user: `medium` (~300) and `long` (2k–4k); load: `medium` only | TTFT for a p50 prompt lies between the two buckets; N\*, goodput and the replica count are optimistic for 900-token prompts (inferred) |
| Output length | p50 200 / p90 600 tok | exactly 256 tokens per request | |
| Arrivals | 4 rps peak, bursty | Poisson at N ÷ e2e_ref | bursts queue more than Poisson arrivals (inferred) |
| Prefix share | 60% | 0: cache off, unique prompts | real TTFT is lower; the warm-prefix track is NOT RUN |
| Concurrency | typical 12, peak 30 | grid 1–64, extended to the first SLO failure | |
| Contexts above 4k tokens | 8% of requests | not measured | NOT RUN |
| GPUs | 2 per replica in production | `hardware.count` | multi-GPU NOT RUN when count = 1 |
