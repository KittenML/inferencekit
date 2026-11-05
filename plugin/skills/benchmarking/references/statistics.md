# Statistics for speed comparisons

## Definitions

- **Round:** one complete measurement of every engine. Each engine runs as a fresh process inside the round: start, ready check, warmup, measured requests, stop. `inferencekit bench rounds` does exactly this for every engine in the run manifest (one `bench serve --rounds 1` call per cell).
- **Per-request decode tok/s:** `(G - k_first) / (t_last_event - t_first_event)` for G generated tokens, where `k_first` is the number of token IDs in the first token-bearing event (1 when IDs are not streamed). The first event is excluded because it carries prefill (TTFT).
- **Per-token latency (TPOT):** the per-request mean `(t_last_event - t_first_event) / (G - k_first)`; report p50 and p95 (p99 under load) across requests. Inter-token latency separately records gaps between token-bearing events.
- **Ratio of medians:** `median(candidate) / median(baseline)` for a higher-is-better metric.
- **Bootstrap CI95 over rounds:** resample whole rounds with replacement many times, recompute the statistic each time, take the 2.5th and 97.5th percentiles. For a ratio, pair engine A's and engine B's results from the same round (they ran next to each other in time). `bench rounds` does the paired version for served engines (`<metric>_ratio_to_best`, ratio to the best engine); `inferencekit compare` does it for `bench decode` records.

## Why rounds, not requests, are the unit

Requests inside one round share the process, allocator state, compiled kernels, GPU clocks and temperature, so a CI over requests is far too narrow. A fresh process per round re-draws all of that: the round-to-round spread is the noise a reader sees on a re-run. Such a CI shows how stable that session was, not where another day would land.

## Fresh process per engine per round

Start each engine fresh in every round. Why: a long-running engine carries warm compile caches, a grown KV pool or fragmented allocator state from earlier requests into later ones, and if one engine stays up while another restarts, the comparison measures that difference.

## Balanced (Williams) order

A fixed or cyclic order puts each engine after the same neighbour every round, so a neighbour that leaves the GPU hot or memory dirty lands on one engine only. A Williams design runs every engine in every position, and right after every other engine, equally often: one block of k rounds for an even k, 2k for an odd k; for two engines, A B B A. Hold the GPU lock for the whole session.

## Interleaving

GPU performance drifts: boost clocks fall as the card heats, other tenants come and go, daemons wake up. If all A rounds run first and all B rounds second, drift lands entirely on one engine and looks like an effect. Interleaving spreads drift evenly; pairing then cancels it in the ratio. `bench rounds` interleaves by construction (every engine once per round). `inferencekit compare` checks interleaving from round timestamps for `bench decode` records, where `--extend <that engine's latest evidence id>` appends rounds to an engine's record, so alternating two `--extend` chains interleaves them. Back-to-back blocks (all A, then all B) are not evidence for a speedup: report the claim UNPROVEN.

## Warmup

Round 1 of a fresh engine typically includes some of: JIT compilation (torch.compile, Triton autotuning), CUDA graph capture, library heuristics, page faults on first touch, and cold L2/TLB state. Run at least 2 warmup requests of every bucket's shape per engine and exclude them. Check: if the first measured request is still slower than the rest by more than the round-to-round spread, warmup was too short. An engine that compiles per prompt length (the early engine captured a prefill graph the second time it saw a length) needs its TTFT reported both ways: first-seen length and repeated length.

## How many rounds

Let `c` be the standard deviation of the per-round log ratio (log(B_i / A_i)) and `h` the CI half-width you need (for example 0.015 for +/-1.5%). With a median-based estimator, roughly:

```
n ~= (2.5 * c / h)^2        # 2.5 ~= 1.96 x 1.25 (medians are ~25% less efficient than means)
```

Worked example (derived): `c = 0.02`, `h = 0.015` gives `n ~= (2.5 * 0.02 / 0.015)^2 ~= 11` rounds per engine. Measure `c` from a short pilot rather than guessing.

**The round count rule (the same in every skill):** R is the pre-registered count (intake `workload.rounds`, default 5, never below 3), sized with the formula above, rounded up to a multiple of the Williams period, and passed explicitly as `--rounds R` (`bench rounds` defaults to the smallest whole Williams-period multiple at least 3: 4 rounds for 2 engines, 6 for 3, 4 for 4). **Exception:** kernel-loop milestones time fixed rounds against the baseline, 2 for a light check (not a claim) and 3 for the full proof (its pre-registered count; [optimizing-kernels](../../optimizing-kernels/SKILL.md), "Checks"); with 2 engines the order is unbalanced and the record says so.

**Adaptive rounds (when `c` is unknown):** pre-register `inferencekit bench rounds --ci-target <h> --min-rounds <n> --rounds <R_max>` in the claim instead of R. It stops at a whole Williams period once the CI half-width is at most `h`, never before `--min-rounds` (at least one period and 3) or after `R_max`. A run that hit `R_max` first is reported as is, never extended by another call.

To detect an effect of size `e`, you need `h` comfortably below `e` (e.g. `h <= e / 2`). If a 2% effect needs 40 rounds at your noise, say so; never report a 4-round result. One round taken from a longer interleaved A/B once overstated the effect several-fold (the baseline ran slow in it). Repetitions inside one process are not independent rounds.

## Reading the result

The rule is `stats.decide`, with the ratio oriented so above 1 is better and `e` the pre-registered effect (`compare --threshold`, default 0.02). The first row that applies wins:

| Verdict (`compare`, or `bench rounds`' ratio CI read the same way) | CI95 of the oriented ratio | Meaning |
|---|---|---|
| `regressed` | entirely below 1, and the ratio <= 1 - e | slower by more than the effect |
| `improved` | entirely above 1, and the ratio >= 1 + e | faster by more than the effect; the CI lower bound is the conservative size |
| `within_noise` | inside [1 - e, 1 + e], and half-width <= 0.75 e | no effect as large as e |
| `inconclusive` | anything else (CI too wide, or it crosses 1 ± e), or the rounds were not interleaved | not enough evidence either way |

Quote the conservative number: "at least 1.06x (CI [1.06, 1.11])", not the point estimate alone. With `bench rounds`, `<metric>_ratio_to_best` is median(engine) / median(best): for tok/s the best engine's advantage over engine X is 1 / X's ratio (conservative: 1 / its CI upper bound); for latencies it is X's ratio (conservative: its CI lower bound).

## Percentiles under load

p95 and p99 need many requests per level: with n requests, the p99 rests on about n/100 of them. Run each load level long enough that its p95 moves less than the SLO margin between two halves of the level, and report n per level. A p99 from 200 requests rests on 2 requests; label it low-confidence.

## Outliers

Do not drop outlier rounds or requests. Medians already resist them, and the outliers are often the real user experience (a hiccup every N requests). Report p95/p99 alongside the median, and investigate any round more than 3x the median spread before publishing.

## Clocks

Leave clocks at their default (boost) policy unless every engine compared, including production, runs locked. Record the policy (`identity.gpu.clocks`). Why: locked clocks give stable numbers for a configuration nobody runs.
