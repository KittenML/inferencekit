# Verdict rules with worked examples

## Pre-registration template

Write this before the first candidate run, and keep the timestamp.

```
Claim C1
Statement:      candidate decode tok/s >= 1.10x baseline
Metric:         decode_tok_s, median per engine, ratio candidate / baseline
Workload:       Qwen3-8B-FP8 @ <revision>, FP8 weights / BF16 KV, prompt 512, gen 256, batch 1, greedy
Environment:    1x RTX 5090, driver <x>, default clocks, GPU otherwise idle
Rounds:         2 warmup requests + 10 measured rounds per engine (bench rounds, Williams order)
Decision rule:  PASS if the paired ratio is improved (bench rounds ratio CI; compare for bench decode),
                CI95 lower > 1.00 and point >= 1.10
                FAIL if regressed, or CI95 upper < 1.10, or correctness FAIL
                otherwise UNPROVEN
Correctness:    teacher-forced top-1 agreement within the A/A band; no large-margin divergence
Written:        <timestamp>, before any candidate measurement
```

Non-inferiority claims (refactors, "no regression") use a margin: "candidate >= 0.98x baseline"; PASS if the CI95 lower bound >= 0.98.

## Worked examples (illustrative numbers)

| Evidence | Precedence step hit | Verdict | Why |
|---|---|---|---|
| ratio 1.13 [1.09, 1.16], improved, correctness PASS, coverage complete | PASS | PASS | rule met; quote "at least 1.09x" |
| ratio 1.06 [1.03, 1.08], improved, claim was >= 1.10 | FAIL | FAIL | CI upper 1.08 < 1.10: the claim is ruled out (a smaller real speedup exists: report it as a separate observed finding) |
| ratio 1.09 [0.97, 1.21], inconclusive, 4 rounds | UNDERPOWERED | UNPROVEN (underpowered) | CI includes 1 and 1.10; add rounds |
| ratio 1.12 [1.08, 1.15], but rounds were back-to-back blocks | INCOMPLETE | UNPROVEN (incomplete) | drift could explain it; rerun interleaved |
| speed PASS, correctness never run | NO-DATA | NOT RUN for correctness, claim UNPROVEN | no claim passes without its correctness gate |
| ratio 1.20 [1.15, 1.25], correctness FAIL (large-margin divergences) | FAIL | FAIL | a faster wrong engine is a failure |
| H100 row planned, no H100 run | NO-DATA | NOT RUN | never extrapolate from another GPU |
| ratio 0.91 [0.88, 0.94], regressed | FAIL | FAIL | publish it as a loss |

## Combining several checks

A claim usually depends on several records (each engine's benchmark, comparison, correctness, coverage of each context length). Evaluate each check, then apply the precedence across all of them: the claim's verdict is the highest-precedence result among its checks. A single FAIL makes the claim FAIL; otherwise a single UNDERPOWERED makes it UNPROVEN, and so on.

## Losses and refuted ideas

Report them in a findings subsection with the same table columns. A refuted idea is a result: "split-K with 8 splits: 1.00x [0.99, 1.01], within_noise at a 2% effect; not adopted" saves the next person a day.
