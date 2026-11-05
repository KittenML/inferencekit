---
name: reporting-perf-claims
description: Writes and audits inference performance claims so they survive review - PASS / FAIL / UNPROVEN / NOT RUN verdicts, ratio CIs, a claims lint (every number has an evidence id), published losses, the review page and the commit ledger. Use when summarising results, writing a PR, README, blog or final report, or reviewing someone else's (or an agent's) numbers. Needs no GPU.
license: Apache-2.0
compatibility: Uses the inferencekit CLI (Python 3.10+, uv or uvx) or its MCP server; needs no GPU. Written for Claude Code, Codex and OpenCode.
metadata:
  version: "0.1.0"
---

# Reporting performance claims

A performance claim is a statement someone will act on: it says what was measured, under which conditions, with what uncertainty, and what was not shown. This skill turns evidence records into claims that survive review, and audits claims others wrote; it is the last stage of [optimizing-inference](../optimizing-inference/SKILL.md).

**Use** to write up any benchmark, capacity, quality, profile or optimisation result, to review a PR description, README table, blog post or agent summary with performance numbers, or to decide what verdict evidence supports. **Not** while you still need to measure ([benchmarking](../benchmarking/SKILL.md), [profiling](../profiling/SKILL.md), [proving-correctness](../proving-correctness/SKILL.md)).

## Preflight checklist

- [ ] The claim was written before the measurement (statement, metric, workload, bucket, decision rule). Why: claims chosen after the data are forking-paths results.
- [ ] Every evidence id the report cites exists (`inferencekit evidence show`). Why: dangling citations are fabricated evidence.
- [ ] `coverage` and `limitations` of every record have been read. Why: a valid record can still not cover the claim.
- [ ] Applicable gates passed: `gates run` for LLMs; structural and semantic checks for media. A gate failure invalidates a win.
- [ ] Losses, refuted ideas (`experiments.jsonl`) and NOT RUN items are listed. Why: a report of only wins misleads about the base rate.

## Hard failure gates

- A number without an evidence id or a formula.
- A speedup without an interleaved A/B whose ratio CI95 excludes 1.
- A speed comparison across precisions (weights, activations, KV cache, lm_head), GPUs or workloads presented as an engine result.
- An engine-timed (`bench decode`) number in a published table, or next to client-timed (`bench serve`) numbers without its label.
- A profile (nsys, ncu) cited as proof of a speedup.
- Missing data shown as 0, blank or PASS.
- A speed claim without a correctness/quality verdict.

## Workflow (copyable)

```
Report:
- [ ] 1. Copy the pre-registered claim(s) verbatim, with their timestamp
- [ ] 2. inferencekit evidence list; select the records for each claim
- [ ] 3. For each claim, evaluate the precedence FAIL > UNDERPOWERED > INCOMPLETE > NO-DATA > PASS
- [ ] 4. Map to PASS / FAIL / UNPROVEN / NOT RUN
- [ ] 5. Label every number observed / derived / inferred
- [ ] 6. inferencekit report <ids...> --format md
- [ ] 7. inferencekit review build runs/<run-id>   (outputs side by side, per bucket and engine)
- [ ] 8. Write REPORT.md using references/wording-guide.md; retain mechanism, losses, NOT RUN items and ledger in Details
- [ ] 9. Claims lint over the final text (below); fix or downgrade every failing line
```

## Exact commands

```bash
inferencekit evidence list
inferencekit evidence show <id>
inferencekit compare <ev_baseline> <ev_candidate> --metric <metric>  # same-kind round records; not load mode
inferencekit report <ids...> --format md                   # fixed layout; --format json for CI
inferencekit review build runs/<run-id>                    # -> runs/<run-id>/review/index.html
inferencekit history                                       # kept commits with their Perf-* trailers
```

MCP tools: `list_evidence`, `inspect_evidence`, `compare`, `report`, `build_review`, `history`. The CLI/MCP report is a fixed technical export; link it or include its tables in Details of the user-facing report. Add quality/gates verdicts from their records; the export has no such columns. Served ratios come from `bench rounds`' `<metric>_ratio_to_best`. Follow [references/wording-guide.md](references/wording-guide.md) for `REPORT.md` and the final reply.

## Verdict precedence

Evaluate in this order; the first that applies is the verdict.

| Step | Condition | Reported as |
|---|---|---|
| FAIL | a required check ran and contradicts the claim: `regressed`, CI95 entirely below the claimed ratio, a quality FAIL, a gate FAIL, or a hard gate in the candidate (except the reasons [hardware.md](../inference-triage/references/hardware.md), note 3, lists as expected on that platform, reported as "not applicable on <platform>") | **FAIL** |
| UNDERPOWERED | checks ran but the CI includes 1 (or the claimed ratio and 1 both), or fewer rounds/items than pre-registered (a kernel-loop full proof pre-registers its fixed 3 rounds) | **UNPROVEN** (underpowered) |
| INCOMPLETE | `coverage.complete` is false, an engine/bucket/level/metric in the claim is missing, or rounds were not interleaved | **UNPROVEN** (incomplete) |
| NO-DATA | a required check produced no data | **NOT RUN** if never attempted; **UNPROVEN** if attempted |
| PASS | none of the above, and the decision rule is met | **PASS** |

Why this order: one contradicting check must never be outvoted by passing ones, and "not enough data" must never look like "fine". Worked examples: [references/verdict-rules.md](references/verdict-rules.md).

## Claims lint (run over every final text)

Result links descriptively to Details or its evidence table for ids, numeric classifications and formulas; every claim stays traceable.

- [ ] Every number has an evidence id, or a formula whose inputs have ids.
- [ ] Every number is labelled observed / derived / inferred.
- [ ] Claim verdicts use only PASS / FAIL / UNPROVEN / NOT RUN; comparison verdicts only `improved | regressed | within_noise | inconclusive`, from `inferencekit compare` or a `bench rounds` paired ratio CI.
- [ ] No speed claim crosses precisions; cross-precision rows are labelled "cross-precision: not an engine comparison", with no ratio verdict. Exception: a format trade-off ([evaluating-quantization](../evaluating-quantization/SKILL.md)) may show its speed ratio with CI95 only next to its quality verdict against the full-precision reference, labelled "cross-precision: format trade-off, not an engine comparison", with no `improved` / `regressed`.
- [ ] Published timing is client-measured: served LLMs or complete media requests. Diagnostic engine timings are labelled.
- [ ] Every speedup has a ratio with its CI95 and the conservative bound quoted; "up to", "slightly faster", "near optimal" are absent.
- [ ] Every engine row states its version, resolved key settings and whether it was tuned for this goal.
- [ ] Capacity claims state the SLO, `N*`, open-loop arrivals, and the price behind any cost.
- [ ] Missing buckets, levels, GPUs and checks appear as NOT RUN, never as blanks or zeros.
- [ ] Losses and refuted ideas are listed with the same columns as wins.
- [ ] The review page exists for the run and is linked.
Line-by-line examples and fixes: [references/claims-lint.md](references/claims-lint.md).

## Review page (`inferencekit review build <run-dir>`, MCP `build_review`)

One self-contained HTML file at `runs/<run-id>/review/index.html` that works offline: rows are inputs grouped by bucket, columns are engines plus the reference, with the first divergence from the reference marked; a summary table at the top; filters by bucket and engine. Link it from the report. Why: a reader can see what each engine actually produced for the inputs that were timed, which catches broken outputs no aggregate metric shows. The full page stays local; share it only after the user has checked it.

## Fixed result table

Open with the plain-English summary from [stop-point.md](../optimizing-inference/references/stop-point.md), "Summary". Put the technical claim table in Details, one row per pre-registered claim, with these columns:

| Claim | Workload / bucket | Baseline | Candidate | Ratio [CI95] | Quality | Mechanism | Gates | Verdict | Evidence |
|---|---|---|---|---|---|---|---|---|---|

Populate every applicable column from evidence; mark missing checks UNPROVEN or NOT RUN.

## Decision and no-go rules

- Quote the conservative bound: "at least 1.09x (CI [1.09, 1.16])". Why: the point estimate is the luckiest honest number.
- Publish losses next to wins, with the same rigour. Why: readers calibrate trust from them.
- Never upgrade a verdict in prose ("UNPROVEN but clearly faster"). Why: the verdict exists to stop that sentence.
- Never extrapolate across GPUs, precisions, context lengths, buckets or concurrency levels you did not measure. Write NOT RUN.
- Refuse a cross-precision speed claim even when asked; offer the same-precision comparison and the cross-precision row labelled as such. The only exception is a format trade-off from evaluating-quantization: the speed ratio with its CI95 next to the row's quality verdict, labelled "cross-precision: format trade-off, not an engine comparison", with no `improved` / `regressed` verdict. Why: a format buys its speed with quality, so only the pair is a result; speed alone across precisions is still refused.

## Reporting rules

- **observed**: measured by a tool in this environment; cite the id and statistic. **derived**: arithmetic on observed values; show the formula. **inferred**: reasoning; name the measurement that would confirm it.
- Verdicts: **PASS / FAIL / UNPROVEN / NOT RUN** for claims; `improved | regressed | within_noise | inconclusive` for comparisons.
- No environment values, secret tokens or raw argv in a report; inferencekit stores argv only as a digest.
