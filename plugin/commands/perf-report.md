---
description: Turn inferencekit evidence into a performance claim with verdicts (PASS / FAIL / UNPROVEN / NOT RUN) and a review page, or audit an existing claim. Needs no GPU.
argument-hint: "[run directory | evidence ids... | path or text of a claim to audit]"
---

Write or audit a performance claim using: $ARGUMENTS

Follow the reporting-perf-claims skill. Read `${CLAUDE_PLUGIN_ROOT}/skills/reporting-perf-claims/SKILL.md` (in a checkout of this repo: `plugin/skills/reporting-perf-claims/SKILL.md`) and its references.

If a run directory or evidence ids were given (or none: use `inferencekit evidence list` and ask which records belong to the claim):
1. Read each record with `inferencekit evidence show <id>`: identity, coverage and limitations. Find its pre-registered claim; a post-hoc claim is at best UNPROVEN.
2. Apply the skill's comparability, gate, quality and verdict rules, including its evaluating-quantization exception. A speedup needs an `improved` comparison from an interleaved A/B with CI95 excluding 1. Missing data is UNPROVEN or NOT RUN, never zero or PASS.
3. Generate `inferencekit report <ids...> --format md` and `inferencekit review build <run-dir>`. Add quality/gates verdicts from their own records; the export has no such columns.
4. Use `references/wording-guide.md` in reporting-perf-claims for `REPORT.md` and the final reply. Start with the plain-English summary specified in optimizing-inference's `references/stop-point.md`, "Summary". Keep mechanisms, losses, refuted ideas, ledger, NOT RUN items and costs in Details. Put ids and observed / derived / inferred labels there or in linked evidence; label `bench decode` values "engine-timed" wherever quoted.
5. Run the skill's claims lint; fix or downgrade every failing line.

If a claim text or file was given to audit: run the claims lint line by line, quote each failing sentence, and state the verdict the evidence actually supports.
