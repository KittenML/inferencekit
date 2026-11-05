# GSM8K parity (`inferencekit quality gsm8k`)

GSM8K (grade-school maths word problems with numeric answers) is a cheap end-to-end task check: a model with a broken kernel usually loses many points, because multi-step reasoning amplifies per-token errors. It complements trajectory scoring; it does not replace it.

## Setup

- Same problems for every engine (all 1,319 test items by default), same zero- or few-shot format, same answer extraction, greedy, same max tokens, same chat template, through each engine's serving endpoint.
- `inferencekit quality gsm8k --base-url <engine endpoint> --model <served-name> --run-dir runs/<id> --label <engine>` (MCP `quality_gsm8k`) writes `runs/<id>/quality/gsm8k/<label>.jsonl` with per-problem outcomes and a `kind=quality_task` record (accuracy with a Wilson CI). Defaults: all 1,319 test items, zero-shot chain of thought, temperature 0, 1,024 max tokens; answer = `#### n`, else the last `\boxed{}`, else the last number. For the paired test, run the baseline first, then each candidate with `--paired-with runs/<id>/quality/gsm8k/<baseline label>.jsonl` (or the baseline's evidence id). Check flags with `--help`.
- Data: the test split is not shipped. The first run downloads it once from the Hugging Face dataset `openai/gsm8k` (main, test) at a pinned revision, checks its sha256 and caches it in `~/.cache/inferencekit/gsm8k/`. With `HF_HUB_OFFLINE=1` or no network it stops and prints the `curl` command that fetches the file by hand; `--data` takes that Parquet file or any GSM8K-format JSONL.
- Keep per-problem outputs, not only the accuracy: the paired test needs them.
- **Concurrency:** `--concurrency 0` sizes it from the server and records the value and its source. A single-stream (queue-only) engine at a fixed high concurrency can time out before 1,319 items finish.
- **Size:** a `--limit` subset screens (only large regressions); light milestone checks use the first 100 (`--limit 100`); every claim, the final candidate and a scored commit use all 1,319 (start it while the report is drafted). `--time-budget-s S` stops sending problems projected to end past S seconds; the record then says `subset: first n of 1319`, not a full-set result.
- Early development runs used another prompt format, 512 max tokens and only the first 100 items (every same-precision engine within a point or two, underpowered): its numbers are not comparable.

## Statistics

- **One engine's accuracy CI (derived):** standard error `sqrt(p (1 - p) / n)`. At `n = 100` and `p = 0.90`: about +/-6 points. At `n = 1,319` and `p = 0.85`: `sqrt(0.85 x 0.15 / 1319) = 0.0098`, about +/-1.9 points.
- **Paired comparison:** both engines answer the same problems. Count `b` = problems only the baseline got right and `c` = problems only the candidate got right. McNemar's test uses only these discordant pairs: exact binomial on `b` out of `b + c` with p = 0.5 for small counts, `chi2 = (b - c)^2 / (b + c)` for large. Pairing gives far more power than comparing two accuracies; with all 1,319 items, effects of about 1-2 points become detectable, against about 10 points with 100 items.
- **What it can show:** "no large regression". It cannot show equivalence within a point; say so.

## Verdicts

| Outcome | Verdict |
|---|---|
| Discordant pairs balanced, difference inside the pre-registered margin | PASS (task smoke test, not proof of equivalence) |
| Candidate significantly worse (McNemar p < 0.05, b > c) or beyond the margin | FAIL |
| Too few problems for the pre-registered effect, or extraction differs between the engines | UNPROVEN |
| Not run | NOT RUN (say whether trajectory scoring covers the gap) |
