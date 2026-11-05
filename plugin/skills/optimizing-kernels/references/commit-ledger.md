# Commit ledger: every kept change is a revertible checkpoint with its numbers

## Repository (from the intake's `repo`: `path`, `branch`, `push: ask|never`)

- **Existing repo:** work on a new branch `perf/<model>-<hardware>`.
- **New repo for this use case:** local `git init`; create a remote (for example with `gh`) only after the user says yes.
- **Push:** never automatically. Ask before each push, even when the intake says `push: ask`.

For a separate post-run [engine contribution](../../optimizing-inference/references/engine-contribution.md), reuse an explicit opt-in covering its branch push and PR. Overriding `push: never` requires a later explicit user decision recorded without editing the pinned intake; unrelated pushes stay unauthorized.

## When to commit

- **Kept change:** it passed its fast correctness check and its screen. Commit it, alone, with the screen numbers in the message. A change the report claims, or a milestone candidate, also needs an interleaved A/B with a CI; a claimed one also a profile showing the mechanism ([SKILL.md](../SKILL.md), "Checks").
- **Failed or slower experiment:** do not commit it to the branch. Revert the code and append one line to `runs/<run-id>/experiments.jsonl` with its numbers. Why: the history stays a clean chain of improvements, and nothing that was tried is lost.

## Commit message format

```
perf(engine): fuse q/k/v into one FP8 GEMV launch

<what changed and why, with the screen numbers, e.g. "step replays: 6.12 -> 5.98 ms/step";
a claimed change adds its profile evidence: e.g. "nsys ev_...: kernels per step 412 -> 340;
attention + qkv critical share 14.1% -> 12.0%">

Perf-Workload: Qwen3-8B FP8, RTX 5090, batch 1, prompt 1024, gen 256
Perf-Decode-Tok-S: 185.3 (95% CI 185.2-185.3) [+2.1% vs 1a2b3c4]
Perf-TTFT-Ms: 60.7
Perf-P95-Latency-Ms: 1452
Perf-Quality: top-1 agreement 98.7% (noise floor 98.8%)
Perf-Cost: $0.91 per 1M output tokens at $0.62/h
Perf-Evidence: ev_3f2a..., ev_9c1d...
Perf-Baseline: 1a2b3c4
```

(Trailer values above are the spec's illustration, not a measurement.) A screened change's Perf-Quality comes from its correctness record.

- Generate the trailer lines with `inferencekit ledger trailers <ev...>` (MCP `ledger_trailers`) from the evidence records this change has: its correctness record and, for a claimed change or a milestone candidate, the new benchmark and the comparison (a `compare` record; for served engines pass `--baseline-ev <the old benchmark>` instead, which gives a ratio of medians without a CI, so quote the CI from `bench rounds`). Never type a number by hand. Why: a hand-typed trailer is a number without an evidence id.
- **Speed headline per modality:** LLM decode tok/s and TTFT; TTS time to first audio and characters/s (or real-time factor); STT real-time factor and latency.
- **Tail latency:** p95 end-to-end, or per chunk when streaming.
- **Quality:** one number per modality (LLM: token agreement or the trajectory score against the reference, with the noise floor).
- **Cost:** $ per 1M tokens (characters, audio hours) at the measured speed and the GPU's hourly price.
- **Capacity goal active:** also `Perf-Capacity-N` (with the SLO it was measured against) and `Perf-Goodput`.
- The user can change the trailer set at intake.

## Reading the ledger

```bash
git log --format='%h %(trailers:key=Perf-Decode-Tok-S,valueonly)'
inferencekit history                 # table of commits with their trailers, sparkline per metric
inferencekit history --repo <path>
git switch -c restore <commit>       # go back to any checkpoint
```

The best commit per headline metric gets a tag `perf/best-<metric>`.

## `experiments.jsonl`

One JSON object per attempted change, kept or not, appended in order. Suggested fields:

```json
{"ts": "2026-10-13T14:02:00Z", "change": "GEMV register pipeline depth 4 -> 8 -> 16",
 "hypothesis": "ncu: ~70% long-scoreboard stalls, 54-66% of DRAM peak -> not enough loads in flight",
 "amdahl_bound": "<derived>", "baseline": "<commit>",
 "result": {"metric": "kernel us in a 36-layer graph", "before": "...", "after": "...", "verdict": "refuted"},
 "correctness": "bitwise", "kept": false, "commit": null, "evidence": ["ev_..."],
 "why": "o_proj unchanged, gate/up slower; the prologue, not bytes in flight, was the cost"}
```

Kept changes get a line too, with `"kept": true` and the commit hash, so the file is the complete list of attempts. Refuted ideas from this file go into the report's findings with the same rigour as wins.
