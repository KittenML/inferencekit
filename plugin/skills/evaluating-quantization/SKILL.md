---
name: evaluating-quantization
description: Choose a compressed model format by measuring memory, complete-request speed and task/perceptual quality on the exact target hardware. Use for quantization, lower precision or fitting a model on cheaper hardware. Respect the chosen quality policy and separate model-change permissions.
license: Apache-2.0
compatibility: Needs inferencekit (Python 3.10+, uv or uvx), a compatible runtime and the workload's quality reference/adapter. Header-based fit research needs no GPU; measurements require authorized hardware. Text examples use text endpoints; media uses native pipelines.
metadata:
  version: "0.1.0"
---

# Evaluating quantization

Quantization can reduce memory and traffic, but conversion/dequantization can erase the speed benefit and outputs may change. This skill answers one question with evidence: **on this GPU and this workload, which format is fastest at a quality loss the user accepts?** The answer is a pair, speed and quality, measured on the same configuration. A faster number on its own is not a result.

Choose the modality before the checks. For image/audio, use [affordability](../optimizing-inference/references/affordability.md); for video, use [video checks](../benchmarking/references/video-checks.md). Record precision per component, peak weights/activations/scratch memory and complete output work. Text token metrics and GSM8K below apply only to text. Other tasks need a registered quality adapter.

Reuse [starting choices](../optimizing-inference/references/starting-questions.md): balanced permits numerical changes with preserved task quality; speed permits stronger approximations within agreed bounds. Neither authorizes a different architecture. Search [current implementations](../researching-optimizations/SKILL.md) for the exact device and shapes before assuming a smaller format has a fast kernel.

**Use** when a format or published quant may be worth it, when fewer bytes per token would help (in any [ceiling band](../inference-triage/SKILL.md#ceiling-bands); exact byte cuts belong to [optimizing-kernels](../optimizing-kernels/SKILL.md)), when the model does not fit or capacity needs KV room, or when [optimizing-inference](../optimizing-inference/SKILL.md)'s `model.precision_allowed` lists several formats. **Not** for same-precision engine comparisons, training a quantization recipe, or ranking different models.

**GPU approval:** steps 1-4 are GPU-free. Launching, `bench rounds`, gates and quality runs occupy a GPU: only with `budget.approved: true` in `intake.yaml` or an explicit yes to a stated cost (GPU, $/h x hours, cap), including a full-precision reference on a second, larger GPU.

For consumer VRAM/unified memory limits or minimum-memory baselines under latency and quality gates, also read [affordability](../optimizing-inference/references/affordability.md).

## The cross-precision rule (read first)

Evidence rule 8 ([evidence-rules.md](../inference-triage/references/evidence-rules.md)) forbids comparing speed across precisions **as an engine result**. This skill allows it under three conditions:
1. **Speed and quality as a pair.** Every format row carries its speed with CI **and** its quality verdict, measured on the exact speed configuration. A row without a quality verdict is NOT RUN on quality and cannot be recommended.
2. **Labelled** `cross-precision: format trade-off, not an engine comparison`. The speed ratio comes from `bench rounds` (`<metric>_ratio_to_best`, paired CI95) and is quoted as a ratio with its CI, never with `improved` / `regressed`.
3. **Same engine isolates the format** (e.g. vLLM BF16 vs FP8 vs AWQ). A row that changes engine **and** format (e.g. llama.cpp Q4_K_M) is a deployment option, labelled with both changes, never used to explain the format effect.

## Inputs (existing intake fields)

`model.*` (the full-precision reference; `precision_allowed` bounds the formats, others NOT RUN; `kv_cache_dtype_allowed: [fp8]` tests an FP8 KV cache), `hardware.*`, `goals.*`, `buckets` and `workload.*` (contexts and concurrency the KV must hold), `budget.wall_clock_h` (drop rows to fit, never rounds or quality), `fidelity` (a format below the stored one needs `fidelity: lossy` and its `precision_allowed` / `kv_cache_dtype_allowed` entry, in its own block: [fidelity.md](../optimizing-inference/references/fidelity.md)), `quality.budget` (replaces the tier defaults), `quality.reference` (a vLLM endpoint for the full-precision model; required when it does not fit the target GPU). `constraints`: `quality margins: <tier> kl=<nats> top1=<fraction> gsm8k=<points>` (tiers `fp8`, `int4`, `gguf5-8`, `sub4`), `offload: allowed`. No intake: ask for the same facts in one message. Margins are agreed before measuring; `non_interactive` takes the tier defaults ([references/measure-and-score.md](references/measure-and-score.md), section 4), logged in `QUESTIONS.md`.

## Preflight checklist

- [ ] The claim written before measuring: metric, bucket(s), the speed threshold that makes a format worth it (e.g. "decode tok/s >= 1.3x BF16, medium bucket") and the quality margin per tier. Why: a threshold chosen after the data is a forking-paths result.
- [ ] The reference is the full-precision checkpoint at a pinned 40-hex sha, served by vLLM (scoring uses its `prompt_logprobs`). Why: two quants agreeing with each other proves nothing.
- [ ] Every candidate pinned to its own repo and 40-hex sha. Why: community quants are re-uploaded under the same name.
- [ ] The KV cache dtype decided per row and identical across rows unless it is what is tested. Why: an FP8 KV cache halves KV bytes on its own.
- [ ] Same prompt set, `max_tokens`, chat template and temperature 0 for every row and the reference (speed at the workload's sampling).

## Hard failure gates

Stop that row (or the study) and report the gate when:
- the resolved config does not show the format you think you run (dequantized to BF16 at load, an unplanned weight-only fallback kernel);
- `inferencekit gates run` FAILs for the row: its speed is void;
- the quality reference is missing: quality is UNPROVEN and no format may be recommended over the reference;
- a row FAILs its pre-registered margin: it stays in the table as a loss, never the recommendation;
- offload would be needed without `offload: allowed` in `constraints`: the row is NOT RUN (does not fit), never quietly offloaded.

## Workflow (copyable)

```
Quantization check:
- [ ] 1. Claim, speed threshold and quality margins per tier written down (state.decisions or the plan)
- [ ] 2. Candidates: quant-scout -> repo@sha, format, size, provenance, engines (inferred), one table
- [ ] 3. Fit + ceiling per candidate, GPU-free: roofline --method headers (safetensors) or
         scripts/gguf_bytes_per_token.py (GGUF) -> bytes/token, predicted ceiling ratio vs the reference, VRAM need
- [ ] 4. Shortlist <= 6 (engine, format) rows that fit and are worth timing; every dropped row gets a reason
- [ ] 5. Approval; launch each row, smoke test, prove format + KV dtype from the log; one manifest entry per row
- [ ] 6. Speed: one bench rounds call over all rows (reference row included) -> ratio_to_best + CI; gates per row
- [ ] 7. Quality: reference A/A trajectories; trajectory per row; quality score per margin tier; gsm8k per row,
         paired with the reference
- [ ] 8. Pareto table, recommendation with its trade-off, labels, NOT RUN rows; claims-skeptic on the draft
```

## Steps

**1-2. Candidates (GPU-free).** Run the [quant-scout](../../agents/quant-scout.md) subagent once per model (Claude Code: `inferencekit:quant-scout`; elsewhere follow that file yourself): official checkpoints first, then the most used community quants, pinned, with recipe, size and engines (**inferred**, sourced). It never picks a winner. Formats: [references/formats.md](references/formats.md).

**3. Fit and predicted ceiling (GPU-free).** `inferencekit roofline --model <repo> --revision <sha> --method headers --context <ctx> --kv-dtype <dtype> --gpu-bw <ev_bw or GB/s>` per safetensors candidate at each bucket context; for GGUF, `python3 scripts/gguf_bytes_per_token.py --repo <repo> --file <name.gguf> --revision <sha>` plus the reference's `kv_bytes` (and `state_bytes`) at the same context. Fields, the Qwen3-8B worked example and the batch and compute-bound cases: [references/bytes-and-fit.md](references/bytes-and-fit.md).
- **Predicted ceiling speed-up** = reference bytes per token / candidate bytes per token (**derived**): an upper bound. 4-bit is never 4x fewer bytes: scales cost bits, lm_head and embeddings often stay BF16, and the KV cache is untouched and grows with context and concurrency.
- **Fit** = stored bytes (every expert, the full embedding table) + sequences x (KV at the context + recurrent state, from a roofline record run **at that context**; never slope x context on sliding-window or hybrid models) + runtime overhead. It is **inferred** until the engine logs its KV capacity in tokens.
- **Shortlist** 3-6 rows (reference included) that fit and clear the threshold with room to spare, same engine across formats first; dropped rows go in the report with the reason.

**4-6. Measure speed (same harness, every row).** Launch each row with the engine's recipe for that format, prefix caching off, CUDA graphs on, and prove from the startup log the quantization method, GEMM kernel family and KV dtype it **resolved** (a weight-only fallback is its own row). One manifest holds every row, each downloading its own pinned checkpoint ([references/measure-and-score.md](references/measure-and-score.md), sections 2-3). One `bench rounds` call interleaves all rows for paired ratio CIs; for capacity measure `N*` per row (KV room often decides it, and W4A16 can lose once compute-bound). `inferencekit gates run` per row while its server is up.

**7. Score quality** against the full-precision reference ([proving-correctness](../proving-correctness/SKILL.md); commands in [references/measure-and-score.md](references/measure-and-score.md), section 5): reference trajectories twice (A/A floor), one per row on its exact speed configuration, `quality score` per margin tier (below 4 bits: the 4-bit margins unless a `sub4` margin was agreed first), `quality gsm8k` per row paired with the reference (reasoning often degrades before KL looks alarming), `quality niah` for FP8 KV or long-context claims. Without logprobs the trajectory is text only: top-1 and NLL are scored by re-tokenizing it, KL and TV are not; GSM8K still counts.

**8. Report.** The Pareto table (all rows, losses, misfits and NOT RUN included) and the recommendation: the fastest row by the conservative end of its speed CI whose quality is PASS on every pre-registered check and which fits. If no quantized row passes, recommend the reference precision; if the fastest passing row's ratio CI includes 1, the format has no proven speed benefit (state any memory benefit). One sentence with both numbers, labelled. Table, frontier rule and wording: [references/measure-and-score.md](references/measure-and-score.md), sections 6-7.

**Big model on a small GPU** ([references/small-gpu.md](references/small-gpu.md)): fit every candidate at the claim's context and concurrency; start with the least aggressive format; quality needs a costed vLLM reference on a larger GPU (none: quality UNPROVEN in the headline); offload only with `offload: allowed`, labelled and measured, never projected.

MCP tools: `roofline`, `inspect_evidence`, `validate_manifest`, `bench_rounds`, `run_gates`, `quality_trajectory`, `quality_score`, `quality_gsm8k`, `quality_niah`, `report`. Check flag spellings with `inferencekit <command> --help`.

## Decision and reporting rules

- No recommendation without a quality verdict from the full-precision reference (the fastest format is the one that computes least). Quote the conservative end of each speed CI.
- Predicted ratios never stand in for measured ones, and never appear in the measured column. Do not extrapolate a result to another GPU, engine version, context, bucket or concurrency. Model-card quality claims are **inferred** context, never evidence.
- Bytes per token, predicted ratios and fit are **derived** or **inferred**; tok/s, KL, agreement and GSM8K are **observed** with evidence ids; `bench rounds` ratios are **derived** with their CI.
- Verdicts PASS / FAIL / UNPROVEN / NOT RUN; a trade-off claim passes only when both its speed and quality halves pass. Publish the losing formats, misfits and NOT RUN rows with the winner ([reporting-perf-claims](../reporting-perf-claims/SKILL.md)).
