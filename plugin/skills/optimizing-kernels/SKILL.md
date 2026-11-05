---
name: optimizing-kernels
description: Profile-driven loop for building or patching inference engines and kernels on NVIDIA, AMD, Apple Silicon or CPU. Use after baseline tuning and profiling to close a measured bottleneck with one change, a correctness check and an end-to-end screen. Covers custom engines, fusion, attention, matrix kernels and memory traffic; routes to the hardware's supported tools.
license: Apache-2.0
compatibility: Uses the inferencekit CLI (Python 3.10+, uv or uvx) or its MCP server, the target engine's toolchain and git. Profilers depend on hardware; the bundled CUDA engine skeleton is NVIDIA-only. Written for Claude Code, Codex and OpenCode.
metadata:
  version: "0.1.0"
---

# Optimising kernels: the profile-driven change loop

Start from the measured critical path: host overhead, synchronization, poor memory/compute utilization, redundant traffic or repeated work. These losses occur in text, vision, audio and other models. This skill closes them one change at a time, each checked fast and kept as a revertible commit with its numbers; the full proof runs at milestones ([optimizing-inference](../optimizing-inference/SKILL.md) stage 7).

## When to use / not use

Use when:
- you have a ceiling, a benchmark and a profile, and want to close the gap (decode, prefill/TTFT, or the batched step);
- you are building a custom engine, patching an engine, or reviewing a kernel change.

Do not use when:
- you have not measured yet ([inference-triage](../inference-triage/SKILL.md) first: most guesses about the bottleneck are wrong);
- an engine flag might close the gap: tune engines first ([benchmarking](../benchmarking/SKILL.md), engine-configs).

## Authorization and parallel implementation

Reuse the run's [starting choices](../optimizing-inference/references/starting-questions.md), including quality and model-change permissions. Ask missing choices only when this skill is invoked alone. Device work stays inside existing authorization; `inferencekit intake budget runs/<run-id>` checks remaining resources, not a requirement to ask again.

Independent workers may write kernels in parallel in distinct files/branches. One coordinator integrates them sequentially, and one owner per device serializes warmup, captures, quality and timing. CPU-heavy builds on the benchmark host wait for measurements to finish ([parallel.md](../optimizing-inference/references/parallel.md)).

## Hardware and workload

Start with `doctor` and the [hardware route](../inference-triage/references/hardware.md). Use Nsight on NVIDIA, supported ROCm tracing/counters on AMD, Instruments/Metal/MLX capture on Apple, or a CPU profiler. For an unfamiliar architecture, runtime, flag or error, retrieve current official docs and verify installed support before selecting a tool. Keep unsupported checks NOT RUN; preserve available timeline or operator attribution with its tool version and source artifact, following [profiling](../profiling/SKILL.md).

The CUDA skeleton, commands and kernel recipes below are NVIDIA examples. Use the target engine's native backend elsewhere; a CUDA API name in PyTorch does not establish CUDA-tool support on ROCm. For images and speech, use the registered media adapter, structural and semantic checks, and complete request timing in [affordability](../optimizing-inference/references/affordability.md). Token checks, GSM8K and the `ikstep` LLM jobs below do not prove media correctness. Keep the same fast-check, milestone and full-proof stages with the workload's own checks.

When the user authorizes numerical changes, judge image/speech kernels against the declared modality quality budget and hash-bound external protocol. Different output bytes alone do not reject that track; exact identity remains a pure-refactor requirement or a diagnostic. Each intermediate change gets one short fixed public correctness/quality screen; promoted milestones and the final candidate need full registered quality and balanced timing. Preserve own-output integrity and all cases. Do not infer permission to prune architecture or use another model from a lossy budget.

An explicit perceptual/task acceptance policy may make numerical similarity diagnostic as well. In that case, do not discard a kernel or baseline solely for LPIPS/SSIM, cross-engine tensor/PCM equality or rounding drift. Measure the prospectively chosen mathematical controls and task/perceptual checks; fresh blinded quality is still required for promotion. No missing perceptual evidence becomes PASS.

## Research and inspect before implementing

Load [researching-optimizations](../researching-optimizations/SKILL.md) before selecting custom work and again when progress stalls or the bottleneck moves. Check existing engine features and emitted compiler code first. Follow [compiler inspection](references/compiler-inspection.md) from source through graph and dispatch to the executed kernel; inspect IR/assembly when it can test a specific hypothesis.

Use the smallest useful experiment, reuse the harness and supervisor, and keep the old path runnable. Avoid unused helpers, headers and abstractions. Independent experiments can be written concurrently; each integrated A/B isolates one change.

## Preflight checklist

- [ ] Ceiling and % of ceiling known (ev_roof, ev_bench). Why: the size of the prize and when to stop.
- [ ] A profile of the steady step with critical-path shares (ev_tl, [profiling](../profiling/SKILL.md)). Why: tells you which of the four losses dominates.
- [ ] A correctness baseline: reference outputs and the A/A noise floor ([proving-correctness](../proving-correctness/SKILL.md)). Why: every change below can silently break numerics.
- [ ] A harness that A/Bs two builds in interleaved fresh-process rounds. Why: the end-to-end effect is the only one that counts.
- [ ] Record build versus patch and both headroom bounds ([stop-point.md](../optimizing-inference/references/stop-point.md)). Any new engine meets the [custom-engine contract](../optimizing-inference/references/candidates.md#custom-engine-contract); the bundled skeleton covers NVIDIA text only.
- [ ] A repo and branch for the ledger (`perf/<model>-<hardware>`, from the intake's `repo`), and `runs/<run-id>/experiments.jsonl`. Why: every kept change becomes a revertible checkpoint; nothing tried is lost.

## Build route: NVIDIA text engine skeleton

The plugin ships a correctness-first engine ([skeleton/README.md](skeleton/README.md)) that already meets the custom-engine contract: an OpenAI streaming server, seeded sampling, top-20 logprobs from the full output layer, a checkpoint loader, a CUDA-graph decode loop and torch modules per architecture.
1. Copy `skeleton/ikengine/` (next to this file; `${CLAUDE_PLUGIN_ROOT}/skills/optimizing-kernels/` when installed) into the run directory and commit it as the first ledger commit.
2. Run its reference server on the model (`python -m ikengine.serve --model <dir> --port <port>`) and validate it as its README says: its model check, the gates, and quality against the reference. It is your correctness anchor and A/A partner. No module for the architecture: write one against the `Model` contract in `ikengine/engine_api.py`, correct first.
3. Replace internals behind the `Model` contract one change at a time (the loop below): kernels, fusion, byte cuts. Keep the server, the sampler, the logprob path and the proofs.
4. Never ship or claim a build without its proofs (below).

## Checks and implementation loop

1. Profile the supported platform once per lever area; repeat when the bottleneck moves. Identify the critical-path limiter, not summed kernel duration.
2. Read compiler/dispatch evidence and current research. Choose one hypothesis with a headroom bound and a falsifying test.
3. Implement behind a switch. Keep the old path runnable; independent alternatives may be written concurrently.
4. Run one fast screen on representative inputs, including awkward shapes. Exact changes preserve the declared outputs. Legacy lossless text changes use token/logprob checks with near-tie analysis; authorized lossy changes use their task/perceptual screen. Compare against a fixed reference as well as the previous build to catch cumulative drift.
5. Screen the complete step/request using the existing timer. Keep a reversible commit and numbers, or log the rejection. A faster microkernel alone is insufficient.
6. Promote promising milestones with the [state.md](../optimizing-inference/references/state.md) schedule: text uses `ikstep job light`; other modalities use registered checks. Baseline proof is reused until its config changes.
7. Run full proof once for the final candidate and any commit handed over (`ikstep job full` for text). A claimed change needs interleaved A/B confidence intervals and a profile showing the mechanism. Missing reference: use the [no-reference route](../proving-correctness/SKILL.md#no-higher-precision-reference).
8. Follow the approved stop rule and `goals.after_target`. A stated deadline: work to it minus the report reserve. Explicit zero reserve means continuous validation/packaging. Log the next untried lever and why.

Do not run full gates, task evaluations, balanced rounds or a new profile after every edit. Exploration screens never become published proof. Final timing includes the complete client request; label step/microkernel measurements separately.

Commit trailers and retained/rejected experiment records: [commit-ledger.md](references/commit-ledger.md). Use the platform profiler and registered client timer; text CLI examples are in [harness-commands.md](../optimizing-inference/references/harness-commands.md).

## Choose a technique from evidence

These apply across model families; select by the measured problem, not a list of fashionable names. For each, estimate headroom, confirm the mechanism, screen quality, then time the whole request.

| Observed problem | NVIDIA | AMD | Apple | Tradeoff / proof |
|---|---|---|---|---|
| CPU launches and tiny kernels | CUDA Graphs, fusion, device sampling | Supported HIP graph/runtime capture, Triton/HIP fusion | Reuse Metal command submission, MLX compiled/fused paths | Verify fewer gaps; include dispatch and synchronization. Sampling applies only where the workload samples. |
| Repeated intermediate traffic | Producer-consumer fusion, layout/producer quantization | HIP/Triton/CK layouts and fused epilogues | Metal/MLX fusion, layouts and quantized producers | Measure bytes and end-to-end cost; quantization needs permission. Fusion can increase registers. |
| Poor matrix utilization | Shape-specific GEMM/GEMV, tiling, split-K, supported matrix instructions | gfx-specific MFMA/WMMA, CK/Triton/other supported compiler | Metal SIMD-group matrices or native MLX kernels | Include packing/reduction overhead and shape tails; matrix paths are generation/dtype specific. |
| Synchronization dominates | Persistent kernels or megakernels | Supported persistent HIP kernels with explicit synchronization | Supported persistent/threadgroup design; do not assume grid-wide barriers | Check residency, occupancy, deadlock hazards and actual saved critical path. HazyResearch examples are hardware-specific research. |
| Model pages or barely fits | Packing, cache sizing, expert residency, staged loading/offload | Same memory strategies with supported allocator/transfer paths | Unified-memory working set, packing, cache limits and staged loading | Peak host/device memory, swap, complete latency and quality; fitting alone is not a latency win. |
| Repeated computation | Validated state/prefix reuse or incremental updates | Same algorithm with native runtime state | Same algorithm with native runtime state | Reset/invalidation across requests and changing inputs; approximate reuse needs quality permission and a separate realistic workload. |

See [research sources](../researching-optimizations/references/sources.md) for current implementations. CPU routes use the same hypotheses with vectorization, threading and cache-aware libraries.

## NVIDIA text playbook: decode (memory-bound; batch 1 examples)

| Technique | Loss it closes | Mechanism evidence | Deep dive |
|---|---|---|---|
| CUDA graph for the whole step | host launch overhead | nsys: one graph launch per step, idle gaps gone | [launch-overhead.md](references/launch-overhead.md) |
| GPU-side sampling, no per-token host sync | CPU on the critical path | nsys: no blocking D2H before the next launch | [launch-overhead.md](references/launch-overhead.md) |
| Fusion: norm+RoPE+KV write, one QKV GEMV, gate-up with SiLU*up epilogue, residual+norm, act-quant into the producer | hand-offs (kernel count) | nsys: kernels per step down, step wall down | [launch-overhead.md](references/launch-overhead.md) |
| FP8 block-scaled GEMV at M=1 (many blocks, 16-byte loads, per-block scales) | library GEMM tiles that cannot fill the GPU at M=1 | ncu: DRAM % of measured peak up, waves >= several | [fp8-gemv.md](references/fp8-gemv.md) |
| Split-K decode attention (flash-decoding) | too few blocks at batch 1, long context | ncu: waves up; nsys: attention's critical share down | [attention-split-k.md](references/attention-split-k.md) |
| Programmatic dependent launch, trigger placed late | remaining hand-off between dependent kernels | step wall down in A/B; overlap between kernels shrinks | [launch-overhead.md](references/launch-overhead.md) |
| Cheaper kernel prologue (hoist weight loads before the PDL wait, stage inputs asynchronously) | fixed per-kernel setup before streaming starts | prologue-only variant timing; in-step time above the bandwidth ideal | [lessons.md](references/lessons.md) |
| lm_head: precision, exact bound pruning (below), fusion with sampling | output-layer bytes measured for this checkpoint | bytes/token down; quality verdict at the logits | [fp8-gemv.md](references/fp8-gemv.md), [byte-cuts.md](references/byte-cuts.md) |
| Exact byte cuts: lossless re-encoding of 16-bit weights, bound-pruned output layer | bytes you did not need to read (moves the ceiling) | bytes/token down; proof tests pass; % of the recomputed ceiling | [byte-cuts.md](references/byte-cuts.md) |
| MoE: gather-GEMV over the top-k experts, routing fused into the first expert kernel, top-k-order sum | library grouped-GEMM tiles and many helper kernels per MoE layer | nsys: kernels per layer down; ncu: DRAM % of measured peak up | [moe-decode.md](references/moe-decode.md) |
| Persistent kernel / megakernel | whatever hand-off is left | measured hand-off sum first | [launch-overhead.md](references/launch-overhead.md) |

Why fusion helps at batch 1: activations are tiny (a 4,096-wide BF16 vector is 8 KB), so fusion saves almost no bytes. It saves *kernels*, and each dependent kernel costs a hand-off of the order of a microsecond or more.

**Exact byte cuts** keep the model identical (lossless track, no precision decision) and move the ceiling: estimate the exact byte floor at the start ([ceilings.md](../optimizing-inference/references/ceilings.md), "Exact byte floor") and work them alongside kernels. Re-encoded weights store out-of-window elements as raw 16-bit values; a pruned head serves logprob requests from the full head. Each needs its proof recipe and a recomputed ceiling, and a `rare_tokens` PASS at the next milestone ([byte-cuts.md](references/byte-cuts.md)); a lossy truncated vocabulary stays a [hack](../benchmarking/references/reward-hacks.md).

## NVIDIA text playbook: prefill / TTFT (compute-bound for long prompts, host-bound for short ones)

For short prompts investigate launch overhead; for long prompts inspect matrix utilization and neighboring elementwise work. NVIDIA text examples and traps: [prefill.md](references/prefill.md).

## Lessons that cost people weeks

- **Tune and screen inside the real step.** A configuration tuned in isolation (warm or flushed L2, no neighbours, no PDL overlap) can lose in context. Re-tune and rank variants by whole-step CUDA-graph replays (seconds); only the best get a `bench serve` screen, and only claimed ones the A/B; a wait loop stops when the server dies.
- **Amdahl first, on the critical path.** A 2x kernel at 4% of the step is ~1.02x; under PDL, use critical-path shares, not summed durations.
- **Counters suggest; timing decides.** ncu pointed at bytes in flight; a pipeline-depth sweep refuted it; the prologue was the cost.
- **A faster kernel can slow its neighbours.** A fusion that saved critical-path time in its own slot gained almost nothing end to end, because the kernels after it slowed by a similar amount. Only the whole-step A/B tells.
More, with sources: [references/lessons.md](references/lessons.md).

## Artifact/evidence contract

Keep the candidate check, screen, commit or rejection, and experiment id. A claimed gain additionally needs interleaved timing, a comparison CI and mechanism evidence from the supported platform profiler. Preserve tool versions and source hashes; never invent Nsight records for other backends.

## Decision and no-go rules

- Host launch overhead dominates? Try supported graph or command-buffer execution first and verify its effect. CUDA graphs are the NVIDIA example; support elsewhere must be established.
- Step at ~85% or more of the bandwidth ceiling ([ceiling bands](../inference-triage/SKILL.md#ceiling-bands)): only fewer bytes per token help: an exact byte cut (above; start these early anyway, and recompute the ceiling for the new format) or lower precision (a precision decision with its own quality bar). Why: the other three losses are nearly gone.
- Never change two things in one A/B. Why: you cannot attribute the result, and one change may hide a regression in the other.
- Losing changes stay out of the branch but go into `experiments.jsonl` and the report. Why: the next person will try the same idea.
- A non-deterministic reduction (atomics in split-K) needs an explicit decision. Why: it breaks byte-identity testing and reproducibility.
- Push only within explicit authorization; do not ask again for an already authorized push. Otherwise commits stay local.

## Reporting rules

When this skill finishes a standalone engine optimization, check for an [optional engine contribution](../optimizing-inference/references/engine-contribution.md) after the final report. Within an optimizing-inference run, leave that step to its coordinator so the same result produces only one PR. Publication requires the user's explicit authorization and the completed engine-level proof, not a faster microbenchmark alone.

- Speedups come only from an interleaved A/B (`bench rounds` ratio CI: derived from observed rounds; `compare`: observed); Amdahl bounds and bytes saved are **derived**; "this worked because X" is **inferred** until the mechanism profile is attached.
- A claimed change without all three (correctness, interleaved A/B, mechanism profile) is UNPROVEN (or NOT RUN for the missing one), never "probably faster"; an unclaimed kept change is reported as screened. Engine-timed A/Bs (`bench decode`) are labelled as such.
