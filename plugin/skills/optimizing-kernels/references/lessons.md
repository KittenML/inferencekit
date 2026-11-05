# Lessons from decode optimisation practice

Each lesson: what goes wrong, why, and the habit that prevents it. These are experience reports from real optimisation work, stated generically; they are not measurements in this repository.

## 1. Isolated tuning picks the wrong configuration

**What happens:** a kernel is autotuned in a loop on its own. With the same weights read every repetition, they stay in L2 (warm), so configurations that rely on cache hits win. Flushing L2 between repetitions fixes that, but still misses what the real step does: the previous kernel leaves its own data in L2, PDL overlaps the prologue with the predecessor, and neighbouring kernels compete for SMs. The configuration chosen in isolation can make the whole decode step markedly slower than the one chosen in place.

**Habit:** shortlist candidates in isolation if you must, then choose by timing whole-step graph replays with each candidate, interleaved.

## 2. A 2x kernel is not a 2x step

**What happens:** a kernel that is 4% of the step gets 2x faster; the step gets 1.02x faster, below the noise of most benchmarks.

**Habit:** compute the Amdahl bound `1 / ((1 - f) + f / s)` before starting, and only start if the bound is comfortably above your benchmark's CI half-width.

## 3. Eager microbenchmarks invert conclusions

**What happens:** a new kernel loses in an eager PyTorch loop (launch overhead dominates both) but wins inside a CUDA graph, or the reverse.

**Habit:** benchmark kernels the way production launches them: inside the graph, with PDL if production uses it.

## 4. PDL trigger placement

**What happens:** triggering dependents at kernel entry lets many waiting kernels occupy SM resources, starving the producer; the step gets slower although every kernel "starts earlier".

**Habit:** prefetch weights before the wait; trigger after the main loop by default and early only for short kernels; compare both placements in an A/B ([PDL placement](launch-overhead.md#pdl-placement)).

## 5. Library GEMMs at M = 1

**What happens:** library FP8/FP4 GEMMs with 64- or 128-row tiles under-fill the GPU at M = 1 and reach well under half of measured bandwidth. Low-precision formats then fail to pay off at decode, because the kernel, not the format, is the limit; and a separate per-GEMM activation-quantise kernel can cost more than the bytes saved.

**Habit:** check DRAM % of measured peak and waves in ncu before concluding "FP8 does not help". Fuse quantisation into the producer.

## 6. Megakernels have a floor too

**What happens:** a persistent kernel removes launches but every op still exchanges its output vector across all SMs through L2; measured, that exchange alone can cost about as much as a kernel boundary with PDL, and one block per SM hides latency worse.

**Habit:** measure the sum of hand-off gaps first; that is the most a megakernel can win.

## 7. "Bitwise equal" comparisons that prove nothing

**What happens:** a new kernel never writes its output; the comparison against the old kernel passes because both buffers hold the same stale data.

**Habit:** fill output buffers with NaN (or a sentinel pattern) before each kernel under test.

## 8. Stream-ordering bugs in test rigs

**What happens:** a test harness resets buffers or uploads inputs on one stream and replays a graph on another non-blocking stream that does not wait for it. Results become load-dependent and look like numerical drift.

**Habit:** make every cross-stream dependency explicit (events), and run the harness's own A/A control under load before trusting differences.

## 9. Profiles explain, A/Bs decide

**What happens:** an ncu run shows higher DRAM throughput, and the change ships, but the step did not get faster (the kernel's overlap with its neighbour shrank).

**Habit:** decide on timing, never on counters: the screen keeps or drops a change; a claimed change needs an interleaved A/B with a CI that excludes 1, plus a profile that shows the mechanism.

## Measured during early development (Qwen3-8B, RTX 5090)

Source: the early engine's development notes (not shipped; this page has what you need) and a llama.cpp Q8_0 test. The engine's code is in the companion repository, folder [qwen3-8b-fp8_rtx-5090](https://github.com/kittenml/inferencekit-engines/tree/main/qwen3-8b-fp8_rtx-5090). The numbers are engine-timed (the engine's own decode harness or `llama-bench`; rule 18), the ratios from interleaved rounds; none is a client-timed number.

| Lesson | What happened | Number |
|---|---|---|
| Profile before guessing | nsys showed only 1.4% idle between kernels in the first engine version, with the whole step already in one CUDA graph, so it was not launch-bound; the loss was inside the kernels | 219 kernels per step, median gap 0.42 us |
| PDL gains match the idle it can hide | Programmatic dependent launch added about what the idle share predicted | +1.43% (7 rounds, identical tokens) |
| Counters suggest, timing decides | ncu: GEMVs at 54-66% of DRAM peak, ~3 warps per SM, ~70% long-scoreboard stalls. A deeper register pipeline (4 -> 8 -> 16) did not speed up o_proj and made gate/up slower | refuted |
| Find the fixed cost | A prologue-only variant cost 5.7-9.2 us against 14.4-40.5 us full kernels; hoisting loads before the PDL wait and staging inputs asynchronously cut it | ~-350 us per token (derived) |
| Load balance was not the limit | o_proj on only 64 SMs was just 5% slower than on all of them | refuted |
| Trigger placement matters | Triggering the next kernel 2 K-blocks before the end of the main loop let its weight prefetch overlap the current kernel's drain; triggering 8 blocks early was neutral to worse | +1.76%; then +0.79% more from splitting the prefetch |
| Access pattern | K-major weight tiles, so all warps read one contiguous window | +0.62% (+0.76% in a second session) |
| Summed time lies under PDL | llama.cpp: activation quantisation 8.7% of summed kernel time, 2.5% of the critical path; quantising once instead of three times per layer gave about what the critical path predicted | +0.8% (byte-identical logits) |
| A faster kernel can slow its neighbours | A fusion that saved critical-path time in its own layer slot gained almost nothing end to end, because neighbouring kernels slowed by a similar amount | within noise of zero |
| One session is not the truth | The same engines moved between sessions (vLLM 1-3% slower in one session, cause not identified); decisions used interleaved ratios, not absolute tok/s | |
