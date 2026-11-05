# Launch overhead: graphs, fusion, PDL, megakernels, GPU-side sampling

## CUDA graphs

A CUDA graph records a sequence of kernel launches once and replays it with one `cudaGraphLaunch`. It removes the per-kernel host cost (Python, dispatcher, driver), which at batch 1 can exceed the GPU time of the kernels themselves.

Requirements and traps:
- **Static addresses and shapes.** Every replay uses the same buffers. Allocate fixed input/output buffers per batch-size bucket and copy new inputs into them before replay.
- **Data-dependent values live on the GPU.** Position, sequence length and KV slot indices must be read by kernels from device memory, not passed as host integers, or the graph bakes in the values from capture time.
- **No host syncs or allocations during capture.** `.item()`, `.tolist()`, `torch.cuda.synchronize()` and new allocations break or invalidate capture.
- **Warm up before capture.** Autotuners, lazy module loading and library heuristics must run before recording.
- **First replay is slower** (graph upload). Exclude it from timing.
- **Capture the whole step.** Engines that capture "piecewise" graphs and run attention eagerly between them keep a launch per layer; at batch 1 that is visible.

## Fusion

| Fusion | Kernels saved per layer | Notes |
|---|---|---|
| residual add + RMSNorm | 1 | writes the new residual and the normalised activation in one pass |
| q/k RMSNorm (Qwen3) + RoPE + KV-cache write | 2-3 | all per-head, all tiny |
| q, k, v projections as one GEMV | 2 | concatenate the weights along N once at load time |
| gate and up projections as one GEMV, SiLU(gate) * up in the epilogue | 2 | the epilogue reads the two halves of the output it just computed |
| activation quantisation into the producer's epilogue (W8A8) | 1 per GEMM | otherwise every FP8 GEMM pays an extra quantise kernel |

At batch 1 fusion saves kernels (hand-offs), not bytes. Verify with nsys: kernels per step should fall, and step time with it.

## Programmatic dependent launch (PDL)

Available on compute capability 9.0 and newer. The next kernel in a stream may launch before the previous one finishes, run a prologue, and then wait until the previous kernel signals.

- Host side: launch with `cudaLaunchKernelEx` and the attribute `cudaLaunchAttributeProgrammaticStreamSerialization` set to 1. In CUDA graphs the dependency becomes a programmatic edge.
- Consumer kernel: do independent work first (index maths, loading the weights it will need, which do not depend on the previous kernel), then call `cudaGridDependencySynchronize()` before reading the producer's output.
- Producer kernel: call `cudaTriggerProgrammaticLaunchCompletion()` once dependents may start; if it never calls it, the trigger happens implicitly when it exits.
- **Trigger placement matters** ([below](#pdl-placement)): triggering at the very start lets dependents hold SM slots (registers, shared memory) while they wait, which can slow the producer.
- **Profiling:** kernels now overlap; their durations include waiting. Judge PDL by step wall time in an A/B.

### PDL placement

In PTX, `griddepcontrol.wait` is the wait and `griddepcontrol.launch_dependents` the trigger (the CUDA calls above).
- **Prefetch weights before the wait.** Weights do not depend on the previous kernel: issue the first weight loads (into registers or shared memory) before the wait, then wait, then read the activation. The weight stream then overlaps the producer's tail.
- **Read the producer's output only after the wait, with ordinary coherent loads.** Drop `__restrict__`/`__ldg` on the activation pointer in a PDL consumer (keep it on weights): such loads count as read-only, and the compiler may move them above the wait. Check with `cuobjdump -sass` that the activation loads follow the wait.
- **Trigger late by default:** once this kernel has issued its own weight loads (at the end of the main loop, or a little before). Dependents prefetch while this kernel drains, and their waiting blocks take no SM slots while it streams.
- **Trigger early only for short kernels** (a few blocks, little to stream: norms, small reductions, routing), so the large kernel after them starts prefetching at once.
- **Measure both placements, and no PDL**, by whole-step A/B: the winner depends on blocks per SM, registers and shared memory.
- **Did not help in batch-1 engines we built** (interleaved in-step timing): triggering every kernel at entry; bulk L2 prefetch of upcoming weights, before or after the wait (inferred cause: it competes for the same DRAM bandwidth); a deep shared-memory staging ring on GPUs with little shared memory per SM (inferred cause: fewer blocks per SM); one block per SM. A build that skips the wait is a race: use it only to measure the boundary cost left, never as a result.

## Persistent kernels and megakernels

One long-lived kernel per step (or per model) whose blocks execute every operation in sequence, synchronising through global-memory flags instead of kernel boundaries.

- **Can win:** when data stays on chip between operations (a fused block that keeps intermediates in shared memory or registers), or when launches are not already hidden by graphs and PDL.
- **Often loses at batch 1:** every operation's output is one small vector that every SM needs for the next operation, so each op still ends in a grid-wide exchange through L2, much like a kernel boundary with PDL. One block per SM also gives less latency hiding than several smaller blocks, and register pressure rises.
- **It owns the GPU:** co-running work (another model, a second engine) cannot get SMs while it runs.
- **Measure the ceiling first:** sum the hand-off gaps in the current step (nsys, or in-kernel timers). The best a megakernel can do is remove them minus its own synchronisation cost. If that bound is below your bar, do not build it.

## GPU-side sampling and the host loop

- Sample on the GPU (argmax for greedy; top-k/top-p kernels otherwise) inside the graph, writing the token id to a device buffer the next step reads.
- Copy the id to pinned host memory asynchronously for streaming to the user; do not block the next step's launch on it.
- Let the host prepare step N+1 (scheduling, detokenising step N-1) while the GPU runs step N. Why: any per-token `cudaStreamSynchronize` or `.item()` on the critical path puts the CPU in series with the GPU.
- Stop conditions (EOS, max length) can be checked on the GPU and read back a step late, at the cost of one wasted step at the end.
