# Launch and hand-off cost at batch 1

## Terms

- **Launch overhead (host side):** CPU time to enqueue one kernel (driver call, argument packing). Eager PyTorch adds Python and dispatcher time on top, typically several microseconds per op.
- **CUDA graph:** a recorded sequence of kernels replayed with one `cudaGraphLaunch`. It removes per-kernel host cost but not the GPU-side cost of starting each kernel.
- **Hand-off (device side):** time between the end of one dependent kernel and the start of useful work in the next: the previous grid drains its last blocks (the tail), the next grid is scheduled and ramps up. It exists even inside a graph.
- **Programmatic dependent launch (PDL):** on Hopper and newer, lets the next kernel start its prologue while the previous one finishes, hiding part of the hand-off.

## Why it matters even when "memory-bound"

Example (illustrative numbers, derived): a Qwen3-8B decode step has on the order of 10-12 kernels per layer x 36 layers plus head and sampling, so roughly 400 kernels. If each dependent hand-off costs ~1.5 us, that is ~0.6 ms per token, about 13% of the 4.66 ms bandwidth floor on an RTX 5090. Fusion (fewer kernels) and PDL (cheaper hand-offs) attack this; faster GEMVs do not.

## Method 1: derive it from an nsys timeline (preferred)

1. Capture steady-state decode: `inferencekit nsys capture -- <decode command>` (skip warmup in the command, or select steps after it).
2. `inferencekit nsys analyze <rep>` gives per-step GPU busy and idle time, gaps between kernels and kernel counts.
3. Derive:
   - kernels per step (observed);
   - median and p90 inter-kernel gap within a step (observed);
   - hand-off floor per token = kernels per step x median gap (derived);
   - idle share = per-step idle / per-step wall (derived).
4. Caveats:
   - Node-level graph tracing (needed to see kernels inside a graph) adds overhead; check that per-step wall time under node tracing is close to wall time under graph-level tracing before trusting gaps.
   - With PDL, the next kernel starts *before* the previous ends, so gaps can be negative and kernel durations include waiting. Use per-step wall time, not sums of kernel durations.
   - A gap measures the scheduler hand-off only; ramp-up and tail inside each kernel show up as low DRAM throughput in ncu, not as gaps.

## Method 2: a trivial-kernel microbenchmark (lower bound, outside inferencekit)

A graph of N dependent trivial kernels, timed with CUDA events, gives the hand-off for a kernel that does almost nothing. Real kernels add ramp-up and tail on top, so treat this as a lower bound.

```python
import torch

x = torch.zeros(1, device="cuda")
side = torch.cuda.Stream()
side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side):          # warm up on a side stream before capture
    for _ in range(3):
        x.add_(1)
torch.cuda.current_stream().wait_stream(side)

N = 1000
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):          # record N dependent one-element kernels
    for _ in range(N):
        x.add_(1)

graph.replay(); torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
REPS = 20
start.record()
for _ in range(REPS):
    graph.replay()
end.record(); torch.cuda.synchronize()
print(f"{start.elapsed_time(end) * 1000 / (REPS * N):.2f} us per dependent kernel")
```

Each `x.add_(1)` is one tiny kernel depending on the previous one; the graph replays N of them, timed by CUDA events. The result is **observed** for a trivial kernel; any extrapolation to real kernels is **inferred**.

## Reading the result

| Hand-off floor / token time | Interpretation | Lever |
|---|---|---|
| < 5% | launches are not the problem | bytes per token, kernel efficiency |
| 5-15% | worth fusing the obvious pairs | fusion, PDL |
| > 15% | launch-bound at batch 1 | CUDA graphs first (if absent), then fusion, PDL; a persistent kernel only after measuring its ceiling |
