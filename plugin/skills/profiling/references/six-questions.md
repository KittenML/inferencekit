# Reading a decode timeline top-down: the six questions

Read from the outside in. Each question has a number that answers it, where to find that number, and the trap that most often produces a wrong answer.

## 0. Orient before answering anything

- Zoom all the way out. Identify the start-up region (model load, compilation, graph capture) and the steady region. Only the steady region counts.
- Confirm the steps look alike: per-step wall time p50 vs p90 from `inferencekit nsys analyze`. If they differ a lot, find out why (graph bucket changes, context growth, profiler buffer flushes) before generalising from one step.
- Check for profiler artefacts inside the window: CUPTI buffer flushes show as a few-millisecond stall on every thread. If one lands inside a step, exclude that step.

## 1. Does this stage matter?

- **Number:** stage share of per-token wall time (derived from per-step timings).
- **Where:** NVTX ranges projected onto the GPU row (not the CPU row), or kernel families summed per step.
- **Trap:** NVTX ranges on the CPU row measure *host* time. GPU work queued inside a range usually runs after the range closes. A 0.5 ms CPU range can own 4 ms of GPU work.

## 2. CPU or GPU on the critical path?

- **Number:** per-step GPU idle time and the longest gaps; count of blocking API calls per step.
- **Where:** gaps between kernels on the stream row; `cudaStreamSynchronize`, `cudaMemcpy` (D2H of the sampled token), `cudaEventSynchronize` on the CUDA API row.
- **Reading:** if the GPU row has gaps while the CPU thread is busy in Python or the scheduler, the CPU is on the critical path. If kernels run back to back and the CPU sits in a blocking call, the GPU is.
- **Trap 1:** a blocking `cudaMemcpyAsync` that "takes 4 ms" is usually the CPU *waiting* for earlier GPU work; the copy itself is microseconds. Compare against the GPU row.
- **Trap 2:** idle helper threads blocked in `poll`, `sem_wait` or `pthread_cond_wait` dominate OS-runtime statistics but cost nothing unless they are on the critical path.

## 3. Which stage?

- **Number:** share of step time per kernel family (attention, GEMV/GEMM, norm, rotary, activation, sampling, copies).
- **Where:** `analyze_nsys` kernel families; or group kernels by name pattern.
- **Expectation for a dense model at short context:** linear layers dominate (they carry the weight bytes), then attention, then the lm_head and sampling. Attention grows with context.

## 4. A few big kernels or a long tail?

- **Numbers:** kernels per step, median kernel duration, share of time in kernels under ~5 us, top-5 cumulative share.
- **Reading:** a long tail of short kernels means hand-off and ramp-up dominate (launch/latency-bound): fuse and use PDL. A few big kernels means kernel efficiency matters: go to ncu.
- **Trap:** "GPU busy 95%" does not mean the GPU is used. A kernel with fewer blocks than SMs leaves most of the GPU idle while counting as busy. Hover a kernel to see its grid size and compare with the SM count.

## 5. What is the speed-of-light gap?

- **Number:** for a big kernel, `bytes it must read / measured bandwidth` vs its observed duration.
- **Example (derived):** a gate+up GEMV for Qwen3-8B FP8 reads 2 x 12,288 x 4,096 = 100.7 MB. At a measured 1,600 GB/s that is ~63 us. If the kernel takes 90 us, it runs at ~70% of the bandwidth ceiling: worth an ncu look.
- **Trap:** with PDL, the kernel's duration includes time spent waiting for its predecessor, so the gap looks bigger than it is. Use ncu (which serialises kernels) for the kernel's own time, and the timeline for the step.

## 6. Amdahl: what is the most this can win?

- **Formula:** step speedup = `1 / ((1 - f) + f / s)` for a part that is fraction `f` of the step, made `s` times faster.
- **Which `f`:** the part's critical-path share of the step (`critical_pct_of_step` from `inferencekit nsys analyze`), never its share of summed kernel durations. Under PDL or multiple streams the summed share overstates it (early engine runs: 8.7% summed vs 2.5% critical for llama.cpp's activation quantisation).
- **Examples (derived):** `f = 0.04, s = 2` gives 1.02x. `f = 0.30, s = 1.5` gives 1.11x. `f = 0.10, s = infinity` (delete it) gives 1.11x.
- **Rule:** if the bound is below what your benchmark can resolve (its CI half-width), the change cannot be proven. Pick a bigger target.

## Useful SQL on the exported .sqlite

Times are in nanoseconds. Kernel names are ids into `StringIds`.

```sql
-- Top kernels by total GPU time
SELECT s.value AS kernel, COUNT(*) AS n,
       SUM(k.end - k.start) / 1e6 AS total_ms,
       AVG(k.end - k.start) / 1e3 AS avg_us
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN StringIds AS s ON s.id = k.shortName
GROUP BY s.value ORDER BY total_ms DESC LIMIT 15;

-- Gaps between consecutive kernels on one device (negative = overlap, e.g. PDL)
WITH ordered AS (
  SELECT start, end, LAG(end) OVER (ORDER BY start) AS prev_end
  FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId = 0
)
SELECT COUNT(*) AS n,
       AVG(start - prev_end) / 1e3 AS mean_gap_us,
       SUM(CASE WHEN start - prev_end > 5000 THEN 1 ELSE 0 END) AS gaps_over_5us
FROM ordered WHERE prev_end IS NOT NULL;
```

Restrict both queries to the steady window (a `WHERE k.start BETWEEN ...` on the step boundaries) before drawing conclusions. `inferencekit nsys analyze` does this per step for you; use SQL for questions it does not answer.

## Summing durations correctly

- **Busy time** is the union of kernel intervals, not the sum of durations. With overlap (PDL, multiple streams) the sum overcounts.
- **Step wall time** is first kernel start to last kernel end of that step, or NVTX step markers projected onto the GPU.
- **Idle time** is wall minus busy.
