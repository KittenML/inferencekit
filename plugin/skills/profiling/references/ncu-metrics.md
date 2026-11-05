# Nsight Compute metrics for decode kernels

## Terms

- **SM (streaming multiprocessor):** one of the GPU's independent processors. A kernel's grid is split into blocks; each block runs on one SM.
- **Wave:** one round of blocks that fit on all SMs at once. A grid of 0.3 waves leaves 70% of SM slots empty.
- **Occupancy:** active warps per SM as a fraction of the maximum. Limited by registers per thread, shared memory per block and block size.
- **Eligible warp:** a warp ready to issue an instruction this cycle. Few eligible warps means the scheduler has nothing to do while loads are in flight.
- **Long scoreboard stall:** a warp waiting on a global/local memory load. The signature of not enough bytes in flight or a dependent chain of loads.
- **Speed of light (SOL):** ncu's section that reports throughput as a percentage of the hardware's theoretical peak for memory and compute.

## The metrics that matter at batch 1

| Question | Metric (ncu name) | Read it as |
|---|---|---|
| How long? | `gpu__time_duration.sum` | replay time; ranking only |
| How many bytes from DRAM? | `dram__bytes_read.sum`, `dram__bytes_write.sum` | compare with the bytes the kernel must read (weights + activations); far more means re-reads |
| DRAM throughput | `dram__bytes_read.sum.per_second` | divide by the **measured** bandwidth from `inferencekit ceiling bw` |
| ncu's own memory % | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | against theoretical peak; reads lower than the measured-peak percentage |
| Compute utilisation | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | high only for compute-bound kernels |
| Grid shape | `launch__grid_size`, `launch__block_size`, `launch__waves_per_multiprocessor` | waves below 1 means idle SMs |
| Occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` (achieved), `launch__occupancy_limit_registers` and siblings | which resource caps it |
| Registers | `launch__registers_per_thread` | high counts cap occupancy |
| Stall reasons | the Warp State Statistics section | top one or two reasons only, see short-kernel caveat |
| Load width | Source view / SASS: `LDG.E.128` vs `LDG.E` | 16-byte vector loads move 4x more per instruction than 4-byte loads |

## Why "% of measured peak", not "% of datasheet"

No kernel reaches datasheet bandwidth; a well-written streaming read on a recent GPU reaches some fraction of it, and that fraction differs per GPU. If you divide by datasheet, a kernel at speed of light looks 10-20% short, and you will chase a gap that does not exist. `summarize_ncu` divides by the measured value when it has one, and writes a limitation when it does not.

## Worked reading: an FP8 GEMV at M = 1 (illustrative)

Suppose the down projection of Qwen3-8B (4,096 x 12,288 FP8 = 50.3 MB) shows:

| Metric | Value |
|---|---|
| duration | 45 us |
| DRAM read | 50.6 MB |
| DRAM throughput | 1,124 GB/s = 70% of a measured 1,600 GB/s |
| SM throughput | 18% |
| grid | 64 blocks on 170 SMs (0.38 waves) |
| top stall | long scoreboard |

Reading (inferred): DRAM bytes match the weight size, so there are no re-reads. Throughput is 70% of measured peak with SMs mostly idle and less than half a wave: latency-bound, not enough bytes in flight. Lever: split K so the grid covers every SM several times, and use 16-byte loads. Expected best case (derived): 50.6 MB / 1,600 GB/s = ~32 us, so up to ~13 us per call, times 36 layers = ~0.47 ms per token. Then check with Amdahl and an A/B; ncu's 45 us is not the in-step time.

## What ncu cannot see

- Overlap with neighbouring kernels (PDL, multiple streams): replay serialises.
- L2 state left by the previous kernel in the real step: replay flushes caches between passes by default.
- Host-side and launch costs: not a kernel property.

## Traps that produced wrong answers in early development

- **L2-resident "bandwidth".** Bytes that fit in L2 (tens of MiB on recent GPUs) come from cache: such a bandwidth test beats the datasheet (the DRAM figure is the read plateau far above L2), and a small model or a tensor re-read across microbenchmark repetitions shows DRAM GB/s no in-step launch reaches. ncu flushes caches between replay passes, so in the real step a kernel can run from its predecessor's L2 and beat ncu, or lose when neighbours evict it.
- **Nominal vs measured peak.** `dram__throughput.avg.pct_of_peak_sustained_elapsed` (ncu's `dram_pct`) divides by the nominal peak, not by your measured bandwidth. Convert with `dram__bytes_read.sum.per_second / measured GB/s` before calling a kernel "at speed of light".
- **A zero is not a measurement.** A derived field that reads exactly 0.0 DRAM traffic for a kernel that must read its weights is a missing or broken metric (current inferencekit builds write null instead), never "no DRAM traffic". Check the raw counter in the `.ncu-rep` before using it.
- **Small launches next to big ones.** In a matrix-vector-heavy decode step, large launches can already run near DRAM peak while small ones (few rows, under one wave) run far below it and are latency-bound. Per-launch DRAM % then points at the small launches, but their share of the step sets the Amdahl bound; compute it before changing anything, and confirm any fix with a whole-step A/B, because a change that speeds one launch can slow its neighbours.
