# ncu capture recipes

## Anchoring the kernel regex

ncu filters kernels by name, then applies `--launch-skip` and `--launch-count` to the *matching* launches. An unanchored pattern such as `gemv` can match a quantisation helper, an lm_head kernel and a split-K reduction as well as the kernel you want, and the skip count then lands somewhere unpredictable.

- Take the exact name from the nsys timeline (the short function name, not the full template signature).
- Anchor it: `^fp8_block_gemv_kernel$`. For a templated family, anchor the prefix and check what matched: `^fp8_block_gemv_kernel`.
- After the capture, read which kernels and launch indices were profiled in the record. If more than one distinct name matched, tighten the regex and recapture.

## Sizing `--launch-skip`

The first launches of a kernel are not representative:

- **warm-up requests:** cold L2, cold TLBs, first-touch page faults;
- **autotuning** (Triton, some CUTLASS paths): the kernel runs with many candidate configurations;
- **graph capture:** engines often run one eager pass per batch size before or while capturing CUDA graphs;
- **prefill:** the same kernel name may run with a much larger M during the prompt.

Recipe:
1. From a node-traced nsys capture of the same command, count launches of the matching kernel before the first steady decode step: `N_pre`.
2. Count launches per decode step: `k` (for a per-layer kernel, usually the number of layers).
3. Set `--launch-skip` to `N_pre + 2k` (skip two more steps for margin) and `--count` to a few launches from the same step.
4. Shapes differ per layer (q/k/v vs MLP): make sure the profiled launches are the shape you care about, or profile a whole step's worth and group by shape.

## Clocks

ncu's default `--clock-control base` locks the GPU to base clocks for stable counters. On a shared or rented GPU this needs privileges, affects other users and makes the kernel slower than in production. inferencekit uses `--clock-control none`: clocks float as in production, so absolute durations vary by a few percent. That is fine because ncu durations are only used to rank and to explain, never to claim.

## Memory during replay

Kernel replay saves the device memory a kernel may write and restores it before each pass, so every pass sees the same inputs. The saved copy needs memory:

- engines that reserve most of the GPU up front (for example a KV cache pool sized to 90% of memory) can fail with out-of-memory, or fall back to slow host-side saving;
- mitigations: lower the reservation (vLLM `--gpu-memory-utilization`, SGLang `--mem-fraction-static`), reduce max context, profile a smaller test harness that calls the same kernel with the same shapes;
- fewer metric sections means fewer passes and less time, but not less saved memory.

Application replay (re-running the whole program once per pass) avoids the save/restore but requires a deterministic program and multiplies run time; inferencekit does not use it by default.

## Counters blocked

`ERR_NVGPUCTRPERM`: the driver bars this user from performance counters (`doctor` says `admin_only`, the driver's `RmProfilingAdminOnly=1`). Ways out: a VM or bare-metal machine whose driver setting you control, or the documented admin setting on your own machine; a container's root does not help without the capability. Never guess counter values ([SKILL.md](../SKILL.md), "Fallbacks").

**A supplied `.ncu-rep` without the ncu CLI:** ask for CSVs exported where ncu is installed (`ncu --import <rep> --page raw --csv` and `--page details --csv`), then `inferencekit ncu summarize raw.csv --details-csv details.csv --peak-gbs <ev_bw or GB/s>`.

## Short kernels

Stall reasons come from periodic warp-state sampling. A 3 us kernel yields very few samples per launch, so its stall percentages are noisy. Profile several launches (`--count 5` or more) and look at the stable top one or two reasons; treat the rest as noise. Throughput metrics (DRAM bytes, instructions) are counted, not sampled, and are reliable even for short kernels.
