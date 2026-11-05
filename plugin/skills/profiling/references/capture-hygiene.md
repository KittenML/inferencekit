# Capture hygiene for Nsight Systems

## Why `env -i`

An `.nsys-rep` stores the profiled process's environment so the report can be reproduced. Anything in that environment travels with the file: cloud API keys, Hugging Face tokens, notebook tokens, database URLs. On rented machines the provider often injects its own credentials (an instance API key, a notebook token) into every shell. Reports get attached to issues, chat threads and pull requests. So:

- `inferencekit nsys capture` and `inferencekit ncu capture` start the profiler under `env -i` with a short allowlist. Use them.
- Before the capture, make the run need no secrets: download weights in advance and set `HF_HUB_OFFLINE=1`.
- `inferencekit nsys capture` refuses secret-looking `--keep-env` names (`*TOKEN*`, `*KEY*`, `*SECRET*`...), because nsys stores the profiled command line and `NAME=value` for every variable it passes. The `.sqlite` copy it keeps has those values, the working directory, user and host name blanked; the `.nsys-rep` cannot be edited and keeps them.
- Before sharing any trace, run `inferencekit nsys analyze` on it: `data.capture_env_names` lists the variables the file stores, and a limitation says when their values and the command line are still in it. Treat `.nsys-rep` files as sensitive even after `env -i`: they contain hostnames, paths and command lines.

A minimal allowlist for a manual capture:

```bash
env -i \
  PATH="$PATH" HOME="$HOME" LANG=C.UTF-8 \
  CUDA_VISIBLE_DEVICES=0 HF_HOME="$HF_HOME" HF_HUB_OFFLINE=1 \
  nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace=graph \
    --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o decode_graph python run_decode.py
```

`$PATH` and friends are expanded by your shell before `env -i` clears everything else, so only the named variables reach the profiler. Add `LD_LIBRARY_PATH` only if the program needs it.

## Choosing what to trace

| Option | Why |
|---|---|
| `--trace=cuda,nvtx,osrt` | CUDA API and kernels, your NVTX markers, OS runtime calls (blocking waits) |
| `--sample=none --cpuctxsw=none` | no CPU call-stack sampling or context-switch tracing: lower overhead, no stack contents in the file. Turn sampling on only when you need Python/C++ hot spots on the CPU |
| `--cuda-graph-trace=graph` | one range per graph launch; realistic host timings |
| `--cuda-graph-trace=node` | every kernel inside graphs; needed for kernel ranking; inflates `cudaGraphLaunch` host time |
| `--capture-range=cudaProfilerApi` | record only between `cudaProfilerStart()` and `cudaProfilerStop()` (`torch.cuda.profiler.start()` / `stop()`); keeps model load and warmup out |

## Marking steps

Wrap each decode step in an NVTX range so the analysis can split steps:

```python
import torch
torch.cuda.nvtx.range_push(f"decode_step {i}")   # host-side marker, start
logits = model_step(...)                          # enqueues GPU work
torch.cuda.nvtx.range_pop()                       # host-side marker, end
```

Remember the range measures host time; nsys projects it onto the GPU row to show when that work ran.

## Programmatic dependent launch (PDL) in a timeline

With PDL (Hopper and newer), kernel N+1 is launched before kernel N finishes. It runs its prologue, then waits in hardware (`cudaGridDependencySynchronize`, PTX `griddepcontrol.wait`) until N signals completion or exits. On the timeline:

- kernels overlap, and gaps between them can be negative;
- each kernel's duration includes its waiting time;
- the sum of kernel durations exceeds the step's wall time.

Report step wall time and busy time as an interval union. To see a single kernel's own execution time, use ncu, which runs kernels serialised.

## Overhead check

Before trusting gaps from a node-traced capture, compare per-step wall time under node tracing with a graph-traced capture of the same command. If node tracing adds more than a few percent per step, read gap sizes as upper bounds.
