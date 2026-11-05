# PyTorch profiler fallback (no hardware counters)

The PyTorch profiler records CPU ops, CUDA runtime calls and GPU kernels through CUPTI activity tracing. It reads **no hardware counters**, so it works where ncu is blocked (`ERR_NVGPUCTRPERM`), and it needs no Nsight install. Use it when nsys cannot run on the host, or when you want PyTorch-op attribution next to the kernels. It gives a timeline (the six questions' 1-4), never a counter-based bound label.

**Limit:** the inferencekit CLI does not parse these traces (`nsys analyze` and `ncu summarize` read Nsight files only). Analyse them by hand (below) or with the `profile-analyst` agent, and label every number **observed (torch profiler trace)**, citing the file. They are not inferencekit evidence records, so `inferencekit report` cannot include them.

**Overhead:** profiling slows the run substantially (stack and shape recording add more), and traces grow fast. **Never time with the profiler on:** keep `--profiler-config` and profiler code out of every manifest launch line and every timed server, send only a few requests while it records, and never quote a trace duration as speed. GPU approval applies as for any capture.

## vLLM (server)

Checked against the vLLM v0.31.0 docs, [Profiling vLLM](https://docs.vllm.ai/en/v0.31.0/contributing/profiling.html) (`--profiler-config` exists since v0.13.0):

```bash
vllm serve <model dir> <same flags as the config you explain> \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile"}'
curl -X POST http://127.0.0.1:<port>/start_profile
# send a few requests (curl, or a short client run whose numbers you discard)
curl -X POST http://127.0.0.1:<port>/stop_profile      # flushing can take minutes for a large run
```

- Config keys: `torch_profiler_with_stack` (on by default), `torch_profiler_record_shapes`, `torch_profiler_with_memory`, `torch_profiler_with_flops` (off), `torch_profiler_use_gzip` (on). Turn stacks off when you only need kernels: less overhead, no source paths in the file.
- Each worker writes a gzipped trace into `torch_profiler_dir`; the docs view them in Perfetto.
- Newer vLLM docs add options to skip warm-up iterations and stop automatically; check the docs of the version you installed before using them.

## SGLang (server)

From the SGLang docs, [Benchmark and Profiling](https://docs.sglang.io/developer_guide/benchmark_and_profiling.html):

```bash
SGLANG_TORCH_PROFILER_DIR=./sglang_profile python -m sglang.launch_server <same args as the config you explain>
curl -X POST http://127.0.0.1:<port>/start_profile -H "Content-Type: application/json" \
  -d '{"output_dir": "./sglang_profile", "start_step": 5, "num_steps": 10, "activities": ["CPU", "GPU"]}'
# send a few requests; with num_steps it stops by itself, otherwise:
curl -X POST http://127.0.0.1:<port>/stop_profile
```

- `start_step` skips warm-up steps; `num_steps` bounds the window. Output is `*.trace.json.gz`, one file per rank (TP/DP/PP/EP in the name), viewable in Perfetto or `chrome://tracing`.

## Custom engines (`torch.profiler`)

[torch.profiler](https://docs.pytorch.org/docs/2.14/profiler.html), around the steady decode loop:

```python
from torch.profiler import profile, schedule, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             schedule=schedule(skip_first=20, wait=1, warmup=2, active=5),   # skip load, warm-up, graph capture
             on_trace_ready=lambda p: p.export_chrome_trace(f"trace_{p.step_num}.json")) as prof:
    for _ in range(40):
        decode_step()
        prof.step()                 # one ProfilerStep#N range per decode step
```

Leave `record_shapes`, `with_stack` and `profile_memory` off unless you need them: the docs warn that shape and stack tracing add overhead. Make the run need no secrets (weights downloaded, `HF_HUB_OFFLINE=1`; [capture-hygiene.md](capture-hygiene.md)).

## Reading a trace (Chrome / Perfetto JSON)

[Perfetto](https://ui.perfetto.dev/) shows the picture; compute the numbers from the JSON (gunzip first). `traceEvents` holds complete events (`"ph": "X"`, `ts` and `dur` in us) by `cat`: `kernel` (stream tracks), `gpu_memcpy` / `gpu_memset`, `cuda_runtime` (`cudaLaunchKernel`, `cudaGraphLaunch`, `cudaStreamSynchronize`, ...), `cpu_op`, `user_annotation` (CPU step ranges: `ProfilerStep#N`, or the engine's own) and, when present, `gpu_user_annotation` (the same ranges on the GPU).

1. **Steps:** a CPU-side range measures host time, and its GPU work often runs after it closes. Assign each `kernel` to the step whose range holds the launch that issued it (the kernel's `args.correlation` equals its `cuda_runtime` launch's), or use the GPU-side ranges. Keep steady steps only: drop warm-up, capture and any step with a profiler flush stall.
2. **Kernel time per step:** busy = the **union** of the step's kernel intervals; span = its first kernel's start to the next step's. With PDL or several streams kernels overlap and summed `dur` exceeds busy: rank families by their share of busy time, never by summed duration (SKILL.md, "Reading the timeline").
3. **GPU idle gaps:** idle = span - busy, plus the largest gaps between consecutive kernels. Above ~10% of the step, the host is on the critical path: look at the `cuda_runtime` row in the gap (launch storms, `cudaStreamSynchronize`, D2H copies of the sampled token).
4. **CUDA graphs:** check that kernels appear inside graph replays. A step that shows only a `cudaGraphLaunch` cannot rank kernels: say so (UNPROVEN).
5. **Profile check:** steady step time under the profiler vs the benchmark's ms/token. The profiler inflates it; a large gap means the shares describe a slowed-down run, so treat them as qualitative.

Shares and idle percentages are **derived** from the trace; any bound label is **inferred, no counters** (bytes a kernel must read / its in-step time vs measured bandwidth). Traces hold host names, user paths and, with stacks on, source paths: never commit or upload one; cite the file and your analysis.
