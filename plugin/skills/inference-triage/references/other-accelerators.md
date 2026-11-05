# Other accelerators: Intel GPUs and Google TPUs

Starting points only (read 14 Oct 2026). The rules in [hardware.md](hardware.md) apply unchanged: `bench rounds`, `gates run` and quality work against any OpenAI-compatible endpoint; `doctor`, `ceiling bw`, nsys and ncu are NOT RUN; measure bandwidth with the platform's own tools and pass the GB/s to `roofline`. Verify every engine by launching it.

## Intel GPUs (Data Center GPU Max, Arc, Arc Pro)

- **Engines:**
  - vLLM's XPU image `vllm/vllm-openai-xpu`, from vLLM v0.26.0: Python 3.12, tensor parallel supported, pipeline parallel in beta ([vLLM GPU install](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html)).
  - Arc Pro B60/B70 through Intel's `intel/llm-scaler-vllm` ([llm-scaler](https://github.com/intel/llm-scaler)).
  - llama.cpp with its SYCL backend ([build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)).
- **Profiler:** `unitrace -d` for device timing, or `--chrome-kernel-logging` for a timeline to open in Perfetto ([unitrace](https://github.com/intel/pti-gpu/tree/master/tools/unitrace)). Read it with the profiling skill's six questions.
- **Bandwidth:** no tool is prescribed here. A STREAM-style read test on the device (for example BabelStream's SYCL model) gives the measured figure; otherwise use the datasheet, labelled "measured ceiling NOT RUN".

## Google TPUs

- **Engine:** vLLM through the `tpu-inference` plugin (JAX, plus PyTorch through torchax), with recipes for v6e and v7x ([tpu-inference](https://github.com/vllm-project/tpu-inference)).
- **Profiler:** the repository's `tpu_profiling.py` example (the JAX profiler), viewed in the TensorBoard profile plugin or Perfetto ([profiling guide](https://github.com/vllm-project/tpu-inference/blob/main/docs/developers_guide/profiling.md)).
- **Bandwidth:** HBM bandwidth comes from the TPU generation's published specification unless you measure it with a JAX copy or read test. Label which one you used.
- **TPUs are rented by the hour too:** the approval and cost rules (GPU, $/h x hours, cap) apply in the same way.
