# Big model on a small GPU

"Model M runs on GPU G" is a capability claim, and a popular one. It survives review only with three proofs: **it fits** (the engine's own memory log, at a stated context and concurrency), **it is usable** (measured speed with a CI, TTFT included), and **it is still the model** (quality against the full-precision reference). This mode finds the configuration that earns all three, or reports honestly that none does.

## Inputs to pin first

- The model: the full-precision repo and sha. A smaller sibling, a distilled variant or a pruned model is a **different model**: never substitute one silently; it becomes a candidate only if the user allows it, and it is then named as such everywhere.
- The target GPU (one GPU unless the intake says otherwise) and its memory in GiB, from `inferencekit doctor`.
- The context the claim covers (a realistic prompt + output, e.g. 4k-8k tokens, not a 16-token prompt) and the concurrency (1 unless the intake asks for more).
- The usable-speed threshold: `goals.targets` (decode tok/s, TTFT), or ask. Without one, report the measured speed and make no "usable" claim.
- Offload: allowed only when `constraints` contains `offload: allowed`.
- The accepted quality loss: a margin tier ([measure-and-score.md](measure-and-score.md), section 4). This mode often reaches formats below 4 bits; agree their margins with the user before measuring.

## Search (GPU-free first)

1. **Candidates:** `quant-scout` lists every format of the model, including aggressive GGUF ones (Q3_K, IQ3, IQ2) when the 4-bit formats are too large.
2. **Fit** each one at the claim's context and concurrency ([bytes-and-fit.md](bytes-and-fit.md), section 4). For MoE models every expert must fit, unless offload is allowed.
3. **Order:** among the formats that fit entirely on the GPU, start with the **least aggressive** (most bits per weight): it has the best chance to pass quality. Go to smaller formats only when it does not fit at the needed context, or measures below the speed threshold.
4. **Offload** (only if allowed) when nothing fits on the GPU at an acceptable format. Prefer offloading MoE routed experts over dense layers: at batch 1 the CPU side then reads only `top_k / num_experts` of the expert bytes per token.
5. **Engine per format:** the aggressive GGUF formats and most offload paths run on llama.cpp; FP8, NVFP4 and INT4 safetensors run on vLLM, SGLang and TensorRT-LLM, which want everything in GPU memory unless their weight-offload option is used. Knob names to check in the pinned version's `--help` (priors): llama.cpp `--n-gpu-layers`, `--n-cpu-moe`, `--override-tensor`; vLLM `--cpu-offload-gb`.
6. **"What is the smallest GPU this model runs on?"** is the same search repeated per GPU size, smallest first; stop at the first size where a row passes fit, the speed threshold and quality. Each size is its own measurement: never extrapolate between GPUs.

## Offload speed, honestly

- The harness measures the GPU's DRAM bandwidth (`inferencekit ceiling bw`) and nothing on the host side. An offload row's ceiling is therefore **inferred**:
  `time per token >= GPU-resident bytes read / GPU bandwidth + host-resident bytes read / host-side bandwidth`,
  where the host-side bandwidth is the system memory bandwidth when the CPU computes those layers (llama.cpp CPU layers and CPU experts), or the host-to-GPU link bandwidth when the engine streams offloaded weights to the GPU every step. Name which one, and the source of its number.
- The reported speed is the measured one (`bench rounds`, client timer), like every other row. Never project it from the GPU-only ceiling, and never quote the GPU-only % of ceiling for an offload row.
- Prefill on offloaded layers is often far slower than decode suggests: TTFT at the claim's prompt length is a required column.
- Offload speed depends on the host: record the CPU model, core count, memory type and size (from `lscpu` and `free`, with no host names), and say the result does not transfer to another host.

## Quality without fitting the reference

- The full-precision reference does not fit the small GPU by construction. Serve it with vLLM on a larger GPU, or several (tensor parallel), as the intake's `quality.reference`, and put that GPU time in the plan and the cost cap.
- If the user will not pay for it, the best available reference is a higher-precision quant that fits somewhere cheaper, **served by vLLM**: `quality score` teacher-forces through vLLM's `prompt_logprobs`, so `--reference-url` must be a vLLM server. In practice that is an FP8 (or other 8-bit safetensors) checkpoint on vLLM. A Q8_0 GGUF qualifies only through vLLM's GGUF loader (experimental), and only after a smoke test shows that server returns `prompt_logprobs`; a Q8_0 on llama.cpp cannot be the KL/top-1 reference. Label every quality number "reference: <format> on vLLM, not full precision": the measured KL then understates the loss against the real model.
- A cheaper reference that is not on vLLM (e.g. Q8_0 on llama.cpp) can still anchor GSM8K (`quality gsm8k` works on any OpenAI-compatible server, and `--paired-with` its results), labelled the same way; KL and top-1 are then UNPROVEN.
- No reference at all: quality is UNPROVEN, the headline says so, and GSM8K accuracy alone is reported as an observation, not as "same quality".
- Formats below 4 bits (Q3_K, IQ3, IQ2, Q2_K) have no looser default margin: see the tier table in [measure-and-score.md](measure-and-score.md), section 4.

## The headline (checklist)

Every one of these appears with the claim, or the claim is not published:
- [ ] model id @ sha (the real model, not a sibling), format and checkpoint repo @ sha, engine @ version;
- [ ] GPU and its memory (GiB); the engine's log lines proving the weights loaded and the KV capacity in tokens;
- [ ] the context length and concurrency actually configured (the limits it ran at, not the model's maximum);
- [ ] decode tok/s and TTFT, medians with CI95, from `bench rounds` at that context; the bucket;
- [ ] what was offloaded, how many bytes per token cross to the host side, and the host's CPU and memory;
- [ ] quality vs the reference (KL and top-1 vs the floor, GSM8K delta), with the tier and the verdict; the reference's own precision;
- [ ] gates PASS; NOT RUN items (longer contexts, more concurrency, other GPUs).

Refuse these, even when asked:
- speed measured on a tiny prompt and quoted as general;
- "runs" meaning "loads", with no quality check on what it generates;
- model-card benchmark scores in place of measured quality;
- TTFT left out because it is embarrassing;
- speculative decoding or MTP without the exactness gate;
- a number from one GPU or host presented as if it holds on another.
