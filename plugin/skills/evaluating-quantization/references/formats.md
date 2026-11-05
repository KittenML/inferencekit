# Formats: what each one stores, and where it can run

Everything about engine and GPU support below is a **prior to verify**: engines add and drop kernels every release, and your memory of them is out of date. `quant-scout` and `engine-scout` source each claim; only a launch on the target GPU, with the format shown in the engine's log, decides.

## Recognise a checkpoint's format

- **safetensors:** `config.json` -> `quantization_config`.
  - `quant_method` / `quant_algo`: `fp8`, `awq`, `gptq`, `compressed-tensors`, `modelopt` (NVIDIA ModelOpt: FP8, NVFP4, INT4 AWQ), `mxfp4`, `bitsandbytes`.
  - Granularity: `bits`, `group_size`, `weight_block_size` (block FP8, e.g. `[128, 128]`), per-tensor or per-channel scales.
  - What stays in full precision: `ignore`, `modules_to_not_convert` or `exclude_modules`. `lm_head` is the usual entry.
  - A declared KV cache format: `kv_cache_scheme` or `kv_cache_quant_algo`. `roofline` warns when the checkpoint declares FP8 KV and you counted BF16 (e.g. `nvidia/Qwen3-8B-NVFP4`). Whether the engine actually used it is read from the engine's log.
- **The roofline record:** `identity.model.dtype` lists the formats carrying at least 1% of the bytes read (e.g. `nvfp4+bf16`), `identity.model.quant` the algorithm, and `data.counts.read_bytes_by_format` the split. AWQ/GPTQ int4 values are packed into int32 tensors, so they show as `i32`.
- **GGUF:** the name ("Q4_K_M") is the recipe of the whole file, not one type per tensor. `scripts/gguf_bytes_per_token.py` lists `read_bytes_by_type`: a Qwen3-8B Q4_K_M file is Q4_K plus Q6_K tensors (Q6_K for the output matrix and some FFN and attention tensors).
- **The Hub's parameter counts lie about bytes.** The `safetensors.parameters` field of the model API counts logical parameters and groups them by the storage dtype, so packed INT4 weights show as billions of "I32" parameters. Never multiply those counts by a dtype size: use the headers.

## Bits per weight, as stored (scales included)

| Format | Layout | Bits per weight | Math precision in the GEMM |
|---|---|---|---|
| BF16 / FP16 | | 16 | 16-bit |
| FP8 E4M3, per-tensor or per-channel scale | | ~8.0 | FP8 x FP8 where the GPU has FP8 tensor cores (W8A8), else weight-only (W8A16) |
| FP8 block-wise | one FP32 scale per 128 x 128 block | ~8.002 | as above |
| GGUF Q8_0 | blocks of 32 int8 + one FP16 scale (34 bytes) | 8.5 | llama.cpp kernels |
| GGUF Q6_K / Q5_K / Q4_K | super-blocks of 256 (210 / 176 / 144 bytes) | 6.5625 / 5.5 / 4.5 | llama.cpp kernels |
| GGUF Q4_K_M (a file recipe) | Q4_K + Q6_K mix | measured 4.79 on Qwen3-8B's linear layers | |
| NVFP4 | FP4 (E2M1) values, one FP8 (E4M3) scale per 16, an FP32 scale per tensor | 4.5 | FP4 x FP4 on Blackwell (W4A4); weight-only elsewhere if the engine has it |
| MXFP4 | FP4 (E2M1) values, one E8M0 scale per 32 | 4.25 | FP4 on Blackwell; weight-only elsewhere |
| INT4 AWQ / GPTQ, group 128 | int4 packed in int32, an FP16 scale and a 4-bit zero point per group | ~4.16 (measured 4.157 on Qwen3-8B-AWQ) | weight-only (W4A16): dequantised to 16-bit inside the GEMM |
| GGUF IQ4_XS | super-blocks of 256 (136 bytes) | 4.25 | llama.cpp kernels |

Bits per weight times parameters gives the linear layers only. The checkpoint as a whole is always larger per weight, because of what stays in BF16.

## What usually stays in BF16 (why 4-bit is not 4x)

- **lm_head:** the vocab x hidden logits matrix, read in full every token. Most FP8, NVFP4 and AWQ recipes exclude it. On Qwen3-8B it is 1.245 GB: 8% of the BF16 model's per-token bytes, but 23-25% of an NVFP4 or AWQ checkpoint's. GGUF K-quant recipes usually quantize it (Q6_K in Q4_K_M: 0.51 GB).
- **Embeddings:** stored in full but only one row is read per token. They cost memory (fit), not bandwidth. Tied embeddings are the lm_head, read in full.
- **Attention projections** in some recipes, **MoE routers**, **norms** (tiny), **vision and audio towers** (not read in text decode, but loaded into memory).
- **MoE recipes that quantize only the experts** (gpt-oss ships MXFP4 experts with BF16 attention): at batch 1 only `top_k` of the experts are read, so the BF16 share of the bytes read is larger than the share of the bytes stored.
- **The KV cache** is never touched by weight quantization. Its share grows with context and with concurrency.

## GPU architecture and kernel paths (priors; verify by launching)

Sources (pages checked 2026-10-14; `latest` docs move, so re-read them for the pinned version): vLLM's per-method hardware table, https://docs.vllm.ai/en/latest/features/quantization/ ; TensorRT-LLM's precision and architecture matrix, https://nvidia.github.io/TensorRT-LLM/reference/precision.html ; llama.cpp backends, https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md .

| Format | Native kernels (low-precision math) | Typical elsewhere |
|---|---|---|
| FP8 W8A8 | GPUs with FP8 tensor cores: Ada (sm_89), Hopper (sm_90), Blackwell (sm_100, sm_120) | Ampere (sm_80, sm_86): weight-only FP8 (W8A16) in some engines, a separate row |
| NVFP4 W4A4 | Blackwell (sm_100, sm_120) | weight-only paths where an engine has them (separate row), otherwise not supported |
| MXFP4 | Blackwell | weight-only kernels on Hopper and Ampere in some engines |
| INT4 AWQ / GPTQ (W4A16) | weight-only by design; Marlin-style kernels need sm_80 or newer | older GPUs: slower kernels or none |
| GGUF | llama.cpp on CUDA, ROCm, Metal, Vulkan and CPU | other engines' GGUF loaders exist but are usually experimental: check their docs |

A fallback that changes the activation precision (W8A8 -> W8A16) is a different row. It is labelled with the path it ran, and it is never merged with the native row.

**AMD GPUs and CPUs:** the method is the same, but formats and kernels differ (e.g. FP8 variants on AMD Instinct GPUs, GGUF on CPUs). Check what `inferencekit doctor` and `inferencekit ceiling bw` support on that host; anything the harness cannot measure there is NOT RUN, never estimated from an NVIDIA result.

## Engines per format (priors; the scouts source them)

| Engine | Formats to look for | KV cache dtype knob (check the pinned version's `--help`) | Where to check (pages checked 2026-10-14) |
|---|---|---|---|
| vLLM | FP8, NVFP4 and INT4 (ModelOpt, compressed-tensors), MXFP4, AWQ, GPTQ, bitsandbytes, GGUF (the GGUF page calls it highly experimental) | `--kv-cache-dtype fp8`; weight offload `--cpu-offload-gb` | https://docs.vllm.ai/en/latest/features/quantization/ , .../quantization/quantized_kvcache.html , .../quantization/gguf.html , https://docs.vllm.ai/en/latest/configuration/engine_args.html |
| SGLang | FP8, NVFP4 (ModelOpt), MXFP4, AWQ, GPTQ, compressed-tensors | `--kv-cache-dtype fp8_e4m3` or `fp8_e5m2` | https://docs.sglang.ai/advanced_features/quantization.html , https://docs.sglang.ai/advanced_features/server_arguments.html |
| TensorRT-LLM (`trtllm-serve`) | FP8, NVFP4, INT4 AWQ (ModelOpt checkpoints) | the checkpoint's quantization config | https://nvidia.github.io/TensorRT-LLM/reference/precision.html , https://nvidia.github.io/TensorRT-LLM/features/quantization.html |
| llama.cpp (`llama-server`) | GGUF: Q8_0, K-quants, IQ-quants, MXFP4 | `--cache-type-k q8_0 --cache-type-v q8_0` (older builds needed flash attention on for a quantized V cache: check); offload `--n-gpu-layers`, `--n-cpu-moe`, `--override-tensor` | https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md |
| LMDeploy (TurboMind) | AWQ INT4, FP8 (version dependent) | its KV quantization policy option (`quant_policy`) | https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html |
| ExLlamaV2 / V3 (TabbyAPI) | their own EXL2 / EXL3 formats (variable bits per weight) | their cache-mode option | https://github.com/turboderp-org/exllamav3 , https://github.com/theroyallab/tabbyAPI |

These pages say what an engine supports in its latest docs, not in the version you pin, and not on your GPU. In the report, the scouts' sourced table (each claim with the URL and date it came from, for the pinned version) replaces this one.

bitsandbytes (NF4, int8) saves memory but its kernels are not built for serving speed: evaluate it for fit, rarely for speed.

## KV cache dtype is its own decision

- An FP8 KV cache halves the KV bytes: it matters at long contexts and under load, and barely at short single-user contexts (Qwen3-8B: 0.15 GB of KV at context 1024 vs 4.83 GB at 32k, BF16).
- Model it with `inferencekit roofline --kv-dtype fp8` (`q8_0` for llama.cpp's cache), and check quality with `quality niah` at the lengths you claim: KV quantization errors grow with context.
- Evaluate it as its own row (same weights, FP8 KV), so the report can say which change bought what.
