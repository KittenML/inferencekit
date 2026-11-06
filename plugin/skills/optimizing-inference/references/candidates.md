# Candidate engines

Search the web before selecting baselines for each run: use the exact model/architecture, modality, precision, workload and hardware. Check current official releases, support tables, repositories and vendor/model recipes; follow benchmark claims to their configurations and measurement conditions. Discover applicable engines beyond the examples below. There is no fixed engine list or count.

Choose the strongest applicable candidates within the approved scope and budget. Record source URLs/access dates and reasons for exclusions or untested candidates in the existing research note. Recheck freshness and compatibility before reusing a previous survey. If web access is unavailable, state that current coverage is unverified; do not claim the latest or best current baseline.

**Verify each candidate by launching it** on the pinned local checkpoint, at its precision, with a smoke request, then tune and measure it. Web rankings do not prove which engine wins locally. Keep versions, workloads and quality checks pinned during each comparison. Flags move between releases: **the installed version's `--help` is the authority**, and the startup log is the proof.

## Common procedure (every engine)

0. **Disk first** (`df -h`): an engine environment takes roughly 5–15 GB (inferred), containers more; delete a failed one's environment when space is short.
1. **One environment per engine** (`uv venv envs/<engine>` or its official container): engines pin different torch/CUDA versions.
2. **Newest release:** install without a version, then record it (`uv pip show <pkg>`, `<engine> --version`, `git rev-parse HEAD`). A model newer than the latest release allows one documented fix: a nightly or the main branch, recorded as such.
3. **Same weights:** one local directory at the resolved sha for every engine; a converted copy (e.g. GGUF) is another artifact (record its sha256 and conversion command).
4. **Launch** on its own port, logging to `logs/<label>.startup.log`; record the startup time (launch → ready).
5. **Ready check:** `GET /v1/models` (or `/health`) returns 200; allow up to ~20 min for compile and graph capture.
6. **Smoke request:** `/v1/chat/completions`, temperature 0, `max_tokens` 32, one short prompt: HTTP 200, coherent text, the expected `usage.completion_tokens`. Garbage (repeated tokens, wrong language) on a quantized checkpoint is a broken kernel path: **NOT SUPPORTED**, not slow.
7. **Resolved config** into `state.candidates.<engine>.resolved_config`, as the engine reports it: quantization/dtype, KV cache dtype, attention backend, CUDA graph mode and capture sizes, max sequences/batch, context length, prefix caching and speculative decoding on/off.
8. **Stop the server** by process group; check `nvidia-smi` (AMD: `amd-smi process`; CPU and Apple: `ps`) shows no compute process before the next engine. A container may outlive its killed `docker run` client: stop it by name.
9. **On failure:** keep the last ~40 log lines with the error (on a new GPU generation typically `no kernel image is available for execution on the device`, i.e. no build for your architecture, the RTX 5090 being compute capability 12.0; `unsupported quantization method`; an unknown architecture; OOM at load). Try **one** documented fix (a wheel for a newer CUDA, a nightly, a required flag or env var, a smaller `max_model_len` for OOM); still failing: **NOT SUPPORTED**, with version, command (relative paths) and error lines.

## LLM engines on NVIDIA

### vLLM
- **Install:** `uv pip install vllm`. Wheel variants per CUDA version exist; check the install docs for the version that matches the driver.
- **Launch:**
  ```bash
  vllm serve <model-dir> --served-model-name <name> --port 8001 --seed 0 \
    --max-model-len <bucket max + max_tokens + margin> --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching
  ```
  Never pass `--enforce-eager`: it turns CUDA graphs off.
- **Endpoint:** `/v1/chat/completions`, `/v1/completions`. Readiness: `/v1/models` or `/health`. Prometheus metrics: `/metrics`.
- **Resolved config:**
  - the startup line `... engine (v<version>) with config: model=..., dtype=..., quantization=..., enforce_eager=..., ...`;
  - the attention backend line (`Using <NAME> backend`);
  - CUDA graph capture lines (mode, sizes, time);
  - the KV cache size line (`GPU KV cache size: N tokens`).
- **Gotchas:**
  - Quantized checkpoints are detected from the checkpoint config; check the logged `quantization=`.
  - Sampling defaults come from the model's `generation_config.json` unless the request sets every sampler field. The harness forwards `workload.extra_body`; it does not set every sampler field.

### SGLang
- **Install:** `uv pip install "sglang[all]"` (check the docs for the CUDA-specific extra index).
- **Launch:**
  ```bash
  python -m sglang.launch_server --model-path <model-dir> --port 8002 --random-seed 0 --disable-radix-cache
  ```
  Never pass `--disable-cuda-graph`.
- **Endpoint:** OpenAI-compatible `/v1/...`. Server info: `/get_server_info` (the name varies by version) returns the resolved server args.
- **Resolved config:**
  - the `server_args=ServerArgs(...)` line at startup;
  - the CUDA graph capture lines (`Capture cuda graph begin/end`, batch sizes);
  - the attention backend in the server args.
- **Gotchas:** the radix cache (prefix caching) is on by default; turn it off. Quantized formats may need `--quantization <method>`; check the logged value.

### llama.cpp (`llama-server`)
- **Install:** build from source at a pinned commit:
  ```bash
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<cc, e.g. 120>
  cmake --build build -j
  ```
  Or use a release binary with CUDA. Record the build number and commit (`llama-server --version`).
- **Weights:** GGUF only. Convert with `convert_hf_to_gguf.py` (`--outtype bf16|q8_0|...`), then `llama-quantize` if needed.
  - **Precision check:** if no GGUF type matches the checkpoint's precision (for example an NVFP4 checkpoint), llama.cpp is **NOT SUPPORTED at equal precision**.
  - It is a separate track only if `precision_allowed` names the GGUF type.
- **Launch:**
  ```bash
  llama-server -m <model.gguf> -ngl 999 -fa on -c <ctx> -np 1 --port 8003 --seed 0
  ```
- **Endpoint:** `/v1/chat/completions` and `/v1/completions`. Introspection: `/props` (context, defaults) and `/slots`.
- **Resolved config:** startup lines `offloaded N/N layers to GPU`, `llama_context: n_ctx = ...`, `n_batch`, `n_ubatch`, `flash_attn`, `type_k`/`type_v`, and the CUDA device and compute capability.
- **Gotchas:**
  - The server reuses a slot's cached prompt prefix between requests by default (the `cache_prompt` request field; `--cache-reuse`). Disable it as the harness/gates require, and check `timings.prompt_n` in responses: if it is smaller than the prompt length, cache was hit.
  - `GGML_CUDA_DISABLE_GRAPHS` must be unset (CUDA graphs on).

### TensorRT-LLM (`trtllm-serve`)
- **Install:** `uv pip install tensorrt_llm` (strict torch/CUDA pins), or NVIDIA's TensorRT-LLM release container, which is the most reliable route if containers are available.
- **Launch:** recent versions use the PyTorch backend by default:
  ```bash
  trtllm-serve <model-dir> --port 8004 --max_batch_size 1 --max_num_tokens 8192 --max_seq_len <ctx> \
    --extra_llm_api_options extra.yaml
  ```
  `extra.yaml` holds e.g. `cuda_graph_config: {enable_padding: true, max_batch_size: 1}` and `kv_cache_config: {enable_block_reuse: false}`.
- **Endpoint:** `/v1/chat/completions`, `/v1/completions`, readiness `/health`.
- **Resolved config:** the startup log prints the resolved LLM arguments (backend, CUDA graph config, KV cache config, quantization). Keep the whole startup log.
- **Gotchas:**
  - Block reuse (prefix caching) may default to on; turn it off.
  - Support for a new GPU architecture or a new quantization format often lands in release candidates first. That is the "one documented fix".

### Additional examples, when applicable
- **LMDeploy:** `lmdeploy serve api_server <model-dir> --server-port 8005 --backend turbomind|pytorch`. The resolved config is logged at startup. KV quantization (`--quant-policy`) is a precision change.
- **ExLlamaV3 via TabbyAPI:** EXL3 weights are a different precision, so it is a separate track only.
- **A custom engine** (`candidates.include`) enters like the others: a reproducible install (git URL, commit, build commands) and the [custom-engine contract](#custom-engine-contract) below, so it can be gated.

Leave out wrappers around an engine already in the list. They add a layer, not a candidate.

### Custom-engine contract

A custom or patched engine serves all of this, or the harness cannot time, gate or quality-check it:
- OpenAI `/v1/chat/completions`, streamed, with usage (`stream_options.include_usage`);
- up to 20 `top_logprobs`, from the full logits even if the fast path prunes the lm_head. A slower full-logits path used only when logprobs are requested is fine. Why: the quality check scores these logprobs against the reference, and the gates' probes check them; an engine that returns none leaves trajectory quality NOT RUN. Pruning rules: [reward-hacks.md](../../benchmarking/references/reward-hacks.md) (an exact bound-pruned lm_head);
- exact length: `ignore_eos`, `min_tokens`;
- token ids: `return_token_ids` (the ids in the stream) and `return_tokens_as_token_ids` (logprob tokens as `token_id:<n>`). The harness picks the request fields it sends (its request profile) from the manifest `name` and the first launch words: a custom engine that honours vLLM's fields puts `vllm` in its name, e.g. "myengine (vLLM-compatible request fields)", and then gets these and the exact-length fields. A name and launch words that match no known engine get the generic profile, which sends neither, so the outputs stop at EOS and their lengths differ from the baseline's in your own `bench rounds`;
- the pinned chat template by default; `chat_template_kwargs` honoured;
- concurrent requests (queueing is fine; `quality gsm8k --concurrency 0` sizes itself to the server);
- seeded sampling (temperature, top_k, top_p; the model's `generation_config.json` defaults when omitted) from the full output layer.

**Build route: start from the [engine skeleton](../../optimizing-kernels/skeleton/README.md).** It serves this contract around correctness-first reference models; copy it and replace the reference forward with fast kernels, one proven change at a time.

## Other hardware (AMD ROCm, x86 CPU, Apple Silicon)

NVIDIA is fully supported; on other hardware the skills guide the flow and the CLI does less ([hardware.md](../../inference-triage/references/hardware.md)). The common procedure above holds. Recheck these examples and exclusions against current support:

| Platform | Candidate examples | Exclusions to recheck |
|---|---|---|
| AMD ROCm (MI300X, MI325X, MI350 series, Radeon) | vLLM (ROCm image, AITER on), SGLang (ROCm image), llama.cpp (HIP build), a custom engine | TensorRT-LLM |
| x86 CPU | llama.cpp `llama-server`, vLLM CPU, OpenVINO Model Server, a custom engine | TensorRT-LLM |
| Apple Silicon | mlx-lm `mlx_lm.server`, llama.cpp (Metal), a custom engine | TensorRT-LLM |
| Intel GPU, Google TPU | [other-accelerators.md](../../inference-triage/references/other-accelerators.md) | TensorRT-LLM |

- Record each exclusion with its source and hardware-specific reason. An engine that was not launched is untested, not NOT SUPPORTED. Research candidates outside this table with `engine-scout`, including the hardware in the query; a launch decides support.
- **Launch lines, the settings to prove and the pitfalls** per platform: [amd-rocm.md](../../inference-triage/references/amd-rocm.md), [cpu.md](../../inference-triage/references/cpu.md), [apple-silicon.md](../../inference-triage/references/apple-silicon.md).
- **Identity in every `bench rounds` call:** `--serve-args "--gpu-name <hardware> --gpu-count <n>"`.
- **Precision names hide formats:** FP8 on MI300X (FNUZ) is not NVIDIA's e4m3fn; MLX 8-bit is not Q8_0; an OpenVINO int8 export is not either of them. Rank only rows whose resolved formats match.

## TTS and STT (measurement coverage is LLM-first)

`bench serve` times OpenAI-compatible **text** endpoints. For audio:
- Candidate verification (stages 3–4) works the same way.
- Speed numbers need a modality client (time to first audio, real-time factor, underruns). A client you write is **not the harness**: its numbers are labelled "custom client" and their claims stay **UNPROVEN** until the harness covers them.
- Quality for TTS (ASR round-trip WER, MOS predictor) and STT (WER on public sets) runs with the tools named in the spec. The harness does not ship them yet: say so.

| Modality | Candidates to try |
|---|---|
| TTS | the model's reference PyTorch code (+ `torch.compile`); vLLM-Omni-style serving; SGLang-Omni-style serving; ONNX Runtime with the TensorRT execution provider; TensorRT-LLM / Triton recipes when the model has one |
| STT | faster-whisper (CTranslate2); whisper.cpp; vLLM (`/v1/audio/transcriptions`); TensorRT-LLM Whisper; NeMo for Parakeet/Canary models |

Pin the decoding settings before comparing STT engines (beam size, temperature fallback, VAD, conditioning on previous text): defaults differ across engines and change WER more than any kernel.

## Recording a result (state.candidates.<engine>)
```json
{"status": "NOT SUPPORTED", "version": "x.y.z", "install": ["uv pip install vllm==x.y.z"],
 "launch": ["vllm", "serve", "models/<dir>", "--port", "8001", "..."],
 "error": "ValueError: ... (startup log lines 212-230)", "attempts": 2, "evidence": [], "minutes": 18}
```
