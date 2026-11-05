# Engine configurations: best documented config, knobs, and proof they took effect

"Best documented config" means the configuration the engine's own documentation recommends for the goal being measured (low-latency single-user, or high-capacity serving), on this model and GPU, at the precision every engine in the comparison shares. Cite the doc page and engine version you followed. Flags change between releases: treat the flags below as things to check, and the engine's `--help` for the version you pinned as the authority.

## Common to every engine

- **Pin the model files.** Download once at a fixed commit and point every engine at the same local path:
  `hf download <org>/<model> --revision <commit-sha> --local-dir ./models/<model>`
  Why: every engine then reads byte-identical weights and config.
- **Fixed output length and sampling.** Every sampler setting explicit with a fixed seed, identical for every engine (the workload's sampling, else `generation_config.json` defaults; quality checks greedy), `ignore_eos` where supported. Why: otherwise engines apply different defaults, and the decode steps and sampler work differ.
- **Prefix caching off (or reset between rounds) for single-user TTFT; unique prompts under load.** Why: a cached prefix skips prefill and makes TTFT look tiny.
- **Keep the startup log and the resolved config of every run.** Why: they state the dtype, KV cache dtype, attention backend, graph mode and scheduler limits actually used.
- **Write the launch line, environment and port into the run manifest** (`engines[].launch`, `env`, `port`, `ready_check`, `resolved_config`). Why: anyone who re-runs the manifest starts each engine from exactly that line.

## Precision table: compare only within a row

| Format | Approx. bytes per weight | Notes | Engines |
|---|---|---|---|
| BF16 | 2.0 | the common denominator | vLLM, SGLang, TRT-LLM, llama.cpp (BF16 GGUF) |
| FP8 E4M3, 128x128 block scales | 1.0 + ~0.0001 (BF16 scales) or ~0.0002 (FP32 scales) | DeepSeek-style fine-grained FP8, used by `Qwen/Qwen3-8B-FP8`; the lm_head is often left in BF16 | vLLM, SGLang |
| Q8_0 (GGUF) | 1.0625 (32 int8 + one FP16 scale per 32 weights) | integer blocks, not FP8 | llama.cpp |

The KV cache dtype is part of precision: FP8 KV halves the bytes attention reads. Record it for every engine. An early-development lesson: vLLM 0.31's default FP8 path on an RTX 5090 (DeepGEMM with E8M0 scales) re-quantises the checkpoint's block scales to powers of two at load time (inferred from vLLM's code), so "same checkpoint" did not mean "same arithmetic". Record the GEMM kernel each engine logs it selected.

## Prove every knob took effect

For each setting the comparison relies on, record four things: requested value, where it was set, resolved value, and the evidence line. A setting is "in effect" only when the resolved value matches.

| Engine | Where the resolved settings are | Early lessons |
|---|---|---|
| vLLM | the `non-default args` startup line; the engine's resolved config object; the backend and GEMM-kernel selection lines; CUDA graph capture lines and the compilation config (graph mode, capture sizes) | releases move settings between environment variables, engine arguments and CLI flags; a channel the installed version no longer reads is dropped with at most a warning, so check the resolved value, not the request. `disable_log_stats=True` appears in offline `LLM` runs without being passed |
| SGLang | the resolved server arguments (`server_args`; recent releases hold them in a `msgspec.Struct` with `resolved_dict()`); "capture cuda graph" log lines | an older reader returned an empty dict for the `msgspec.Struct`; refuse to record an empty configuration rather than record "nothing set" |
| llama.cpp | `llama-bench` JSON fields and the server log (offloaded layers, backend, flash-attention status); build flags from `CMakeCache.txt` (`GGML_CUDA_GRAPHS`, `GGML_CUDA_FA`) | requested flags are not proof; record llama-bench's own report and the build flags |
| TRT-LLM | the resolved LLM/serve config the server logs at startup | check `max_batch_size`, `max_num_tokens`, CUDA graph padding |
| Custom engine | whatever it logs about itself: graphs on/off, kernels selected, dtypes | if it logs nothing, add that first; the record is only as good as what the engine reports |

Read every run's log for warnings, not only failures: the one line saying a setting was ignored is not in the JSON rows. `bench rounds` keeps each cell's server log (`logs/<label>.r<k>.log`) and its parse in `resolved_config/<label>.r<k>.json` (RESOLVED / PARTIAL / UNPROVEN, warnings, unknown settings); a custom engine can print one line `{"inferencekit_resolved_config": {...}}` with its own settings.

## Single-user (low-latency) configuration

### vLLM

```bash
vllm serve ./models/<model> --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching --seed 0 --port 8000
```
- **CUDA graphs on.** `--enforce-eager` turns them off; do not pass it. Record the CUDA graph mode (piecewise vs full) and that the capture sizes include 1. Why: in piecewise mode attention runs outside the graph, which costs launches at batch 1; an uncaptured batch size runs eagerly.
- **Attention backend:** set with the engine argument/flag your pinned version documents and confirm the resolved backend from its log before timing. A backend switch can be worth a measurable share at batch 1; measure it with an interleaved A/B and a quality check, never assume it.
- **Speculative decoding off** in an engine ranking unless every engine uses the same lossless method (the exactness gate checks it). A speedup with it is against a baseline the task or person fixed, else the faster of each baseline's rows without and with its best documented exact speculation ([speculative.md](../../optimizing-inference/references/speculative.md)).

### SGLang

```bash
python -m sglang.launch_server --model-path ./models/<model> --disable-radix-cache --random-seed 0 --port 8000
```
- **CUDA graphs on** (`--disable-cuda-graph` turns them off); `--cuda-graph-max-bs` sets the largest captured batch.
- **`--enable-torch-compile`** is documented as helping small batches and lengthens startup; early development did not try it (its published SGLang row says "default settings, not tuned"). If you use it, warm up until compilation finishes.
- **Attention backend** (`--attention-backend`): record the resolved one.

### llama.cpp

```bash
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j
./build/bin/llama-server -m ./models/<model>.gguf -ngl 99 -fa on --port 8000
```
- **All layers on the GPU** (`-ngl 99`) and **flash attention on**; the CUDA backend uses graphs for single-token decode unless `GGML_CUDA_DISABLE_GRAPHS=1` is set.
- **`llama-bench` depth:** its generation test starts from an empty context unless a depth is set (`-d`). Decode at depth 0 reads no KV cache and overstates speed versus a real prompt.
- **Precision:** use a BF16 GGUF to compare with BF16 runs; Q8_0 is reported separately against its own ceiling (`inferencekit roofline --dtype q8_0`).

## Capacity (serving) configuration: knobs to sweep

Tune every engine for capacity separately, inside its GPU budget, starting from its documented defaults; stop tuning a knob once its gain is `within_noise`. Candidate knobs (check names against your pinned version):
- **vLLM:** `--max-num-seqs`, `--max-num-batched-tokens`, `--gpu-memory-utilization`, the CUDA graph mode with a capture size at least `max_num_seqs`, the attention backend, speculative decoding (only lossless, and gated).
- **SGLang:** `--mem-fraction-static`, `--max-running-requests`, `--chunked-prefill-size`, `--schedule-conservativeness`.
- **TRT-LLM (`trtllm-serve`):** `max_batch_size`, `max_num_tokens`, CUDA graph padding.
- **llama.cpp (`llama-server`):** `-np` (parallel slots), `-b` / `-ub`, flash attention.

A bounded sweep: short runs of many configs on one bucket at the target concurrency, keep the top few, then run the full protocol on them. Record startup and compile time separately from measured time.

## Passing settings to `bench decode` (engine-timed, no server)

`inferencekit bench decode` drives vLLM's `LLM(...)` and SGLang's `Engine(...)` in-process, or `llama-bench`, so server flags map to `--engine-args` JSON keys (check `inferencekit bench decode --help`):

| Server flag | `--engine-args` key |
|---|---|
| vLLM `--enforce-eager` / `--no-enable-prefix-caching` / `--max-model-len` / `--gpu-memory-utilization` | `enforce_eager` / `enable_prefix_caching` / `max_model_len` / `gpu_memory_utilization` |
| SGLang `--disable-cuda-graph` / `--disable-radix-cache` / `--enable-torch-compile` / `--mem-fraction-static` | `disable_cuda_graph` / `disable_radix_cache` / `enable_torch_compile` / `mem_fraction_static` |
| llama-bench `-ngl 99 -fa 1` | `{"ngl": 99, "fa": 1}` |

The record's `identity.engine.flags` shows what the engine reported back, which is what counts.
