# Tuning: bounded sweeps, every knob proven

## Procedure (per supported engine, inside its share of the stage 4 box)

1. **Baseline recipe.** Use the engine's documented low-latency configuration for this model class and GPU, with prefix caching off and CUDA graphs on. Measure it, label it `<engine>-recipe` (manifest role `baseline-recipe`), and keep it. If quoted numbers for this engine beat the recipe (e.g. a lower TTFT), find and measure their config first.
2. **Candidate list.** Pick up to ~12 configs from the knob table below, most promising first, at most 2 knobs changed per config. Order them by the lever ladder ([stop-point.md](stop-point.md)): overhead first, then batching and scheduler (capacity), then precision only within the intake. A vendor autotuner may propose some of them ("Vendor autotuners", below).
3. **Screening runs:** two buckets, `short` and `long` (or the intake's target buckets), one round, the same prompts and `max_tokens`: `bench serve --buckets short --buckets long --rounds 1 --engine-kind <kind>` ([harness-commands.md](harness-commands.md); never the engine's own benchmark). Why two: a knob that cuts launch overhead can help short prompts and hurt long ones (prefill, KV), and one bucket hides that. A single-round point estimate only **ranks** configs to pick the top 3: never keep or drop, never a reported result.
4. **Successive halving:** the top 3 by the lead metric plus the recipe, each its own manifest entry (label, port, launch line), go to the full protocol (all buckets, interleaved rounds; `<R>`, its floor and fitting the stage box: [harness-commands.md](harness-commands.md), section 2): one `bench rounds --engines <recipe>,<c1>,<c2>,<c3>` over pairwise calls, dropping candidates before rounds.
   - **Also promote** a screened config that beats the pre-registered effect `e` on another metric the person cares about (the intake's SLO or goals, e.g. TTFT or TPOT) in either bucket, without screening worse than the recipe on the lead metric. It joins the same call. If `bench rounds` shows that metric `improved` and the lead metric not `regressed`, report it as an option for that metric: it does not replace the headline config, and the `REPORT.md` summary lists it on its own line ([stop-point.md](stop-point.md), "Summary").
5. **Keep or drop, per knob.** Read each `bench rounds` call with the rule in [harness-commands.md](harness-commands.md), "Reading the result" and "Keep or drop" (the same rule `compare` applies):
   - `improved` (CI entirely above 1 **and** beyond the pre-registered effect) for the candidate over the incumbent, and `regressed` in no target bucket: keep it.
   - `regressed`: drop it.
   - `within_noise` (CI inside ± the effect **and** narrow): drop it and stop exploring that knob.
   - `inconclusive` (anything else, usually a CI too wide after a few rounds of a large model): run more rounds in a new call if the budget allows; otherwise record it as UNPROVEN and do not keep it. Keep exploring the knob only if the box allows.
   - `inferencekit compare` accepts matching decode, single-user serve, clip, image or audio records. Use `bench rounds`' own paired results for its aggregate; load-mode and round-summary records are not supported.
6. **Every config:** relaunch, prove each changed knob in effect, measure, append to `experiments.jsonl`. A config whose knob is not in effect gets `in_effect: false, kept: false`, whatever its number.
7. **Budget.** Run `inferencekit intake budget runs/<run-id>` before each relaunch. On `stage_over`, finish the current measurement, record the remaining configs as NOT RUN, and move on.
8. **End of stage 4: clean the manifest.** Remove every dropped or unproven candidate entry from `manifest.json` (it stays in `experiments.jsonl`), so the manifest holds only what you will report: each engine's recipe, its kept tuned config (if any), any option kept for another metric (step 4), and its capacity entry (load mode). Run `manifest validate`, then copy it to the host again ([remote-host.md](remote-host.md)). A re-run runs every entry that is left; stage 5 passes `--engines` (single-user entries in single mode, capacity entries in load mode).

Startup and compile time is recorded separately and is not part of the speed numbers. It does count against the budget.

## How to prove a knob is in effect

| Source | What it shows |
|---|---|
| Startup log | The engine's resolved config line (`with config:` in vLLM, `server_args=` in SGLang, `llama_context:` lines in llama.cpp, the LLM args in TRT-LLM), the attention backend line, and the CUDA graph capture lines with sizes. |
| Server-info endpoints | SGLang `/get_server_info`, llama.cpp `/props` and `/slots`, vLLM `/v1/models` (`max_model_len`). |
| `/metrics` (Prometheus) | Prefix-cache hits (must stay 0), speculative decoding drafted/accepted tokens, running/waiting requests (concurrency actually reached). |
| Behaviour | `timings.prompt_n` (llama.cpp); the harness's caching and length gates. |

Quote the exact log line in `experiments.jsonl` (`resolved.source`). A flag in your launch command proves nothing: engines silently ignore unknown or overridden settings, environment variables included.

## Knobs per engine

Precision column: **P** means the knob changes numerics or precision. It is allowed only with `fidelity: lossy` and within `precision_allowed` / `kv_cache_dtype_allowed` (or by a task's precision rule: [fidelity.md](fidelity.md)), and it must pass the quality gates.

### vLLM
| Knob | Flag (check `--help`) | P | Proof |
|---|---|---|---|
| CUDA graph / compile mode | `-O<level>`, `--compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'` | | config line `compilation_config`; capture log lines |
| Capture sizes | `--compilation-config '{"cudagraph_capture_sizes": [1,2,4,...]}'` or a max capture size | | capture log: batch 1 (and every load-mode batch) captured |
| Compile sizes | `-cc '{"compile_sizes": [1]}'` (`--compilation-config`; one JSON for all keys): compiles batch 1, for single-user decode | | config line `compile_sizes`; longer startup |
| Attention backend | `--attention-backend` (newer) or an env var (older; may be ignored) | (P if a lower-precision kernel) | `Using <NAME> backend` line |
| Max sequences / batched tokens | `--max-num-seqs`, `--max-num-batched-tokens` | | config line |
| Chunked prefill | `--enable-chunked-prefill` / its default | | config line |
| Async scheduling | `--async-scheduling` (if present) | | config line |
| KV cache dtype | `--kv-cache-dtype fp8` | **P** | config line `kv_cache_dtype` |
| Memory | `--gpu-memory-utilization`, `--max-model-len` | | KV cache size line |
| Speculative / MTP | `--speculative-config '{...}'` | exactness gate | spec-decode metrics in `/metrics` (acceptance) |
| Prefix caching | `--no-enable-prefix-caching` (**required off**) | | config line; prefix-cache hits = 0 |

### SGLang
| Knob | Flag | P | Proof |
|---|---|---|---|
| CUDA graph max batch | `--cuda-graph-max-bs` (and `--cuda-graph-bs` list) | | capture log lines |
| torch.compile | `--enable-torch-compile` | | server args; longer startup |
| Attention backend | `--attention-backend` | (P) | server args |
| Memory | `--mem-fraction-static` | | server args; KV tokens line |
| Max running requests | `--max-running-requests` | | server args |
| Chunked prefill | `--chunked-prefill-size` | | server args |
| Scheduling | `--schedule-conservativeness`, `--schedule-policy` | | server args |
| KV cache dtype | `--kv-cache-dtype` | **P** | server args |
| Speculative / MTP | `--speculative-algorithm ...` + draft settings | exactness gate | server args; acceptance in logs/metrics |
| Radix cache | `--disable-radix-cache` (**required**) | | server args |

### llama.cpp (`llama-server`)
| Knob | Flag | P | Proof |
|---|---|---|---|
| Flash attention | `-fa on` | | `flash_attn = 1` |
| All layers on GPU | `-ngl 999` | | `offloaded N/N layers` |
| Batch / micro-batch | `-b`, `-ub` | | `n_batch`, `n_ubatch` lines |
| Parallel slots | `-np` (capacity) | | `n_seq_max` / `/slots` |
| KV cache type | `--cache-type-k`, `--cache-type-v` | **P** | `type_k`, `type_v` |
| Threads | `-t`, `-tb` | | startup `system_info` |
| CUDA graphs | on by default; `GGML_CUDA_DISABLE_GRAPHS` unset | | no "disabling CUDA graphs" lines in the log |
| Speculative | `-md <draft.gguf>` + draft settings | exactness gate | server log acceptance stats |
| Prompt cache | off for benchmarks | | `timings.prompt_n` = prompt length |

### TensorRT-LLM (`trtllm-serve`)
| Knob | Where | P | Proof |
|---|---|---|---|
| CUDA graphs | `cuda_graph_config: {enable_padding: true, max_batch_size: N}` | | resolved LLM args; warmup capture lines |
| Batch / tokens | `--max_batch_size`, `--max_num_tokens` | | resolved args |
| KV memory | `--kv_cache_free_gpu_memory_fraction` | | resolved args |
| KV dtype | `kv_cache_config: {dtype: fp8}` | **P** | resolved args |
| Block reuse | `kv_cache_config: {enable_block_reuse: false}` (**required off**) | | resolved args |
| Chunked prefill | `enable_chunked_prefill: true` | | resolved args |
| Attention backend | `attn_backend` | (P) | resolved args |
| torch.compile | `torch_compile_config` | | resolved args |
| Speculative / MTP | `speculative_config` | exactness gate | acceptance in logs |

## Equal-precision rule

These count as precision:
- weight format;
- activation format;
- KV cache dtype;
- lm_head and embedding precision;
- MoE expert precision;
- any "fast math" kernel that quantizes on the fly.

Two rows are comparable only if all of these match, or the intake allows the difference. A precision change is a separate track: it is never ranked against the equal-precision rows, and it needs its own quality gate result.

## Capacity goal (load mode)

Tune **separately** from the single-user config. Single-user wants small graphs and low latency. Capacity wants batch size, KV memory and scheduling tuned for throughput inside the SLO.

| Engine | Knobs for N* |
|---|---|
| vLLM | `--max-num-seqs`, `--max-num-batched-tokens`, `--gpu-memory-utilization`, graph capture sizes ≥ max-num-seqs |
| SGLang | `--max-running-requests`, `--mem-fraction-static`, `--chunked-prefill-size`, `--schedule-conservativeness` |
| llama.cpp | `-np`, `-c` (total context shared by the slots), `-b`/`-ub` |
| TRT-LLM | `--max_batch_size`, `--max_num_tokens`, `--kv_cache_free_gpu_memory_fraction`, `cuda_graph_config.max_batch_size` |

- Sweep with `bench rounds --mode load` on the `dev-load` set, with `workload.concurrency` as repeated `--concurrency` flags inside `--serve-args` ([harness-commands.md](harness-commands.md)). A one-off `bench serve --mode load` against a running server is a screening run only.
- N* is the highest level at which the SLO holds at that level and every lower one.
- **As experienced engineers do** ([capacity.md](capacity.md)): at least 100 requests per level (`--duration-s`, `--max-requests-per-level`, one shared `--ref-e2e-ms`); the grid extended until the SLO fails; the batch cap checked against the measured in-flight count, not N; a Pareto frontier across every capacity config.
- **Multi-GPU hosts** (`hardware.count` > 1): add the parallel configs from [parallelism.md](parallelism.md) as candidates, each using all the host's GPUs.
- Each engine's best serving config goes into the manifest as its own engine entry (e.g. label `vllm-capacity`).

## Vendor autotuners: candidate generators only (stage 4)

Inside the stage-4 time box (their run time counts), an autotuner may **propose** configs. Its numbers are **screening only** (never ranked, kept or dropped on, or reported): its client, timer, prompts and cache settings differ from the harness (hard rules 1 and 2).

| Tool | What it does | Notes |
|---|---|---|
| vLLM `auto_tune.sh` (in the vLLM repository, under benchmarks/auto_tune) | Tries every `max-num-seqs` × `max-num-batched-tokens` pair from two lists, at a P99 end-to-end latency limit (`MAX_LATENCY_ALLOWED_MS`), and lowers the request rate until the limit holds. Finds the highest `gpu-memory-utilization` that does not OOM, starting at 0.98 [vLLM auto_tune]. | Set **`MIN_CACHE_HIT_PCT=0`**, and set `INPUT_LEN` / `OUTPUT_LEN` from the workload. Its path must not contain "vllm" (it runs `pkill -f vllm`). |
| `vllm bench sweep serve`, then `vllm bench sweep plot_pareto` | Runs a grid of `vllm serve` × `vllm bench serve` parameters (`--serve-params`, `--bench-params` JSON files) and plots tok/s/GPU against tok/s/user [vLLM sweeps]. | Put `--no-enable-prefix-caching` in its `--serve-cmd`. |
| SGLang tuning | The hyperparameter guide: watch `#queue-req` and `token usage` in the decode log, then tune `--schedule-conservativeness`, `--mem-fraction-static`, `--chunked-prefill-size` and `--max-running-requests` [SGLang tuning]. For MoE, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py --tune` writes kernel config files. | Tuned MoE kernel configs are files copied into the engine's package: that is an engine patch, so it needs a reproducible manifest install step ([manifest.md](manifest.md)). |
| NVIDIA AIConfigurator, `aiconfigurator cli default --model <id> --total-gpus <k> --system <gpu> --ttft <ms> --tpot <ms>` | **Models** (does not run) TP/PP, batch size and aggregated vs disaggregated serving for TRT-LLM (vLLM and SGLang in evaluation), from per-GPU databases (h100_sxm, h200_sxm, b200_sxm, gb200, a100_sxm) [AIConfigurator]. | Its output is **inferred**: a starting point, never a number. Other GPUs are rough estimates only. |
| BentoML `llm-optimizer` | Runs real sweeps of SGLang or vLLM server and client arguments, with constraints such as `ttft:p95<500ms`. `llm-optimizer estimate` models without running [llm-optimizer]. | `estimate` is **inferred**. Keep prefix caching off in its server args. |

Rules:
1. **Prefix caching off:** `MIN_CACHE_HIT_PCT=0` for auto_tune, the engine's cache-off flag in every server command it launches.
2. **The intake's workload:** input lengths (bucket), `max_tokens`, SLO.
3. **Every proposal takes the normal path:** manifest entry, knob proven in effect, `bench serve` screening, `bench rounds`.
4. **Re-prove a `gpu-memory-utilization` it picked** (auto_tune pushes it to 0.98): a config that fails to start or OOMs in any fresh round, or on `long`, is dropped.
5. **Log** tool, version, command and each proposal in `experiments.jsonl` (`"value": null`, `"note": "proposed by <tool>"`).
6. **It is timed GPU work:** nothing else runs on the host meanwhile.

Sources (read 2026-10-14): [vLLM auto_tune] https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune/README.md · [vLLM sweeps] https://docs.vllm.ai/en/latest/benchmarking/sweeps/ · [SGLang tuning] https://docs.sglang.io/advanced_features/hyperparameter_tuning.html and https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton · [AIConfigurator] https://github.com/ai-dynamo/aiconfigurator · [llm-optimizer] https://github.com/bentoml/llm-optimizer
