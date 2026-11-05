# Multi-GPU: tensor, pipeline, data and expert parallelism

Use this page when `hardware.count` > 1. **With `hardware.count: 1`, multi-GPU is NOT RUN**: one verdict row, reason "hardware.count = 1". The replica plan ([capacity.md](capacity.md), section 7) still applies, with one GPU per replica.

Engine flags change between releases. The flags below were checked against each engine's documentation on 2026-10-14 (sources at the end). Check `--help` of the version you installed, and prove every setting from the resolved config (hard rule 4).

## Terms

- **TP (tensor parallelism):** every layer's weight matrices are split across k GPUs. Each GPU reads 1/k of the weights per token, and the GPUs exchange partial results (an all-reduce, about twice per layer) on every step.
- **PP (pipeline parallelism):** consecutive groups of layers live on different GPUs. One sequence passes through the stages one after another, so PP adds throughput (several batches in flight), not single-user speed.
- **DP (data parallelism), or replicas:** k full copies of the model, each serving its own requests. Per-user speed stays that of one GPU; capacity scales up to k×.
- **EP (expert parallelism):** for mixture-of-experts (MoE) models, the experts are spread across GPUs and tokens are routed to the GPU holding their expert.
- **NVLink:** a direct GPU-to-GPU link with hundreds of GB/s. **PCIe:** the host bus, roughly 32 GB/s per direction for Gen4 x16 and 64 GB/s for Gen5 (datasheet labels, inferred). Consumer RTX cards have no NVLink.

## When to use what

1. **The model fits on one GPU:** weights + KV cache for the target concurrency at the workload's context + about 5–8 GB for activations and graphs (derived; the activation figure is from [SGLang tuning]).
   - Do not shard it. "The best communication overhead is no communication overhead" [TRT-LLM sharding].
   - **Capacity lead:** run replicas, k copies behind one endpoint (DP = k). The SGLang guide prefers data parallelism for throughput when memory allows [SGLang tuning].
   - **Single-user lead:** TP = k can lower TPOT, because k GPUs read the weights in parallel, but only with fast links. Over PCIe, the all-reduce per layer often costs more than it saves for small models (inferred: about 2 × layers × the all-reduce latency per token). Measure TP = k against TP = 1; never assume.
2. **It does not fit on one GPU, but fits on the host:**
   - with NVLink: TP = k [TRT-LLM sharding; vLLM parallelism];
   - PCIe only: try PP (or llama.cpp's layer split) as well as TP. On GPUs without NVLink, vLLM recommends pipeline parallelism over tensor parallelism for throughput [vLLM parallelism]. TRT-LLM says to start with TP and check whether PP is better [TRT-LLM sharding];
   - a GPU count the model does not divide evenly: PP, which allows uneven splits [vLLM parallelism].
3. **MoE models:** EP for the expert layers, with TP or DP for attention [vLLM data parallel; SGLang server args].
4. **Not supported yet (NOT RUN, say so):** more than one host (TP inside a node, PP across nodes), prefill/decode disaggregation (pays off only at scale, when TTFT and TPOT are both tight), and several servers behind an external router.

## How to launch each engine on k GPUs

| | vLLM `vllm serve` | SGLang `sglang.launch_server` | TRT-LLM `trtllm-serve` | llama.cpp `llama-server` |
|---|---|---|---|---|
| TP | `--tensor-parallel-size k` | `--tp-size k` (alias of `--tensor-parallel-size`) | `--tp_size k` | `-sm row` (or `-sm tensor`, experimental) |
| PP | `--pipeline-parallel-size k` | `--pp-size k` | `--pp_size k` | `-sm layer` (the default) with `-ts` proportions |
| DP, one endpoint | `--data-parallel-size k`; `--max-num-seqs` is **per rank** | `--dp-size k`, with `--load-balance-method` | separate servers (not supported yet) | separate servers (not supported yet) |
| EP (MoE) | `--enable-expert-parallel` (with TP or DP) | `--ep-size k`; DP attention is `--attn-dp-size k` (it replaces `--enable-dp-attention`) | `--ep_size k` | — |
| Which GPUs | `CUDA_VISIBLE_DEVICES` | `CUDA_VISIBLE_DEVICES` | `CUDA_VISIBLE_DEVICES` | `-dev`, or `-mg` with `-sm none` |

- **Prove it in effect:**
  - the startup log's resolved config shows the parallel sizes and the world size;
  - `nvidia-smi` shows the model's memory on every GPU it should use, and on no other;
  - the startup log says which all-reduce path the engine chose (for example, a warning that a custom all-reduce is disabled). Record it.
- **Record the topology once:** `nvidia-smi topo -m` (`NV#` = NVLink; `PIX`, `PXB`, `PHB`, `NODE`, `SYS` = PCIe paths, `SYS` crossing CPU sockets). It goes into `state`, the hardware decision record and `REPORT.md`.
- **Manifest:** the parallel flags go in the engine's `launch` argv, and a GPU selection goes in its `env`. The manifest's `hardware.count` is the host's GPU count.

## What changes in measurement

- **Per-GPU normalisation.** In load mode, pass `--gpu-count <GPUs behind the server>` and `--gpu-price-usd-h <price per GPU-hour>` (= `hardware.price_usd_h` ÷ `hardware.count`) inside `--serve-args`. `bench serve` divides throughput by the GPU count, and its cost formula uses the per-GPU price.
- **Every capacity entry on a k-GPU host uses all k GPUs** as one endpoint (TP = k, PP = k, DP = k, or a mix). A re-run passes `--gpu-count` = `hardware.count` for every entry, so a one-GPU capacity entry on a k-GPU manifest would be normalised wrongly. Measure one-GPU capacity on a one-GPU run, or report it as agent-measured only.
- **Cost check:** a `$/1M` worked out from the whole-host `price_usd_h` as if it were the per-GPU price is count× too high when count > 1. Recompute it from tok/s/GPU and the per-GPU price, and say so.
- **GPU count is part of a row's identity.** Single-user rows on different GPU counts go in separate blocks of the ranked table, with a GPUs column. Never compare their tok/s without saying how many GPUs produced it.
- **N\* per GPU** = N\* ÷ GPUs per replica. Goodput and $/1M are per replica and per GPU-hour; show both.
- **The bandwidth ceiling.**
  - Measure each GPU, one at a time: `CUDA_VISIBLE_DEVICES=<i> inferencekit ceiling bw`.
  - **TP = k:** ceiling tok/s = (sum of the k measured bandwidths) ÷ bytes per token. This holds **only if TP actually splits the weights**: check that each GPU's weight memory in the startup log is about 1/k of the checkpoint. KV bytes split only when the KV heads ≥ k; otherwise the KV cache is replicated, so count it k times. Pass the summed GB/s as a number to `inferencekit roofline --gpu-bw <GB/s>`, and cite every `ceiling bw` record. The ceiling ignores all-reduce time, so TP's % of ceiling is lower than TP = 1's. That gap is the communication cost: report it.
  - **PP and DP:** a single sequence's ceiling is one GPU's (PP runs the stages one after another). The aggregate is up to k× at a batch that fills every stage or replica.
  - **EP:** derive the bytes per token per GPU from the experts each GPU holds and the routing (inferred), and label it a modelling assumption.
- **Interconnect:** report NVLink or PCIe next to every TP or EP row. A TP row on PCIe that loses to TP = 1 is a finding, not a failure.
- **GPU hygiene:** while timing, nothing else runs on any GPU of the host (`doctor` and remote-host.md, "GPU hygiene"). With configs that use several GPUs, screening never runs one engine per GPU in parallel.

## Capacity planning with replicas

- A replica is one server endpoint: one GPU, or one TP/PP group.
- Replicas = ceil(peak RPS ÷ goodput per replica at N\*), and GPUs = replicas × GPUs per replica ([capacity.md](capacity.md), section 7).
- Compare plans at equal SLO. For example, one TP = 2 replica against two one-GPU replicas: pick the plan with the fewer GPU-hours per million requests. Say which plan the numbers support, and which one is inferred.
- Production usually adds a spare replica for failover and headroom for bursts. Say whether the plan includes them.

## Sources (read 2026-10-14)

- [vLLM parallelism] https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html
- [vLLM data parallel] https://docs.vllm.ai/en/latest/serving/data_parallel_deployment.html
- [SGLang server args] https://docs.sglang.io/advanced_features/server_arguments.html
- [SGLang tuning] https://docs.sglang.io/advanced_features/hyperparameter_tuning.html
- [TRT-LLM serve] https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html
- [TRT-LLM sharding] https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/deciding-model-sharding-strategy.html
- [llama.cpp server] https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
