# Bytes per token, predicted ceilings and fit

At batch 1, decode is memory-bound: every step reads the weights it uses once, plus the KV cache of the context so far. So the fastest a format can possibly decode is

```
ceiling tok/s           = measured DRAM bandwidth / bytes read per token            (derived)
predicted ceiling ratio = reference bytes per token / candidate bytes per token     (derived; the bandwidth cancels)
```

The predicted ratio needs no GPU, only the checkpoints' headers. It is an upper bound on the measured speed-up, never a substitute for it.

## 1. Bytes per token for each candidate

**safetensors checkpoints** (BF16, FP8, NVFP4, MXFP4, AWQ, GPTQ, compressed-tensors):

```bash
inferencekit roofline --model <repo> --revision <sha> --method headers --context <ctx> --kv-dtype bf16 --gpu-bw <ev_bw or GB/s>
```

- `--method headers` sums the stored bytes of every tensor from the safetensors headers (a few MB of range requests, no weights downloaded), grouped by role. Packed FP4 values, FP8 values and every scale tensor count exactly as stored.
- `--context`: each bucket's prompt length + about half of `max_tokens`. Run it once per bucket you report.
- `--kv-dtype`: the KV dtype the row will run (`bf16`, `fp8`, `q8_0`). Use the dtype the engine **logged** when you redo it after the launch.
- `--gpu-bw`: the `ceiling bw` evidence id when you have one, else a number labelled datasheet-based. The predicted ratio does not depend on it.

Fields to read with `inferencekit evidence show <id> --path <dotted path>`. The record's `metrics` is a list (`--path metrics` prints all of them: `bytes_per_token`, `weight_bytes`, `kv_bytes`, `state_bytes`, `bytes_<group>`), so a metric name alone is not a path; the same values are plain keys under `data.counts`:

| Path | Meaning |
|---|---|
| `data.counts.bytes_per_token`, `data.counts.weight_bytes`, `data.counts.kv_bytes`, `data.counts.state_bytes` | the total read per token at this context and its terms (`state_bytes` only on hybrids) |
| `data.counts.lm_head_bytes`; `data.counts.groups` | lm_head bytes read per token; read and stored bytes per role (`attention`, `mlp_dense`, `moe_experts`, `lm_head`, ...) |
| `data.counts.read_bytes_by_format` | read bytes per format (`bf16`, `fp8`, `nvfp4`, `mxfp4`, `i32` for packed int4) |
| `data.counts.checkpoint.stored_bytes` | everything stored at the pinned commit: the weight memory to fit (all experts, full embedding table, towers) |
| `data.counts.kv_bytes` | the KV cache of one sequence at this context, sliding-window layers capped at their window: the fit's KV term |
| `data.counts.kv.layers`, `data.counts.kv.bytes_per_element` | per layer group: kind (`full` / `sliding`), elements per token, tokens held, layer list |
| `data.counts.state` | recurrent state of linear-attention layers (hybrids): `layers`, `per_layer.recurrent_bytes`, `per_layer.conv_bytes`, and `bytes` (= `state_bytes`, read **and** written per token, so twice the state held) |
| `data.counts.moe` | `num_experts`, `top_k`, `experts_stored_bytes`, `experts_read_bytes` (batch 1) |
| `data.counts.kv_bytes_per_context_token` | the local slope d(KV)/d(context) **at this context**: sliding-window layers past their window add 0 and recurrent state is not in it. Not a fit input (section 4) |
| `identity.model.dtype`, `identity.model.quant` | format label and algorithm |
| `limitations` | e.g. "the checkpoint's quant config declares an fp8 KV cache but KV was counted as bf16" |

**GGUF files** (llama.cpp): `roofline` cannot read them. Use this skill's header reader, which range-reads only the GGUF header:

```bash
python3 scripts/gguf_bytes_per_token.py --repo <repo> --file <name.gguf> --revision <sha>     # or local paths
```

- It prints `weight_bytes_per_token` (every tensor once; one embedding row; MoE routed experts x `expert_used_count / expert_count`), `stored_bytes`, `read_bytes_by_type`, the lm_head type, `linear_bits_per_weight` and the commit it read. Split models: pass every shard (`--file` once per shard).
- It does not compute the KV cache. KV bytes do not depend on the weight format: add `kv_bytes` (and `state_bytes` on a hybrid) from the reference's `roofline` record at the same context and KV dtype.
- It exits 2 instead of guessing (an unknown tensor type, missing expert counts, a file that is not GGUF). Fix the input; never patch the numbers.

## 2. Worked example: Qwen3-8B, four formats, real headers

All values **derived** from the checkpoints' headers (`roofline --method headers`, and `gguf_bytes_per_token.py` for the GGUF rows, KV from the BF16 record), at a BF16 KV cache. KV is 147,456 bytes per context token (36 layers x 8 KV heads x 128 dims x K and V x 2 bytes): 0.15 GB at context 1024, 4.83 GB at 32k.

| Checkpoint @ revision | Stored GB | Weights read per token GB | lm_head GB (share at ctx 1k) | Bytes/token ctx 1k | Ratio vs BF16 | Bytes/token ctx 32k | Ratio |
|---|---|---|---|---|---|---|---|
| `Qwen/Qwen3-8B` @ b968826d | 16.38 | 15.14 | 1.25 BF16 (8%) | 15.29 | 1.00 | 19.97 | 1.00 |
| `Qwen/Qwen3-8B-FP8` @ 220b46e3 | 9.44 | 8.19 | 1.25 BF16 (15%) | 8.34 | 1.83 | 13.02 | 1.53 |
| `nvidia/Qwen3-8B-NVFP4` @ ccd10a89 | 6.40 | 5.15 | 1.25 BF16 (23%) | 5.30 | 2.88 | 9.98 | 2.00 |
| `Qwen/Qwen3-8B-AWQ` @ 4da05a8e | 6.10 | 4.85 | 1.25 BF16 (25%) | 5.00 | 3.05 | 9.69 | 2.06 |
| `Qwen/Qwen3-8B-GGUF` Q8_0 @ 7c41481f | 8.70 | 8.04 | 0.66 Q8_0 (8%) | 8.19 | 1.87 | 12.87 | 1.55 |
| `Qwen/Qwen3-8B-GGUF` Q6_K @ 7c41481f | 6.72 | 6.21 | 0.51 Q6_K (8%) | 6.36 | 2.40 | 11.04 | 1.81 |
| `Qwen/Qwen3-8B-GGUF` Q4_K_M @ 7c41481f | 5.02 | 4.67 | 0.51 Q6_K (11%) | 4.82 | 3.17 | 9.50 | 2.10 |

What the table teaches:
- **The linear layers alone come close to the bit ratio.** BF16 attention + MLP read 13.89 GB; NVFP4's read 3.91 GB (3.56x = 16 / 4.5 bits) and AWQ's 3.61 GB (3.85x = 16 / 4.16 bits). The scales are the first loss.
- **The BF16 lm_head is the second.** NVFP4 and AWQ keep it at 1.25 GB, a quarter of what they read per token. The GGUF Q4_K_M file quantizes it to Q6_K, which is why it reads the fewest bytes of the 4-bit rows despite more bits per linear weight (4.79).
- **The KV cache is the third, and it grows.** At 32k context every row adds the same 4.83 GB, and NVFP4's ceiling ratio drops from 2.88x to 2.00x. NVFP4 plus an FP8 KV cache at 32k reads 7.57 GB (2.64x vs BF16 with BF16 KV), but that is two precision changes in one row.
- **Bytes are not speed.** These are ceilings. Measured speed-ups are lower whenever a format's kernels reach a smaller fraction of the bandwidth than the BF16 kernels do.

Reproduce (the revisions are the commits the headers were read from): `inferencekit roofline --model <repo> --revision <sha> --method headers --context 1024` with `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218`, `Qwen/Qwen3-8B-FP8` @ `220b46e3b2180893580a4454f21f22d3ebb187d3`, `nvidia/Qwen3-8B-NVFP4` @ `ccd10a893cbca613259517c3efe08e151ddf2b8e` and `Qwen/Qwen3-8B-AWQ` @ `4da05a8edb55c6046cce958586c33b61da07bb79`; and `python3 scripts/gguf_bytes_per_token.py --repo Qwen/Qwen3-8B-GGUF --file Qwen3-8B-Q4_K_M.gguf --revision 7c41481f57cb95916b40956ab2f0b139b296d974`.

## 3. Batches, prefill and compute-bound work

- **Under load, dense weights are shared by the batch and the KV cache is not.** For a dense model with B sequences at context C, the bytes per generated token are `weights / B + KV(C) + state` (state on hybrids only). On Qwen3-8B at B = 16 and context 1024, NVFP4's ceiling ratio vs BF16 is 2.32x instead of 2.88x, and it keeps falling as B and C grow (derived from the table). Quantizing weights helps capacity mostly by freeing memory for more KV (a larger batch), which only a load run measures.
- **MoE under load reads more experts, so `weights / B` is wrong for it.** At batch 1 each MoE layer reads `top_k` experts; at batch B the tokens of one step pick up to `B x top_k` distinct experts, so the expert bytes read per step grow with B, up to all of them. With uniform, independent routing (an assumption: real routing is skewed, so this over-counts):
  `expert bytes per step ~= experts_stored_bytes x (1 - (1 - top_k / num_experts)^B)` and
  `bytes per generated token ~= (weight_bytes - experts_read_bytes + expert bytes per step) / B + KV(C) + state`
  (inputs from `data.counts.moe`; for GGUF, experts stored = `groups_read_bytes.moe_experts x expert_count / expert_used_count`). Example (derived from the formula): with 128 experts and `top_k` 8 (Qwen3-30B-A3B), a step at B = 16 reads about 64% of the expert bytes (6.25% at batch 1), so the expert bytes per generated token fall only about 1.55x, not 16x. The batch-1 ratio between formats therefore does not predict the ratio under load for MoE models, least of all for recipes that quantize only the experts.
- **Past the ridge point the math precision decides, not the bytes.** Prefill and large decode batches are compute-bound. Weight-only formats (W4A16, W8A16) still do 16-bit math and add dequantization work, so they can be slower than BF16 there: TTFT on long prompts often gets no gain or gets worse. W8A8 FP8 and W4A4 NVFP4 use faster tensor-core math where the GPU has it.
- So a batch-1 ceiling ratio never predicts TTFT or `N*`. Measure them per row.

## 4. Will it fit?

```
memory needed  ~= stored weight bytes                        (data.counts.checkpoint.stored_bytes, or GGUF stored_bytes)
                + concurrent sequences x (KV(C) + state held)  per sequence, at the context C each one holds
                + runtime overhead                           (CUDA context, activations, CUDA graphs, allocator slack)

KV(C)          = data.counts.kv_bytes of a roofline record run with --context C (sliding-window layers capped)
state held     = data.counts.state_bytes / 2                 (hybrids; the record counts it read + written per token)
```

- **Run `roofline` at the claim's context; do not scale a slope.** `kv_bytes_per_context_token` is the local slope at the record's context: sliding-window layers past their window add 0 to it, and recurrent state is not in it. Slope x context is the KV only for a dense model with full attention in every layer; for sliding-window or hybrid models (Gemma, gpt-oss, Qwen3-Next) it undercounts and declares a fit that is not one.
- **Some engines and versions keep the full context for sliding-window layers too.** Then KV(C) is the uncapped value: the sum over `data.counts.kv.layers` of `elements_per_token x (number of layers) x bytes_per_element x C`. Use it as the conservative bound until the engine's log shows the KV capacity it really allocated. Example (derived, `roofline --method headers --context 32768` on `openai/gpt-oss-20b` @ `6cee5e81ee83917806bbde320786a8fb61efebee`): 12 full-attention layers and 12 sliding-window layers (window 128) hold 0.81 GB per sequence capped and 1.61 GB uncapped. The slope there is 24,576 bytes per token, from the full layers only: slope x context misses the sliding layers' window, a small error for a 128-token window and a large one just past a long window.
- **Units:** GPU memory is sold and reported in GiB (1 GiB = 1.074 GB). The header byte counts are bytes: convert before comparing.
- **The overhead is inferred**, typically 1-3 GiB on top of what the engine leaves free by design (vLLM uses a fraction of memory set by `--gpu-memory-utilization`, 0.9 by default). Say so, and let the launch decide.
- **Every MoE expert must fit**, even though batch 1 reads only `top_k` of them. MoE checkpoints are large to store and cheap to read: Qwen3-30B-A3B Q4_K_M stores 18.55 GB but reads 1.92 GB of weights per token at batch 1 (derived, GGUF header).
- **Example (derived, then inferred):** Qwen3-8B (dense, full attention in every layer, so KV(C) = 147,456 bytes x C) on a 16 GiB (17.18 GB) GPU at 32k context and one sequence: BF16 16.38 + 4.83 = 21.2 GB, does not fit; FP8 9.44 + 4.83 = 14.3 GB, tight once overhead is added; AWQ 6.10 + 4.83 = 10.9 GB, fits. With vLLM's default 0.9 memory fraction (15.5 GB) and about 1.5 GB of overhead, roughly 7.9 GB is left for KV: about 1.6 sequences of 32k (inferred until the launch logs it).
- **Proof of fit is the engine's log**, recorded in `resolved_config` (wording varies by version): vLLM's model-loading memory line and its KV cache size in tokens plus the maximum concurrency for your context; SGLang's KV cache allocation line (number of tokens); llama.cpp's per-device model and KV buffer sizes and how many layers went to the GPU. KV capacity in tokens / context = the sequences that really fit.
- A row that only fits with a shorter context or fewer sequences than the intake asks for is reported with the limits it actually ran at, never with the intake's.
