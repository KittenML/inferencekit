# Worked example: bytes per token and ceiling for Qwen3-8B FP8

Everything here is **derived** arithmetic from the model config and published GPU constants. Use it to sanity-check `inferencekit roofline` output, and as a template for other models.

## 1. Config values used

From the model's `config.json` (Qwen3-8B): `num_hidden_layers` 36, `hidden_size` 4096, `num_attention_heads` 32, `num_key_value_heads` 8, `head_dim` 128, `intermediate_size` 12288, `vocab_size` 151936, `tie_word_embeddings` false.

The FP8 checkpoint stores the linear-layer weights as FP8 E4M3 with one BF16 scale (`weight_scale_inv`) per 128x128 weight block, and keeps the embeddings and lm_head in BF16. Check `quantization_config` in the checkpoint you deploy; other FP8 checkpoints differ.

## 2. Weights read per decode token

Per layer, shapes are `[out, in]`:

| Matrix | Shape | Elements |
|---|---|---|
| q_proj | 4096 x 4096 | 16,777,216 |
| k_proj | 1024 x 4096 | 4,194,304 |
| v_proj | 1024 x 4096 | 4,194,304 |
| o_proj | 4096 x 4096 | 16,777,216 |
| gate_proj | 12288 x 4096 | 50,331,648 |
| up_proj | 12288 x 4096 | 50,331,648 |
| down_proj | 4096 x 12288 | 50,331,648 |
| **per layer** | | **192,937,984** |

- 36 layers: 6,945,767,424 elements x 1 byte = **6.946 GB**.
- Block scales: per layer 1,024 + 256 + 256 + 1,024 + 3 x 3,072 = 11,776 blocks; x 36 x 2 bytes (BF16) = **0.85 MB** as stored, or 1.7 MB if an engine upcasts them to FP32 (negligible, but not zero).
- lm_head: 151,936 x 4,096 = 622,329,856 elements x 2 bytes = **1.245 GB**. That is 15% of all weight bytes: at batch 1 the vocabulary projection is not a rounding error.
- Embedding: one row per token, 4,096 x 2 bytes = 8 KB. Norm weights: about 0.6 MB.
- **Weights total: ~8.19 GB per token.**

## 3. KV cache read per decode token

Each new token's attention reads K and V for every previous token, in every layer:

```
KV bytes per context token = layers x 2 (K and V) x kv_heads x head_dim x bytes
                           = 36 x 2 x 8 x 128 x 2 (BF16) = 147,456 bytes (144 KiB)
```

| Context length | KV read per token (BF16 KV) | Total bytes per token | Ceiling at 1,792 GB/s |
|---|---|---|---|
| 1,024 | 0.151 GB | 8.34 GB | ~215 tok/s (4.66 ms) |
| 8,192 | 1.208 GB | 9.40 GB | ~191 tok/s |
| 32,768 | 4.832 GB | 13.02 GB | ~138 tok/s |

- Crossover where KV bytes equal weight bytes: 8.193e9 / 147,456 = **~55,600 tokens**.
- An FP8 KV cache halves the KV column (and is a precision change: compare only with other FP8-KV runs).

## 4. Other GPUs (datasheet bandwidth, 1,024-token context)

| GPU | Datasheet bandwidth | Ceiling |
|---|---|---|
| RTX 5090 (GDDR7) | 1,792 GB/s | ~215 tok/s |
| H100 SXM (HBM3) | 3,350 GB/s | ~401 tok/s |
| H100 PCIe (HBM2e) | 2,000 GB/s | ~240 tok/s |

Replace the datasheet value with `inferencekit ceiling bw` before quoting any of these as a ceiling. Streaming-read microbenchmarks land below datasheet, often around 85-95% (inferred from general experience; measure yours).

## 5. Variants worth computing

- **lm_head in FP8 too:** weights drop to ~7.57 GB, ceiling at 1K context rises to ~232 tok/s (+8%). It changes logits, so it needs a correctness proof before anyone calls it a speedup.
- **BF16 everything:** 6.946 x 2 + 1.245 = ~15.1 GB of weights, ~117 tok/s at 1K context on 1,792 GB/s. This is the ceiling to compare against llama.cpp BF16 runs.
- **Q8_0 (llama.cpp):** linear weights at 1.0625 bytes per weight = 7.38 GB, plus the lm_head in whatever type the GGUF stores it. Read the GGUF tensor types rather than assuming.

## 6. What the ceiling ignores (and why that is fine)

- **Activations:** at batch 1 the hidden state is 4,096 x 2 bytes = 8 KB per layer boundary. Tiny next to 8 GB.
- **Compute:** count parameters, not bytes (the lm_head is 2 bytes per parameter): about 2 x 7.57e9 parameters = ~15 GFLOPs per token, plus ~0.6 GFLOPs of attention at a 1K context (4 x 36 layers x 32 heads x 128 x 1,024). At ~215 tok/s that is ~3.4 TFLOP/s, a few percent at most of an RTX 5090's tensor throughput. Decode is not compute-bound at batch 1.
- **L2 reuse:** none across tokens for an 8 GB working set on a 96 MB L2. For small models whose weights approach the L2 size, the DRAM ceiling understates what is possible.
- **Launch and hand-off costs:** not included; see launch-cost.md. They are why real engines sit below the ceiling even with perfect kernels.
