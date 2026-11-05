# Ceilings: how fast could batch-1 decode possibly go?

Under load, and on several GPUs, the ceiling changes: see [capacity.md](capacity.md) (bandwidth use per load level) and [parallelism.md](parallelism.md) (TP sums the GPUs' bandwidth only when it splits the weights).

At batch 1, every decode step reads the weights it uses once, plus the KV cache of the context so far. So:

```
ceiling tok/s = measured DRAM bandwidth (bytes/s) / bytes read per token
% of ceiling  = measured decode tok/s / ceiling tok/s          (derived)
```

1. **Bandwidth.** On NVIDIA, run `inferencekit ceiling bw` and use its evidence id. Other platforms use the [hardware protocol](../../inference-triage/references/hardware.md#ceilings-and-profiles), recording measured GB/s and the source artifact. Never use datasheet bandwidth as a measured ceiling.
2. **Bytes per token, any safetensors checkpoint.**
   ```bash
   inferencekit roofline --model <dir> --gpu-bw <ev id> --context <ctx> --kv-dtype <bf16|fp8>
   ```
   - Run it once per bucket context: prompt length + about half of `max_tokens`.
   - By default (`--method auto`) it sums the real stored bytes from the safetensors headers. It handles NVFP4, FP8, MoE (top-k of the routed experts, plus shared experts), sliding-window and hybrid linear-attention models. `--dtype` applies only to `--method config`, the dense estimate from `config.json` (no safetensors, or a what-if for another weight format).
   - Pass the KV dtype the engine **resolved** (`--kv-dtype`). Read its `limitations`: the MoE caveat goes into the report.
   - Older `intake validate` builds warn that `roofline` covers dense models only. That warning is out of date: use `roofline`, and keep the script below as the cross-check.
3. **Cross-check, or when `roofline` cannot model the checkpoint** (it refuses, the headers are unavailable, or a format it does not know): `scripts/bytes_per_token.py` (below). Label the result **derived**, with the formula and inputs in `state` and the report. When the two disagree by more than a few percent, find out why before you use either.

## Bytes per token from safetensors headers (any format, exact stored bytes)

A safetensors file starts with an 8-byte little-endian header length, then a JSON header. The header gives each tensor's dtype, shape and byte range. Summing byte ranges counts exactly what is stored, including packed 4-bit values and their scale tensors. No format arithmetic is needed.

This skill ships the calculation as `scripts/bytes_per_token.py` (stdlib only, reads headers only). Copy it to the machine that holds the checkpoint and run it once per bucket context:

```bash
scp -P <port> -i <key> <this skill's directory>/scripts/bytes_per_token.py <user>@<host>:~/ik/   # ${CLAUDE_PLUGIN_ROOT}/skills/optimizing-inference
$SSH 'python3 ~/ik/bytes_per_token.py ~/ik/models/<dir> <context> --kv-bytes <2|1>'
```

- Output (JSON): `bytes_per_token`, `weights_per_token`, `kv_bytes`, `groups_bytes`, `checkpoint_bytes`, `sample_tensors_per_group`, `moe`, `attention` (layer counts by kind and the KV formula used), `notes`, `formula`. Record all of it in `state` and the report: the result is **derived**.
- **It exits 2 instead of guessing** when a guess would overstate bytes per token. Read the message, fix the input, and record what you passed and why:
  - **MoE config but no routed-expert tensors matched** (default pattern `\.experts\.`): look at `sample_tensors_per_group` and pass `--expert-pattern '<regex>'`. Pass `--topk` / `--n-experts` when the config uses other key names; record where they came from.
  - **Non-full-attention keys but no `layer_types`** (`sliding_window`, `sliding_window_pattern`, `full_attention_interval`, `attention_chunk_size`, `linear_*`, SSM keys): pass `--layer-types` as a comma list tiled to the layer count, e.g. `sliding,sliding,sliding,sliding,sliding,full`. `--layer-types full` is an explicit statement that every layer is full attention: say why.
  - **Per-layer-type KV shapes, MLA (`kv_lora_rank`) or shared KV:** derive the KV bytes by hand and pass `--kv-bytes-total`, with the formula written down.
- When `config.json` has `layer_types`, the KV term is computed per layer: full-attention layers read `context` tokens, sliding/chunked layers `min(context, window)`, and linear-attention/SSM layers 0 per-token KV (their fixed-size state is not counted: that understates bytes, the conservative direction).

### Check the result before using it
- **Direction of error:** a bytes model that is too high lowers the ceiling and **inflates % of ceiling**. Every doubt goes the other way: when unsure, count fewer bytes, never more.
- **Totals:** `checkpoint_bytes` ≈ the checkpoint size on disk (the download's file sizes).
- **MoE:** `weights_per_token` should be close to the model card's active parameter count × stored bytes per parameter. If it is much larger, expert tensors landed in `every_token`: fix `--expert-pattern`.
- **Shared experts** are read every token. Make sure their tensor names landed in `every_token`, not in `routed_experts`.
- **KV bytes per element:** 2 for bf16/fp16, 1 for fp8. Use the KV dtype the engine **resolved**, not the one you asked for.
- **Skipped groups** (vision or audio towers, MTP heads) are not read during text decode. Check `groups_bytes.skipped` is what you expect. With speculative decoding via MTP, the MTP head *is* read: add it back.
- **MoE at batch 1:** different tokens route to different experts, so the per-token average is `topk/n_experts` of the expert bytes. The ceiling assumes no expert is re-read within a step. Say this in the report: it is a modelling assumption (inferred).
- **Faster than the ceiling** means a bug: in the bytes model, in the bandwidth, or in the token counting. Stop and find it.

## Per bucket
Compute the ceiling at each bucket's mean context. Decode slows as the KV cache grows, so the `long` bucket has a lower ceiling. A `decode_ceiling_fraction` target is checked bucket by bucket, against that bucket's ceiling.

## Exact byte floor (headroom estimate 2)

Part of the stored bytes can go without changing one output bit. `scripts/bytes_per_token.py` reports (derived): `bytes_by_dtype`; `packable_16bit_bytes`, the 16-bit weights per token that lossless re-encoding shrinks; `lm_head_bytes` and `lm_head_share`, which a bound-pruned output layer mostly skips; and `floor_bytes_per_token` with `floor_formula`, after both cuts.

Bytes headroom = `bytes_per_token` ÷ `floor_bytes_per_token`, counting only cuts the fidelity allows ([fidelity.md](fidelity.md); recipes and proofs: [optimizing-kernels](../../optimizing-kernels/SKILL.md)). The floor is an estimate: re-measure what the built engine reads and recompute the ceiling.
