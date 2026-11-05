# Ceilings: the speed of light for decode, batched decode and prefill

Every speed number is reported against a ceiling, so the reader knows whether the gap is worth chasing. The ceiling uses **measured** bandwidth, never the datasheet value.

## Why decode has a bandwidth ceiling

At batch 1, every decode step reads all the weights once, and each weight takes part in one multiply-add. Arithmetic intensity (FLOPs per byte moved) is about 2 FLOPs per byte at FP8, far below any GPU's ridge point (the intensity where compute, not memory, becomes the limit). So the time per token cannot be lower than **bytes read per token / DRAM bandwidth**.

## Preflight

- [ ] `inferencekit doctor` this session. Why: bandwidth depends on the exact GPU SKU, memory clock and power state.
- [ ] GPU idle (no other compute process). Why: a neighbour's traffic lowers measured bandwidth.
- [ ] The deployed dtypes are known per tensor group (linear weights, lm_head, embeddings, KV cache). Why: FP8 checkpoints often keep the lm_head in BF16, a large share of the bytes.
- [ ] The context lengths you will report are fixed. Why: KV cache bytes grow linearly with context.

## Hard failure gates

- **Measured bandwidth above the datasheet value:** an L2-resident buffer, so the measurement is wrong. The DRAM figure is the **read** plateau at buffer sizes far above L2 (decode streams weights read-only: a read test, not a copy test).
- **Measured bandwidth below ~70% of datasheet.** Usually a power cap, low memory clock, a shared GPU or a virtualised slice; fix or record it before trusting any ceiling.
- **Roofline dtype different from what the engine loads.** The ceiling would describe a different model.
- **Measured decode tok/s above the ceiling.** Impossible: the token count, dtype, or bytes-per-token model is wrong (after a byte cut: Decision rules).

## Workflow

```
Ceilings:
- [ ] 1. inferencekit doctor                                         -> ev_env
- [ ] 2. inferencekit ceiling bw                                     -> ev_bw (read plateau at sizes >> L2)
- [ ] 3. measured / datasheet bandwidth (derived)
- [ ] 4. inferencekit roofline --model M --dtype D --gpu-bw <ev_bw> --context L   -> ev_roof, per context reported
- [ ] 5. Check bytes/token by hand for one tensor group (worked-example-qwen3-8b.md)
- [ ] 6. Launch cost: nsys capture of steady decode -> kernels per step and median hand-off gap (launch-cost.md)
- [ ] 7. Batched / prefill ceilings by hand when those goals are active (below)
```

```bash
inferencekit doctor
inferencekit ceiling bw                                                         # -> ev_bw
inferencekit roofline --model Qwen/Qwen3-8B-FP8 --dtype fp8 --gpu-bw <ev_bw> --context 1152
inferencekit roofline --model Qwen/Qwen3-8B-FP8 --dtype fp8 --gpu-bw 1792      # datasheet, for contrast only
```

MCP tools: `doctor`, `measure_bandwidth`, `roofline`. `--model` takes a Hugging Face id or a local `config.json`; `--dtype` accepts `fp8`, `bf16`, `q8_0`; `--kv-dtype`, `--lm-head-dtype` and `--revision` refine it; `--gpu-bw <ev_bw>` reuses a stored measurement. `--gpu-bw measure` runs the microbenchmark (GPU-occupying). The context to use for a decode benchmark is the mean context of the timed tokens: prompt + about half the generated tokens.

## Without `ceiling bw` (AMD ROCm, x86 CPU, Apple Silicon)

`ceiling bw` is NVIDIA-only for now: NOT RUN elsewhere, even when it starts on ROCm (an inflated, identity-less figure: never cite it; [hardware.md](../../inference-triage/references/hardware.md), note 4, and its platform files). The ceiling is the same division with another bandwidth source.

| Platform | Measured read bandwidth (preferred) | Fallback |
|---|---|---|
| AMD ROCm | BabelStream HIP, the **Dot** kernel, arrays far above the 256 MB Infinity Cache on MI300X | datasheet (MI300X 5.3 TB/s, MI325X 6 TB/s) |
| x86 CPU | `likwid-bench` read-only kernel, or Intel MLC "ALL Reads" | STREAM Triad (counts writes too: label it), then channels x MT/s x 8 B |
| Apple Silicon | an MLX read test (`mx.sum` over 1 GiB, timed around `mx.eval`) | Apple's published bandwidth |

- **Save the tool's output** with the command and version in `runs/<run-id>/hw/bandwidth.txt`. The number is **observed (vendor tool)**, cited by that file, not an evidence id.
- **Pass the GB/s to `roofline`:** `inferencekit roofline --model <dir> --dtype <fmt> --gpu-bw <GB/s> --context <ctx>`. Its record notes the bandwidth is user-supplied; cite the file beside it.
- **Above the datasheet** means a cache-resident buffer, as on NVIDIA. **Far below it,** check for a power cap, a low-power mode or a shared device. The ~70% rule of thumb above comes from NVIDIA GPUs; CPUs often sustain a smaller share of their theoretical figure.
- **Datasheet only:** "datasheet-based; measured ceiling NOT RUN", and every % of ceiling is UNPROVEN.

## Worked example (Qwen3-8B FP8)

| Quantity | Value | Class |
|---|---|---|
| Weights per token (FP8 linear layers, BF16 lm_head) | 8.19 GB | derived |
| KV per context token (BF16, 36 layers x 2 x 8 heads x 128 x 2 B) | 147,456 B | derived |
| Bytes per token at context L | 8.19 GB + L x 147,456 B | derived |
| Ceiling at context L | measured read plateau (GB/s) / bytes per token | derived, with ev_bw |
| % of ceiling | measured tok/s / ceiling tok/s | derived, with ev_bench and ev_roof |

The early engine (engine-timed, see rule 18) reached about nine-tenths of its measured-bandwidth ceiling at a 1,024-token prompt; past that point only fewer bytes per token help. Full arithmetic, context variants and datasheet contrasts for other GPUs: [worked-example-qwen3-8b.md](worked-example-qwen3-8b.md).

## Batched decode (capacity) ceiling (derived by hand)

With N sequences decoding together, one step reads the weights once (shared) and each sequence's own KV cache:

```
step time   >= (W + N x KV(L)) / BW          W = weight bytes, KV(L) = KV bytes per sequence at context L
tok/s total <=  N x BW / (W + N x KV(L))
```

As N grows the KV term dominates, so long contexts lower the batched ceiling sharply. At large N the step also becomes compute-bound: `2 x params x N` FLOPs per step against the GPU's tensor throughput. Take the lower of the two. `inferencekit roofline` computes the batch-1 case; compute the batched case by hand and show the formula.

## Prefill ceiling (TTFT)

Prefill multiplies all prompt tokens at once, so it is compute-bound for all but tiny prompts: `FLOPs ~= 2 x params x prompt_tokens` plus attention (`~4 x layers x heads x head_dim x prompt_tokens^2 / 2` for causal attention). Divide by the tensor throughput for the precision used. inferencekit has no FLOP microbenchmark, so a datasheet-based prefill ceiling is labelled "datasheet-based; measured compute ceiling NOT RUN". Short prompts are usually host-bound (launches), not compute-bound: the early engine spent ~45 ms on eager launches around 11 ms of GPU work at 128 tokens before it captured prefill as a CUDA graph.

## Launch cost

The bandwidth ceiling assumes the GPU streams bytes continuously. Each dependent kernel adds a hand-off (the previous kernel drains, the next ramps up). Derive it from an nsys timeline (kernels per step x median gap, plus per-step idle): [launch-cost.md](launch-cost.md).

## Fixed result table

| Quantity | Value | Class | Evidence / source |
|---|---|---|---|
| Datasheet DRAM bandwidth | | published constant | vendor spec sheet |
| Measured read bandwidth (plateau) | | observed | ev_bw |
| Measured / datasheet | % | derived | ev_bw |
| Weight bytes per token | GB | derived | ev_roof |
| KV bytes per token at context L | GB | derived | ev_roof |
| Ceiling tok/s at measured bandwidth (per context) | | derived | ev_roof |
| Batched ceiling at N (if capacity is a goal) | | derived | formula above |
| Hand-off floor per token (kernels x gap) | ms | derived | ev_nsys |
| Measured tok/s and % of ceiling | | observed / derived | ev_bench |

## Decision and no-go rules

- Use measured bandwidth for every "% of ceiling". Why: datasheet numbers are not reachable by any kernel, so they make every engine look worse than it is.
- Byte cuts are not a last resort: estimate the exact byte floor at the start ([optimizing-inference ceilings](../../optimizing-inference/references/ceilings.md)) and work exact cuts alongside kernel work. Why: the gains multiply, and a kernel tuned for the old format is redone after the cut.
- Near the ceiling (~85% or more; the [ceiling bands](../../inference-triage/SKILL.md#ceiling-bands)), stop tuning kernels. Why: only reading fewer bytes per token can help (exact byte cuts: lossless weight re-encoding or an exact bound-pruned lm_head, [reward-hacks.md](reward-hacks.md); lower precision with a quality verdict; a smaller model; lossless speculative decoding).
- The ceiling is per storage format: after a byte cut, recompute bytes per token for what the engine reads (by hand if `roofline --dtype` lacks it); then above 100% of the old ceiling is expected.
- Hand-off floor above ~10% of the token time: CUDA graphs, fusion and programmatic dependent launch are worth more than faster GEMVs. Why: they remove gaps, not bytes.
- Recompute the ceiling for every context length you report. Why: at long context the KV cache can rival the weights (about 55,600 tokens for Qwen3-8B with a BF16 KV cache).

## Reporting rules

- Datasheet bandwidth is a published constant; measured bandwidth is **observed**; bytes per token, ceilings and % of ceiling are **derived** (show the division); "this kernel is memory-bound" is **inferred** until ncu shows DRAM throughput near the measured peak.
- If bandwidth was not measured, the ceiling row says "datasheet-based; measured ceiling NOT RUN", and any % of ceiling is UNPROVEN.
