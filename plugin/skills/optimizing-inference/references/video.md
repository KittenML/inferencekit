# Video clip engines (text-to-video, image-to-video)

A video model makes one clip per request, after many denoising steps over tens of thousands of video tokens: it is **compute-bound**, not memory-bound like batch-1 LLM decode. The examples use the LTX-2.x family; the method holds for any diffusion video model.

## The workload: pin it before timing anything

- **Seconds per clip** at a fixed width, height, frame count, fps, step count, guidance scale and seed. Text-to-video (`t2v`) and image-to-video (`i2v`) are separate tasks with separate numbers.
- **Clips per hour**, one clip at a time: 3,600,000 x clips / the sum of their clip times, per round (derived), and per GPU.
- **Stage times** (text encode, denoise, video and audio decode): self-reported by the engine, so use them to find where time goes, never as the headline.

Write the shapes into a clip set (JSONL, one clip per line: `id`, `task`, `prompt`, `image` for i2v, `width`, `height`, `num_frames`, `fps`, `steps`, `guidance_scale`, `audio`), one task per set. In the intake: `modality: video`, targets `clip_ms` or `clips_per_hour`. Fewer steps, a smaller shape or a dropped guidance pass is a different workload, not a speed-up ([video-checks.md](../../benchmarking/references/video-checks.md)).

## Candidate engines

- **ComfyUI** with its native nodes for the model and an API-format workflow; record its own memory and offload policy.
- **The model's official pipelines** (for LTX-2.x, the official LTX code): the reference implementation.
- **Diffusers** (the model's pipeline, often from the main branch). Note any CPU offload it turns on.

Give every engine the same checkpoint, steps, shape and seed. The reference pipeline in BF16 (96 GB for a 20B-class model) is the quality reference.

## Serving

A clip server ships with the harness; install the inferencekit wheel into the pipeline's venv:

```bash
<venv>/bin/python -m inferencekit.engines.clip.server --pipeline diffusers_ltx --model <dir> --task t2v --port 8000
```

`--pipeline official_ltx --pipeline-arg builder=inferencekit.engines.clip.official_ltx:one_stage --pipeline-arg builder.gemma_root=<dir>` serves the official one-stage pipeline with its own defaults (no `--attention`, no reference mode; it rebuilds the transformer every clip, as shipped). LTX-2.5-Diffusers' `transformer/` is the distilled model: the full one is `--pipeline-arg transformer_subfolder=transformer_full --pipeline-arg 'scheduler={"use_dynamic_shifting": true, "shift_terminal": 0.1}'`. The BF16 reference is `--pipeline diffusers_ltx --pipeline-arg reference=true --pipeline-arg default_negative_prompt=true` (reference mode wants the negative prompt named); its `--reference-alt` floor is that command without `reference=true` and with `--dtype bf16 --attention _native_efficient` (exact, memory linear in tokens; `native` picks flash without a mask; `_native_math` can run out of memory). Make one alt clip at the target shape before you pre-register it; if it is bit-identical to the reference's, it ran the same kernel: try `_native_cudnn`. One process serves one task: time t2v and i2v with separate clip sets. A server of your own must draw the initial noise with the shared recipe (`inferencekit.engines.clip.noise`: CPU generator seeded with `noise_seed`, float32) and write the per-step dumps (`inferencekit.engines.clip.stepdump`), or the quality checks cannot pass it.

## The clip HTTP contract

- `GET /health` answers 200 once the model is loaded.
- `POST /v1/clips` takes JSON `{task, prompt, image?, width, height, num_frames, fps, steps, guidance_scale, seed, noise_seed?, audio?}`; `image` is base64 for i2v.
- It answers with the clip, as `video_b64` or as `video_path` with the finished file's `video_sha256` and `video_bytes` (so it cannot answer before the file is written), plus `format`, `stage_ms` per stage, `cache` (cached stages: only `negative_text_encode`, a fixed negative prompt encoded once, is allowed) and `effective`: `steps_run`, `width`, `height`, `frames`, `fps`, `dtype`, `attention`. Report what really ran: a mismatch or a cached stage makes the claim invalid.

ComfyUI needs no wrapper: `--endpoint-kind comfyui --workflow <api-format workflow.json>`. Its node cache may reuse only the fixed negative prompt's encoder, listed in the workflow file's `allow_cached` (allowed for every engine, stated in the record); any other cached stage is CLAIM INVALID. A graph that always makes audio says `"audio": true` there. Launch ComfyUI with `--cache-classic`: its default cache keeps earlier prompts' encodings, so a prompt repeated in a later round is reused (CLAIM INVALID). Give the clip set distinct prompts: bench clip warms up with its own and orders clips so neighbours differ.

## The ceiling: FLOPs, not bytes

Per denoising step, from the transformer's `config.json` (write formula and inputs into `state`, label it derived):

```
tokens N        = (1 + (frames - 1) / temporal) x (height / spatial) x (width / spatial) / (patch_t x patch_h x patch_w)
linear FLOPs    = 2 x N x (parameters in the linear layers)
attention FLOPs = 4 x N^2 x hidden size x layers         (QK^T and AV; add cross-attention to the text)
per step        = (linear + attention) x forward passes   (2 with classifier-free guidance)
```

`temporal` and `spatial` are the VAE's compression (causal video VAEs keep the first frame). Compute both terms: with about 14 x hidden^2 linear parameters per layer, the attention share is about N / (N + 7 x hidden). With a 32x spatial VAE (LTX-2.x) the linear layers dominate at 720p and attention reaches parity near 1080p; 8x-VAE models are attention-bound at the same size. Attack the larger term first.

Measure the dense matmul rate on this GPU per dtype (BF16, and FP8/FP4 when the card has them): a warmed-up 8192 x 8192 x 8192 matmul, median of 20 runs. The datasheet is a label, not a reachable number. The ideal denoising time is `per step x steps / measured FLOP/s`; % of ceiling = ideal / measured denoising time (derived).

## Levers

**Lossless** (same checkpoint, same numerics within the A/A floor; `quality clip-steps` decides):
- attention kernels: FlashAttention-class BF16 kernels for the GPU's architecture, or a faster SDPA backend;
- fusion: norm + modulation, QKV projections, activations;
- `torch.compile` and CUDA graphs: the shapes are fixed, so capture once per timed shape (bench clip's warm-up covers every distinct shape);
- faster video and audio decode (a large share of short clips);
- overlapping stages (audio decode beside video decode, text encoding before the loop);
- offload only when memory forces it, then prefetch the next block on a side stream.

**Lossy** (a separate track with a stated quality budget, never ranked with lossless): FP8 or FP4 matmuls; low-precision attention (SageAttention-style); a distilled few-step checkpoint; step caching (reusing transformer work across denoising steps).

## Commands

With the engine up (`/health` answers 200):

```bash
inferencekit bench clip --base-url http://127.0.0.1:8000 --clips clips/dev.jsonl --label tuned \
  --model <repo id> --rounds 3 --warmup 1 --run-dir runs/<id>
```

It times every clip from the client (`clip_ms`, observed; a round's value is the mean over its clips), adds clips per hour (derived) and stage medians (self-reported), and keeps every clip under `outputs/<label>/r<round>/`. Seeds are fixed per round, the same for every engine; a re-run elsewhere picks a new `--seed-base`. To compare engines, interleave them:

```bash
inferencekit bench rounds --manifest runs/<id>/manifest.json --prompts clips/dev.jsonl --mode clip \
  --rounds 6 --warmup 1 --run-dir runs/<id>
```

Manifest engines carry `endpoint_kind` (and `workflow` for ComfyUI); `ready_check` defaults to `/health`. Keep `--warmup 1` (a whole clip per distinct shape). Then run `inferencekit gates clip` per live engine (`--python <video venv>/bin/python`, `--bench-evidence <bench clip record>`), `inferencekit quality clip-steps` and `inferencekit quality clip` of each engine's clips against the BF16 reference; what each checks and needs: [video-checks.md](../../benchmarking/references/video-checks.md).

`inferencekit review build runs/<id>` shows each clip beside the reference's, one table per round, with clip time, stages and quality.

An engine that cannot reach the contract, or a check that cannot run, is NOT RUN, never a pass.

## License

Benchmark publication depends on the model's license terms. Many video models ship under a community license with a gated download, not an OSI license (LTX-2.x does): read the terms on publishing benchmarks and sharing clips before anything leaves the run directory.
