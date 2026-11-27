# Image and speech measurements

`bench image` and `bench audio` time complete media requests on the client. They run against an already-started server implementing the contracts below. Both are also MCP tools, `bench_image` and `bench_audio`. CPU tests exercise synthetic servers; real hardware results remain NOT RUN until measured.

Use `inferencekit doctor` on the engine host to identify its hardware and profiling route. Pin the model revision, complete precision/quantization recipe, engine revision and hardware in the benchmark arguments. Declare unquantized weights explicitly as `none`. Effective settings are checked against the server response, but remain self-reported: retain the engine's resolved configuration and logs too.

## Images

Create a JSONL fixture file, one request per line:

```json
{"id":"one","prompt":"A red ceramic mug on a wooden table","width":1024,"height":1024,"steps":50,"guidance_scale":3.0,"settings":{}}
```

```bash
inferencekit bench image --base-url http://127.0.0.1:8000 --model org/model \
  --model-revision CHECKPOINT_SHA --dtype bf16 --quant none \
  --engine-version ENGINE_SHA --gpu-name 'Exact GPU model' \
  --prompts images.jsonl --label baseline --rounds 1 --first-round 0 --run-dir runs/images
```

The server accepts `POST /v1/images/generations` with `model`, `prompt`, `size` (`WxH`), `width`, `height`, `steps`, `guidance_scale`, `seed`, plus the supplied settings. Its JSON response contains:

```json
{"image_b64":"BASE64_PNG","format":"png","effective":{"model":"org/model","width":1024,"height":1024,"steps_run":50,"guidance_scale":3.0,"seed":123,"model_revision":"CHECKPOINT_SHA","dtype":"bf16","quant":"none"},"cache":[]}
```

Report actual executed settings in `effective`, including every requested setting. `cache` lists reused stages; an empty list means none. The client validates PNG dimensions, CRCs and decompressed pixels, rejects repeated outputs for different requests, and includes decoding in `image_ms`. `images_per_hour` is derived from image latency. Loading and compilation are separate; warmup images are excluded. Structural validity does not prove prompt adherence or image quality.

By default any reported cached stage invalidates the run. To permit reuse of an unchanged empty-negative embedding, use `--allow-cache fixed_negative_embedding` and pin `"negative_prompt":""` in every fixture's `settings`. The server reports `cache:["fixed_negative_embedding"]` only when it actually reuses that embedding. This allowance is bound into the request-set hash and must be available equally to both engines. It permits no positive-prompt, denoising, VAE or output reuse. Retain independent execution proof of the reused stage and fixed model/settings; a server declaration alone cannot prove the policy was followed.

## Speech

Create a JSONL fixture file:

```json
{"id":"one","text":"The meeting starts at ten tomorrow morning.","unseen":true}
```

For voice conditioning, add `reference_audio` (a path relative to the fixture file) and its `reference_text` transcript. The evidence binds fixture text and reference bytes by hash. `unseen` is an attestation, not a proof about the server's history.

```bash
inferencekit bench audio --base-url http://127.0.0.1:8000 --model org/model \
  --model-revision CHECKPOINT_SHA --model-dtype bf16 --model-quant none \
  --engine-version ENGINE_SHA --hardware-name 'Apple M4, 24 GB unified memory' \
  --fixtures speech.jsonl --label baseline --rounds 1 --first-round 0 --run-dir runs/speech
```

The server accepts `POST /v1/audio/speech` with `model`, `input`, `seed`, `sample_rate`, `channels`, `encoding: "pcm_s16le"`, `stream: true` and `settings`. Reference bytes, when present, arrive as `reference_audio_b64` with `reference_text`. Return flushed UTF-8 NDJSON events in this order:

```json
{"type":"metadata","sample_rate":44100,"channels":1,"encoding":"pcm_s16le"}
{"type":"audio","pcm_b64":"BASE64_PCM"}
{"type":"final","sample_count":44100,"effective":{"model":"org/model","seed":123,"model_revision":"CHECKPOINT_SHA","dtype":"bf16","quant":"none","engine_version":"ENGINE_SHA"}}
```

Repeat audio events as needed. `sample_count` counts frames per channel. The final `effective` object must include every requested setting. Model/engine pins travel inside request `settings`; the CLI's `--model-dtype` and `--model-quant` use wire keys `dtype` and `quant`. A final event is mandatory and terminal; timing consumes the complete response body. The example seed and sample count are placeholders for actual output.

Failed request rows have a null `sample_count` and no valid speed metrics. Their `received_pcm_frames` and `chunks` preserve how much decoded PCM arrived and when, including partial delivery before a timeout. These diagnostics do not establish a complete or usable speech output.

The client reports first PCM arrival (`first_pcm_p95_ms` for round comparisons), first nonzero signal arrival, simulated nonzero playback onset, complete generation time, real-time factor (wall time / generated audio duration), gaps and underruns with zero extra prebuffer. `first_audio_p95_ms` in round comparisons uses simulated nonzero onset, accounting for leading zero PCM and stalls. A nonzero sample is not proof of audible speech. Default timing limits are p95 onset <=2 seconds, maximum RTF <=1 and zero underruns; use the SLO flags to register different limits before measuring.

`inferencekit review build RUN_DIR` displays saved PNG images and speech outputs. Raw PCM is wrapped in a WAV under `review/media/` only after its hash, byte count, encoding, sample rate and channel count match `outputs/LABEL/index.json`; the source PCM stays unchanged. The review data lists each playback copy and its source hash. For speech captions, pass `--prompts FIXTURES.jsonl`: unbucketed fixture texts fill every observed round of the matching id. Attach the original evidence records to display per-request timing and failures. The page keeps simulated nonzero onset distinct from first PCM arrival and semantic quality remains UNPROVEN until separate quality evidence is supplied.

## Comparisons and proof

Use `bench rounds` to launch each engine fresh per round, check readiness, warm up, measure and stop its process group in balanced order. Set manifest `workload.modes` to `["audio"]` or `["image"]`; `--mode auto` also recognizes modality and endpoint declarations. Engines use `audio_http` or `image_http` and default to `/health` readiness. Keep the same fixture set, precision, generation settings and round seeds for both engines.

```bash
inferencekit bench rounds --manifest manifest.json --prompts speech.jsonl --mode audio --rounds 4
# Use the two final arm IDs from plan.audio_evidence_final for compare and quality audio.
```

The manifest supplies `model.id`, `model.revision`, `model.precision`, each engine's version/launch/environment/port, and `workload.quant` (`none` for unquantized). Audio also accepts `workload.sample_rate`, `channels`, `seed` and `settings`; image settings stay in each fixture. `--serve-args` can supply client timeouts and SLOs but cannot override the scheduler's identity, seeds or round indices. `--mode clip` uses the video client.

For CPU servers, set `hardware` to `{"backend":"cpu","cpu":{"model":"Exact CPU model"},"count":1}`. The runner uses a CPU-exclusive advisory lock and records sampled process-group RSS, OS thread counts, affinity and selected runtime environment controls. Requested thread settings do not prove a runtime used that many workers. RSS is host RAM, may miss brief peaks and counts shared pages per process; it is not VRAM or swap. CPU load/capacity mode is not supported. Direct CPU media calls use `--device cpu --hardware-name 'Exact CPU model'`.

Manual drivers can still use one `bench audio`/`bench image` round per invocation and `--extend EVIDENCE_ID`. Continuation requires identical configuration and intact artifacts. For audio ports that change on restart, set the same `--endpoint-id` from the first round; otherwise keep the endpoint fixed. Scheduler-final records are immutable; start a new scheduler run instead of extending them.

```bash
inferencekit compare BASELINE_ID CANDIDATE_ID --metric image_ms
inferencekit compare BASELINE_ID CANDIDATE_ID --metric first_audio_p95_ms
inferencekit report BASELINE_ID CANDIDATE_ID COMPARISON_ID --format md
```

`compare` requires matching request hashes and pinned model identities, at least three complete matching pairs, interleaved non-overlapping timing windows, and no structural failures. Use at least four rounds for a balanced two-engine final run. Missing scheduled rounds prevent a conclusive comparison; manual chains can declare `--expected-rounds 4` to check missing tail rounds. Known memory measurements appear beside timing, with their method and coverage.

The paired bootstrap CI measures timing only. Scheduler-final records can establish fresh-process lifecycle. Memory fit, semantic quality and sustained reliability need their own evidence; failed speech SLOs remain failures. Register speech acceptance before capture with `quality audio-policy`, then evaluate saved PCM using [`quality audio`](audio-quality.md). Cross-precision results remain format trade-offs requiring separate quality reporting.

Keep output artifacts for quality evaluation (`--save-outputs all`, the default). Retain cold-start/compile timing, whole-device and process memory, pressure/swap, sustained long outputs, task quality and mechanism profiles separately. Do not infer a smaller-memory-device result from a run on larger hardware.

### Image speed/quality tradeoffs

When the user permits fewer steps or a different precision, record each arm with its actual settings. New image records include a task hash covering prompts, resolution, seed policy and other settings, excluding only step count and numerical format. For matching task hashes, compare timing explicitly as a tradeoff:

```bash
inferencekit compare BASELINE_ID CANDIDATE_ID --metric image_ms --image-tradeoff
```

The same operation is available over MCP with `image_tradeoff: true`. This is a **format/work tradeoff**, not a same-work engine comparison. The timing verdict is not a quality verdict: retain independent checks against the agreed quality budget, and report quality, lifecycle and reliability separately. Mismatched prompts, dimensions, model revision, seed policy, other settings, failed outputs or unverified interleaving still prevent a valid comparison. Older image records without the task hash require new records; do not fabricate a hash.

## Speech recognition

In every experiment (quantisation, precision, activation bit-width, kernel or graph change, approximation, any optimisation), measure the speed of every variant even when its quality fails or is broken. Never skip, defer or drop a timing because quality broke. Record quality and speed side by side, and label a broken-quality row as such, but still time it. If a timing truly cannot run (the build is refused, or it crashes), report it as NOT RUN with the exact error, never "skipped because of quality". Pass this rule verbatim into every subagent and workflow brief.

Each JSONL row names an uncompressed PCM WAV relative to the fixture file and the expected text:

```json
{"id":"read-01","audio":"clips/read-01.wav","reference_text":"The meeting starts at ten.","language":"en","unseen":true}
```

The client hashes exact WAV bytes and references. Only the WAV, model, language and temperature go to
`POST /v1/audio/transcriptions` as multipart, with `response_format=json` and optional `stream=true`.
The reference text stays with the benchmark. The JSON response returns `text`. Streaming sends
`transcript.text.delta` events with `delta`, followed by `transcript.text.done` with the final `text`.
The JSON response or terminal SSE event supplies an `effective` object:

```json
{"model":"example/model","model_revision":"pinned-checkpoint","language":"en","dtype":"fp32","quant":"none","engine_version":"pinned-engine","temperature":0,"audio_duration_s":1.2,"cache":false}
```

Supply actual effective settings from the server or an adapter to the server's pinned launch. Include
`beam_size`, `vad_filter` and `chunking` when the plan declares them. inferencekit sends temperature and
language per request. Configure the other decoding knobs at launch. A plain OpenAI-compatible server
without the effective extension can still return diagnostic timings, but its claim validity fails.

```bash
inferencekit quality stt-policy --policy stt-policy.json --fixtures speech.jsonl --run-dir runs/stt
inferencekit bench stt --base-url http://127.0.0.1:8000 --fixtures speech.jsonl \
  --model example/model --model-revision pinned-checkpoint --model-dtype fp32 --model-quant none \
  --engine-version pinned-engine --hardware-name 'Exact CPU model' --device cpu --rounds 1
inferencekit bench rounds --mode stt --manifest runs/stt/manifest.json --prompts speech.jsonl \
  --rounds 4 --warmup 0
inferencekit compare BASELINE_ID CANDIDATE_ID --metric rtf
inferencekit quality stt --protocol POLICY_ID --baseline BASELINE_ID --candidate CANDIDATE_ID \
  --fixtures speech.jsonl --run-dir runs/stt
inferencekit report COMPARISON_ID QUALITY_ID --format md
```

The manifest uses `workload.modes: ["stt"]`, each engine's `endpoint_kind: "stt_http"`, and
`workload.quant` plus model and engine pins. Use `workload.stream: true` to time SSE
or set `--stream` on direct calls. To warm up, set a positive `--warmup` and pass
`--warmup-fixtures warmup.jsonl` through `--serve-args`. Use WAVs separate from the measured fixtures.
Get the per-engine benchmark IDs from `plan.stt_evidence_final`. The scheduler attempts every live
engine even when transcript validation fails, and invalid rows remain visible with diagnostic timing.
The default `--save-outputs all` retains text for quality and review. Check these artifacts before sharing.

The client measures complete HTTP response time. Streaming additionally measures arrival of the first
nonempty delta. `stt_ms` is mean request completion time per round, `first_partial_p95_ms`
is the streaming p95, and `rtf` is summed wall time divided by summed input WAV duration.
`audio_s_per_s` is its reciprocal. Failure rows retain elapsed time and retain completion
time when the client consumed the full response. Invalid rows cannot support a speed claim.

Run `inferencekit gates stt` with the same fixture, model and endpoint arguments on each live engine.
The probes repeat audio under distinct request IDs, add seeded low-level noise, compare held-out
and tuning WER, and check effective decode settings and duration. An optional cross-language probe
records a warning using server-reported detection confidence. An `unseen` flag attests that a
fixture was held out. Use both seen and unseen inputs for the held-out check.

For device-side engines, `inferencekit import stt --help` describes the import. The JSON
document uses `format: "inferencekit.import.stt.v1"`, `timer`, capture start/end Unix
seconds (`capture_start_unix_s`, `capture_end_unix_s`) and `rows`. Each row has `id`,
`round`, `audio_sha256`, `elapsed_ms`, `complete_ms`, `text`, `effective` and optional
`first_partial_ms` or `error`. Retain failed rows with their elapsed time. Device timings
and reported settings are capture attestations. Quality can use these transcripts, and the
HTTP client speed comparison requires its own capture.
