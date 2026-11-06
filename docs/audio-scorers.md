# Local speech scorers

Use `quality audio-score` to transcribe saved speech with Faster Whisper, predict naturalness with UTMOSv2, and optionally compare a voice with SpeechBrain ECAPA. It writes observations for [`quality audio`](audio-quality.md), which computes WER/CER, paired confidence intervals and the registered acceptance verdict. Scoring runs after the timed benchmark.

For output-preserving changes, start with the exact PCM policy: no scorer models are needed. When outputs change, register task metrics and margins before capture. Predicted naturalness and speaker similarity are proxies; keep listening comparisons alongside them.

## Prepare the scorer environment

Keep scorer dependencies in a separate Python environment. The CLI itself does not depend on PyTorch or any scorer package. Install pinned versions of Faster Whisper, UTMOSv2 and its dependencies, and SpeechBrain only if speaker checks apply. Download the matching checkpoints separately before scoring:

| Scorer | Local assets |
|---|---|
| Faster Whisper | Converted model directory, tokenizer and configuration |
| UTMOSv2 | Fusion checkpoint, Wav2Vec2 directory and EfficientNet weights |
| SpeechBrain ECAPA | Model directory including hyperparameters, embedding weights and normalization files |

Use the upstream [Faster Whisper setup](https://github.com/SYSTRAN/faster-whisper), [UTMOSv2 inference guide](https://github.com/sarulab-speech/UTMOSv2/blob/main/docs/inference.md) and [ECAPA model card](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb). Model licences and scorer applicability remain those of their publishers. Scoring requires local files and does not download checkpoints.

Example CPU environment used for the Linux smoke checks:

```bash
uv venv --python 3.11 .venv-speech
uv pip install --python .venv-speech/bin/python \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0
uv pip install --python .venv-speech/bin/python \
  faster-whisper==1.2.1 speechbrain==1.0.3 transformers==4.49.0 \
  timm==1.0.15 librosa==0.11.0 numpy==1.26.4 huggingface-hub==0.30.2 \
  'utmosv2 @ git+https://github.com/sarulab-speech/UTMOSv2.git@cc2700db57bb83ee13dc31ebe1b868c254e15d09'
```

Retain your resolved environment lock with the run. The descriptor records installed versions and hashes; a package version alone does not pin its dependencies. Installation and checkpoint download require network access; the scoring phase does not.

Save a configuration such as `audio-scorers.json`, replacing paths with your installed environment and assets:

```json
{
  "format": "inferencekit.quality.audio.scorers.v1",
  "python": "/path/to/scorers/bin/python",
  "threads": 1,
  "seed": 0,
  "batch_size": 4,
  "timeout_s": 3600,
  "scorers": {
    "asr": {
      "backend": "faster-whisper",
      "model": "/path/to/faster-whisper-model",
      "settings": {
        "language": "en", "beam_size": 5, "compute_type": "int8",
        "vad_filter": false, "condition_on_previous_text": false
      }
    },
    "naturalness": {
      "backend": "utmosv2",
      "model": "/path/to/fold0_s42_best_model.pth",
      "assets": {
        "facebook/wav2vec2-base": "/path/to/wav2vec2-base",
        "tf_efficientnetv2_s.in21k_ft_in1k": "/path/to/efficientnet-model.safetensors"
      },
      "settings": {
        "model_name": "fusion_stage3", "window_seconds": 3,
        "fold": 0, "checkpoint_seed": 42, "predict_dataset": "sarulab",
        "num_repetitions": 1
      }
    }
  }
}
```

Choose the ASR language and UTMOS domain for your workload. Both arms use the same configuration. The adapters hash backend code and model assets and record dependency versions for registration; changing these inputs requires a new registered policy and fresh measurements.

## Register, measure, score

1. Run `inferencekit quality audio-scorers audio-scorers.json`. Copy its `scorers` object into your [quality policy](audio-quality.md#import-task-observations), then register it with `quality audio-policy` before capturing either arm.
2. Use a fixed representative fixture set and [`bench rounds --mode audio`](media-benchmarks.md#comparisons-and-proof). Retain complete PCM. The benchmark reports streaming onset, completion, realtime factor and available memory measurements separately from quality.
3. Stop the benchmark engines before running the scorers, so quality work cannot compete with timed requests. Use the final baseline and candidate evidence IDs from the rounds plan.

```bash
inferencekit quality audio-score --config audio-scorers.json \
  --baseline BASELINE_ID --candidate CANDIDATE_ID \
  --fixtures speech.jsonl --protocol POLICY_ID --out audio-scores.json
inferencekit quality audio --baseline BASELINE_ID --candidate CANDIDATE_ID \
  --fixtures speech.jsonl --protocol POLICY_ID --scores audio-scores.json \
  --run-dir runs/speech
inferencekit report COMPARISON_ID QUALITY_ID --format md
```

The score file binds each observation to its source PCM, text, seed, fixture and round. Models are reused within a scoring invocation. Missing or failed observations cannot become a quality PASS. Keep the observations with the evidence; they contain transcripts and are not automatically safe to publish.

## Optional checks

- **CER:** add `metrics.cer` to the policy, alongside WER or instead of it. It reuses ASR transcripts and counts normalized Unicode code points with whitespace removed. See [normalization and margins](audio-quality.md#import-task-observations).
- **ASR outliers:** register an `asr_retry` scorer using Faster Whisper with VAD enabled and `condition_on_previous_text: false`. Retain both decodes; the quality evaluator does not choose the best transcript.
- **Voice conditioning:** add a `speaker` scorer with `backend: "speechbrain-ecapa"`, the local model directory and `settings: {}`; register a speaker margin. It compares output with each fixture's `reference_audio`. Omit it when no suitable reference exists. A similarity score does not establish voice identity or consent.
- **Listening:** retain blinded, paired listener ratings under the registered listening protocol. Add them to the generated rows using the [listening schema](audio-quality.md#listening-and-speaker-scores), preserving hashes, descriptors and identities. No adapter generates human ratings.

UTMOS windows cover the complete output, including the tail, in original PCM coordinates. The quality evaluator weights their scores by duration. Native seeded spectrogram views and short-input tiling are retained; identical PCM windows reuse one prediction. Scorer resampling and downmixing apply only to model inputs; PCM preservation always compares the original bytes. These checks assess the registered fixture set, not every possible utterance or voice.
