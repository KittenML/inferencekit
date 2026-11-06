# Speech quality evidence

`quality audio` compares the complete PCM saved by `bench audio` against a named baseline. It produces `quality_audio` evidence for reports. The core uses only CPU arithmetic and imported observations; it does not install, download or run ASR, MOS or speaker models.

The acceptance policy is explicit:

| Policy | Required evidence | Meaning of PASS |
|---|---|---|
| `exact` | Identical PCM, frame counts, rate and channels | Preservation of the named baseline outputs |
| `task_noninferiority` | Paired WER and predicted naturalness; optional speaker metric | Noninferiority under the registered task/proxy metrics |
| `perceptual_and_task_noninferiority` | Task metrics plus blinded, paired listening ratings | Noninferiority under the registered task and listening protocol |

A baseline's intrinsic intelligibility or suitability is not established by preserving it. Predicted MOS, ASR and speaker scores do not substitute for listening. Timing, lifecycle, memory and sustained reliability remain separate requirements.

## Register before capture

Create a JSON policy and register it before either benchmark begins. For an exact-output refactor:

```json
{"acceptance":"exact","min_fixtures":2,"min_rounds":1,"resamples":2000,"seed":0}
```

```bash
inferencekit quality audio-policy --policy audio-policy.json --fixtures speech.jsonl --run-dir runs/speech
# Retain the returned POLICY_ID, then capture both benchmark arms with --save-outputs all.
inferencekit quality audio --baseline BASELINE_ID --candidate CANDIDATE_ID \
  --fixtures speech.jsonl --protocol POLICY_ID --run-dir runs/speech
inferencekit report COMPARISON_ID QUALITY_ID --format md
```

The registration binds fixture text, reference-voice content, acceptance, margins, scorer configuration and bootstrap plan. It must precede every measured round, including rounds retained by `--extend`. A late policy produces exploratory results with an UNPROVEN acceptance verdict. The original evidence is never rewritten.

Source arms must have matching fixture IDs, reference content, round indices, seeds, model revision, generation settings, PCM rate and channels. Different numerical formats require `allow_numerics_change: true` in the policy. This permission does not authorize a different model or generation task. Model execution or hardware spending requires its own authorization.

## Import task observations

For task noninferiority, supply `metrics` and pinned `scorers` in the policy. These margins are illustrative, not recommended universal tolerances; register values justified for the actual workload.

```json
{
  "acceptance": "task_noninferiority",
  "min_fixtures": 20,
  "min_rounds": 1,
  "resamples": 2000,
  "seed": 0,
  "normalization": "nfkc_casefold_punctuation_v1",
  "metrics": {
    "wer": {"margin": 0.01, "outlier_wer": 1.0},
    "naturalness": {"margin": 0.1}
  },
  "scorers": {
    "asr": {
      "name": "faster-whisper", "version": "PINNED_PACKAGE_VERSION",
      "model": "LOCAL_MODEL_ID", "revision": "CHECKPOINT_CONTENT_HASH",
      "config": {"device": "cpu", "compute_type": "int8", "language": "en",
                 "temperature": 0, "beam_size": 5, "vad_filter": true,
                 "condition_on_previous_text": false}
    },
    "naturalness": {
      "name": "UTMOSv2", "version": "PINNED_CODE_REVISION",
      "model": "fusion_stage3", "revision": "CHECKPOINT_CONTENT_HASH",
      "scale": [1, 5],
      "config": {"predict_dataset": "somos", "fold": 0, "num_repetitions": 5,
                 "seed": 0, "preprocessing": "DECLARE_RESAMPLING_DOWNMIX_AND_WINDOW_RULE"}
    }
  }
}
```

Use local, pinned scorer implementations in their own environment. Faster Whisper returns a segment generator: consume it to finish transcription, and retain the concatenated transcript and actual decoder settings. Its official implementation supports VAD and independent-window decoding. UTMOSv2 exposes dataset/domain selection, checkpoint, fold and repeated predictions; retain these settings and any preprocessing in `config`. See [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) and [UTMOSv2 inference](https://github.com/sarulab-speech/UTMOSv2/blob/main/docs/inference.md). Avoid an unpinned `torch.hub` download during scoring.

Import one observation per arm, fixture and round. `scorers` must exactly equal the registered descriptor object. Use hashes and seeds from the source benchmark's `requests.jsonl`; bind scores to the original PCM even when the scorer resamples it internally.

```json
{
  "format": "inferencekit.quality.audio.scores.v1",
  "protocol": "POLICY_ID",
  "baseline": "BASELINE_ID",
  "candidate": "CANDIDATE_ID",
  "scorers": {"asr": "COPY_REGISTERED_DESCRIPTOR", "naturalness": "COPY_REGISTERED_DESCRIPTOR"},
  "rows": [{
    "arm": "candidate", "id": "one", "round": 0, "seed": 123,
    "pcm_sha256": "HASH_FROM_REQUEST_ROW", "text_sha256": "HASH_FROM_REQUEST_ROW",
    "asr": {"text": "The meeting starts at ten tomorrow morning."},
    "naturalness": {"windows": [
      {"start_frame": 0, "end_frame": 44100, "score": 4.0},
      {"start_frame": 44100, "end_frame": 66150, "score": 3.8}
    ]}
  }]
}
```

The shown row and descriptor placeholders are a template, not a complete score file. Include both arms and every measured fixture/round. A scorer failure may retain an `error` string and omit the affected metric; missing required observations remain UNPROVEN. Unknown or duplicate identities, different scorer pins, changed PCM/text hashes, nonfinite scores and invalid windows are rejected.

```bash
inferencekit quality audio --baseline BASELINE_ID --candidate CANDIDATE_ID \
  --fixtures speech.jsonl --protocol POLICY_ID --scores audio-scores.json --run-dir runs/speech
```

WER is recomputed as word edit distance divided by reference words, without a cap. `nfkc_casefold_punctuation_v1` applies Unicode NFKC, case folding, punctuation-to-space and whitespace splitting; `whitespace_v1` only splits whitespace. Neither expands numbers or supplies language-specific tokenization. Choose a suitable declared protocol for the fixture language; empty normalized references are rejected.

Naturalness windows use original PCM frame coordinates, must be contiguous and cover the whole output exactly once, including the final short window. Scores are weighted by window duration within an utterance. The same method and preprocessing apply to both arms. Scorer provenance is an attestation by the importer: hashes establish binding and reproducibility inputs, not that the external model or listening study was honestly run.

## ASR outliers

Register `wer.outlier_wer` before decoding. When either arm exceeds it, the affected pair needs a corroborating decode of both arms. Register an `asr_retry` descriptor with VAD enabled and `condition_on_previous_text: false`; supply each retry as `asr.retry_text`. Retain both transcripts. The policy does not choose whichever transcript has lower WER.

The paired primary result remains visible. The corroborating result substitutes both affected arms for a second calculation; disagreement between verdicts, or a missing corroborating decode, makes the WER check UNPROVEN. A corroborated regression remains FAIL. High waveform SNR never waives WER. Structural failures and failures of other required quality checks remain FAIL regardless of ASR uncertainty.

## Listening and speaker scores

To require listening, use `perceptual_and_task_noninferiority`, register `metrics.listening` with `margin` and `min_raters` (at least two), and add a `scorers.listening` descriptor with `name`, `version`, `protocol_sha256`, `scale: [1,5]` and `config: {"blinded":true}` plus the study's actual configuration. Each observation supplies `listening.ratings` as `[{"rater":"anonymous-id","score":4.0}, ...]`. The same anonymous panel must rate both arms of each pair. The protocol hash binds the externally retained listening study design; blinding remains an attestation.

Speaker similarity is optional. Register `metrics.speaker.margin` and a scorer with pinned model/version/revision, preprocessing config and increasing score scale (larger means more similar). Each row supplies `speaker: {"score":0.8,"reference_sha256":"FIXTURE_REFERENCE_HASH"}`. A fixture voice reference is required. For an example local scorer, see the publisher's [SpeechBrain ECAPA model card](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb).

## Statistics and reports

The paired bootstrap resamples fixtures, retaining all rounds within each drawn fixture. WER pools edit and reference-word counts; naturalness, speaker and listening scores average per-utterance values. Listening inference is conditional on the supplied listener panel. The CI is not a power guarantee: register an adequate fixture count for the intended margin.

For WER the worse-direction difference is candidate minus baseline; for larger-is-better scores it is baseline minus candidate. PASS requires the entire CI inside the registered degradation margin; FAIL requires the entire CI above it; a straddling CI is UNPROVEN. Incomplete coverage or fewer than the registered fixture/round counts cannot PASS. Exact policy also requires bit-identical PCM for every pair; non-exact policies retain parity as diagnostics.

SNR is `10*log10(sum(baseline²)/sum((baseline-candidate)²))`; maxabs is the largest signed-sample difference divided by 32768. Both use unmodified, aligned PCM. Unequal lengths have no SNR/maxabs value. Identical PCM has `snr_status: "identical"` and a null numeric SNR, avoiding a fabricated finite or nonfinite JSON value.

The retained `audio_score.json` lists per-pair parity, source IDs, policy, coverage and checks; `imported_scores.json` retains external observations. Report comparison quality can use only evidence bound to its exact baseline/candidate IDs. Proxy acceptance remains labelled as such; missing/changed observations become UNPROVEN, previous FAIL records remain visible, and lifecycle/reliability requirements are not satisfied by quality evidence.
