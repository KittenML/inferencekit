# On-device speech and concurrency

Identify the phone model, SoC, OS build, runtime and actual CPU, GPU or NPU placement. For Android gather `adb shell getprop`, process affinity and `adb shell dumpsys thermalservice` before and after each cell. Read VmHWM or PSS separately with the process and measurement method identified. For iOS use the selected device's OS and runtime facts plus Instruments or native timers when available. Report inaccessible counters as NOT RUN.

Run the client and engine on the phone. With Android device labs, push a native arm64 test or start an app with ADB, collect its artifacts, and use the same phone for the baseline and candidate. A forwarded HTTP endpoint adds transport time. Label native measurements by where the timer starts and ends, and verify settings, operator placement and process isolation from runtime logs or vendor tools.

## Capacity under speech SLOs

- Pin the model, precision, voice, seed policy, text and sentence boundaries. Start with sentence streaming for interactive output. Benchmark whole-utterance delivery on the same text and report its first audio and complete latency. Native smaller chunks can be a separate workload.
- Set the first-audio and RTF limits before testing. Schedule numbered waves of distinct fixtures at fixed times regardless of previous completion. At each level dispatch the stated number of requests together. Use the same ordered fixture set and period across configurations. Count time from scheduled release, including queueing. Warm up each fresh process outside timing.
- In each measured round all stream slots must meet both p95 limits. Simulate playback starting at first PCM with no extra buffer and require zero underruns. Track first PCM separately from the first nonzero playback sample. A response running beyond its wave period fails that level.
- Start at one stream and measure consecutive levels, including the first failing level above a contiguous passing prefix. A screen can identify a candidate; repeat with balanced fresh-process rounds, enough requests per stream to characterize the tail and a sustained thermal check. If the top measured level passes, report a lower bound. Report the wave period and output audio seconds per second of scheduled wall time.
- Establish completion and speech quality on saved output from the exact configurations under test. Native timing import does not attest to fixture contents, correctness, actual audio hardware playback, or the capture implementation.

Use `hardware.device`, `workload.client: audio_capacity` and `goals.lead: capacity` in the intake. Set `goals.slo.first_audio_p95_ms`, `rtf_p95` and `max_underruns: 0`.

Register the plan before timing. Fields are `levels` (consecutive from 1), `rounds`, `waves`, `period_ms`, `sample_rate`, `mode` (`stream` or `full`), `chunks_per_request` (1 for full), `fixture_ids` and `fixture_sha256` (digest of the fixture file). List one distinct ID per wave and stream at the largest level. For each run use fixture index `wave * concurrency + stream`.

Export one JSONL row per request with zero-based `round`, `wave`, `stream`, and fields `concurrency`, `id`, `complete_ms`, and `chunks`. Each chunk is an array of arrival milliseconds since scheduled release, PCM frame count and leading silent frame count. Use null completion for failed requests. Retain a separate record of the process launches, resolved settings, thermal state and capture code.

```bash
inferencekit import audio-capacity --timings requests.jsonl --plan plan.json --fixtures fixtures.tsv --label candidate \
  --model MODEL --model-revision REV --engine-version VER --hardware-name DEVICE \
  --slo-first-audio-p95-ms 400 --slo-rtf-p95 1
```
