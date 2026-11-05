# Examples

A few things to ask for. These are starting points: you can also just describe your problem in plain words ("TTFT got worse after the vLLM upgrade", "which engine should I use for this model?", "is this benchmark believable?") and the plugin picks the right skill.

Commands are shown for Claude Code. In OpenCode, drop the `inferencekit:` prefix (`/optimize-inference`).

## Make a model faster

```text
# Fastest single-user decode
/inferencekit:optimize-inference Qwen/Qwen3-8B-FP8 on 1x RTX 5090, fastest single-user decode, by 6h

# Most users at once while time to first token stays low (N* = how many users it serves at once within that limit)
/inferencekit:optimize-inference Qwen/Qwen3-8B-FP8 on 1x RTX 5090, N* >= 32 at p95 TTFT <= 300 ms, lengths: 900/200

# Cheapest serving for your traffic
/inferencekit:optimize-inference Qwen/Qwen3-8B-FP8 on 1x RTX 5090, lowest $/1M tokens, arrival rate: 4 req/s, lengths: 900/200

# Video
/inferencekit:optimize-inference Lightricks/LTX-2.5-Diffusers on 1x RTX PRO 6000, fastest clip
```

The agent compares and tunes supported engines for your workload, then profiles and implements custom optimizations if requested. It asks only for missing preferences and spending authorization, reusing answers you already gave. A plan is optional.

## Contribute a completed engine

You can opt in up front: “When the optimization is finished, if the result meets the engines repository's contribution requirements, open a public PR to KittenML/inferencekit-engines with the engine source and shareable evidence. You may create a fork if needed.”

Otherwise, the agent prepares a qualifying contribution locally and asks before publishing. Unattended runs leave it awaiting permission unless already authorized. The PR describes the measured improvement and quality limits; it is not automatically merged. [Contribution workflow](../plugin/skills/optimizing-inference/references/engine-contribution.md).

## Measure what you already have

```text
# Speed of a running server, one user at a time
/inferencekit:bench http://127.0.0.1:8000 Qwen/Qwen3-8B-FP8

# How many users it can serve under a latency limit
/inferencekit:bench http://127.0.0.1:8000 Qwen/Qwen3-8B-FP8 --capacity --slo-ttft-p95-ms 300

# Engine against engine, same hardware, same prompts
/inferencekit:bench http://127.0.0.1:8000 Qwen/Qwen3-8B-FP8 --vs "python -m sglang.launch_server --model-path Qwen/Qwen3-8B-FP8"
```

## Find out why

```text
# Where the time goes, kernel by kernel
/inferencekit:profile trace.nsys-rep

# Is a quantized model good enough?
Is the NVFP4 build of Gemma 4 26B good enough on my RTX 5090, compared with BF16?

# Check a speed claim before you trust it
/inferencekit:perf-report "Our engine is 2x faster than vLLM on an H100"
```

For a repeatable setup, write your workload to a file once and reuse it: `inferencekit intake template --out intake.yaml`, then `/inferencekit:optimize-inference intake.yaml`.

These cover the common cases. The plugin can do much more: TTFT and prefill tuning, regressions after an upgrade, KV-cache and batching settings, speculative decoding, roofline ceilings for your hardware, and turning results into a report with PASS/FAIL verdicts. Ask in plain words.
