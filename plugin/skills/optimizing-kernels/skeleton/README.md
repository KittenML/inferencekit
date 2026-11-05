# Engine skeleton (`ikengine`)

A small inference engine written for correctness first: plain PyTorch, one request at a time, an
OpenAI-compatible server, and the checks that prove a faster version still computes the same thing.
It has no fast kernels. It computes exactly what the checkpoint declares, slowly, so you have a
working, checkable server on day one and a reference to test every kernel against.

Your job, when you build a new engine from it: make the model fast behind the same contract, and
keep the proofs passing.

## What is inside

| File | What it owns |
|---|---|
| `ikengine/engine_api.py` | The contract between server, engine and model. Do not change it. |
| `ikengine/serve.py` | The HTTP server: OpenAI API, chat template, tokenizer, queue, stop rules, ENGINE.json. Leave it alone. |
| `ikengine/engine.py` | The generic engine over any model: prefill, decode loop, sampling, logprobs. |
| `ikengine/graph_loop.py` | The decode step (model + next-token choice + position) captured in a CUDA graph. |
| `ikengine/sampler.py` | Greedy, temperature, top-k, top-p, min-p and seeded draws, all on the GPU. |
| `ikengine/loader.py` | Reads a checkpoint: config, quantization scheme, every tensor as stored. |
| `ikengine/qlinear.py` | Reference linear layers for 16-bit, block FP8 and NVFP4 weights, plus `dequant()`. |
| `ikengine/ref_dense.py` | A config-driven dense decoder (Llama, Mistral, Qwen2, Qwen3) and its building blocks. |
| `ikengine/moe.py` | Mixture-of-experts routing and an expert MLP that works inside a CUDA graph. |
| `ikengine/models/` | One module per model family, chosen by config.json's `architectures` (Gemma 4, Qwen3.5/3.8, dense). |
| `ikengine/tools/validate_model.py` | GPU check: greedy tokens and top-20 logprobs against a reference, plus speed. |

## Run it

The skeleton needs PyTorch with CUDA, `safetensors` and `transformers` (for the tokenizer and chat
template). There is nothing to build.

```bash
mkdir -p /work/run/engine
cp -r <plugin dir>/skills/optimizing-kernels/skeleton/ikengine /work/run/engine/
cd /work/run/engine
python -m ikengine.serve --model /path/to/checkpoint --port 8000 --served-model-name my-model
```

Useful options: `--chat-template FILE`, `--default-chat-template-kwargs '{"enable_thinking": false}'`,
`--max-seq N` (the KV cache size), `--max-queue N`, and the numerics options `--act` and `--kv`
(below), and `--device` (default `cuda`). At startup the server captures the decode graphs, runs a short
warm-up, prints its resolved configuration and writes `ENGINE.json` at the top of the git repository it
runs in (here /work/run/ENGINE.json), with a launch command that works from a fresh copy of the
repository, no .git needed. `--engine-json PATH` moves the file and `''` skips it.

Then check it like any other engine:

```bash
python -m ikengine.tools.validate_model --model /path/to/checkpoint --ref http://REFERENCE:8000
inferencekit gates run --base-url http://127.0.0.1:8000 --model my-model
```

If the reference server needs a key, put it in the `OPENAI_API_KEY` environment variable (or name
another variable with `--api-key-env`). The key is never passed on the command line or written out.
For an instruction-tuned model, add `--chat` (and `--chat-template-kwargs '{"enable_thinking": false}'`
if the baseline uses it, and `--chat-template FILE` if the server was given one) so the prompts go
through the chat template the way the server sends them.

Then quality against the reference, as for any engine
([proving-correctness](../../proving-correctness/SKILL.md)): `inferencekit quality trajectory` on this
server (`--engine-kind vllm`: it honours vLLM's request fields) and `inferencekit quality score` against
the reference's noise floor; when the workload times sampling, also `inferencekit quality sampling`
on this server (its sampled first tokens against its own top-20 log-probabilities).

`validate_model` decodes greedily on 8 built-in prompts (or yours: a text file, or JSONL with `prompt`
or `messages`, such as the inferencekit dev set with `--per-bucket 8`), asks the reference for its own
greedy tokens and for its logprobs teacher-forced on the engine's tokens (vLLM `prompt_logprobs`,
SGLang's native `/generate`, or Hugging Face), and prints one line: identical greedy paths, each first
divergence with the gap between the two tokens on both sides (a near tie when every gap is at most
`--tie-nats`, 0.5 by default), top-1 agreement (also counting exactly tied 16-bit logits), mean KL and
decode tok/s. It exits non-zero when a threshold fails. `--ref hf` uses Hugging Face transformers on
the same GPU instead of a server (the engine is freed first, so either one fitting is enough).
`--ref none --save-engine FILE`, then `--load-engine FILE` with the reference, let the engine and a
reference server take the GPU one at a time. Set the thresholds from the reference's own noise:
`--ref-aa` compares its decode path with its own teacher-forced path on the same tokens, and a second
greedy run with the first (run to run).

## The numerics rule: as shipped, or declared

The skeleton computes what the checkpoint declares:

- **Weights and scales are used as stored.** Nothing is re-rounded at load time. A block-FP8 product
  is formed per block in fp32 and scaled there; the weight times its scale is never rounded into a
  16-bit weight first.
- **Activations follow the declared scheme.** Block FP8 (`quant_method: fp8` with
  `weight_block_size`): each token's activations are quantized to e4m3 per group of 128 with
  scale = amax / 448, as vLLM's block-FP8 path does. NVFP4 from NVIDIA ModelOpt (`hf_quant_config.json`
  or a `config_groups` quantization config): activations are quantized to FP4 (e2m1, round to nearest
  even) with an e4m3 scale per 16 values under the checkpoint's static `input_scale`, so the layer is
  W4A4, as vLLM runs it. `qlinear.py` cites the vLLM functions it follows.
- **The KV cache uses the declared dtype**: e4m3 with the stored k/v scales when the checkpoint
  declares an FP8 KV cache (scale 1.0 when the checkpoint stores none, as vLLM does), else the model
  dtype.

`--act a16` (16-bit activations) and `--kv bf16` (16-bit KV cache) are allowed changes. The model
declares them in its `numerics`, ENGINE.json says "numerics changed", and the result has to show its
quality against the 16-bit reference model is no worse than the baseline's. Any other change (lower
precision, dropped scales, different rounding) is also a numerics change and is declared the same way.

Logprobs are always the log-softmax of the raw logits over the full vocabulary, before temperature,
top-k or top-p, as vLLM returns them.

## Make it fast

Change the model's internals; keep the contract, the server and the checks.

1. **Work inside `Model.decode`.** It takes a token and a position (both on the GPU) and writes the
   logits into a buffer it owns. It must stay CUDA-graph capturable: fixed shapes, no host
   synchronisation, no Python branches on tensor values. `graph_loop.py` captures it with the
   sampler, so a capturable decode gets graphs for free. Check `"cuda_graphs": true` in the startup
   line; if capture failed, `graph_error` says why and the engine runs eagerly.
2. **Replace references with kernels one at a time**, behind the same function signature, and test
   each against the reference it replaces (`qlinear`, `attend`, `rms_norm`, `moe.expert_mlp`) on real
   weights and activations before you time it. The usual order for single-user decoding: the
   quantized GEMVs (weights dominate the bytes), attention decode, then fusing norms, RoPE and
   residual adds to cut launches. See [fp8-gemv.md](../references/fp8-gemv.md),
   [launch-overhead.md](../references/launch-overhead.md),
   [attention-split-k.md](../references/attention-split-k.md), [prefill.md](../references/prefill.md)
   and [lessons.md](../references/lessons.md). For MoE decode see
   [moe-decode.md](../references/moe-decode.md); for exact byte cuts (packing, a bound-pruned output
   layer) see [byte-cuts.md](../references/byte-cuts.md).
3. **A pruned output layer is for plain decoding only.** The engine computes logprobs from the logits
   `decode` returns, so a request with logprobs must take the full output layer; route it there. Give
   pruned rows a finite value below the kept threshold (their bound), never -inf: a faster sampler that
   estimates its top-k cut-off from the values can then still find it ([byte-cuts.md](../references/byte-cuts.md)).
4. **A faster sampler must make the same choices**: same ties at the top-k boundary, same top-p rule,
   same noise per token id for a given seed. Prove it against `sampler.py` with fixed noise.
5. **Keep the proofs passing after every change you keep**: `validate_model` against the reference,
   your kernel-vs-reference tests, the gates, and the quality checks.

The reference is slow by design: every quantized linear decodes its weights to fp32 on every call,
and attention reads the whole KV cache each step. Expect it to be far below a tuned engine; measure
it before you start so the gains you report have a real starting point.

## What the server promises

The server meets the custom-engine contract the inferencekit checks expect (see
[candidates.md](../../optimizing-inference/references/candidates.md)):

- OpenAI `/v1/chat/completions` and `/v1/completions`, streaming one chunk per token, with usage and
  token ids; `/health`, `/v1/models`, `/tokenize`, `/detokenize`, `/engine/config`.
- Sampling as vLLM: temperature, top_k, top_p, min_p and seed. Fields a request leaves out take the
  model's `generation_config.json` defaults, so a request with no sampling fields samples the way the
  baseline does. The same seed gives the same tokens. Temperature 0 is greedy, ties to the lowest id.
- Logprobs: the chosen token plus up to 20 alternatives, from the full vocabulary.
- Chat template defaults from `--default-chat-template-kwargs`; a request's own kwargs win.
- One request runs at a time and the rest wait in a first-in, first-out queue. `/get_server_info`
  reports this, so `inferencekit quality gsm8k --concurrency 0` picks a matching concurrency.

## Add or change a model family

A model module is a file in `ikengine/models/` that registers the Hugging Face architecture names it
implements:

```python
from . import register

@register("MyModelForCausalLM")
class MyModel:
    @classmethod
    def from_checkpoint(cls, ck, device, max_seq, act="as_shipped", kv="as_shipped"): ...
    def alloc_state(self, max_seq): ...       # KV cache, recurrent state, static buffers
    def reset(self, state): ...
    def prefill(self, input_ids, state): ...  # fp32 logits of the last prompt position
    def decode(self, token, pos, state): ...  # fp32 logits in a buffer owned by state
```

It also sets `vocab_size`, `max_seq`, `eos_token_ids` and `numerics`. Build it from the pieces here:
`Checkpoint.linear(prefix)` for every weight as stored, `qlinear` for every product, `ref_dense`'s
`rms_norm`, `apply_rope`, `rope_inv_freq` and `attend`, and `moe.route` with `moe.ExpertStack` for
experts. Check it against Hugging Face on a tiny random config (see how the repository's tests do it)
and then on the real checkpoint with `validate_model`.

## Limits

- One request at a time (batch 1). There is no continuous batching.
- `ref_dense.py` covers Llama, Mistral, Qwen2 and Qwen3 dense decoders. Gemma 4 and the Qwen3.5/3.8
  hybrid family have their own modules in `models/`. Dynamic-NTK and LongRoPE position scaling are not
  implemented and raise an error rather than run with other frequencies.
- Checkpoint formats: 16-bit, block FP8, ModelOpt NVFP4 and ModelOpt FP8. Checkpoints in the
  compressed-tensors format store their scales differently and are refused rather than misread.
- The reference runs on any CUDA GPU that PyTorch supports: FP4 math is emulated in fp32, so it needs
  no FP4 hardware. Any kernel you add states which GPU architectures it supports.
