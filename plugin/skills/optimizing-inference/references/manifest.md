# manifest.json and reproduce.sh

`runs/<run-id>/manifest.json` is what anyone re-runs to check your numbers: on a clean host, with the same harness wheel, each engine's `install` commands, then the same measurements (ideally on prompts you never saw). **Your numbers are agent-measured until such a re-run matches them.** An engine that cannot be installed or launched from the manifest alone cannot be re-run, so it has no reproducible number.

## Shape (contract; validated by `inferencekit manifest validate`)
```json
{
  "manifest_version": 1,
  "model": {"id": "org/name", "revision": "<40-hex sha>", "precision": "nvfp4", "tokenizer": "org/name",
            "local_dir": "models/<dir>"},
  "hardware": {"gpu": "RTX 5090", "count": 1, "price_usd_h": 0.6},
  "engines": [
    {"name": "vllm", "label": "vllm-recipe", "version": "x.y.z",
     "install": ["uv pip install vllm==x.y.z"],
     "launch": ["vllm", "serve", "models/<dir>", "--port", "8001", "--seed", "0", "--no-enable-prefix-caching"],
     "env": {}, "port": 8001, "ready_check": "/v1/models",
     "resolved_config": {"quantization": "...", "kv_cache_dtype": "auto", "attention_backend": "...",
                         "cudagraph_mode": "...", "max_num_seqs": 1, "source": "startup log lines ..."},
     "role": "baseline-recipe"},
    {"name": "vllm", "label": "vllm-tuned", "...": "...", "role": "tuned"},
    {"name": "myengine (vLLM-compatible request fields)", "label": "myengine-v3", "...": "...", "role": "custom"}
  ],
  "workload": {"modes": ["single"], "buckets": ["very_short", "short", "medium", "long"], "max_tokens": 256,
               "chat_template_kwargs": {"enable_thinking": false},
               "extra_body": {"temperature": 0.7, "top_k": 50, "top_p": 0.9, "seed": 1234},
               "slo": {"ttft_p95_ms": 500, "tpot_p95_ms": 50, "max_fail": 0}},
  "harness": {"inferencekit_version": "0.1.0", "wheel_sha256": "<sha256 of the wheel you ran>",
              "dev_prompt_set_sha256": "<from inferencekit prompts verify>"},
  "agent_run": {"run_id": "<run-id>", "started": "<ISO>", "ended": "<ISO>"}
}
```

## Rules
- **One entry per configuration you report:**
  - the baseline recipe and the best tuned config for each supported engine;
  - the capacity config, when load mode ran;
  - every custom or patched engine.

  Labels are unique; ports are distinct.
- **Custom engines** (`role: custom`) meet the [custom-engine contract](candidates.md#custom-engine-contract), or a re-run cannot gate them. One that honours vLLM's request fields has `vllm` in its `name`, as above: the harness picks its request profile from the name.
- **`workload.extra_body`:** the timed sampling, merged last into every timed request to every engine; absent means greedy. Write temperature, top_k, top_p and seed explicitly: vLLM fills a missing top_k or top_p from the checkpoint's `generation_config.json`, so a temperature-only body samples from a cut distribution there and perhaps not on another engine. `manifest validate` refuses fields that change what is timed or how tokens are counted (`EXTRA_BODY_RESERVED`: what the harness sends, exact-length and token-count fields, stop/stop_token_ids/logit_bias, n/best_of, echo/suffix), `chat_template_kwargs` and secret-looking keys.
- **`workload.chat_template_kwargs`:** the chat-template setting sent with every chat request to every engine, the same value in every harness call ([harness-commands.md](harness-commands.md), "Chat template").
- **`install`:** commands that work on a **fresh** host, in a fresh venv, with pinned versions.
  - Source builds: `git clone <url>`, `git checkout <commit>`, `git apply <patch>`, and the build commands.
  - Put patches in the run directory and reference them by relative path.
  - Nothing from your shell history: no "already installed" assumptions, no local wheels without a path and a sha256.
- **`launch`:** an argv list, not a shell string.
  - `launch[0]` is a bare program name (`vllm`, `trtllm-serve`, `llama-server`, or a unique `python-<engine>` for `python -m` launchers), never a path. A re-run replaces it with that name from the engine's own venv; your `bench rounds` finds it through `~/ik/bin` (remote-host.md, "Engine programs and paths").
  - Paths are relative to the run directory, never under your home directory: the model (`models/<dir>`, the same as `model.local_dir`), config files, engine source. Your `bench rounds` runs from the run directory on the host (with `models` symlinked there); a re-run runs from its copy of your run directory and downloads the model to `model.local_dir`.
  - Pin the served name (`--served-model-name <name>` or the engine's equivalent) so `--model` is the same in every harness call.
- **`env`:** only non-secret values, e.g. `{"VLLM_USE_V1": "1"}`. A secret is listed with the value `null` (`{"HF_TOKEN": null}` = inherit it from the caller's environment), never written here.
- **`resolved_config`:** copied from the engine's log or info endpoint (tuning.md), with where it came from. Both a re-run and the reviewers compare it to what they see.
- **`model.revision`:** the resolved 40-hex sha.
- **`harness.wheel_sha256`:** `sha256sum` of the wheel installed on the GPU host (remote-host.md, "Install the harness", says where to find it when the host came prepared). An empty value is a warning here and a refusal at a re-run.
- **`harness.dev_prompt_set_sha256`:** the `sha256` that `inferencekit prompts verify --set dev` prints.
- **No host names, IPs, users, key paths or rental-provider names anywhere.**

Run `inferencekit manifest validate runs/<run-id>/manifest.json` (strict by default: exit 1 on any error), and fix everything it reports. `--no-strict` prints the same result with exit 0, which is useful for a draft. Copy the manifest to the GPU host after every change, because `bench rounds` reads it there.

## reproduce.sh
One script in `runs/<run-id>/` that a stranger can run on a clean host with the same GPU:

```bash
#!/usr/bin/env bash
# Reproduce <run-id>: <model id>@<sha> on <gpu>. Agent-measured; re-run on a clean host to check.
set -euo pipefail
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH
uv venv envs/ik && uv pip install --python envs/ik/bin/python "inferencekit-cli==0.1.0" "huggingface_hub[cli]"
envs/ik/bin/hf download <model id> --revision <sha> --local-dir models/<dir>
# per engine: venv + install commands from manifest.json, then launch, ready check, benchmark, stop
# ... (generated from manifest.json; keep it in sync)
envs/ik/bin/inferencekit manifest validate manifest.json --strict
envs/ik/bin/inferencekit bench rounds --help   # the same harness commands you ran, with the same arguments
```

- Generate the engine sections from `manifest.json`, so the two cannot drift.
- Use the exact harness commands and arguments you ran. They are recorded in each evidence record's `data.request`.
