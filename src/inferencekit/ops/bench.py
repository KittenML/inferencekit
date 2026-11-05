"""bench decode: batch-1 decode speed, TTFT and per-token latency, with the engine's real flags.

Definitions (per round):
  ttft_ms       wall time of a 1-token generation: prefill + first sample (offline, no network)
  decode_tok_s  (tokens - 1) / (t_total - ttft), t_total = wall time of a gen_len generation
  tpot_ms       (t_total - ttft) / (tokens - 1): time per output token after the first

Reported value = median over rounds, with a 95% bootstrap CI over rounds. Rounds are
the unit of repetition because tokens inside one generation are not independent.

Interleaving: to compare A and B fairly, alternate them (A, B, A, B ...) so slow
drift (clocks, temperature, neighbours) hits both equally. ``--extend <ev_id>`` runs
more rounds of the same configuration and writes a new record holding the old and
new rounds, each with wall-clock timestamps; ``compare`` checks the alternation.

Environment: the engine runs with the caller's full environment (unlike profiler
captures, which use an allowlist). The record lists the *names* of the engine-
configuration variables that were set (``data.env_config_names``), so a profile taken
with a different configuration can be spotted. Secret-looking keys inside
``--engine-args`` (``hf_token``, ``api_key``...) are stored as <redacted>.
"""

from __future__ import annotations

import json
import sys

from inferencekit import evidence as ev
from inferencekit import stats
from inferencekit.engines import adapters, gates
from inferencekit.gpuinfo import gpu_identity
from inferencekit.ops.doctor import gpu_processes
from inferencekit.proc import config_env_names
from inferencekit.registry import Arg, OpError, operation

ENGINES = ("vllm", "sglang", "llamacpp", "cmd")
# Arguments that must match for --extend: they define "the same configuration".
SAME_CONFIG = ("engine", "model", "prompt_len", "gen_len", "seed", "engine_args", "cmd")


def _engine_args(text: str) -> dict:
    try:
        value = json.loads(text or "{}")
    except ValueError as e:
        raise OpError(f"engine_args is not valid JSON: {e}") from None
    if not isinstance(value, dict):
        raise OpError("engine_args must be a JSON object, e.g. '{\"max_model_len\": 4096}'")
    return value


def _ci_metric(name: str, unit: str, values: list[float]) -> dict:
    if not values:
        return ev.metric(name, None, unit, "median", n=0)
    ci = stats.bootstrap_ci(values, ps=(50,))[50]
    return ev.metric(name, stats.summarize(values)["median"], unit, "median", n=len(values), ci95=ci)


@operation("bench_decode", ("bench", "decode"), "Batch-1 decode tok/s, per-token latency and TTFT for vllm, "
           "sglang, llama.cpp or any command; records effective engine flags and hard-gate violations.",
           [Arg("engine", "string", "Engine adapter.", required=True, choices=ENGINES),
            Arg("model", "string", "HF id or local path (a .gguf file for llamacpp). Optional for cmd."),
            Arg("prompt_len", "integer", "Prompt length in tokens.", default=128),
            Arg("gen_len", "integer", "Generated tokens per round (>= 2).", default=128),
            Arg("rounds", "integer", "Timed rounds.", default=5),
            Arg("warmup", "integer", "Untimed warm-up generations (the first also records reference tokens).",
                default=1),
            Arg("seed", "integer", "Seed for synthetic prompts and the engine.", default=0),
            Arg("engine_args", "string", "JSON object passed to the engine (vLLM LLM(...), SGLang Engine(...), "
                "or llama-bench flags such as {\"ngl\": 99, \"fa\": 1}). Values of secret-looking keys "
                "(hf_token, api_key...) are stored as <redacted>.", default="{}"),
            Arg("python", "string", "Interpreter that has the engine installed (vllm/sglang). Default: this one."),
            Arg("cmd", "string", "For engine=cmd: command printing a JSON line {\"decode_tok_s\": ..., "
                "\"ttft_ms\": ...}. Stored only as a digest.", sensitive=True),
            Arg("extend", "string", "Evidence id of an earlier run of the same configuration; its rounds are kept "
                "and new rounds appended (use to interleave A/B rounds)."),
            Arg("record_tokens", "boolean", "Store the reference generation's token ids (for correctness).",
                default=True),
            Arg("timeout", "integer", "Seconds before the engine run (and every process it started) is killed.",
                default=3600)])
def bench_decode(call, engine, model, prompt_len, gen_len, rounds, warmup, seed, engine_args, python, cmd, extend,
                 record_tokens, timeout):
    if gen_len < 2:
        raise OpError("gen_len must be >= 2 (decode speed needs tokens after the first)")
    if rounds < 1:
        raise OpError("rounds must be >= 1")
    if engine != "cmd" and not model:
        raise OpError(f"engine={engine} needs --model")
    if engine == "cmd" and not cmd:
        raise OpError("engine=cmd needs --cmd")
    eargs = _engine_args(engine_args)
    store = ev.Store()

    prev, prev_rounds = None, []
    if extend:
        prev = store.load(extend)
        if prev.get("kind") != "benchmark":
            raise OpError(f"--extend {extend}: not a benchmark record")
        problem = ev.integrity_problem(prev)
        if problem:
            raise OpError(f"--extend: {problem}; refusing to build on it")
        prev_args = prev["data"].get("request", {}).get("args", {})
        mine = ev.redact(call.request()["args"])  # stored args went through the same redaction
        diff = [k for k in SAME_CONFIG if prev_args.get(k) != mine.get(k)]
        if diff:
            raise OpError(f"--extend needs the same configuration; these differ: {diff}")
        prev_rounds = prev["data"].get("rounds", [])
    offset = len(prev_rounds)

    # Leftovers from an earlier run (an orphaned engine, someone else's job) steal bandwidth.
    others = gpu_processes()
    with store.staging_dir() as staging:  # removed again if the run fails
        cfg = {"model": model, "prompt_len": prompt_len, "gen_len": gen_len, "rounds": rounds, "warmup": warmup,
               "seed": seed, "engine_args": eargs, "round_offset": offset}
        if engine in ("vllm", "sglang"):
            result = adapters.run_python_worker(engine, python or sys.executable, cfg, staging, timeout)
        elif engine == "llamacpp":
            result = adapters.run_llamacpp(model, prompt_len, gen_len, rounds, eargs, staging, timeout)
            for r in result["rounds"]:
                r["index"] += offset
        else:
            result = adapters.run_cmd(cmd, rounds, offset, timeout)

        flags = result.get("flags") or {}
        backend = gates.attention_backend_from_log(result.get("log_text", "")) if engine == "vllm" else None
        gate = gates.check(engine, flags, backend)
        limitations = list(result.get("limitations", []))
        if others:
            limitations.append(f"{len(others)} other compute process(es) were on the GPU when the run started ("
                               + ", ".join(f"{o['name']} {o['used_mib']} MiB" for o in others[:4])
                               + "): timings may be contended")
        limitations += [f"HARD GATE: {v}" for v in gate["violations"]]
        limitations += gate["warnings"]
        if gate["valid"] is None:
            limitations.append("engine flags not visible to inferencekit: CUDA graphs / attention backend not checked")
        if prev is not None:
            old_flags = prev["identity"]["engine"].get("flags", {})
            if old_flags != ev.redact(flags):
                limitations.append(f"effective engine flags changed since {prev['evidence_id'][:15]}")

        all_rounds = prev_rounds + result["rounds"]
        decode = [r["decode_tok_s"] for r in all_rounds if r.get("decode_tok_s") is not None]
        ttft = [r["ttft_ms"] for r in all_rounds if r.get("ttft_ms") is not None]
        tpot = [r["tpot_ms"] for r in all_rounds if r.get("tpot_ms") is not None]
        missing = []
        if not ttft:
            missing.append("ttft_ms")
        if len(decode) < 2:
            limitations.append("fewer than 2 rounds: no confidence interval")
        metrics = [_ci_metric("decode_tok_s", "tok/s", decode), _ci_metric("ttft_ms", "ms", ttft),
                   _ci_metric("tpot_ms", "ms", tpot)]

        eng = result.get("engine") or {}
        ident = ev.identity(
            gpu=gpu_identity(),
            engine={"name": eng.get("name") or engine, "version": eng.get("version"), "commit": eng.get("commit"),
                    "flags": flags},
            model={"id": model, "revision": eargs.get("revision"),
                   "dtype": str(flags["dtype"]) if flags.get("dtype") is not None else None,
                   "quant": str(flags.get("quantization")) if flags.get("quantization") else None},
            workload={"prompt_len": prompt_len, "gen_len": gen_len, "batch": 1, "seed": seed})
        med = metrics[0]["value"]
        lo, hi = metrics[0]["ci95"] or (None, None)
        summary = f"{ident['engine']['name']} {model or ''}".strip() + ": decode "
        summary += f"{med:.1f} tok/s" if med is not None else "n/a"
        if lo is not None and hi is not None:
            summary += f" (95% CI {lo:.1f}-{hi:.1f}, n={len(decode)})"
        if metrics[1]["value"] is not None:
            summary += f", TTFT {metrics[1]['value']:.1f} ms"
        if gate["violations"]:
            summary += " [HARD GATE: " + "; ".join(gate["violations"]) + "]"
        tokens = result.get("tokens") if record_tokens else None
        if tokens is None and prev is not None:
            tokens = prev["data"].get("tokens")
        record = ev.build(
            call, kind="benchmark", claim_class="observed", summary=summary, identity_=ident, metrics=metrics,
            artifacts=[ev.artifact(p, staging, role) for p, role in result.get("files", []) if p.is_file()],
            missing=missing, limitations=limitations,
            data={"rounds": all_rounds, "tokens": tokens, "prompt": result.get("prompt"),
                  "claim_validity": gate, "extends": prev["evidence_id"] if prev else None,
                  "env_config_names": config_env_names(), "gpu_processes_at_start": len(others),
                  "definitions": {"ttft_ms": "wall time of a 1-token generation (prefill + first sample)",
                                  "decode_tok_s": "(tokens - 1) / (t_total - ttft) per round",
                                  "tpot_ms": "(t_total - ttft) / (tokens - 1) per round"}})
        return store.save(record, staging)
