"""Run one engine's benchmark and return a common result shape.

Every adapter returns::

    {"engine": {"name", "version", "commit"}, "flags": {...}, "rounds": [round, ...],
     "tokens": [...] or None, "log_text": str, "limitations": [...], "files": [(path, role)]}

where each round is ``{"index", "t_start", "t_end", "ttft_ms", "decode_tok_s",
"tpot_ms", "gen_tokens"}`` (times in epoch seconds; None when unknown).
"""

from __future__ import annotations

import json
import re
import shlex
import time
from pathlib import Path

from inferencekit.evidence import redact, redact_text
from inferencekit.proc import run, which
from inferencekit.registry import OpError

WORKER_DIR = Path(__file__).resolve().parent
MARK = "INFERKIT_RESULT "


def _redact_file(path: Path) -> None:
    """Engine logs can echo configuration; scrub secret-looking env values before keeping them."""
    if path.is_file():
        path.write_text(redact_text(path.read_text(errors="replace")))


def run_python_worker(engine: str, python: str, cfg: dict, staging: Path, timeout: int) -> dict:
    """vLLM / SGLang: run the worker script under the engine's interpreter."""
    cfg_path = staging / "worker_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))  # the worker needs the real values while it runs
    out_path, log_path = staging / "worker_stdout.txt", staging / "engine.log"
    r = run([python, str(WORKER_DIR / f"{engine}_worker.py"), str(cfg_path)], timeout=timeout,
            stdout_path=out_path, stderr_path=log_path)
    # The kept copy must not carry a token typed into --engine-args ({"hf_token": ...}):
    # redact() blanks secret-looking keys, redact_text secret env values and the home dir.
    cfg_path.write_text(redact_text(json.dumps(redact(cfg), indent=2)))
    _redact_file(log_path)
    _redact_file(out_path)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith(MARK)]
    if not lines:
        tail = log_path.read_text(errors="replace")[-1500:] if log_path.is_file() else r.stderr[-1500:]
        raise OpError(f"{engine} worker failed (exit {r.returncode}); log tail:\n{tail}")
    result = json.loads(lines[-1][len(MARK):])
    result["log_text"] = log_path.read_text(errors="replace") if log_path.is_file() else ""
    result["files"] = [(log_path, "engine_log"), (cfg_path, "worker_config")]
    result.setdefault("limitations", [])
    return result


# --------------------------------------------------------------------------- llama.cpp

def _llama_flag(key: str) -> str:
    """{'ngl': 99} -> '-ngl'; {'flash-attn': 1} -> '--flash-attn'; keys with dashes pass through."""
    if key.startswith("-"):
        return key
    return ("-" if len(key) <= 3 else "--") + key


def run_llamacpp(model: str, prompt_len: int, gen_len: int, rounds: int, engine_args: dict, staging: Path,
                 timeout: int) -> dict:
    bench = which("llama-bench")
    if bench:
        return _run_llama_bench(bench, model, prompt_len, gen_len, rounds, engine_args, staging, timeout)
    cli = which("llama-cli")
    if cli:
        return _run_llama_cli(cli, model, prompt_len, gen_len, rounds, engine_args, staging, timeout)
    raise OpError("llama-bench / llama-cli not found on PATH (build llama.cpp with -DGGML_CUDA=ON)")


def _run_llama_bench(bench, model, prompt_len, gen_len, rounds, engine_args, staging, timeout) -> dict:
    extra = []
    for k, v in engine_args.items():
        extra += [_llama_flag(k), str(v)]
    argv = [bench, "-m", model, "-p", str(prompt_len), "-n", str(gen_len), "-r", str(rounds), "-o", "json", *extra]
    out_path, log_path = staging / "llama_bench.json", staging / "llama_bench.log"
    t0 = time.time()
    r = run(argv, timeout=timeout, stdout_path=out_path, stderr_path=log_path)
    t1 = time.time()
    _redact_file(log_path)
    if not r.ok:
        raise OpError(f"llama-bench failed (exit {r.returncode}): {redact_text(r.stderr[-1000:])}")
    entries = json.loads(r.stdout)
    pp = next((e for e in entries if e.get("n_prompt", 0) > 0 and e.get("n_gen", 0) == 0), None)
    tg = next((e for e in entries if e.get("n_gen", 0) > 0 and e.get("n_prompt", 0) == 0), None)
    if tg is None:
        raise OpError("llama-bench output has no generation (tg) test")
    tg_ts = tg.get("samples_ts") or [tg.get("avg_ts")]
    pp_ns = (pp or {}).get("samples_ns") or ([pp["avg_ns"]] if pp and pp.get("avg_ns") else [])
    out_rounds = []
    for i, ts in enumerate(tg_ts):
        ttft = pp_ns[i] / 1e6 if i < len(pp_ns) else None
        out_rounds.append({"index": i, "t_start": None, "t_end": None, "ttft_ms": ttft, "decode_tok_s": ts,
                           "tpot_ms": 1e3 / ts if ts else None, "gen_tokens": gen_len})
    keys = ("n_gpu_layers", "flash_attn", "type_k", "type_v", "n_batch", "n_ubatch", "n_threads", "backends",
            "split_mode", "gpu_info", "model_type")
    flags = {k: tg.get(k) for k in keys if k in tg}
    limitations = [
        "llama-bench runs prefill (pp) and generation (tg) as separate tests: TTFT here is the pp time without "
        "sampling, and tg decodes from an empty KV cache unless a depth (-d) is passed",
        f"rounds ran back to back inside one llama-bench call ({t0:.0f}-{t1:.0f} epoch s): no per-round timestamps",
    ]
    if not tg.get("samples_ts"):
        limitations.append("this llama-bench build reports only the average over repetitions (n=1)")
    return {"engine": {"name": "llama.cpp", "version": str(tg.get("build_number") or "") or None,
                       "commit": tg.get("build_commit")},
            "flags": flags, "rounds": out_rounds, "tokens": None, "log_text": r.stderr, "limitations": limitations,
            "files": [(out_path, "raw_output"), (log_path, "engine_log")]}


_PROMPT_EVAL = re.compile(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens")
_EVAL = re.compile(r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:runs|tokens)")


def _run_llama_cli(cli, model, prompt_len, gen_len, rounds, engine_args, staging, timeout) -> dict:
    extra = []
    for k, v in engine_args.items():
        extra += [_llama_flag(k), str(v)]
    prompt = " ".join(["hello"] * prompt_len)  # roughly one token per word; exact length is not controlled
    out_rounds, logs = [], []
    for i in range(rounds):
        argv = [cli, "-m", model, "-p", prompt, "-n", str(gen_len), "--temp", "0", "--ignore-eos", "-no-cnv", *extra]
        t_start = time.time()
        r = run(argv, timeout=timeout)
        if not r.ok and "no-cnv" in r.stderr:  # newer builds dropped the flag
            r = run([a for a in argv if a != "-no-cnv"], timeout=timeout)
        t_end = time.time()
        logs.append(r.stderr)
        pe, de = _PROMPT_EVAL.search(r.stderr), _EVAL.search(r.stderr)
        if not de:
            raise OpError(f"could not find llama.cpp timing lines in llama-cli output (exit {r.returncode})")
        ms, runs = float(de.group(1)), int(de.group(2))
        out_rounds.append({"index": i, "t_start": t_start, "t_end": t_end,
                           "ttft_ms": float(pe.group(1)) if pe else None,
                           "decode_tok_s": runs / (ms / 1e3) if ms > 0 else None,
                           "tpot_ms": ms / runs if runs else None, "gen_tokens": runs + 1})
    log_path = staging / "llama_cli.log"
    log_path.write_text(redact_text("\n\n".join(logs)))
    return {"engine": {"name": "llama.cpp", "version": None, "commit": None}, "flags": dict(engine_args),
            "rounds": out_rounds, "tokens": None, "log_text": "\n".join(logs),
            "limitations": ["llama-cli fallback (no llama-bench): prompt length is approximate and TTFT is the "
                            "reported prompt-eval time"],
            "files": [(log_path, "engine_log")]}


# --------------------------------------------------------------------------- any command

def run_cmd(cmd: str, rounds: int, round_offset: int, timeout: int) -> dict:
    """Run ``cmd`` once per round; it must print a JSON line containing decode_tok_s.

    Optional keys in that line: ttft_ms, tpot_ms, gen_tokens, tokens (list of ids),
    flags (dict, e.g. {"cuda_graphs": true, "attention_backend": "FLASH_ATTN"}),
    engine ({"name", "version", "commit"}).
    """
    argv = shlex.split(cmd)
    out_rounds, last = [], {}
    for k in range(rounds):
        t_start = time.time()
        r = run(argv, timeout=timeout)
        t_end = time.time()
        if not r.ok:
            raise OpError(f"command failed in round {k} (exit {r.returncode}): {redact_text(r.stderr[-800:])}")
        found = None
        for line in reversed(r.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{") and "decode_tok_s" in line:
                try:
                    found = json.loads(line)
                    break
                except ValueError:
                    continue
        if found is None:
            raise OpError(f"round {k}: no JSON line with decode_tok_s in the command's stdout")
        last = found
        tok_s = found.get("decode_tok_s")
        out_rounds.append({"index": round_offset + k, "t_start": t_start, "t_end": t_end,
                           "ttft_ms": found.get("ttft_ms"), "decode_tok_s": tok_s,
                           "tpot_ms": found.get("tpot_ms") or (1e3 / tok_s if tok_s else None),
                           "gen_tokens": found.get("gen_tokens")})
    engine = last.get("engine") if isinstance(last.get("engine"), dict) else {}
    return {"engine": {"name": engine.get("name") or "cmd", "version": engine.get("version"),
                       "commit": engine.get("commit")},
            "flags": last.get("flags") if isinstance(last.get("flags"), dict) else {},
            "rounds": out_rounds, "tokens": last.get("tokens"), "log_text": "", "limitations": [], "files": []}
