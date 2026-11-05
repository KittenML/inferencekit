"""Batch-1 decode timing inside vLLM's own Python. Standard library + vllm only.

Usage: python vllm_worker.py CONFIG.json
Prints exactly one line ``INFERKIT_RESULT {...}`` on stdout; logs go to stderr.

Timing per round (offline engine, no HTTP, no network):
  ttft   = wall time of a generation with max_tokens=1 (prefill + first sample)
  total  = wall time of a generation with max_tokens=gen_len
  decode tok/s = (tokens - 1) / (total - ttft)
Every call gets a fresh synthetic prompt: vLLM caches prompt prefixes, and a cached
prompt would make TTFT look far better than a real request's.
"""

import json
import os
import random
import sys
import time

MARK = "INFERKIT_RESULT "


def prompt_ids(seed, label, n, vocab):
    """Deterministic synthetic token ids, avoiding the low ids where special tokens live."""
    lo = 1000 if vocab > 2000 else 10
    hi = min(vocab, 30000)
    rng = random.Random(f"{seed}:{label}")  # str seeds hash with SHA-512: stable everywhere
    return [lo + int(rng.random() * (hi - lo)) for _ in range(n)]


def get(obj, path):
    for part in path.split("."):
        if obj is None:
            return None
        obj = getattr(obj, part, None)
    return obj


def plain(x):
    """Enums and config objects become strings; containers are kept small."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (list, tuple, set)):
        return [plain(v) for v in list(x)[:256]]
    return str(x)


def effective_flags(llm):
    """What the engine actually runs with (after its own defaults), from its config objects."""
    eng = llm.llm_engine
    cfg = get(eng, "vllm_config")
    if cfg is None and hasattr(eng, "get_vllm_config"):
        try:
            cfg = eng.get_vllm_config()
        except Exception:  # noqa: BLE001
            cfg = None
    sizes = plain(get(cfg, "compilation_config.cudagraph_capture_sizes"))
    flags = {
        "enforce_eager": plain(get(cfg, "model_config.enforce_eager")),
        "dtype": plain(get(cfg, "model_config.dtype")),
        "quantization": plain(get(cfg, "model_config.quantization")),
        "max_model_len": plain(get(cfg, "model_config.max_model_len")),
        "kv_cache_dtype": plain(get(cfg, "cache_config.cache_dtype")),
        "block_size": plain(get(cfg, "cache_config.block_size")),
        "enable_prefix_caching": plain(get(cfg, "cache_config.enable_prefix_caching")),
        "gpu_memory_utilization": plain(get(cfg, "cache_config.gpu_memory_utilization")),
        "tensor_parallel_size": plain(get(cfg, "parallel_config.tensor_parallel_size")),
        "compilation_level": plain(get(cfg, "compilation_config.level")),
        "compilation_mode": plain(get(cfg, "compilation_config.mode")),
        "cudagraph_mode": plain(get(cfg, "compilation_config.cudagraph_mode")),
        "speculative_decoding": get(cfg, "speculative_config") is not None,
    }
    if isinstance(sizes, list):
        flags["cudagraph_capture_sizes"] = {"count": len(sizes), "min": min(sizes) if sizes else None,
                                            "max": max(sizes) if sizes else None, "includes_1": 1 in sizes}
    return flags


def main():
    cfg = json.load(open(sys.argv[1]))
    import vllm
    from vllm import LLM, SamplingParams

    engine_args = dict(cfg.get("engine_args", {}))
    engine_args.setdefault("seed", cfg["seed"])
    llm = LLM(model=cfg["model"], **engine_args)
    try:
        vocab = len(llm.get_tokenizer())
    except Exception:  # noqa: BLE001 - e.g. skip_tokenizer_init
        vocab = 32000

    def generate(ids, max_tokens):
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        t0 = time.perf_counter()
        try:
            outs = llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)
        except TypeError:  # older vLLM: token ids as a keyword argument
            outs = llm.generate(prompt_token_ids=[ids], sampling_params=sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        return dt, [int(t) for t in outs[0].outputs[0].token_ids]

    n, g = cfg["prompt_len"], cfg["gen_len"]
    # Warm-up on the fixed reference prompt; its tokens feed the correctness check.
    _, ref_tokens = generate(prompt_ids(cfg["seed"], "reference", n, vocab), g)
    for w in range(cfg.get("warmup", 1) - 1):
        generate(prompt_ids(cfg["seed"], f"warmup{w}", n, vocab), g)
    rounds = []
    for k in range(cfg["rounds"]):
        i = cfg.get("round_offset", 0) + k
        t_start = time.time()
        ttft, _ = generate(prompt_ids(cfg["seed"], f"round{i}:first", n, vocab), 1)
        total, toks = generate(prompt_ids(cfg["seed"], f"round{i}:full", n, vocab), g)
        t_end = time.time()
        decode_s = total - ttft
        ok = len(toks) > 1 and decode_s > 0
        rounds.append({"index": i, "t_start": t_start, "t_end": t_end, "ttft_ms": ttft * 1e3,
                       "total_ms": total * 1e3, "gen_tokens": len(toks),
                       "decode_tok_s": (len(toks) - 1) / decode_s if ok else None,
                       "tpot_ms": decode_s * 1e3 / (len(toks) - 1) if ok else None})
    result = {"engine": {"name": "vllm", "version": getattr(vllm, "__version__", None)},
              "flags": effective_flags(llm), "rounds": rounds, "tokens": ref_tokens,
              "prompt": {"vocab_bound": min(vocab, 30000), "rule": "ids from random.Random('<seed>:<label>')"}}
    sys.stdout.write(MARK + json.dumps(result) + "\n")
    sys.stdout.flush()
    os._exit(0)  # skip engine teardown, which can hang on some versions; the OS frees the GPU


if __name__ == "__main__":
    main()
