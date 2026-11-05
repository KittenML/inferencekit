"""Batch-1 decode timing inside SGLang's own Python (offline Engine). Standard library + sglang only.

Usage: python sglang_worker.py CONFIG.json
Same protocol and timing definitions as vllm_worker.py. SGLang's radix cache reuses
prompt prefixes, so each call again gets a fresh synthetic prompt.
"""

import json
import os
import random
import sys
import time

MARK = "INFERKIT_RESULT "
# Only these server arguments are recorded. Never dump the whole object: it can hold an API key.
FLAG_FIELDS = ("disable_cuda_graph", "cuda_graph_max_bs", "cuda_graph_bs", "attention_backend",
               "prefill_attention_backend", "decode_attention_backend", "sampling_backend", "quantization", "dtype",
               "kv_cache_dtype", "enable_torch_compile", "tp_size", "disable_radix_cache", "mem_fraction_static",
               "context_length", "speculative_algorithm")


def prompt_ids(seed, label, n, vocab):
    lo = 1000 if vocab > 2000 else 10
    hi = min(vocab, 30000)
    rng = random.Random(f"{seed}:{label}")
    return [lo + int(rng.random() * (hi - lo)) for _ in range(n)]


def plain(x):
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in list(x)[:256]]
    return str(x)


def main():
    cfg = json.load(open(sys.argv[1]))
    import sglang

    engine_args = dict(cfg.get("engine_args", {}))
    engine_args.setdefault("random_seed", cfg["seed"])
    engine = sglang.Engine(model_path=cfg["model"], **engine_args)
    try:
        vocab = len(engine.tokenizer_manager.tokenizer)
    except Exception:  # noqa: BLE001
        vocab = 32000

    def generate(ids, max_new):
        params = {"temperature": 0.0, "max_new_tokens": max_new, "ignore_eos": True}
        t0 = time.perf_counter()
        out = engine.generate(input_ids=ids, sampling_params=params)
        dt = time.perf_counter() - t0
        toks = out.get("output_ids") or []
        n_out = (out.get("meta_info") or {}).get("completion_tokens", len(toks))
        return dt, [int(t) for t in toks], int(n_out)

    n, g = cfg["prompt_len"], cfg["gen_len"]
    _, ref_tokens, _ = generate(prompt_ids(cfg["seed"], "reference", n, vocab), g)
    for w in range(cfg.get("warmup", 1) - 1):
        generate(prompt_ids(cfg["seed"], f"warmup{w}", n, vocab), g)
    rounds = []
    for k in range(cfg["rounds"]):
        i = cfg.get("round_offset", 0) + k
        t_start = time.time()
        ttft, _, _ = generate(prompt_ids(cfg["seed"], f"round{i}:first", n, vocab), 1)
        total, _, n_out = generate(prompt_ids(cfg["seed"], f"round{i}:full", n, vocab), g)
        t_end = time.time()
        decode_s = total - ttft
        ok = n_out > 1 and decode_s > 0
        rounds.append({"index": i, "t_start": t_start, "t_end": t_end, "ttft_ms": ttft * 1e3,
                       "total_ms": total * 1e3, "gen_tokens": n_out,
                       "decode_tok_s": (n_out - 1) / decode_s if ok else None,
                       "tpot_ms": decode_s * 1e3 / (n_out - 1) if ok else None})
    args = getattr(engine, "server_args", None)
    flags = {f: plain(getattr(args, f, None)) for f in FLAG_FIELDS if hasattr(args, f)}
    result = {"engine": {"name": "sglang", "version": getattr(sglang, "__version__", None)}, "flags": flags,
              "rounds": rounds, "tokens": ref_tokens,
              "prompt": {"vocab_bound": min(vocab, 30000), "rule": "ids from random.Random('<seed>:<label>')"}}
    sys.stdout.write(MARK + json.dumps(result) + "\n")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
