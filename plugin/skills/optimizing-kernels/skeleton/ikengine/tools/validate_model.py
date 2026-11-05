"""Check a model module against a reference on the same prompts: greedy ids, top-20 logprobs and speed.

    python -m ikengine.tools.validate_model --model DIR --ref hf             # Hugging Face transformers
    python -m ikengine.tools.validate_model --model DIR --ref http://H:8000  # a vLLM or SGLang server

A reference server that needs a key reads it from the environment variable that --api-key-env names
(default OPENAI_API_KEY); the key is never a command-line value and is never printed or written.

Run it from the directory that holds ikengine/. For each prompt the engine (ikengine.engine.build_engine)
decodes greedily with top-20 logprobs, then the reference is asked for:
- its own greedy continuation (free running), to find the first step where the two greedy paths diverge;
- its logprobs teacher-forced on the engine's tokens (HF: one forward pass; vLLM: `prompt_logprobs`; SGLang: the
  native /generate with `logprob_start_len`, alignment checked by token id), so every step compares the two
  distributions on the same context. `--ref-kind auto` tries vLLM's request, then SGLang's; a server with
  neither is compared only on the steps before the divergence, on its own path.
Reported per prompt and overall: first divergence (step index, or none), top-1 agreement (the reference's argmax
equals the engine's token; "top1_agreement_ties" also counts an exactly tied reference top-1, and the gap at every
step that still disagrees is kept), mean KL(reference || engine) on the reference's top-20 plus one bucket for the
rest (merging tokens into buckets can only lower a KL, so with exact probabilities this is at most the
full-vocabulary KL; an engine probability missing from its own top-20 is imputed, which makes it approximate), and
the engine's decode tok/s (one untimed warm-up, then the timed greedy runs without logprobs).

At each first divergence the report gives the gap between the two tokens on each side, in nats: the reference's
lp(its token) - lp(engine token) teacher-forced (ref_tf_gap) and on its own free-running path (ref_free_gap), and
the engine's lp(its token) - lp(reference token) (eng_gap). The contexts are identical up to that step, so all
three describe the same next-token choice. It is a near tie when every gap is at most --tie-nats: logits come
out of a 16-bit output layer (bf16 logits in [16, 32) sit 0.125 apart), so gaps come in such steps, 0 is an
exact tie and a few steps are rounding. A large reference gap means the engine chose a token the reference
clearly does not. A ref_tf_gap at or below 0 means the reference's own teacher-forced pass prefers the engine's
token, so the reference disagrees with itself there; such a divergence is flagged ref_tf_prefers_engine.

Prompts: the 8 built-in ones, or --prompts FILE: plain text (one per line) or JSONL (optionally .gz) with "prompt"
(text) or "messages" (a chat, rendered with the chat template), such as inferencekit's dev set. --per-bucket K
takes the first K records of each "bucket"; --buckets a,b keeps only those buckets, in that order.
--ref-aa also measures the reference against itself, the A/A floor to set the thresholds from: its free-running
logprobs against its own teacher-forced logprobs on the same tokens (top-1, the gaps where they differ, KL), and a
second free run against the first (run to run: the first step where the greedy paths differ). It needs a
reference that gives teacher-forced logprobs (exit 2 otherwise).

--save-engine PATH writes the engine's runs; --load-engine PATH compares them instead of building the engine, so
the engine and a reference server never need the GPU at the same time (--ref none stops after the engine). The
engine runs first and is freed before a Hugging Face reference loads, so both fit on one GPU when either fits
alone (`--ref-device` puts the reference elsewhere). The exit code is 1 when a threshold fails, 2 on a setup error.
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import math
import os
import sys
import time
import urllib.request

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Q: A train travels 120 km in 1.5 hours. What is its average speed in km/h?\nA:",
    "Once upon a time, in a village at the edge of a dark forest,",
    "List three uses of copper:\n1.",
    "Translate to German: The weather is nice today.\nGerman:",
    "The derivative of x^3 + 2x is",
    "In 1969, the first humans to walk on the Moon were",
]
TOP = 20
TOKENS = 64                     # greedy tokens per prompt unless --tokens (or a loaded engine run) says otherwise
OBSERVED = "observed (this host, this run)"


# --- metrics --------------------------------------------------------------------------------------------
def _probs(top: dict[int, float]) -> dict[int, float]:
    return {t: math.exp(lp) for t, lp in top.items()}


def bucket_kl(ref_top: dict[int, float], eng_top: dict[int, float], eps: float = 1e-10) -> float:
    """KL(ref || engine) over the reference's top-k tokens plus one bucket for the rest of the vocabulary.

    Both arguments map token id -> logprob (each a top-k list). An engine token missing from its own top-k gets
    min(its smallest top-k probability, its remaining mass shared evenly), never more than it could have.
    """
    pr, pe = _probs(ref_top), _probs(eng_top)
    missing = [t for t in pr if t not in pe]
    floor = min(pe.values()) if pe else eps
    left = max(0.0, 1.0 - sum(pe.values()))
    imputed = min(floor, left / len(missing)) if missing else 0.0
    q = {t: pe.get(t, imputed) for t in pr}
    kl = sum(p * (math.log(max(p, eps)) - math.log(max(q[t], eps))) for t, p in pr.items())
    rest_r, rest_e = max(0.0, 1.0 - sum(pr.values())), max(0.0, 1.0 - sum(q.values()))
    if rest_r > eps:
        kl += rest_r * (math.log(rest_r) - math.log(max(rest_e, eps)))
    return max(0.0, kl)


def _gap(top: dict | None, mine: int, other: int) -> tuple[float | None, bool]:
    """lp(mine) - lp(other) in one top-k list, and whether that is only a lower bound (`other` is missing from the
    list, so it counts at the list's smallest logprob). None when `mine` is missing."""
    if not top or mine not in top:
        return None, False
    if other not in top:
        return top[mine] - min(top.values()), True
    return top[mine] - top[other], False


def divergence(step: int | None, eng_ids, ref_ids, eng_tops, ref_tf_tops, ref_free_tops=None,
               tie_nats: float = 0.5) -> dict | None:
    """The first divergence and the gap between the two tokens on each side (nats): ref_tf_gap and ref_free_gap are
    the reference's lp(reference token) - lp(engine token), teacher-forced on the engine's path and on its own
    free-running path; eng_gap is the engine's lp(engine token) - lp(reference token). A gap whose other token is
    missing from that side's top-k is a lower bound, marked <key>_bound ">=". near_tie: every gap known is exact
    and at most tie_nats (bf16 logits move in steps of 0.125 nats for logits of 16 to 32, so 0 is an exact tie and
    one or two steps are rounding). None when the paths do not diverge."""
    if step is None or step >= len(eng_ids) or step >= len(ref_ids):
        return None
    e, r = eng_ids[step], ref_ids[step]
    out, known = {"step": step, "engine_token": e, "ref_token": r}, []
    for key, tops, mine, other in (("ref_tf_gap", ref_tf_tops, r, e), ("ref_free_gap", ref_free_tops, r, e),
                                   ("eng_gap", eng_tops, e, r)):
        g, bound = _gap(tops[step] if tops is not None and step < len(tops) else None, mine, other)
        out[key] = g
        if bound:
            out[key + "_bound"] = ">="
        if g is not None:
            known.append(g if not bound else math.inf)                     # a bound cannot show a tie
    out["near_tie"] = bool(known) and all(g <= tie_nats + 1e-6 for g in known)   # logprobs carry float noise
    # the reference's own teacher-forced pass prefers (or ties) the engine's token: it disagrees with itself here
    out["ref_tf_prefers_engine"] = out["ref_tf_gap"] is not None and "ref_tf_gap_bound" not in out \
        and out["ref_tf_gap"] <= 1e-6
    return out


def compare(eng_ids: list[int], eng_tops: list[dict], ref_ids: list[int], ref_tops: list[dict],
            ref_free_tops: list[dict] | None = None, tie_nats: float = 0.5) -> dict:
    """eng_tops[i] / ref_tops[i]: token -> logprob at step i on the ENGINE's path (teacher-forced reference).
    ref_free_tops[i]: the reference's own logprobs on ITS path (free running), used only for the divergence gaps."""
    n = min(len(eng_ids), len(ref_tops))
    div = next((i for i, (a, b) in enumerate(zip(eng_ids, ref_ids)) if a != b), None)
    if div is None and len(eng_ids) != len(ref_ids):
        div = min(len(eng_ids), len(ref_ids))
    agree = [max(ref_tops[i], key=ref_tops[i].get) == eng_ids[i] for i in range(n)]
    # 16-bit logits tie exactly: the engine's token counts as the reference's top-1 when its reference logprob
    # equals the maximum; the gap (reference max - its logprob of the engine token) is kept where they still differ
    gaps = [None if eng_ids[i] not in ref_tops[i] else max(ref_tops[i].values()) - ref_tops[i][eng_ids[i]]
            for i in range(n)]
    tied = [g is not None and g <= 1e-6 for g in gaps]
    kls = [bucket_kl(ref_tops[i], eng_tops[i]) for i in range(n)]
    return {"steps": n, "first_divergence": div, "top1_agreement": sum(agree) / n if n else None,
            "top1_agreement_ties": sum(tied) / n if n else None,
            "top1_gaps": [g for g, t in zip(gaps, tied) if not t],
            "mean_kl": sum(kls) / n if n else None, "max_kl": max(kls) if kls else None,
            "divergence": divergence(div, eng_ids, ref_ids, eng_tops, ref_tops, ref_free_tops, tie_nats)}


def reference_aa(gen: list[int], free: list[dict], own: list[dict], gen2: list[int], tie_nats: float = 0.5) -> dict:
    """The reference against itself. Decode against prefill: its free-running top-20 (`free`, in the engine's place)
    against its own teacher-forced top-20 on the same tokens (`own`), compared as the engine is (KL(own || free),
    top-1 = argmax(own) equals its token). Run to run: the first step where a second free run `gen2` leaves `gen`
    (None: identical)."""
    c = compare(gen, free, gen, own, tie_nats=tie_nats)
    run2 = next((i for i, (a, b) in enumerate(zip(gen, gen2)) if a != b), None)
    if run2 is None and len(gen) != len(gen2):
        run2 = min(len(gen), len(gen2))
    c["run_to_run_first_divergence"] = run2
    return c


def _gap_summary(gaps: list) -> dict:
    """The top-1 disagreement gaps (nats; None: the token is missing from the reference's top-20), pooled."""
    exact = sorted(g for g in gaps if g is not None)
    return {"n": len(gaps), "missing_from_ref_top20": len(gaps) - len(exact), "max": exact[-1] if exact else None,
            "le_0.125": sum(g <= 0.125 + 1e-6 for g in exact), "le_0.5": sum(g <= 0.5 + 1e-6 for g in exact),
            "sorted": exact}


def _pooled(items: list[dict], key: str) -> float:
    """A per-prompt metric averaged over all compared steps."""
    steps = sum(p["steps"] for p in items) or 1
    return sum((p[key] or 0) * p["steps"] for p in items) / steps


@contextlib.contextmanager
def _cwd_off_path():
    """Keep the run directory off sys.path while transformers loads: a directory named `kernels` there (yours,
    next to ikengine/) would shadow the Hub `kernels` package that transformers probes for (as serve.py does)."""
    here = os.path.abspath(os.getcwd())
    saved = list(sys.path)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]
    try:
        yield
    finally:
        sys.path[:] = saved


# --- the engine ---------------------------------------------------------------------------------------
def run_engine(args, prompts: list[list[int]]) -> tuple[list[dict], float, dict]:
    import torch
    from ..engine import build_engine
    from ..engine_api import SamplingParams
    eng = build_engine(args.model, device=args.device, max_seq=args.max_seq, act=args.act, kv=args.kv)
    greedy = SamplingParams(temperature=0.0)
    runs = []
    for ids in prompts:
        outs = list(eng.stream(ids, args.tokens, greedy, TOP, {}))
        runs.append({"ids": [o.token_id for o in outs], "tops": [dict(o.top) for o in outs]})
    list(eng.stream(prompts[0], 4, greedy, None, {}))                         # untimed warm-up of the fast path
    toks, secs = 0, 0.0
    for ids in prompts:
        gen = eng.stream(ids, args.tokens, greedy, None, {})
        next(gen)                       # prefill's token: the engine queues no decode step before sending it,
        t0 = time.perf_counter()        # so the clock starts with the device idle and times decode steps only
        toks += sum(1 for _ in gen)
        secs += time.perf_counter() - t0
    config = eng.config()
    del eng
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return runs, (toks / secs if secs else 0.0), config


def read_engine_runs(path: str) -> dict:
    """A file --save-engine wrote: prompts (token ids), labels, tokens, runs (ids and top-20 per step),
    decode_tok_s and the engine's config. JSON keys are strings; the top-20 token ids come back as ints."""
    with open(path) as f:
        saved = json.load(f)
    saved["runs"] = [{"ids": r["ids"], "tops": [{int(t): v for t, v in d.items()} for d in r["tops"]]}
                     for r in saved["runs"]]
    saved["labels"] = saved.get("labels") or [f"prompt{i}" for i in range(len(saved["prompts"]))]
    return saved


def _engine_runs(args, prompts: list[list[int]], labels: list[str] | None = None):
    """Run the engine (writing its runs to --save-engine), or reuse the runs --load-engine saved for these prompts
    (the saved ones, or the first of them, token id for token id) and --tokens."""
    if args.load_engine:
        saved = read_engine_runs(args.load_engine)
        if saved["prompts"][:len(prompts)] != prompts:
            raise ValueError(f"{args.load_engine} was saved for other prompts")
        if saved.get("tokens") != args.tokens:
            raise ValueError(f"{args.load_engine} was saved for --tokens {saved.get('tokens')}, not {args.tokens}")
        if saved.get("model") and os.path.normpath(str(saved["model"])) != os.path.normpath(str(args.model)):
            print(f"note: {args.load_engine} was saved for --model {saved['model']}, not {args.model}; check that "
                  "both name the same checkpoint", file=sys.stderr)
        return saved["runs"][:len(prompts)], saved["decode_tok_s"], saved["engine"]
    runs, tok_s, config = run_engine(args, prompts)
    if args.save_engine:
        with open(args.save_engine, "w") as f:
            json.dump({"prompts": prompts, "labels": labels or [f"prompt{i}" for i in range(len(prompts))],
                       "tokens": args.tokens, "model": args.model, "decode_tok_s": tok_s, "engine": config,
                       "runs": runs, "label": OBSERVED}, f)
    return runs, tok_s, config


# --- references ---------------------------------------------------------------------------------------
def _load_hf(model_dir: str, device: str):
    from transformers import AutoModelForCausalLM
    dtype_kw = {"dtype": "auto"}
    try:
        return AutoModelForCausalLM.from_pretrained(model_dir, device_map=device, **dtype_kw)
    except TypeError:                                                          # transformers before 4.56
        dtype_kw = {"torch_dtype": "auto"}
        return AutoModelForCausalLM.from_pretrained(model_dir, device_map=device, **dtype_kw)
    except (ImportError, ValueError):                                          # no accelerate for device_map
        return AutoModelForCausalLM.from_pretrained(model_dir, **dtype_kw).to(device)


def _hf_forced(model, ids: list[int], cont: list[int], device) -> list[dict]:
    """The model's top-20 logprobs at every position of `cont` after `ids`, in one forward pass."""
    import torch
    lp = torch.log_softmax(model(torch.tensor([ids + cont], device=device)).logits[0, len(ids) - 1:-1].float(), -1)
    vals, idx = lp.topk(TOP, dim=-1)
    return [dict(zip(i.tolist(), v.tolist())) for i, v in zip(idx, vals)]


def _hf_greedy(model, ids: list[int], tokens: int, device) -> tuple[list[int], list[dict]]:
    """Plain greedy decode with the KV cache (no generation_config processors, no EOS stop): (ids, top-20 per step)."""
    import torch
    gen, free, past, cur = [], [], None, torch.tensor([ids], device=device)
    for _ in range(tokens):
        o = model(cur, past_key_values=past, use_cache=True)
        past = o.past_key_values
        lp1 = torch.log_softmax(o.logits[0, -1].float(), dim=-1)
        v1, i1 = lp1.topk(TOP)
        free.append(dict(zip(i1.tolist(), v1.tolist())))
        gen.append(int(lp1.argmax()))
        cur = torch.tensor([[gen[-1]]], device=device)
    return gen, free


def ref_hf(args, prompts, runs) -> list[dict]:
    import torch
    device = args.ref_device or args.device
    with _cwd_off_path():
        model = _load_hf(args.model, device)
    model.eval()
    out = []
    with torch.no_grad():
        for ids, run in zip(prompts, runs):
            gen, free = _hf_greedy(model, ids, args.tokens, device)
            rec = {"ids": gen, "tops": _hf_forced(model, ids, run["ids"], device), "free_tops": free,
                   "teacher_forced": True}
            if getattr(args, "ref_aa", False):          # A/A: its cached decode against one pass on the same tokens
                rec["aa"] = reference_aa(gen, free, _hf_forced(model, ids, gen, device),
                                         _hf_greedy(model, ids, args.tokens, device)[0], args.tie_nats)
            out.append(rec)
    return out


def _api_key(env_name: str | None) -> str | None:
    """The reference key from the environment variable `env_name` (never from the command line)."""
    return os.environ.get(env_name) if env_name else None


def _request(url: str, body: dict | None, key_env: str | None) -> urllib.request.Request:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    key = _api_key(key_env)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    return req


def _post(url: str, body: dict, key_env: str | None) -> dict:
    with urllib.request.urlopen(_request(url, body, key_env), timeout=600) as r:
        return json.loads(r.read())


def _served_name(args, base: str) -> str:
    if args.ref_model:
        return args.ref_model
    with urllib.request.urlopen(_request(base + "/v1/models", None, args.api_key_env), timeout=60) as r:
        return json.loads(r.read())["data"][0]["id"]


def _vllm_free(args, base, name, ids):
    """/v1/completions, greedy, no EOS stop: (ids, top-20 per step on its own path)."""
    free = _post(base + "/v1/completions", {"model": name, "prompt": ids, "max_tokens": args.tokens,
                                            "temperature": 0, "logprobs": TOP, "return_tokens_as_token_ids": True,
                                            "ignore_eos": True}, args.api_key_env)
    lp = free["choices"][0].get("logprobs") or {}
    gen = [int(str(t).split(":")[-1]) for t in lp.get("tokens") or []]
    tops = [{int(str(t).split(":")[-1]): v for t, v in (d or {}).items()} for d in lp.get("top_logprobs") or []]
    return gen, [dict(sorted(d.items(), key=lambda kv: -kv[1])[:TOP]) for d in tops]


def _vllm_forced(args, base, name, ids, cont):
    """vLLM's prompt_logprobs on prompt + `cont`: entry len(ids) + i scores token i of `cont`."""
    tf = _post(base + "/v1/completions", {"model": name, "prompt": ids + cont, "max_tokens": 1,
                                          "temperature": 0, "prompt_logprobs": TOP}, args.api_key_env)
    plp = tf["choices"][0].get("prompt_logprobs")
    if not plp:
        raise ValueError("no prompt_logprobs in the response")
    tops = [{int(t): (v["logprob"] if isinstance(v, dict) else float(v)) for t, v in d.items()}
            for d in plp[len(ids):]]
    return [dict(sorted(d.items(), key=lambda kv: -kv[1])[:TOP]) for d in tops]


def _sglang_top(entries) -> dict:
    """SGLang's top-logprob list [[logprob, token_id, text], ...] -> {token_id: logprob}."""
    return {int(e[1]): float(e[0]) for e in (entries or []) if e is not None and e[0] is not None}


def _sglang_free(args, base, ids):
    """SGLang native /generate, greedy, no EOS stop: (ids, top-20 per step on its own path)."""
    r = _post(base + "/generate", {"input_ids": ids, "return_logprob": True, "top_logprobs_num": TOP,
                                   "logprob_start_len": -1,
                                   "sampling_params": {"temperature": 0, "max_new_tokens": args.tokens,
                                                       "ignore_eos": True}}, args.api_key_env)
    meta = r.get("meta_info") or {}
    gen = [int(e[1]) for e in meta.get("output_token_logprobs") or []] or [int(t) for t in r.get("output_ids") or []]
    return gen, [_sglang_top(t) for t in meta.get("output_top_logprobs") or []]


def _sglang_forced(args, base, ids, cont):
    """SGLang native /generate on prompt + `cont`. SGLang returns input logprobs from position logprob_start_len
    on, the first entry empty (observed on 0.5.21: [None, id, None] and a null top list), so the request starts
    one token early, at the last prompt token: entry 1 + i is the logprob of token i of `cont` given everything
    before it. The token ids SGLang echoes are checked against [last prompt token] + `cont`, so a shifted
    alignment raises instead of comparing the wrong positions."""
    start = len(ids) - 1
    r = _post(base + "/generate", {"input_ids": ids + cont, "return_logprob": True, "top_logprobs_num": TOP,
                                   "logprob_start_len": start,
                                   "sampling_params": {"temperature": 0, "max_new_tokens": 1}}, args.api_key_env)
    meta = r.get("meta_info") or {}
    toks = [int(e[1]) for e in meta.get("input_token_logprobs") or []]
    tops = (meta.get("input_top_logprobs") or [])[1:1 + len(cont)]
    want = [int(ids[-1])] + [int(t) for t in cont]
    if toks[:len(want)] != want or len(tops) < len(cont) or not all(tops):
        raise ValueError(f"SGLang input logprobs do not line up with the engine tokens (got {len(toks)} entries, "
                         f"first ids {toks[:3]} vs {want[:3]})")
    return [_sglang_top(t) for t in tops]


def ref_openai(args, prompts, runs) -> list[dict]:
    """A reference server. kind vllm: /v1/completions (+ prompt_logprobs); sglang: native /generate;
    auto: vLLM's request first, SGLang's when the server has no prompt_logprobs."""
    base = args.ref.rstrip("/")
    kind = getattr(args, "ref_kind", "auto") or "auto"
    name = None if kind == "sglang" else _served_name(args, base)
    out = []
    for ids, run in zip(prompts, runs):
        forced = None
        if kind == "sglang":
            gen, free = _sglang_free(args, base, ids)
            forced = _sglang_forced(args, base, ids, run["ids"])
        else:
            gen, free = _vllm_free(args, base, name, ids)
            try:                                                               # vLLM: teacher-forced on our path
                forced = _vllm_forced(args, base, name, ids, run["ids"])
            except Exception as e:  # noqa: BLE001
                if kind == "vllm":
                    print(f"note: no prompt_logprobs from the reference ({type(e).__name__}); using its own path",
                          file=sys.stderr)
                else:
                    try:
                        forced = _sglang_forced(args, base, ids, run["ids"])
                        kind = "sglang"
                        gen, free = _sglang_free(args, base, ids)           # same API for both paths
                    except Exception as e2:  # noqa: BLE001
                        print(f"note: no teacher-forced logprobs from the reference ({type(e).__name__}, "
                              f"{type(e2).__name__}); using its own path", file=sys.stderr)
        if forced is None:          # other servers: compare only the steps before the divergence, on its own path
            if getattr(args, "ref_aa", False):
                raise ValueError("--ref-aa needs teacher-forced logprobs, which this reference did not give")
            div = next((i for i, (a, b) in enumerate(zip(run["ids"], gen)) if a != b), len(gen))
            out.append({"ids": gen, "tops": free[:div], "free_tops": free, "teacher_forced": False})
            continue
        rec = {"ids": gen, "tops": forced, "free_tops": free, "teacher_forced": True}
        if getattr(args, "ref_aa", False):                                     # A/A: the reference against itself
            own = (_sglang_forced(args, base, ids, gen) if kind == "sglang"
                   else _vllm_forced(args, base, name, ids, gen))
            gen2 = (_sglang_free(args, base, ids) if kind == "sglang" else _vllm_free(args, base, name, ids))[0]
            rec["aa"] = reference_aa(gen, free, own, gen2, getattr(args, "tie_nats", 0.5))
        out.append(rec)
    args.ref_kind_used = kind
    return out


# --- prompts ------------------------------------------------------------------------------------------
def _split(v) -> list[str] | None:
    return [x for x in str(v).split(",") if x] if v else None


def read_prompt_file(path: str, per_bucket: int | None = None, buckets: list[str] | None = None
                     ) -> tuple[list, list[str]]:
    """Prompts from a text file (one per line) or JSONL, optionally gzipped, and a label for each (a record's "id",
    else its line). A JSONL record gives "prompt" (text) or "messages" (a chat, kept as the message list).
    buckets: keep only records of these buckets, in this order. per_bucket K: the first K records of each
    "bucket" (records without one are kept). Returns (prompts, labels)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        lines = [ln for ln in f if ln.strip()]
    recs = []
    for i, ln in enumerate(lines):
        rec = json.loads(ln) if ln.lstrip().startswith("{") else {"prompt": ln.rstrip("\n")}
        recs.append((str(rec.get("id") or f"line{i + 1}"), rec))
    if buckets:
        recs = [x for b in buckets for x in recs if x[1].get("bucket") == b]
    if per_bucket is not None:
        seen: dict = {}
        kept = []
        for x in recs:
            b = x[1].get("bucket")
            seen[b] = seen.get(b, 0) + 1
            if b is None or seen[b] <= per_bucket:
                kept.append(x)
        recs = kept
    return [r["messages"] if "messages" in r else r["prompt"] for _, r in recs], [name for name, _ in recs]


def load_prompts(args, labels: list | None = None) -> list[list[int]]:
    """Token ids for the prompts the flags name; `labels`, when given, gets one name per prompt."""
    if args.prompt_ids:
        with open(args.prompt_ids) as f:
            out = [list(map(int, p)) for p in json.load(f)]
        if labels is not None:
            labels.extend(f"ids{i}" for i in range(len(out)))
        return out
    texts, names = PROMPTS, [f"builtin{i}" for i in range(len(PROMPTS))]
    if args.prompts:
        texts, names = read_prompt_file(args.prompts, getattr(args, "per_bucket", None),
                                        _split(getattr(args, "buckets", None)))
    if labels is not None:
        labels.extend(names)
    with _cwd_off_path():
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(getattr(args, "tokenizer", None) or args.model)
    if getattr(args, "chat_template", None):                                 # the template the server is given
        with open(args.chat_template) as f:
            tok.chat_template = f.read()
    chat = args.chat or any(isinstance(t, list) for t in texts)              # message lists are chats
    kw = json.loads(args.chat_template_kwargs or "{}") if chat else None
    return tokenize(tok, texts, kw)


def tokenize(tok, texts: list, chat_kwargs: dict | None = None) -> list[list[int]]:
    """Token ids as the skeleton's server makes them. Plain prompts: encode with the special tokens (BOS).
    Chat (chat_kwargs not None; a text is one user turn): render the template to text, then encode without adding
    special tokens, because the template holds them (serve.py and vLLM's chat path do the same). Rendering to text
    also avoids apply_chat_template(tokenize=True), which returns a BatchEncoding, not ids, in transformers 5.
    A prompt that is a list of messages is always a chat."""
    out = []
    for t in texts:
        if chat_kwargs is None and isinstance(t, str):
            out.append([int(i) for i in tok.encode(t)])
            continue
        msgs = t if isinstance(t, list) else [{"role": "user", "content": t}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **(chat_kwargs or {}))
        out.append([int(i) for i in tok.encode(text, add_special_tokens=False)])
    return out


# --- main ---------------------------------------------------------------------------------------------
def _fmt_gap(d: dict, key: str) -> str:
    v = d.get(key)
    return "-" if v is None else d.get(key + "_bound", "") + f"{round(v, 3):g}"    # 1.125 stays 1.125


def _tie_word(d: dict) -> str:
    if d["near_tie"]:
        return "tie"
    return "NOT tie but ref teacher-forced prefers eng" if d.get("ref_tf_prefers_engine") else "NOT tie"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ikengine.tools.validate_model", description=__doc__.split("\n\n")[0],
                                 allow_abbrev=False)          # "--api-key KEY" must not pass as --api-key-env
    ap.add_argument("--model", required=True, help="checkpoint directory")
    ap.add_argument("--ref", default="hf", help="'hf' (transformers), a vLLM or SGLang server URL, or 'none' (the "
                    "engine only, saved with --save-engine)")
    ap.add_argument("--ref-kind", default="auto", choices=["auto", "vllm", "sglang"],
                    help="how to ask a reference server for teacher-forced logprobs (default: vLLM's, then SGLang's)")
    ap.add_argument("--ref-model", help="served model name at --ref (default: the first one listed)")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY",
                    help="name of the environment variable holding the --ref server's key (value never recorded)")
    ap.add_argument("--ref-device", help="device for the hf reference (default: --device)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--act", default="as_shipped", choices=["as_shipped", "a16"])
    ap.add_argument("--kv", default="as_shipped", choices=["as_shipped", "bf16"])
    ap.add_argument("--tokens", type=int, help=f"greedy tokens per prompt (default {TOKENS}; a loaded run's own)")
    ap.add_argument("--n", type=int, help="use the first N prompts (default: all)")
    ap.add_argument("--prompts", help="text file (one prompt per line), or JSONL (.gz too) with 'prompt' or 'messages'")
    ap.add_argument("--buckets", help="with a JSONL prompt file: only these buckets, comma-separated, in this order")
    ap.add_argument("--per-bucket", type=int, help="with a JSONL prompt file: the first K records of each bucket")
    ap.add_argument("--prompt-ids", help="JSON list of token-id lists (skips the tokenizer)")
    ap.add_argument("--chat", action="store_true", help="wrap each prompt in the chat template")
    ap.add_argument("--chat-template-kwargs", help="JSON, e.g. '{\"enable_thinking\": false}'")
    ap.add_argument("--chat-template", help="chat template file (as the server's --chat-template); default: the "
                    "tokenizer's own")
    ap.add_argument("--tokenizer", help="tokenizer directory (default: --model)")
    ap.add_argument("--min-top1", type=float, default=0.98, help="fail below this mean top-1 agreement")
    ap.add_argument("--max-kl", type=float, default=0.02, help="fail above this mean KL")
    ap.add_argument("--require-identical", action="store_true", help="fail on any greedy divergence")
    ap.add_argument("--tie-nats", type=float, default=0.5,
                    help="a first divergence is a near tie when every gap there is at most this (nats); bf16 "
                         "logits move in steps of 0.125-0.25 nats")
    ap.add_argument("--ref-aa", action="store_true",
                    help="also compare the reference's free-running logprobs with its own teacher-forced ones on "
                         "the same tokens (its A/A floor)")
    saved_runs = ap.add_mutually_exclusive_group()
    saved_runs.add_argument("--save-engine", "--engine-out", dest="save_engine", metavar="PATH",
                            help="write the engine's prompts, tokens, top-20 logprobs and speed here (JSON)")
    saved_runs.add_argument("--load-engine", "--engine-in", dest="load_engine", metavar="PATH",
                            help="compare runs --save-engine wrote instead of building the engine, so the engine "
                                 "and a reference server never share the GPU; its prompts and --tokens by default "
                                 "(prompt flags given with it must select the same prompts, or the first of them)")
    ap.add_argument("--out", help="write the full report as JSON here")
    args = ap.parse_args(argv)
    label = OBSERVED
    try:
        saved = read_engine_runs(args.load_engine) if args.load_engine else None
        if saved is not None and not (args.prompts or args.prompt_ids):  # the prompts the run was saved with
            prompts, labels = saved["prompts"], saved["labels"]
        else:
            labels: list[str] = []
            prompts = load_prompts(args, labels)
        if args.n:
            prompts, labels = prompts[: args.n], labels[: args.n]
        if args.tokens is None:
            args.tokens = saved["tokens"] if saved is not None else TOKENS
        runs, tok_s, config = _engine_runs(args, prompts, labels)
        if saved is not None:
            label = f"observed (engine: the run saved in {args.load_engine}; reference: this run)"
        if args.ref == "none":
            print(f"engine only: {len(runs)} prompts x {args.tokens} tokens; decode {tok_s:.1f} tok/s (observed"
                  + (", saved run)" if saved is not None else ")")
                  + (f"; saved to {args.save_engine}" if args.save_engine else ""))
            return 0
        refs = ref_hf(args, prompts, runs) if args.ref == "hf" else ref_openai(args, prompts, runs)
    except Exception as e:  # noqa: BLE001
        print(f"validate_model: setup failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    per = [compare(r["ids"], r["tops"], f["ids"], f["tops"], f.get("free_tops"), args.tie_nats)
           for r, f in zip(runs, refs)]
    for p, name, ids, f in zip(per, labels, prompts, refs):
        p["prompt"], p["prompt_tokens"], p["teacher_forced"] = name, len(ids), f.get("teacher_forced", False)
        if "aa" in f:
            p["reference_aa"] = {k: f["aa"][k] for k in ("steps", "top1_agreement", "top1_agreement_ties",
                                                         "top1_gaps", "mean_kl", "max_kl",
                                                         "run_to_run_first_divergence")}
    top1, top1_ties, kl = (_pooled(per, k) for k in ("top1_agreement", "top1_agreement_ties", "mean_kl"))
    identical = sum(p["first_divergence"] is None for p in per)
    divs = [p["divergence"] for p in per if p["divergence"]]
    ties = sum(d["near_tie"] for d in divs)
    self_disagree = sum(d["ref_tf_prefers_engine"] and not d["near_tie"] for d in divs)
    off = [g for p in per for g in p["top1_gaps"]]
    aa = [p["reference_aa"] for p in per if "reference_aa" in p]
    fails = []
    if top1 < args.min_top1:
        fails.append(f"top-1 agreement {top1:.4f} < {args.min_top1}")
    if kl > args.max_kl:
        fails.append(f"mean KL {kl:.5f} > {args.max_kl}")
    if args.require_identical and identical < len(per):
        fails.append(f"greedy paths differ on {len(per) - identical} of {len(per)} prompts")
    report = {"verdict": "FAIL" if fails else "PASS", "fails": fails, "reference": args.ref,
              "reference_kind": getattr(args, "ref_kind_used", "hf" if args.ref == "hf" else args.ref_kind),
              "prompts": len(per), "tokens": args.tokens, "steps_compared": sum(p["steps"] for p in per),
              "teacher_forced_prompts": sum(p["teacher_forced"] for p in per),
              "identical_greedy": identical, "divergences": len(divs), "near_tie_divergences": ties,
              "not_tie_ref_tf_prefers_engine": self_disagree,
              "tie_nats": args.tie_nats, "top1_agreement": top1, "top1_agreement_ties": top1_ties,
              "top1_disagreement_gaps": _gap_summary(off),
              "mean_kl": kl, "max_kl": max((p["max_kl"] or 0) for p in per) if per else None,
              "reference_aa": ({"prompts": len(aa), "steps": sum(a["steps"] for a in aa),
                                "run_to_run_identical": sum(a["run_to_run_first_divergence"] is None for a in aa),
                                "top1_agreement": _pooled(aa, "top1_agreement"),
                                "top1_agreement_ties": _pooled(aa, "top1_agreement_ties"),
                                "top1_disagreement_gaps": _gap_summary([g for a in aa for g in a["top1_gaps"]]),
                                "mean_kl": _pooled(aa, "mean_kl"), "max_kl": max(a["max_kl"] or 0 for a in aa)}
                               if aa else None),
              "decode_tok_s": tok_s, "label": label, "engine": config, "per_prompt": per}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    gaps = ", ".join(f"{d['step']}:{_tie_word(d)} (ref {_fmt_gap(d, 'ref_tf_gap')}/"
                     f"{_fmt_gap(d, 'ref_free_gap')}, eng {_fmt_gap(d, 'eng_gap')})" for d in divs)
    a = report["reference_aa"]
    print(f"{report['verdict']}: {identical}/{len(per)} greedy paths identical (first divergence per prompt: "
          f"{[p['first_divergence'] for p in per]})"
          + (f"; near ties {ties}/{len(divs)} (every gap <= {args.tie_nats} nats; gaps: ref teacher-forced/free, "
             f"eng) [{gaps}]" if divs else "")
          + f"; top-1 agreement {top1:.4f} ({top1_ties:.4f} counting exact ties); mean KL {kl:.5f}"
          + (f"; reference A/A top-1 {a['top1_agreement']:.4f} ({a['top1_agreement_ties']:.4f} counting exact "
             f"ties), mean KL {a['mean_kl']:.5f}, run to run {a['run_to_run_identical']}/{a['prompts']} identical"
             if a else "")
          + f"; decode {tok_s:.1f} tok/s (observed{', saved run' if saved is not None else ''})"
          + (f"; {'; '.join(fails)}" if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
