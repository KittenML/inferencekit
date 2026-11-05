"""correctness: does an optimisation change what the model generates?

With greedy decoding (temperature 0) the same model and prompt should give the
same tokens. Faster kernels change floating-point summation order, so a near-tie
between the top two logits can flip, and from that position on the two sequences
follow different contexts. So the useful numbers are:
  * exact match rate: sequences identical end to end;
  * first-divergence index per sequence (later = smaller numerical difference);
  * prefix agreement: mean of (first divergence / length);
  * position-wise agreement (counts matches even after divergence; shown, not judged).
Optional: per-token logprobs within the agreeing prefix, checked against a tolerance.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from inferencekit import evidence as ev
from inferencekit.registry import Arg, OpError, operation

HIST_BINS = [(0, 0), (1, 7), (8, 31), (32, 127), (128, 511), (512, None)]


def load_sequences(store: ev.Store, ref: str) -> tuple[list[dict], dict | None]:
    """[{tokens, logprobs?}] from a JSONL file or an evidence record with data.tokens."""
    p = Path(ref)
    if p.is_file() and p.suffix in (".jsonl", ".json", ".txt"):
        seqs = []
        for n, line in enumerate(p.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                raise OpError(f"{p}:{n}: not JSON") from None
            if isinstance(obj, list):
                seqs.append({"tokens": obj})
            elif isinstance(obj, dict) and isinstance(obj.get("tokens"), list):
                seqs.append({"tokens": obj["tokens"], "logprobs": obj.get("logprobs")})
            else:
                raise OpError(f"{p}:{n}: expected a list of token ids or {{\"tokens\": [...]}}")
        return seqs, None
    rec = store.load(ref)
    tokens = rec.get("data", {}).get("tokens")
    if not tokens:
        raise OpError(f"{ref}: record has no data.tokens (run bench decode with record_tokens)")
    if tokens and isinstance(tokens[0], list):
        return [{"tokens": t} for t in tokens], rec
    return [{"tokens": tokens}], rec


def first_divergence(a: list, b: list) -> int | None:
    """Index of the first differing token; None if identical (including length)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _bin_label(lo, hi):
    return f"{lo}" if lo == hi else (f"{lo}+" if hi is None else f"{lo}-{hi}")


@operation("check_correctness", ("correctness",), "Greedy token agreement between a reference and a candidate: "
           "exact matches, first-divergence positions, optional logprob tolerance.",
           [Arg("reference", "string", "Evidence id (with data.tokens) or a JSONL file of token-id lists.",
                required=True),
            Arg("candidate", "string", "Evidence id or JSONL file, same order as the reference.", required=True),
            Arg("min_prefix_agreement", "number", "PASS needs mean prefix agreement >= this (1.0 = identical).",
                default=1.0),
            Arg("logprob_tol", "number", "Max |logprob difference| allowed inside the agreeing prefix "
                "(only when both inputs carry logprobs).")])
def check_correctness(call, reference, candidate, min_prefix_agreement, logprob_tol):
    store = ev.Store()
    ref, ref_rec = load_sequences(store, reference)
    cand, cand_rec = load_sequences(store, candidate)
    limitations, missing = [], []
    if len(ref) != len(cand):
        limitations.append(f"different sequence counts ({len(ref)} vs {len(cand)}): compared the first "
                           f"{min(len(ref), len(cand))}")
    n = min(len(ref), len(cand))
    divergences, prefix_fracs, posmatch, postotal = [], [], 0, 0
    max_lp_diff = None
    for r, c in zip(ref[:n], cand[:n]):
        a, b = r["tokens"], c["tokens"]
        d = first_divergence(a, b)
        divergences.append(d)
        length = max(len(a), len(b)) or 1
        prefix_fracs.append((length if d is None else d) / length)
        posmatch += sum(1 for x, y in zip(a, b) if x == y)
        postotal += length
        if r.get("logprobs") and c.get("logprobs"):
            upto = len(a) if d is None else d
            diffs = [abs(x - y) for x, y in zip(r["logprobs"][:upto], c["logprobs"][:upto])
                     if x is not None and y is not None]
            if diffs:
                max_lp_diff = max(max_lp_diff or 0.0, max(diffs))
    if n == 0:
        missing.append("sequences")
    exact = sum(1 for d in divergences if d is None)
    prefix = statistics.mean(prefix_fracs) if prefix_fracs else None
    diverged = [d for d in divergences if d is not None]
    hist = {}
    for lo, hi in HIST_BINS:
        hist[_bin_label(lo, hi)] = sum(1 for d in diverged if d >= lo and (hi is None or d <= hi))
    hist["identical"] = exact

    if n == 0:
        verdict, why = "UNPROVEN", "no sequences to compare"
    elif prefix is not None and prefix >= min_prefix_agreement - 1e-12:
        verdict, why = "PASS", f"mean prefix agreement {prefix:.4f} >= {min_prefix_agreement}"
    else:
        verdict, why = "FAIL", f"mean prefix agreement {prefix:.4f} < {min_prefix_agreement}"
    if logprob_tol is not None:
        if max_lp_diff is None:
            limitations.append("logprob tolerance requested but inputs carry no logprobs: not checked")
            missing.append("logprobs")
            if verdict == "PASS":
                verdict, why = "UNPROVEN", why + "; logprob tolerance not checked"
        elif max_lp_diff > logprob_tol:
            verdict, why = "FAIL", f"max logprob difference {max_lp_diff:.3g} > {logprob_tol}"
    if n == 1:
        limitations.append("a single sequence: one near-tie decides the result; use several prompts")
    for rec in (ref_rec, cand_rec):
        problem = ev.integrity_problem(rec) if rec else None
        if problem:  # an edited input must not become a fresh, verifying PASS
            limitations.append(problem)
            missing.append("input integrity")
            if verdict == "PASS":
                verdict, why = "UNPROVEN", why + "; an input failed its integrity check"

    metrics = [ev.metric("exact_match_pct", 100 * exact / n if n else None, "%", "rate", n=n),
               ev.metric("prefix_agreement_pct", 100 * prefix if prefix is not None else None, "%", "mean", n=n),
               ev.metric("positionwise_agreement_pct", 100 * posmatch / postotal if postotal else None, "%", "rate",
                         n=postotal)]
    if diverged:
        metrics.append(ev.metric("first_divergence_index", statistics.median(diverged), "tokens", "median",
                                 n=len(diverged)))
    if max_lp_diff is not None:
        metrics.append(ev.metric("max_logprob_abs_diff", max_lp_diff, "nats", "max", n=n))
    summary = f"{verdict}: {exact}/{n} sequences identical, prefix agreement " + (
        f"{100 * prefix:.1f}%" if prefix is not None else "n/a")
    ident = (cand_rec or ref_rec or {}).get("identity") or ev.identity()
    record = ev.build(call, kind="correctness", claim_class="derived", summary=summary, identity_=ident,
                      metrics=metrics, missing=missing, limitations=limitations,
                      data={"verdict": verdict, "reason": why, "sequences": n,
                            "first_divergence": divergences[:1000], "divergence_histogram": hist,
                            "reference": ref_rec["evidence_id"] if ref_rec else str(reference),
                            "candidate": cand_rec["evidence_id"] if cand_rec else str(candidate)})
    return store.save(record)
