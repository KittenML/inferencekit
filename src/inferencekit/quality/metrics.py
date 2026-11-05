"""Quality statistics: KL and TV on truncated top-k lists, position buckets, CIs, Wilson, McNemar.

KL(reference || engine) from top-k lists
----------------------------------------
Servers return only the k most likely tokens per position (k <= 20 for vLLM's default
``--max-logprobs``), so neither distribution is known in full. ``kl_topk`` builds two
comparable distributions over the same finite support:

1. Support U = the union of both top-k lists.
2. A token in U that one side did not list gets, on that side, an imputed probability
   ``min(p_min, tail / (m + 1))``: ``tail`` is that side's unlisted mass
   (1 - sum of its listed probabilities), ``m`` is how many tokens of U it is missing,
   and ``p_min`` is its smallest listed probability (an unlisted token cannot be more
   likely than the k-th listed one). The "+1" leaves a share of the tail for the rest of
   the vocabulary.
3. Everything outside U is merged into one OTHER bucket on each side: 1 - sum over U.
4. Probabilities are floored at 1e-10 and each side is renormalised, then
   KL = sum_u p(u) log(p(u) / q(u)) over U and OTHER.

Properties: identical inputs give exactly 0; when the lists cover the whole vocabulary
the result is the exact KL; merging the tail into OTHER can only lower KL (data
processing inequality), while step 2 is an approximation in either direction. Treat
the value as a well-defined, comparable divergence between engines scored the same
way, not as the exact full-vocabulary KL.

Both lists must have the same length. A top-20 reference list against a top-5 engine
list puts 15 tokens on the engine side at imputed tail values, which reads as divergence
even for identical distributions (0.03 nats on a peaked 200-token toy model, more on
real vocabularies). ``quality score`` therefore cuts both lists to the engine's k (plus
the token actually generated) before calling ``kl_topk``, and pairs the noise floor at
that same k. A smaller k makes the value coarser (less sensitive), never biased upward.

Total variation from top-k lists
--------------------------------
``tv_topk`` builds the same two distributions (union support, imputed missing entries,
one OTHER bucket) and returns TV = 1/2 * sum |p(u) - q(u)| over U and OTHER: the share of
probability mass the engine moved, between 0 and 1. Identical inputs give exactly 0, full
lists give the exact TV, and merging the tail into OTHER can only lower it. Unlike KL it
stays bounded when one side gives a token almost no mass, so a single bad imputation
cannot dominate the mean. ``tv_counts`` is the same distance between two sample
histograms (``quality sampling`` tests draws against a known distribution instead:
``quality/sampling.py``).

Confidence intervals
--------------------
Tokens inside one generation are correlated, so every CI resamples whole prompts
(cluster bootstrap) and pools the per-token values of the drawn prompts. Paired CIs
(engine minus noise floor) draw the same prompts for both arms.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction
from typing import Hashable, Mapping, Sequence

from inferencekit.stats import Z95, _draw, percentile_ci

FLOOR = 1e-10
OTHER = "__other__"
# Position buckets (token index inside the generated text, 0-based).
BUCKETS = ((0, 15), (16, 63), (64, 255), (256, None))


def bucket_label(lo: int, hi: int | None) -> str:
    return f"{lo}+" if hi is None else f"{lo}-{hi}"


BUCKET_LABELS = tuple(bucket_label(lo, hi) for lo, hi in BUCKETS)


def bucket_of(position: int) -> str:
    for lo, hi in BUCKETS:
        if position >= lo and (hi is None or position <= hi):
            return bucket_label(lo, hi)
    raise ValueError(position)


# --------------------------------------------------------------------------- KL

def _complete(known: Mapping[Hashable, float], support: set) -> dict:
    """Probabilities on ``support`` plus OTHER, from one side's listed logprobs."""
    probs = {k: math.exp(lp) for k, lp in known.items() if lp is not None}
    listed = sum(probs.values())
    tail = max(0.0, 1.0 - listed)
    missing = [k for k in support if k not in probs]
    if missing:
        cap = min(probs.values()) if probs else 1.0
        each = max(min(cap, tail / (len(missing) + 1)), FLOOR)
        for k in missing:
            probs[k] = each
    probs = {k: max(v, FLOOR) for k, v in probs.items() if k in support}
    probs[OTHER] = max(1.0 - sum(probs.values()), FLOOR)
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()}


def kl_topk(reference: Mapping[Hashable, float], engine: Mapping[Hashable, float]) -> float:
    """KL(reference || engine) in nats from two {token: logprob} top-k lists (see module doc)."""
    support = {k for k, v in reference.items() if v is not None} | {k for k, v in engine.items() if v is not None}
    p = _complete(reference, support)
    q = _complete(engine, support)
    kl = 0.0
    for k, pk in p.items():
        if pk > 0:
            kl += pk * math.log(pk / q[k])
    return max(kl, 0.0)


def kl_full(p_logprobs: Sequence[float], q_logprobs: Sequence[float]) -> float:
    """Exact KL(p || q) for two full log-distributions over the same vocabulary."""
    return sum(math.exp(a) * (a - b) for a, b in zip(p_logprobs, q_logprobs))


# --------------------------------------------------------------------------- TV

def tv_topk(reference: Mapping[Hashable, float], engine: Mapping[Hashable, float]) -> float:
    """Total variation between two {token: logprob} top-k lists, on the support ``kl_topk`` uses."""
    support = {k for k, v in reference.items() if v is not None} | {k for k, v in engine.items() if v is not None}
    p = _complete(reference, support)
    q = _complete(engine, support)
    return min(1.0, 0.5 * sum(abs(p[k] - q[k]) for k in p))


def tv_full(p_logprobs: Sequence[float], q_logprobs: Sequence[float]) -> float:
    """Exact TV for two full log-distributions over the same vocabulary."""
    return 0.5 * sum(abs(math.exp(a) - math.exp(b)) for a, b in zip(p_logprobs, q_logprobs))


def tv_counts(a: Mapping[Hashable, int], b: Mapping[Hashable, int]) -> float | None:
    """TV between the empirical distributions of two sample histograms ({outcome: count}); None if one is empty."""
    na, nb = sum(a.values()), sum(b.values())
    if na <= 0 or nb <= 0:
        return None
    return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in set(a) | set(b))


# --------------------------------------------------------------------------- CIs over prompts

def pooled(units: Sequence[tuple[float, int]]) -> float | None:
    """Pooled mean sum(sums) / sum(counts) of per-prompt (sum, count) pairs."""
    s = sum(u[0] for u in units)
    n = sum(u[1] for u in units)
    return s / n if n else None


def cluster_ci(units: Sequence[tuple[float, int]], resamples: int = 2000, seed: int = 0) -> tuple:
    """95% CI of the pooled mean, resampling prompts (each unit = one prompt's (sum, count))."""
    units = [u for u in units if u[1] > 0]
    k = len(units)
    if k < 2 or resamples <= 0:
        return (None, None)
    rng = random.Random(seed)
    reps = []
    for _ in range(resamples):
        s = n = 0.0
        for _ in range(k):
            a, b = units[_draw(rng, k)]
            s += a
            n += b
        reps.append(s / n if n else float("nan"))
    return percentile_ci(reps)


def paired_delta_ci(engine: Sequence[tuple[float, int]], floor: Sequence[tuple[float, int]],
                    resamples: int = 2000, seed: int = 0) -> dict:
    """pooled(engine) - pooled(floor) with a paired bootstrap 95% CI.

    ``engine[i]`` and ``floor[i]`` must be the same prompt; prompts where either arm has no
    tokens are dropped. Each resample draws prompt indices once and uses them for both arms.
    """
    pairs = [(e, f) for e, f in zip(engine, floor) if e[1] > 0 and f[1] > 0]
    k = len(pairs)
    out = {"n_prompts": k, "delta": None, "ci95": (None, None)}
    if k == 0:
        return out
    out["delta"] = pooled([p[0] for p in pairs]) - pooled([p[1] for p in pairs])
    if k < 2 or resamples <= 0:
        return out
    rng = random.Random(seed)
    reps = []
    for _ in range(resamples):
        es = en = fs = fn = 0.0
        for _ in range(k):
            (a, b), (c, d) = pairs[_draw(rng, k)]
            es += a
            en += b
            fs += c
            fn += d
        reps.append(es / en - fs / fn)
    out["ci95"] = percentile_ci(reps)
    return out


def margin_verdict(worse_lo: float | None, worse_hi: float | None, margin: float) -> tuple[str, str]:
    """PASS when the whole CI of "how much worse than the floor" is <= margin, FAIL when all of it is above.

    ``worse`` is oriented so that positive = the engine is worse than the noise floor.
    """
    if worse_lo is None or worse_hi is None:
        return "UNPROVEN", "no confidence interval (need >= 2 prompts scored for both the engine and the floor)"
    if worse_hi <= margin + 1e-12:
        return "PASS", f"CI [{worse_lo:.4g}, {worse_hi:.4g}] <= margin {margin:g}"
    if worse_lo > margin + 1e-12:
        return "FAIL", f"CI [{worse_lo:.4g}, {worse_hi:.4g}] > margin {margin:g}"
    return "UNPROVEN", f"CI [{worse_lo:.4g}, {worse_hi:.4g}] straddles margin {margin:g}: score more prompts"


# --------------------------------------------------------------------------- task accuracy

def wilson_ci(successes: int, n: int, z: float = Z95) -> tuple[float | None, float | None]:
    """Wilson score interval for a binomial proportion (good coverage even near 0 or 1)."""
    if n <= 0:
        return (None, None)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts.

    b = items only arm A got right, c = items only arm B got right. Under H0 (same
    accuracy) each discordant item is a fair coin, so min(b, c) ~ Binomial(b + c, 1/2).
    Computed with integers (2**-n underflows a float for n > 1074).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return float(min(Fraction(1), Fraction(2 * tail, 2 ** n)))
