"""The first-token distribution a server's own log-probabilities imply under the request's sampling settings.

``quality sampling`` asks a server for its top-20 log-probabilities of the first generated token (raw
log-softmax, before temperature, top-k and top-p, as vLLM returns them and the server contract requires),
builds the distribution its sampler should draw from, and tests the server's seeded draws against it.

Order of the cuts (Hugging Face and vLLM): temperature, then top-k, then top-p on the top-k set
renormalised, with min-p checked both before and after top-p (vLLM and Hugging Face differ on where
min-p goes; a prompt counts only when both orders keep the same tokens). SGLang ("joint") takes top-p
over the whole distribution and the first top_k tokens by position, then min-p; its kept set can only
be larger. A prompt counts only when that order keeps the same tokens too, so a correct sampler in
either order passes.

Only 20 tokens are listed, so the distribution is known exactly only when the kept set provably lies
inside them. ``implied`` proves that or says why not:

* top_k < K (the number listed): the top-k set is the first top_k listed tokens; a near-tie across
  that boundary is ambiguous.
* top_k >= K, or off: unlisted tokens may sit in the top-k set. Each has a log-probability <= the
  K-th's and together they hold at most R of the raw mass, R = 1 - the listed mass with every
  listed log-probability lowered by ``tol``. Hugging Face and vLLM keep every token tied with the
  k-th value, so there are top_k - K of them plus any number of ties: at most top_k - K + R / p_K
  (p_K the K-th's probability), and no count bound above temperature 1 with top_k > K (a tie at a
  low k-th value can add any number of small tokens) or with no top-k. ``unseen_bound`` turns that
  into the most tempered weight they can add; the top-p cut must be the same for every unlisted
  weight from 0 to that bound. A bound below ``UNSEEN_EPS`` of the kept weight is ignored.
* Every top-p and min-p comparison must hold with logits moved by ``tol`` (logprobs are rounded in
  JSON, and two identical requests can differ slightly); a cut between two tokens whose
  log-probabilities are within ``tol`` is ambiguous (the engine may keep either).

Exclusion reasons (``implied(...)["status"]``): ``beyond_top`` (the kept set may reach past the
listed tokens), ``boundary`` (a cut within the tolerance), ``min_p_order`` (the two min-p orders
keep different tokens), ``top_p_order`` (top-p over the top-k set and over the whole distribution
keep different tokens), ``bad_logprobs`` (none, or not log-probabilities).

``expected_tv`` is the mean of TV(empirical, p) when n draws come from p exactly, the parametric
bootstrap's mean in closed form: TV is half the sum over tokens of |count/n - p|, each count is
Binomial(n, p_t), and E|X - np| has De Moivre's closed form, so no Monte Carlo noise enters.
"""

from __future__ import annotations

import math
from typing import Hashable, Mapping, Sequence

LOGPROB_TOL = 1e-3  # nats: smallest logit tolerance for the cut comparisons
UNSEEN_EPS = 1e-4  # unlisted weight below this share of the kept weight is treated as absent
GREEDY_T = 1e-5  # vLLM samples greedily below this temperature
_R_EPS = 1e-9  # unlisted raw mass below this: the listed tokens are the whole vocabulary


def sampling_settings(extra: Mapping) -> tuple[dict, list[str]]:
    """{temperature, top_k, top_p, min_p} from a request body (None = off), and the fields not sent (those
    vLLM fills from generation_config.json when absent).

    temperature is required (a server's default is not knowable from outside); top_k <= 0 or -1, top_p >= 1
    and min_p <= 0 are off. Raises ValueError on a missing temperature or a value of the wrong type.
    """
    def num(key, cast):
        v = extra.get(key)
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{key} must be a number, got {v!r}")
        return cast(v)

    t = num("temperature", float)
    if t is None:
        raise ValueError("temperature is not set: the check needs the sampling settings the draws use")
    if t < 0 or not math.isfinite(t):
        raise ValueError(f"temperature must be >= 0, got {t}")
    k = num("top_k", float)
    if k is not None and k != int(k):
        raise ValueError(f"top_k must be an integer, got {k}")
    p, mp = num("top_p", float), num("min_p", float)
    if p is not None and not 0 < p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {p}")
    if mp is not None and not 0 <= mp <= 1:
        raise ValueError(f"min_p must be in [0, 1], got {mp}")
    out = {"temperature": t, "top_k": int(k) if k is not None and k > 0 else None,
           "top_p": p if p is not None and p < 1 else None, "min_p": mp if mp else None}
    return out, [f for f in ("top_k", "top_p", "min_p", "repetition_penalty") if f not in extra]


def unseen_bound(rest: float, p_last: float, p_top: float, temperature: float, m: float) -> float:
    """Largest total tempered weight (relative to the top token's) of at most ``m`` unlisted tokens, each with
    raw probability <= ``p_last`` and together <= ``rest``. inf when it has no bound (T > 1 with no top-k)."""
    if rest <= 0 or m <= 0 or p_last <= 0:
        return 0.0
    a = 1.0 / temperature
    if a >= 1:  # convex: as many at the cap as the mass allows
        return (p_last / p_top) ** a * min(m, rest / p_last)
    if math.isinf(m):  # concave: spreading the mass over more tokens raises the sum without limit
        return math.inf
    return m * (min(rest / m, p_last) / p_top) ** a


def _cut_range(w: Sequence[float], n_set: int, u_max: float, top_p: float | None, t: float) -> tuple:
    """Top-p over the first ``n_set`` weights plus 0..``u_max`` of unlisted weight sorted after them.

    Returns (lo, hi): the fewest and the most listed tokens the nucleus can keep when every logit moves by up
    to ``t`` (in tempered units) and the unlisted weight is anywhere in its range; hi is None when the
    nucleus may not close inside the listed tokens. A token j is kept when the mass before it is < top_p.
    """
    if top_p is None:
        return (n_set, n_set) if u_max == 0 else (n_set, None)
    total = sum(w[:n_set])
    e = math.exp(2 * t)
    lo = hi = None
    acc = 0.0
    for j in range(n_set):
        acc += w[j]
        rest = total - acc
        # the nucleus closes at token j+1 (1-based) at the latest / earliest moves of the inputs
        if lo is None and acc / (acc + rest / e) >= top_p:
            lo = j + 1
        if hi is None and acc / (acc + (rest + u_max) * e) >= top_p:
            hi = j + 1
            break
    if lo is None:  # rounding: the listed set always closes when nothing unlisted is in it
        lo = n_set
    return lo, hi


def implied(top: Sequence[tuple[Hashable, float]], temperature: float, top_k: int | None = None,
            top_p: float | None = None, min_p: float | None = None, tol: float = LOGPROB_TOL) -> dict:
    """The kept tokens and their probabilities, from the listed (key, raw logprob) pairs, or why not provable.

    ``top`` is the server's top-K list for one position (any order). Returns {"status": "ok", "probs":
    {key: p}, "kept": n, "listed": K, "max_p", "unseen_bound"} or {"status": <reason>, "detail": text}.
    """
    items = [(k, float(v)) for k, v in top if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not items or len(items) != len(top) or any(math.isnan(v) or v > 1e-6 for _, v in items):
        return {"status": "bad_logprobs", "detail": "missing, non-numeric or positive log-probabilities"}
    items.sort(key=lambda kv: -kv[1])  # stable: exact ties keep the server's order
    keys, lp = [k for k, _ in items], [v for _, v in items]
    big_k = len(lp)
    tol = max(tol, 0.0)
    listed = sum(math.exp(v) for v in lp)
    # the unlisted raw mass, at most: the listed log-probabilities may each be up to tol too high
    rest = max(0.0, 1.0 - listed * math.exp(-tol)) if 1.0 - listed > _R_EPS else 0.0

    def done(n: int, u: float, u_raw: float) -> dict:
        # a cut between two near-tied tokens is ambiguous (the engine may keep either); past the last listed
        # token the next one is unlisted, and a tie with it cannot be ruled out while unlisted weight remains
        if n < big_k and lp[n - 1] - lp[n] < tol:
            return {"status": "boundary", "detail": f"the cut after token {n} is a near-tie with token {n + 1}"}
        if n >= big_k and u > 0:
            return {"status": "beyond_top", "detail": "every listed token is kept and unlisted ones may tie"}
        z = sum(w[:n])
        probs: dict = {}
        for key, x in zip(keys[:n], w[:n]):
            probs[key] = probs.get(key, 0.0) + x / z
        return {"status": "ok", "probs": probs, "kept": n, "listed": big_k, "max_p": max(probs.values()),
                "unseen_bound": u_raw / (z + u_raw) if u_raw else 0.0}

    if temperature < GREEDY_T:  # greedy: the argmax only
        w = [1.0]
        clear = lp[0] - lp[1] >= tol if big_k > 1 else rest < math.exp(lp[0] - tol)
        if not clear:
            return {"status": "boundary", "detail": "greedy: the top two tokens are within the tolerance"}
        return done(1, 0.0, 0.0)
    t = tol / temperature
    w = [math.exp((v - lp[0]) / temperature) for v in lp]
    p_last, p_top = math.exp(lp[-1]), math.exp(lp[0])
    if top_k is not None and top_k < big_k:
        if lp[top_k - 1] - lp[top_k] < tol:
            return {"status": "boundary", "detail": f"top_k {top_k}: tokens {top_k} and {top_k + 1} are a near-tie"}
        n_k, m = top_k, 0.0
    elif top_k is None or (top_k > big_k and temperature > 1):
        n_k, m = big_k, math.inf
    else:  # top_k - K unlisted tokens, plus any number tied with the k-th value (each <= p_K, together <= rest)
        n_k, m = big_k, (top_k - big_k) + rest / p_last
    u_raw = unseen_bound(rest, p_last, p_top, temperature, m) if rest > 0 else 0.0
    u_max = 0.0 if u_raw <= UNSEEN_EPS * sum(w[:n_k]) else u_raw
    m_c, unseen_pass = n_k, u_max > 0
    if min_p is not None:
        thr = math.log(min_p)
        lw = [(v - lp[0]) / temperature for v in lp[:n_k]]
        if any(abs(x - thr) < 2 * t for x in lw):
            return {"status": "boundary", "detail": f"min_p {min_p}: a token within the tolerance of the threshold"}
        m_c = sum(1 for x in lw if x >= thr)
        unseen_pass = u_max > 0 and m_c == n_k  # an unlisted token sorts after every listed one

    def orders(tt: float) -> list[tuple[int, int | None]]:
        """(fewest, most) listed tokens kept, most None when unlisted ones may be kept too; one entry per
        min-p order: top-p then min-p (Hugging Face), and min-p then top-p (vLLM)."""
        lo, hi = _cut_range(w, n_k, u_max, top_p, tt)
        if hi is None:
            a = (min(lo, m_c), None if unseen_pass else m_c)
        else:
            a = (min(lo, m_c), min(hi, m_c))
        if min_p is None:
            return [a]
        lo, hi = _cut_range(w, min(n_k, m_c), u_max if unseen_pass else 0.0, top_p, tt)
        return [a, (lo, hi)]

    got = orders(t)
    if any(hi is None for _, hi in got):
        return {"status": "beyond_top", "detail": "the kept set may include unlisted tokens"}
    if any(lo != hi for lo, hi in got):
        moved = any(hi is None or lo != hi for lo, hi in orders(0.0))  # the unlisted weight alone moves it
        return {"status": "beyond_top" if moved else "boundary",
                "detail": "the nucleus depends on the unlisted weight" if moved
                else f"top_p {top_p}: a cumulative probability within the tolerance"}
    if len({lo for lo, _ in got}) > 1:
        return {"status": "min_p_order", "detail": f"min_p {min_p} before and after top_p keep "
                                                   f"{got[1][0]} and {got[0][0]} tokens"}
    n = got[0][0]
    if top_k is not None and top_p is not None:  # with no top-k the two orders are the same
        joint = _joint_most(w, lp, rest, top_k, top_p, min_p, temperature, t)
        if joint != n:
            return {"status": "top_p_order", "detail": f"top_p {top_p} over the top-k set keeps {n} tokens, over "
                    f"the whole distribution (SGLang) up to {'unlisted ones too' if joint is None else joint}"}
    return done(n, u_max, u_raw)


def _joint_most(w: Sequence[float], lp: Sequence[float], rest: float, top_k: int, top_p: float,
                min_p: float | None, temperature: float, t: float) -> int | None:
    """The most listed tokens SGLang's joint order can keep (None: unlisted ones too): top-p over the whole tempered
    distribution (every unlisted token's weight counts, at its largest), the first top_k by position, then min-p."""
    big_k = len(w)
    u_all = unseen_bound(rest, math.exp(lp[-1]), math.exp(lp[0]), temperature, math.inf) if rest > 0 else 0.0
    _, hi = _cut_range(w, big_k, u_all, top_p, t)
    n = (top_k if top_k <= big_k else None) if hi is None else min(hi, top_k)
    if min_p is not None:
        thr = math.log(min_p)
        n_min = sum(1 for v in lp if (v - lp[0]) / temperature >= thr)
        if n_min < big_k:  # every unlisted token is below the K-th, so below the threshold too
            n = n_min if n is None else min(n, n_min)
    return n


def binomial_mad(n: int, p: float) -> float:
    """E|X - n p| for X ~ Binomial(n, p): 2 m C(n, m) p^m (1-p)^(n-m+1) with m = floor(n p) + 1 (De Moivre)."""
    if n <= 0 or p <= 0.0 or p >= 1.0:
        return 0.0
    m = math.floor(n * p) + 1
    if m > n:
        return 0.0
    log = (math.log(2 * m) + math.lgamma(n + 1) - math.lgamma(m + 1) - math.lgamma(n - m + 1)
           + m * math.log(p) + (n - m + 1) * math.log1p(-p))
    return math.exp(log)


def expected_tv(probs: Mapping[Hashable, float], n: int) -> float:
    """Mean TV between the histogram of n draws from ``probs`` and ``probs`` itself (exact, see the module doc)."""
    return 0.5 * sum(binomial_mad(n, p) for p in probs.values()) / n if n > 0 else 0.0


def tv_to(hist: Mapping[Hashable, int], probs: Mapping[Hashable, float]) -> float | None:
    """TV between a histogram of draws and a distribution; draws outside ``probs`` count in full."""
    n = sum(hist.values())
    if n <= 0:
        return None
    keys = set(hist) | set(probs)
    return 0.5 * sum(abs(hist.get(k, 0) / n - probs.get(k, 0.0)) for k in keys)


def tv_between(a: Mapping[Hashable, float], b: Mapping[Hashable, float]) -> float:
    """TV between two distributions over the same kind of keys."""
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))
