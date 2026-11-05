"""Small, dependency-free statistics for benchmark rounds.

Benchmarks here produce few numbers (one per round, typically 5-30), so plain Python
is fast enough and keeps the logic easy to read. Randomness comes from
``random.Random(seed)`` and only through ``rng.random()``, whose sequence Python
guarantees to be reproducible for a given seed; so a CI computed today and next
year from the same data is identical.

Contents
  quantile / nearest_rank   two percentile conventions (interpolated / an actual sample)
  percentile_ci             95% percentile interval from bootstrap replicates
  bootstrap_ci              CI of a percentile, resampling whole *units* (clusters)
  paired_ratio_ci           CI of median(candidate) / median(baseline), resampling pairs
  unpaired_ratio_ci         same without pairing, for runs that cannot be paired
  rounds_for_power          how many rounds would make the CI narrow enough for decide()
  decide                    verdict for a comparison (improved / regressed / ...)
  combine_claims            PASS / FAIL / UNPROVEN / NOT RUN across several checks
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Hashable, Iterable, Sequence

Z95 = 1.959964
DEFAULT_RESAMPLES = 10_000
# A CI counts as "powered" (narrow enough to support a "no change" call) when its
# half-width is at most this fraction of the practical threshold.
POWER_RATIO = 0.75
# rounds_for_power multiplies its 1/sqrt(n) estimate by this, so that most reruns (not
# half of them) come out powered. Calibrated by simulation through paired_ratio_ci +
# decide: 85-100% of reruns powered at most noise/drift mixes (tests/test_stats.py).
POWER_MARGIN = 2.0


# --------------------------------------------------------------------------- percentiles

def quantile(xs: Sequence[float], p: float) -> float | None:
    """Linear-interpolated percentile (p in 0..100), the numpy default. For summaries."""
    if not xs:
        return None
    v = sorted(xs)
    k = (len(v) - 1) * p / 100.0
    lo = int(math.floor(k))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def nearest_rank(sorted_xs: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile: the smallest sample with at least p% of samples <= it.

    Unlike interpolation, the answer is always a value that was actually observed,
    which is what you want inside a bootstrap (no invented in-between values).
    """
    n = len(sorted_xs)
    if n == 0:
        return None
    rank = max(1, math.ceil(p / 100.0 * n))
    return sorted_xs[min(rank, n) - 1]


def summarize(xs: Iterable[float]) -> dict:
    """n / median / p95 / min / max / mean of a list (None-free)."""
    v = [float(x) for x in xs if x is not None]
    if not v:
        return {"n": 0}
    return {"n": len(v), "median": statistics.median(v), "p95": quantile(v, 95), "min": min(v), "max": max(v),
            "mean": sum(v) / len(v)}


def percentile_ci(replicates: Sequence[float]) -> tuple[float | None, float | None]:
    """95% percentile interval of bootstrap replicates.

    With B sorted replicates the bounds are element floor(0.025 (B-1)) and
    ceil(0.975 (B-1)): slightly conservative (never narrower than the exact quantiles).
    NaN in any replicate means some resample had no defined value: the CI is undefined.
    """
    reps = [float(r) for r in replicates]
    if not reps or any(math.isnan(r) for r in reps):
        return None, None
    reps.sort()
    b = len(reps)
    return reps[int(0.025 * (b - 1))], reps[int(math.ceil(0.975 * (b - 1)))]


def _draw(rng: random.Random, k: int) -> int:
    # rng.random() is the only stream Python guarantees reproducible across versions.
    return min(int(rng.random() * k), k - 1)


# --------------------------------------------------------------------------- bootstrap CIs

def bootstrap_ci(values: Sequence[float], units: Sequence[Hashable] | None = None, ps: Sequence[float] = (50,),
                 resamples: int = DEFAULT_RESAMPLES, seed: int = 0) -> dict[float, tuple[float | None, float | None]]:
    """95% CI of nearest-rank percentiles ``ps`` of ``values``, by cluster bootstrap.

    ``units[i]`` names the resampling unit of ``values[i]``. Values that share a unit
    (e.g. all per-token latencies of one round) are correlated, so resampling them
    one by one would understate the uncertainty. Instead each resample draws whole
    units with replacement (as many draws as there are units) and keeps every value
    of every drawn unit. With ``units=None`` each value is its own unit (ordinary
    bootstrap). Returns ``{p: (lo, hi)}``; ``(None, None)`` with < 2 values or units.
    """
    if units is None:
        units = list(range(len(values)))
    by_unit: dict[Hashable, list[float]] = {}
    for v, u in zip(values, units):
        by_unit.setdefault(u, []).append(float(v))
    groups = list(by_unit.values())
    k = len(groups)
    if len(values) < 2 or k < 2 or resamples <= 0:
        return {p: (None, None) for p in ps}
    rng = random.Random(seed)
    reps: dict[float, list[float]] = {p: [] for p in ps}
    for _ in range(resamples):
        sample: list[float] = []
        for _ in range(k):
            sample.extend(groups[_draw(rng, k)])
        sample.sort()
        for p in ps:
            reps[p].append(nearest_rank(sample, p))
    return {p: percentile_ci(reps[p]) for p in ps}


def paired_ratio_ci(baseline: Sequence[float], candidate: Sequence[float], resamples: int = DEFAULT_RESAMPLES,
                    seed: int = 0) -> dict:
    """Ratio of medians, candidate / baseline, with a paired bootstrap 95% CI.

    ``baseline[i]`` and ``candidate[i]`` must come from the same round i (run back to
    back). Resampling *pairs* keeps that link: anything that slowed round i for both
    arms (a thermal dip, a noisy neighbour) moves numerator and denominator together
    and largely cancels, which gives a narrower and more honest CI than resampling
    the two arms independently.
    """
    n = min(len(baseline), len(candidate))
    base, cand = [float(x) for x in baseline[:n]], [float(x) for x in candidate[:n]]
    out = {"n_pairs": n, "ratio": None, "ci95": (None, None), "log_ratio_sd": None}
    if n == 0 or statistics.median(base) == 0:
        return out
    out["ratio"] = statistics.median(cand) / statistics.median(base)
    logs = [math.log(c / b) for b, c in zip(base, cand) if b > 0 and c > 0]
    out["log_ratio_sd"] = statistics.stdev(logs) if len(logs) > 1 else (0.0 if logs else None)
    if n < 2:
        return out
    rng = random.Random(seed)
    reps = []
    for _ in range(resamples):
        idx = [_draw(rng, n) for _ in range(n)]
        mb = statistics.median([base[i] for i in idx])
        mc = statistics.median([cand[i] for i in idx])
        reps.append(mc / mb if mb > 0 else float("nan"))
    out["ci95"] = percentile_ci(reps)
    return out


def unpaired_ratio_ci(baseline: Sequence[float], candidate: Sequence[float], resamples: int = DEFAULT_RESAMPLES,
                      seed: int = 0) -> dict:
    """Ratio of medians with each arm resampled independently (no pairing available)."""
    base, cand = [float(x) for x in baseline], [float(x) for x in candidate]
    out = {"n_baseline": len(base), "n_candidate": len(cand), "ratio": None, "ci95": (None, None)}
    if not base or not cand or statistics.median(base) == 0:
        return out
    out["ratio"] = statistics.median(cand) / statistics.median(base)
    if len(base) < 2 or len(cand) < 2:
        return out
    rng = random.Random(seed)
    reps = []
    for _ in range(resamples):
        mb = statistics.median([base[_draw(rng, len(base))] for _ in base])
        mc = statistics.median([cand[_draw(rng, len(cand))] for _ in cand])
        reps.append(mc / mb if mb > 0 else float("nan"))
    out["ci95"] = percentile_ci(reps)
    return out


# --------------------------------------------------------------------------- power and verdicts

def rounds_for_power(n: int, half_width: float | None, threshold: float, power_ratio: float = POWER_RATIO,
                     margin: float = POWER_MARGIN) -> int | None:
    """Rounds per arm that would make decide() call the CI powered, from a pilot comparison.

    ``n`` rounds gave a CI with this ``half_width`` (decide()'s, on the ratio scale); the
    target is ``power_ratio * threshold``. A CI's half-width shrinks like 1/sqrt(n), so
    n * (half_width / target)^2 rounds would hit the target on average -- which means only
    about half of the reruns would. ``margin`` (2x) buys a large majority.

    Why not the textbook 1.96 * SD / sqrt(n) of per-pair log ratios: decide() tests a
    bootstrap CI of a *ratio of medians*. A median is noisier than a mean (about 1.25x the
    standard error for normal data), and slow drift cancels only partly when each arm's
    median is taken on its own. Scaling the half-width of the very CI that decide() uses
    includes both effects. (Simulated with 2-3% per-round noise: n from the textbook
    formula came out powered in only 3-15% of reruns.) With fewer than about 5 pilot rounds
    the estimate is rough.
    """
    if half_width is None or n < 1 or not threshold:
        return None
    target = power_ratio * abs(threshold)
    return max(2, int(math.ceil(margin * n * (half_width / target) ** 2 - 1e-9)))


def decide(ratio: float | None, lo: float | None, hi: float | None, threshold: float, higher_is_better: bool,
           power_ratio: float = POWER_RATIO) -> dict:
    """Turn a ratio CI into ``improved | regressed | within_noise | inconclusive``.

    Work with g = the ratio oriented so that g > 1 is better. The checks run in this
    order and the first that applies wins:
      1. no ratio or no CI                                    -> inconclusive (no data)
      2. CI entirely below 1 and g <= 1 - threshold           -> regressed
      3. CI entirely above 1 and g >= 1 + threshold           -> improved
      4. CI inside [1 - threshold, 1 + threshold] and powered -> within_noise
      5. anything else                                        -> inconclusive
    "Powered" means the CI half-width is at most power_ratio x threshold: a CI that
    happens to sit inside the band but is nearly as wide as it is not evidence of
    "no change"; more rounds are needed.
    """
    if ratio is None or lo is None or hi is None or ratio <= 0 or lo <= 0:
        return {"verdict": "inconclusive", "reason": "no ratio or no confidence interval (need >= 2 rounds per arm)"}
    if higher_is_better:
        g, g_lo, g_hi = ratio, lo, hi
    else:  # invert so that "faster" (smaller latency) is still g > 1
        g, g_lo, g_hi = 1 / ratio, 1 / hi, 1 / lo
    half_width = max(g - g_lo, g_hi - g)
    powered = half_width <= power_ratio * threshold * (1 + 1e-9)
    base = {"oriented_ratio": g, "oriented_ci95": [g_lo, g_hi], "half_width": half_width, "powered": powered}
    if g_hi < 1 and g <= 1 - threshold:
        return {**base, "verdict": "regressed", "reason": f"CI below 1 and change beyond -{threshold:.1%}"}
    if g_lo > 1 and g >= 1 + threshold:
        return {**base, "verdict": "improved", "reason": f"CI above 1 and change beyond +{threshold:.1%}"}
    if g_lo >= 1 - threshold and g_hi <= 1 + threshold and powered:
        return {**base, "verdict": "within_noise", "reason": f"CI inside +/-{threshold:.1%} and narrow enough"}
    why = "CI too wide to decide" if not powered else "CI crosses the practical threshold"
    return {**base, "verdict": "inconclusive", "reason": why}


def combine_claims(verdicts: Iterable[str]) -> str:
    """Combine claim verdicts: any FAIL -> FAIL; else any UNPROVEN -> UNPROVEN; else PASS.

    "NOT RUN" entries are ignored unless nothing else ran. Missing data is never a pass.
    """
    vs = [v for v in verdicts if v != "NOT RUN"]
    if not vs:
        return "NOT RUN"
    if "FAIL" in vs:
        return "FAIL"
    if "UNPROVEN" in vs:
        return "UNPROVEN"
    return "PASS"
