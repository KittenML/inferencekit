import math
import random

from inferencekit import stats


def test_bootstrap_ci_of_median_brackets_the_sample_median():
    values = list(range(1, 101))
    lo, hi = stats.bootstrap_ci(values, ps=(50,), resamples=2000)[50]
    assert lo <= 50.5 <= hi
    assert 35 <= lo and hi <= 65


def test_bootstrap_ci_is_deterministic_for_a_seed():
    values = [3.1, 2.9, 3.3, 3.0, 2.8, 3.2]
    a = stats.bootstrap_ci(values, resamples=1000, seed=7)
    b = stats.bootstrap_ci(values, resamples=1000, seed=7)
    assert a == b


def test_bootstrap_ci_edge_cases():
    assert stats.bootstrap_ci([5.0], resamples=100)[50] == (None, None)
    assert stats.bootstrap_ci([4.0, 4.0, 4.0], resamples=100)[50] == (4.0, 4.0)


def test_cluster_bootstrap_is_not_narrower_than_treating_values_as_independent():
    # 4 rounds of 20 identical tokens each: really only 4 independent observations.
    values = [v for v in (10.0, 12.0, 11.0, 13.0) for _ in range(20)]
    units = [i for i in range(4) for _ in range(20)]
    naive = stats.bootstrap_ci(values, ps=(50,), resamples=2000)[50]
    clustered = stats.bootstrap_ci(values, units=units, ps=(50,), resamples=2000)[50]
    assert (clustered[1] - clustered[0]) >= (naive[1] - naive[0])


def test_percentile_ci_index_convention():
    reps = [float(i) for i in range(10_000)]
    lo, hi = stats.percentile_ci(reps)
    assert lo == reps[int(0.025 * 9999)] and hi == reps[math.ceil(0.975 * 9999)]
    assert stats.percentile_ci([1.0, float("nan")]) == (None, None)


def test_nearest_rank_returns_an_observed_value():
    xs = [1.0, 2.0, 3.0, 4.0]
    assert stats.nearest_rank(xs, 50) == 2.0
    assert stats.nearest_rank(xs, 95) == 4.0
    assert stats.quantile(xs, 50) == 2.5


def test_paired_ratio_ci_tracks_a_shared_drift():
    # Both arms drift by the same factor per round; the candidate is 10% faster throughout.
    drift = [1.0, 0.95, 1.05, 0.9, 1.1, 1.0, 0.97]
    base = [100 * d for d in drift]
    cand = [110 * d for d in drift]
    res = stats.paired_ratio_ci(base, cand, resamples=2000)
    assert res["ratio"] == 1.1 or math.isclose(res["ratio"], 1.1)
    lo, hi = res["ci95"]
    assert math.isclose(lo, 1.1) and math.isclose(hi, 1.1)
    unpaired = stats.unpaired_ratio_ci(base, cand, resamples=2000)["ci95"]
    assert (unpaired[1] - unpaired[0]) > (hi - lo)


def test_decide_verdicts():
    assert stats.decide(1.10, 1.08, 1.12, 0.02, True)["verdict"] == "improved"
    assert stats.decide(0.90, 0.88, 0.92, 0.02, True)["verdict"] == "regressed"
    assert stats.decide(1.001, 0.995, 1.006, 0.02, True)["verdict"] == "within_noise"
    # inside the band but too wide to be evidence of "no change"
    assert stats.decide(1.0, 0.981, 1.019, 0.02, True)["verdict"] == "inconclusive"
    assert stats.decide(1.05, 0.95, 1.15, 0.02, True)["verdict"] == "inconclusive"
    assert stats.decide(None, None, None, 0.02, True)["verdict"] == "inconclusive"
    # latency: a smaller ratio is an improvement
    assert stats.decide(0.90, 0.88, 0.92, 0.02, False)["verdict"] == "improved"


def test_rounds_for_power_scales_the_observed_half_width():
    # 10 rounds gave a half-width of 3%; the target is 0.75 x 2% = 1.5%. The half-width shrinks
    # like 1/sqrt(n), so (3 / 1.5)^2 = 4x the rounds hits it on average; the 2x margin makes 80.
    assert stats.rounds_for_power(10, 0.03, 0.02) == 80
    assert stats.rounds_for_power(10, 0.01, 0.02) < stats.rounds_for_power(10, 0.03, 0.02)
    assert stats.rounds_for_power(10, None, 0.02) is None
    assert stats.rounds_for_power(10, 0.0, 0.02) == 2


def _ab_run(rng, n, drift_sd=0.01, noise_sd=0.014):
    """One interleaved A/B with no real change: drift shared by both arms of a round, noise per arm."""
    base, cand = [], []
    for _ in range(n):
        drift = rng.gauss(0, drift_sd)
        base.append(100 * math.exp(drift + rng.gauss(0, noise_sd)))
        cand.append(100 * math.exp(drift + rng.gauss(0, noise_sd)))
    return base, cand


def _decision(base, cand, seed):
    r = stats.paired_ratio_ci(base, cand, resamples=500, seed=seed)
    return stats.decide(r["ratio"], *r["ci95"], 0.02, True), r


def _share_powered(rng, n, runs=40):
    return sum(_decision(*_ab_run(rng, n), seed=k)[0]["powered"] for k in range(runs)) / runs


def test_rounds_for_power_is_enough_for_most_reruns():
    # A pilot of 10 rounds is too noisy to decide. Rerun at the recommended length many times,
    # through the same paired_ratio_ci + decide path that compare uses: most reruns must be powered.
    rng = random.Random(11)
    pilot, res = _decision(*_ab_run(rng, 10), seed=0)
    assert pilot["powered"] is False
    n = stats.rounds_for_power(10, pilot["half_width"], 0.02)
    assert _share_powered(rng, n) >= 0.75
    # The textbook formula (1.96 x SD of per-pair log ratios / sqrt(n)) treats a ratio of medians
    # like a mean and solves for the *average* half-width: far too few rounds.
    textbook = math.ceil((stats.Z95 * res["log_ratio_sd"] / (0.75 * math.log1p(0.02))) ** 2)
    assert textbook < n / 2
    assert _share_powered(rng, textbook) < 0.5


def test_combine_claims_precedence():
    assert stats.combine_claims(["PASS", "UNPROVEN", "FAIL"]) == "FAIL"
    assert stats.combine_claims(["PASS", "UNPROVEN"]) == "UNPROVEN"
    assert stats.combine_claims(["PASS", "NOT RUN"]) == "PASS"
    assert stats.combine_claims([]) == "NOT RUN"
