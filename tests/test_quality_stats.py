"""Quality statistics and GSM8K answer extraction (no servers)."""

import math

import pytest

from inferencekit.quality import gsm8k as g8
from inferencekit.quality import metrics as qm
from inferencekit.quality.fake import FakeLM


def _logsoftmax(xs):
    m = max(xs)
    lse = m + math.log(sum(math.exp(x - m) for x in xs))
    return [x - lse for x in xs]


def test_kl_topk_is_zero_for_identical_lists():
    lp = _logsoftmax([2.0, 1.0, 0.5, -1.0, -3.0])
    top3 = {i: lp[i] for i in range(3)}
    assert qm.kl_topk(top3, dict(top3)) == 0.0


def test_kl_topk_equals_exact_kl_when_lists_cover_the_vocabulary():
    p = _logsoftmax([2.0, 1.0, 0.5, -1.0, -3.0])
    q = _logsoftmax([1.8, 1.3, 0.2, -0.5, -3.5])
    exact = qm.kl_full(p, q)
    approx = qm.kl_topk(dict(enumerate(p)), dict(enumerate(q)))
    assert exact > 0.01
    assert approx == pytest.approx(exact, rel=1e-6)


def test_kl_topk_with_truncated_lists_is_finite_and_nonnegative():
    lm_ref, lm_eng = FakeLM(vocab=64), FakeLM(vocab=64, perturb=0.8)
    ctx = [1, 5, 9]
    p, q = lm_ref.logprobs(ctx), lm_eng.logprobs(ctx)
    ref_top = {j: p[j] for j in lm_ref.top(p, 20)}
    eng_top = {j: q[j] for j in lm_eng.top(q, 5)}
    kl = qm.kl_topk(ref_top, eng_top)
    assert math.isfinite(kl) and kl >= 0
    # Merging the unlisted tail into one bucket never increases KL over the exact value
    # when both sides list the same support (data processing inequality).
    same = {j: p[j] for j in lm_ref.top(p, 20)}
    same_q = {j: q[j] for j in same}
    assert qm.kl_topk(same, same_q) <= qm.kl_full(p, q) + 1e-9


def test_tv_topk_matches_exact_tv_and_is_bounded():
    p = _logsoftmax([2.0, 1.0, 0.5, -1.0, -3.0])
    q = _logsoftmax([1.8, 1.3, 0.2, -0.5, -3.5])
    top3 = {i: p[i] for i in range(3)}
    assert qm.tv_topk(top3, dict(top3)) == 0.0
    exact = qm.tv_full(p, q)
    assert exact > 0.01
    assert qm.tv_topk(dict(enumerate(p)), dict(enumerate(q))) == pytest.approx(exact, rel=1e-6)
    # Pinsker: TV <= sqrt(KL / 2)
    assert exact <= math.sqrt(qm.kl_full(p, q) / 2)
    # truncated lists: a probability distance in [0, 1]; on the same support, merging the tail never adds distance
    lm_ref, lm_eng = FakeLM(vocab=64), FakeLM(vocab=64, perturb=0.8)
    a, b = lm_ref.logprobs([1, 5, 9]), lm_eng.logprobs([1, 5, 9])
    tv = qm.tv_topk({j: a[j] for j in lm_ref.top(a, 20)}, {j: b[j] for j in lm_eng.top(b, 5)})
    assert 0.0 < tv <= 1.0
    same = {j: a[j] for j in lm_ref.top(a, 20)}
    assert qm.tv_topk(same, {j: b[j] for j in same}) <= qm.tv_full(a, b) + 1e-9
    # disjoint one-hot lists are as far apart as two distributions can be
    assert qm.tv_topk({"x": 0.0}, {"y": 0.0}) == pytest.approx(1.0)


def test_tv_counts_between_histograms():
    assert qm.tv_counts({"a": 2, "b": 2}, {"a": 4}) == pytest.approx(0.5)
    assert qm.tv_counts({"a": 3, "b": 1}, {"a": 30, "b": 10}) == 0.0  # counts are normalised per side
    assert qm.tv_counts({"a": 1}, {"b": 5}) == 1.0
    assert qm.tv_counts({}, {"a": 1}) is None


def test_position_buckets():
    assert [qm.bucket_of(i) for i in (0, 15, 16, 63, 64, 255, 256, 5000)] == [
        "0-15", "0-15", "16-63", "16-63", "64-255", "64-255", "256+", "256+"]


def test_wilson_ci_known_values():
    lo, hi = qm.wilson_ci(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    lo, hi = qm.wilson_ci(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-4)
    assert qm.wilson_ci(0, 0) == (None, None)


def test_mcnemar_exact():
    assert qm.mcnemar_exact(0, 0) == 1.0
    assert qm.mcnemar_exact(10, 0) == pytest.approx(2 * 0.5 ** 10)
    assert qm.mcnemar_exact(1, 4) == pytest.approx(2 * (1 + 5) / 32)
    assert qm.mcnemar_exact(4, 1) == qm.mcnemar_exact(1, 4)
    assert qm.mcnemar_exact(5, 5) == 1.0
    p = qm.mcnemar_exact(1000, 1150)  # n > 1074: 2**-n would underflow a float
    assert 0 < p < 0.05


def test_cluster_and_paired_cis():
    units = [(9.0, 10), (10.0, 10), (8.0, 10), (10.0, 10)]
    lo, hi = qm.cluster_ci(units, resamples=500)
    assert lo <= qm.pooled(units) <= hi
    d = qm.paired_delta_ci(units, units, resamples=200)
    assert d["delta"] == 0 and d["ci95"] == (0.0, 0.0)


def test_margin_verdict():
    assert qm.margin_verdict(-0.01, 0.005, 0.01)[0] == "PASS"
    assert qm.margin_verdict(0.02, 0.05, 0.01)[0] == "FAIL"
    assert qm.margin_verdict(0.0, 0.05, 0.01)[0] == "UNPROVEN"
    assert qm.margin_verdict(None, None, 0.01)[0] == "UNPROVEN"


@pytest.mark.parametrize("text,expected,rule", [
    ("16 - 3 - 4 = 9 eggs. 9 * 2 = 18\n#### 18", "18", "hash"),
    ("So the total is $1,234.50.\n#### $1,234.50", "$1,234.50", "hash"),
    ("The answer is \\boxed{42}.", "42", "boxed"),
    ("First 3 apples, then 5 more, so 8 apples.", "8", "last"),
    ("I am not sure.", None, "none"),
    ("#### -7", "-7", "hash"),
])
def test_gsm8k_extraction(text, expected, rule):
    assert g8.extract(text) == (expected, rule)


def test_gsm8k_numeric_equality():
    assert g8.is_correct("1,234.50", "1234.5")
    assert g8.is_correct("$18.", "18")
    assert g8.is_correct("18.00", "18")
    assert not g8.is_correct("17", "18")
    assert not g8.is_correct(None, "18")


def test_builtin_gsm8k_split_is_pinned():
    # Downloaded on first use (offline tests: tests/test_quality_gsm8k_data.py). The JSONL hash never
    # changes, so dataset_sha256 in older evidence records still matches.
    assert g8.JSONL_SHA256 == "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
    assert len(g8.HF_REVISION) == 40 and len(g8.PARQUET_SHA256) == 64 and g8.N_TEST == 1319
    cached = g8.cache_dir() / f"gsm8k-test-{g8.HF_REVISION[:12]}.jsonl"
    if cached.is_file():  # this machine has it: check the real split too
        items, sha = g8.load(cached)
        assert len(items) == 1319 and items[0]["gold"] == "18" and sha == g8.JSONL_SHA256
        assert all(g8.normalize(it["gold"]) is not None for it in items)
