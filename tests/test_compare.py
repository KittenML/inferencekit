import json

from inferencekit import registry

from conftest import make_benchmark

N = 10
# Small deterministic "noise" (+/-0.4%), different pattern per arm.
BASE = [100.0 * (1 + 0.002 * ((i * 7) % 5 - 2)) for i in range(N)]


def arm(level):
    return [level * (1 + 0.002 * ((i * 3) % 5 - 2)) for i in range(N)]


def run_compare(store, base, cand, **kw):
    a = make_benchmark(store, base, t0=0.0, step=10.0)
    b = make_benchmark(store, cand, t0=kw.pop("cand_t0", 5.0), step=10.0, **kw)  # interleaved: A B A B ...
    return registry.call("compare", {"baseline": a["evidence_id"], "candidate": b["evidence_id"], "resamples": 2000})


def test_improved(store):
    rec = run_compare(store, BASE, arm(110.0))
    assert rec["kind"] == "comparison"
    assert rec["data"]["verdict"] == "improved"
    assert rec["data"]["interleaved"] is True
    ratio = rec["metrics"][0]
    assert 1.08 < ratio["value"] < 1.12 and ratio["ci95"][0] > 1.0


def test_within_noise(store):
    rec = run_compare(store, BASE, arm(100.0))
    assert rec["data"]["verdict"] == "within_noise"
    assert rec["data"]["decision"]["powered"] is True


def test_regressed(store):
    assert run_compare(store, BASE, arm(90.0))["data"]["verdict"] == "regressed"


def test_noisy_short_runs_are_inconclusive(store):
    a = make_benchmark(store, [100.0, 80.0, 120.0], t0=0.0)
    b = make_benchmark(store, [105.0, 125.0, 85.0], t0=5.0)
    rec = registry.call("compare", {"baseline": a["evidence_id"], "candidate": b["evidence_id"], "resamples": 2000})
    assert rec["data"]["verdict"] == "inconclusive"
    # too wide, so more rounds would help: the record says roughly how many
    assert rec["data"]["rounds_for_power"] > 3
    assert any("rounds per arm" in lim for lim in rec["limitations"])


def test_no_rounds_advice_when_more_rounds_cannot_help(store):
    rec = run_compare(store, BASE, arm(110.0), prompt_len=512)  # different workloads
    assert rec["data"]["verdict"] == "inconclusive"
    assert rec["data"]["rounds_for_power"] is None
    assert not any("rounds per arm" in lim for lim in rec["limitations"])


def test_edited_input_is_not_laundered_into_a_passing_comparison(store):
    a = make_benchmark(store, BASE, t0=0.0)
    b = make_benchmark(store, arm(100.0), t0=5.0)
    path = store.run_dir(b["evidence_id"]) / "evidence.json"
    rec = json.loads(path.read_text())
    for r in rec["data"]["rounds"]:  # hand-edit the candidate to look 20% faster
        r["decode_tok_s"] *= 1.2
    path.write_text(json.dumps(rec))
    cmp_rec = registry.call("compare", {"baseline": a["evidence_id"], "candidate": b["evidence_id"],
                                        "resamples": 2000})
    assert cmp_rec["data"]["verdict"] == "inconclusive"
    assert "does not match its content" in cmp_rec["data"]["decision"]["reason"]
    assert "input integrity" in cmp_rec["coverage"]["missing"]
    out = registry.call("report", {"ids": [cmp_rec["evidence_id"]], "format": "json"})
    assert out["verdict"] == "UNPROVEN"


def test_not_interleaved_is_flagged(store):
    rec = run_compare(store, BASE, arm(110.0), cand_t0=10_000.0)
    assert rec["data"]["interleaved"] is False
    assert any("not interleaved" in lim for lim in rec["limitations"])


def test_different_workloads_are_inconclusive(store):
    rec = run_compare(store, BASE, arm(110.0), prompt_len=512)
    assert rec["data"]["verdict"] == "inconclusive"
    assert "workloads differ" in rec["data"]["decision"]["reason"]


def test_latency_metric_is_oriented(store):
    # candidate decodes faster, so its per-token latency is lower: that is an improvement
    a = make_benchmark(store, BASE, t0=0.0)
    b = make_benchmark(store, arm(110.0), t0=5.0)
    rec = registry.call("compare", {"baseline": a["evidence_id"], "candidate": b["evidence_id"],
                                    "metric": "tpot_ms", "resamples": 2000})
    assert rec["data"]["higher_is_better"] is False
    assert rec["data"]["verdict"] == "improved"


def test_report_on_a_comparison(store):
    rec = run_compare(store, BASE, arm(110.0))
    out = registry.call("report", {"ids": [rec["evidence_id"]], "format": "md"})
    text = out["markdown"]
    order = [text.index(h) for h in ("## Verdict", "## Environment card", "## Results", "## Findings",
                                     "## Limitations", "## Reproduce")]
    assert order == sorted(order)
    assert out["verdict"] == "PASS"


def test_report_with_hard_gate_violation_is_not_a_pass(store):
    a = make_benchmark(store, BASE, t0=0.0)
    b = make_benchmark(store, arm(110.0), t0=5.0, valid=False)
    cmp_rec = registry.call("compare", {"baseline": a["evidence_id"], "candidate": b["evidence_id"],
                                        "resamples": 2000})
    out = registry.call("report", {"ids": [cmp_rec["evidence_id"], b["evidence_id"]], "format": "json"})
    assert out["verdict"] == "UNPROVEN"
