import json

import pytest

from inferencekit import registry
from inferencekit.ops.correctness import first_divergence


def test_first_divergence():
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert first_divergence([1, 2], [1, 2, 3]) == 2


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_agreement_from_jsonl(tmp_path, store):
    ref, cand = tmp_path / "ref.jsonl", tmp_path / "cand.jsonl"
    write_jsonl(ref, [[1, 2, 3, 4], [5, 6, 7, 8], {"tokens": [9, 9, 9, 9]}])
    write_jsonl(cand, [[1, 2, 3, 4], [5, 6, 0, 0], {"tokens": [9, 9, 9, 9]}])
    rec = registry.call("check_correctness", {"reference": str(ref), "candidate": str(cand)})
    m = {x["name"]: x["value"] for x in rec["metrics"]}
    assert m["exact_match_pct"] == pytest.approx(100 * 2 / 3)
    assert m["prefix_agreement_pct"] == pytest.approx(100 * (1 + 0.5 + 1) / 3)
    assert rec["data"]["first_divergence"] == [None, 2, None]
    assert rec["data"]["verdict"] == "FAIL"  # default requires identical outputs
    relaxed = registry.call("check_correctness", {"reference": str(ref), "candidate": str(cand),
                                                  "min_prefix_agreement": 0.8})
    assert relaxed["data"]["verdict"] == "PASS"


def test_logprob_tolerance(tmp_path, store):
    ref, cand = tmp_path / "ref.jsonl", tmp_path / "cand.jsonl"
    write_jsonl(ref, [{"tokens": [1, 2], "logprobs": [-0.10, -0.20]}])
    write_jsonl(cand, [{"tokens": [1, 2], "logprobs": [-0.10, -0.35]}])
    rec = registry.call("check_correctness", {"reference": str(ref), "candidate": str(cand), "logprob_tol": 0.1})
    assert rec["data"]["verdict"] == "FAIL"


def test_an_edited_input_record_cannot_pass(store):
    from conftest import make_benchmark

    ref = make_benchmark(store, [100.0, 101.0])
    cand = make_benchmark(store, [100.0, 102.0], t0=5.0)
    path = store.run_dir(cand["evidence_id"]) / "evidence.json"
    edited = json.loads(path.read_text())
    edited["summary"] = "edited by hand"  # tokens unchanged: they still agree with the reference
    path.write_text(json.dumps(edited))
    rec = registry.call("check_correctness", {"reference": ref["evidence_id"], "candidate": cand["evidence_id"]})
    assert rec["data"]["verdict"] == "UNPROVEN"
    assert "input integrity" in rec["coverage"]["missing"]
