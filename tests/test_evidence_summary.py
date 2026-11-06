"""Compact inspection retains decision evidence and routes back to exact stored values."""

import json

import pytest

from inferencekit import cli, evidence as ev, registry
from inferencekit.mcp_server import fit
from inferencekit.ops.store import get_path
from inferencekit.registry import OpError

from conftest import fake_call


def large_record(store):
    checks = [{"gate": f"check_{i}", "verdict": "PASS", "applicable": True, "reason": "tokens agree",
               "samples": [{"token": j, "logprobs": [-0.1, -2.3], "text": "sample " * 40}
                           for j in range(30)]} for i in range(80)]
    checks[-2].update(verdict="FAIL", reason="late token mismatch")
    checks[-1].update(verdict="NOT RUN", reason="reference unavailable")
    return store.save(ev.build(fake_call("run_gates", ("gates", "run")), kind="gates", claim_class="observed",
                               summary="FAIL: one mismatch; quality remains UNPROVEN",
                               identity_=ev.identity(engine={"name": "candidate"},
                                                     workload={"prompt_len": 128, "gen_len": 64}),
                               metrics=[ev.metric("decode_tok_s_ratio", 1.25, "x", "ratio_of_medians", n=6,
                                                  ci95=[1.2, 1.3])],
                               missing=["reference quality proof"], limitations=["screening workload only"],
                               data={"verdict": "FAIL", "checks": checks,
                                     "claim_validity": {"valid": False, "violations": ["quality proof absent"]},
                                     "required_proofs": {"quality": "UNPROVEN"}}))


def inspect(rec, **kwargs):
    return registry.call("inspect_evidence", {"id": rec["evidence_id"], **kwargs})


def test_summary_preserves_decision_inputs_and_full_record(store):
    rec = large_record(store)
    path = store.run_dir(rec["evidence_id"]) / "evidence.json"
    before = path.read_bytes()
    full = inspect(rec)
    summary = inspect(rec, view="summary")
    assert full == {"integrity": ev.verify(rec), "record": rec}
    assert inspect(rec, view="full") == full
    assert summary["integrity"] == {"id_ok": True, "schema_ok": True, "schema_errors": []}
    for key in ("evidence_id", "kind", "claim_class", "summary", "identity", "provenance", "metrics",
                "coverage", "limitations"):
        assert summary[key] == rec[key]
    assert summary["finding_counts"]["verdict:FAIL"] == 2
    assert summary["finding_counts"]["verdict:NOT RUN"] == 1
    assert summary["finding_counts"]["verdict:UNPROVEN"] == 1
    assert summary["finding_counts"]["valid:False"] == 1
    assert summary["data_index"]["checks"] == {"path": "data.checks", "type": "array", "count": 80}
    assert len(json.dumps(summary)) < len(json.dumps(full)) / 10
    assert path.read_bytes() == before and store.load(rec["evidence_id"]) == rec


def test_late_failures_first_and_all_findings_recoverable_without_skips(store):
    rec = large_record(store)
    summary = inspect(rec, view="summary", limit=6)
    first = {x["path"]: x for x in summary["items"]}
    assert first["data.checks.78.verdict"]["reason"] == "late token mismatch"
    assert first["data.checks.79.verdict"]["applicable"] is True
    assert first["data.checks.79.verdict"]["value"] == "NOT RUN"
    assert "view='summary', offset=6, limit=6" in summary["more"]
    found = []
    while True:
        for item in summary["items"]:
            assert get_path(rec, item["path"]) == item["value"]
            found.append(item["path"])
        if summary["next_offset"] is None:
            break
        summary = inspect(rec, view="summary", offset=summary["next_offset"], limit=6)
    assert len(found) == len(set(found)) == summary["total"] == 83
    exact = inspect(rec, path="data.checks", offset=78, limit=2)
    assert exact["items"] == rec["data"]["checks"][78:] and exact["next_offset"] is None
    assert inspect(rec, path="data.checks.78")['value'] == rec["data"]["checks"][78]


def test_summary_does_not_invent_a_verdict_and_retains_bad_integrity(store):
    rec = store.save(ev.build(fake_call(), kind="benchmark", claim_class="observed", summary="no checks",
                              data={"samples": [1, 2, 3]}))
    summary = inspect(rec, view="summary")
    assert summary["finding_counts"] == {} and summary["items"] == []
    assert "No overall verdict is inferred" in summary["note"]
    path = store.run_dir(rec["evidence_id"]) / "evidence.json"
    rec["summary"] = "changed after capture"
    path.write_text(json.dumps(rec))
    assert inspect(rec, view="summary")["integrity"]["id_ok"] is False


def test_cli_summary_and_path_combination(store, capsys, monkeypatch):
    rec = large_record(store)
    monkeypatch.delenv("INFERENCEKIT_OUTPUT", raising=False)
    assert cli.main(["evidence", "show", rec["evidence_id"], "--view", "summary", "--limit", "3"]) == 0
    view = json.loads(capsys.readouterr().out)
    assert view["view"] == "summary" and len(view["items"]) == 3
    with pytest.raises(OpError, match="cannot be combined"):
        inspect(rec, view="summary", path="metrics")


def test_mcp_summary_page_hint_keeps_view_and_does_not_skip(store):
    rec = large_record(store)
    summary = inspect(rec, view="summary")
    view = fit(summary, max_chars=6000)
    assert len(json.dumps(view)) <= 6000
    assert 1 <= len(view["items"]) < len(summary["items"])
    assert view["next_offset"] == len(view["items"])
    assert f"view='summary', offset={view['next_offset']}" in view["more"]
    assert view["finding_counts"] == summary["finding_counts"] and view["integrity"] == summary["integrity"]


@pytest.mark.parametrize("width", [725, 735, 745])
def test_summary_page_hint_overhead_never_skips_findings(store, width):
    rec = store.save(ev.build(fake_call(), kind="gates", claim_class="observed", summary="50 checks",
                              data={"checks": [{"verdict": "PASS", "reason": "x" * width}
                                               for _ in range(50)]}))
    offset, seen = 0, []
    while offset is not None:
        source = inspect(rec, view="summary", offset=offset, limit=50)
        before = json.dumps(source)
        shown = fit(source)
        assert json.dumps(source) == before
        assert len(json.dumps(shown, ensure_ascii=False)) <= 32000
        assert isinstance(shown["items"], list) and shown["items"]
        seen.extend(row["path"] for row in shown["items"])
        if shown["next_offset"] is not None:
            assert shown["next_offset"] == offset + len(shown["items"])
            assert f"view='summary', offset={shown['next_offset']}" in shown["more"]
        offset = shown["next_offset"]
    assert seen == [f"data.checks.{i}.verdict" for i in range(50)]


def test_mcp_summary_qualifier_hint_points_to_original_record(store):
    rec = store.save(ev.build(fake_call(), kind="benchmark", claim_class="observed", summary="invalid",
                              data={"claim_validity": {"valid": False,
                                      "violations": [f"problem {i}: " + "x" * 200 for i in range(100)]}}))
    view = fit(inspect(rec, view="summary"), max_chars=8000)
    cut = view["items"][0]["violations"]
    assert cut["truncated"]
    assert "path='data.claim_validity.violations', offset=10" in cut["more"]
    assert "items.0" not in cut["more"]


def test_mcp_summary_last_resort_retains_failure_counts_and_integrity(store):
    rec = large_record(store)
    summary = inspect(rec, view="summary")
    summary["identity"] = {f"field{i}": "x" * 200 for i in range(100)}
    view = fit(summary, max_chars=3000)
    assert view["truncated"] and view["view"] == "summary"
    assert view["finding_counts"]["verdict:FAIL"] == 2 and view["integrity"]["id_ok"] is True
    assert view["limitation_count"] == view["missing_count"] == 1
    assert "omitted details cannot establish PASS" in view["more"]


def test_ambiguous_engine_keys_use_exact_json_pointers(store):
    rec = store.save(ev.build(fake_call(), kind="gates", claim_class="observed", summary="FAIL",
                              data={"verdicts": {"engine.v1": "FAIL", "engine": {"v1": "PASS"},
                                                 "a./~b": "UNPROVEN", "": "NOT RUN"},
                                    "checks.v1": [{"verdict": "FAIL", "reason": "exact path"}]}))
    summary = inspect(rec, view="summary")
    values = {f["path"]: f["value"] for f in summary["items"]}
    assert values["/data/verdicts/engine.v1"] == "FAIL"
    assert values["data.verdicts.engine.v1"] == "PASS"
    assert values["/data/verdicts/a.~1~0b"] == "UNPROVEN"
    assert values["/data/verdicts/"] == "NOT RUN"
    assert values["/data/checks.v1/0/verdict"] == "FAIL"
    assert summary["data_index"]["checks.v1"]["path"] == "/data/checks.v1"
    for path, value in values.items():
        assert inspect(rec, path=path)["value"] == value
    assert inspect(rec, path="data.verdicts.engine.v1")["value"] == "PASS"  # unchanged dotted semantics


def test_mcp_qualifier_hint_keeps_json_pointer_escaping(store):
    rec = store.save(ev.build(fake_call(), kind="gates", claim_class="observed", summary="FAIL",
                              data={"checks.v1": {"a/b~c": {"verdict": "FAIL",
                                      "violations": [f"problem {i}: " + "x" * 200 for i in range(100)]}}}))
    view = fit(inspect(rec, view="summary"), max_chars=8000)
    cut = view["items"][0]["violations"]
    assert "path='/data/checks.v1/a~1b~0c/violations', offset=10" in cut["more"]
    assert len(inspect(rec, path="/data/checks.v1/a~1b~0c/violations")["items"]) == 50


def test_mcp_nested_list_hints_follow_pointer_pages_and_object_wrappers(store):
    rec = store.save(ev.build(fake_call(), kind="gates", claim_class="observed", summary="FAIL",
                              data={"checks.v1": [{"verdict": "FAIL", "probe.details": ["x" * 200] * 100}]
                                    * 10}))
    for source_path, kwargs, wrapper in [("/data/checks.v1", {"offset": 7, "limit": 1}, "items"),
                                         ("/data/checks.v1/7", {}, "value")]:
        view = fit(inspect(rec, path=source_path, **kwargs), max_chars=4000)
        item = view["items"][0] if wrapper == "items" else view["value"]
        cut = item["probe.details"]
        assert "path='/data/checks.v1/7/probe.details', offset=10" in cut["more"]
        exact = inspect(rec, path="/data/checks.v1/7/probe.details", offset=10, limit=50)
        assert exact["items"] == rec["data"]["checks.v1"][7]["probe.details"][10:60]


def test_mcp_full_record_hint_escapes_ambiguous_keys(store):
    rec = store.save(ev.build(fake_call(), kind="gates", claim_class="observed", summary="FAIL",
                              data={"engine.v1": {"probe/details": ["x" * 200] * 100}}))
    view = fit(inspect(rec), max_chars=5000)
    cut = view["record"]["data"]["engine.v1"]["probe/details"]
    assert "path='/data/engine.v1/probe~1details', offset=10" in cut["more"]
    assert inspect(rec, path="/data/engine.v1/probe~1details", offset=10)["total"] == 100


@pytest.mark.parametrize("status", ["launch_failed", "serve_failed", "startup_timeout", "warmup_failed",
                                    "stop_failed", "exited_early", "port_busy", "serve_no_metrics"])
def test_runtime_failures_precede_unproven_and_keep_detail(store, status):
    rec = store.save(ev.build(fake_call(), kind="serve_benchmark", claim_class="observed", summary="failed cell",
                              data={"checks": [{"verdict": "UNPROVEN"}] * 50,
                                    "cells": [{"status": status, "detail": "engine failed to finish", "round": 1}]}))
    first = inspect(rec, view="summary", limit=1)["items"][0]
    assert first["value"] == status and first["detail"] == "engine failed to finish"
    assert inspect(rec, path=first["path"])["value"] == status


@pytest.mark.parametrize("statuses", [["X" * 50_000], [f"unique_{i}" for i in range(3000)]])
def test_unbounded_status_values_keep_exact_total_without_unbounded_count_keys(store, statuses):
    rec = store.save(ev.build(fake_call(), kind="serve_benchmark", claim_class="observed", summary="status fixture",
                              data={"cells": [{"status": status} for status in statuses]}))
    summary = inspect(rec, view="summary")
    assert summary["finding_counts"] == {"status:<other>": len(statuses)}
    assert summary["total"] == len(statuses)
    assert summary["items"][0]["value"] == statuses[0]
    view = fit(summary, max_chars=3000)
    assert len(json.dumps(view, ensure_ascii=False)) <= 3000
    assert view["finding_counts"] == summary["finding_counts"]
    assert view["integrity"]["id_ok"] is True


def test_mcp_summary_bounds_malformed_integrity_errors_and_counts_all(store):
    rec = large_record(store)
    path = store.run_dir(rec["evidence_id"]) / "evidence.json"
    rec["metrics"] = [{"name": f"bad_{i}", "value": "x" * 50_000,
                       "unit": "x", "stat": "x", "n": 1} for i in range(25)]
    path.write_text(json.dumps(rec))
    error_count = len(ev.validate(rec))
    assert error_count > 20  # verify exposes at most 20 messages, but the total must remain exact
    summary = inspect(rec, view="summary")
    view = fit(summary)
    assert len(json.dumps(view, ensure_ascii=False)) <= 32_000
    assert view["truncated"]
    assert view["integrity"] == {"id_ok": False, "schema_ok": False,
                                 "schema_error_count": error_count, "schema_errors_omitted": True}
    assert view["finding_counts"]["verdict:FAIL"] == 2
    assert "inferencekit evidence show" in view["more"] and "original id or path" in view["more"]
    assert "inferencekit.evidence.validate" in view["more"]
    assert inspect(rec)["integrity"]["schema_errors"]  # ordinary full inspection is unchanged


@pytest.mark.parametrize("field,value", [("artifacts", 7), ("limitations", 7),
                                         ("coverage", 7), ("coverage", {"missing": 7})])
def test_invalid_top_level_counts_remain_unknown(store, field, value):
    rec = large_record(store)
    path = store.run_dir(rec["evidence_id"]) / "evidence.json"
    rec[field] = value
    # Force the bounded fallback without relying on the malformed field to make it large.
    rec["identity"] = {f"field{i}": "x" * 200 for i in range(100)}
    path.write_text(json.dumps(rec))
    summary = inspect(rec, view="summary")
    assert summary["integrity"]["schema_ok"] is False
    assert summary["artifact_count"] == (None if field == "artifacts" else len(rec["artifacts"]))
    view = fit(summary, max_chars=3000)
    assert view["truncated"] and view["integrity"]["schema_ok"] is False
    assert view["limitation_count"] == (None if field == "limitations" else len(rec["limitations"]))
    assert view["missing_count"] == (None if field == "coverage" else len(rec["coverage"]["missing"]))
