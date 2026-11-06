"""Scheduler evidence must match its cells and retain partial-memory qualifiers."""
import pytest

from inferencekit.serve.round_evidence import finalize_arm_records
from conftest import make_benchmark


def fixture_run(store):
    source = make_benchmark(store, [100, 100, 100], t0=5, step=10)
    cells = [{"label": "baseline", "round": i + 1, "status": "ok", "t_start": i * 10,
              "t_end": i * 10 + 9, "ready": {"status": "ready"}, "stop": {"group_gone": True},
              "server_env": {"controls": {"OMP_NUM_THREADS": "2"}, "ld_preload": [], "omitted": []},
              "resources": {"status": "observed", "peak_rss_mib": 50 + i}} for i in range(3)]
    plan = {"rounds": 3, "design": {"design": "Williams", "balanced": True}}
    return source, cells, plan


def test_enriched_evidence_keeps_raw_capture_and_partial_memory(store):
    source, cells, plan = fixture_run(store)
    cells[1]["resources"]["status"] = "partial"
    final = finalize_arm_records({"baseline": source["evidence_id"]}, cells, plan)
    record = store.load(final["baseline"])
    assert record["data"]["memory"]["complete"] is False
    assert record["data"]["memory"]["verdict"] == "UNPROVEN"
    assert record["data"]["memory"]["peak_rss_mib"] == 52
    assert record["data"]["lifecycle"]["verdict"] == "PASS"
    assert store.load(source["evidence_id"])["data"].get("scheduler_final") is None


@pytest.mark.parametrize("bad", ["duplicate", "outside_window"])
def test_lifecycle_requires_unique_cells_bound_to_capture_windows(store, bad):
    source, cells, plan = fixture_run(store)
    if bad == "duplicate":
        cells[1]["round"] = 1
    else:
        cells[1]["t_end"] = 11  # client round 1 was actually captured at 15..16
    final = finalize_arm_records({"baseline": source["evidence_id"]}, cells, plan)
    assert store.load(final["baseline"])["data"]["lifecycle"]["verdict"] == "UNPROVEN"
