"""Incomplete rounds cannot disappear through pair intersection; memory stays per arm."""
import copy

import pytest

from inferencekit import evidence as ev, registry
from inferencekit.registry import OpError
from inferencekit.serve.round_evidence import MAX_ROUNDS, round_coverage
from conftest import make_benchmark
from test_compare_media import media_record


def compare(a, b, **kwargs):
    return registry.call("compare", {"baseline": a["evidence_id"], "candidate": b["evidence_id"],
                                      "resamples": 100, **kwargs})


def arms(store):
    return (make_benchmark(store, [100] * 4, t0=0), make_benchmark(store, [200] * 4, t0=5))


@pytest.mark.parametrize("mutate,reason", [
    (lambda rows: rows.pop(), "unmatched metric rounds"),
    (lambda rows: rows.pop(1), "gaps in observed rounds"),
    (lambda rows: rows.append(copy.deepcopy(rows[0])), "duplicate round indices"),
    (lambda rows: rows[1].pop("decode_tok_s"), "missing finite decode_tok_s"),
    (lambda rows: rows[1].update(index="1"), "invalid round indices"),
])
def test_partial_or_duplicate_metric_rounds_cannot_win(store, mutate, reason):
    a, b = arms(store)
    mutate(b["data"]["rounds"])
    b = store.save(b)
    result = compare(a, b)
    assert result["data"]["verdict"] == "inconclusive"
    assert reason in result["data"]["decision"]["reason"]


def test_matching_truncated_tails_need_a_plan_or_explicit_count(store):
    a, b = arms(store)
    observed = compare(a, b)
    assert observed["data"]["verdict"] == "improved"
    assert observed["data"]["arm_round_coverage"]["baseline"]["complete"] is None
    assert any("planned round count unknown" in x for x in observed["limitations"])
    checked = compare(a, b, expected_rounds=5)
    assert checked["data"]["verdict"] == "inconclusive"
    assert checked["data"]["arm_round_coverage"]["baseline"]["missing_indices"] == [4]
    a["data"]["round_plan"] = {"expected_indices": [0, 1, 2, 3, 4]}
    planned = compare(store.save(a), b)
    assert planned["data"]["verdict"] == "inconclusive"


def test_explicit_count_cannot_weaken_a_recorded_plan(store):
    a, b = arms(store)
    a["data"]["round_plan"] = {"expected_indices": [0, 1, 2, 3, 4]}
    result = compare(store.save(a), b, expected_rounds=4)
    assert result["data"]["verdict"] == "inconclusive"
    assert "conflicts with its recorded round plan" in result["data"]["decision"]["reason"]
    with pytest.raises(OpError, match="positive"):
        compare(a, b, expected_rounds=0)


@pytest.mark.parametrize("recorded", [False, True])
def test_rounds_outside_the_expected_plan_are_inconclusive(store, recorded):
    a, b = arms(store)
    if recorded:
        for rec in (a, b):
            rec["data"]["round_plan"] = {"expected_indices": [0, 1, 2]}
        a, b = store.save(a), store.save(b)
    result = compare(a, b, **({} if recorded else {"expected_rounds": 3}))
    assert result["data"]["verdict"] == "inconclusive"
    assert "rounds outside the requested plan" in result["data"]["decision"]["reason"]
    assert result["data"]["arm_round_coverage"]["baseline"]["unexpected_indices"] == [3]
    assert result["data"]["arm_round_coverage"]["baseline"]["complete"] is False


@pytest.mark.parametrize("mutate,flag", [
    (lambda data: data["rounds"][0].update(index=10**30), "invalid_indices"),
    (lambda data: data.update(round_plan={"expected_indices": [10**30]}), "oversized_plan"),
    (lambda data: data.update(round_plan={"expected_indices": [0, 0]}), "invalid_plan"),
    (lambda data: data.update(round_plan=[0, 1]), "invalid_plan"),
])
def test_unbounded_or_invalid_indices_and_plans_are_refused_safely(store, mutate, flag):
    a, b = arms(store)
    mutate(b["data"])
    result = compare(a, store.save(b))
    assert result["data"]["verdict"] == "inconclusive"
    assert result["data"]["arm_round_coverage"]["candidate"][flag] is True


def test_huge_expected_count_is_bounded_before_allocation(store):
    a, b = arms(store)
    with pytest.raises(OpError, match="expected_rounds"):
        compare(a, b, expected_rounds=10**30)
    coverage = round_coverage(a, expected_rounds=10**30)
    assert coverage["oversized_plan"] is True and coverage["expected_indices"] is None
    excessive = {"data": {"rounds": [{"index": 0}] * (MAX_ROUNDS + 1)}}
    coverage = round_coverage(excessive)
    assert coverage["oversized_rows"] is True and coverage["complete"] is False


def test_memory_measurement_and_method_retained_per_arm(store):
    a, b = arms(store)
    a["data"]["memory"] = {"peak_rss_mib": 80, "method": "sampled_process_group_rss"}
    b["metrics"].append(ev.metric("peak_rss_mib", 120, "MiB", "max"))
    result = compare(store.save(a), store.save(b))
    assert result["data"]["verdict"] == "improved"
    assert result["data"]["memory"]["baseline"]["peak_rss_mib"]["method"] == "sampled_process_group_rss"
    assert result["data"]["memory"]["candidate"]["peak_rss_mib"]["value"] == 120
    assert "peak RSS baseline 80.0 MiB (sampled, coverage unknown)" in result["summary"]
    assert "candidate 120.0 MiB (coverage unknown)" in result["summary"]
    assert {m["subject"] for m in result["metrics"] if m["name"] == "peak_rss_mib"} == {"baseline", "candidate"}
    assert not any(m["name"] == "peak_rss_mib_ratio" for m in result["metrics"])


def test_partial_memory_remains_qualified_without_invalidating_timing(store):
    a, b = arms(store)
    a["data"]["memory"] = {"peak_rss_mib": 80, "method": "sampled_process_group_rss", "complete": False,
                             "verdict": "UNPROVEN", "limitations": ["One requested round has no RSS samples."]}
    result = compare(store.save(a), b)
    assert result["data"]["verdict"] == "improved"
    memory = result["data"]["memory"]["baseline"]["peak_rss_mib"]
    assert memory["complete"] is False and memory["verdict"] == "UNPROVEN"
    assert "80.0 MiB (sampled, partial)" in result["summary"] and "candidate unknown" in result["summary"]
    assert any("One requested round has no RSS samples" in line for line in result["limitations"])
    assert any("lower bound on the true peak" in line for line in result["limitations"])


def test_mixed_server_tuning_within_an_arm_is_inconclusive(store):
    a, b = arms(store)
    b["identity"]["engine"]["flags"]["server_env"] = {"varied_between_rounds": [
        {"controls": {"OMP_NUM_THREADS": "1"}}, {"controls": {"OMP_NUM_THREADS": "2"}}]}
    result = compare(a, store.save(b))
    assert result["data"]["verdict"] == "inconclusive"
    assert "server tuning environment varied" in result["data"]["decision"]["reason"]


def test_different_stable_tuning_between_arms_can_improve(store):
    a, b = arms(store)
    a["identity"]["engine"]["flags"]["server_env"] = {"controls": {"OMP_NUM_THREADS": "1"}}
    b["identity"]["engine"]["flags"]["server_env"] = {"controls": {"OMP_NUM_THREADS": "2"}}
    a["identity"]["engine"]["flags"]["private_path"] = "/private/unused"
    b["identity"]["engine"]["flags"]["server_env"]["ld_preload"] = [
        {"name": "libmkl.so", "sha256": "a" * 64, "status": "hashed"}]
    result = compare(store.save(a), store.save(b))
    assert result["data"]["verdict"] == "improved"
    assert result["data"]["arm_environment"]["baseline"]["controls"]["OMP_NUM_THREADS"] == "1"
    assert result["data"]["arm_environment"]["candidate"]["controls"]["OMP_NUM_THREADS"] == "2"
    assert "private_path" not in result["data"]["arm_environment"]["baseline"]
    assert "server env differs: OMP_NUM_THREADS, LD_PRELOAD" in result["summary"]
    assert "/private/unused" not in result["summary"] and "libmkl.so" not in result["summary"]


def test_unknown_requested_controls_and_preload_are_visible_limitations(store):
    a, b = arms(store)
    b["identity"]["engine"]["flags"]["server_env"] = {"controls": {"OMP_NUM_THREADS": None},
        "omitted": ["OMP_NUM_THREADS"], "ld_preload": [{"name": "libfoo.so", "sha256": None, "status": "unavailable"}]}
    result = compare(a, store.save(b))
    assert result["data"]["verdict"] == "improved"
    assert any("requested CPU controls were omitted or invalid" in x for x in result["limitations"])
    assert any("preload library identity" in x for x in result["limitations"])


def cpu_record(store, *, offset=0, scale=1, model="Fixture CPU", topology="a" * 64, allowed=None):
    rec = media_record(store, offset=offset, scale=scale)
    rec["identity"]["gpu"] = ev.identity()["gpu"]
    rec["identity"]["cpu"] = ev.identity(cpu={"model": model, "architecture": "x86_64",
                                             "topology_sha256": topology, "allowed_cpus": allowed or [0, 1]})["cpu"]
    return store.save(rec)


def test_cpu_media_identity_allows_thread_and_affinity_tuning(store):
    a = cpu_record(store)
    b = cpu_record(store, offset=3, scale=.5, allowed=[0])
    result = compare(a, b, metric="image_ms")
    assert result["data"]["verdict"] == "improved"


@pytest.mark.parametrize("kwargs", [{"model": "Another CPU"}, {"topology": "b" * 64}])
def test_cpu_model_or_topology_mismatch_is_inconclusive(store, kwargs):
    result = compare(cpu_record(store), cpu_record(store, offset=3, scale=.5, **kwargs), metric="image_ms")
    assert result["data"]["verdict"] == "inconclusive"
    assert "different CPU hardware" in result["data"]["decision"]["reason"]


def test_cpu_vs_gpu_media_is_inconclusive(store):
    result = compare(cpu_record(store), media_record(store, offset=3, scale=.5), metric="image_ms")
    assert result["data"]["verdict"] == "inconclusive"
    assert "hardware backends differ" in result["data"]["decision"]["reason"]


def scheduler_proof(rec):
    rec["data"].update(scheduler_final=True,
                       round_plan={"expected_indices": list(range(4))},
                       lifecycle={"source": "bench rounds", "fresh_process_per_cell": True, "valid": True, "verdict": "PASS"},
                       runtime_rounds=[{"index": i, "status": "ok", "ready": {"status": "ready"},
                                        "stop": {"group_gone": True}} for i in range(4)])
    return rec


def test_scheduler_lifecycle_proof_does_not_claim_quality_or_reliability(store):
    a = store.save(scheduler_proof(cpu_record(store)))
    b = store.save(scheduler_proof(cpu_record(store, offset=3, scale=.5)))
    result = compare(a, b, metric="image_ms")
    assert result["data"]["required_proofs"] == {"media quality": "UNPROVEN", "fresh-process lifecycle": "PASS",
                                                  "repeated-run reliability": "UNPROVEN"}
    b["data"]["runtime_rounds"][0]["stop"]["group_gone"] = False
    result = compare(a, store.save(b), metric="image_ms")
    assert result["data"]["required_proofs"]["fresh-process lifecycle"] == "UNPROVEN"


def test_lifecycle_flag_alone_does_not_pass(store):
    a, b = cpu_record(store), cpu_record(store, offset=3, scale=.5)
    for rec in (a, b):
        rec["data"]["lifecycle"] = {"source": "bench rounds", "fresh_process_per_cell": True, "valid": True, "verdict": "PASS"}
    result = compare(store.save(a), store.save(b), metric="image_ms")
    assert result["data"]["required_proofs"]["fresh-process lifecycle"] == "UNPROVEN"
