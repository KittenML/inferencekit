"""Both arms' observed memory and requested CPU controls remain readable in a report."""
from inferencekit import evidence as ev, registry
from inferencekit.ops.report import build_report, to_markdown
from conftest import make_benchmark


def test_comparison_environment_card_has_both_cpu_arms_and_memory(store):
    arms = []
    for label, threads, memory, start in (("base", "2", 80, 0), ("candidate", "4", 160, 5)):
        record = make_benchmark(store, [100] * 4, t0=start)
        record["identity"]["gpu"] = ev.identity()["gpu"]
        record["identity"]["cpu"] = {"model": "Fixture CPU"}
        record["identity"]["engine"].update(name=label, flags={"server_env": {
            "controls": {"OMP_NUM_THREADS": threads}, "ld_preload": [], "omitted": []}})
        record["data"]["memory"] = {"peak_rss_mib": memory, "method": "sampled_process_group_rss",
                                     "complete": label == "base", "verdict": "PASS" if label == "base" else "UNPROVEN"}
        arms.append(store.save(record))
    compared = registry.call("compare", {"baseline": arms[0]["evidence_id"], "candidate": arms[1]["evidence_id"],
                                        "resamples": 100})
    report = build_report([compared], "improved", "CPU report")
    rows = {r["engine"]: r for r in report["environment"]}
    assert rows["base"]["hardware"] == rows["candidate"]["hardware"] == "CPU: Fixture CPU"
    assert rows["base"]["server tuning (requested)"] == "OMP_NUM_THREADS=2"
    assert rows["candidate"]["server tuning (requested)"] == "OMP_NUM_THREADS=4"
    assert "80" in rows["base"]["peak memory"] and "160" in rows["candidate"]["peak memory"]
    assert "partial" in rows["candidate"]["peak memory"]
    markdown = to_markdown(report)
    assert "peak memory" in markdown and "server tuning (requested)" in markdown
    assert "sampled_process_group_rss" in markdown
