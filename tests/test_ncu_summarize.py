"""ncu CSV parsing on small hand-written exports (same layout as `ncu --import X --csv`)."""

import pytest

from inferencekit import registry
from inferencekit.analysis import ncu_csv

RAW_CSV = """==PROF== Disconnected from process 123
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","gpu__time_duration.sum","dram__bytes_read.sum","dram__bytes_write.sum","gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed","sm__throughput.avg.pct_of_peak_sustained_elapsed","sm__warps_active.avg.pct_of_peak_sustained_active","launch__registers_per_thread","launch__waves_per_multiprocessor","launch__occupancy_limit_registers","launch__occupancy_limit_shared_mem","launch__occupancy_limit_warps","launch__occupancy_limit_blocks","smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio","smsp__average_warps_issue_stalled_selected_per_issue_active.ratio","smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"
"","","","","","","","","","","","usecond","Kbyte","Kbyte","%","%","%","register/thread","","block","block","block","block","inst","inst","inst"
"0","123","python","host","gemv_kernel","1","7","(256, 1, 1)","(132, 1, 1)","0","9.0","10.00","16,000.00","4.00","82.50","20.10","45.00","40","1.50","3","8","8","32","12.00","1.00","3.00"
"1","123","python","host","tiny_kernel","1","7","(64, 1, 1)","(1, 1, 1)","0","9.0","2.50","0.01","0.00","1.20","3.40","6.00","16","0.01","16","32","24","32","0.50","1.00","4.50"
"2","123","python","host","mma_kernel","1","7","(128, 1, 1)","(264, 1, 1)","0","9.0","100.00","50.00","10.00","12.00","88.00","70.00","128","2.00","2","4","8","32","1.00","1.00","2.00"
"""

_PREFIX = '"0","123","python","host","gemv_kernel","1","7","(256, 1, 1)","(132, 1, 1)","0","9.0",'
DETAILS_CSV = "\n".join([
    '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size",'
    '"Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"',
    _PREFIX + '"GPU Speed Of Light Throughput","Duration","usecond","10.00"',
    _PREFIX + '"GPU Speed Of Light Throughput","Memory Throughput","%","83.00"',
    _PREFIX + '"Memory Workload Analysis","Memory Throughput","Gbyte/second","1,600.00"',
    _PREFIX + '"Memory Workload Analysis","L2 Hit Rate","%","12.50"',
    _PREFIX + '"Occupancy","Theoretical Occupancy","%","50.00"',
]) + "\n"


def test_units_and_commas():
    assert ncu_csv.to_number("1,234.5", "usecond") == pytest.approx(1_234_500.0)
    assert ncu_csv.to_number("n/a") is None


@pytest.mark.parametrize("unit,kind,factor", [
    ("usecond", "time", 1e3), ("us", "time", 1e3), ("msecond", "time", 1e6), ("ns", "time", 1.0),
    ("Kbyte", "bytes", 1e3), ("MB", "bytes", 1e6), ("byte", "bytes", 1.0),
    ("Gbyte/second", "byte_rate", 1e9), ("Gbyte/s", "byte_rate", 1e9), ("Tbyte/s", "byte_rate", 1e12),
    ("Mbyte/s", "byte_rate", 1e6), ("%", "percent", 1.0), ("register/thread", "count", 1.0), ("", "count", 1.0),
])
def test_unit_spellings_old_and_new(unit, kind, factor):
    # Newer Nsight Compute releases write "us" and "Gbyte/s" where older ones wrote "usecond" and "Gbyte/second".
    assert ncu_csv.unit_kind(unit) == (kind, factor)


def test_unknown_or_mismatched_units_are_never_read_as_base_units():
    assert ncu_csv.unit_kind("cycle/nsecond") is None and ncu_csv.unit_kind("Gbyte/ms") is None
    assert ncu_csv.to_number("763.13", "Gbyte/ms", "byte_rate") is None  # unknown unit: no guess
    assert ncu_csv.to_number("43.3", "%", "byte_rate") is None  # a percentage is not a byte rate
    assert ncu_csv.to_number("763.13", "Gbyte/s", "byte_rate") == pytest.approx(763.13e9)
    assert ncu_csv.to_number("12.5", "", "percent") == pytest.approx(12.5)  # a bare number is accepted as %


def _details_only(unit, value, extra=()):
    rows = ['"ID","Kernel Name","Section Name","Metric Name","Metric Unit","Metric Value"',
            '"0","gemv_kernel","GPU Speed Of Light Throughput","Duration","us","10.00"',
            '"0","gemv_kernel","GPU Speed Of Light Throughput","Memory Throughput","%","45.40"']
    if unit is not None:
        rows.append(f'"0","gemv_kernel","Memory Workload Analysis","Memory Throughput","{unit}","{value}"')
    return "\n".join(rows + list(extra)) + "\n"


@pytest.mark.parametrize("unit,value,gbs", [("Gbyte/second", "763.13", 763.1), ("Gbyte/s", "763.13", 763.1),
                                            ("Tbyte/s", "1.33", 1330.0)])
def test_details_page_byte_rate_spellings(unit, value, gbs):
    k = ncu_csv.summarize("", _details_only(unit, value), peak_gbs=1680.0)[0]
    assert k["duration_us"] == pytest.approx(10.0)
    assert k["dram_gbs"] == pytest.approx(gbs)
    assert k["dram_pct_of_measured_peak"] == pytest.approx(round(100 * gbs / 1680.0, 1))
    assert k["unparsed_units"] == []


def test_basic_set_without_memory_section_gives_no_bandwidth():
    # --set basic: no Memory Workload Analysis section. The only "Memory Throughput" left is the
    # Speed Of Light one in %, which must not be read as 45.4 bytes/s (0.0 GB/s).
    k = ncu_csv.summarize("", _details_only(None, None), peak_gbs=1680.0)[0]
    assert k["dram_gbs"] is None and k["dram_pct_of_measured_peak"] is None
    assert k["mem_pct"] == pytest.approx(45.4)


def test_unknown_unit_is_flagged_not_guessed(tmp_path, store):
    k = ncu_csv.summarize("", _details_only("Gbyte/ms", "0.76"), peak_gbs=1680.0)[0]
    assert k["dram_gbs"] is None
    assert k["unparsed_units"] == ["dram_bytes_per_s: '0.76' 'Gbyte/ms'"]
    det = tmp_path / "details.csv"
    det.write_text(_details_only("Gbyte/ms", "0.76"))
    raw = tmp_path / "raw.csv"
    raw.write_text("")
    rec = registry.call("summarize_ncu", {"source": str(raw), "details_csv": str(det)})
    assert any("unknown or unexpected unit" in lim and "Gbyte/ms" in lim for lim in rec["limitations"])


def test_raw_page_summary():
    ks = ncu_csv.summarize(RAW_CSV, peak_gbs=2000.0)
    assert [k["kernel"] for k in ks] == ["gemv_kernel", "tiny_kernel", "mma_kernel"]
    gemv = ks[0]
    assert gemv["duration_us"] == pytest.approx(10.0)
    # (16,000 KB + 4 KB) / 10 us = 1600.4 GB/s
    assert gemv["dram_gbs"] == pytest.approx(1600.4)
    assert gemv["dram_pct_of_measured_peak"] == pytest.approx(80.0)
    assert gemv["bound"] == "memory-bound"
    assert gemv["occupancy_limiter"] == "registers"
    assert gemv["top_stalls"][0]["reason"] == "long_scoreboard"
    assert gemv["top_stalls"][0]["share_pct"] == pytest.approx(80.0)  # 12 / (12 + 3); "selected" excluded
    assert all(s["reason"] != "selected" for s in gemv["top_stalls"])
    assert ks[1]["bound"] == "latency-bound"
    assert ks[2]["bound"] == "compute-bound"


def test_details_fill_what_raw_lacks():
    ks = ncu_csv.summarize(RAW_CSV, DETAILS_CSV)
    gemv = ks[0]
    assert gemv["occupancy_theoretical_pct"] == pytest.approx(50.0)
    assert gemv["l2_hit_pct"] == pytest.approx(12.5)
    assert gemv["mem_pct"] == pytest.approx(83.0)  # the SOL-section "Memory Throughput", not the GB/s one


def test_details_only():
    ks = ncu_csv.summarize("", DETAILS_CSV)
    assert len(ks) == 1
    assert ks[0]["duration_us"] == pytest.approx(10.0)
    assert ks[0]["dram_gbs"] == pytest.approx(1600.0)  # from Memory Workload Analysis throughput


def test_bound_label_rules():
    assert ncu_csv.bound_label(75, 10) == "memory-bound"
    assert ncu_csv.bound_label(10, 75) == "compute-bound"
    assert ncu_csv.bound_label(20, 30) == "latency-bound"
    assert ncu_csv.bound_label(50, 30) == "mixed"
    assert ncu_csv.bound_label(None, None) == "unknown"


def test_summarize_operation_from_csv_files(tmp_path, store):
    raw = tmp_path / "raw.csv"
    raw.write_text(RAW_CSV)
    det = tmp_path / "details.csv"
    det.write_text(DETAILS_CSV)
    rec = registry.call("summarize_ncu", {"source": str(raw), "details_csv": str(det), "peak_gbs": "2000"})
    assert rec["kind"] == "ncu_kernels"
    assert [k["bound"] for k in rec["data"]["kernels"]] == ["memory-bound", "latency-bound", "compute-bound"]
    assert any("under 5 us" in lim for lim in rec["limitations"])
