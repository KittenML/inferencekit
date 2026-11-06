"""bench rounds: order design, server lifecycle (process groups, readiness), log scraping, end to end.

The engines are tests/fakes_rounds/fake_server.py (a stdlib OpenAI-compatible fake) and the
measurement is tests/fakes_rounds/stub_bench_serve.py, injected with --serve-cmd. No GPU needed.
"""

import json
import os
import re
import socket
import sys
import time
from itertools import permutations
from pathlib import Path

import pytest

from inferencekit import cli
from inferencekit.evidence import compute_id, set_default_root
from inferencekit.ops import bench_rounds as ops_rounds
from inferencekit.registry import OpError
from inferencekit.serve import rounds as rr
from inferencekit.serve import servers

FAKES = Path(__file__).parent / "fakes_rounds"
FAKE_SERVER = str(FAKES / "fake_server.py")
STUB_SERVE = str(FAKES / "stub_bench_serve.py")
LOGS = FAKES / "logs"

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _gone(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except FileNotFoundError:
        return True


def _wait_gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while not _gone(pid) and time.time() < deadline:
        time.sleep(0.05)
    return _gone(pid)


def fake_engine(label: str, port: int, *extra: str, name: str = "custom", env: dict | None = None) -> dict:
    return {"name": name, "label": label, "version": "0.0.1", "install": [],
            "launch": [sys.executable, FAKE_SERVER, "--port", str(port), "--label", label, *extra],
            "env": env or {}, "port": port, "ready_check": "/v1/models", "role": "custom"}


def write_run(tmp_path: Path, engines: list[dict]) -> tuple[Path, Path]:
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    manifest = {"manifest_version": 1, "model": {"id": "org/fake", "revision": "abc", "precision": "bf16",
                                                 "tokenizer": "org/fake"},
                "hardware": {"gpu": "none", "count": 1}, "engines": engines,
                "workload": {"modes": ["single"], "buckets": ["very_short", "short"], "max_tokens": 16, "slo": {}},
                "harness": {}, "agent_run": {}}
    (run / "manifest.json").write_text(json.dumps(manifest))
    prompts = tmp_path / "dev.jsonl"
    rows = [{"id": f"{b}-{i}", "bucket": b, "messages": [{"role": "user", "content": f"hi {i}"}], "max_tokens": 4,
             "source": "test", "kind": "chat"} for b in ("very_short", "short") for i in range(2)]
    prompts.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return run / "manifest.json", prompts


# --------------------------------------------------------------------------- order design

@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_williams_design_is_a_balanced_latin_square(n):
    labels = [f"e{i}" for i in range(n)]
    design = rr.round_orders(labels, 2 * n)
    period = design["period"]
    assert period == (n if n % 2 == 0 or n == 1 else 2 * n)
    block = design["orders"][:period]
    for row in block:  # every engine once per round
        assert sorted(row) == labels
    for pos in range(n):  # every engine equally often in every position
        counts = {lb: sum(1 for row in block if row[pos] == lb) for lb in labels}
        assert len(set(counts.values())) == 1
    if n > 1:  # every ordered pair "b right after a" equally often (first-order carry-over balance)
        counts = rr.carryover_counts(block)
        assert set(counts) == set(permutations(labels, 2))
        assert len(set(counts.values())) == 1


def test_default_rounds_balance_the_design():
    assert [rr.default_rounds(n) for n in (1, 2, 3, 4, 5, 6)] == [3, 4, 6, 4, 10, 6]
    for n in (2, 3, 4, 5):
        assert rr.round_orders([f"e{i}" for i in range(n)], rr.default_rounds(n))["balanced"]


def test_round_orders_flags_an_unbalanced_number_of_rounds():
    assert rr.round_orders(["a", "b", "c"], 6)["balanced"] is True
    d = rr.round_orders(["a", "b", "c"], 4)
    assert d["balanced"] is False and len(d["orders"]) == 4
    assert rr.round_orders(["a", "b"], 3)["orders"] == [["a", "b"], ["b", "a"], ["a", "b"]]


# --------------------------------------------------------------------------- servers

def test_server_env_is_clean_and_records_names_only(monkeypatch):
    monkeypatch.setenv("MY_SECRET_THING", "do-not-pass-1234")
    monkeypatch.setenv("HF_TOKEN", "hf_not_a_real_token_123")
    monkeypatch.setenv("VLLM_SOMETHING", "1")
    env, info = servers.server_env({"VLLM_USE_V1": "1", "HF_TOKEN": None, "ABSENT_VAR": None, "N": 4,
                                    "FLAG": True})
    assert "MY_SECRET_THING" not in env and "VLLM_SOMETHING" not in env  # not listed: not passed
    assert env["VLLM_USE_V1"] == "1" and env["HF_TOKEN"] == "hf_not_a_real_token_123"  # null = inherit
    assert env["N"] == "4" and env["FLAG"] == "true" and "PATH" in env
    assert info["inherited"] == ["HF_TOKEN"] and info["missing"] == ["ABSENT_VAR"]
    assert "hf_not_a_real_token_123" not in json.dumps(info)


@linux_only
def test_stop_kills_the_whole_process_group(tmp_path):
    port = free_port()
    pid_file = tmp_path / "child.pid"
    env, _ = servers.server_env({})
    srv = servers.launch([sys.executable, FAKE_SERVER, "--port", str(port), "--spawn-child", str(pid_file)], env,
                         tmp_path / "s.log")
    ready = servers.wait_ready(srv, f"http://127.0.0.1:{port}/v1/models", 20, 0.1)
    assert ready["status"] == "ready"
    child = int(pid_file.read_text())
    assert not _gone(child)
    res = servers.stop(srv, grace_s=5)
    assert res["group_gone"] and res["method"] == "sigterm"
    assert _wait_gone(child) and _wait_gone(srv.popen.pid)
    assert not servers.port_in_use("127.0.0.1", port)


@linux_only
def test_stop_escalates_to_sigkill_when_sigterm_is_ignored(tmp_path):
    port = free_port()
    pid_file = tmp_path / "child.pid"
    env, _ = servers.server_env({})
    srv = servers.launch([sys.executable, FAKE_SERVER, "--port", str(port), "--ignore-sigterm", "--spawn-child",
                          str(pid_file)], env, tmp_path / "s.log")
    assert servers.wait_ready(srv, f"http://127.0.0.1:{port}/health", 20, 0.1)["status"] == "ready"
    child = int(pid_file.read_text())
    # The grandchild dies on SIGTERM; the leader ignores it, so the group survives the grace period.
    res = servers.stop(srv, grace_s=0.5)
    assert res["method"] == "sigkill" and res["group_gone"]
    assert _wait_gone(child) and _wait_gone(srv.popen.pid)


def test_readiness_times_out_and_reports_early_exit(tmp_path):
    env, _ = servers.server_env({})
    port = free_port()
    srv = servers.launch([sys.executable, FAKE_SERVER, "--port", str(port), "--never-ready"], env, tmp_path / "a.log")
    try:
        r = servers.wait_ready(srv, f"http://127.0.0.1:{port}/v1/models", 1.0, 0.1)
        assert r["status"] == "timeout" and r["last"] == 503
    finally:
        assert servers.stop(srv, 5)["group_gone"]
    srv = servers.launch([sys.executable, FAKE_SERVER, "--port", str(free_port()), "--exit-early"], env,
                         tmp_path / "b.log")
    r = servers.wait_ready(srv, "http://127.0.0.1:1/v1/models", 20, 0.1)
    assert r["status"] == "exited" and r["exit_code"] == 3
    assert servers.stop(srv, 1)["method"] == "already_exited"
    assert "fake failure" in (tmp_path / "b.log").read_text()


def test_gpu_lock_is_exclusive(tmp_path):
    lock = tmp_path / "gpu.lock"
    with servers.gpu_lock(lock, "first") as held:
        assert held["owner"] == "first"
        with pytest.raises(OpError, match="held by"):
            with servers.gpu_lock(lock, "second"):
                pass
    with servers.gpu_lock(lock, "third"):  # released after the block
        pass


def test_cpu_lock_uses_its_own_default_and_is_exclusive(tmp_path, monkeypatch):
    cpu, gpu = tmp_path / "cpu.lock", tmp_path / "gpu.lock"
    monkeypatch.setenv("INFERENCEKIT_CPU_LOCK", str(cpu))
    monkeypatch.setenv("INFERENCEKIT_GPU_LOCK", str(gpu))
    with servers.gpu_lock(None, "cpu-first", resource="CPU") as held:
        assert held["path"] == str(cpu)
        with pytest.raises(OpError, match="CPU lock"):
            with servers.gpu_lock(None, "cpu-second", resource="CPU"):
                pass
        with servers.gpu_lock(None, "independent-gpu") as held_gpu:
            assert held_gpu["path"] == str(gpu)


def test_gpu_memory_wait_detects_a_leak():
    readings = iter([[9000], [5000], [1100]])
    ok = servers.wait_gpu_memory([1000], 256, 5, reader=lambda: next(readings))
    assert ok["checked"] and ok["returned"] and ok["after_mib"] == [1100]
    leak = servers.wait_gpu_memory([1000], 256, 0.2, reader=lambda: [4000])
    assert leak["checked"] and not leak["returned"] and leak["excess_mib"] == [3000]
    assert servers.wait_gpu_memory(None, 256, 1)["checked"] is False


def test_warmup_messages_are_salted_copies():
    msgs = [{"role": "user", "content": "What is 2+2?"}]
    salted = servers.salted_copy(msgs, "ik-rounds-warmup-r1-0")
    assert salted[0]["content"] == "[ik-rounds-warmup-r1-0] What is 2+2?" and msgs[0]["content"] == "What is 2+2?"
    sysmsg = servers.salted_copy([{"role": "system", "content": "Be brief."}, *msgs], "t")
    assert sysmsg[0]["content"].startswith("[t] ") and sysmsg[1]["content"] == "What is 2+2?"  # earliest text
    parts = servers.salted_copy([{"role": "user", "content": [{"type": "text", "text": "hi"}]}], "t")
    assert parts[0]["content"][0] == {"type": "text", "text": "[t] "} and parts[0]["content"][1]["text"] == "hi"


def test_prometheus_deltas_count_counters_only():
    before = ('# TYPE req counter\nreq_total{m="a b"} 3\n# TYPE lat histogram\nlat_count 2\nlat_sum 0.5\n'
              '# TYPE running gauge\nrunning 1\nuntyped 7\n')
    after = ('# TYPE req counter\nreq_total{m="a b"} 10\n# TYPE lat histogram\nlat_count 9\nlat_sum 2.5\n'
             '# TYPE running gauge\nrunning 0\nuntyped 9\n')
    d = servers.metrics_delta(before, after)
    assert d["counter_deltas"] == {'req_total{m="a b"}': 7.0, "lat_count": 7.0, "lat_sum": 2.0}
    assert d["gauges_after"] == {"running": 0.0}
    assert servers.metrics_delta(None, after) == {"available": False}


# --------------------------------------------------------------------------- resolved config from logs

def _parse(kind, name):
    return servers.parse_resolved_config(kind, (LOGS / f"{name}.log").read_text())


def test_vllm_log_resolved_config():
    r = _parse("vllm", "vllm")
    s = r["settings"]
    assert r["status"] == "RESOLVED" and r["version"] == "0.11.0"
    assert s["engine_config"]["enforce_eager"] is False and s["engine_config"]["max_seq_len"] == 8192
    assert s["engine_config"]["dtype"] == "torch.bfloat16" and s["engine_config"]["kv_cache_dtype"] == "auto"
    assert s["cudagraph_mode"] == "FULL_AND_PIECEWISE" and s["cudagraph_capture_sizes_count"] == 7
    assert s["attention_backend"] == "FLASH_ATTN" and s["cuda_graphs"]["captured"]
    assert s["kv_cache_tokens"] == 79232 and s["non_default_args"]["max_model_len"] == 8192
    assert r["unknown_settings"] == ["env VLLM_ATTENTON_BACKEND"]
    assert r["validity"]["valid"] is True and len(r["warnings"]) == 2


def test_vllm_eager_and_fallback_attention_are_hard_gate_violations():
    text = (LOGS / "vllm.log").read_text().replace("enforce_eager=False", "enforce_eager=True")
    text = text.replace("Using FLASH_ATTN attention backend", "Using TORCH_SDPA attention backend")
    r = servers.parse_resolved_config("vllm", text)
    assert r["validity"]["valid"] is False
    assert any("enforce_eager" in v for v in r["validity"]["violations"])
    assert any("TORCH_SDPA" in v for v in r["validity"]["violations"])


def test_sglang_logs_new_and_old_server_args_formats():
    r = _parse("sglang", "sglang")
    assert r["status"] == "RESOLVED" and r["settings"]["server_args"]["attention_backend"] == "flashinfer"
    assert r["settings"]["scheduler"]["max_total_num_tokens"] == 96512 and r["settings"]["cuda_graphs"]["captured"]
    assert r["validity"]["valid"] is True and r["warnings"]
    old = _parse("sglang", "sglang_old")
    assert old["status"] == "RESOLVED" and old["settings"]["server_args"]["kv_cache_dtype"] == "fp8_e5m2"
    assert "CUDA graphs off (disable_cuda_graph=True)" in old["validity"]["violations"]


def test_llamacpp_logs_gpu_and_cpu_builds():
    r = _parse("llamacpp", "llamacpp")
    s = r["settings"]
    assert r["status"] == "RESOLVED" and r["version"] == "b6743 (a1b2c3d4)"
    assert s["context"]["n_ctx"] == 16384 and s["context"]["flash_attn"] == "auto"
    assert s["offloaded_layers"] == {"gpu": 37, "total": 37} and s["kv_cache_type"] == {"k": "f16", "v": "f16"}
    assert s["n_slots"] == 4 and r["validity"]["valid"] is True
    cpu = _parse("llamacpp", "llamacpp_cpu")
    assert cpu["settings"]["context"] == {"n_ctx": 4096, "flash_attn": 0}
    assert cpu["validity"]["valid"] is False and len(cpu["validity"]["violations"]) == 2
    assert cpu["unknown_settings"] == ["args --flash-attention"] and cpu["errors"]


def test_trtllm_log_and_generic_fallbacks():
    r = _parse("trtllm", "trtllm")
    args = r["settings"]["llm_args"]
    assert r["status"] == "RESOLVED" and r["version"] == "1.0.0"
    assert args["max_batch_size"] == 2048 and args["attn_backend"] == "TRTLLM" and args["cuda_graphs"] is True
    assert args["kv_cache_config"]["dtype"] == "auto" and r["warnings"]
    g = _parse("generic", "generic")
    assert g["status"] == "UNPROVEN" and "raw log" in g["reason"] and g["warnings"]
    # an unknown format for a known engine is UNPROVEN too, never guessed
    assert servers.parse_resolved_config("vllm", "hello\nworld\n")["status"] == "UNPROVEN"
    custom = servers.parse_resolved_config(
        "generic", 'boot\n{"inferencekit_resolved_config": {"cuda_graphs": false, "kv_dtype": "fp8"}}\n')
    assert custom["status"] == "SELF_REPORTED" and custom["validity"]["valid"] is False
    assert custom["validity"]["source"] == "self_reported" and custom["self_reported"] == ["cuda_graphs", "kv_dtype"]
    # a self-report can show a violation but never prove there is none
    clean = servers.parse_resolved_config(
        "generic", '{"inferencekit_resolved_config": {"cuda_graphs": true, "attention_backend": "flash"}}\n')
    assert clean["status"] == "SELF_REPORTED" and clean["validity"]["valid"] is None


def test_self_report_never_upgrades_a_known_engine():
    """The process under test can print anything: for vLLM, only vLLM's own lines count."""
    lines = [ln for ln in (LOGS / "vllm.log").read_text().splitlines()
             if "Initializing a V1 LLM engine" not in ln and "attention backend" not in ln.lower()
             and "Using " not in ln]
    lines.append('{"inferencekit_resolved_config": {"enforce_eager": false, "cudagraph_mode": "FULL", '
                 '"attention_backend": "FLASH_ATTN"}}')
    r = servers.parse_resolved_config("vllm", "\n".join(lines))
    assert r["status"] == "PARTIAL"  # not RESOLVED, not SELF_REPORTED
    assert r["self_reported"] == ["attention_backend", "cudagraph_mode", "enforce_eager"]
    assert "attention_backend" not in r["settings"] and "attention_backend" not in r["validity"]["checked"]
    sdpa = (LOGS / "vllm.log").read_text().replace("Using FLASH_ATTN attention backend",
                                                   "Using TORCH_SDPA attention backend")
    r = servers.parse_resolved_config("vllm", sdpa + '\n{"inferencekit_resolved_config": '
                                                    '{"attention_backend": "FLASH_ATTN"}}\n')
    assert r["status"] == "RESOLVED" and any("TORCH_SDPA" in v for v in r["validity"]["violations"])
    cells = [{"label": "x", "resolved_config": {"status": "RESOLVED", "warnings": 0, "unknown_settings": [],
                                                "self_reported": [], "file": "a"}},
             {"label": "x", "resolved_config": {"status": "SELF_REPORTED", "warnings": 0, "unknown_settings": [],
                                                "self_reported": ["cuda_graphs"], "file": "b"}},
             {"label": "y", "resolved_config": {"status": "PARTIAL", "warnings": 0, "unknown_settings": [],
                                                "self_reported": [], "file": "c"}},
             {"label": "y", "resolved_config": {"status": "SELF_REPORTED", "warnings": 0, "unknown_settings": [],
                                                "self_reported": [], "file": "d"}}]
    summ = rr.resolved_summary(cells, ["x", "y"])
    assert summ["x"]["status"] == "SELF_REPORTED" and summ["x"]["self_reported"] == ["cuda_graphs"]
    assert summ["y"]["status"] == "SELF_REPORTED"  # ranked below PARTIAL, above UNPROVEN


def test_engine_kind_mapping():
    assert servers.engine_kind("vllm", ["vllm", "serve"]) == "vllm"
    assert servers.engine_kind("custom", ["python", "-m", "sglang.launch_server"]) == "sglang"
    assert servers.engine_kind("llama.cpp", ["llama-server"]) == "llamacpp"
    assert servers.engine_kind("trtllm", ["trtllm-serve"]) == "trtllm"
    assert servers.engine_kind("mine", ["./serve"]) == "generic"


# --------------------------------------------------------------------------- bench serve output

def test_bucket_metrics_reads_several_record_shapes():
    rec = {"metrics": [{"name": "ttft_ms", "value": 5.0, "unit": "ms", "stat": "median", "subject": "bucket=short"},
                       {"name": "ttft_ms", "value": 9.0, "unit": "ms", "stat": "p95", "bucket": "short"},
                       {"name": "goodput", "value": 3.0, "unit": "req/s", "stat": "median",
                        "subject": "concurrency=8"}],
           "data": {"per_bucket": {"long": {"tpot_ms": {"median": 20.0, "unit": "ms"}, "decode_tok_s": 50}}}}
    m = rr.bucket_metrics(rec)
    assert m["short"]["ttft_ms"]["value"] == 5.0 and m["short"]["ttft_ms_p95"]["value"] == 9.0
    assert m["concurrency=8"]["goodput"]["value"] == 3.0
    assert m["long"]["tpot_ms"]["value"] == 20.0 and m["long"]["decode_tok_s"]["value"] == 50.0
    assert rr.parse_serve_output("warming up\n" + json.dumps(rec)) == rec
    assert rr.parse_serve_output("") is None


def test_serve_argv_uses_the_pinned_bench_serve_flags():
    ctx = {"base_url": "http://127.0.0.1:1", "prompts": "p.jsonl", "mode": "single", "label": "a", "round": 2,
           "run_dir": "runs/x", "model": "m", "engine_kind": "vllm", "engine_version": "0.11.0", "extend": "ev_1"}
    argv = rr.serve_argv("mybench", ctx, "--cache-salt")
    flags = dict(zip(argv[1::2], argv[2::2]))
    assert argv[0] == "mybench" and flags["--label"] == "a" and flags["--engine-kind"] == "vllm"
    assert flags["--model"] == "m" and flags["--rounds"] == "1" and flags["--extend"] == "ev_1"
    assert flags["--engine-version"] == "0.11.0" and argv[-1] == "--cache-salt"
    load = rr.serve_argv(None, {**ctx, "mode": "load", "extend": ""}, "--concurrency 1 --concurrency 2")
    assert load[1:5] == ["-m", "inferencekit", "bench", "serve"] and "--rounds" not in load and "--extend" not in load
    assert load[-4:] == ["--concurrency", "1", "--concurrency", "2"]
    assert rr.serve_argv("x --url {base_url} --tag {label}-{round}", ctx, None) == \
        ["x", "--url", "http://127.0.0.1:1", "--tag", "a-2"]
    shared = rr.serve_argv("b", {**ctx, "mode": "load", "extend": "", "ref_e2e_ms": 31.25}, "--concurrency 4")
    assert shared[-4:] == ["--ref-e2e-ms", "31.25", "--concurrency", "4"]
    assert "--ref-e2e-ms" not in rr.serve_argv("b", {**ctx, "ref_e2e_ms": 31.25}, None)  # single mode: never
    assert rr.schedule_fixed_by("--rates 1 --rates 2") == "--rates"
    assert rr.schedule_fixed_by("--ref-e2e-ms=40") == "--ref-e2e-ms" and rr.schedule_fixed_by("--concurrency 4") is None
    with pytest.raises(OpError):
        rr.serve_argv("x {nope}", ctx, None)


def test_round_metrics_come_from_the_latest_rounds_requests():
    rows = [{"bucket": "short", "round": 0, "ok": True, "ttft_ms": 100.0},
            {"bucket": "short", "round": 1, "ok": True, "ttft_ms": 10.0},
            {"bucket": "short", "round": 1, "ok": True, "ttft_ms": 30.0},
            {"bucket": "long", "round": 1, "ok": False, "ttft_ms": None}]
    k, m = rr.round_metrics_from_rows(rows)
    assert k == 1 and m["short"]["ttft_ms"]["value"] == 20.0 and m["all"]["ttft_ms"]["value"] == 20.0
    assert m["long"]["failed_requests"]["value"] == 1.0 and "ttft_ms" not in m["long"]
    assert rr.higher_is_better("usd_per_1m_output_tokens_at_n_star") is False
    assert rr.higher_is_better("n_star") is True and rr.higher_is_better("ttft_ms_p95") is False
    # the base metric decides, not a stat suffix that mentions N*
    assert rr.higher_is_better("ttft_ms_p95_at_n_star") is False and rr.higher_is_better("tpot_ms_p95_at_n_star") \
        is False
    assert rr.higher_is_better("goodput_rps_at_n_star") is True and rr.higher_is_better("rate_star_rps") is True
    assert rr.higher_is_better("output_tok_s_per_gpu_at_n_star") is True
    assert rr.higher_is_better("decode_tok_s") is True and rr.higher_is_better("e2e_ms") is False
    assert rr.higher_is_better("mean_inflight") is None


def _capacity_record(ttft_p95, n_star, goodput):
    return {"evidence_id": "x", "kind": "capacity", "data": {"measurement": "m"},
            "metrics": [{"name": "n_star", "value": n_star, "unit": "streams", "stat": "threshold"},
                        {"name": "goodput_rps", "value": goodput, "unit": "req/s", "stat": "at_n_star"},
                        {"name": "ttft_ms", "value": ttft_p95, "unit": "ms", "stat": "p95_at_n_star"},
                        {"name": "tpot_ms", "value": ttft_p95 / 10, "unit": "ms", "stat": "p95_at_n_star"}]}


def test_capacity_records_rank_latencies_low_and_rates_high():
    cells = []
    for k in (1, 2):
        cells.append({"label": "fast", "round": k, "status": "ok",
                      "metrics": rr.bucket_metrics(_capacity_record(6.5 + k, 8, 300.0))})
        cells.append({"label": "slow", "round": k, "status": "ok",
                      "metrics": rr.bucket_metrics(_capacity_record(21.3 + k, 4, 90.0))})
    assert "ttft_ms_p95_at_n_star" in cells[0]["metrics"]["all"]
    s = rr.summarize(cells, ["fast", "slow"], resamples=200, mode="load")
    rows = {(r["engine"], r["metric"]): r for r in s["rows"]}
    for metric in ("ttft_ms_p95_at_n_star", "tpot_ms_p95_at_n_star", "n_star", "goodput_rps_at_n_star"):
        assert rows[("fast", metric)]["best"] == "fast", metric
    assert rows[("slow", "ttft_ms_p95_at_n_star")]["ratio_to_best"] > 1
    assert rows[("slow", "n_star")]["ratio_to_best"] == pytest.approx(0.5)
    assert "(derived)" in s["definitions"]["median"]


def _rows_cell(label, k, ms, fail_long):
    rows = [{"bucket": "short", "round": k, "ok": True, "ttft_ms": ms, "e2e_ms": 3 * ms} for _ in range(4)]
    rows += [{"bucket": "long", "round": k, "ok": i >= fail_long, "ttft_ms": ms if i >= fail_long else None,
              "e2e_ms": 3 * ms if i >= fail_long else None} for i in range(4)]
    _, metrics = rr.round_metrics_from_rows(rows)
    return {"label": label, "round": k, "status": rr.cell_status(0, {"x": 1}, metrics)[0], "metrics": metrics}


def test_an_engine_with_failed_requests_is_never_best():
    """Fast errors (e.g. a context too short for the long prompts) must not win on latency."""
    cells = [c for k in (1, 2, 3, 4) for c in (_rows_cell("flaky", k, 10.0, 3), _rows_cell("honest", k, 100.0, 0))]
    assert all(c["status"] == "ok" for c in cells)
    s = rr.summarize(cells, ["flaky", "honest"], resamples=200)
    rows = {(r["engine"], r["bucket"], r["metric"]): r for r in s["rows"]}
    for bucket in ("long", "all"):
        for metric in ("ttft_ms", "e2e_ms"):
            assert rows[("honest", bucket, metric)]["best"] == "honest"
            flaky = rows[("flaky", bucket, metric)]
            assert flaky["ratio_to_best"] is None and "failed request" in flaky["excluded_from_best"]
    assert rows[("flaky", "short", "ttft_ms")]["best"] == "flaky"  # no failures in that bucket: ranked
    assert rows[("flaky", "long", "failed_requests")]["total"] == 12.0
    assert rows[("flaky", "long", "failed_requests")]["best"] is None  # counts are reported, never ranked
    assert {(f["engine"], f["bucket"]): f["failed_requests"] for f in s["failures"]} == \
        {("flaky", "long"): 12.0, ("flaky", "all"): 12.0}
    plan = {"engines": [{"label": "flaky"}, {"label": "honest"}], "rounds": 4, "mode": "single",
            "design": {"balanced": True, "period": 2, "design": "williams"},
            "prompts": {"name": "p", "sha256": "0" * 64, "n": 8}}
    lim, missing = ops_rounds._limitations({"plan": plan, "cells": cells, "resolved_config": {}, "summary": s})
    assert any(x.startswith("flaky/long: 12 of 16 requests failed over 4 round(s)") for x in lim)
    assert not any(x.startswith("flaky/all") for x in lim) and not missing
    md = rr.markdown(s, plan, cells, {})
    assert "| flaky [FAILURES] | long | ttft_ms |" in md and "## Failed requests" in md


def test_a_cell_whose_requests_all_failed_is_serve_failed():
    rows = [{"bucket": "short", "round": 0, "ok": False, "ttft_ms": None} for _ in range(3)]
    _, metrics = rr.round_metrics_from_rows(rows)
    assert rr.cell_status(0, {"x": 1}, metrics) == ("serve_failed", "all 3 requests failed")
    assert rr.cell_status(0, None, metrics)[0] == "serve_failed"
    assert rr.cell_status(0, {"x": 1}, {})[0] == "serve_no_metrics"


def test_missing_or_failed_cells_never_make_an_engine_best():
    cells = [_rows_cell(label, k, 1 if label == "incomplete" else 100, 0)
             for label in ("complete", "incomplete") for k in (1, 2)]
    cells[-1] = {"label": "incomplete", "round": 2, "status": "startup_timeout"}
    summary = rr.summarize(cells, ["complete", "incomplete"], resamples=100, expected_rounds=2)
    row = next(r for r in summary["rows"] if r["engine"] == "incomplete" and r["metric"] == "ttft_ms")
    assert row["median"] == 1 and row["best"] == "complete" and row["ratio_to_best"] is None
    assert row["missing_rounds"] == [2] and "incomplete" in row["excluded_from_best"]
    assert summary["incomplete_rounds"] == [{"engine": "incomplete", "missing_rounds": [2]}]
    assert rr.higher_is_better("rtf") is False


def test_cpu_load_is_refused_before_any_launch(tmp_path):
    manifest, prompts = write_run(tmp_path, [fake_engine("a", free_port())])
    with pytest.raises(OpError, match="CPU load mode.*single"):
        rr.run_rounds(str(manifest), str(prompts), 2, "load", resource_mode="cpu")


@pytest.mark.parametrize("cleanup", [
    {"stop": {"group_gone": False}, "status": "startup_timeout"},
    {"stop": {"group_gone": True}, "gpu_memory": {"checked": True, "returned": False}, "status": "ok"},
])
def test_cleanup_failure_skips_every_remaining_text_cell(tmp_path, monkeypatch, cleanup):
    manifest, prompts = write_run(tmp_path, [fake_engine("a", free_port()), fake_engine("b", free_port())])
    launched = []

    def fake_cell(spec, k, position, cfg, prompts):
        launched.append(spec["label"])
        return {"round": k, "position": position, "label": spec["label"], "kind": "generic", **cleanup}

    monkeypatch.setattr(rr, "run_cell", fake_cell)
    result = rr.run_rounds(str(manifest), str(prompts), 4, "single", gpu_lock_path=str(tmp_path / "lock"))
    assert launched == ["a"]
    assert len(result["cells"]) == 8
    assert all(c["status"] == "skipped_after_cleanup_failure" for c in result["cells"][1:])
    assert result["plan"]["aborted_after_cleanup_failure"]["label"] == "a"
    assert result["plan"]["rounds"] == 4
    assert result["cells"][0]["status"] != "ok"
    assert "Incomplete rounds" in result["markdown"]


# --------------------------------------------------------------------------- end to end


@linux_only
def test_cpu_rounds_measure_server_group_without_gpu_probes(tmp_path, store, monkeypatch):
    engines = [fake_engine("a", free_port(), "--ttft-ms", "5", env={"OMP_NUM_THREADS": "2"}),
               fake_engine("b", free_port(), "--ttft-ms", "10", env={"OMP_NUM_THREADS": "4"})]
    manifest, prompts = write_run(tmp_path, engines)
    monkeypatch.setattr(rr, "cpu_identity", lambda: {"model": "Test CPU", "architecture": "test"})
    result = rr.run_rounds(str(manifest), str(prompts), 2, "single", resource_mode="cpu",
                           cpu_lock_path=str(tmp_path / "cpu.lock"), startup_timeout_s=20, stop_timeout_s=3,
                           warmup=0, poll_s=.05, serve_cmd=f"{sys.executable} {STUB_SERVE}",
                           gpu_reader=lambda: pytest.fail("CPU runs must not read GPU memory"))
    assert result["plan"]["resource_mode"] == "cpu" and "gpu_lock" not in result["plan"]
    assert result["plan"]["cpu"]["model"] == "Test CPU"
    assert result["plan"]["unfinalized_evidence"] == ["a", "b"]  # deliberately unstored test-client records
    for cell in result["cells"]:
        assert cell["status"] == "ok" and cell["stop"]["group_gone"]
        assert "gpu_memory" not in cell
        assert cell["resources"]["peak_rss_mib"] > 0
        assert cell["resources"]["peak_threads"] >= 1
        assert cell["resources"]["observations"]["settled"]["processes"] >= 1
        assert cell["server_env"]["controls"]["OMP_NUM_THREADS"] == ("2" if cell["label"] == "a" else "4")
    memory = [r for r in result["summary"]["rows"] if r["metric"] == "peak_rss_mib"]
    assert len(memory) == 2 and all(r["higher_is_better"] is False and r["n"] == 2 for r in memory)

def test_end_to_end_two_engines_via_cli(tmp_path, capsys, monkeypatch):
    p_fast, p_slow = free_port(), free_port()
    env_dump = tmp_path / "env.json"
    monkeypatch.setenv("ROUNDS_TEST_SECRET_TOKEN", "should-not-reach-the-server")
    req_log = tmp_path / "fast_requests.jsonl"
    engines = [fake_engine("fast", p_fast, "--ttft-ms", "10", "--log-fixture", str(LOGS / "vllm.log"),
                           "--env-dump", str(env_dump), "--request-log", str(req_log), name="vllm",
                           env={"VLLM_USE_V1": "1"}),
               fake_engine("slow", p_slow, "--ttft-ms", "20", "--log-fixture", str(LOGS / "llamacpp.log"),
                           name="llama.cpp")]
    manifest, prompts = write_run(tmp_path, engines)
    lock = tmp_path / "gpu.lock"
    try:
        code = cli.main(["--runs-dir", str(tmp_path / "store"), "bench", "rounds", "--manifest", str(manifest),
                         "--prompts", str(prompts), "--gpu-lock", str(lock),  # default rounds: 4 for 2 engines
                         "--startup-timeout-s", "30", "--stop-timeout-s", "5", "--warmup", "2",
                         "--serve-cmd", f"{sys.executable} {STUB_SERVE}"])
    finally:
        set_default_root(None)
    assert code == 0
    rec = json.loads(capsys.readouterr().out)
    run = manifest.parent
    assert rec["kind"] == "serve_benchmark" and rec["claim_class"] == "observed"
    assert rec["data"]["metric_claim_classes"]["<metric>"].startswith("observed")
    assert compute_id(rec) == rec["evidence_id"]
    assert (tmp_path / "store" / rec["evidence_id"] / "evidence.json").is_file()
    assert (run / "evidence" / f"{rec['evidence_id']}.json").is_file()
    plan = rec["data"]["plan"]
    # without --ci-target or an extra body the summary has its original shape (the terse view adds the ratios)
    assert re.fullmatch(r"bench rounds \(single\): 2 engines x 4 rounds, 8/8 cells ok, Williams order balanced; "
                        r"best \S+ \[\S+\]: (fast|slow) \S+ \[observed\]", rec["summary"]), rec["summary"]
    assert plan["rounds"] == 4 and not any("not balanced" in x for x in rec["limitations"])
    assert plan["design"]["balanced"] and plan["design"]["orders"][:2] == [["fast", "slow"], ["slow", "fast"]]
    assert all(c["status"] == "ok" for c in rec["data"]["cells"]) and len(rec["data"]["cells"]) == 8
    # single mode chains each engine's bench serve records with --extend; values are per round, not cumulative
    assert plan["serve_evidence_final"] == {"fast": "ev_fast_3_single", "slow": "ev_slow_3_single"}
    assert "round 3" in [c for c in rec["data"]["cells"] if c["label"] == "fast"][-1]["metrics_source"]
    # medians per engine x bucket, with CIs; ratio to best with a paired CI
    rows = {(r["engine"], r["bucket"], r["metric"]): r for r in rec["data"]["rows"]}
    fast, slow = rows[("fast", "short", "ttft_ms")], rows[("slow", "short", "ttft_ms")]
    assert fast["best"] == "fast" and fast["ratio_to_best"] == 1.0 and fast["n"] == 4
    assert fast["median"] == pytest.approx(15.15, rel=0.01)  # 10 ms x 1.5 (short) x round jitter; never 999
    assert slow["ratio_to_best"] == pytest.approx(2.0, rel=0.02) and slow["ratio_ci95"][0] is not None
    assert rows[("slow", "very_short", "decode_tok_s")]["ratio_to_best"] == pytest.approx(0.5, rel=0.02)
    assert ("fast", "all", "e2e_ms") in rows and ("fast", "all", "failed_requests") in rows
    names = {(m["name"], m.get("subject")) for m in rec["metrics"]}
    assert ("ttft_ms_ratio_to_best", "slow/short") in names
    # logs, resolved config, prometheus scrapes, summary table
    for label in ("fast", "slow"):
        for k in range(1, 5):
            assert (run / "logs" / f"{label}.r{k}.log").is_file()
            assert (run / "resolved_config" / f"{label}.r{k}.json").is_file()
            assert (run / "prometheus" / f"{label}.r{k}.after.txt").is_file()
    cell = rec["data"]["cells"][0]
    assert cell["stop"]["group_gone"] and cell["warmup"]["ok"] == 2 and cell["served_model"].startswith("fake-model")
    assert "salted" in cell["warmup"]["kind"] and "salted" in plan["settings"]["warmup_kind"]
    # the warm-up never sent a timed prompt unsalted: it cannot seed the prefix cache of a timed prompt
    seen = [json.loads(x)["first"] for x in req_log.read_text().splitlines()]
    warm = [x for x in seen if x.startswith("[ik-rounds-warmup-r")]
    timed = [x for x in seen if not x.startswith("[ik-")]
    assert len(warm) == 8 and len(set(warm)) == 8 and len(timed) == 16 and not set(timed) & set(warm)
    assert {x.split("] ", 1)[1] for x in warm} <= set(timed)  # same prompts, salted
    res = rec["data"]["resolved_config"]
    assert res["fast"]["status"] == "RESOLVED" and res["slow"]["status"] == "RESOLVED"
    assert res["fast"]["unknown_settings"] == ["env VLLM_ATTENTON_BACKEND"]
    md = (run / "rounds" / "summary.md").read_text()
    assert "| slow | short | ttft_ms |" in md and "[derived]" in md and "RESOLVED" in md
    # the server saw only the listed variables plus the base allowlist
    seen = set(json.loads(env_dump.read_text()))
    assert "VLLM_USE_V1" in seen and "ROUNDS_TEST_SECRET_TOKEN" not in seen
    assert seen <= set(servers.SERVER_BASE_ENV) | {"VLLM_USE_V1", "PWD", "SHLVL", "_", "LC_CTYPE"}
    # the lock was released
    with servers.gpu_lock(lock, "after"):
        pass
    assert not servers.port_in_use("127.0.0.1", p_fast) and not servers.port_in_use("127.0.0.1", p_slow)


def test_end_to_end_failures_are_recorded_and_the_run_continues(tmp_path, store):
    p_ok, p_hang, p_busy = free_port(), free_port(), free_port()
    engines = [fake_engine("ok", p_ok, "--ttft-ms", "5"),
               fake_engine("hang", p_hang, "--never-ready"),
               fake_engine("busy", p_busy)]
    manifest, prompts = write_run(tmp_path, engines)
    squatter = socket.socket()
    squatter.bind(("127.0.0.1", p_busy))
    squatter.listen(1)
    try:
        result = rr.run_rounds(str(manifest), str(prompts), 1, "single", gpu_lock_path=str(tmp_path / "l"),
                               startup_timeout_s=1.0, stop_timeout_s=3, warmup=0,
                               serve_cmd=f"{sys.executable} {STUB_SERVE}", poll_s=0.1, gpu_reader=lambda: None)
    finally:
        squatter.close()
    by = {c["label"]: c for c in result["cells"]}
    assert by["ok"]["status"] == "ok"
    assert by["hang"]["status"] == "startup_timeout" and by["hang"]["stop"]["group_gone"]
    assert by["busy"]["status"] == "port_busy" and "stop" not in by["busy"]
    assert not result["plan"]["design"]["balanced"]  # 3 engines need 6 rounds
    assert {r["engine"] for r in result["summary"]["rows"]} == {"ok"}
    assert "Failed cells" in result["markdown"] and "startup_timeout" in result["markdown"]


def test_end_to_end_engine_whose_requests_all_fail(tmp_path, store):
    engines = [fake_engine("a", free_port(), "--ttft-ms", "5"), fake_engine("b", free_port(), "--ttft-ms", "1")]
    manifest, prompts = write_run(tmp_path, engines)
    result = rr.run_rounds(str(manifest), str(prompts), 2, "single", gpu_lock_path=str(tmp_path / "l"),
                           startup_timeout_s=20, stop_timeout_s=3, warmup=0, poll_s=0.1, gpu_reader=lambda: None,
                           serve_cmd=f"{sys.executable} {STUB_SERVE}", serve_args="--fail-requests b")
    by = {(c["label"], c["round"]): c for c in result["cells"]}
    assert by[("b", 1)]["status"] == "serve_failed" and by[("b", 1)]["detail"] == "all 4 requests failed"
    assert by[("a", 1)]["status"] == "ok"
    assert {r["engine"] for r in result["summary"]["rows"]} == {"a"}


def test_load_mode_reads_the_capacity_record(tmp_path, capsys):
    engines = [fake_engine("a", free_port(), "--ttft-ms", "10"), fake_engine("b", free_port(), "--ttft-ms", "20")]
    manifest, prompts = write_run(tmp_path, engines)
    try:
        code = cli.main(["--runs-dir", str(tmp_path / "store"), "bench", "rounds", "--manifest", str(manifest),
                         "--prompts", str(prompts), "--rounds", "2", "--mode", "load", "--gpu-lock",
                         str(tmp_path / "l"), "--startup-timeout-s", "20", "--stop-timeout-s", "3", "--warmup", "0",
                         "--serve-cmd", f"{sys.executable} {STUB_SERVE}", "--serve-args", "--engine-version 9"])
    finally:
        set_default_root(None)
    assert code == 0
    rec = json.loads(capsys.readouterr().out)
    # load-mode values come from capacity records: derived, in the record and in the table
    assert rec["claim_class"] == "derived" and rec["data"]["metric_claim_classes"]["<metric>"].startswith("derived")
    md = (manifest.parent / "rounds" / "summary.md").read_text()
    assert "median [derived]" in md and "median [observed]" not in md
    rows = {(r["engine"], r["metric"]): r for r in rec["data"]["rows"]}
    assert rows[("a", "n_star")]["best"] == "a" and rows[("b", "n_star")]["ratio_to_best"] == pytest.approx(0.5)
    cost = rows[("b", "usd_per_1m_output_tokens_at_n_star")]
    assert cost["best"] == "a" and cost["ratio_to_best"] == pytest.approx(2.0)
    ttft = rows[("b", "ttft_ms_p95_at_n_star")]  # lower p95 TTFT wins, although the name mentions N*
    assert ttft["best"] == "a" and ttft["ratio_to_best"] == pytest.approx(2.0)
    assert rows[("b", "goodput_rps_at_n_star")]["best"] == "a"
    plan = rec["data"]["plan"]
    assert plan["serve_evidence_final"] == {}
    # one arrival schedule for every engine: the first cell's warm-up e2e, passed to every later cell
    sched = plan["load_schedule"]
    assert sched["ref_e2e_ms"] == 30.0 and sched["shared"] and sched["pilot"] == "a r1"
    cells = rec["data"]["cells"]
    assert cells[0]["load_schedule"]["ref_e2e_ms"] is None
    assert all(c["load_schedule"]["ref_e2e_ms"] == 30.0 for c in cells[1:])
    assert all(c["load_schedule"]["ref_e2e_ms_used"] == 30.0 for c in cells)
    assert "Arrival schedule: the same for every engine" in md
    assert not any("schedules differ" in x for x in rec["limitations"])


def test_load_mode_schedule_given_or_fixed_by_serve_args(tmp_path, store):
    engines = [fake_engine("a", free_port(), "--ttft-ms", "10"), fake_engine("b", free_port(), "--ttft-ms", "20")]
    manifest, prompts = write_run(tmp_path, engines)
    common = dict(gpu_lock_path=str(tmp_path / "l"), startup_timeout_s=20, stop_timeout_s=3, warmup=0, poll_s=0.1,
                  gpu_reader=lambda: None, serve_cmd=f"{sys.executable} {STUB_SERVE}")
    given = rr.run_rounds(str(manifest), str(prompts), 2, "load", ref_e2e_ms=12.5, **common)
    assert given["plan"]["load_schedule"]["shared"]
    assert all(c["load_schedule"]["ref_e2e_ms_used"] == 12.5 for c in given["cells"])
    fixed = rr.run_rounds(str(manifest), str(prompts), 2, "load", serve_args="--rates 1 --rates 2", **common)
    sched = fixed["plan"]["load_schedule"]
    assert sched["fixed_by"] == "--rates" and sched["shared"] and sched["ref_e2e_ms"] is None
    assert all(c["load_schedule"]["ref_e2e_ms"] is None for c in fixed["cells"])
    # each engine used its own warm-up (no --ref-e2e-ms passed): the stub reports 30 and 60
    assert {c["load_schedule"]["ref_e2e_ms_used"] for c in fixed["cells"]} == {30.0, 60.0}


def test_engines_filter_and_manifest_errors(tmp_path):
    manifest, prompts = write_run(tmp_path, [fake_engine("a", 1), fake_engine("b", 2)])
    specs = rr.engine_specs(rr.load_manifest(manifest), "b")
    assert [s["label"] for s in specs] == ["b"]
    with pytest.raises(OpError, match="unknown label"):
        rr.engine_specs(rr.load_manifest(manifest), "zzz")
    bad = {"engines": [{"label": "x", "launch": "vllm serve", "port": "8000"}]}
    with pytest.raises(OpError, match="launch"):
        rr.engine_specs(bad, None)
    with pytest.raises(OpError, match="appears twice"):
        rr.engine_specs({"engines": [fake_engine("a", 1), fake_engine("a", 2)]}, None)


# --------------------------------------------------------------------------- adaptive stop (--ci-target)

def _cells(values: dict[str, list[float | None]]) -> list[dict]:
    """Single-mode cells holding only decode_tok_s [all]: values[label][k - 1] for round k (None: a failed cell)."""
    return [{"label": lb, "round": k, "status": "ok" if v is not None else "serve_failed",
             "metrics": {"all": {"decode_tok_s": {"value": v, "unit": "tok/s"},
                                 "ttft_ms": {"value": 1.0, "unit": "ms"}}} if v is not None else {}}
            for lb, vs in values.items() for k, v in enumerate(vs, 1)]


def test_adaptive_check_reads_each_engines_ratio_ci():
    tight = rr.adaptive_check(_cells({"a": [100, 101, 99, 100], "b": [50, 50.5, 49.5, 50]}), ["a", "b"], 4, 0.01)
    assert tight["met"] and tight["round"] == 4
    assert tight["widths"]["a"] == {"width": None, "basis": "best engine (the reference of every ratio)",
                                    "counted": False}
    b = tight["widths"]["b"]
    assert b["basis"] == "ratio_ci95" and b["width"] == pytest.approx(0.0) and b["n"] == 4 and b["counted"]
    noisy = rr.adaptive_check(_cells({"a": [100, 100, 100, 100], "b": [40, 60, 45, 55]}), ["a", "b"], 4, 0.01)
    assert not noisy["met"] and noisy["widths"]["b"]["width"] > 0.05
    assert rr.adaptive_check(_cells({"a": [100, 100, 100, 100], "b": [40, 60, 45, 55]}), ["a", "b"], 4, 0.5)["met"]
    # one engine: the CI of its own median; an engine with no successful round never meets the target
    solo = rr.adaptive_check(_cells({"a": [100, 102, 98, 100]}), ["a"], 4, 0.05)
    assert solo["met"] and solo["widths"]["a"]["basis"] == "ci95 of the median"
    dead = rr.adaptive_check(_cells({"a": [100, 100, 100, 100], "b": [None] * 4}), ["a", "b"], 4, 0.5)
    assert not dead["met"] and dead["widths"]["b"] == {"width": None, "basis": "no successful round", "counted": True}


def test_adaptive_plan_rounds_and_refusals():
    plan, rounds = rr._adaptive_plan(2, None, None, 0.01, "single")
    assert rounds == 12 and plan["max_rounds"] == 12 and plan["min_rounds"] == 4 and plan["period"] == 2
    assert plan["first_check_round"] == 4 and plan["checks"] == [] and "at most 12 rounds" in plan["rule"]
    plan, rounds = rr._adaptive_plan(3, None, None, 0.02, "single")  # 3 engines: period 6
    assert rounds == 12 and plan["period"] == 6 and plan["min_rounds"] == 6 and plan["first_check_round"] == 6
    plan, rounds = rr._adaptive_plan(5, None, None, 0.02, "single")  # 5 engines: period 10
    assert rounds == 20 and plan["first_check_round"] == 10
    plan, rounds = rr._adaptive_plan(2, 10, 3, 0.02, "single")
    assert rounds == 10 and plan["min_rounds"] == 3 and plan["first_check_round"] == 4
    with pytest.raises(OpError, match="single mode"):
        rr._adaptive_plan(2, 8, None, 0.01, "load")
    for bad in (0, 1, -0.1, 1.5):
        with pytest.raises(OpError, match="between 0 and 1"):
            rr._adaptive_plan(2, 8, None, bad, "single")
    with pytest.raises(OpError, match="at least 2"):
        rr._adaptive_plan(2, 8, 1, 0.01, "single")
    with pytest.raises(OpError, match="cannot stop anything"):
        rr._adaptive_plan(2, 4, None, 0.01, "single")  # the first check (round 4) is the last round


def test_min_rounds_needs_a_ci_target(tmp_path):
    manifest, prompts = write_run(tmp_path, [fake_engine("a", free_port()), fake_engine("b", free_port())])
    with pytest.raises(OpError, match="only with --ci-target"):
        rr.run_rounds(str(manifest), str(prompts), 8, "single", min_rounds=4, gpu_lock_path=str(tmp_path / "l"))


def test_adaptive_stop_end_to_end_via_cli(tmp_path, capsys):
    """The stub's two engines keep a constant speed ratio, so the CI is met at the first whole period
    from --min-rounds on (round 4 of at most 8)."""
    engines = [fake_engine("a", free_port(), "--ttft-ms", "10"), fake_engine("b", free_port(), "--ttft-ms", "20")]
    manifest, prompts = write_run(tmp_path, engines)
    try:
        code = cli.main(["--runs-dir", str(tmp_path / "store"), "bench", "rounds", "--manifest", str(manifest),
                         "--prompts", str(prompts), "--rounds", "8", "--ci-target", "0.02", "--min-rounds", "3",
                         "--gpu-lock", str(tmp_path / "l"), "--startup-timeout-s", "20", "--stop-timeout-s", "3",
                         "--warmup", "0", "--serve-cmd", f"{sys.executable} {STUB_SERVE}"])
    finally:
        set_default_root(None)
    assert code == 0
    rec = json.loads(capsys.readouterr().out)
    plan = rec["data"]["plan"]
    assert plan["rounds"] == 4 and len(rec["data"]["cells"]) == 8
    assert plan["design"]["balanced"] and len(plan["design"]["orders"]) == 4
    stop = plan["stopped_early"]
    assert stop["round"] == 4 and stop["rule"] == plan["adaptive"]["rule"] and stop["widths"]["b"]["width"] <= 0.02
    assert plan["adaptive"]["stopped_early"] and plan["adaptive"]["met"]
    assert [c["round"] for c in plan["adaptive"]["checks"]] == [4]  # min 3 -> the first whole period is round 4
    assert "(adaptive: stopped early, CI target 0.02 met)" in rec["summary"]
    assert any(x.startswith("adaptive stop: stopped after round 4 of at most 8") and "optional stopping" in x
               for x in rec["limitations"])
    saved = json.loads((manifest.parent / "rounds" / "plan.json").read_text())
    assert saved["rounds"] == 4 and saved["stopped_early"]["round"] == 4
    assert "Adaptive stop: stopped early after round 4 of at most 8" in \
        (manifest.parent / "rounds" / "summary.md").read_text()


def test_adaptive_target_not_met_runs_every_round(tmp_path, store, monkeypatch):
    engines = [fake_engine("a", free_port(), "--ttft-ms", "10"), fake_engine("b", free_port(), "--ttft-ms", "20")]
    manifest, prompts = write_run(tmp_path, engines)
    wide = {"a": {"width": None, "basis": "best engine (the reference of every ratio)", "counted": False},
            "b": {"width": 0.2, "basis": "ratio_ci95", "n": 2, "counted": True}}
    monkeypatch.setattr(rr, "adaptive_check", lambda cells, labels, k, target: {"round": k, "met": False,
                                                                                "widths": wide})
    result = rr.run_rounds(str(manifest), str(prompts), 4, "single", ci_target=0.01, min_rounds=2,
                           gpu_lock_path=str(tmp_path / "l"), startup_timeout_s=20, stop_timeout_s=3, warmup=0,
                           poll_s=0.1, gpu_reader=lambda: None, serve_cmd=f"{sys.executable} {STUB_SERVE}")
    plan = result["plan"]
    assert plan["rounds"] == 4 and len(result["cells"]) == 8 and "stopped_early" not in plan
    assert [c["round"] for c in plan["adaptive"]["checks"]] == [2, 4]
    assert plan["adaptive"]["met"] is False and plan["adaptive"]["stopped_early"] is False
    lim, _ = ops_rounds._limitations(result)
    assert any(x.startswith("adaptive stop: the CI target 0.01 was NOT met by round 4") and "b 0.2" in x
               for x in lim)


def test_without_ci_target_the_plan_is_unchanged(tmp_path, store):
    engines = [fake_engine("a", free_port(), "--ttft-ms", "10"), fake_engine("b", free_port(), "--ttft-ms", "20")]
    manifest, prompts = write_run(tmp_path, engines)
    result = rr.run_rounds(str(manifest), str(prompts), None, "single", gpu_lock_path=str(tmp_path / "l"),
                           startup_timeout_s=20, stop_timeout_s=3, warmup=0, poll_s=0.1, gpu_reader=lambda: None,
                           serve_cmd=f"{sys.executable} {STUB_SERVE}")
    plan = result["plan"]
    assert plan["rounds"] == 4 and len(result["cells"]) == 8
    assert not {"adaptive", "stopped_early", "extra_body"} & set(plan)
    assert "Adaptive stop" not in result["markdown"] and "--extra-body" not in result["markdown"]
    lim, _ = ops_rounds._limitations(result)
    assert not any("adaptive" in x for x in lim)


# --------------------------------------------------------------------------- sampling workloads (extra body)

SAMPLING = {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "seed": 1234}


def test_extra_body_sources_and_refusals():
    manifest = {"workload": {"extra_body": SAMPLING}}
    assert rr.extra_body_for(None, manifest) == (SAMPLING, "manifest workload.extra_body", [])
    assert rr.extra_body_for('{"temperature": 0.7}', manifest) == ({"temperature": 0.7}, "--extra-body", [])
    assert rr.extra_body_for(None, {}) == ({}, "none", [])
    assert rr.extra_body_for(None, {"workload": {"extra_body": {}}}) == ({}, "none", [])
    inner = "--cache-salt --extra-body '{\"temperature\": 1, \"seed\": 1}'"
    one = {"temperature": 1, "seed": 1}
    assert rr.extra_body_for(None, {}, inner) == (one, "--extra-body in --serve-args", [])
    assert rr.extra_body_for(json.dumps(one), {}, inner) == (one, "--extra-body", [])  # same value: fine
    # warnings come back with the value (bench rounds records them as limitations)
    value, source, warnings = rr.extra_body_for('{"seed": 1}', {})
    assert value == {"seed": 1} and len(warnings) == 1 and "greedy" in warnings[0]
    with pytest.raises(OpError, match="give one value"):
        rr.extra_body_for(None, manifest, inner)
    with pytest.raises(OpError, match="not valid JSON"):
        rr.extra_body_for("{temperature: 1}", {})
    with pytest.raises(OpError, match="JSON object"):
        rr.extra_body_for("[1]", {})
    with pytest.raises(OpError, match="the harness sets max_tokens itself"):
        rr.extra_body_for('{"max_tokens": 8}', {})
    with pytest.raises(OpError, match="manifest workload.extra_body/chat_template_kwargs"):
        rr.extra_body_for(None, {"workload": {"extra_body": {"chat_template_kwargs": {}}}})
    # the manifest schema's rules apply to --extra-body too, and so do the refused fields
    with pytest.raises(OpError, match="--extra-body/top_p: 5 is greater than the maximum of 1"):
        rr.extra_body_for('{"top_p": 5}', {})
    with pytest.raises(OpError, match="--extra-body/ignore_eos: bench serve sets ignore_eos"):
        rr.extra_body_for('{"ignore_eos": false}', {})
    with pytest.raises(OpError, match="ends generations early"):
        rr.extra_body_for('{"stop": ["\\n"]}', {})


def test_serve_argv_sends_one_extra_body():
    ctx = {"base_url": "http://127.0.0.1:1", "prompts": "p.jsonl", "mode": "single", "label": "a", "round": 1,
           "run_dir": "runs/x", "model": "m", "engine_kind": "vllm", "extend": "", "extra_body": SAMPLING}
    argv = rr.serve_argv("b", ctx, "--cache-salt --extra-body={\"seed\":1} --warmup 1")
    assert argv.count("--extra-body") == 1 and not any(w.startswith("--extra-body=") for w in argv)
    assert json.loads(argv[argv.index("--extra-body") + 1]) == SAMPLING
    assert argv[-3:] == ["--cache-salt", "--warmup", "1"]
    plain = rr.serve_argv("b", {**ctx, "extra_body": None}, "--extra-body {}")
    assert plain[-2:] == ["--extra-body", "{}"]  # nothing resolved: --serve-args passes through untouched


def _recording_serve_cmd(tmp_path: Path) -> tuple[str, Path]:
    """A bench serve stand-in that logs the --extra-body it was given (None when absent) and its
    INFERENCEKIT_OUTPUT, then runs the stub without --extra-body."""
    log, wrap = tmp_path / "extra_bodies.jsonl", tmp_path / "wrap_serve.py"
    wrap.write_text(
        "import json, os, runpy, sys\n"
        "argv = sys.argv[1:]\n"
        "i = argv.index('--extra-body') if '--extra-body' in argv else None\n"
        f"with open({str(log)!r}, 'a') as f:\n"
        "    f.write(json.dumps({'label': argv[argv.index('--label') + 1],\n"
        "                        'body': argv[i + 1] if i is not None else None,\n"
        "                        'output': os.environ.get('INFERENCEKIT_OUTPUT')}) + '\\n')\n"
        "if i is not None:\n"
        "    del argv[i:i + 2]\n"
        f"sys.argv = [{STUB_SERVE!r}] + argv\n"
        f"runpy.run_path({STUB_SERVE!r}, run_name='__main__')\n")
    return f"{sys.executable} {wrap}", log


def test_manifest_sampling_reaches_every_bench_serve_call_and_the_record(tmp_path, capsys):
    engines = [fake_engine("a", free_port(), "--ttft-ms", "10"), fake_engine("b", free_port(), "--ttft-ms", "20")]
    manifest, prompts = write_run(tmp_path, engines)
    m = json.loads(manifest.read_text())
    m["workload"]["extra_body"] = SAMPLING
    manifest.write_text(json.dumps(m))
    serve_cmd, log = _recording_serve_cmd(tmp_path)
    try:
        code = cli.main(["--runs-dir", str(tmp_path / "store"), "bench", "rounds", "--manifest", str(manifest),
                         "--prompts", str(prompts), "--rounds", "2", "--gpu-lock", str(tmp_path / "l"),
                         "--startup-timeout-s", "20", "--stop-timeout-s", "3", "--warmup", "0",
                         "--serve-cmd", serve_cmd])
    finally:
        set_default_root(None)
    assert code == 0
    rec = json.loads(capsys.readouterr().out)
    sent = [json.loads(x) for x in log.read_text().splitlines()]
    assert sorted(s["label"] for s in sent) == ["a", "a", "b", "b"]
    assert all(json.loads(s["body"]) == SAMPLING for s in sent)
    assert rec["data"]["plan"]["extra_body"] == {"value": SAMPLING, "source": "manifest workload.extra_body"}
    assert rec["identity"]["engine"]["flags"] == {"engines": ["a", "b"], "extra_body": SAMPLING}
    assert "; extra body {" in rec["summary"] and all(c["status"] == "ok" for c in rec["data"]["cells"])
    assert "Every request carried --extra-body" in (manifest.parent / "rounds" / "summary.md").read_text()


def test_extra_body_flag_overrides_the_manifest(tmp_path, store):
    engines = [fake_engine("a", free_port(), "--ttft-ms", "10")]
    manifest, prompts = write_run(tmp_path, engines)
    m = json.loads(manifest.read_text())
    m["workload"]["extra_body"] = SAMPLING
    manifest.write_text(json.dumps(m))
    serve_cmd, log = _recording_serve_cmd(tmp_path)
    result = rr.run_rounds(str(manifest), str(prompts), 1, "single", extra_body='{"temperature": 0.6, "seed": 7}',
                           gpu_lock_path=str(tmp_path / "l"), startup_timeout_s=20, stop_timeout_s=3, warmup=0,
                           poll_s=0.1, gpu_reader=lambda: None, serve_cmd=serve_cmd)
    assert result["plan"]["extra_body"] == {"value": {"temperature": 0.6, "seed": 7}, "source": "--extra-body"}
    assert [json.loads(json.loads(x)["body"]) for x in log.read_text().splitlines()] == \
        [{"seed": 7, "temperature": 0.6}]


def test_sampling_warnings_become_limitations(tmp_path, store):
    engines = [fake_engine("a", free_port(), "--ttft-ms", "10")]
    manifest, prompts = write_run(tmp_path, engines)
    serve_cmd, _ = _recording_serve_cmd(tmp_path)
    result = rr.run_rounds(str(manifest), str(prompts), 1, "single", extra_body='{"top_k": 64}',
                           gpu_lock_path=str(tmp_path / "l"), startup_timeout_s=20, stop_timeout_s=3, warmup=0,
                           poll_s=0.1, gpu_reader=lambda: None, serve_cmd=serve_cmd)
    body = result["plan"]["extra_body"]
    assert body["value"] == {"top_k": 64} and len(body["warnings"]) == 1
    lim, _ = ops_rounds._limitations(result)
    assert any(x.startswith("extra body sent to every engine: --extra-body: sets top_k but no temperature above 0")
               for x in lim)


# --------------------------------------------------------------------------- output mode of the bench serve child

def test_bench_serve_child_prints_json_when_the_caller_wants_terse(tmp_path, store, monkeypatch):
    """bench rounds reads each bench serve record from the child's stdout: the child must get
    INFERENCEKIT_OUTPUT=json even when the person running bench rounds has the terse default."""
    monkeypatch.setenv("INFERENCEKIT_OUTPUT", "terse")
    engines = [fake_engine("a", free_port(), "--ttft-ms", "10"), fake_engine("b", free_port(), "--ttft-ms", "20")]
    manifest, prompts = write_run(tmp_path, engines)
    serve_cmd, log = _recording_serve_cmd(tmp_path)
    result = rr.run_rounds(str(manifest), str(prompts), 2, "single", gpu_lock_path=str(tmp_path / "l"),
                           startup_timeout_s=20, stop_timeout_s=3, warmup=0, poll_s=0.1, gpu_reader=lambda: None,
                           serve_cmd=serve_cmd)
    seen = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(seen) == 4 and {s["output"] for s in seen} == {"json"}
    assert all(c["status"] == "ok" for c in result["cells"])


def test_real_bench_serve_child_under_the_terse_default(tmp_path, store, monkeypatch):
    """The default child (``python -m inferencekit bench serve``, no --serve-cmd) against a fake engine, with the
    caller's output mode unset (terse): the cell still parses a record and has metrics."""
    monkeypatch.delenv("INFERENCEKIT_OUTPUT", raising=False)
    monkeypatch.delenv(rr.SERVE_CMD_ENV, raising=False)
    src = str(Path(__file__).resolve().parents[1] / "src")
    monkeypatch.setenv("PYTHONPATH", src + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
    manifest, prompts = write_run(tmp_path, [fake_engine("a", free_port(), "--ttft-ms", "5")])
    result = rr.run_rounds(str(manifest), str(prompts), 1, "single", gpu_lock_path=str(tmp_path / "l"),
                           startup_timeout_s=20, stop_timeout_s=3, warmup=0, poll_s=0.1, gpu_reader=lambda: None)
    cell = result["cells"][0]
    assert cell["status"] == "ok", (cell.get("status"), cell.get("detail"))
    assert (cell.get("serve") or {}).get("evidence_id") and cell["metrics"].get("all")
