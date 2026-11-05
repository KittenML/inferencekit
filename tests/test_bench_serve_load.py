"""bench serve, load mode: open-loop arrivals, N*, goodput and cost, against a fake server."""

import json

import pytest

from inferencekit import evidence as ev
from inferencekit import registry
from inferencekit.serve import loadgen, metrics
from inferencekit.serve.prompts_io import Prompt

from fakes.fake_openai_server import FakeConfig, FakeServer
from test_bench_serve import write_prompts


@pytest.fixture(autouse=True)
def endpoint_key(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCEKIT_KEY_FILE", str(tmp_path / "endpoint.key"))


def run_load(srv, prompts, **kw):
    args = {"base_url": srv.base_url, "model": "fake-model", "prompts": str(prompts), "engine_kind": "vllm",
            "mode": "load", "warmup": 1, "duration_s": 1.0, "max_requests_per_level": 40, "resamples": 200}
    args.update(kw)
    return registry.call("bench_serve", args)


def test_poisson_schedule_is_seeded_and_scales_with_rate():
    a = loadgen.poisson_schedule(10, 100, seed=7)
    assert a == loadgen.poisson_schedule(10, 100, seed=7) and a != loadgen.poisson_schedule(10, 100, seed=8)
    assert 900 <= len(a) <= 1100 and all(x < y for x, y in zip(a, a[1:]))
    # same seed, double rate: the same arrivals at half the offsets (identical shape for every engine)
    b = loadgen.poisson_schedule(20, 50, seed=7)
    assert b[:50] == pytest.approx([x / 2 for x in a[:50]])
    assert len(loadgen.poisson_schedule(1000, 10, seed=0, max_requests=25)) == 25


def test_prompt_pool_never_repeats_text_in_a_run():
    pool = loadgen.PromptPool([Prompt(f"p{i}", "short", ({"role": "user", "content": f"q{i}"},), 4)
                               for i in range(5)])
    it = pool.stream(1, "c1")
    first = [next(it) for _ in range(5)]
    assert {p.id for p, _ in first} == {f"p{i}" for i in range(5)} and all(s is None for _, s in first)
    it2 = pool.stream(2, "c2")
    texts = [loadgen.salted_messages(p, s)[0]["content"] for p, s in [next(it2) for _ in range(12)]]
    texts += [loadgen.salted_messages(p, s)[0]["content"] for p, s in first]
    assert len(set(texts)) == len(texts) and pool.salted == 12


def _lv(level, ok):
    return {"level": level, "summary": {"slo_ok": ok, "slo_checks": {"ttft_p95_ms": {"ok": ok}}}}


def test_n_star_needs_the_slo_at_every_lower_level():
    assert metrics.n_star([_lv(1, True), _lv(2, True), _lv(4, False), _lv(8, True)])[0] == 2
    assert metrics.n_star([_lv(1, True), _lv(2, False), _lv(4, True)])[0] == 1  # 4 passing does not count
    star, why = metrics.n_star([_lv(1, False)])
    assert star is None and "fails at level 1" in why
    star, why = metrics.n_star([_lv(1, True), _lv(2, True)])
    assert star == 2 and "above the grid" in why
    assert metrics.n_star([{"level": 1, "summary": {"slo_ok": None}}]) == (None, "no SLO given")


def test_cost_per_million_tokens():
    # 1000 tok/s per GPU at $3.60/GPU-h: 3.6M tokens per $3.60 -> $1 per 1M
    assert metrics.usd_per_1m_tokens(1000, 3.6) == pytest.approx(1.0)
    assert metrics.usd_per_1m_tokens(None, 3.6) is None and metrics.usd_per_1m_tokens(1000, None) is None


def test_load_mode_finds_n_star_on_a_server_that_saturates(store, tmp_path):
    # 4 slots, ~45 ms per request: concurrency 1 and 2 keep TTFT near 20 ms; 16 queues for hundreds of ms.
    prompts = write_prompts(tmp_path / "dev.jsonl", buckets=("short", "medium"), per_bucket=10, max_tokens=8)
    with FakeServer(FakeConfig(ttft_s=0.020, itl_s=0.004, max_concurrency=4)) as srv:
        cap = run_load(srv, prompts, concurrency=[1, 2, 16, 32], ref_e2e_ms=50, slo_ttft_p95_ms=120,
                       slo_tpot_p95_ms=20, gpu_price_usd_h=2.0, run_dir=str(tmp_path / "run"), label="fake")
        assert srv.max_active_seen <= 4
    assert ev.validate(cap) == [] and cap["kind"] == "capacity" and cap["claim_class"] == "derived"
    assert cap["data"]["n_star"] == 2, cap["data"]["curve"]
    assert "first fails at level 16" in cap["data"]["reason"]
    star = next(m for m in cap["metrics"] if m["name"] == "n_star")
    assert star["value"] == 2 and star["unit"] == "streams"
    curve = {c["level"]: c for c in cap["data"]["curve"]}
    assert set(curve) == {1, 2, 16}  # 32 not run after the first failure
    assert curve[16]["ttft_p95_ms"] > 120 and curve[2]["ttft_p95_ms"] <= 120
    tok_s = curve[2]["output_tok_s_per_gpu"]
    cost = next(m for m in cap["metrics"] if m["name"] == "usd_per_1m_output_tokens")["value"]
    assert cost == pytest.approx(2.0 / (tok_s * 3600) * 1e6)
    meas = store.load(cap["data"]["measurement"])
    assert ev.validate(meas) == [] and meas["kind"] == "serve_benchmark" and meas["claim_class"] == "observed"
    w = meas["identity"]["workload"]
    assert w["mode"] == "load" and w["concurrency"] == [1, 2, 16, 32] and w["slo"]["ttft_p95_ms"] == 120
    assert any("levels not run after the SLO failed: 32" in x for x in meas["limitations"])
    subjects = {m.get("subject") for m in meas["metrics"]}
    assert {"c1", "c2", "c16"} <= subjects
    inflight = next(m for m in cap["metrics"] if m["name"] == "mean_inflight")
    assert inflight["stat"] == "at_n_star" and inflight["value"] == curve[2]["mean_inflight"]
    assert cap["data"]["mean_inflight_at_n_star"] == inflight["value"]
    # every request's output and exact request body are kept, per level
    assert (tmp_path / "run/outputs/fake/load-c16").is_dir()
    some = next((tmp_path / "run/inputs/fake/load-c16").rglob("*.json"))
    assert json.loads(some.read_text())["ignore_eos"] is True
    rows = [json.loads(x) for x in (store.run_dir(meas["evidence_id"]) / "requests.jsonl").read_text().splitlines()]
    assert {r["level"] for r in rows} == {"c1", "c2", "c16"} and all(r["ok"] for r in rows)


def test_load_mode_counts_failures_against_the_slo(store, tmp_path):
    prompts = write_prompts(tmp_path / "dev.jsonl", buckets=("short",), per_bucket=10, max_tokens=4)
    with FakeServer(FakeConfig(ttft_s=0.004, itl_s=0.001, fail_rate=0.3, seed=1)) as srv:
        cap = run_load(srv, prompts, rates=[20.0], slo_ttft_p95_ms=1000, warmup=0, max_requests_per_level=20)
    assert cap["data"]["n_star"] is None and "failed" in cap["data"]["reason"]
    assert next(m for m in cap["metrics"] if m["name"] == "rate_star_rps")["unit"] == "req/s"
    meas = store.load(cap["data"]["measurement"])
    failed = next(m for m in meas["metrics"] if m["name"] == "failed_requests")
    assert failed["value"] > 0
    assert any("gpu price" in x for x in cap["coverage"]["missing"])


def test_load_without_slo_measures_but_claims_no_n_star(store, tmp_path):
    prompts = write_prompts(tmp_path / "dev.jsonl", buckets=("short",), per_bucket=10, max_tokens=4)
    with FakeServer(FakeConfig(ttft_s=0.003, itl_s=0.001)) as srv:
        cap = run_load(srv, prompts, rates=[10.0, 20.0], duration_s=0.5)
    assert cap["data"]["n_star"] is None and cap["data"]["reason"] == "no SLO given"
    assert any(x.startswith("slo") for x in cap["coverage"]["missing"])
    assert len(cap["data"]["curve"]) == 2


def test_closed_loop_is_labelled_a_stress_test(store, tmp_path):
    prompts = write_prompts(tmp_path / "dev.jsonl", buckets=("short",), per_bucket=6, max_tokens=4)
    with FakeServer(FakeConfig(ttft_s=0.003, itl_s=0.001)) as srv:
        cap = run_load(srv, prompts, concurrency=[2], closed_loop=True, duration_s=0.3, slo_ttft_p95_ms=1000)
    meas = store.load(cap["data"]["measurement"])
    assert meas["limitations"][0].startswith("CLOSED-LOOP STRESS TEST") and "closed-loop" in meas["summary"]
    assert cap["data"]["n_star"] is None and "closed-loop" in cap["data"]["reason"]


def test_e2e_ref_comes_from_reference_requests_not_slow_first_requests(store, tmp_path):
    # The first 2 requests take 400 ms longer (lazy init / JIT). They are the warm-up; e2e_ref is the median
    # of 3 later reference requests, so the offered rate is not lowered by the cold start.
    prompts = write_prompts(tmp_path / "dev.jsonl", buckets=("short",), per_bucket=10, max_tokens=6)
    with FakeServer(FakeConfig(ttft_s=0.010, itl_s=0.002, slow_first_n=2, slow_first_s=0.4)) as srv:
        cap = run_load(srv, prompts, concurrency=[1], warmup=2, ref_requests=3, duration_s=0.5,
                       slo_ttft_p95_ms=1000)
        tags = [r["messages"][0]["content"].split("]")[0] for r in srv.requests[:5]]
    assert tags == ["[ik-warm-0", "[ik-warm-1", "[ik-ref-0", "[ik-ref-1", "[ik-ref-2"]
    meas = store.load(cap["data"]["measurement"])
    d = meas["data"]
    assert d["e2e_ref_source"] == "reference_requests" and len(d["reference_e2e_ms"]) == 3
    assert min(d["warmup_e2e_ms"]) > 400 and d["e2e_ref_ms"] < 100, (d["warmup_e2e_ms"], d["e2e_ref_ms"])
    assert any("median of 3 reference request(s)" in x for x in meas["limitations"])


def test_n_star_served_with_too_little_load_in_flight_is_flagged(store, tmp_path):
    # --ref-e2e-ms far above the real latency: "concurrency 2" is offered at 8 req/s while each request
    # takes ~15 ms, so almost nothing is in flight. N* = 2 is nominal and the capacity record says so.
    prompts = write_prompts(tmp_path / "dev.jsonl", buckets=("short",), per_bucket=10, max_tokens=4)
    with FakeServer(FakeConfig(ttft_s=0.005, itl_s=0.002)) as srv:
        cap = run_load(srv, prompts, concurrency=[1, 2], ref_e2e_ms=250, duration_s=1.5, slo_ttft_p95_ms=1000)
    assert cap["data"]["n_star"] == 2, cap["data"]
    inflight = cap["data"]["mean_inflight_at_n_star"]
    assert inflight < 0.9 * 2
    assert cap["limitations"][0].startswith("N* = 2 is the nominal level") and "measured mean in-flight" in cap["summary"]


def test_load_mode_errors_and_outputs_carry_no_address(store, tmp_path):
    prompts = write_prompts(tmp_path / "dev.jsonl", buckets=("short",), per_bucket=4, max_tokens=4)
    with FakeServer(FakeConfig(host="127.0.0.2", refuse_after_models=True)) as srv:
        port = srv.port
        cap = run_load(srv, prompts, rates=[20.0], warmup=0, max_requests_per_level=4, duration_s=0.5,
                       slo_ttft_p95_ms=1000, run_dir=str(tmp_path / "run"))
    meas = store.load(cap["data"]["measurement"])
    texts = [json.dumps(cap), json.dumps(meas)]
    for root in (store.run_dir(meas["evidence_id"]), tmp_path / "run"):
        texts += [p.read_text() for p in root.rglob("*") if p.is_file()]
    assert any("<server>" in t for t in texts)
    for t in texts:
        assert "127.0.0.2" not in t and f":{port}" not in t and f"', {port}" not in t
