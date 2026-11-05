"""ledger trailers (formatting from evidence) and history (parsing git log on a temp repo)."""

import copy
import json
import subprocess

import pytest
from conftest import fake_call, make_benchmark

from inferencekit import cli, registry
from inferencekit import evidence as ev
from inferencekit.ops.ledger import leading_number, sparkline
from inferencekit.registry import OpError


def trailers(*ids, **kw) -> dict:
    return registry.call("ledger_trailers", {"ids": list(ids), **kw})


def lines(res) -> dict:
    return dict(line.split(": ", 1) for line in res["markdown"].splitlines())


def write_record(tmp_path, rec: dict, name: str) -> str:
    """A finalized (id-stamped) record as a JSON file: lets tests use v2 kinds the v1 store does not accept."""
    rec = ev.finalize(rec)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(rec))
    return str(path)


def bench(store, values, model="Qwen/Qwen3-8B", **kw):
    rec = make_benchmark(store, values, **kw)
    if model:  # make_benchmark leaves the model unset: add one, re-stamped so integrity still holds
        body = {k: v for k, v in rec.items() if k != "evidence_id"}
        body["identity"]["model"].update({"id": model, "quant": "fp8"})
        body["identity"]["gpu"].update({"name": "RTX 5090", "count": 1})
        rec = store.save(body)
    return rec


# --------------------------------------------------------------------------- trailers

def test_benchmark_trailers_with_comparison_delta_and_quality(store):
    base = bench(store, [100.0, 101.0, 99.0, 100.5, 100.2], t0=0.0, step=20.0)
    cand = bench(store, [110.0, 111.0, 109.0, 110.5, 110.2], t0=10.0, step=20.0)
    comp = registry.call("compare", {"baseline": base["evidence_id"], "candidate": cand["evidence_id"],
                                     "resamples": 500})
    corr = registry.call("check_correctness", {"reference": base["evidence_id"], "candidate": cand["evidence_id"]})
    res = trailers(cand["evidence_id"], comp["evidence_id"], corr["evidence_id"], baseline_commit="1a2b3c4d5e")
    t = lines(res)
    assert t["Perf-Workload"] == "Qwen/Qwen3-8B FP8, RTX 5090, synthetic, batch 1, prompt 128, gen 128"
    assert t["Perf-Decode-Tok-S"].startswith("110.2 [+10.0% (95% CI +")
    assert "Perf-TTFT-Ms" not in t  # the synthetic record has no TTFT metric: omitted
    assert "[+10.0% (95% CI +" in t["Perf-Decode-Tok-S"] and "improved vs 1a2b3c4]" in t["Perf-Decode-Tok-S"]
    assert t["Perf-Quality"] == "greedy prefix agreement 100.00% (1/1 identical) PASS"
    assert t["Perf-Baseline"] == "1a2b3c4d5e"
    assert t["Perf-Evidence"].count("ev_") == 3
    keys = [line.split(":")[0] for line in res["markdown"].splitlines()]
    assert keys == ["Perf-Workload", "Perf-Decode-Tok-S", "Perf-Quality", "Perf-Evidence", "Perf-Baseline"]
    omitted = {o["key"] for o in res["omitted"]}
    assert {"Perf-TTFT-Ms", "Perf-P95-Latency-Ms", "Perf-Cost", "Perf-Capacity-N", "Perf-Goodput"} <= omitted
    classes = {x["key"]: x["claim_class"] for x in res["trailers"]}
    assert classes["Perf-Decode-Tok-S"] == "derived" and classes["Perf-Workload"] == "observed"


def test_missing_metrics_are_omitted_never_zero(store):
    rec = ev.build(fake_call(), kind="benchmark", claim_class="observed", summary="s",
                   identity_=ev.identity(workload={"batch": 1}),
                   metrics=[ev.metric("decode_tok_s", 50.0, "tok/s", "median", n=3),
                            ev.metric("ttft_ms", None, "ms", "median", n=0)])
    rec = store.save(rec)
    res = trailers(rec["evidence_id"])
    assert "Perf-TTFT-Ms" not in res["markdown"] and "Perf-Cost" not in res["markdown"]
    assert lines(res)["Perf-Decode-Tok-S"] == "50"
    assert ": 0" not in res["markdown"]


def test_plain_delta_from_baseline_ev_and_single_stream_cost(store):
    base = bench(store, [180.0, 180.0, 180.0])
    cand = bench(store, [185.3, 185.3, 185.3], t0=5.0)
    res = trailers(cand["evidence_id"], baseline_ev=base["evidence_id"], gpu_price_usd_h=0.62)
    t = lines(res)
    assert t["Perf-Decode-Tok-S"].endswith(f"[+2.9% vs {base['evidence_id'][:15]}, ratio of medians, no CI]")
    assert t["Perf-Cost"] == "$0.93 per 1M output tokens at $0.62/GPU-h (1 stream)"
    assert t["Perf-Baseline"] == base["evidence_id"][:15]
    assert any("ratio of medians without a CI" in x for x in res["limitations"])
    assert any("single-stream" in x for x in res["limitations"])


def test_capacity_trailers_from_v2_records(store, tmp_path):
    w = {"prompt_len": None, "gen_len": None, "batch": None, "seed": 0, "mode": "load", "concurrency": None,
         "prompt_set": {"name": "dev", "sha256": "ab" * 32},
         "slo": {"ttft_p95_ms": 300, "tpot_p95_ms": 50, "max_fail": 0}}
    ident = ev.identity(gpu={"name": "RTX 5090", "count": 1}, engine={"name": "vllm", "version": "0.11.0"},
                        model={"id": "org/model", "quant": "nvfp4"})
    ident["workload"] = w
    cap = ev.build(fake_call("bench_serve", ("bench", "serve")), kind="benchmark", claim_class="observed",
                   summary="capacity", identity_=ident,
                   metrics=[ev.metric("n_star", 24, "streams", "max"),
                            ev.metric("goodput_rps", 18.2, "req/s", "rate", subject="N*"),
                            ev.metric("e2e_ms", 1452.0, "ms", "p95", n=480),
                            ev.metric("output_tok_s_per_gpu", 2400.0, "tok/s", "rate")])
    cap["kind"] = "capacity"  # a v2 kind (written by bench serve); built here by hand
    path = write_record(tmp_path, cap, "cap")
    res = trailers(path, gpu_price_usd_h=2.0)
    t = lines(res)
    assert t["Perf-Workload"] == "org/model NVFP4, RTX 5090, vllm 0.11.0, open-loop load, prompts dev (abababababab)"
    assert t["Perf-Capacity-N"] == "24 streams (SLO p95 TTFT <= 300 ms, p95 TPOT <= 50 ms, failures <= 0)"
    assert t["Perf-Goodput"] == "18.2 req/s at N*=24"
    assert "Perf-P95-Latency-Ms" not in t  # p95 e2e is a single-user headline; load levels do not feed it
    assert t["Perf-Cost"] == "$0.23 per 1M output tokens at $2.00/GPU-h (at N*)"
    assert "Perf-Decode-Tok-S" not in t


def test_serve_benchmark_and_quality_lane_records(store, tmp_path):
    ident = ev.identity(gpu={"name": "RTX 5090", "count": 1}, engine={"name": "vllm", "version": "0.11.0"},
                        model={"id": "org/model", "quant": "fp8"})
    ident["workload"] = {"prompt_len": None, "gen_len": 256, "batch": 1, "seed": 0, "mode": "single",
                         "prompt_set": {"name": "dev-v1", "sha256": "cd" * 32}}
    mets = []
    for subj, dec, ttft, e2e in (("short", 150.0, 20.0, 1700.0), ("long", 140.0, 90.0, 1900.0),
                                 ("all", 145.0, 30.0, 1850.0)):
        mets += [ev.metric("decode_tok_s", dec, "tok/s", "median", n=30, ci95=[dec - 1, dec + 1], subject=subj),
                 ev.metric("decode_tok_s", dec + 5, "tok/s", "p95", n=30, subject=subj),
                 ev.metric("ttft_ms", ttft, "ms", "median", n=30, ci95=[ttft - 1, ttft + 1], subject=subj),
                 ev.metric("e2e_ms", e2e, "ms", "p95", n=30, subject=subj)]
    single = ev.build(fake_call("bench_serve", ("bench", "serve")), kind="benchmark", claim_class="observed",
                      summary="serve", identity_=ident, metrics=mets, data={"label": "vllm-tuned"})
    single["kind"] = "serve_benchmark"
    load = ev.build(fake_call("bench_serve", ("bench", "serve")), kind="benchmark", claim_class="observed",
                    summary="load", identity_=copy.deepcopy(ident), metrics=[ev.metric("ttft_ms", 400.0, "ms", "p50", subject="c8")],
                    data={"label": "vllm-tuned"})
    load["kind"], load["identity"]["workload"] = "serve_benchmark", {**ident["workload"], "mode": "load"}
    traj = ev.build(fake_call("quality_trajectory"), kind="correctness", claim_class="derived", summary="q",
                    metrics=[ev.metric("top1_agreement_pct", 98.86, "%", "pooled_mean", subject="vllm-tuned"),
                             ev.metric("top1_agreement_pct", 99.2, "%", "pooled_mean", subject="vllm-tuned @pos 0-63"),
                             ev.metric("top1_agreement_pct", 98.99, "%", "pooled_mean", subject="hf-aa")],
                    data={"v2_kind": "quality_trajectory", "verdict": "PASS", "noise_floor": "hf-aa",
                          "verdicts": {"vllm-tuned": "PASS"}})
    task = ev.build(fake_call("quality_gsm8k"), kind="correctness", claim_class="observed", summary="g",
                    metrics=[ev.metric("gsm8k_accuracy_pct", 85.5, "%", "rate", n=200, ci95=[80.0, 90.0],
                                       subject="vllm-tuned")],
                    data={"v2_kind": "quality_task", "task": "gsm8k", "label": "vllm-tuned"})
    cap = ev.build(fake_call("bench_serve", ("bench", "serve")), kind="benchmark", claim_class="derived",
                   summary="cap", identity_=copy.deepcopy(ident), metrics=[ev.metric("n_star", None, "streams", "threshold")],
                   data={"label": "vllm-tuned", "n_star": None, "reason": "SLO missed at c1"})
    cap["kind"] = "capacity"
    paths = [write_record(tmp_path, r, n) for r, n in ((single, "s"), (load, "l"), (traj, "t"), (task, "g"),
                                                        (cap, "c"))]
    res = trailers(*paths)
    t = lines(res)
    assert t["Perf-Workload"] == ("org/model FP8, RTX 5090, vllm 0.11.0, single-user, batch 1, gen 256, "
                                  "prompts dev-v1 (cdcdcdcdcdcd)")
    assert t["Perf-Decode-Tok-S"] == "145 (95% CI 144-146)"  # the 'all' subject is the overall value
    assert t["Perf-TTFT-Ms"] == "30 (95% CI 29-31)" and t["Perf-P95-Latency-Ms"] == "1850"
    assert t["Perf-Quality"] == ("top1-agreement 98.86% (noise floor 98.99%) PASS [vllm-tuned]; "
                                 "gsm8k accuracy 85.5% (95% CI 80-90%) (n=200) [vllm-tuned]")
    assert "Perf-Capacity-N" not in t
    assert any(o["key"] == "Perf-Capacity-N" and "SLO missed at c1" in o["reason"] for o in res["omitted"])


def sampling_record(tmp_path, label="myengine", verdict="PASS", excess=0.0011) -> str:
    """A `quality sampling` record as the op writes it with --baseline-url: no accuracy metric, its number is the
    excess first-token TV (draws vs the server's own logprobs); the baseline's own excess comes after it."""
    rec = ev.build(fake_call("quality_sampling", ("quality", "sampling")), kind="correctness", claim_class="derived",
                   summary="sampling",
                   metrics=[ev.metric("first_token_tv_excess", excess, "probability", "mean", n=32,
                                      ci95=[-0.0085, 0.0116] if excess is not None else [None, None], subject=label),
                            ev.metric("first_token_tv", 0.21, "probability", "mean", n=32, subject=label),
                            ev.metric("first_token_tv_expected", 0.209, "probability", "mean", n=32, subject=label),
                            ev.metric("first_token_out_of_support", 0.0, "fraction", "rate", n=8192,
                                      ci95=[0.0, 0.0005], subject=label),
                            ev.metric("first_token_tv_excess", 0.004, "probability", "mean", n=30,
                                      ci95=[-0.002, 0.01], subject="baseline"),
                            ev.metric("sampling_requests", 16512, "requests", "count")],
                   data={"v2_kind": "quality_task", "task": "sampling", "label": label, "verdict": verdict})
    return write_record(tmp_path, rec, "sampling")


def test_sampling_record_gives_its_excess_tv_in_the_quality_trailer(store, tmp_path):
    t = lines(trailers(sampling_record(tmp_path)))
    assert t["Perf-Quality"] == "first token tv excess 0.0011 (95% CI -0.0085 to 0.0116) (n=32) PASS [myengine]"
    assert "0.0011%" not in t["Perf-Quality"]  # a probability, not a percentage


def test_sampling_trailer_never_shows_the_baselines_number(store, tmp_path):
    """No informative prompt on the engine (its excess is null): the baseline's excess in the same record must not
    stand in for it."""
    t = lines(trailers(sampling_record(tmp_path, verdict="UNPROVEN", excess=None)))
    assert t["Perf-Quality"] == "first token tv excess n/a UNPROVEN [myengine]"


def capacity_record(tmp_path, name="cap", price=0.62, tok_s=1000.0, gpus=1, level_kind="concurrency",
                    reason="SLO first fails at level 32 (ttft_p95_ms)", with_cost=True, with_tput=True):
    """A capacity record shaped like bench serve's (cost computed at ``price`` from per-GPU throughput)."""
    ident = ev.identity(gpu={"name": "RTX 5090", "count": gpus}, engine={"name": "vllm", "version": "0.11.0"},
                        model={"id": "org/model", "quant": "fp8"})
    ident["workload"] = {"prompt_len": None, "gen_len": None, "batch": None, "seed": 0, "mode": "load"}
    star = "n_star" if level_kind == "concurrency" else "rate_star_rps"
    mets = [ev.metric(star, 24 if level_kind == "concurrency" else 4.0, "streams", "threshold")]
    if with_tput:
        mets.append(ev.metric("output_tok_s_per_gpu", tok_s, "tok/s", "at_n_star"))
    if with_cost:
        mets.append(ev.metric("usd_per_1m_output_tokens", price / (tok_s * 3600) * 1e6, "USD", "at_n_star"))
    rec = ev.build(fake_call("bench_serve", ("bench", "serve")), kind="benchmark", claim_class="derived",
                   summary="cap", identity_=ident, metrics=mets,
                   data={"label": "vllm-tuned", "level_kind": level_kind, "n_star": 24, "reason": reason,
                         "slo": {"ttft_p95_ms": 300}, "gpu_price_usd_h": price, "gpu_count": gpus})
    rec["kind"] = "capacity"
    return write_record(tmp_path, rec, name)


def test_capacity_cost_is_never_labelled_with_a_price_it_was_not_computed_at(store, tmp_path):
    path = capacity_record(tmp_path)  # $0.62/GPU-h, 1000 tok/s/GPU at N* -> $0.17 per 1M
    for kw in ({}, {"gpu_price_usd_h": 0.62}):
        res = trailers(path, **kw)
        assert lines(res)["Perf-Cost"] == "$0.17 per 1M output tokens at $0.62/GPU-h (at N*)"
    res = trailers(path, gpu_price_usd_h=2.0)  # another price: repriced from the throughput, derived
    assert lines(res)["Perf-Cost"] == "$0.56 per 1M output tokens at $2.00/GPU-h (at N*, repriced)"
    assert {x["key"]: x["claim_class"] for x in res["trailers"]}["Perf-Cost"] == "derived"
    assert any("repriced at $2.00/GPU-h" in x and "$0.62/GPU-h ($0.17)" in x for x in res["limitations"])
    # No throughput to reprice from: the record's own cost at its own price, and a note that the flag was ignored.
    res = trailers(capacity_record(tmp_path, "cap2", with_tput=False), gpu_price_usd_h=2.0)
    assert lines(res)["Perf-Cost"] == "$0.17 per 1M output tokens at $0.62/GPU-h (at N*)"
    assert any("--gpu-price-usd-h 2 ignored" in x for x in res["limitations"])


def test_capacity_n_is_server_wide_and_multi_gpu_cost_counts_every_gpu(store, tmp_path):
    t = lines(trailers(capacity_record(tmp_path, gpus=2)))
    assert t["Perf-Capacity-N"] == "24 streams across 2 GPUs (SLO p95 TTFT <= 300 ms)" and "/GPU" not in t[
        "Perf-Capacity-N"]
    t = lines(trailers(capacity_record(tmp_path, "rate", level_kind="rate",
                                       reason="SLO holds at every level tried: the true N* may be above the grid")))
    assert t["Perf-Capacity-N"] == "max rate at least 4 req/s (SLO p95 TTFT <= 300 ms)"
    # Single-stream fallback on a tensor-parallel (2-GPU) engine: one stream occupies both GPUs.
    rec = ev.build(fake_call(), kind="benchmark", claim_class="observed", summary="s",
                   identity_=ev.identity(gpu={"name": "H100", "count": 2}, workload={"batch": 1}),
                   metrics=[ev.metric("decode_tok_s", 100.0, "tok/s", "median", n=3)])
    rec = store.save(rec)
    t = lines(trailers(rec["evidence_id"], gpu_price_usd_h=1.0))
    assert t["Perf-Cost"] == "$5.56 per 1M output tokens at $1.00/GPU-h x 2 GPUs (1 stream)"


def labelled(store, label, value):
    rec = ev.build(fake_call(), kind="benchmark", claim_class="observed", summary="s",
                   identity_=ev.identity(engine={"name": label}, workload={"batch": 1}),
                   metrics=[ev.metric("decode_tok_s", value, "tok/s", "median", n=3, ci95=[value - 1, value + 1])])
    return store.save(rec)


def test_several_engines_are_refused_unless_label_picks_one(store):
    base, tuned = labelled(store, "vllm-recipe", 100.0), labelled(store, "vllm-tuned", 120.0)
    with pytest.raises(OpError, match="--label"):
        trailers(base["evidence_id"], tuned["evidence_id"], baseline_commit="abcdef1")
    t = lines(trailers(base["evidence_id"], tuned["evidence_id"], label="vllm-tuned"))
    assert t["Perf-Decode-Tok-S"] == "120 (95% CI 119-121)"
    with pytest.raises(OpError, match="no-such"):
        trailers(tuned["evidence_id"], label="no-such")
    # Two single-user records of one engine (a baseline and a candidate) are ambiguous too.
    again = labelled(store, "vllm-tuned", 90.0)
    with pytest.raises(OpError, match="2 single-user benchmark records"):
        trailers(tuned["evidence_id"], again["evidence_id"])


def test_text_mode_prints_caveats_on_stderr_not_in_the_trailers(store, capsys):
    base = bench(store, [180.0, 180.0, 180.0])
    cand = bench(store, [185.3, 185.3, 185.3], t0=5.0)
    assert cli.main(["--runs-dir", str(store.root), "ledger", "trailers", cand["evidence_id"],
                     "--baseline-ev", base["evidence_id"], "--baseline-commit", "abcdef1"]) == 0
    io = capsys.readouterr()
    assert "Perf-Decode-Tok-S: 185.3 [+2.9% vs abcdef1, ratio of medians, no CI]" in io.out
    assert "limitation" not in io.out and "omitted" not in io.out
    assert "limitation: decode_tok_s delta is a ratio of medians without a CI" in io.err
    assert "omitted Perf-Cost" in io.err
    side = json.loads(next((store.root / "ledger").glob("trailers-*.json")).read_text())
    assert any(t["key"] == "Perf-Decode-Tok-S" and t["claim_class"] == "derived" for t in side["trailers"])


def test_newlines_in_evidence_cannot_forge_trailers(store):
    rec = ev.build(fake_call(), kind="benchmark", claim_class="observed", summary="s",
                   identity_=ev.identity(model={"id": "evil\nPerf-Decode-Tok-S: 9999"}, workload={"batch": 1}),
                   metrics=[ev.metric("decode_tok_s", 10.0, "tok/s", "median", n=2)])
    rec = store.save(rec)
    res = trailers(rec["evidence_id"])
    assert [ln for ln in res["markdown"].splitlines() if ln.startswith("Perf-Decode-Tok-S")] == \
        ["Perf-Decode-Tok-S: 10"]


def test_edited_evidence_is_refused(store, tmp_path):
    rec = bench(store, [100.0, 101.0])
    rec["metrics"][0]["value"] = 999.0
    path = tmp_path / "edited.json"
    path.write_text(json.dumps(rec))
    with pytest.raises(OpError, match="refusing"):
        trailers(str(path))
    with pytest.raises(OpError):
        trailers(rec["evidence_id"], baseline_commit="not a sha; rm -rf")


def test_json_format_and_cli_prints_the_block(store, capsys):
    rec = bench(store, [100.0, 101.0, 102.0])
    res = trailers(rec["evidence_id"], format="json")
    assert "markdown" not in res and res["text"].startswith("Perf-Workload: ")
    assert cli.main(["--runs-dir", str(store.root), "ledger", "trailers", rec["evidence_id"][:12]]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Perf-Workload: ") and "Perf-Decode-Tok-S: 101" in out


# --------------------------------------------------------------------------- history

def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                           "-c", "commit.gpgsign=false", *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "no-global-gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "perf/test")
    wl = "Perf-Workload: m FP8, GPU, batch 1, prompt 128, gen 128"
    msgs = [
        ("perf: other workload", "Perf-Workload: m FP8, GPU, batch 8\nPerf-Decode-Tok-S: 900"),
        ("perf: baseline", f"{wl}\nPerf-Decode-Tok-S: 170.1 (95% CI 169.9-170.3)\nPerf-TTFT-Ms: 62.0"),
        ("docs: no numbers", None),
        ("perf: fuse qkv", f"{wl}\nPerf-Decode-Tok-S: 185.3 (95% CI 185.2-185.3) [+8.9% vs abc1234]\n"
                           "Perf-TTFT-Ms: 60.7\nPerf-Cost: $0.91 per 1M output tokens at $0.62/h (1 stream)"),
        ("perf: try graphs", f"{wl}\nPerf-Decode-Tok-S: 180.0\nPerf-TTFT-Ms: 59.9\nPerf-Evidence: ev_1, ev_2"),
    ]
    for subject, body in msgs:
        args = ["commit", "-q", "--allow-empty", "-m", subject] + (["-m", body] if body else [])
        git(r, *args)
    return r


def test_history_table_and_sparklines(repo, store):
    res = registry.call("history", {"repo": str(repo)})
    assert len(res["commits"]) == 4  # the commit without trailers is skipped
    md = res["markdown"]
    assert "| commit | date | subject | Decode-Tok-S | TTFT-Ms | Cost |" in md
    rows = [ln for ln in md.splitlines() if ln.startswith("| ") and "---" not in ln][1:]
    assert "try graphs" in rows[0] and "other workload" in rows[-1]  # newest first, like git log
    assert rows[-1].endswith("| 900 | · | · |")  # missing values are a dot, never 0
    dec = res["metrics"]["Perf-Decode-Tok-S"]
    assert dec["sparkline"] == "▁█▆" and dec["best"]["value"] == 185.3  # other workload left out
    assert res["metrics"]["Perf-TTFT-Ms"]["best"]["value"] == 59.9
    assert res["metrics"]["Perf-Cost"]["sparkline"] == "·▅·"
    assert any("another Perf-Workload" in n for n in res["notes"])
    assert "Perf-Evidence" not in res["metrics"]


def test_history_metric_filter_and_tag_best(repo, store):
    res = registry.call("history", {"repo": str(repo), "metric": "decode_tok_s", "tag_best": True})
    assert list(res["metrics"]) == ["Perf-Decode-Tok-S"]
    assert res["tags"] == [{"tag": "perf/best-decode-tok-s", "commit": res["metrics"]["Perf-Decode-Tok-S"]["best"]
                            ["short"], "value": 185.3}]
    tagged = git(repo, "log", "-1", "--format=%s", "perf/best-decode-tok-s").strip()
    assert tagged == "perf: fuse qkv"
    with pytest.raises(OpError, match="(?i)perf-nope"):
        registry.call("history", {"repo": str(repo), "metric": "nope"})


def test_history_errors(tmp_path, store):
    with pytest.raises(OpError):
        registry.call("history", {"repo": str(tmp_path / "missing")})
    with pytest.raises(OpError):
        registry.call("history", {"repo": str(tmp_path), "rev": "--output=x"})


def test_quality_direction_follows_the_metric():
    from inferencekit.ops.ledger import direction

    assert direction("Perf-Quality", "greedy prefix agreement 100.00% PASS") is True
    assert direction("Perf-Quality", "asr-wer 2.1% (noise floor 2.3%)") is False
    assert direction("Perf-TTFT-Ms") is False and direction("Perf-My-Thing") is None


def test_number_parsing_and_sparkline():
    assert leading_number("185.3 (95% CI 185.2-185.3) [+2.1% vs 1a2b3c4]") == 185.3
    assert leading_number("$0.91 per 1M output tokens") == 0.91
    assert leading_number("gsm8k accuracy 85.5% (n=100)") == 85.5
    assert leading_number("tf-agreement 98.86% (noise floor 98.99%)") == 98.86
    assert leading_number("none") is None
    assert sparkline([1, None, 3, 2]) == "▁·█▅"
    assert sparkline([5, 5]) == "▅▅" and sparkline([None]) == ""
