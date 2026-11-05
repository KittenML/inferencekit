"""bench serve, single-user mode and the streaming client, against a fake OpenAI server (no GPU)."""

import asyncio
import json

import pytest

from inferencekit import evidence as ev
from inferencekit import registry
from inferencekit.registry import OpError
from inferencekit.serve import client as cl
from inferencekit.serve import metrics, profiles, prompts_io

from fakes.fake_openai_server import FakeConfig, FakeServer

MSG = [{"role": "user", "content": "hello"}]


@pytest.fixture(autouse=True)
def endpoint_key(tmp_path, monkeypatch):
    """Keep the per-machine endpoint key of bench serve inside the test's temp dir."""
    path = tmp_path / "endpoint.key"
    monkeypatch.setenv("INFERENCEKIT_KEY_FILE", str(path))
    return path


def write_prompts(path, buckets=("short", "long"), per_bucket=2, max_tokens=16):
    lines = [json.dumps({"id": f"{b}-{i}", "bucket": b, "messages": [{"role": "user", "content": f"{b} {i}"}],
                         "max_tokens": max_tokens, "source": "test", "kind": "chat"})
             for b in buckets for i in range(per_bucket)]
    path.write_text("\n".join(lines) + "\n")
    return path


def one(srv, engine="vllm", api="chat", n=16, token_ids="auto", tokenizer=None):
    ep = cl.Endpoint.parse(srv.base_url)
    body, _ = profiles.build_body(api, engine, "fake-model", MSG, n, token_ids=token_ids)
    res = asyncio.run(cl.stream(ep, api, body, rid="x", bucket="short"))
    cl.count_tokens(res, tokenizer)
    return res, metrics.request_row(res)


# --------------------------------------------------------------------------- client timing

def test_measured_ttft_and_tpot_match_the_servers_delays():
    with FakeServer(FakeConfig(ttft_s=0.040, itl_s=0.008)) as srv:
        rows = [one(srv, n=16)[1] for _ in range(3)]
    ttft = sorted(r["ttft_ms"] for r in rows)[1]
    tpot = sorted(r["tpot_ms"] for r in rows)[1]
    assert 39.0 <= ttft <= 50.0, ttft  # connect + loopback overhead only
    assert 7.5 <= tpot <= 8.8, tpot
    assert all(r["n_tokens"] == 16 for r in rows)
    assert 1000 / 8.8 <= sorted(r["decode_tok_s"] for r in rows)[1] <= 1000 / 7.5


def test_several_tokens_per_event_are_counted_by_ids_not_events():
    # 4 tokens per SSE event at 5 ms/token: 200 tok/s. Counting events would report 50 tok/s.
    with FakeServer(FakeConfig(ttft_s=0.02, itl_s=0.005, tokens_per_event=4)) as srv:
        res, row = one(srv, n=32)
    assert res.token_ids == list(range(1000, 1032))
    assert row["n_tokens"] == 32 and row["n_token_events"] == 8 and row["tokens_per_event_max"] == 4
    assert row["token_count_method"] == "stream_token_ids:choice_token_ids"
    # (32 - 4 tokens in the first event) / (7 events x 20 ms) = 200 tok/s
    assert 175 <= row["decode_tok_s"] <= 205, row["decode_tok_s"]


@pytest.mark.parametrize("cfg,engine,api,token_ids,method", [
    (dict(ids="choice"), "vllm", "chat", "auto", "stream_token_ids:choice_token_ids"),
    (dict(ids="choice"), "vllm", "completions", "auto", "stream_token_ids:choice_token_ids"),
    (dict(ids="logprobs_id"), "llamacpp", "chat", "logprobs", "stream_token_ids:logprobs_ids"),
    (dict(ids="token_id_str"), "vllm", "completions", "logprobs", "stream_token_ids:logprobs_ids"),
    (dict(ids="token_id_str", ids_on_request_only=False), "generic", "completions", "auto",
     "stream_token_ids:logprobs_ids"),
    (dict(ids="token_id_str", ids_on_request_only=False), "generic", "chat", "auto",
     "stream_token_ids:logprobs_ids"),
    (dict(ids="logprobs_id"), "llamacpp", "chat", "auto", "usage"),  # ids would cost decode time: not asked
    (dict(ids="none"), "sglang", "chat", "auto", "usage"),
    (dict(ids="none", usage=False), "generic", "chat", "auto", "sse_events"),
])
def test_token_count_methods(cfg, engine, api, token_ids, method):
    with FakeServer(FakeConfig(ttft_s=0.005, itl_s=0.001, **cfg)) as srv:
        res, row = one(srv, engine=engine, api=api, n=12, token_ids=token_ids)
    assert res.ok, res.error
    assert row["token_count_method"] == method
    assert row["n_tokens"] == 12
    if method.startswith("stream_token_ids"):
        assert res.token_ids == list(range(1000, 1012))


def test_padded_ids_cannot_raise_the_token_count():
    # The server streams 3 extra ids but reports the true count in usage: the smaller count is used.
    with FakeServer(FakeConfig(ttft_s=0.005, itl_s=0.002, pad_ids=3)) as srv:
        res, row = one(srv, n=12)
    assert res.ok and len(res.token_ids) == 15
    assert row["n_tokens"] == 12 and row["n_ids"] == 15 and row["n_usage"] == 12
    assert row["token_count_method"] == "min(stream_token_ids:choice_token_ids,usage)"
    assert "15 streamed ids vs usage.completion_tokens=12" in row["count_mismatch"]
    # decode speed from 11 tokens after the first over 11 x 2 ms, not 14 tokens
    assert row["decode_tok_s"] <= 1000 / 2 * 1.05, row["decode_tok_s"]


def test_retokenize_when_no_ids_and_no_usage():
    with FakeServer(FakeConfig(ttft_s=0.005, itl_s=0.001, ids="none", usage=False)) as srv:
        _, row = one(srv, engine="generic", n=10, tokenizer=lambda text: text.split())
    assert row["token_count_method"] == "retokenize" and row["n_tokens"] == 10


def test_unchunked_streams_and_api_prefix_are_handled():
    with FakeServer(FakeConfig(ttft_s=0.005, itl_s=0.001, chunked=False)) as srv:
        ep = cl.Endpoint.parse(srv.base_url + "/v1/")
        body, _ = profiles.build_body("chat", "vllm", "fake-model", MSG, 8)
        res = asyncio.run(cl.stream(ep, "chat", body, rid="x", bucket="b"))
    cl.count_tokens(res)
    assert res.ok and res.n_tokens == 8 and res.finish_reason == "length"


def test_profiles_send_only_documented_fields():
    with FakeServer(FakeConfig(ttft_s=0.001, itl_s=0.0005)) as srv:
        for engine in profiles.ENGINE_KINDS:
            one(srv, engine=engine, n=4)
        reqs = {e: r for e, r in zip(profiles.ENGINE_KINDS, srv.requests)}
    assert reqs["vllm"]["ignore_eos"] and reqs["vllm"]["min_tokens"] == 4 and reqs["vllm"]["return_token_ids"]
    assert reqs["sglang"]["ignore_eos"] and "return_token_ids" not in reqs["sglang"]  # rejected when streaming
    assert reqs["llamacpp"]["ignore_eos"] and "min_tokens" not in reqs["llamacpp"] and "logprobs" not in reqs["llamacpp"]
    assert reqs["trtllm"]["min_tokens"] == 4 and "return_token_ids" not in reqs["trtllm"]
    assert not {"ignore_eos", "min_tokens", "return_token_ids"} & set(reqs["generic"])
    for r in reqs.values():
        assert r["stream"] is True and r["stream_options"] == {"include_usage": True} and r["temperature"] == 0.0


def test_failures_are_reported_not_raised():
    with FakeServer(FakeConfig(ttft_s=0.002, itl_s=0.001, fail_rate=1.0)) as srv:
        res, row = one(srv, n=4)
    assert not res.ok and res.status == 500 and "fake overload" in res.error
    assert row["ttft_ms"] is None and row["e2e_ms"] is None
    with FakeServer(FakeConfig(ttft_s=0.002, itl_s=0.001, fail_rate=1.0, fail_mode="midstream")) as srv:
        res, _ = one(srv, n=8)
    assert not res.ok and "ended early" in res.error  # a dropped stream is a failure, not a short success
    ep = cl.Endpoint.parse("http://127.0.0.1:9")  # nothing listens on the discard port
    body, _ = profiles.build_body("chat", "generic", "m", MSG, 4)
    res = asyncio.run(cl.stream(ep, "chat", body, rid="x", bucket="b", timeout_s=2))
    assert not res.ok and res.error


# --------------------------------------------------------------------------- prompts

def test_prompt_set_loading_and_checks(tmp_path):
    p = write_prompts(tmp_path / "dev.jsonl", buckets=("long", "short", "very_short"), per_bucket=3)
    ps = prompts_io.load(p)
    assert ps.buckets() == ["very_short", "short", "long"] and len(ps.prompts) == 9 and len(ps.sha256) == 64
    assert len(prompts_io.load(p, buckets=["short"], limit_per_bucket=2).prompts) == 2
    (tmp_path / "dev.manifest.json").write_text(json.dumps({"name": "dev", "sha256": "0" * 64}))
    with pytest.raises(OpError, match="does not match its manifest"):
        prompts_io.load(p)
    dup = tmp_path / "dup.jsonl"
    dup.write_text(p.read_text().splitlines()[0] + "\n" + p.read_text().splitlines()[0] + "\n")
    with pytest.raises(OpError, match="duplicate id"):
        prompts_io.load(dup)
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "../x", "bucket": "short", "messages": MSG}) + "\n")
    with pytest.raises(OpError, match="'id'"):
        prompts_io.load(bad)


# --------------------------------------------------------------------------- the operation, single mode

def run_single(srv, prompts, **kw):
    args = {"base_url": srv.base_url, "model": "fake-model", "prompts": str(prompts), "engine_kind": "vllm",
            "rounds": 2, "warmup": 1, "resamples": 300}
    args.update(kw)
    return registry.call("bench_serve", args)


def metric(rec, name, stat="median", subject="all"):
    return next(m for m in rec["metrics"] if m["name"] == name and m["stat"] == stat and m.get("subject") == subject)


def test_single_mode_record(store, tmp_path):
    prompts = write_prompts(tmp_path / "dev.jsonl")
    with FakeServer(FakeConfig(ttft_s=0.030, itl_s=0.006)) as srv:
        rec = run_single(srv, prompts, run_dir=str(tmp_path / "run"), label="fake-vllm")
        n_requests = len(srv.requests)
    assert ev.validate(rec) == [] and ev.verify(rec)["id_ok"]
    assert rec["schema"] == "inferencekit.evidence.v2" and rec["kind"] == "serve_benchmark"
    assert rec["claim_class"] == "observed"
    assert n_requests == 1 + 2 * 4  # warm-up + rounds x prompts
    assert srv.base_url not in json.dumps(rec) and rec["data"]["base_url_kind"] == "local"  # address as digest only
    w = rec["identity"]["workload"]
    assert w["mode"] == "single" and w["gen_len"] == 16 and w["endpoint_kind"] == "openai_chat"
    assert w["prompt_set"]["sha256"] == prompts_io.load(prompts).sha256
    assert 29 <= metric(rec, "ttft_ms")["value"] <= 42
    assert 5.6 <= metric(rec, "tpot_ms")["value"] <= 6.8
    for subject in ("short", "long", "all"):
        m = metric(rec, "decode_tok_s", subject=subject)
        assert m["n"] == 4 if subject != "all" else m["n"] == 8
        assert m["ci95"][0] is not None and m["ci95"][0] <= m["value"] <= m["ci95"][1]
    assert metric(rec, "e2e_ms", "p95")["value"] >= metric(rec, "e2e_ms")["value"]
    assert len(rec["data"]["rounds"]) == 2 and {"decode_tok_s", "t_start", "total_ms"} <= set(rec["data"]["rounds"][0])
    assert rec["data"]["token_count_methods"] == {"stream_token_ids:choice_token_ids": 8}
    assert rec["data"]["request_profile"]["asked"]["exact_length"] == ["ignore_eos", "min_tokens"]
    # outputs: text, token ids, finish reason, usage, timings for every prompt
    out = json.loads((tmp_path / "run/outputs/fake-vllm/short/short-0.json").read_text())
    assert out["token_ids"] == list(range(1000, 1016)) and out["finish_reason"] == "length"
    assert out["usage"]["completion_tokens"] == 16 and len(out["timings"]["event_ms"]) == 16
    index = json.loads((tmp_path / "run/outputs/fake-vllm/index.json").read_text())
    assert len(index["files"]) == 4
    assert json.loads((tmp_path / "run/metrics/fake-vllm.json").read_text())["speed"]["per_bucket"]["all"]
    roles = {a["role"] for a in rec["artifacts"]}
    assert roles == {"requests", "outputs_index"}
    rows = (store.run_dir(rec["evidence_id"]) / "requests.jsonl").read_text().splitlines()
    assert len(rows) == 8


def test_single_mode_without_run_dir_keeps_outputs_with_the_record(store, tmp_path):
    prompts = write_prompts(tmp_path / "dev.jsonl", buckets=("short",), per_bucket=2, max_tokens=6)
    with FakeServer(FakeConfig(ttft_s=0.005, itl_s=0.001)) as srv:
        rec = run_single(srv, prompts, rounds=1, warmup=0)
    assert (store.run_dir(rec["evidence_id"]) / "outputs/vllm/short/short-1.json").is_file()
    assert "fewer than 2 rounds: no confidence interval" in rec["limitations"]


def test_early_eos_and_failures_are_flagged(store, tmp_path):
    prompts = write_prompts(tmp_path / "dev.jsonl", buckets=("short",), per_bucket=4)
    with FakeServer(FakeConfig(ttft_s=0.003, itl_s=0.001, stop_at=5, fail_rate=0.3, seed=3)) as srv:
        rec = run_single(srv, prompts, engine_kind="generic", rounds=2)
    lim = " ".join(rec["limitations"])
    assert "no exact-length control" in lim
    assert "request(s) failed" in lim and "failed" in rec["summary"]
    rows = [json.loads(r) for r in (store.run_dir(rec["evidence_id"]) / "requests.jsonl").read_text().splitlines()]
    assert any(not r["ok"] for r in rows) and all(r["n_tokens"] == 5 for r in rows if r["ok"])
    # A server that silently ignores ignore_eos is caught by the length check.
    with FakeServer(FakeConfig(ttft_s=0.003, itl_s=0.001, stop_at=5, honour_ignore_eos=False)) as srv:
        rec = run_single(srv, prompts, engine_kind="sglang", rounds=1, warmup=0)
    assert any("did not generate exactly max_tokens although ignore_eos/min_tokens was sent" in x
               for x in rec["limitations"])
    assert ev.validate(rec) == []


def test_by_default_no_timed_prompt_can_hit_the_prefix_cache(store, tmp_path):
    prompts = write_prompts(tmp_path / "dev.jsonl", buckets=("short",), per_bucket=2, max_tokens=4)
    with FakeServer(FakeConfig(ttft_s=0.002, itl_s=0.001)) as srv:
        rec = run_single(srv, prompts, rounds=3, warmup=1, run_dir=str(tmp_path / "run"), label="a")
        texts = [r["messages"][0]["content"] for r in srv.requests]
    assert len(set(texts)) == len(texts) == 1 + 3 * 2  # warm-up + 3 rounds x 2 prompts, all different
    assert texts[0].startswith("[ik-warm-0] ")
    assert texts[1:3] == ["short 0", "short 1"]  # round 0: untagged (cold: the warm-up was tagged)
    assert texts[3:] == ["[ik-r1] short 0", "[ik-r1] short 1", "[ik-r2] short 0", "[ik-r2] short 1"]
    assert rec["identity"]["workload"]["prefix_cache"] == "cold" and rec["data"]["prefix_cache"]["mode"] == "cold"
    assert not any("prefix cache" in x for x in rec["limitations"]) and "cold prefix cache" in rec["summary"]
    out = json.loads((tmp_path / "run/outputs/a/short/short-0.json").read_text())
    assert out["round"] == 0 and out["salt"] is None  # outputs come from the untagged round
    body = json.loads((tmp_path / "run/inputs/a/short/short-0.json").read_text())
    assert body["messages"] == [{"role": "user", "content": "short 0"}]


def test_no_cache_salt_measures_warm_cache_on_purpose_and_says_so(store, tmp_path):
    prompts = write_prompts(tmp_path / "dev.jsonl", buckets=("short",), per_bucket=2, max_tokens=4)
    with FakeServer(FakeConfig(ttft_s=0.002, itl_s=0.001)) as srv:
        rec = run_single(srv, prompts, rounds=2, warmup=1, cache_salt=False)
        texts = [r["messages"][0]["content"] for r in srv.requests]
        assert texts[1:] == ["short 0", "short 1"] * 2
        with pytest.raises(OpError, match="cache_salt"):  # cold and warm rounds never merge
            run_single(srv, prompts, rounds=1, warmup=0, extend=rec["evidence_id"])
    assert rec["identity"]["workload"]["prefix_cache"] == "warm"
    assert any("warm-cache TTFT" in x for x in rec["limitations"]) and "warm prefix cache" in rec["summary"]


def test_errors_never_carry_the_server_address(store, tmp_path):
    # 127.0.0.2 is loopback but not "local" to the client: it stands in for a remote host whose engine
    # died after the readiness check, so every request is refused with the address in the OS error.
    prompts = write_prompts(tmp_path / "dev.jsonl", buckets=("short",), per_bucket=2, max_tokens=4)
    with FakeServer(FakeConfig(host="127.0.0.2", refuse_after_models=True)) as srv:
        port = srv.port
        rec = run_single(srv, prompts, rounds=1, warmup=0, run_dir=str(tmp_path / "run"), label="remote")
    assert rec["data"]["base_url_kind"] == "remote" and "every request failed" in rec["coverage"]["missing"]
    texts = [json.dumps(rec)]
    texts += [p.read_text() for p in store.run_dir(rec["evidence_id"]).rglob("*") if p.is_file()]
    texts += [p.read_text() for p in (tmp_path / "run").rglob("*") if p.is_file()]
    rows = [json.loads(x) for x in (store.run_dir(rec["evidence_id"]) / "requests.jsonl").read_text().splitlines()]
    assert rows and all(not r["ok"] and "Connect call failed ('<server>', <port>)" in r["error"] for r in rows)
    for t in texts:
        assert "127.0.0.2" not in t and f"', {port}" not in t and f":{port}" not in t
    assert any("('<server>', <port>)" in x for x in rec["limitations"])
    # a host name that resolves to an address: both are scrubbed
    ep = cl.Endpoint.parse("http://gpu-box.example.net:8000")
    msg = "OSError: Multiple exceptions: Connect call failed ('203.0.113.7', 8000), ('2001:db8::5', 8000, 0, 0)"
    assert cl.scrub(msg, ep) == ("OSError: Multiple exceptions: Connect call failed ('<addr>', <port>), "
                                 "('<addr>', <port>, 0, 0)")
    assert cl.scrub("HTTP 502: bad gateway gpu-box.example.net:8000; max length 8000", ep) == \
        "HTTP 502: bad gateway <server>:<port>; max length 8000"


def test_base_url_is_stored_as_a_keyed_tag_not_a_plain_digest(store, tmp_path, endpoint_key, monkeypatch):
    from inferencekit.canonical import digest

    prompts = write_prompts(tmp_path / "dev.jsonl", buckets=("short",), per_bucket=1, max_tokens=4)
    with FakeServer(FakeConfig(ttft_s=0.002, itl_s=0.001)) as srv:
        a = run_single(srv, prompts, rounds=1, warmup=0)
        b = run_single(srv, prompts, rounds=1, warmup=0, extend=a["evidence_id"])  # same key: same tag
        monkeypatch.setenv("INFERENCEKIT_KEY_FILE", str(tmp_path / "other.key"))
        c = run_single(srv, prompts, rounds=1, warmup=0)
    stored = a["data"]["request"]["args"]["base_url"]
    assert stored == b["data"]["request"]["args"]["base_url"] != c["data"]["request"]["args"]["base_url"]
    assert stored != {"digest": digest(srv.base_url)} and digest(srv.base_url) not in json.dumps(a)
    assert endpoint_key.is_file() and len(endpoint_key.read_text().strip()) == 64
    assert (endpoint_key.stat().st_mode & 0o777) == 0o600


def test_inputs_hold_the_exact_bodies_the_gates_check(store, tmp_path):
    # Contract: inputs/<bucket>/<id>.json is the request body; with ignore_eos, the output must be exactly
    # max_tokens long. A server that ignores ignore_eos is visible from the run dir alone.
    prompts = write_prompts(tmp_path / "dev.jsonl", buckets=("short",), per_bucket=2, max_tokens=8)
    run = tmp_path / "run"
    with FakeServer(FakeConfig(ttft_s=0.002, itl_s=0.001, stop_at=5, honour_ignore_eos=False)) as srv:
        run_single(srv, prompts, rounds=1, warmup=0, run_dir=str(run), label="cheat")
        sent = srv.requests
    for i, req in enumerate(sent):
        assert json.loads((run / f"inputs/cheat/short/short-{i}.json").read_text()) == req  # exact body
        assert json.loads((run / f"inputs/short/short-{i}.json").read_text()) == req  # contract path
        out = json.loads((run / f"outputs/cheat/short/short-{i}.json").read_text())
        assert req["ignore_eos"] is True and req["max_tokens"] == 8 and len(out["token_ids"]) == 5


def test_gates_static_output_length_check_runs_on_a_bench_serve_run_dir(store, tmp_path):
    gates = pytest.importorskip("inferencekit.ops.gates")  # the gates module
    prompts = write_prompts(tmp_path / "dev.jsonl", buckets=("short",), per_bucket=2, max_tokens=8)
    run = tmp_path / "run"
    with FakeServer(FakeConfig(ttft_s=0.002, itl_s=0.001, stop_at=5, honour_ignore_eos=False)) as srv:
        run_single(srv, prompts, rounds=1, warmup=0, run_dir=str(run), label="cheat")
    with FakeServer(FakeConfig(ttft_s=0.002, itl_s=0.001, stop_at=5)) as srv:
        run_single(srv, prompts, rounds=1, warmup=0, run_dir=str(tmp_path / "run2"), label="honest")
    bad = gates.gate_output_length_static(run, "cheat", None, 0.1, 0.1)
    good = gates.gate_output_length_static(tmp_path / "run2", "honest", None, 0.1, 0.1)
    assert "FAIL" in json.dumps(bad) and "0/2" in json.dumps(bad)
    assert "PASS" in json.dumps(good) and "2/2" in json.dumps(good)


def test_outputs_index_is_a_snapshot_later_runs_cannot_change(store, tmp_path):
    from inferencekit.canonical import file_sha256

    prompts = write_prompts(tmp_path / "dev.jsonl", buckets=("short",), per_bucket=2, max_tokens=6)
    run = tmp_path / "run"
    with FakeServer(FakeConfig(ttft_s=0.002, itl_s=0.001)) as srv:
        first = run_single(srv, prompts, rounds=1, warmup=0, run_dir=str(run), label="x")
        run_single(srv, prompts, rounds=1, warmup=0, run_dir=str(run), label="x", max_tokens=4)  # overwrites run/
    art = next(a for a in first["artifacts"] if a["role"] == "outputs_index")
    assert art["path"] == "outputs/x/index.json"  # inside the record's directory, not the run dir
    rec_dir = store.run_dir(first["evidence_id"])
    assert file_sha256(rec_dir / art["path"])[0] == art["sha256"]
    snap = json.loads((rec_dir / art["path"]).read_text())
    assert set(snap["files"]) == set(snap["inputs"]) == {"short/short-0.json", "short/short-1.json"}
    for rel, sha in snap["files"].items():
        assert file_sha256(rec_dir / "outputs/x" / rel)[0] == sha
        assert json.loads((rec_dir / "outputs/x" / rel).read_text())["max_tokens"] == 6
    assert json.loads((run / "outputs/x/short/short-0.json").read_text())["max_tokens"] == 4


def test_extend_appends_rounds_for_interleaving(store, tmp_path):
    prompts = write_prompts(tmp_path / "dev.jsonl", buckets=("short",), per_bucket=2, max_tokens=6)
    with FakeServer(FakeConfig(ttft_s=0.004, itl_s=0.001)) as srv:
        first = run_single(srv, prompts, rounds=1, warmup=0)
        second = run_single(srv, prompts, rounds=1, warmup=0, extend=first["evidence_id"])
        assert [r["index"] for r in second["data"]["rounds"]] == [0, 1]
        assert second["data"]["extends"] == first["evidence_id"]
        assert metric(second, "ttft_ms")["n"] == 4
        with pytest.raises(OpError, match="these differ"):
            run_single(srv, prompts, rounds=1, warmup=0, max_tokens=4, extend=first["evidence_id"])
        with pytest.raises(OpError, match="engine_version"):  # another build of the same engine kind
            run_single(srv, prompts, rounds=1, warmup=0, engine_version="0.2", extend=first["evidence_id"])
    with FakeServer(FakeConfig(ttft_s=0.004, itl_s=0.001)) as other:  # same engine kind, another server
        with pytest.raises(OpError, match="base_url"):
            run_single(other, prompts, rounds=1, warmup=0, extend=first["evidence_id"])


def test_unreachable_server_and_bad_arguments(store, tmp_path):
    prompts = write_prompts(tmp_path / "dev.jsonl")
    with pytest.raises(OpError, match="not reachable"):
        registry.call("bench_serve", {"base_url": "http://127.0.0.1:9", "model": "m", "prompts": str(prompts)})
    with pytest.raises(OpError, match="exactly one of"):
        registry.call("bench_serve", {"base_url": "http://127.0.0.1:9", "model": "m", "prompts": str(prompts),
                                      "mode": "load"})
    with pytest.raises(OpError, match="label"):
        registry.call("bench_serve", {"base_url": "http://127.0.0.1:9", "model": "m", "prompts": str(prompts),
                                      "label": "../x"})


def test_cli_parses_bench_serve():
    from inferencekit import cli

    ns = cli.build_parser().parse_args(["bench", "serve", "--base-url", "http://h:8000", "--model", "m", "--prompts",
                                        "p.jsonl", "--mode", "load", "--concurrency", "1", "--concurrency", "8",
                                        "--slo-ttft-p95-ms", "500", "--gpu-price-usd-h", "0.6"])
    assert ns._op == "bench_serve" and ns.concurrency == [1, 8] and ns.slo_ttft_p95_ms == 500.0
