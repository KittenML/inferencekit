"""compare over bench serve records: it must refuse arms that did not send the same requests."""

import json

import pytest

from inferencekit import registry

from fakes.fake_openai_server import FakeConfig, FakeServer


@pytest.fixture(autouse=True)
def endpoint_key(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCEKIT_KEY_FILE", str(tmp_path / "endpoint.key"))


def _prompts(path):
    rows = [{"id": f"{b}-{i}", "bucket": b, "messages": [{"role": "user", "content": f"{b} prompt {i}"}],
             "max_tokens": 8, "source": "test", "kind": "chat"}
            for b in ("very_short", "short", "long") for i in range(3)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _serve(srv, prompts, label, **kw):
    args = {"base_url": srv.base_url, "model": "fake-model", "prompts": str(prompts), "engine_kind": "vllm",
            "rounds": 2, "warmup": 0, "resamples": 200, "label": label}
    args.update(kw)
    return registry.call("bench_serve", args)


def _compare(a, b, **kw):
    return registry.call("compare", {"baseline": a["evidence_id"], "candidate": b["evidence_id"],
                                     "resamples": 200, **kw})


def test_compare_refuses_different_bucket_subsets_and_template_settings(store, tmp_path):
    prompts = _prompts(tmp_path / "p.jsonl")
    with FakeServer(FakeConfig(ttft_s=0.005, itl_s=0.001)) as srv:
        a = _serve(srv, prompts, "a", buckets=["very_short", "short"], limit_per_bucket=3,
                   chat_template_kwargs='{"enable_thinking": false}')
        b = _serve(srv, prompts, "b", buckets=["very_short", "long"], limit_per_bucket=1,
                   chat_template_kwargs='{"enable_thinking": true}')
        same = _serve(srv, prompts, "c", buckets=["short", "very_short"], limit_per_bucket=3,
                      chat_template_kwargs='{"enable_thinking":false}')
    # Same prompt-set file, so the file sha alone cannot tell them apart.
    assert a["identity"]["workload"]["prompt_set"] == b["identity"]["workload"]["prompt_set"]
    assert a["identity"]["workload"]["request_set"]["sha256"] != b["identity"]["workload"]["request_set"]["sha256"]
    res = _compare(a, b)
    assert res["data"]["verdict"] == "inconclusive"
    reason = res["data"]["decision"]["reason"]
    assert "request_set" in reason, reason
    # Records without request_set (older builds): the stored bench serve arguments are compared instead.
    from inferencekit.ops.compare import serve_workload_reasons
    strip = lambda r: {**r, "identity": {**r["identity"], "workload": {k: v for k, v in r["identity"]["workload"].items()
                                                                        if k != "request_set"}}}
    reasons = " ".join(serve_workload_reasons(strip(a), strip(b)))
    assert "--buckets" in reasons and "--limit-per-bucket" in reasons and "--chat-template-kwargs" in reasons, reasons
    assert serve_workload_reasons(strip(a), strip(same)) == []

    # The same requests written differently (bucket order, JSON spelling) compare as the same workload.
    assert a["identity"]["workload"]["request_set"] == same["identity"]["workload"]["request_set"]
    res = _compare(a, same)
    assert "workloads differ" not in (res["data"]["decision"].get("reason") or "")


def test_compare_is_inconclusive_when_an_arm_had_failed_requests(store, tmp_path):
    prompts = _prompts(tmp_path / "p.jsonl")
    with FakeServer(FakeConfig(ttft_s=0.005, itl_s=0.001)) as ok_srv:
        a = _serve(ok_srv, prompts, "a")
    with FakeServer(FakeConfig(ttft_s=0.005, itl_s=0.001, fail_rate=0.3, seed=1)) as bad_srv:
        b = _serve(bad_srv, prompts, "b")
    assert sum(r["n_failed"] for r in b["data"]["rounds"]) > 0
    res = _compare(a, b)
    assert res["data"]["verdict"] == "inconclusive"
    assert "failed request" in res["data"]["decision"]["reason"]


def test_compare_names_a_metric_bench_serve_does_not_measure(store, tmp_path):
    prompts = _prompts(tmp_path / "p.jsonl")
    with FakeServer(FakeConfig(ttft_s=0.005, itl_s=0.001)) as srv:
        a = _serve(srv, prompts, "a", buckets=["short"])
        b = _serve(srv, prompts, "b", buckets=["short"])
    res = _compare(a, b, metric="prefill_tok_s")
    reason = res["data"]["decision"]["reason"]
    assert "not measured by bench serve" in reason and "more rounds will not help" in reason, reason
    assert not res["data"].get("rounds_for_power")


def test_extend_accepts_the_same_template_settings_written_differently(store, tmp_path):
    prompts = _prompts(tmp_path / "p.jsonl")
    with FakeServer(FakeConfig(ttft_s=0.005, itl_s=0.001)) as srv:
        a = _serve(srv, prompts, "a", rounds=1, buckets=["short"], chat_template_kwargs='{"enable_thinking": false}')
        b = _serve(srv, prompts, "a", rounds=1, buckets=["short"], chat_template_kwargs='{"enable_thinking":false}',
                   extend=a["evidence_id"])
    assert len(b["data"]["rounds"]) == 2
