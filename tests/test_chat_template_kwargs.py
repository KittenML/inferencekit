"""chat_template_kwargs reach every request the same way (bench serve, bench rounds' command line, gates),
and records made with a run dir are copied into <run-dir>/evidence/."""

import json

import pytest

from inferencekit import evidence as ev
from inferencekit import registry
from inferencekit.ops.bench_serve import chat_template_kwargs_for
from inferencekit.registry import OpError
from inferencekit.serve import rounds as rr
from fakes.fake_openai_server import FakeConfig, FakeServer


@pytest.fixture(autouse=True)
def endpoint_key(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCEKIT_KEY_FILE", str(tmp_path / "endpoint.key"))


def _prompts(path):
    rows = [{"id": "short-0", "bucket": "short", "messages": [{"role": "user", "content": "a b c"}], "max_tokens": 4},
            {"id": "short-1", "bucket": "short", "messages": [{"role": "user", "content": "d e f"}], "max_tokens": 4,
             "chat_template_kwargs": {"enable_thinking": True, "extra": 1}}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_merge_order_flag_then_record():
    assert chat_template_kwargs_for({"chat_template_kwargs": {"a": 2}}, {"a": 0, "b": 0}, {"a": 1}) == {"a": 2, "b": 0}
    assert chat_template_kwargs_for({}, None, {"x": False}) == {"x": False}
    assert chat_template_kwargs_for({}, None, None) == {}


def test_bench_serve_sends_flag_and_record_kwargs_and_copies_the_record(store, tmp_path):
    prompts = _prompts(tmp_path / "p.jsonl")
    run = tmp_path / "run"
    run.mkdir()
    with FakeServer(FakeConfig(ttft_s=0.002, itl_s=0.001)) as srv:
        rec = registry.call("bench_serve", {"base_url": srv.base_url, "model": "fake-model", "prompts": str(prompts),
                                            "engine_kind": "vllm", "rounds": 1, "warmup": 1, "resamples": 50,
                                            "run_dir": str(run), "label": "x",
                                            "chat_template_kwargs": '{"enable_thinking": false}'})
        sent = [r for r in srv.requests if "messages" in r]
    assert sent and all("chat_template_kwargs" in r for r in sent)
    by_text = {r["messages"][0]["content"].split("] ")[-1]: r["chat_template_kwargs"] for r in sent}
    assert by_text["a b c"] == {"enable_thinking": False}
    assert by_text["d e f"] == {"enable_thinking": True, "extra": 1}  # the record's own field wins per key
    assert (run / "evidence" / f"{rec['evidence_id']}.json").is_file()
    assert json.loads((run / "evidence" / f"{rec['evidence_id']}.json").read_text()) == rec


def test_bench_rounds_passes_manifest_kwargs_to_every_cell():
    value, src = rr.template_kwargs(None, {"workload": {"chat_template_kwargs": {"enable_thinking": False}}})
    assert value == {"enable_thinking": False} and "manifest" in src
    value, src = rr.template_kwargs('{"enable_thinking": true}', {"workload": {"chat_template_kwargs": {"x": 1}}})
    assert value == {"enable_thinking": True} and src == "--chat-template-kwargs"
    with pytest.raises(OpError):
        rr.template_kwargs("[1]", {})
    ctx = {"base_url": "http://127.0.0.1:1", "model": "m", "prompts": "p.jsonl", "mode": "single", "label": "a",
           "run_dir": "r", "engine_kind": "vllm", "chat_template_kwargs": {"enable_thinking": False}}
    argv = rr.serve_argv(None, ctx, "--max-tokens 8")
    i = argv.index("--chat-template-kwargs")
    assert json.loads(argv[i + 1]) == {"enable_thinking": False}
    # Also in --serve-args: sent once, the resolved value (run_rounds refuses a contradicting one).
    argv = rr.serve_argv(None, ctx, "--chat-template-kwargs '{\"enable_thinking\": false}' --max-tokens 8")
    assert argv.count("--chat-template-kwargs") == 1 and argv[-2:] == ["--max-tokens", "8"]
    assert json.loads(argv[argv.index("--chat-template-kwargs") + 1]) == {"enable_thinking": False}


def test_bench_rounds_reconciles_serve_args_kwargs_and_the_intake_pin(tmp_path):
    value, src = rr._serve_args_template_kwargs("--chat-template-kwargs '{\"enable_thinking\": false}'", {}, "none")
    assert value == {"enable_thinking": False} and "--serve-args" in src
    with pytest.raises(OpError, match="give one value"):
        rr._serve_args_template_kwargs("--chat-template-kwargs='{\"enable_thinking\":true}'",
                                       {"enable_thinking": False}, "manifest workload.chat_template_kwargs")
    value, _ = rr._serve_args_template_kwargs("--chat-template-kwargs '{\"enable_thinking\":false}'",
                                              {"enable_thinking": False}, "manifest")
    assert value == {"enable_thinking": False}
    # The intake pin (intake.normalized.json in the run dir) is added, and a contradiction is refused.
    (tmp_path / "intake.normalized.json").write_text(json.dumps(
        {"intake": {"workload": {"chat_template_kwargs": {"enable_thinking": False}}}}))
    value, src = rr._pinned_template_kwargs({}, "none", [tmp_path])
    assert value == {"enable_thinking": False} and "intake pin" in src
    with pytest.raises(OpError, match="pinned"):
        rr._pinned_template_kwargs({"enable_thinking": True}, "--chat-template-kwargs", [tmp_path])


def test_manifest_validate_enforces_the_intake_pin(tmp_path):
    from inferencekit.ops.manifest import check_manifest

    (tmp_path / "intake.normalized.json").write_text(json.dumps(
        {"intake": {"workload": {"chat_template_kwargs": {"enable_thinking": False}}}}))
    errs = check_manifest({"workload": {}}, tmp_path)["errors"]
    assert any("intake's pin" in e and "missing" in e for e in errs), errs
    errs = check_manifest({"workload": {"chat_template_kwargs": {"enable_thinking": True}}}, tmp_path)["errors"]
    assert any("intake's pin" in e for e in errs), errs
    errs = check_manifest({"workload": {"chat_template_kwargs": {"enable_thinking": False}}}, tmp_path)["errors"]
    assert not any("intake's pin" in e for e in errs), errs


def test_default_runs_dir_env(monkeypatch, tmp_path):
    ev.set_default_root(None)
    monkeypatch.delenv("INFERENCEKIT_RUNS_DIR", raising=False)
    monkeypatch.setenv("INFERKIT_RUNS_DIR", str(tmp_path / "old"))
    assert ev.default_root() == tmp_path / "old"  # deprecated name still read
    monkeypatch.setenv("INFERENCEKIT_RUNS_DIR", str(tmp_path / "new"))
    assert ev.default_root() == tmp_path / "new"  # the new name wins


def test_gsm8k_lm_eval_refuses_template_kwargs_given_or_pinned(tmp_path):
    base = {"base_url": "http://127.0.0.1:9", "model": "m", "run_dir": str(tmp_path), "backend": "lm-eval"}
    with pytest.raises(OpError, match="lm-eval cannot send chat_template_kwargs"):
        registry.call("quality_gsm8k", {**base, "chat_template_kwargs": '{"enable_thinking": false}'})
    (tmp_path / "intake.normalized.json").write_text(json.dumps(
        {"intake": {"workload": {"chat_template_kwargs": {"enable_thinking": False}}}}))
    with pytest.raises(OpError, match="lm-eval cannot send chat_template_kwargs"):
        registry.call("quality_gsm8k", base)
    with pytest.raises(OpError, match="pinned"):
        registry.call("quality_gsm8k", {**base, "backend": "builtin", "chat_template_kwargs": '{"enable_thinking": true}'})
