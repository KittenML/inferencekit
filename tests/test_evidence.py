import json

import pytest

from inferencekit import evidence as ev
from inferencekit.canonical import canonical_json, digest
from inferencekit.registry import Arg, Call, Operation

from conftest import fake_call


@pytest.mark.parametrize("value,text", [
    (1.0, "1"), (100.0, "100"), (0.5, "0.5"), (1.5, "1.5"), (123.456, "123.456"), (-0.0025, "-0.0025"),
    (1e-6, "0.000001"), (1e-7, "1e-7"), (1.23e-18, "1.23e-18"), (1e20, "100000000000000000000"),
    (1e21, "1e+21"), (5e-324, "5e-324"), (1.7976931348623157e308, "1.7976931348623157e+308"), (-0.0, "0"),
])
def test_numbers_follow_ecmascript_formatting(value, text):
    assert canonical_json(value) == text


def test_canonical_json_sorts_keys_and_drops_whitespace():
    assert canonical_json({"b": [1, 2.5, None], "a": {"y": True, "x": "é"}}) == \
        '{"a":{"x":"é","y":true},"b":[1,2.5,null]}'
    with pytest.raises(ValueError):
        canonical_json(float("nan"))


def _record(summary="s", value=1.0):
    rec = ev.build(fake_call(), kind="benchmark", claim_class="observed", summary=summary,
                   metrics=[ev.metric("decode_tok_s", value, "tok/s", "median", n=3, ci95=(0.9, 1.1))])
    rec["provenance"]["ended"] = "2026-01-01T00:00:01.000Z"  # fixed, so two builds are identical
    return rec


def test_evidence_id_is_stable_and_content_addressed():
    a, b = ev.finalize(_record()), ev.finalize(_record())
    assert a["evidence_id"] == b["evidence_id"]
    assert a["evidence_id"].startswith("ev_") and len(a["evidence_id"]) == 3 + 64
    assert ev.finalize(_record(value=1.01))["evidence_id"] != a["evidence_id"]
    # the id never covers itself: re-finalizing a finalized record gives the same id
    assert ev.finalize(a)["evidence_id"] == a["evidence_id"]


def test_saved_record_validates_and_detects_tampering(store):
    rec = store.save(_record())
    assert ev.validate(rec) == []
    loaded = store.load(rec["evidence_id"][:12])  # unique prefix lookup
    assert ev.verify(loaded)["id_ok"]
    loaded["metrics"][0]["value"] = 2.0
    assert not ev.verify(loaded)["id_ok"]
    on_disk = json.loads((store.run_dir(rec["evidence_id"]) / "evidence.json").read_text())
    assert on_disk == rec


def test_sensitive_arguments_are_stored_as_digests_only():
    op = Operation("t", ("t",), "test", (Arg("cmd", "string", "c", sensitive=True), Arg("n", "integer", "n")),
                   lambda call, **kw: {})
    call = Call(op, {"cmd": "python run.py --api-key abc123", "n": 3})
    req = call.request()
    assert req["args"]["n"] == 3
    assert req["args"]["cmd"] == {"digest": digest("python run.py --api-key abc123")}
    assert "abc123" not in json.dumps(req)
    assert call.argv_digest().startswith("sha256:")


def test_secret_environment_values_are_redacted(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_not_a_real_secret_value")
    rec = ev.finalize(_record(summary="leaked hf_not_a_real_secret_value here"))
    assert "hf_not_a_real_secret_value" not in json.dumps(rec)
    assert "<redacted:HF_TOKEN>" in rec["summary"]


def test_non_finite_numbers_become_null():
    rec = ev.finalize(_record(value=float("inf")))
    assert rec["metrics"][0]["value"] is None


def test_secret_looking_keys_are_redacted_in_dicts_and_json_text():
    text = '{"hf_token": "hf_typed_inline_123", "max_num_batched_tokens": 8192, "tokenizer": "Qwen/Qwen3-8B"}'
    assert ev.redact_text(text) == ('{"hf_token": "<redacted>", "max_num_batched_tokens": 8192, '
                                    '"tokenizer": "Qwen/Qwen3-8B"}')
    assert ev.redact_text("{'api_key': 'sk-123456789'}") == "{'api_key': '<redacted>'}"  # a Python dict repr
    assert ev.redact({"engine_args": {"hf_token": "hf_x", "max_model_len": 4096, "tokens": [1, 2]}}) == \
        {"engine_args": {"hf_token": "<redacted>", "max_model_len": 4096, "tokens": [1, 2]}}
    assert ev.looks_secret_key("password") and not ev.looks_secret_key("max_num_batched_tokens")


def test_integrity_problem_flags_an_edited_record(store):
    rec = store.save(_record())
    assert ev.integrity_problem(rec) is None
    edited = json.loads(json.dumps(rec))
    edited["metrics"][0]["value"] = 2.0
    assert "does not match its content" in ev.integrity_problem(edited)


def test_staging_dir_is_removed_when_the_operation_fails(store):
    with pytest.raises(RuntimeError), store.staging_dir() as staging:
        (staging / "stdout.txt").write_text("raw output, maybe with a secret")
        raise RuntimeError("the profiler failed")
    assert not staging.exists()
    with store.staging_dir() as staging:  # success: save() moves the files into the run dir
        (staging / "log.txt").write_text("ok")
        rec = store.save(_record(), staging)
    assert not staging.exists() and (store.run_dir(rec["evidence_id"]) / "log.txt").is_file()
