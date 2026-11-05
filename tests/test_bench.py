"""bench decode without a GPU: engine=cmd and a fake engine interpreter stand in for real engines."""

import json
import stat
import sys

import pytest

from inferencekit import registry
from inferencekit.engines import adapters
from inferencekit.registry import OpError

SECRET = "hf_typed_into_engine_args_123"
FAKE_RESULT = {"engine": {"name": "fake"}, "flags": {}, "rounds": [], "tokens": None}


@pytest.fixture
def engine_cmd(tmp_path):
    """engine=cmd: a command that prints one JSON result line per round."""
    script = tmp_path / "fake_engine.py"
    script.write_text("import json\nprint(json.dumps({'decode_tok_s': 100.0, 'ttft_ms': 20.0}))\n")
    return f"{sys.executable} {script}"


def test_engine_args_secrets_are_not_stored(store, engine_cmd):
    args = json.dumps({"hf_token": SECRET, "max_model_len": 4096})
    rec = registry.call("bench_decode", {"engine": "cmd", "cmd": engine_cmd, "engine_args": args, "rounds": 2})
    text = json.dumps(rec)
    assert SECRET not in text
    stored = rec["data"]["request"]["args"]["engine_args"]
    assert json.loads(stored) == {"hf_token": "<redacted>", "max_model_len": 4096}  # still readable
    assert "env_config_names" in rec["data"]
    on_disk = (store.run_dir(rec["evidence_id"]) / "evidence.json").read_text()
    assert SECRET not in on_disk


def test_worker_config_kept_as_artifact_is_redacted(tmp_path, monkeypatch):
    # A stand-in for the engine's python: it checks it was given the real token, then prints a result.
    fake = tmp_path / "fake-python"
    fake.write_text("#!/bin/sh\n"
                    f"grep -q {SECRET} \"$2\" || exit 9\n"
                    f"echo '{adapters.MARK}{json.dumps(FAKE_RESULT)}'\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    staging = tmp_path / "staging"
    staging.mkdir()
    cfg = {"model": "m", "engine_args": {"hf_token": SECRET, "max_model_len": 4096}}
    result = adapters.run_python_worker("vllm", str(fake), cfg, staging, timeout=60)
    kept = (staging / "worker_config.json").read_text()
    assert SECRET not in kept and json.loads(kept)["engine_args"]["max_model_len"] == 4096
    assert (staging / "worker_config.json", "worker_config") in result["files"]


def test_failed_run_leaves_no_staging_dir(store):
    with pytest.raises(OpError):
        registry.call("bench_decode", {"engine": "cmd", "cmd": f"{sys.executable} -c \"raise SystemExit(4)\"",
                                       "rounds": 1})
    staging_root = store.root / ".staging"
    assert not staging_root.exists() or not any(staging_root.iterdir())


def test_extend_refuses_an_edited_record(store, engine_cmd):
    rec = registry.call("bench_decode", {"engine": "cmd", "cmd": engine_cmd, "rounds": 2})
    path = store.run_dir(rec["evidence_id"]) / "evidence.json"
    edited = json.loads(path.read_text())
    edited["data"]["rounds"][0]["decode_tok_s"] = 150.0
    path.write_text(json.dumps(edited))
    with pytest.raises(OpError, match="does not match its content"):
        registry.call("bench_decode", {"engine": "cmd", "cmd": engine_cmd, "rounds": 2,
                                       "extend": rec["evidence_id"]})
