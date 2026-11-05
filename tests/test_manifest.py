"""manifest validate: schema + semantic checks on the run manifest a re-run replays."""

import copy
import json

import jsonschema
import pytest

from inferencekit import cli, registry
from inferencekit.evidence import set_default_root
from inferencekit.ops import manifest as mf
from inferencekit.registry import OpError

REV = "220b46e3b2180893580a4454f21f22d3ebb187d3"
WHEEL = "a" * 64


def good_manifest() -> dict:
    return {
        "manifest_version": 1,
        "model": {"id": "Qwen/Qwen3-8B", "revision": REV, "precision": "bf16", "tokenizer": "Qwen/Qwen3-8B"},
        "hardware": {"gpu": "RTX 5090", "count": 1},
        "engines": [
            {"name": "vllm", "label": "vllm-tuned", "version": "0.31.0", "install": ["uv pip install vllm==0.31.0"],
             "launch": ["vllm", "serve", "Qwen/Qwen3-8B", "--revision", REV, "--port", "8000"],
             "env": {"VLLM_USE_V1": "1"}, "port": 8000, "ready_check": "/v1/models",
             "resolved_config": {"max_num_seqs": 64}, "role": "tuned"},
            {"name": "sglang", "label": "sglang-recipe", "version": "0.5.21",
             "install": ["uv pip install 'sglang[all]==0.5.21'"],
             "launch": ["python", "-m", "sglang.launch_server", "--model-path", "Qwen/Qwen3-8B", "--revision", REV,
                        "--port", "8001"],
             "port": 8001, "ready_check": "/health", "resolved_config": {"mem_fraction_static": 0.9},
             "role": "baseline-recipe"},
        ],
        "workload": {"modes": ["single"], "buckets": ["very_short", "short", "medium", "long"], "max_tokens": 256,
                     "slo": {}},
        "harness": {"inferencekit_version": "0.1.0", "wheel_sha256": WHEEL, "dev_prompt_set_sha256": "b" * 64},
        "agent_run": {"run_id": "run-001", "started": "2026-10-13T10:00:00Z", "ended": "2026-10-13T12:00:00Z"},
    }


def test_schema_is_valid_json_schema():
    jsonschema.Draft202012Validator.check_schema(mf.load_schema())


def test_good_manifest_is_valid_without_warnings():
    res = mf.check_manifest(good_manifest())
    assert res == {"valid": True, "errors": [], "warnings": []}


def test_full_example_manifest_passes_the_schema():
    # A full example manifest, with placeholders filled.
    m = good_manifest()
    m["engines"] = [m["engines"][0]]
    m["harness"]["wheel_sha256"] = ""
    res = mf.check_manifest(m)
    assert res["valid"], res["errors"]
    assert any("wheel_sha256 is empty" in w for w in res["warnings"])


@pytest.mark.parametrize("mutate, needle", [
    (lambda m: m["engines"][1].update(label="vllm-tuned"), "labels must be unique"),
    (lambda m: m["engines"][1].update(port=8000), "ports must be distinct"),
    (lambda m: m["engines"][0].update(install=[]), "install"),
    (lambda m: m["engines"][0].update(install=["   "]), "install"),
    (lambda m: m["engines"][0].update(launch=[]), "launch"),
    (lambda m: m["engines"][0].update(launch=[""]), "launch"),
    (lambda m: m["model"].update(revision="main"), "full 40-character"),
    (lambda m: m["model"].update(revision=REV[:12]), "full 40-character"),
    (lambda m: m["model"].update(revision=REV.upper()), "full 40-character"),
    (lambda m: m["engines"][0]["env"].update(HF_TOKEN="hf_secretvalue123"), "looks like a secret"),
    (lambda m: m["engines"][0].update(ready_check="v1/models"), "must be an HTTP path"),
    (lambda m: m["harness"].update(wheel_sha256="deadbeef"), "64-character"),
    (lambda m: m.update(manifest_version=2), "manifest_version"),
    (lambda m: m["workload"].update(modes=["burst"]), "modes"),
    (lambda m: m["engines"][0].update(port=70000), "port"),
    (lambda m: m.update(extra_field=1), "extra_field"),
    (lambda m: m.pop("harness"), "harness"),
    (lambda m: m["engines"][0].update(host="10.1.2.3"), "not a loopback address"),
    (lambda m: m["engines"][0].update(cwd="/opt/engine"), "cwd"),
    (lambda m: m["engines"][0].update(cwd="../outside"), "cwd"),
    (lambda m: m["model"].update(local_dir="/data/models/x"), "local_dir"),
    (lambda m: m["engines"][0]["env"].update(N=3), "env/N"),
])
def test_bad_manifests_are_rejected(mutate, needle):
    m = copy.deepcopy(good_manifest())
    mutate(m)
    res = mf.check_manifest(m)
    assert not res["valid"]
    assert any(needle in e for e in res["errors"]), res["errors"]


def test_rounds_lane_engine_fields_are_accepted():
    """bench rounds documents env values of null (inherit from the caller, e.g. a token) and optional host and
    cwd; a manifest written that way must pass validation."""
    m = good_manifest()
    m["engines"][0]["env"]["HF_TOKEN"] = None
    m["engines"][0]["host"] = "127.0.0.1"
    m["engines"][1]["host"] = "localhost"
    m["engines"][1]["cwd"] = "engine-src/build"
    res = mf.check_manifest(m)
    assert res == {"valid": True, "errors": [], "warnings": []}


def test_clip_manifest_is_accepted_and_needs_its_clip_set():
    """bench rounds --mode clip (video): clip engines carry endpoint_kind (and workflow for ComfyUI); a clip
    workload names its clip set instead of the text buckets and max_tokens."""
    m = good_manifest()
    m["engines"][0].update(endpoint_kind="clip_http", ready_check="/health")
    m["engines"][1].update(endpoint_kind="comfyui", workflow="workflows/ltx-t2v.json", ready_check="/system_stats")
    m["workload"] = {"modes": ["clip"], "clips": "clips/dev.jsonl"}
    assert mf.check_manifest(m) == {"valid": True, "errors": [], "warnings": []}
    for mutate, needle in ((lambda w: w.pop("clips"), "clips"),
                           (lambda w: w.update(modes=["single"]), "buckets")):
        bad = copy.deepcopy(m)
        mutate(bad["workload"])
        res = mf.check_manifest(bad)
        assert not res["valid"] and any(needle in e for e in res["errors"]), res["errors"]
    bad = copy.deepcopy(m)
    bad["engines"][0]["endpoint_kind"] = "grpc"
    assert not mf.check_manifest(bad)["valid"]


def test_local_model_dir_counts_as_pinned():
    m = good_manifest()
    m["model"]["local_dir"] = "models/qwen3-8b"
    m["engines"][0]["launch"] = ["vllm", "serve", "models/qwen3-8b", "--port", "8000"]
    m["engines"][1]["launch"] = ["python", "-m", "sglang.launch_server", "--model-path", "../models/qwen3-8b"]
    assert mf.check_manifest(m) == {"valid": True, "errors": [], "warnings": []}
    del m["model"]["local_dir"]
    res = mf.check_manifest(m)
    assert res["valid"]
    assert any("model.local_dir is not set" in w for w in res["warnings"])
    assert not any("does not mention model/revision" in w for w in res["warnings"])


def test_semantic_checks_survive_a_schema_invalid_manifest():
    res = mf.check_manifest({"engines": [{"label": "a", "port": 1}, {"label": "a", "port": 1}, "junk"]})
    assert not res["valid"]
    assert any("labels must be unique" in e for e in res["errors"])
    assert any("ports must be distinct" in e for e in res["errors"])
    assert mf.check_manifest([1, 2])["valid"] is False


def test_warnings_for_unpinned_launch_and_load_without_slo():
    m = good_manifest()
    m["engines"][0]["launch"] = ["vllm", "serve", "Qwen/Qwen3-8B"]
    m["workload"]["modes"] = ["single", "load"]
    res = mf.check_manifest(m)
    assert res["valid"]
    assert any("does not mention model/revision" in w for w in res["warnings"])
    assert any("N* cannot be computed" in w for w in res["warnings"])


def test_operation_returns_sha_and_strict_raises(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(good_manifest()))
    out = registry.call("validate_manifest", {"file": str(p)})
    assert out["valid"] and len(out["sha256"]) == 64 and out["engines"] == ["vllm-tuned", "sglang-recipe"]
    bad = good_manifest()
    bad["model"]["revision"] = "main"
    p.write_text(json.dumps(bad))
    out = registry.call("validate_manifest", {"file": str(p), "strict": False})
    assert not out["valid"]
    with pytest.raises(OpError, match="invalid run manifest"):  # strict is the default
        registry.call("validate_manifest", {"file": str(p)})
    with pytest.raises(OpError, match="invalid run manifest"):
        registry.call("validate_manifest", {"file": str(p), "strict": True})
    p.write_text("{not json")
    with pytest.raises(OpError, match="not valid JSON"):
        registry.call("validate_manifest", {"file": str(p)})
    with pytest.raises(OpError, match="not found"):
        registry.call("validate_manifest", {"file": str(tmp_path / "missing.json")})


def test_cli_manifest_validate(tmp_path, capsys):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(good_manifest()))
    try:
        assert cli.main(["--runs-dir", str(tmp_path / "runs"), "manifest", "validate", str(p)]) == 0
        assert json.loads(capsys.readouterr().out)["valid"] is True
        bad = good_manifest()
        bad["engines"][1]["port"] = 8000
        p.write_text(json.dumps(bad))
        assert cli.main(["--runs-dir", str(tmp_path / "runs"), "manifest", "validate", str(p), "--strict"]) == 1
        assert "ports must be distinct" in capsys.readouterr().err
        # `manifest validate m.json && ...` must stop on an invalid manifest without any flag.
        assert cli.main(["--runs-dir", str(tmp_path / "runs"), "manifest", "validate", str(p)]) == 1
        assert "ports must be distinct" in capsys.readouterr().err
        assert cli.main(["--runs-dir", str(tmp_path / "runs"), "manifest", "validate", str(p), "--no-strict"]) == 0
        assert json.loads(capsys.readouterr().out)["valid"] is False
    finally:
        set_default_root(None)


# --------------------------------------------------------------------------- workload.extra_body (sampling)

SAMPLING = {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "seed": 1234}


def test_sampling_extra_body_is_valid():
    m = good_manifest()
    # other engine fields pass too, including ones whose names contain "token"
    m["workload"]["extra_body"] = dict(SAMPLING, min_p=0.0, repetition_penalty=1.0, skip_special_tokens=False)
    assert mf.check_manifest(m) == {"valid": True, "errors": [], "warnings": []}


@pytest.mark.parametrize("body,needle", [
    ({"temperature": -1}, "temperature"),
    ({"top_p": 0}, "top_p"),
    ({"top_p": 1.5}, "top_p"),
    ({"top_k": 2.5}, "top_k"),
    ({"seed": "1234"}, "seed"),
    ([{"temperature": 1.0}], "extra_body"),
    ({"max_tokens": 4}, "the harness sets max_tokens itself"),
    ({"stream": False}, "the harness sets stream itself"),
    ({"messages": []}, "the harness sets messages itself"),
    ({"chat_template_kwargs": {"enable_thinking": False}}, "workload.chat_template_kwargs"),
    ({"api_key": "x"}, "looks like a secret"),
    ({"hf_token": "x"}, "looks like a secret"),
    # fields that change what is timed or how tokens are counted
    ({"ignore_eos": False}, "exact-length decoding"),
    ({"min_tokens": 1}, "exact-length decoding"),
    ({"stop": ["\n"]}, "ends generations early"),
    ({"stop_token_ids": [2]}, "ends generations early"),
    ({"best_of": 2}, "extra sequences"),
    ({"logit_bias": {"2": -100}}, "steers which tokens"),
    ({"return_token_ids": False}, "count streamed tokens"),
    ({"return_tokens_as_token_ids": False}, "count streamed tokens"),
    ({"logprobs": False}, "counts tokens through them"),
    ({"top_logprobs": 5}, "through logprobs"),
    ({"echo": True}, "returns the prompt"),
    ({"suffix": "x"}, "changes the prompt"),
])
def test_bad_extra_bodies_are_rejected(body, needle):
    m = good_manifest()
    m["workload"]["extra_body"] = body
    res = mf.check_manifest(m)
    assert not res["valid"] and any(needle in e for e in res["errors"]), res["errors"]


def test_sampling_settings_without_a_temperature_are_flagged():
    m = good_manifest()
    m["workload"]["extra_body"] = {"top_k": 64, "seed": 1}
    res = mf.check_manifest(m)
    assert res["valid"] and any("top_k, seed but no temperature above 0" in w for w in res["warnings"])
    m["workload"]["extra_body"] = {"top_k": 64, "temperature": 0}
    assert any("greedy" in w for w in mf.check_manifest(m)["warnings"])
    m["workload"]["extra_body"] = SAMPLING
    assert mf.check_manifest(m)["warnings"] == []


def test_extra_body_problems_name_where_the_body_came_from():
    errors, warnings = mf.extra_body_problems({"model": "x", "min_tokens": 1, "top_p": 0.9}, "--extra-body")
    assert errors == ["--extra-body/model: the harness sets model itself (from the manifest); an extra body must not "
                      "set it",
                      "--extra-body/min_tokens: bench serve sets min_tokens for exact-length decoding (bench serve "
                      "--no-exact-length turns it off); an extra body must not set it"]
    assert warnings and warnings[0].startswith("--extra-body: sets top_p but no temperature")
    assert mf.extra_body_problems("[]", "--extra-body") == (["--extra-body: must be a JSON object"], [])


def test_extra_body_types_are_checked_outside_manifest_validate_too():
    """bench rounds --extra-body never passes the manifest schema: extra_body_problems applies its rules."""
    errors, _ = mf.extra_body_problems({"top_p": 5, "temperature": -1, "seed": "1"}, "--extra-body")
    assert [e.split(":")[0] for e in errors] == ["--extra-body/seed", "--extra-body/temperature", "--extra-body/top_p"]
    # manifest validate reports each type error once (from the schema)
    m = good_manifest()
    m["workload"]["extra_body"] = {"top_p": 5}
    assert len([e for e in mf.check_manifest(m)["errors"] if "top_p" in e]) == 1
