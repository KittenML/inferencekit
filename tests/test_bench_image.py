"""Image benchmark evidence and anti-cheat regressions; no GPU required."""
import json

import pytest

from inferencekit import evidence as ev, registry
from inferencekit.ops import bench_image as bi
from inferencekit.registry import OpError
from test_image_client import image_server


def fixtures(tmp_path, rows=None):
    rows = rows if rows is not None else [{"id": "one", "prompt": "a fox", "width": 4, "height": 3,
                                          "steps": 5, "guidance_scale": 3.0,
                                          "settings": {"scheduler": "euler", "negative_prompt": ""}}]
    path = tmp_path / "images.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def bench(url, path, **changes):
    return registry.call("bench_image", {"base_url": url, "prompts": str(path), "model": "model",
                                       "model_revision": "a" * 40, "dtype": "bf16", "quant": "none",
                                       "engine_version": "engine@pin", "gpu_name": "Test Radeon", "rounds": 2,
                                       "resamples": 100, **changes})


def test_image_evidence_rounds_and_saved_outputs(tmp_path, store, monkeypatch):
    monkeypatch.setenv("INFERENCEKIT_KEY_FILE", str(tmp_path / "endpoint.key"))
    path = fixtures(tmp_path)
    with image_server() as (url, requests):
        record = bench(url, path, run_dir=str(tmp_path / "run"))
    assert not ev.validate(record) and ev.integrity_problem(record) is None
    assert record["kind"] == "image_benchmark" and record["identity"]["workload"]["task"] == "t2i"
    assert record["data"]["quality"]["verdict"] == "UNPROVEN"
    assert not record["coverage"]["complete"] and record["data"]["claim_validity"]["valid"]
    rounds = record["data"]["rounds"]
    assert [r["index"] for r in rounds] == [0, 1]
    assert rounds[0]["seed"] != rounds[1]["seed"]
    assert all(r["image_ms"] > 0 and r["n_failed"] == 0 for r in rounds)
    assert rounds[0]["images_per_hour"] == pytest.approx(3_600_000 / rounds[0]["image_ms"])
    assert len(requests) == 3 and "warm-up" in requests[0][1]["prompt"]
    assert len(list((tmp_path / "run" / "outputs").rglob("*.png"))) == 2
    serialized = json.dumps(record)
    assert url not in serialized and "127.0.0.1" not in serialized
    assert len(record["identity"]["workload"]["request_set"]["sha256"]) == 64


@pytest.mark.parametrize("mode", ["cached", "repeat", "ignore_steps", "missing_effective", "wrong_size"])
def test_false_claims_invalid(tmp_path, store, mode):
    with image_server(mode) as (url, _):
        record = bench(url, fixtures(tmp_path))
    assert not record["data"]["claim_validity"]["valid"]
    assert record["data"]["claim_validity"]["violations"]
    if mode == "wrong_size":
        assert all(r["image_ms"] is None and r["n_failed"] == 1 for r in record["data"]["rounds"])


def test_warmup_covers_each_shape_and_is_excluded(tmp_path, store):
    rows = [{"id": "one", "prompt": "fox", "width": 4, "height": 3, "steps": 5, "guidance_scale": 3.0},
            {"id": "two", "prompt": "cat", "width": 5, "height": 3, "steps": 5, "guidance_scale": 3.0}]
    with image_server() as (url, requests):
        record = bench(url, fixtures(tmp_path, rows), rounds=1, warmup=2)
    assert len(requests) == 6
    assert len(record["data"]["warmup"]["requests"]) == 4
    assert record["data"]["rounds"][0]["n_images"] == 2


def test_extend_requires_same_configuration_and_artifact_integrity(tmp_path, store, monkeypatch):
    monkeypatch.setenv("INFERENCEKIT_KEY_FILE", str(tmp_path / "endpoint.key"))
    path = fixtures(tmp_path)
    with image_server() as (url, requests):
        first = bench(url, path, rounds=1)
        second = bench(url, path, rounds=1, extend=first["evidence_id"])
        assert [r["index"] for r in second["data"]["rounds"]] == [0, 1]
        assert second["data"]["extends"] == first["evidence_id"]
        before = len(requests)
        for changes in [{"dtype": "fp16"}, {"model": "different"}, {"gpu_name": "Different GPU"},
                        {"seed_base": 2}, {"engine_version": "other"}, {"first_round": 0}, {"first_round": 2}]:
            with pytest.raises(OpError):
                bench(url, path, rounds=1, extend=first["evidence_id"], **changes)
        assert len(requests) == before
        art = next(a for a in first["artifacts"] if a["role"] == "requests")
        (store.run_dir(first["evidence_id"]) / art["path"]).write_text("tampered")
        with pytest.raises(OpError, match="missing or changed"):
            bench(url, path, rounds=1, extend=first["evidence_id"])


def test_request_set_covers_every_effective_choice(tmp_path):
    rows = bi.load_image_set(fixtures(tmp_path))
    base = bi.request_set(rows, "model", {"dtype": "bf16"}, "per_round", 0, None)
    assert base != bi.request_set(rows, "model", {"dtype": "fp16"}, "per_round", 0, None)
    assert base != bi.request_set(rows, "model", {"dtype": "bf16"}, "per_round", 1, None)
    assert base != bi.request_set(rows, "model", {"dtype": "bf16"}, "list", 0, [1, 2])
    assert base != bi.request_set(rows, "model", {"dtype": "bf16"}, "per_round", 0, None,
                                 "fixed_negative_embedding")
    rows[0]["settings"]["scheduler"] = "other"
    assert base != bi.request_set(rows, "model", {"dtype": "bf16"}, "per_round", 0, None)


def test_only_registered_fixed_empty_negative_reuse_is_permitted(tmp_path, store):
    path = fixtures(tmp_path)
    with image_server("fixed_negative_cached") as (url, _):
        forbidden = bench(url, path)
        allowed = bench(url, path, allow_cache="fixed_negative_embedding")
    assert not forbidden["data"]["claim_validity"]["valid"]
    assert allowed["data"]["claim_validity"]["valid"]
    assert allowed["data"]["claim_validity"]["fixed_negative_reused_requests"] == 2
    assert forbidden["identity"]["workload"]["request_set"] != allowed["identity"]["workload"]["request_set"]
    with image_server("positive_and_negative_cached") as (url, _):
        wrong = bench(url, path, allow_cache="fixed_negative_embedding")
    assert not wrong["data"]["claim_validity"]["valid"]


@pytest.mark.parametrize("settings", [{}, {"negative_prompt": "different"}])
def test_fixed_negative_allowance_requires_explicit_empty_prompt_before_network(tmp_path, settings):
    row = {"id": "one", "prompt": "fox", "width": 4, "height": 3, "steps": 5,
           "guidance_scale": 3.0, "settings": settings}
    with image_server() as (url, requests):
        with pytest.raises(OpError, match="negative_prompt"):
            bench(url, fixtures(tmp_path, [row]), allow_cache="fixed_negative_embedding")
        assert requests == []


def test_extend_cannot_change_fixed_negative_cache_policy(tmp_path, store):
    with image_server() as (url, requests):
        path = fixtures(tmp_path)
        old = bench(url, path, rounds=1)
        before = len(requests)
        with pytest.raises(OpError, match="identical workload"):
            bench(url, path, rounds=1, extend=old["evidence_id"], allow_cache="fixed_negative_embedding")
        assert len(requests) == before


@pytest.mark.parametrize("change", [{"width": 0}, {"steps": True}, {"guidance_scale": float("nan")},
                                   {"id": "../escape"}, {"ignored_setting": 2},
                                   {"settings": {"seed": 5}}, {"settings": {"api_key": "secret"}}])
def test_invalid_fixture_rejected_before_request(tmp_path, change):
    row = {"id": "one", "prompt": "fox", "width": 4, "height": 3, "steps": 5, "guidance_scale": 3.0, **change}
    with pytest.raises(OpError):
        bi.load_image_set(fixtures(tmp_path, [row]))


def test_seed_list_bounds_and_fixed_policy(tmp_path, store):
    path = fixtures(tmp_path)
    with image_server() as (url, requests):
        with pytest.raises(OpError, match="cover"):
            bench(url, path, seed_policy="list", seeds=[1])
        assert not requests
        fixed = bench(url, path, seed_policy="fixed")
    assert not fixed["data"]["claim_validity"]["valid"]
    assert any("output-cache absence" in limitation for limitation in fixed["limitations"])


def test_pin_conflict_rejected_before_request(tmp_path):
    row = {"id": "one", "prompt": "fox", "width": 4, "height": 3, "steps": 5,
           "guidance_scale": 3.0, "settings": {"dtype": "fp16"}}
    with image_server() as (url, requests):
        with pytest.raises(OpError, match="conflicts"):
            bench(url, fixtures(tmp_path, [row]))
        assert not requests
