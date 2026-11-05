import json

import pytest

from inferencekit import registry
from inferencekit.ops.roofline import decode_bytes, matrix_bytes
from inferencekit.registry import OpError

# Shaped like Qwen3-8B: untied lm_head, q/k norms.
QWEN3_8B_LIKE = {
    "model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36, "intermediate_size": 12288,
    "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936,
    "tie_word_embeddings": False,
}
GB = 1e9


def test_fp8_with_bf16_lm_head_is_about_8_2_to_8_4_gb_per_token():
    c = decode_bytes(QWEN3_8B_LIKE, "fp8", context=1024)
    assert c["dtypes"]["lm_head"] == "bf16"
    # linear layers: 36 x (2 x 4096^2 + 2 x 1024 x 4096 + 3 x 12288 x 4096) parameters
    assert c["shape"]["linear_params"] == 6_945_767_424
    assert c["lm_head_bytes"] == 151936 * 4096 * 2
    assert 8.18 * GB < c["weight_bytes"] < 8.21 * GB
    assert 8.2 * GB < c["bytes_per_token"] < 8.4 * GB
    # KV per context token: 2 (K,V) x 8 heads x 128 dims x 2 bytes x 36 layers
    assert c["kv_bytes_per_context_token"] == 2 * 8 * 128 * 2 * 36
    assert c["kv_bytes"] == c["kv_bytes_per_context_token"] * 1024


def test_fp8_block_scales_are_negligible():
    rows, cols = 4096, 4096
    overhead = matrix_bytes(rows, cols, "fp8") - rows * cols
    assert overhead == (rows // 128) * (cols // 128) * 4  # one fp32 scale per 128x128 block
    assert overhead / (rows * cols) < 1e-3


def test_bf16_and_q8_0_scale_with_bytes_per_weight():
    bf16 = decode_bytes(QWEN3_8B_LIKE, "bf16", context=0)
    q8 = decode_bytes(QWEN3_8B_LIKE, "q8_0", context=0)
    params = bf16["shape"]["linear_params"]
    assert bf16["layers_bytes"] == pytest.approx(params * 2, rel=1e-3)
    assert q8["layers_bytes"] == pytest.approx(params * 8.5 / 8, rel=1e-3)
    assert q8["lm_head_bytes"] == pytest.approx(151936 * 4096 * 8.5 / 8)


def test_tied_lm_head_is_still_read_in_full():
    tied = dict(QWEN3_8B_LIKE, tie_word_embeddings=True)
    a = decode_bytes(QWEN3_8B_LIKE, "bf16", context=0)
    b = decode_bytes(tied, "bf16", context=0)
    assert a["lm_head_bytes"] == b["lm_head_bytes"]
    assert b["embedding_row_bytes"] == 4096 * 2  # one row, never the whole matrix


def test_moe_configs_are_rejected():
    with pytest.raises(OpError):
        decode_bytes(dict(QWEN3_8B_LIKE, num_experts=64), "bf16", context=0)


def test_roofline_operation_ceiling(tmp_path, store):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(QWEN3_8B_LIKE))
    rec = registry.call("roofline", {"model": str(cfg), "dtype": "fp8", "gpu_bw": "1600", "context": 1024})
    m = {x["name"]: x["value"] for x in rec["metrics"]}
    assert m["ceiling_tok_s"] == pytest.approx(1600 * GB / m["bytes_per_token"])
    assert 185 < m["ceiling_tok_s"] < 200
    assert rec["kind"] == "roofline" and rec["claim_class"] == "derived"
    assert any("user-supplied" in lim for lim in rec["limitations"])


def _ceiling(store, gbs=1600.0):
    from conftest import fake_call

    from inferencekit import evidence as ev

    rec = ev.build(fake_call("measure_bandwidth", ("ceiling", "bw")), kind="ceiling", claim_class="observed",
                   summary="synthetic ceiling",
                   metrics=[ev.metric("dram_read_gbs", gbs, "GB/s", "best_sustained", n=30)])
    return store.save(rec)


def test_an_edited_ceiling_record_is_refused(tmp_path, store):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(QWEN3_8B_LIKE))
    ceiling = _ceiling(store)
    ok = registry.call("roofline", {"model": str(cfg), "dtype": "fp8", "gpu_bw": ceiling["evidence_id"]})
    assert ok["data"]["bandwidth"]["evidence_id"] == ceiling["evidence_id"]
    path = store.run_dir(ceiling["evidence_id"]) / "evidence.json"
    edited = json.loads(path.read_text())
    edited["metrics"][0]["value"] = 3000.0
    path.write_text(json.dumps(edited))
    with pytest.raises(OpError, match="does not match its content"):
        registry.call("roofline", {"model": str(cfg), "dtype": "fp8", "gpu_bw": ceiling["evidence_id"]})


def test_report_flags_a_roofline_at_another_context(tmp_path, store):
    from conftest import make_benchmark

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(QWEN3_8B_LIKE))
    bench = make_benchmark(store, [150.0, 151.0], prompt_len=128)  # gen_len 128: mean context 192
    far = registry.call("roofline", {"model": str(cfg), "dtype": "fp8", "gpu_bw": "1600", "context": 32768})
    near = registry.call("roofline", {"model": str(cfg), "dtype": "fp8", "gpu_bw": "1600", "context": 192})
    out = registry.call("report", {"ids": [bench["evidence_id"], far["evidence_id"], near["evidence_id"]],
                                   "format": "json"})
    ceiling_lines = [f for f in out["report"]["findings"] if "bandwidth ceiling" in f]
    far_line = next(f for f in ceiling_lines if far["evidence_id"][:15] in f)
    near_line = next(f for f in ceiling_lines if near["evidence_id"][:15] in f)
    assert "roofline context 32768 vs benchmark mean context 192" in far_line
    assert "rerun roofline --context 192" in far_line
    assert "roofline context" not in near_line
