"""The byte-floor fields of the optimizing-inference skill's bytes_per_token.py, on synthetic safetensors headers."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "plugin" / "skills" / "optimizing-inference" / "scripts" / \
    "bytes_per_token.py"
CFG = {"num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
       "hidden_size": 32, "vocab_size": 10, "tie_word_embeddings": False}
KV = 2 * 2 * 8 * 2 * 100 * 2  # K and V x kv_heads x head_dim x 2 bytes x context 100 x 2 layers


def checkpoint(d: Path, tensors: dict, cfg: dict = CFG) -> Path:
    """config.json plus one safetensors file holding only a header: name -> (dtype, byte size)."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg))
    off, hdr = 0, {"__metadata__": {"format": "pt"}}
    for name, (dtype, size) in tensors.items():
        hdr[name] = {"dtype": dtype, "shape": [size], "data_offsets": [off, off + size]}
        off += size
    blob = json.dumps(hdr).encode()
    (d / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob)
    return d


def run(model: Path, *flags) -> tuple[int, dict | None, str]:
    p = subprocess.run([sys.executable, "-I", str(SCRIPT), str(model), "100", *map(str, flags)],
                       capture_output=True, text=True)
    return p.returncode, (json.loads(p.stdout) if p.returncode == 0 else None), p.stderr


# An FP8 checkpoint with BF16 norms, a BF16 output layer and a BF16 embedding.
FP8 = {"model.embed_tokens.weight": ("BF16", 640), "lm_head.weight": ("BF16", 640),
       "model.layers.0.mlp.w": ("F8_E4M3", 1000), "model.layers.0.mlp.w_scale": ("F32", 40),
       "model.layers.1.mlp.w": ("F8_E4M3", 1000), "model.layers.0.norm.weight": ("BF16", 64),
       "model.layers.1.norm.weight": ("BF16", 64)}


def test_floor_fields_on_an_fp8_checkpoint(tmp_path):
    rc, out, err = run(checkpoint(tmp_path / "m", FP8))
    assert rc == 0, err
    weights = 2000 + 40 + 128 + 640 + 64  # every-token tensors + output layer + one embedding row
    assert out["weights_per_token"] == weights and out["bytes_per_token"] == weights + KV  # unchanged keys
    assert out["bytes_by_dtype"] == {"F8_E4M3": 2000, "F32": 40, "BF16": 128 + 640 + 64}
    assert sum(out["bytes_by_dtype"].values()) == out["weights_per_token"]
    assert out["packable_16bit_bytes"] == 128  # the BF16 norms; never the output layer or the embedding row
    assert out["lm_head_bytes"] == 640
    assert out["lm_head_share"] == pytest.approx(640 / (weights + KV))
    want = weights + KV - 128 * (1 - 12 / 16) - 640 * (1 - 8.5 / 16)
    assert out["floor_bytes_per_token"] == pytest.approx(want)
    assert "12/16" in out["floor_formula"] and "8.5/16" in out["floor_formula"]


def test_bit_widths_are_flags_and_bounded(tmp_path):
    m = checkpoint(tmp_path / "m", FP8)
    rc, out, err = run(m, "--pack-bits", 16, "--head-proxy-bits", 16)
    assert rc == 0, err
    assert out["floor_bytes_per_token"] == out["bytes_per_token"]  # no cut at 16 bits
    rc, out, _ = run(m, "--pack-bits", 8, "--head-proxy-bits", 4)
    assert out["floor_bytes_per_token"] == pytest.approx(out["bytes_per_token"] - 128 * 0.5 - 640 * 0.75)
    rc, _, err = run(m, "--pack-bits", 17)
    assert rc == 2 and "--pack-bits" in err


def test_bf16_moe_packs_the_routed_experts_read_per_token(tmp_path):
    cfg = {**CFG, "num_experts": 4, "num_experts_per_tok": 1}
    tensors = {**FP8, "model.layers.0.mlp.w": ("BF16", 1000), "model.layers.1.mlp.w": ("BF16", 1000),
               **{f"model.layers.0.mlp.experts.{i}.w": ("BF16", 400) for i in range(4)}}
    rc, out, err = run(checkpoint(tmp_path / "m", tensors, cfg))
    assert rc == 0, err
    assert out["packable_16bit_bytes"] == 2000 + 128 + 1600 * 0.25
    assert out["bytes_by_dtype"]["BF16"] == pytest.approx(2000 + 128 + 400 + 640 + 64)
    assert sum(out["bytes_by_dtype"].values()) == pytest.approx(out["weights_per_token"])


def test_tied_output_layer_is_the_embedding(tmp_path):
    tensors = {k: v for k, v in FP8.items() if k != "lm_head.weight"}
    rc, out, err = run(checkpoint(tmp_path / "m", tensors, {**CFG, "tie_word_embeddings": True}))
    assert rc == 0, err
    assert out["tied_lm_head"] is True and out["lm_head_bytes"] == 640
    assert out["bytes_by_dtype"]["BF16"] == pytest.approx(128 + 640 + 64)
    assert out["floor_bytes_per_token"] == pytest.approx(out["bytes_per_token"] - 128 * (1 - 12 / 16)
                                                         - 640 * (1 - 8.5 / 16))


def test_quantized_output_layer_and_f16_are_not_cut(tmp_path):
    tensors = {**FP8, "lm_head.weight": ("F8_E4M3", 320), "model.layers.0.norm.weight": ("F16", 64)}
    rc, out, err = run(checkpoint(tmp_path / "m", tensors))
    assert rc == 0, err
    assert out["lm_head_bytes"] == 320 and out["packable_16bit_bytes"] == 64  # only the BF16 norm
    assert out["floor_bytes_per_token"] == pytest.approx(out["bytes_per_token"] - 64 * (1 - 12 / 16))
    notes = " ".join(out["notes"])
    assert "F16 weight bytes" in notes and "16-bit part" in notes
