"""The evaluating-quantization skill's GGUF header reader, on synthetic GGUF files (no network)."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parent.parent / "plugin" / "skills" / "evaluating-quantization" / "scripts"
          / "gguf_bytes_per_token.py")
TYPES = {"F32": (0, 1, 4), "F16": (1, 1, 2), "Q8_0": (8, 32, 34), "Q4_K": (12, 256, 144), "Q6_K": (14, 256, 210)}


def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def _gguf(path: Path, meta: dict, tensors: list[tuple[str, list[int], str]]) -> None:
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta))
    for k, v in meta.items():
        out += _s(k)
        if isinstance(v, str):
            out += struct.pack("<I", 8) + _s(v)
        elif isinstance(v, list):  # array of strings, like a tokenizer table
            out += struct.pack("<IIQ", 9, 8, len(v)) + b"".join(_s(x) for x in v)
        else:
            out += struct.pack("<II", 4, v)
    offset = 0
    for name, dims, t in tensors:
        out += _s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<IQ", TYPES[t][0], offset)
        offset += 1024
    path.write_bytes(out + b"\0" * 64)


def _run(*paths: Path) -> dict:
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), *map(str, paths)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def _nbytes(dims: list[int], t: str) -> int:
    n = 1
    for d in dims:
        n *= d
    return n // TYPES[t][1] * TYPES[t][2]


def test_dense_untied(tmp_path):
    tensors = [("token_embd.weight", [256, 10], "Q8_0"), ("output.weight", [256, 10], "Q6_K"),
               ("blk.0.attn_q.weight", [256, 4], "Q4_K"), ("blk.0.attn_norm.weight", [256], "F32")]
    _gguf(tmp_path / "m.gguf", {"general.architecture": "llama", "tokenizer.ggml.tokens": [f"t{i}" for i in range(40)]},
          tensors)
    d = _run(tmp_path / "m.gguf")
    embd = _nbytes([256, 10], "Q8_0")
    want = embd / 10 + _nbytes([256, 10], "Q6_K") + _nbytes([256, 4], "Q4_K") + _nbytes([256], "F32")
    assert d["weight_bytes_per_token"] == want
    assert d["stored_bytes"] == sum(_nbytes(dims, t) for _, dims, t in tensors)
    assert d["lm_head"] == {"type": "Q6_K", "tied": False, "bytes": _nbytes([256, 10], "Q6_K")}
    assert d["linear_bits_per_weight"] == 4.5


def test_tied_and_moe(tmp_path):
    tensors = [("token_embd.weight", [256, 10], "F16"), ("blk.0.ffn_up_exps.weight", [256, 2, 8], "Q4_K"),
               ("blk.0.ffn_up_shexp.weight", [256, 2], "Q8_0")]
    meta = {"general.architecture": "qwen3moe", "qwen3moe.expert_count": 8, "qwen3moe.expert_used_count": 2}
    _gguf(tmp_path / "moe.gguf", meta, tensors)
    d = _run(tmp_path / "moe.gguf")
    assert d["lm_head"]["tied"] is True and d["lm_head"]["bytes"] == _nbytes([256, 10], "F16")
    assert d["groups_read_bytes"]["moe_experts"] == _nbytes([256, 2, 8], "Q4_K") * 2 / 8
    assert d["weight_bytes_per_token"] == (_nbytes([256, 10], "F16") + _nbytes([256, 2, 8], "Q4_K") / 4
                                           + _nbytes([256, 2], "Q8_0"))


def test_refuses_to_guess(tmp_path):
    _gguf(tmp_path / "x.gguf", {"general.architecture": "qwen3moe"}, [("blk.0.ffn_up_exps.weight", [256, 2, 8], "Q4_K")])
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), str(tmp_path / "x.gguf")], capture_output=True, text=True)
    assert r.returncode == 2 and "expert_count" in r.stderr
    (tmp_path / "bad.gguf").write_bytes(b"NOPE" + b"\0" * 32)
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), str(tmp_path / "bad.gguf")], capture_output=True, text=True)
    assert r.returncode == 2 and "magic" in r.stderr
