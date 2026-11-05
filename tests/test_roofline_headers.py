"""roofline --method headers: exact bytes per token from safetensors headers (synthetic files, no torch)."""

import hashlib
import http.server
import json
import math
import os
import socket
import struct
import sys
import threading

import pytest

from inferencekit import checkpoint, registry
from inferencekit.ops.report import reproduce_command
from inferencekit.ops.roofline import _precision_label, decode_bytes
from inferencekit.registry import OpError
from inferencekit.roofline_headers import header_decode_bytes, tensor_formats

GB = 1e9


@pytest.fixture(autouse=True)
def hermetic_hf(tmp_path, monkeypatch):
    """An empty HF cache and online mode, whatever the machine running the tests has set."""
    cache = tmp_path / "hf-cache"
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HOME", "HUGGINGFACE_HUB_CACHE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    try:
        from huggingface_hub import constants
    except ImportError:
        return cache
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    return cache


def set_offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    try:
        from huggingface_hub import constants
    except ImportError:
        return
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
BITS = {"BF16": 16, "F32": 32, "F8_E4M3": 8, "U8": 8, "F16": 16}


def nbytes(dtype, shape):
    return math.ceil(math.prod(shape) * BITS[dtype] / 8)


def header_bytes(tensors: dict) -> bytes:
    """The 8-byte length prefix + JSON header of a safetensors file holding ``tensors``."""
    header, off = {}, 0
    for name, (dtype, shape) in tensors.items():
        size = nbytes(dtype, shape)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + size]}
        off += size
    header["__metadata__"] = {"format": "pt"}
    raw = json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw


def write_safetensors(path, tensors: dict, data: bool = True):
    """A safetensors file; with data=False only the header (the reader never touches the data)."""
    blob = header_bytes(tensors)
    if data:
        blob += bytes(sum(nbytes(d, s) for d, s in tensors.values()))
    path.write_bytes(blob)
    return blob


def nvfp4(prefix, rows, cols):
    """ModelOpt NVFP4 linear: packed U8 values (2 per byte), FP8 scale per 16, two fp32 scalars."""
    return {f"{prefix}.weight": ("U8", (rows, cols // 2)), f"{prefix}.weight_scale": ("F8_E4M3", (rows, cols // 16)),
            f"{prefix}.weight_scale_2": ("F32", ()), f"{prefix}.input_scale": ("F32", ())}


def fp8(prefix, rows, cols):
    """ModelOpt FP8 per-tensor linear."""
    return {f"{prefix}.weight": ("F8_E4M3", (rows, cols)), f"{prefix}.weight_scale": ("F32", ()),
            f"{prefix}.input_scale": ("F32", ())}


def bf16(prefix, rows, cols=None):
    return {f"{prefix}.weight": ("BF16", (rows, cols) if cols else (rows,))}


def size_of(tensors: dict) -> int:
    return sum(nbytes(d, s) for d, s in tensors.values())


# --------------------------------------------------------------------------- a Qwen3.5-style hybrid (GDN + full)

H, I, V = 64, 128, 1000
GDN = {"linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16,
       "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4}
CONV_DIM = 2 * 2 * 16 + 4 * 16  # q, k (key heads) + v (value heads) = 128
HYBRID_CFG = {"model_type": "qwen3_5", "tie_word_embeddings": False,
              "quantization_config": {"quant_algo": "MIXED_PRECISION", "quant_method": "modelopt"},
              "text_config": {"hidden_size": H, "num_hidden_layers": 4, "intermediate_size": I,
                              "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
                              "vocab_size": V, "tie_word_embeddings": False, "mamba_ssm_dtype": "float32",
                              "dtype": "bfloat16", "mtp_num_hidden_layers": 1,
                              "layer_types": ["linear_attention"] * 3 + ["full_attention"], **GDN}}


def hybrid_tensors():
    p = "model.language_model"
    t = {f"{p}.embed_tokens.weight": ("BF16", (V, H)), f"{p}.norm.weight": ("BF16", (H,))}
    t.update(nvfp4("lm_head", V, H))
    groups = {"attention": {}, "linear_attention": {}, "mlp_dense": {}, "norms": {}}
    for i in range(4):
        L = f"{p}.layers.{i}"
        groups["mlp_dense"].update({**nvfp4(f"{L}.mlp.gate_proj", I, H), **nvfp4(f"{L}.mlp.up_proj", I, H),
                                    **nvfp4(f"{L}.mlp.down_proj", H, I)})
        groups["norms"].update({**bf16(f"{L}.input_layernorm", H), **bf16(f"{L}.post_attention_layernorm", H)})
        if i < 3:
            a = f"{L}.linear_attn"
            groups["linear_attention"].update({
                **fp8(f"{a}.in_proj_qkv", CONV_DIM, H), **fp8(f"{a}.in_proj_z", 64, H), **fp8(f"{a}.out_proj", H, 64),
                **bf16(f"{a}.in_proj_a", 4, H), **bf16(f"{a}.in_proj_b", 4, H),
                f"{a}.conv1d.weight": ("BF16", (CONV_DIM, 1, 4)), f"{a}.A_log": ("F32", (4,)),
                f"{a}.dt_bias": ("BF16", (4,))})
            groups["norms"].update(bf16(f"{a}.norm", 16))
        else:
            a = f"{L}.self_attn"
            groups["attention"].update({**fp8(f"{a}.q_proj", 2 * 64, H), **fp8(f"{a}.k_proj", 32, H),
                                        **fp8(f"{a}.v_proj", 32, H), **fp8(f"{a}.o_proj", H, 64)})
            groups["norms"].update({**bf16(f"{a}.q_norm", 16), **bf16(f"{a}.k_norm", 16)})
    for g in groups.values():
        t.update(g)
    groups["norms"].update(bf16(f"{p}.norm", H))
    vision = {"model.visual.blocks.0.attn.qkv.weight": ("BF16", (96, 32)),
              "model.visual.merger.linear_fc1.weight": ("BF16", (H, 128))}
    mtp = {**bf16("mtp.fc", H, 2 * H), **bf16("mtp.layers.0.mlp.gate_proj", I, H), **bf16("mtp.norm", H)}
    t.update(vision)
    t.update(mtp)
    return t, groups, vision, mtp


def test_hybrid_nvfp4_gdn_counts_every_group_exactly(tmp_path):
    tensors, groups, vision, mtp = hybrid_tensors()
    write_safetensors(tmp_path / "model.safetensors", tensors)
    ck = checkpoint.load(str(tmp_path))
    c = header_decode_bytes(ck, HYBRID_CFG, context=100, kv_dtype="bf16")
    for g, ts in groups.items():
        assert c["groups"][g]["read_bytes"] == size_of(ts), g
    # NVFP4 lm_head: V x H/2 packed bytes + V x H/16 fp8 scales + two fp32 scalars, read in full
    assert c["lm_head_bytes"] == V * H // 2 + V * H // 16 + 8
    assert c["groups"]["embedding_row"]["read_bytes"] == H * 2  # one bf16 row, not the table
    assert c["groups"]["embedding_row"]["stored_bytes"] == V * H * 2
    assert c["excluded"]["towers"]["stored_bytes"] == size_of(vision)
    assert c["excluded"]["mtp"]["stored_bytes"] == size_of(mtp)
    # stored bytes reconcile: every tensor is in exactly one group or excluded
    assert sum(g["stored_bytes"] for g in c["groups"].values()) + size_of(vision) + size_of(mtp) \
        == ck.stored_bytes == size_of(tensors)
    # full-attention KV, one layer: 2 x (2 kv heads x 16) elements x 2 bytes x context
    assert c["layer_kinds"] == {"linear": 3, "full": 1}
    assert c["kv_bytes"] == 2 * 32 * 2 * 100
    assert c["kv_bytes_per_context_token"] == 2 * 32 * 2
    # GDN state per layer: 4 value heads x 16 x 16 fp32 (config mamba_ssm_dtype) + conv 128 x 3 bf16; read + write
    per_layer = 2 * (4 * 16 * 16 * 4 + CONV_DIM * 3 * 2)
    assert c["state"]["per_layer"]["state_dtype"] == "fp32"
    assert c["state_bytes"] == 3 * per_layer
    assert c["bytes_per_token"] == c["weight_bytes"] + c["kv_bytes"] + c["state_bytes"]
    assert not c["unproven"] and not c["other_patterns"]
    bf = header_decode_bytes(ck, HYBRID_CFG, context=100, state_dtype="bf16")
    assert bf["state_bytes"] == 3 * 2 * (4 * 16 * 16 * 2 + CONV_DIM * 3 * 2)
    with_mtp = header_decode_bytes(ck, HYBRID_CFG, context=100, include_mtp=True)
    assert with_mtp["weight_bytes"] == c["weight_bytes"] + size_of(mtp)


def test_unknown_linear_attention_state_is_unproven_not_guessed(tmp_path, store):
    tensors, *_ = hybrid_tensors()
    write_safetensors(tmp_path / "model.safetensors", tensors)
    cfg = json.loads(json.dumps(HYBRID_CFG))
    for k in GDN:
        cfg["text_config"].pop(k)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    c = header_decode_bytes(checkpoint.load(str(tmp_path)), cfg, context=10)
    assert c["state_bytes"] is None and c["state"]["status"] == "UNPROVEN"
    assert any("layers 0, 1, 2" in u and "state layout unknown" in u for u in c["unproven"])
    rec = registry.call("roofline", {"model": str(tmp_path), "gpu_bw": "1000"})
    assert rec["coverage"]["complete"] is False
    assert any(m.startswith("UNPROVEN") for m in rec["coverage"]["missing"])
    assert "UNPROVEN" in rec["summary"] and "upper bound" in rec["summary"]


def test_gdn_config_that_contradicts_the_tensors_is_unproven(tmp_path):
    tensors, *_ = hybrid_tensors()
    write_safetensors(tmp_path / "model.safetensors", tensors)
    cfg = json.loads(json.dumps(HYBRID_CFG))
    cfg["text_config"]["linear_num_value_heads"] = 8  # conv1d / A_log say 4
    c = header_decode_bytes(checkpoint.load(str(tmp_path)), cfg, context=10)
    assert c["state_bytes"] is None
    assert any("does not match the checkpoint" in u for u in c["unproven"])


# --------------------------------------------------------------------------- a Gemma-4-style MoE (sliding + full)

GH, GI, GE, GK, GV = 32, 48, 4, 2, 500
GEMMA_CFG = {"model_type": "gemma4", "tie_word_embeddings": True,
             "quantization_config": {"quant_algo": "NVFP4", "kv_cache_scheme": {"num_bits": 8, "type": "float"}},
             "text_config": {"hidden_size": GH, "num_hidden_layers": 2, "intermediate_size": GI,
                             "moe_intermediate_size": 16, "num_experts": GE, "top_k_experts": GK,
                             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                             "num_global_key_value_heads": 1, "global_head_dim": 16, "attention_k_eq_v": True,
                             "sliding_window": 64, "vocab_size": GV, "tie_word_embeddings": True,
                             "layer_types": ["sliding_attention", "full_attention"]}}


def gemma_tensors(fused=False):
    p = "model.language_model"
    t = {f"{p}.embed_tokens.weight": ("BF16", (GV, GH)), f"{p}.norm.weight": ("BF16", (GH,))}
    experts, always = {}, {}
    for i in range(2):
        L = f"{p}.layers.{i}"
        kv_rows = 2 * 8 if i == 0 else 16
        attn = {**bf16(f"{L}.self_attn.q_proj", 32 if i == 0 else 64, GH), **bf16(f"{L}.self_attn.k_proj", kv_rows, GH),
                **bf16(f"{L}.self_attn.o_proj", GH, 32 if i == 0 else 64)}
        if i == 0:
            attn.update(bf16(f"{L}.self_attn.v_proj", kv_rows, GH))  # full layers have K = V: no v_proj
        always.update(attn)
        always.update({**bf16(f"{L}.mlp.gate_proj", GI, GH), **bf16(f"{L}.mlp.up_proj", GI, GH),
                       **bf16(f"{L}.mlp.down_proj", GH, GI), **bf16(f"{L}.router.proj", GE, GH),
                       f"{L}.router.scale": ("BF16", (GH,)), **bf16(f"{L}.input_layernorm", GH),
                       f"{L}.layer_scalar": ("BF16", (1,))})
        if fused:
            experts.update({f"{L}.experts.gate_up_proj": ("BF16", (GE, 2 * 16, GH)),
                            f"{L}.experts.down_proj": ("BF16", (GE, GH, 16))})
        else:
            for e in range(GE):
                E = f"{L}.experts.{e}"
                experts.update({**nvfp4(f"{E}.gate_proj", 16, GH), **nvfp4(f"{E}.up_proj", 16, GH),
                                **nvfp4(f"{E}.down_proj", GH, 16)})
    t.update(always)
    t.update(experts)
    t["model.embed_vision.embedding_projection.weight"] = ("BF16", (GH, 24))
    return t, experts, always


@pytest.mark.parametrize("fused", [False, True])
def test_moe_reads_top_k_of_n_experts_and_tied_lm_head_once(tmp_path, fused):
    tensors, experts, always = gemma_tensors(fused)
    write_safetensors(tmp_path / "model.safetensors", tensors)
    c = header_decode_bytes(checkpoint.load(str(tmp_path)), GEMMA_CFG, context=1000, kv_dtype="fp8")
    assert c["groups"]["moe_experts"]["stored_bytes"] == size_of(experts)
    assert c["groups"]["moe_experts"]["read_bytes"] == size_of(experts) * GK / GE
    assert c["moe"]["top_k"] == GK and c["moe"]["num_experts"] == GE and "top_k" in c["moe"]["caveat"]
    assert not c["notes"] or all("expert tensors" not in n and "fused expert" not in n for n in c["notes"])
    # tied: the embedding is the lm_head, read in full once; no extra embedding row
    assert c["lm_head_bytes"] == GV * GH * 2
    assert "embedding_row" not in c["groups"]
    assert c["weight_bytes"] == size_of(always) + size_of(experts) * GK / GE + GV * GH * 2 + GH * 2
    # KV at fp8: sliding layer reads min(1000, 64) tokens of 2 x 16 elements; the full layer (K = V, 1 x 16)
    # reads all 1000 tokens of 2 x 16 elements
    assert c["kv_bytes"] == 64 * 32 * 1 + 1000 * 32 * 1
    assert c["kv_bytes_per_context_token"] == 32  # only the full layer grows past the window
    assert any("attention_k_eq_v" in n for n in c["notes"])
    assert c["state"] is None and c["state_bytes"] == 0


def test_moe_record_carries_caveat_and_kv_dtype_hint(tmp_path, store):
    tensors, *_ = gemma_tensors()
    write_safetensors(tmp_path / "model.safetensors", tensors)
    (tmp_path / "config.json").write_text(json.dumps(GEMMA_CFG))
    rec = registry.call("roofline", {"model": str(tmp_path), "context": 1000})
    assert rec["data"]["counts"]["method"] == "headers"
    assert any("Batch-1 decode reads exactly top_k" in lim for lim in rec["limitations"])
    assert any("quant config declares an fp8 KV cache" in lim for lim in rec["limitations"])  # counted bf16
    assert rec["identity"]["model"]["quant"] == "nvfp4"
    # the bf16 attention / dense MLP / tied lm_head outweigh the 2-of-4 NVFP4 experts read per token
    assert rec["identity"]["model"]["dtype"] == "bf16+nvfp4"
    assert any("towers" in lim for lim in rec["limitations"])


def test_moe_without_top_k_in_config_is_an_error(tmp_path):
    tensors, *_ = gemma_tensors()
    write_safetensors(tmp_path / "model.safetensors", tensors)
    cfg = json.loads(json.dumps(GEMMA_CFG))
    cfg["text_config"].pop("top_k_experts")
    with pytest.raises(OpError, match="top-k"):
        header_decode_bytes(checkpoint.load(str(tmp_path)), cfg, context=10)


# --------------------------------------------------------------------------- headers vs the config method

QWEN3_8B_FP8_CFG = {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36, "intermediate_size": 12288,
                    "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936,
                    "tie_word_embeddings": False}


def qwen3_8b_fp8_tensors():
    """Qwen3-8B-FP8 as published: e4m3 weights + fp32 128x128 block scales, bf16 norms/embeddings/lm_head."""
    hdim, inter = 4096, 12288
    t = {"model.embed_tokens.weight": ("BF16", (151936, hdim)), "lm_head.weight": ("BF16", (151936, hdim)),
         "model.norm.weight": ("BF16", (hdim,))}
    shapes = {"self_attn.q_proj": (4096, hdim), "self_attn.k_proj": (1024, hdim), "self_attn.v_proj": (1024, hdim),
              "self_attn.o_proj": (hdim, 4096), "mlp.gate_proj": (inter, hdim), "mlp.up_proj": (inter, hdim),
              "mlp.down_proj": (hdim, inter)}
    for i in range(36):
        L = f"model.layers.{i}"
        for name, (r, c) in shapes.items():
            t[f"{L}.{name}.weight"] = ("F8_E4M3", (r, c))
            t[f"{L}.{name}.weight_scale_inv"] = ("F32", (math.ceil(r / 128), math.ceil(c / 128)))
        for n in ("input_layernorm", "post_attention_layernorm"):
            t[f"{L}.{n}.weight"] = ("BF16", (hdim,))
        for n in ("q_norm", "k_norm"):
            t[f"{L}.self_attn.{n}.weight"] = ("BF16", (128,))
    return t


def test_headers_agree_with_the_config_method_on_qwen3_8b_fp8(tmp_path):
    # header-only shards (no tensor data): the reader only ever looks at the headers
    t = qwen3_8b_fp8_tensors()
    names = sorted(t)
    halves = {"model-00001-of-00002.safetensors": names[: len(names) // 2],
              "model-00002-of-00002.safetensors": names[len(names) // 2:]}
    for f, ns in halves.items():
        write_safetensors(tmp_path / f, {n: t[n] for n in ns}, data=False)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {n: f for f, ns in halves.items() for n in ns}}))
    (tmp_path / "stray-copy.safetensors").write_bytes(header_bytes({"x": ("BF16", (10**6,))}))  # not in the index
    ck = checkpoint.load(str(tmp_path))
    assert [f["file"] for f in ck.files] == sorted(halves)
    heads = header_decode_bytes(ck, QWEN3_8B_FP8_CFG, context=1152)
    conf = decode_bytes(QWEN3_8B_FP8_CFG, "fp8", context=1152)
    assert heads["weight_bytes"] == pytest.approx(conf["weight_bytes"], rel=0.01)
    assert heads["weight_bytes"] == pytest.approx(8.19 * GB, rel=0.01)
    assert heads["kv_bytes"] == conf["kv_bytes"]
    assert heads["bytes_per_token"] == pytest.approx(conf["bytes_per_token"], rel=0.01)
    assert heads["lm_head_bytes"] == conf["lm_head_bytes"]


def test_record_lists_each_header_digest(tmp_path, store):
    blob = write_safetensors(tmp_path / "model.safetensors", qwen3_8b_fp8_tensors(), data=False)
    (tmp_path / "config.json").write_text(json.dumps(QWEN3_8B_FP8_CFG))
    rec = registry.call("roofline", {"model": str(tmp_path), "gpu_bw": "1600", "context": 1152, "dtype": "bf16"})
    ck = rec["data"]["counts"]["checkpoint"]
    (n,) = struct.unpack("<Q", blob[:8])
    assert ck["files"] == [{"file": "model.safetensors", "header_len": n,
                            "header_sha256": hashlib.sha256(blob[8:8 + n]).hexdigest(),
                            "tensors": len(qwen3_8b_fp8_tensors()), "data_bytes": ck["stored_bytes"]}]
    assert len(ck["headers_digest"]) == 64
    m = {x["name"]: x["value"] for x in rec["metrics"]}
    assert m["ceiling_tok_s"] == pytest.approx(1600 * GB / m["bytes_per_token"])
    assert {"bytes_attention", "bytes_mlp_dense", "bytes_lm_head"} <= set(m)
    assert rec["claim_class"] == "derived"
    assert any("--dtype bf16 ignored" in lim for lim in rec["limitations"])
    cfg_rec = registry.call("roofline", {"model": str(tmp_path), "method": "config", "dtype": "fp8",
                                         "context": 1152})
    assert cfg_rec["data"]["counts"]["method"] == "config"
    assert m["bytes_per_token"] == pytest.approx(
        next(x["value"] for x in cfg_rec["metrics"] if x["name"] == "bytes_per_token"), rel=0.01)


def test_method_headers_without_safetensors_is_an_error(tmp_path, store):
    (tmp_path / "config.json").write_text(json.dumps(QWEN3_8B_FP8_CFG))
    with pytest.raises(OpError, match="no safetensors"):
        registry.call("roofline", {"model": str(tmp_path / "config.json"), "method": "headers"})
    rec = registry.call("roofline", {"model": str(tmp_path / "config.json"), "dtype": "fp8"})  # auto -> config
    assert rec["data"]["counts"]["method"] == "config"


def test_header_sizes_must_match_their_data_offsets(tmp_path):
    raw = json.dumps({"w": {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 16]}}).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    with pytest.raises(OpError, match="data_offsets span 16"):
        checkpoint.load(str(tmp_path))
    assert checkpoint.tensor_bytes("U8", (8, 4)) == 32 and checkpoint.tensor_bytes("F4", (3,)) == 2


# --------------------------------------------------------------------------- the Hub path (a local fake Hub)

class _FakeHub(http.server.BaseHTTPRequestHandler):
    files: dict = {}
    log: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):  # noqa: N802
        type(self).log.append((self.path, self.headers.get("Range"), self.headers.get("Authorization")))
        if self.path.startswith("/org/model/resolve/abc123/") and self.path.endswith(".safetensors"):
            self.send_response(302)  # shards redirect to pre-signed storage, as the Hub does
            self.send_header("Location", "/storage/" + self.path.rsplit("/", 1)[1])
            self.send_header("X-Repo-Commit", "abc123def")
            self.end_headers()
            return
        name = self.path.rsplit("/", 1)[1]
        body = self.files.get(name)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        rng = self.headers.get("Range")
        if rng:
            a, b = (int(x) for x in rng.removeprefix("bytes=").split("-"))
            body = body[a:b + 1]
            self.send_response(206)
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_hub_reads_only_headers_by_range_and_keeps_the_token_off_redirects(tmp_path, store, monkeypatch):
    tensors, *_ = gemma_tensors()
    big = dict(tensors, **{f"pad.{i}.weight": ("BF16", (1, 1)) for i in range(3000)})  # header > first read
    blob = header_bytes(big) + bytes(10**6)
    _FakeHub.files = {"model.safetensors": blob, "config.json": json.dumps(GEMMA_CFG).encode()}
    _FakeHub.log = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeHub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{srv.server_address[1]}")
        monkeypatch.setenv("HF_TOKEN", "hf_test_secret")
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # force the urllib path
        rec = registry.call("roofline", {"model": "org/model", "revision": "abc123", "context": 100})
    finally:
        srv.shutdown()
    ck = rec["data"]["counts"]["checkpoint"]
    assert ck["source"] == "hub" and ck["commit"] == "abc123def"
    assert rec["identity"]["model"]["revision"] == "abc123def"
    (n,) = struct.unpack("<Q", blob[:8])
    assert n > checkpoint.FIRST_READ
    assert ck["files"][0]["header_sha256"] == hashlib.sha256(blob[8:8 + n]).hexdigest()
    storage = [(p, r, a) for p, r, a in _FakeHub.log if p.startswith("/storage/")]
    assert [r for _, r, _ in storage] == [f"bytes=0-{checkpoint.FIRST_READ - 1}",
                                          f"bytes={checkpoint.FIRST_READ}-{8 + n - 1}"]  # never the data
    assert all(a is None for _, _, a in storage)  # the token is not forwarded to the storage host
    assert all(a == "Bearer hf_test_secret" for p, _, a in _FakeHub.log if p.startswith("/org/"))
    assert "hf_test_secret" not in json.dumps(rec)
    assert rec["data"]["counts"]["groups"]["moe_experts"]["read_bytes"] > 0
    # the config came from the same commit as the headers
    assert ("/org/model/resolve/abc123def/config.json", None, "Bearer hf_test_secret") in _FakeHub.log
    # the stored request pins what ran (headers) and the commit read, so the reproduce command is exact
    assert rec["data"]["request"]["args"]["method"] == "headers"
    assert rec["data"]["request"]["args"]["revision"] == "abc123def"
    assert rec["data"]["request"]["requested"] == {"method": "auto", "revision": "abc123", "dtype": None}
    assert "--method headers" in reproduce_command(rec) and "--revision abc123def" in reproduce_command(rec)


# --------------------------------------------------------------------------- the local HF cache and offline mode

SHA = "0123456789abcdef0123456789abcdef01234567"


def fake_cache(cache, repo, files: dict[str, bytes], ref="main"):
    """The huggingface_hub cache layout: blobs/<hash>, snapshots/<commit>/<file> -> blob symlinks, refs/<ref>."""
    folder = cache / ("models--" + repo.replace("/", "--"))
    (folder / "refs").mkdir(parents=True, exist_ok=True)
    (folder / "refs" / ref).write_text(SHA)
    snap = folder / "snapshots" / SHA
    snap.mkdir(parents=True, exist_ok=True)
    (folder / "blobs").mkdir(exist_ok=True)
    for name, body in files.items():
        blob = folder / "blobs" / hashlib.sha256(body).hexdigest()
        blob.write_bytes(body)
        os.symlink(os.path.relpath(blob, snap), snap / name)
    return snap


def test_offline_reads_the_headers_from_the_local_hf_cache(hermetic_hf, store, monkeypatch):
    t = qwen3_8b_fp8_tensors()
    fake_cache(hermetic_hf, "org/cached", {"model.safetensors": header_bytes(t),
                                           "config.json": json.dumps(QWEN3_8B_FP8_CFG).encode()})
    set_offline(monkeypatch)
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9")  # any network use would fail
    rec = registry.call("roofline", {"model": "org/cached", "context": 1152})
    c = rec["data"]["counts"]
    assert c["method"] == "headers"
    assert c["checkpoint"]["source"] == "hub-cache" and c["checkpoint"]["commit"] == SHA
    assert rec["identity"]["model"]["revision"] == SHA and rec["data"]["config_source"] == "hub-cache"
    assert c["weight_bytes"] == pytest.approx(8.19 * GB, rel=0.01)
    assert any("local Hugging Face cache" in lim for lim in rec["limitations"])
    assert rec["data"]["request"]["args"]["revision"] == SHA
    assert rec["identity"]["model"]["dtype"] == "fp8+bf16"


def test_offline_without_cached_shards_falls_back_to_config(hermetic_hf, store, monkeypatch):
    fake_cache(hermetic_hf, "org/cfgonly", {"config.json": json.dumps(QWEN3_8B_FP8_CFG).encode()})
    set_offline(monkeypatch)
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9")
    rec = registry.call("roofline", {"model": "org/cfgonly", "dtype": "fp8", "context": 1152})
    assert rec["data"]["counts"]["method"] == "config"
    assert any("HF_HUB_OFFLINE" in lim and "config-only estimate" in lim for lim in rec["limitations"])
    assert rec["data"]["request"]["args"]["method"] == "config"  # reproduces this record, not 'auto'
    assert "--method config" in reproduce_command(rec)
    with pytest.raises(checkpoint.HubUnavailable, match="HF_HUB_OFFLINE"):
        registry.call("roofline", {"model": "org/cfgonly", "method": "headers"})


def test_unreachable_hub_falls_back_to_config_in_auto(hermetic_hf, store, monkeypatch):
    fake_cache(hermetic_hf, "org/cfgonly", {"config.json": json.dumps(QWEN3_8B_FP8_CFG).encode()})
    with socket.socket() as s:  # a port nothing listens on: connection refused
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{port}")
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # the urllib path
    with pytest.raises(checkpoint.HubUnavailable):
        checkpoint.load("org/cfgonly")
    rec = registry.call("roofline", {"model": "org/cfgonly", "dtype": "fp8", "context": 1152})
    assert rec["data"]["counts"]["method"] == "config"
    assert any("safetensors headers unavailable" in lim for lim in rec["limitations"])
    assert rec["identity"]["model"]["revision"] == SHA  # the config's cached snapshot


@pytest.mark.parametrize("quant,dtype,flagged", [
    ({"quant_method": "fp8", "weight_block_size": [128, 128]}, "fp8", False),
    ({"quant_method": "modelopt", "quant_algo": "NVFP4"}, "bf16", True),
    (None, "bf16", False)])
def test_auto_fallback_takes_the_weight_format_from_the_config(hermetic_hf, store, monkeypatch, quant, dtype, flagged):
    cfg = dict(QWEN3_8B_FP8_CFG, **({"quantization_config": quant} if quant else {}))
    fake_cache(hermetic_hf, "org/cfgonly", {"config.json": json.dumps(cfg).encode()})
    set_offline(monkeypatch)
    rec = registry.call("roofline", {"model": "org/cfgonly", "context": 1152})  # no --dtype
    assert rec["data"]["counts"]["method"] == "config"
    assert rec["data"]["counts"]["dtypes"]["weights"] == dtype
    assert any("does not model" in m for m in rec["coverage"]["missing"]) is flagged
    assert rec["data"]["request"]["args"]["dtype"] == dtype  # the reproduce command counts the same format
    explicit = registry.call("roofline", {"model": "org/cfgonly", "method": "config", "context": 1152})
    assert explicit["data"]["counts"]["dtypes"]["weights"] == "bf16"  # the --method config behaviour is kept


# --------------------------------------------------------------------------- labels and names

def test_format_labels_name_nvfp4_not_its_storage_dtypes(tmp_path):
    tensors, *_ = hybrid_tensors()
    write_safetensors(tmp_path / "model.safetensors", tensors)
    ck = checkpoint.load(str(tmp_path))
    f = tensor_formats(ck.tensors)
    lm = "lm_head"
    assert {f[f"{lm}.weight"], f[f"{lm}.weight_scale"], f[f"{lm}.weight_scale_2"], f[f"{lm}.input_scale"]} \
        == {"nvfp4"}
    a = "model.language_model.layers.3.self_attn.q_proj"
    assert f[f"{a}.weight"] == f[f"{a}.weight_scale"] == "fp8"
    assert f["model.language_model.layers.0.linear_attn.A_log"] == "fp32"
    c = header_decode_bytes(ck, HYBRID_CFG, context=10)
    assert set(c["read_bytes_by_format"]) <= {"nvfp4", "fp8", "bf16", "fp32"}
    assert sum(c["read_bytes_by_format"].values()) == pytest.approx(c["weight_bytes"])
    cfg = json.loads(json.dumps(HYBRID_CFG))
    cfg["quantization_config"]["quantized_layers"] = {"a": {"quant_algo": "FP8"}, "b": {"quant_algo": "FP8"},
                                                     "c": {"quant_algo": "NVFP4", "group_size": 16}}
    dtype, quant = _precision_label(cfg, c["read_bytes_by_format"])
    assert dtype.startswith("nvfp4+fp8") and "u8" not in dtype and "f8_e4m3" not in dtype  # (tiny model: bf16 >1%)
    assert quant == "mixed_precision(fp8+nvfp4)"
    # fused expert tensors and gpt-oss style MXFP4 blocks
    fused = {"e.gate_up_proj": checkpoint.Tensor("e.gate_up_proj", "U8", (4, 32, 16), 2048, "f"),
             "e.gate_up_proj_scale": checkpoint.Tensor("e.gate_up_proj_scale", "F8_E4M3", (4, 32, 2), 256, "f"),
             "m.down_proj_blocks": checkpoint.Tensor("m.down_proj_blocks", "U8", (4, 8, 2, 16), 1024, "f"),
             "m.down_proj_scales": checkpoint.Tensor("m.down_proj_scales", "U8", (4, 8, 2), 64, "f")}
    assert tensor_formats(fused) == {"e.gate_up_proj": "nvfp4", "e.gate_up_proj_scale": "nvfp4",
                                     "m.down_proj_blocks": "mxfp4", "m.down_proj_scales": "mxfp4"}


def test_nested_lm_head_of_a_multimodal_save_is_the_lm_head(tmp_path):
    """Older Llava / Gemma3 / Mistral3 saves put the logits matrix at language_model.lm_head.weight."""
    h, v = 32, 100
    p = "language_model.model"
    t = {f"{p}.embed_tokens.weight": ("BF16", (v, h)), "language_model.lm_head.weight": ("BF16", (v, h)),
         f"{p}.norm.weight": ("BF16", (h,)), **bf16(f"{p}.layers.0.self_attn.q_proj", h, h),
         **bf16(f"{p}.layers.0.self_attn.k_proj", 16, h), **bf16(f"{p}.layers.0.self_attn.v_proj", 16, h),
         **bf16(f"{p}.layers.0.self_attn.o_proj", h, h), **bf16(f"{p}.layers.0.mlp.gate_proj", 64, h),
         **bf16("multi_modal_projector.linear_1", h, 24)}
    write_safetensors(tmp_path / "model.safetensors", t)
    cfg = {"model_type": "llava", "text_config": {"hidden_size": h, "num_hidden_layers": 1, "num_attention_heads": 4,
                                                  "num_key_value_heads": 2, "head_dim": 8, "vocab_size": v,
                                                  "tie_word_embeddings": False}}
    c = header_decode_bytes(checkpoint.load(str(tmp_path)), cfg, context=10)
    assert c["lm_head_bytes"] == v * h * 2
    assert "other" not in c["groups"] and not c["unproven"]
