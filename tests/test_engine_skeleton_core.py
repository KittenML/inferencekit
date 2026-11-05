"""The engine skeleton's core (plugin/skills/optimizing-kernels/skeleton/ikengine): checkpoint formats, reference
quantized linears, sampler, decode loop, generic engine, dense reference and MoE blocks, all on CPU.

Needs torch and safetensors; the Hugging Face comparisons also need transformers >= 5 (skipped below 5).
Everything is skipped cleanly without them (CI has no torch). Locally, with an env that has them:
    python -m pytest -q --noconftest tests/test_engine_skeleton_core.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import save_file  # noqa: E402

SKELETON = Path(__file__).resolve().parents[1] / "plugin" / "skills" / "optimizing-kernels" / "skeleton"
sys.path.insert(0, str(SKELETON))

from ikengine import graph_loop, models, moe, ref_dense, sampler  # noqa: E402
from ikengine import qlinear as Q  # noqa: E402
from ikengine.engine import Engine, build_engine  # noqa: E402
from ikengine.engine_api import Model, SamplingParams  # noqa: E402
from ikengine.loader import Checkpoint, QWeight  # noqa: E402

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]       # the FP4 (e2m1) magnitudes, codes 0..7, by hand


# --- independent number-format references -----------------------------------------------------------------
def e4m3_table() -> list[float]:
    """Every finite float8_e4m3fn value from its bit layout (sign, 4 exponent bits with bias 7, 3 mantissa bits;
    exponent 15 with mantissa 7 is NaN), without torch."""
    vals = []
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 0xF, b & 0x7
        if e == 15 and m == 7:
            continue
        v = (m / 8) * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
        vals.append(-v if s else v)
    return vals


def nearest_even(x: float, grid: list[float], code_of) -> float:
    """Round-to-nearest on a value grid; a tie goes to the value whose code is even."""
    best = min(grid, key=lambda g: abs(g - x))
    ties = [g for g in grid if abs(abs(g - x) - abs(best - x)) == 0.0]
    if len(ties) > 1:
        best = next(g for g in ties if code_of(g) % 2 == 0)
    return best


def e2m1_ref(x: float) -> float:
    """vLLM's cast_to_fp4 thresholds, written out: saturate at 6, ties to the even code."""
    a = min(abs(x), 6.0)
    v = nearest_even(a, E2M1, E2M1.index)
    return -v if x < 0 else v


# --- e2m1 / e4m3 ---------------------------------------------------------------------------------------------
def test_e2m1_codes_decode_to_the_hand_table():
    codes = torch.arange(16, dtype=torch.uint8)
    got = Q.e2m1_decode(codes).tolist()
    assert got[:8] == E2M1 and got[8:] == [-v for v in E2M1]


def test_e2m1_ties_round_to_even_and_saturate():
    x = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0, 0.26, 0.74, -2.6, -0.25, 100.0, 4.9])
    assert Q.e2m1_round(x).tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 0.5, 0.5, -3.0, -0.0, 6.0, 4.0]


def test_e2m1_round_matches_the_threshold_rule():
    rng = random.Random(0)
    xs = [rng.uniform(-8, 8) for _ in range(2000)] + [m + d for m in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
                                                      for d in (-1e-6, 0.0, 1e-6)]
    got = Q.e2m1_round(torch.tensor(xs, dtype=torch.float32)).tolist()
    want = [e2m1_ref(float(torch.tensor(x, dtype=torch.float32))) for x in xs]
    assert got == want


def test_e2m1_pack_unpack_round_trip_low_nibble_first():
    codes = torch.randint(0, 16, (5, 64), dtype=torch.uint8)
    packed = Q.e2m1_pack(codes)
    assert packed.shape == (5, 32)
    assert int(packed[0, 0]) == int(codes[0, 0]) | (int(codes[0, 1]) << 4)
    assert torch.equal(Q.e2m1_unpack(packed), codes)
    assert torch.equal(Q.e2m1_encode(Q.e2m1_decode(codes)) & 0x7, codes & 0x7)


def test_e4m3_round_is_nearest_even_on_the_hand_table():
    table = e4m3_table()
    assert max(table) == 448.0 and min(v for v in table if v > 0) == 2.0 ** -9

    def code(v):                                       # the mantissa bits' parity decides ties
        b = torch.tensor([v]).to(torch.float8_e4m3fn).view(torch.uint8)
        return int(b)
    rng = random.Random(1)
    xs = [rng.uniform(-460, 460) for _ in range(500)] + [rng.uniform(-0.02, 0.02) for _ in range(500)]
    xs += [(table[i] + table[i + 1]) / 2 for i in range(0, len(table) - 1, 7) if table[i] * table[i + 1] > 0]
    xs = [float(torch.tensor(x, dtype=torch.float32)) for x in xs]
    got = Q.e4m3_round(torch.tensor(xs, dtype=torch.float32)).tolist()
    want = [nearest_even(max(-448.0, min(448.0, x)), table, code) for x in xs]
    assert got == want


# --- activation quantizers ------------------------------------------------------------------------------------
def test_nvfp4_activation_known_vector():
    # input_scale 0.5 -> global scale 2; amax 12 -> block scale e4m3(2 * 12 / 6) = 4; x * 1/(4 * 0.5) on e2m1
    x = torch.tensor([[12.0, 5.0, 3.0, -2.5, 0.5, 1.5, 7.0] + [0.0] * 9])
    xq, s = Q.quant_act_nvfp4(x, torch.tensor(0.5))
    assert s.tolist() == [[4.0]]
    assert xq[0, :7].tolist() == [6.0, 2.0, 1.5, -1.0, 0.0, 1.0, 4.0]
    assert (xq * s.repeat_interleave(16, -1) * 0.5)[0, :7].tolist() == [12.0, 4.0, 3.0, -2.0, 0.0, 2.0, 8.0]


def test_nvfp4_activation_block_scale_is_e4m3_and_values_fit():
    x = torch.randn(4, 64) * 3
    xq, s = Q.quant_act_nvfp4(x, torch.tensor(0.01))
    assert torch.equal(s, s.to(torch.float8_e4m3fn).float())
    assert torch.equal(xq, Q.e2m1_round(xq)) and xq.abs().max() <= 6.0
    rel = ((xq * s.repeat_interleave(16, -1) * 0.01) - x).abs().max() / x.abs().max()
    assert rel < 0.2


def test_fp8_group_quant_scale_is_amax_over_448():
    x = torch.randn(3, 300)
    xq, s = Q.quant_act_fp8_group(x, 128)
    assert s.shape == (3, 3)
    for g in range(3):
        blk = x[:, g * 128:(g + 1) * 128]
        assert torch.equal(s[:, g], blk.abs().amax(-1) / 448.0)
    assert torch.equal(xq, Q.e4m3_round(xq)) and xq.abs().max() == 448.0


def test_fp8_tensor_activation_static_and_dynamic():
    xq, s = Q.quant_act_fp8_tensor(torch.tensor([[1000.0, 0.55, -1.0]]), torch.tensor(0.5))
    assert xq.tolist() == [[448.0, 1.125, -2.0]] and float(s) == 0.5        # saturate; 1.1 -> 1.125 on e4m3
    x = torch.tensor([[3.0, -896.0], [1.0, 0.0]])
    xq, s = Q.quant_act_fp8_tensor(x)                                        # one dynamic amax/448 scale
    assert float(s) == 2.0 and xq.tolist() == [[1.5, -448.0], [0.5, 0.0]]


# --- weights and products ---------------------------------------------------------------------------------------
def _nvfp4(n=8, k=64, seed=0, input_scale=0.02):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (n, k), dtype=torch.uint8, generator=g)
    scale = (torch.rand(n, k // 16, generator=g) * 8 + 0.5).to(torch.float8_e4m3fn)
    return QWeight("nvfp4", Q.e2m1_pack(codes), (n, k), scale=scale, scale_2=torch.tensor(0.003),
                   input_scale=torch.tensor(input_scale), block=(1, 16)), codes


def _fp8_block(n=96, k=200, bn=32, bk=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(n, k, generator=g) * 100).to(torch.float8_e4m3fn)
    s = torch.rand(-(-n // bn), -(-k // bk), generator=g) * 0.01 + 1e-3
    return QWeight("fp8_block", w, (n, k), scale=s, block=(bn, bk))


def test_nvfp4_dequant_is_value_times_block_scale_times_scale_2():
    w, codes = _nvfp4()
    d = Q.dequant(w)
    for r, c in [(0, 0), (3, 17), (7, 63), (5, 32)]:
        code = int(codes[r, c])
        val = (-1 if code & 8 else 1) * E2M1[code & 7]
        assert d[r, c].item() == pytest.approx(val * float(w.scale[r, c // 16].float()) * 0.003, rel=1e-6)


def test_fp8_block_dequant_uses_each_block_scale():
    w = _fp8_block()
    d = Q.dequant(w)
    for r, c in [(0, 0), (40, 70), (95, 199), (63, 128)]:
        assert d[r, c].item() == pytest.approx(float(w.weight[r, c].float()) * float(w.scale[r // 32, c // 64]), rel=1e-6)


def test_qlinear_nvfp4_w4a4_is_the_quantized_product_times_alpha():
    w, _ = _nvfp4()
    x = torch.randn(3, 64, dtype=torch.bfloat16)
    xq, xs = Q.quant_act_nvfp4(x, w.input_scale)
    xd = (xq.double().reshape(3, 4, 16) * xs.double().unsqueeze(-1)).reshape(3, 64)
    wd = (Q.e2m1_decode(Q.e2m1_unpack(w.weight)).double().reshape(8, 4, 16) * w.scale.double().unsqueeze(-1)).reshape(8, 64)
    want = (xd @ wd.t()) * (0.02 * 0.003)
    assert torch.allclose(Q.qlinear(x, w).double(), want.to(torch.bfloat16).double(), rtol=1e-2, atol=1e-6)
    a16 = Q.qlinear(x, w, Q.ACT_A16).double()
    assert torch.allclose(a16, (x.double() @ Q.dequant(w).double().t()).to(torch.bfloat16).double(), rtol=1e-2)
    assert not torch.equal(a16, Q.qlinear(x, w).double())


def test_qlinear_fp8_tensor_known_vector_with_row_scales():
    # ModelOpt FP8: x_q = e4m3(x / input_scale); y = (x_q @ w_q^T) * weight_scale (per row here) * input_scale
    wq = torch.tensor([[1.0, 2.0, 0.0, -1.0], [0.5, 0.5, 0.5, 0.5], [4.0, 0.0, 0.0, 1.0]]).to(torch.float8_e4m3fn)
    w = QWeight("fp8_tensor", wq, (3, 4), scale=torch.tensor([0.1, 2.0, 0.25]), input_scale=torch.tensor(0.5))
    x = torch.tensor([[1.0, -2.0, 0.55, 3.0]])                               # x / 0.5 = [2, -4, 1.1 -> 1.125, 6]
    y = Q.qlinear(x, w)
    assert y[0].tolist() == pytest.approx([-12 * 0.1 * 0.5, 2.5625 * 2.0 * 0.5, 14 * 0.25 * 0.5], rel=1e-6)
    assert Q.dequant(w)[1].tolist() == [1.0, 1.0, 1.0, 1.0] and Q.dequant(w)[2, 0].item() == 1.0
    a16 = Q.qlinear(x, w, Q.ACT_A16)                                          # 16-bit activations: 1.1, not 1.125
    assert a16[0, 1].item() == pytest.approx((1.0 - 2.0 + 0.55 + 3.0) * 0.5 * 2.0, rel=1e-6)
    one = QWeight("fp8_tensor", wq, (3, 4), scale=torch.tensor(0.1), input_scale=torch.tensor(0.5))
    assert Q.qlinear(x, one)[0, 2].item() == pytest.approx(14 * 0.1 * 0.5, rel=1e-6)


def test_qlinear_fp8_block_sums_block_products_in_fp32():
    w = _fp8_block()
    x = torch.randn(4, 200, dtype=torch.float32)
    y = Q.qlinear(x, w)
    xq, xs = Q.quant_act_fp8_group(x, 64)
    want = torch.zeros(4, 96, dtype=torch.float64)
    for kb in range(4):
        ks = slice(kb * 64, min(200, (kb + 1) * 64))
        part = xq[:, ks].double() @ w.weight[:, ks].float().double().t()             # [4, 96]
        ws = w.scale[:, kb].double().repeat_interleave(32)[:96]
        want += part * xs[:, kb:kb + 1].double() * ws
    assert torch.allclose(y.double(), want, rtol=1e-5, atol=1e-4)
    with pytest.raises(ValueError):
        Q.qlinear(x, w, act="w4")


@pytest.mark.parametrize("kind", ["nvfp4", "fp8_block", "bf16"])
def test_qlinear_stacked_rows_use_their_own_weight(kind):
    if kind == "nvfp4":
        ws = [_nvfp4(seed=s, input_scale=0.01 * (s + 1))[0] for s in range(3)]
    elif kind == "fp8_block":
        ws = [_fp8_block(n=64, k=128, seed=s) for s in range(3)]
    else:
        ws = [QWeight("bf16", torch.randn(16, 32).to(torch.bfloat16), (16, 32)) for _ in range(3)]
    stack = moe.ExpertStack.from_linears(ws)
    ids = torch.tensor([2, 0, 1, 2])
    k = ws[0].shape[1]
    x = torch.randn(4, k, dtype=torch.bfloat16 if kind == "bf16" else torch.float32)
    y = Q.qlinear(x, stack.gather(ids)).float()
    tol = dict(rtol=1e-2, atol=1e-2) if kind == "bf16" else dict(rtol=1e-4, atol=1e-3)
    for r, e in enumerate(ids.tolist()):
        assert torch.allclose(y[r], Q.qlinear(x[r:r + 1], ws[e])[0].float(), **tol)


def test_kv_fp8_quant_round_trip():
    x = torch.randn(2, 5, 8) * 3
    q = Q.kv_quant(x, torch.tensor(0.5))
    assert q.dtype == torch.float8_e4m3fn
    assert torch.allclose(Q.kv_dequant(q, torch.tensor(0.5)), x, rtol=0.07, atol=1e-2)


# --- checkpoint reading -----------------------------------------------------------------------------------------
def _write(d: Path, config: dict, tensors: dict | None = None, hq: dict | None = None):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(config))
    if hq is not None:
        (d / "hf_quant_config.json").write_text(json.dumps(hq))
    if tensors:
        save_file(tensors, str(d / "model.safetensors"))
    return d


def test_scheme_modelopt_nvfp4_from_hf_quant_config(tmp_path):
    hq = {"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8",
                                                              "group_size": 16, "exclude_modules": ["lm_head", "model.layers.0.mlp*"]}}
    ck = Checkpoint(str(_write(tmp_path, {"architectures": ["X"]}, hq=hq)))
    assert ck.quant["method"] == "nvfp4" and ck.quant["act"] == "nvfp4_static" and ck.quant["kv"] == "fp8"
    assert ck.is_excluded("lm_head") and ck.is_excluded("model.layers.0.mlp.gate_proj")
    assert not ck.is_excluded("model.layers.1.mlp.gate_proj")
    assert ck.layer_algo("model.layers.1.experts.0.up_proj") == "nvfp4" and ck.layer_algo("lm_head") == "none"


def test_scheme_config_groups_as_the_gemma4_nvfp4_export(tmp_path):
    grp = {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16}
    qc = {"config_groups": {"group_0": {"input_activations": grp, "weights": grp, "targets": ["Linear"]}},
          "ignore": ["lm_head", "model.language_model.layers.0.self_attn*"], "quant_algo": "NVFP4",
          "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"}, "quant_method": "modelopt"}
    ck = Checkpoint(str(_write(tmp_path, {"architectures": ["X"], "quantization_config": qc})))
    assert (ck.quant["method"], ck.quant["act"], ck.quant["kv"]) == ("nvfp4", "nvfp4_static", "fp8")
    assert ck.is_excluded("model.language_model.layers.0.self_attn.q_proj")
    qc2 = {**qc, "quant_algo": None, "config_groups": {"g": {"weights": grp, "targets": ["Linear"]}}}
    ck2 = Checkpoint(str(_write(tmp_path / "w4a16", {"quantization_config": qc2})))
    assert ck2.quant["method"] == "nvfp4" and ck2.quant["act"] is None


def test_scheme_modelopt_keys_from_config_json_only(tmp_path):
    # An export with no hf_quant_config.json: ModelOpt's own key names sit in config.json's quantization_config.
    qc = {"quant_method": "modelopt", "quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8", "group_size": 16,
          "exclude_modules": ["lm_head", "model.layers.0.self_attn*"]}
    ck = Checkpoint(str(_write(tmp_path / "a", {"quantization_config": qc})))
    assert (ck.quant["method"], ck.quant["act"], ck.quant["kv"]) == ("nvfp4", "nvfp4_static", "fp8")
    assert ck.is_excluded("lm_head") and ck.is_excluded("model.layers.0.self_attn.k_proj")
    with pytest.raises(NotImplementedError):
        Checkpoint(str(_write(tmp_path / "b", {"quantization_config": {**qc, "group_size": 32}})))
    no_kv = {k: v for k, v in qc.items() if k != "kv_cache_quant_algo"}
    assert Checkpoint(str(_write(tmp_path / "c", {"quantization_config": no_kv}))).quant["kv"] is None


def test_kv_scales_single_and_one_sided(tmp_path):
    tensors = {"a.kv_scale": torch.tensor(0.3), "b.k_proj.k_scale": torch.tensor(0.2),
               "c.attn.v_scale": torch.tensor(0.7), "d.k_proj.kv_scale": torch.tensor(0.4),
               "e.k_scale": torch.tensor(0.1), "e.v_scale": torch.tensor(0.9), "e.kv_scale": torch.tensor(5.0),
               "w.weight": torch.zeros(2, 2)}
    ck = Checkpoint(str(_write(tmp_path, {}, tensors)))
    got = {p: tuple(round(float(t), 6) for t in ck.kv_scales(p)) for p in "abcde"}
    assert got == {"a": (0.3, 0.3), "b": (0.2, 0.2), "c": (0.7, 0.7), "d": (0.4, 0.4), "e": (0.1, 0.9)}
    assert ck.kv_scales("w") == (None, None)


def test_mixed_precision_layers_take_their_own_kind(tmp_path):
    w4, _ = _nvfp4(n=8, k=32)
    hq = {"producer": {"name": "modelopt"},
          "quantization": {"quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": None,
                           "quantized_layers": {"m.a": {"quant_algo": "NVFP4"}, "m.b": {"quant_algo": "FP8"}}}}
    wq = (torch.randn(4, 32) * 50).to(torch.float8_e4m3fn)
    tensors = {"m.a.weight": w4.weight, "m.a.weight_scale": w4.scale, "m.a.weight_scale_2": w4.scale_2,
               "m.a.input_scale": w4.input_scale, "m.b.weight": wq, "m.b.weight_scale": torch.tensor(0.01),
               "m.b.input_scale": torch.tensor(0.05), "m.c.weight": torch.randn(4, 32).to(torch.bfloat16)}
    ck = Checkpoint(str(_write(tmp_path / "mix", {}, tensors, hq=hq)))
    assert (ck.quant["method"], ck.quant["act"], ck.quant["kv"]) == ("mixed", "per layer", None)
    assert [ck.layer_algo(p) for p in ("m.a", "m.b", "m.c")] == ["nvfp4", "fp8_tensor", "none"]
    a, b, c = ck.linear("m.a"), ck.linear("m.b"), ck.linear("m.c")
    assert (a.kind, b.kind, c.kind) == ("nvfp4", "fp8_tensor", "bf16")
    x = torch.randn(2, 32)
    xq = Q.e4m3_round(x / torch.tensor(0.05))
    assert torch.allclose(Q.qlinear(x, b), (xq @ wq.float().t()) * 0.01 * 0.05, rtol=1e-5, atol=1e-6)
    assert torch.allclose(Q.qlinear(x, a), Q.qlinear(x, w4)) and not torch.allclose(Q.qlinear(x, a), Q.qlinear(x, a, "a16"))
    bad = {**hq, "quantization": {**hq["quantization"], "quantized_layers": {"m.a": {"quant_algo": "W4A8_AWQ"}}}}
    with pytest.raises(NotImplementedError):
        Checkpoint(str(_write(tmp_path / "bad", {}, tensors, hq=bad)))


def test_scheme_block_fp8_and_refusals(tmp_path):
    qc = {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128],
          "modules_to_not_convert": ["lm_head", "model.layers.0.linear_attn.in_proj_a"]}
    ck = Checkpoint(str(_write(tmp_path / "a", {"quantization_config": qc})))
    assert ck.quant["method"] == "fp8_block" and ck.quant["block"] == [128, 128]
    assert ck.quant["act"] == "fp8_dynamic_group" and ck.is_excluded("model.layers.0.linear_attn.in_proj_a")
    assert Checkpoint(str(_write(tmp_path / "b", {}))).quant["method"] == "none"
    with pytest.raises(NotImplementedError):
        Checkpoint(str(_write(tmp_path / "c", {"quantization_config": {"quant_method": "compressed-tensors"}})))
    with pytest.raises(NotImplementedError):
        Checkpoint(str(_write(tmp_path / "d", {"quantization_config": {"quant_method": "gptq"}})))


def test_linear_kinds_follow_the_stored_tensors_and_keep_values(tmp_path):
    w4, _ = _nvfp4()
    w8 = _fp8_block(n=64, k=128, bn=128, bk=128)
    tensors = {"a.weight": w4.weight, "a.weight_scale": w4.scale, "a.weight_scale_2": w4.scale_2,
               "a.input_scale": w4.input_scale, "b.weight": w8.weight, "b.weight_scale_inv": w8.scale,
               "c.weight": torch.randn(4, 8).to(torch.bfloat16), "c.bias": torch.randn(4).to(torch.bfloat16),
               "attn.k_proj.k_scale": torch.tensor(0.25), "attn.v_scale": torch.tensor(0.5)}
    qc = {"quant_method": "fp8", "weight_block_size": [128, 128], "activation_scheme": "dynamic"}
    ck = Checkpoint(str(_write(tmp_path, {"quantization_config": qc}, tensors)))
    a, b, c = ck.linear("a"), ck.linear("b"), ck.linear("c")
    assert (a.kind, a.shape, b.kind, b.block, c.kind) == ("nvfp4", (8, 64), "fp8_block", (128, 128), "bf16")
    assert torch.equal(a.weight, w4.weight) and torch.equal(a.scale.view(torch.uint8), w4.scale.view(torch.uint8))
    assert torch.equal(b.weight.view(torch.uint8), w8.weight.view(torch.uint8)) and torch.equal(b.scale, w8.scale)
    assert c.bias is not None and ck.linear("c", bias=False).bias is None
    ks, vs = ck.kv_scales("attn")
    assert float(ks) == 0.25 and float(vs) == 0.5 and ck.kv_scales("other") == (None, None)


# --- sampler ------------------------------------------------------------------------------------------------------
def _params(t=1.0, k=0, p=1.0, m=0.0):
    return (torch.tensor([t]), torch.tensor([k]), torch.tensor([p]), torch.tensor([m]))


def test_greedy_ties_go_to_the_lowest_id():
    assert sampler.greedy(torch.tensor([[1.0, 3.0, 3.0, 2.0]])).item() == 1
    assert sampler.is_greedy(0.0) and sampler.is_greedy(1e-6) and not sampler.is_greedy(0.1)


def test_top_k_keeps_boundary_ties_and_top_p_keeps_the_best():
    logits = torch.tensor([[0.0, 2.0, 1.0, 2.0, -1.0]])
    kept = sampler.filter_logits(logits, *_params(k=1)).isfinite()[0].tolist()
    assert kept == [False, True, False, True, False]
    kept = sampler.filter_logits(logits, *_params(p=1e-6)).isfinite()[0].tolist()
    assert sum(kept) == 1
    m = sampler.filter_logits(torch.tensor([[math.log(0.5), math.log(0.3), math.log(0.2)]]), *_params(m=0.5))
    assert m.isfinite()[0].tolist() == [True, True, False]


def test_filter_matches_hugging_face_warpers():
    tr = pytest.importorskip("transformers")
    from transformers.generation import logits_process as lp
    g = torch.Generator().manual_seed(3)
    for t, k, p, m in [(0.7, 20, 0.95, 0.0), (1.0, 64, 0.9, 0.05), (1.3, 0, 0.8, 0.0), (1.0, 5, 1.0, 0.1)]:
        logits = torch.randn(1, 300, generator=g) * 3
        want = lp.TemperatureLogitsWarper(t)(None, logits.clone())
        if k:
            want = lp.TopKLogitsWarper(k)(None, want)
        if p < 1.0:
            want = lp.TopPLogitsWarper(p)(None, want)
        if m:
            want = lp.MinPLogitsWarper(m)(None, want)
        got = sampler.filter_logits(logits, *_params(t, k, p, m))
        assert torch.equal(got.isfinite(), want.isfinite()), (t, k, p, m, tr.__version__)
        assert torch.allclose(got[got.isfinite()], want[want.isfinite()])


def test_sampling_frequencies_match_the_filtered_distribution():
    logits = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -1.0, -3.0, -5.0])
    params = sampler.DeviceParams.alloc(8, "cpu")
    params.set(SamplingParams(temperature=0.8, top_k=6, top_p=0.95))
    want = sampler.filter_logits(logits.reshape(1, -1), params.temperature, params.top_k, params.top_p,
                                 params.min_p).softmax(-1)[0]
    gen = sampler.make_generator("cpu", 7)
    counts = torch.zeros(8)
    n = 20000
    for _ in range(n):
        params.refill_noise(gen)
        counts[int(params.choose(logits))] += 1
    assert (counts / n - want).abs().max() < 0.015
    assert counts[want == 0].sum() == 0


def test_seeded_draws_repeat_and_logprobs_are_raw():
    logits = torch.randn(50)
    params = sampler.DeviceParams.alloc(50, "cpu")
    params.set(SamplingParams(temperature=1.0, top_k=20, top_p=0.95))

    def run(seed):
        g = sampler.make_generator("cpu", seed)
        out = []
        for _ in range(30):
            params.refill_noise(g)
            out.append(int(params.choose(logits)))
        return out
    assert run(11) == run(11) and run(11) != run(12)
    chosen, ids, vals = sampler.logprobs(logits, torch.tensor([4]), 5)
    full = torch.log_softmax(logits, -1)
    assert chosen.item() == pytest.approx(full[4].item())
    assert ids.tolist() == full.topk(5).indices.tolist() and vals.tolist() == sorted(vals.tolist(), reverse=True)
    assert sampler.logprobs(logits, torch.tensor([4]), 0)[1].numel() == 0


# --- decode loop and engine over a fake model ------------------------------------------------------------------
class CountModel:
    """Logits favour token (last + 1) % V, with position-dependent noise; counts its calls. Implements Model."""

    def __init__(self, vocab=32, max_seq=64, boost=20.0):
        self.vocab_size, self.max_seq, self.boost = vocab, max_seq, boost
        self.eos_token_ids = frozenset({0})
        self.numerics = {"weights": "fp32 as stored"}
        self.calls = {"prefill": 0, "decode": 0, "reset": 0}

    def alloc_state(self, max_seq):
        return {"logits": torch.zeros(self.vocab_size)}

    def reset(self, state):
        self.calls["reset"] += 1

    def _logits(self, tok, pos, state):
        g = torch.Generator().manual_seed(int(pos) * 1000 + int(tok))
        out = torch.randn(self.vocab_size, generator=g)
        out[(int(tok) + 1) % self.vocab_size] += self.boost
        state["logits"].copy_(out)
        return state["logits"]

    def prefill(self, input_ids, state):
        self.calls["prefill"] += 1
        return self._logits(input_ids[-1], input_ids.shape[0] - 1, state)

    def decode(self, token, pos, state):
        self.calls["decode"] += 1
        return self._logits(token[0], pos[0], state)


def test_fake_model_meets_the_protocol():
    assert isinstance(CountModel(), Model)


def test_engine_greedy_stream_and_logprobs_only_when_asked():
    m = CountModel()
    eng = Engine(m, device="cpu")
    info = {}
    outs = list(eng.stream([3, 4, 5], 6, SamplingParams(temperature=0.0), None, info))
    assert [o.token_id for o in outs] == [6, 7, 8, 9, 10, 11]
    assert all(o.logprob is None and o.top == [] for o in outs)
    assert info["decode"] == "eager" and "prefill_ms" in info
    assert m.calls["prefill"] == 1 and m.calls["decode"] == 5
    outs = list(eng.stream([3, 4, 5], 3, SamplingParams(temperature=0.0), 20, {}))
    assert len(outs[0].top) == 20 and outs[0].top[0][0] == outs[0].token_id
    assert outs[0].logprob == pytest.approx(outs[0].top[0][1])
    assert all(a[1] >= b[1] for a, b in zip(outs[1].top, outs[1].top[1:]))
    assert list(eng.stream([3], 2, SamplingParams(temperature=0.0), 0, {}))[0].top == []


@pytest.mark.parametrize("sp", [SamplingParams(temperature=0.0), SamplingParams(temperature=1.2, top_k=8, seed=3)])
def test_engine_lookahead_ring_gives_the_same_stream(sp):
    def run(lookahead, k=5, tokens=11):
        return [(o.token_id, o.logprob, o.top) for o in
                Engine(CountModel(boost=1.0), device="cpu", lookahead=lookahead).stream([4, 9], tokens, sp, k, {})]
    base = run(0)
    assert len(base) == 11 and len(base[3][2]) == 5
    for la in (1, 2, 5):
        assert run(la) == base, la
    assert run(2, k=None) == [(t, None, []) for t, _, _ in base]
    m = CountModel(boost=1.0)
    eng = Engine(m, device="cpu", lookahead=2)
    info = {}
    gen = eng.stream([4, 9], 11, sp, 5, info)
    first = next(gen)                       # prefill's token comes before any decode step is launched
    assert m.calls["decode"] == 0 and "prefill_ms" in info and (first.token_id, first.logprob, first.top) == base[0]
    next(gen)
    assert m.calls["decode"] == 3           # step 1 and two more in flight
    gen.close()
    assert [(o.token_id, o.logprob, o.top) for o in eng.stream([4, 9], 11, sp, 5, {})] == base


def test_engine_sampling_is_seeded_and_close_is_safe():
    eng = Engine(CountModel(boost=0.5), device="cpu")
    sp = SamplingParams(temperature=1.5, top_k=0, top_p=1.0, seed=5)
    a = [o.token_id for o in eng.stream([1, 2], 12, sp, None, {})]
    b = [o.token_id for o in eng.stream([1, 2], 12, sp, None, {})]
    c = [o.token_id for o in eng.stream([1, 2], 12, SamplingParams(temperature=1.5, seed=6), None, {})]
    assert a == b and a != c
    gen = eng.stream([1, 2], 10, sp, 5, {})
    next(gen)
    gen.close()                                     # the server stops early
    assert [o.token_id for o in eng.stream([1, 2], 12, sp, None, {})] == a


def test_engine_caps_at_max_seq_and_refuses_overlap():
    eng = Engine(CountModel(max_seq=8), device="cpu")
    info = {}
    assert len(list(eng.stream([1, 2, 3, 4, 5], 10, SamplingParams(temperature=0.0), None, info))) == 3
    assert info["max_seq_capped"] == 3
    with pytest.raises(ValueError):
        list(eng.stream(list(range(8)), 1, SamplingParams(temperature=0.0), None, {}))
    g1 = eng.stream([1], 3, SamplingParams(temperature=0.0), None, {})
    next(g1)
    with pytest.raises(RuntimeError):
        list(eng.stream([1], 3, SamplingParams(temperature=0.0), None, {}))
    g1.close()
    assert eng.config()["cuda_graphs"] is False


def test_pipelined_keeps_lookahead_steps_in_flight():
    log = []
    out = list(graph_loop.pipelined(5, 2, lambda i: log.append(("L", i)), lambda i: log.append(("C", i)) or i))
    assert out == [0, 1, 2, 3, 4]
    assert log[:6] == [("L", 0), ("C", 0), ("L", 1), ("L", 2), ("L", 3), ("C", 1)]   # token 0 waits for nothing
    assert log.count(("L", 4)) == 1 and log.index(("L", 4)) < log.index(("C", 2))
    log.clear()
    assert list(graph_loop.pipelined(3, 0, lambda i: log.append(("L", i)), lambda i: log.append(("C", i)) or i)) == [0, 1, 2]
    assert log == [("L", 0), ("C", 0), ("L", 1), ("C", 1), ("L", 2), ("C", 2)]


# --- MoE blocks ------------------------------------------------------------------------------------------------------
def test_route_softmax_and_sigmoid():
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.0]])
    w, ids = moe.route(logits, 2)
    p = logits.softmax(-1)[0]
    assert ids.tolist() == [[1, 2]]
    assert torch.allclose(w, torch.stack([p[1], p[2]]).unsqueeze(0) / (p[1] + p[2]))
    w, ids = moe.route(logits, 2, renormalize=False, per_expert_scale=torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(w, torch.stack([p[1] * 2, p[2] * 3]).unsqueeze(0))
    w, ids = moe.route(logits, 2, scoring="sigmoid", correction_bias=torch.tensor([0.0, 0.0, 0.0, 5.0]))
    s = logits.sigmoid()[0]
    assert ids.tolist() == [[3, 1]] and torch.allclose(w, torch.stack([s[3], s[1]]).unsqueeze(0) / (s[3] + s[1]))


def test_expert_mlp_gather_and_loop_paths_agree_with_a_naive_loop():
    g = torch.Generator().manual_seed(0)
    e, h, i = 6, 32, 48
    gate = [QWeight("fp32", torch.randn(i, h, generator=g) / 6, (i, h)) for _ in range(e)]
    up = [QWeight("fp32", torch.randn(i, h, generator=g) / 6, (i, h)) for _ in range(e)]
    down = [QWeight("fp32", torch.randn(h, i, generator=g) / 7, (h, i)) for _ in range(e)]
    stacks = [moe.ExpertStack.from_linears(s) for s in (gate, up, down)]
    x = torch.randn(5, h, generator=g)
    w, ids = moe.route(torch.randn(5, e, generator=g), 2)
    silu = torch.nn.functional.silu
    naive = torch.zeros(5, h)
    for t in range(5):
        for j in range(2):
            k = int(ids[t, j])
            naive[t] += w[t, j] * (silu(x[t] @ gate[k].weight.t()) * (x[t] @ up[k].weight.t())) @ down[k].weight.t()
    a = moe.expert_mlp(x, w, ids, *stacks, silu)
    b = moe.expert_mlp(x, w, ids, *stacks, silu, gather_max_rows=0)
    assert torch.allclose(a, naive, atol=1e-5) and torch.allclose(b, naive, atol=1e-5)
    with pytest.raises(ValueError):
        moe.ExpertStack.from_linears([gate[0], QWeight("fp32", torch.zeros(i, h), (i, h), bias=torch.zeros(i))])


# --- dense reference vs Hugging Face ------------------------------------------------------------------------------
TINY = dict(vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, max_position_embeddings=256, rms_norm_eps=1e-6)

CASES = {
    "qwen3": ("Qwen3Config", "Qwen3ForCausalLM", dict(tie_word_embeddings=True)),
    "llama-llama3-rope": ("LlamaConfig", "LlamaForCausalLM", dict(rope_parameters={
        "rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0, "low_freq_factor": 1.0,
        "high_freq_factor": 4.0, "original_max_position_embeddings": 64})),
    "llama-yarn": ("LlamaConfig", "LlamaForCausalLM", dict(rope_parameters={
        "rope_type": "yarn", "rope_theta": 10000.0, "factor": 4.0, "original_max_position_embeddings": 64},
        attention_bias=True)),
    "llama-linear": ("LlamaConfig", "LlamaForCausalLM", dict(rope_parameters={
        "rope_type": "linear", "rope_theta": 10000.0, "factor": 2.0}, mlp_bias=True)),
    "qwen2-window": ("Qwen2Config", "Qwen2ForCausalLM", dict(use_sliding_window=True, sliding_window=5,
                                                              max_window_layers=1)),
    "mistral-window": ("MistralConfig", "MistralForCausalLM", dict(sliding_window=6)),
}


def _transformers_5():
    """transformers 5 or later, else skip: the Hugging Face comparisons fail with 4.57.3 and pass with 5.12.1 (CPU),
    so below 5 a failure would say nothing about the skeleton."""
    transformers = pytest.importorskip("transformers")
    if int(str(transformers.__version__).split(".")[0]) < 5:
        pytest.skip(f"needs transformers >= 5 for the Hugging Face comparisons (found {transformers.__version__})")
    return transformers


def _hf_tiny(tmp_path, case, dtype=torch.float32):
    transformers = _transformers_5()
    cfg_name, arch, extra = CASES[case]
    cfg = getattr(transformers, cfg_name)(**TINY, **extra)
    torch.manual_seed(0)
    model = getattr(transformers, arch)(cfg).eval().to(torch.float32)
    with torch.no_grad():                           # larger weights, and norms and biases away from 1 and 0
        for name, p in model.named_parameters():
            p.add_(torch.randn_like(p) * (0.1 if ("norm" in name or name.endswith("bias")) else 0.08))
    model = model.to(dtype)                         # saved in this dtype; config.json records it
    d = tmp_path / f"{case}-{str(dtype).replace('torch.', '')}"
    model.save_pretrained(str(d), safe_serialization=True)
    return model, d


@pytest.mark.parametrize("case", sorted(CASES))
def test_ref_dense_matches_hugging_face(tmp_path, case):
    hf, d = _hf_tiny(tmp_path, case)
    ck = Checkpoint(str(d))
    model = models.model_class(ck.config["architectures"][0]).from_checkpoint(ck, device="cpu", max_seq=48)
    assert isinstance(model, ref_dense.DenseDecoder) and isinstance(model, Model)
    state = model.alloc_state(48)
    model.reset(state)
    ids = torch.randint(0, 96, (12,), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        logits = model.prefill(ids, state).clone()
        seq = ids.tolist()
        for step in range(9):
            want = hf(torch.tensor([seq])).logits[0, -1].float()
            assert torch.allclose(logits, want, atol=2e-4, rtol=1e-4), (case, step, (logits - want).abs().max())
            nxt = int(logits.argmax())
            logits = model.decode(torch.tensor([nxt]), torch.tensor([len(seq)], dtype=torch.int32), state).clone()
            seq.append(nxt)


@pytest.mark.parametrize("case", sorted(CASES))
def test_ref_dense_bf16_matches_hugging_face(tmp_path, case):
    # The path real checkpoints take: bf16 weights, bf16 RoPE tables, bf16 rounding of each block's output.
    # Hugging Face computes attention in bf16 and the reference in fp32, so logits agree only to `tol`. Both
    # follow Hugging Face's greedy path; the greedy choice must match wherever its top-2 margin exceeds 2 * tol
    # (random tiny weights give near-ties that bf16 noise may flip; real ones are no different there).
    tol = 0.05
    hf, d = _hf_tiny(tmp_path, case, torch.bfloat16)
    ck = Checkpoint(str(d))
    assert ck.dtype_name == "bfloat16"
    model = ref_dense.DenseDecoder.from_checkpoint(ck, device="cpu", max_seq=48)
    assert model.dtype == torch.bfloat16 and model.layers[0].q.weight.dtype == torch.bfloat16
    state = model.alloc_state(48)
    model.reset(state)
    ids = torch.randint(0, 96, (12,), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        logits = model.prefill(ids, state).clone()
        seq = ids.tolist()
        clear = 0
        for step in range(9):
            want = hf(torch.tensor([seq])).logits[0, -1].float()
            assert torch.allclose(logits, want, atol=tol), (case, step, (logits - want).abs().max())
            top2 = want.topk(2).values
            if float(top2[0] - top2[1]) > 2 * tol:
                clear += 1
                assert int(logits.argmax()) == int(want.argmax()), (case, step)
            nxt = int(want.argmax())
            logits = model.decode(torch.tensor([nxt]), torch.tensor([len(seq)], dtype=torch.int32), state).clone()
            seq.append(nxt)
        assert clear >= 3, (case, clear)


def test_build_engine_greedy_matches_hugging_face_generate(tmp_path):
    hf, d = _hf_tiny(tmp_path, "qwen3")
    eng = build_engine(str(d), device="cpu", max_seq=64, warmup=False)
    prompt = [5, 17, 33, 2, 9, 61]
    got = [o.token_id for o in eng.stream(prompt, 10, SamplingParams(temperature=0.0), None, {})]
    seq = list(prompt)
    with torch.no_grad():
        for _ in range(10):
            seq.append(int(hf(torch.tensor([seq])).logits[0, -1].argmax()))
    assert got == seq[len(prompt):]
    assert eng.config()["numerics"]["weights"] == "16-bit as stored"


def _quantize_fp8_block(w, bn, bk):
    """A test-side block-FP8 export: e4m3 values with fp32 weight_scale_inv = amax/448 per (bn, bk) block."""
    n, k = w.shape
    s = torch.zeros(-(-n // bn), -(-k // bk))
    q = torch.zeros(n, k)
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            blk = w[i * bn:(i + 1) * bn, j * bk:(j + 1) * bk]
            s[i, j] = blk.abs().amax().clamp(min=1e-12) / 448.0
            q[i * bn:(i + 1) * bn, j * bk:(j + 1) * bk] = blk / s[i, j]
    return q.to(torch.float8_e4m3fn), s


def test_ref_dense_reads_a_block_fp8_checkpoint(tmp_path):
    hf, d = _hf_tiny(tmp_path, "qwen3")
    from safetensors.torch import load_file
    sd = load_file(str(d / "model.safetensors"))
    out = {}
    for name, t in sd.items():
        if name.endswith("_proj.weight"):
            q, s = _quantize_fp8_block(t, 32, 32)
            out[name], out[name[:-len("weight")] + "weight_scale_inv"] = q, s
        else:
            out[name] = t
    cfg = json.loads((d / "config.json").read_text())
    cfg["quantization_config"] = {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [32, 32]}
    q_dir = _write(tmp_path / "fp8", cfg, out)
    ck = Checkpoint(str(q_dir))
    shipped = ref_dense.DenseDecoder.from_checkpoint(ck, device="cpu", max_seq=32)
    a16 = ref_dense.DenseDecoder.from_checkpoint(ck, device="cpu", max_seq=32, act="a16")
    assert shipped.numerics["activations"].startswith("fp8 e4m3, dynamic per token in groups of 32")
    assert a16.numerics["activations"].startswith("16-bit (changed")
    assert (shipped.numerics["as_shipped"], shipped.numerics["changed"]) == (True, [])
    assert a16.numerics["as_shipped"] is False and a16.numerics["changed"][0].startswith("activations: 16-bit")
    kv16 = ref_dense.DenseDecoder.from_checkpoint(ck, device="cpu", max_seq=32, kv="bf16")   # no FP8 KV declared
    assert (kv16.numerics["as_shipped"], kv16.numerics["changed"]) == (True, [])
    for name, t in sd.items():                                          # HF with the dequantized weights
        if name.endswith("_proj.weight"):
            w = Q.dequant(ck.linear(name[:-len(".weight")]))
            hf.get_parameter(name).data.copy_(w)
    ids = torch.tensor([3, 14, 15, 92, 65, 35, 8])
    with torch.no_grad():
        want = hf(ids.unsqueeze(0)).logits[0, -1]
        got16 = a16.prefill(ids, a16.alloc_state(32))
        got8 = shipped.prefill(ids, shipped.alloc_state(32))
    assert torch.allclose(got16, want, atol=2e-4)
    assert torch.nn.functional.cosine_similarity(got8, want, dim=0) > 0.99 and not torch.allclose(got8, want, atol=1e-4)


def test_ref_dense_fp8_kv_cache_as_declared(tmp_path):
    hf, d = _hf_tiny(tmp_path, "llama-linear")
    from safetensors.torch import load_file
    sd = load_file(str(d / "model.safetensors"))
    sd["model.layers.0.self_attn.k_proj.k_scale"] = torch.tensor(0.05)    # ModelOpt's names; layer 1 has none
    sd["model.layers.0.self_attn.v_proj.v_scale"] = torch.tensor(0.02)
    hq = {"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8"}}
    q_dir = _write(tmp_path / "kv8", json.loads((d / "config.json").read_text()), sd, hq=hq)
    ck = Checkpoint(str(q_dir))
    fp8 = ref_dense.DenseDecoder.from_checkpoint(ck, device="cpu", max_seq=32)
    k16 = ref_dense.DenseDecoder.from_checkpoint(ck, device="cpu", max_seq=32, kv="bf16")
    assert fp8.kv_dtype == torch.float8_e4m3fn and fp8.layers[0].k_scale.item() == pytest.approx(0.05)
    assert fp8.layers[1].k_scale is None and "fp8 e4m3" in fp8.numerics["kv"]
    assert "changed" in k16.numerics["kv"]
    assert (fp8.numerics["as_shipped"], fp8.numerics["changed"]) == (True, [])
    assert k16.numerics["as_shipped"] is False and k16.numerics["changed"] == ["kv: float32 instead of the declared fp8 e4m3"]
    ids = torch.tensor([7, 1, 44, 3, 90, 12])
    with torch.no_grad():
        want = hf(ids.unsqueeze(0)).logits[0, -1]
        s8, s16 = fp8.alloc_state(32), k16.alloc_state(32)
        got8, got16 = fp8.prefill(ids, s8).clone(), k16.prefill(ids, s16).clone()
        step8 = fp8.decode(torch.tensor([5]), torch.tensor([6], dtype=torch.int32), s8).clone()
    assert torch.allclose(got16, want, atol=2e-4)
    assert torch.nn.functional.cosine_similarity(got8, want, dim=0) > 0.99 and not torch.equal(got8, got16)
    assert s8.k[0].dtype == torch.float8_e4m3fn and s8.k[0][:, 6].float().abs().sum() > 0 and step8.isfinite().all()


def test_ref_dense_numerics_unchanged_by_no_op_flags(tmp_path):
    _, d = _hf_tiny(tmp_path, "qwen3", torch.bfloat16)
    ck = Checkpoint(str(d))
    for act, kv in (("as_shipped", "as_shipped"), ("a16", "as_shipped"), ("as_shipped", "bf16"), ("a16", "bf16")):
        n = ref_dense.DenseDecoder.from_checkpoint(ck, device="cpu", max_seq=16, act=act, kv=kv).numerics
        assert (n["as_shipped"], n["changed"]) == (True, []), (act, kv)
        assert n["kv"] == "bfloat16" and n["weights"] == "16-bit as stored"


def test_ref_dense_moves_every_weight_to_its_device(tmp_path):
    # The Checkpoint reads onto the CPU; the model's device decides where every tensor lives ("meta" stands in
    # for a GPU here: a tensor left behind would still be on the CPU).
    _, d = _hf_tiny(tmp_path, "llama-yarn")
    from safetensors.torch import load_file
    sd = load_file(str(d / "model.safetensors"))
    sd["model.layers.0.self_attn.k_proj.k_scale"] = torch.tensor(0.05)
    hq = {"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8"}}
    ck = Checkpoint(str(_write(tmp_path / "kv8", json.loads((d / "config.json").read_text()), sd, hq=hq)))
    m = ref_dense.DenseDecoder.from_checkpoint(ck, device="meta", max_seq=16)
    lins = [getattr(L, f) for L in m.layers for f in ("q", "k", "v", "o", "gate", "up", "down")] + [m.head]
    tensors = [t for w in lins for t in (w.weight, w.scale, w.scale_2, w.input_scale, w.bias) if t is not None]
    tensors += [m.embed, m.norm, m.cos, m.sin, m.layers[0].k_scale, m.layers[0].v_scale, m.layers[0].ln_attn]
    assert any(w.bias is not None for w in lins) and all(t.device.type == "meta" for t in tensors)


def test_registry_and_option_checks(tmp_path):
    for arch in ref_dense.ARCHITECTURES:
        assert models.model_class(arch) is ref_dense.DenseDecoder
    _, d = _hf_tiny(tmp_path, "qwen3")
    with pytest.raises(ValueError):
        models.load_model(str(d), device="cpu", max_seq=16, kv="fp4")
    with pytest.raises(NotImplementedError):
        ref_dense.rope_inv_freq({"rope_scaling": {"rope_type": "dynamic", "factor": 2.0}}, 16)


# --- the GPU check script, run on CPU against Hugging Face -------------------------------------------------------
def test_validate_model_metrics():
    from ikengine.tools import validate_model as vm
    top = {1: math.log(0.6), 2: math.log(0.3), 3: math.log(0.05)}
    assert vm.bucket_kl(top, top) == pytest.approx(0.0, abs=1e-12)
    assert vm.bucket_kl(top, {1: math.log(0.3), 2: math.log(0.6), 3: math.log(0.05)}) > 0.1
    assert vm.bucket_kl({1: 0.0}, {2: math.log(0.9), 1: math.log(0.1)}) == pytest.approx(math.log(10))
    r = vm.compare([1, 2, 3], [top] * 3, [1, 2, 4], [top, {2: 0.0}, top])
    assert r["first_divergence"] == 2 and r["top1_agreement"] == pytest.approx(2 / 3)


def test_validate_model_divergence_margins():
    from ikengine.tools import validate_model as vm
    eng = {1: math.log(0.51), 2: math.log(0.49)}                              # the engine picks 1 by a hair
    ref = {2: math.log(0.52), 1: math.log(0.48)}                              # the reference picks 2 by a hair
    r = vm.compare([5, 1, 7], [eng] * 3, [5, 2, 9], [ref] * 3, [ref] * 3)
    d = r["divergence"]
    assert r["first_divergence"] == 1 and d["step"] == 1 and (d["engine_token"], d["ref_token"]) == (1, 2)
    assert d["eng_gap"] == pytest.approx(math.log(0.51 / 0.49))
    assert d["ref_free_gap"] == pytest.approx(math.log(0.52 / 0.48)) and d["near_tie"] is True
    far = {2: math.log(0.9), 1: math.log(0.01)}
    assert vm.compare([1], [eng], [2], [far], [far])["divergence"]["near_tie"] is False
    half = {1: -1.0, 2: -1.50000006}                                          # 0.5 nats, as float32 logprobs give it
    assert vm.divergence(0, [1], [2], [half], [{2: -1.0, 1: -1.25}])["near_tie"] is True
    assert vm.compare([1, 2], [eng] * 2, [1, 2], [ref] * 2)["divergence"] is None
    assert vm.divergence(0, [1], [2], [{1: 0.0}], [{3: 0.0}])["ref_tf_gap"] is None      # its own token unseen
    bound = vm.divergence(0, [1], [2], [{1: -0.1, 3: -0.2}], [{2: -0.1, 1: -0.2}])        # 2 not in the engine's list
    assert bound["eng_gap_bound"] == ">=" and bound["near_tie"] is False                # a bound cannot show a tie
    assert d["ref_tf_prefers_engine"] is False and bound["ref_tf_prefers_engine"] is False
    # the reference decodes 2 but its own teacher-forced pass prefers the engine's 1: flagged, still not a tie
    flip = vm.divergence(0, [1], [2], [{1: -0.125, 2: -1.875}], [{1: -0.375, 2: -1.0}], [{2: -0.5, 1: -0.625}])
    assert flip["ref_tf_gap"] == pytest.approx(-0.625) and flip["ref_free_gap"] == pytest.approx(0.125)
    assert flip["near_tie"] is False and flip["ref_tf_prefers_engine"] is True
    assert vm._fmt_gap(flip, "eng_gap") == "1.75" and vm._fmt_gap({"g": 1.125}, "g") == "1.125"
    assert vm._fmt_gap(bound, "eng_gap") == ">=0.1"
    assert vm._tie_word(flip) == "NOT tie but ref teacher-forced prefers eng" and vm._tie_word(d) == "tie"


def test_validate_model_reads_dev_style_prompt_files(tmp_path):
    import gzip
    from ikengine.tools import validate_model as vm
    rows = [{"id": f"{b}-{i}", "bucket": b, "messages": [{"role": "user", "content": f"{b} {i}"}]}
            for b in ("very_short", "short") for i in range(3)] + [{"prompt": "plain"}]
    path = tmp_path / "dev.jsonl.gz"
    with gzip.open(path, "wt") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    prompts, labels = vm.read_prompt_file(str(path), per_bucket=2)
    assert labels == ["very_short-0", "very_short-1", "short-0", "short-1", "line7"]
    assert prompts[0] == [{"role": "user", "content": "very_short 0"}] and prompts[-1] == "plain"
    assert len(vm.read_prompt_file(str(path))[0]) == 7


def _chat_tokenizer(d: Path):
    """A small real tokenizer (word level) with BOS and a chat template, saved as a checkpoint would ship it."""
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    words = ["<unk>", "<s>", "<|user|>", "<|end|>", "<|assistant|>", "hello", "world", "think", "no"]
    tk = tokenizers.Tokenizer(tokenizers.models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tk.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tk.post_processor = tokenizers.processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", unk_token="<unk>")
    tok.chat_template = ("{{ bos_token }}{% for m in messages %} <|user|> {{ m['content'] }} <|end|>{% endfor %}"
                         "{% if add_generation_prompt %} <|assistant|>{% endif %}"
                         "{% if enable_thinking is defined and not enable_thinking %} no think{% endif %}")
    tok.save_pretrained(str(d))
    return tok


def test_validate_model_chat_prompts_are_token_ids(tmp_path):
    from types import SimpleNamespace
    from ikengine.tools import validate_model as vm
    _chat_tokenizer(tmp_path / "tok")
    args = SimpleNamespace(model=str(tmp_path / "tok"), prompt_ids=None, prompts=None, chat=True,
                           chat_template_kwargs=None)
    got = vm.load_prompts(args)                                              # the built-in prompts
    assert len(got) == len(vm.PROMPTS) and all(type(i) is int for p in got for i in p)
    assert all(p[:2] == [1, 2] and p[-2:] == [3, 4] for p in got)
    (tmp_path / "p.txt").write_text("hello world\n" + json.dumps({"prompt": "world"}) + "\n")
    args.prompts = str(tmp_path / "p.txt")
    assert vm.load_prompts(args) == [[1, 2, 5, 6, 3, 4], [1, 2, 6, 3, 4]]   # one BOS, from the template
    args.chat_template_kwargs = '{"enable_thinking": false}'
    assert vm.load_prompts(args)[0] == [1, 2, 5, 6, 3, 4, 8, 7]
    args.chat = False
    assert vm.load_prompts(args) == [[1, 5, 6], [1, 6]]                      # plain prompts get the BOS
    (tmp_path / "t.jinja").write_text("{% for m in messages %}<|user|> {{ m['content'] }}{% endfor %} <|assistant|>")
    args.chat, args.chat_template, args.chat_template_kwargs = True, str(tmp_path / "t.jinja"), None
    assert vm.load_prompts(args) == [[2, 5, 6, 4], [2, 6, 4]]                # the given template, not the tokenizer's


def test_validate_model_key_comes_from_the_environment(monkeypatch):
    from ikengine.tools import validate_model as vm
    monkeypatch.setenv("IK_TEST_REF_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert vm._request("http://x/v1/models", None, "IK_TEST_REF_KEY").get_header("Authorization") == "Bearer sk-test"
    assert vm._request("http://x/v1/completions", {"a": 1}, "OPENAI_API_KEY").get_header("Authorization") is None
    with pytest.raises(SystemExit):                                           # no key value on the command line,
        vm.main(["--model", "m", "--api-key", "sk-test"])                    # not even as a prefix of --api-key-env


# A fake reference for the A/A checks: 20 tokens, all in every top-20 list (nothing is imputed), and the next
# token peaks one after the last. Its decode (free
# running) and its teacher-forced (prefill) logprobs differ on purpose: prefill is softer, and after token 3 it
# peaks on 3 again, so the reference disagrees with itself at that step. Its second free run of a prompt leaves the
# first at step 2.
AA_VOCAB, AA_PROMPT = 20, [1, 2]


def _aa_lp(peak: int, sharp: float) -> list[float]:
    z = [-sharp * abs(v - peak) - 0.01 * v for v in range(AA_VOCAB)]        # distinct values: no ties
    m = math.log(sum(math.exp(x) for x in z))
    return [x - m for x in z]


def _aa_decode(ctx: list[int], run: int) -> list[float]:
    return _aa_lp(ctx[-1] + 1 + (3 if run == 2 and len(ctx) == len(AA_PROMPT) + 2 else 0), 1.0)


def _aa_prefill(ctx: list[int]) -> list[float]:
    return _aa_lp(ctx[-1] + (0 if ctx[-1] == 3 else 1), 0.5)


def _aa_top(row: list[float]) -> dict[int, float]:
    return dict(sorted(enumerate(row), key=lambda kv: -kv[1])[:20])


def _aa_free(prompt: list[int], n: int, run: int) -> tuple[list[int], list[dict]]:
    ids, tops = [], []
    for _ in range(n):
        row = _aa_decode(prompt + ids, run)
        tops.append(_aa_top(row))
        ids.append(max(range(AA_VOCAB), key=row.__getitem__))
    return ids, tops


def _aa_server(kind: str, forced_on: list):
    """A fake vLLM (/v1/completions) or SGLang (/generate) server on the fake reference; `forced_on` gets the
    tokens each teacher-forced request scores after the prompt."""
    runs: dict = {}

    def post(url, body, key_env):
        if kind == "vllm":
            assert url.endswith("/v1/completions")
            full = body["prompt"]
            if "prompt_logprobs" not in body:
                runs[tuple(full)] = runs.get(tuple(full), 0) + 1
                ids, tops = _aa_free(full, body["max_tokens"], runs[tuple(full)])
                return {"choices": [{"logprobs": {"tokens": [f"token_id:{t}" for t in ids], "top_logprobs": [
                    {f"token_id:{t}": v for t, v in d.items()} for d in tops]}}]}
            forced_on.append(full[len(AA_PROMPT):])
            return {"choices": [{"prompt_logprobs": [None] + [
                {str(t): {"logprob": v} for t, v in _aa_top(_aa_prefill(full[:j])).items()}
                for j in range(1, len(full))]}]}
        assert url.endswith("/generate")
        full, s = body["input_ids"], body["logprob_start_len"]
        if s == -1:
            runs[tuple(full)] = runs.get(tuple(full), 0) + 1
            ids, tops = _aa_free(full, body["sampling_params"]["max_new_tokens"], runs[tuple(full)])
            return {"meta_info": {"output_token_logprobs": [[d[t], t, None] for d, t in zip(tops, ids)],
                                  "output_top_logprobs": [[[v, t, None] for t, v in d.items()] for d in tops]}}
        forced_on.append(full[len(AA_PROMPT):])
        return {"meta_info": {"input_token_logprobs": [[None, full[s], None]] + [[0.0, t, None] for t in full[s + 1:]],
                              "input_top_logprobs": [None] + [[[v, t, None] for t, v in
                                                               _aa_top(_aa_prefill(full[:j])).items()]
                                                              for j in range(s + 1, len(full))]}}
    return post


class _AALM:
    """The fake reference as a Hugging Face causal LM: calls with use_cache are the greedy decode (the cache is the
    context so far), calls without are one teacher-forced pass over the whole sequence."""

    def __init__(self, forced_on: list):
        self.free_runs, self.forced_on = 0, forced_on

    def eval(self):
        return self

    def __call__(self, x, past_key_values=None, use_cache=False):
        from types import SimpleNamespace
        seq = [int(t) for t in x[0].tolist()]
        if use_cache:
            if past_key_values is None:
                self.free_runs += 1
            ctx = (past_key_values or []) + seq
            return SimpleNamespace(logits=torch.tensor([[_aa_decode(ctx, self.free_runs)]]), past_key_values=ctx)
        self.forced_on.append(seq[len(AA_PROMPT):])
        return SimpleNamespace(logits=torch.tensor([[_aa_prefill(seq[:j + 1]) for j in range(len(seq))]]))


def test_validate_model_reference_aa_scores_its_own_teacher_forced_path(monkeypatch):
    """--ref-aa for vLLM, SGLang and Hugging Face: the reference's decode top-20 in the engine's place against its
    own teacher-forced top-20 on its own tokens, KL(forced || decode) and top-1 = argmax(forced) against its token,
    plus a second free run against the first."""
    from types import SimpleNamespace
    from ikengine.tools import validate_model as vm
    n = 3
    gen, free = _aa_free(AA_PROMPT, n, 1)
    own = [_aa_top(_aa_prefill(AA_PROMPT + gen[:i])) for i in range(n)]
    want_kl = sum(vm.bucket_kl(o, f) for o, f in zip(own, free)) / n
    assert gen == [3, 4, 5] and [max(o, key=o.get) for o in own] == [3, 3, 5]   # it disagrees with itself at step 1
    assert abs(want_kl - sum(vm.bucket_kl(f, o) for o, f in zip(own, free)) / n) > 0.01   # the directions differ
    eng = {"ids": [3, 9, 10], "tops": [{3: 0.0}, {9: 0.0}, {10: 0.0}]}         # the engine leaves it at step 1
    for kind in ("vllm", "sglang", "hf"):
        forced_on: list = []
        args = SimpleNamespace(ref="http://ref:1", ref_kind=kind, ref_model="m", api_key_env=None, tokens=n,
                               ref_aa=True, tie_nats=0.5, model="m", device="cpu", ref_device=None)
        if kind == "hf":
            monkeypatch.setattr(vm, "_load_hf", lambda d, dev: _AALM(forced_on))
            rec = vm.ref_hf(args, [AA_PROMPT], [eng])[0]
        else:
            monkeypatch.setattr(vm, "_post", _aa_server(kind, forced_on))
            rec = vm.ref_openai(args, [AA_PROMPT], [eng])[0]
        aa = rec["aa"]
        assert rec["ids"] == gen and forced_on == [eng["ids"], gen], kind     # the engine's tokens, then its own
        assert aa["steps"] == n and aa["mean_kl"] == pytest.approx(want_kl, rel=1e-4), kind
        assert aa["top1_agreement"] == pytest.approx(2 / 3) and aa["top1_agreement_ties"] == pytest.approx(2 / 3)
        assert aa["top1_gaps"] == [pytest.approx(max(own[1].values()) - own[1][4], rel=1e-4)], kind
        assert aa["run_to_run_first_divergence"] == 2, kind                    # its second free run: 3, 4, 8

    def no_forced(url, body, key_env):                                       # a server without prompt_logprobs
        if "prompt_logprobs" in body:
            raise ValueError("unsupported")
        return _aa_server("vllm", [])(url, body, key_env)
    monkeypatch.setattr(vm, "_post", no_forced)
    args.ref_kind = "vllm"
    with pytest.raises(ValueError, match="teacher-forced"):                  # an A/A it cannot run: an error
        vm.ref_openai(args, [AA_PROMPT], [eng])
    args.ref_aa = False                                                      # without --ref-aa: its own path only
    assert vm.ref_openai(args, [AA_PROMPT], [eng])[0]["teacher_forced"] is False


def test_validate_model_main_on_a_saved_engine_run(tmp_path, monkeypatch, capsys):
    """--load-engine: the saved prompts, labels and --tokens by default; --n takes the first; --prompt-ids must be
    the first of the saved prompts; the report and the printed speed say the engine run was loaded."""
    from ikengine.tools import validate_model as vm
    prompts = [AA_PROMPT, [2, 6]]
    runs = [dict(zip(("ids", "tops"), _aa_free(p, 3, 1))) for p in prompts]   # the engine decodes as the reference
    saved = tmp_path / "engine.json"
    rec = {"prompts": prompts, "labels": ["a", "b"], "tokens": 3, "model": "m", "decode_tok_s": 9.5,
           "engine": {"engine": "fake"}, "runs": runs}
    saved.write_text(json.dumps(rec))
    monkeypatch.setattr(vm, "run_engine", lambda a, p: pytest.fail("the engine must not be built"))
    monkeypatch.setattr(vm, "_post", _aa_server("vllm", []))
    out = tmp_path / "report.json"
    base = ["--model", "m", "--ref", "http://ref:1", "--ref-kind", "vllm", "--ref-model", "m", "--min-top1", "0",
            "--max-kl", "100", "--load-engine", str(saved), "--out", str(out)]

    assert vm.main(base + ["--n", "1", "--ref-aa"]) == 0
    rep, printed = json.loads(out.read_text()), capsys.readouterr()
    assert rep["prompts"] == 1 and [p["prompt"] for p in rep["per_prompt"]] == ["a"] and rep["tokens"] == 3
    assert rep["label"] == f"observed (engine: the run saved in {saved}; reference: this run)"
    assert rep["decode_tok_s"] == 9.5 and rep["engine"] == {"engine": "fake"}
    assert "decode 9.5 tok/s (observed, saved run)" in printed.out and "note:" not in printed.err
    aa = rep["reference_aa"]
    assert aa["prompts"] == 1 and aa["steps"] == 3 and aa["run_to_run_identical"] == 0
    assert aa["top1_disagreement_gaps"]["n"] == 1 and len(aa["top1_disagreement_gaps"]["sorted"]) == 1
    assert rep["per_prompt"][0]["reference_aa"]["run_to_run_first_divergence"] == 2
    assert len(rep["per_prompt"][0]["reference_aa"]["top1_gaps"]) == 1 and "run to run 0/1 identical" in printed.out

    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps([AA_PROMPT]))                                  # the first of the saved prompts
    assert vm.main(base + ["--prompt-ids", str(ids)]) == 0
    assert [p["prompt"] for p in json.loads(out.read_text())["per_prompt"]] == ["ids0"]
    ids.write_text(json.dumps([[2, 6]]))                                     # a saved prompt, but not the first
    assert vm.main(base + ["--prompt-ids", str(ids)]) == 2
    assert vm.main(base + ["--tokens", "4"]) == 2                            # saved for --tokens 3
    capsys.readouterr()
    assert vm.main(["--model", "other"] + base[2:]) == 0                     # another --model: a note, not a refusal
    assert "saved for --model m, not other" in capsys.readouterr().err
    del rec["labels"]                                                        # a file from before labels were saved
    saved.write_text(json.dumps(rec))
    assert vm.main(base) == 0
    assert [p["prompt"] for p in json.loads(out.read_text())["per_prompt"]] == ["prompt0", "prompt1"]
    with pytest.raises(SystemExit):                                          # load and save at once: refused
        vm.main(base + ["--save-engine", str(tmp_path / "again.json")])


def test_validate_model_passes_against_hugging_face_on_cpu(tmp_path):
    _, d = _hf_tiny(tmp_path, "llama-llama3-rope")
    from ikengine.tools import validate_model as vm
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps([[1, 2, 3, 4], [50, 7, 9]]))
    out = tmp_path / "report.json"
    rc = vm.main(["--model", str(d), "--ref", "hf", "--device", "cpu", "--prompt-ids", str(ids), "--tokens", "6",
                  "--max-seq", "32", "--require-identical", "--out", str(out)])
    rep = json.loads(out.read_text())
    assert rc == 0 and rep["verdict"] == "PASS" and rep["identical_greedy"] == 2
    assert rep["top1_agreement"] == 1.0 and rep["mean_kl"] < 1e-6
    assert rep["divergences"] == 0 and rep["per_prompt"][0]["prompt_tokens"] == 4
    assert vm.main(["--model", str(tmp_path / "missing"), "--device", "cpu", "--prompt-ids", str(ids)]) == 2
    saved, out2 = tmp_path / "engine.json", tmp_path / "report2.json"    # the engine alone, then the reference
    assert vm.main(["--model", str(d), "--ref", "none", "--device", "cpu", "--prompt-ids", str(ids), "--tokens", "6",
                    "--max-seq", "32", "--engine-out", str(saved)]) == 0
    assert vm.main(["--model", str(d), "--ref", "hf", "--device", "cpu", "--engine-in", str(saved),
                    "--require-identical", "--ref-aa", "--out", str(out2)]) == 0
    rep2 = json.loads(out2.read_text())
    assert rep2["identical_greedy"] == 2 and rep2["mean_kl"] < 1e-6 and rep2["per_prompt"][1]["steps"] == 6
    assert rep2["reference_aa"]["top1_agreement"] == 1.0 and rep2["reference_aa"]["mean_kl"] < 1e-6
    assert vm.main(["--model", str(d), "--ref", "hf", "--device", "cpu", "--load-engine", str(saved),
                    "--tokens", "5"]) == 2                                    # saved for --tokens 6: refused
