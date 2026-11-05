"""The Gemma 4 model module (skeleton ikengine/models/gemma4.py) against Hugging Face's reference forward.

Tiny random Gemma 4 text models are built with transformers, saved as safetensors, read back through
ikengine's Checkpoint and compared in fp32 on the CPU: last-position prefill logits and 8 greedy decode steps,
with prompts longer than the sliding window. A bf16 copy of each runs the shipped dtype path against HF's own
bf16 error. The config parsing and numerics tests need nothing; the rest need torch and a transformers release
with Gemma 4 and skip without them (CI has neither).
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKELETON = ROOT / "plugin" / "skills" / "optimizing-kernels" / "skeleton"
if str(SKELETON) not in sys.path:
    sys.path.insert(0, str(SKELETON))

from ikengine.models import gemma4 as g4  # noqa: E402  (imports without torch)
from ikengine.models import model_class  # noqa: E402

S, F = g4.SLIDING, g4.FULL


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


HAS_TORCH = _has("torch") and _has("safetensors")
HAS_HF = HAS_TORCH and _has("transformers") and _has("transformers.models.gemma4")
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="needs torch and safetensors")
needs_hf = pytest.mark.skipif(not HAS_HF, reason="needs torch and transformers with Gemma 4")


# --- config parsing (no torch) ------------------------------------------------------------------------
def test_registered_for_both_architectures():
    assert model_class("Gemma4ForConditionalGeneration") is g4.Gemma4
    assert model_class("Gemma4ForCausalLM") is g4.Gemma4


def test_layer_plan_from_a_hub_style_config():
    """Hub configs carry global_head_dim / num_global_key_value_heads; they apply to the full layers."""
    tc = {"num_hidden_layers": 6, "num_key_value_heads": 8, "head_dim": 256, "global_head_dim": 512,
          "num_global_key_value_heads": 2, "attention_k_eq_v": True, "sliding_window": 1024,
          "layer_types": [S, S, S, S, S, F]}
    plan = g4.layer_plan(tc)
    assert [a.kind for a in plan] == [S] * 5 + [F]
    assert {(a.head_dim, a.n_kv_heads, a.k_eq_v, a.window) for a in plan[:5]} == {(256, 8, False, 1024)}
    assert (plan[5].head_dim, plan[5].n_kv_heads, plan[5].k_eq_v, plan[5].window) == (512, 2, True, None)
    assert [a.kv_layer for a in plan] == list(range(6))


def test_layer_plan_global_kv_heads_need_k_eq_v():
    tc = {"num_hidden_layers": 2, "num_key_value_heads": 4, "head_dim": 8, "global_head_dim": 16,
          "num_global_key_value_heads": 1, "layer_types": [S, F]}
    assert (g4.layer_plan(tc)[1].head_dim, g4.layer_plan(tc)[1].n_kv_heads) == (16, 4)


def test_layer_plan_from_per_layer_config():
    """transformers >= 5.19 saves per_layer_config with zero-padded layer keys (or a layer type name)."""
    tc = {"num_hidden_layers": 12, "num_key_value_heads": 2, "head_dim": 8, "sliding_window": 4,
          "layer_types": [S] * 5 + [F] + [S] * 5 + [S],
          "per_layer_config": {"05": {"head_dim": 16, "num_key_value_heads": 1}, "11": {"head_dim": 32}}}
    plan = g4.layer_plan(tc)
    assert plan[-1].kind == F                                          # the last layer is always full
    assert (plan[5].head_dim, plan[5].n_kv_heads) == (16, 1)
    assert (plan[11].head_dim, plan[11].n_kv_heads) == (32, 2)
    assert plan[0].head_dim == 8
    tc2 = dict(tc, per_layer_config={F: {"head_dim": 16}})
    assert [a.head_dim for a in g4.layer_plan(tc2)] == [16 if a.kind == F else 8 for a in g4.layer_plan(tc2)]
    with pytest.raises(NotImplementedError):
        g4.layer_plan(dict(tc, per_layer_config={"05": {"intermediate_size": 3}}))


def test_layer_plan_kv_sharing_reads_the_last_owner_of_the_same_type():
    tc = {"num_hidden_layers": 6, "num_key_value_heads": 1, "head_dim": 8, "global_head_dim": 16,
          "layer_types": [S, S, F, S, S, F], "num_kv_shared_layers": 2}
    assert [a.kv_layer for a in g4.layer_plan(tc)] == [0, 1, 2, 3, 3, 2]
    with pytest.raises(ValueError):
        g4.layer_plan(dict(tc, layer_types=[S, S, S, S, F, F]))   # no full layer before the shared ones
    with pytest.raises(ValueError):
        g4.layer_plan(dict(tc, layer_types=[S, S, "linear_attention", S, S, F]))


# --- the numerics block (no torch) ----------------------------------------------------------------------
W4A4 = "nvfp4 with each module's static input_scale (W4A4)"


def _numerics(kinds, act="as_shipped", kv_fp8=False, declared_fp8=False, expert_scales=None):
    return g4._numerics(Counter(kinds), "bfloat16", "bfloat16", act, kv_fp8, declared_fp8, False, expert_scales)


def test_numerics_follow_the_stored_input_scales():
    """W4A4 when the quantized linears store an input_scale; weight-only NVFP4 (no input_scale) is W4A16, whose
    activations are 16-bit as shipped, so act="a16" changes nothing there."""
    n = _numerics({("nvfp4", True): 60, ("bf16", False): 40})
    assert n["activations"] == f"as shipped: {W4A4} for 60 linears; bfloat16 into 40 16-bit linears"
    assert n["as_shipped"] and n["changed"] == []
    n = _numerics({("nvfp4", True): 60, ("bf16", False): 40}, act="a16")
    assert n["changed"] == [f"activations: a16 instead of {W4A4} for 60 linears"] and not n["as_shipped"]
    for act in ("as_shipped", "a16"):
        n = _numerics({("nvfp4", False): 60, ("bf16", False): 40}, act=act)
        assert n["activations"] == ("as shipped: bfloat16 into 60 weight-only nvfp4 linears (W4A16); "
                                    "bfloat16 into 40 16-bit linears")
        assert n["as_shipped"] and n["changed"] == []
    n = _numerics({("nvfp4", True): 2, ("nvfp4", False): 3}, act="a16")     # mixed: only the W4A4 ones change
    assert n["changed"] == [f"activations: a16 instead of {W4A4} for 2 linears"]
    assert n["weights"] == "as stored: 5 nvfp4 linears; embedding bfloat16"
    assert _numerics({("bf16", False): 9}, act="a16")["changed"] == []


def test_numerics_name_every_quantized_kind():
    for kinds, text in (({("fp8_tensor", True): 7}, "fp8 e4m3 per tensor with each module's static input_scale"),
                        ({("fp8_tensor", False): 7}, "fp8 e4m3 per tensor, one dynamic scale per tensor"),
                        ({("fp8_block", False): 7}, "fp8 e4m3, dynamic per-token groups")):
        assert _numerics(kinds)["activations"] == f"as shipped: {text} for 7 linears"
        assert _numerics(kinds, act="a16")["changed"] == [f"activations: a16 instead of {text} for 7 linears"]


def test_numerics_carry_the_expert_input_scales():
    """ENGINE.json gets the expert input-scale facts (the engine copies only `numerics` into it)."""
    q = {("nvfp4", True): 12}
    same = {"one_value_per_layer": {"gate": True, "up": True, "down": True}, "gate_equals_up": True}
    differ = {"one_value_per_layer": {"gate": False, "up": True, "down": True}, "gate_equals_up": False}
    assert "expert_input_scales" not in _numerics(q)
    assert _numerics(q, expert_scales=same)["expert_input_scales"].endswith("computes the same")
    t = _numerics(q, expert_scales=differ)["expert_input_scales"]
    assert "gate equals up no" in t and "gate no, up yes, down yes" in t and t.endswith("when greedy ids differ")
    assert _numerics(q, act="a16", expert_scales=differ)["expert_input_scales"].startswith("unused (a16")
    n = _numerics(q, kv_fp8=False, declared_fp8=True, expert_scales=same)
    assert n["changed"] == ["kv: bfloat16 instead of the declared fp8 e4m3"] and json.loads(json.dumps(n)) == n


# --- tiny Hugging Face models ---------------------------------------------------------------------------
MOE = dict(  # the 26B-A4B shape family: sliding + full layers, k = v on full layers, dense MLP beside MoE
    vocab_size=96, hidden_size=32, intermediate_size=48, num_hidden_layers=5, num_attention_heads=4,
    num_key_value_heads=2, head_dim=8, global_head_dim=16, num_global_key_value_heads=1, attention_k_eq_v=True,
    sliding_window=6, layer_types=[S, S, F, S, F], final_logit_softcapping=5.0, hidden_size_per_layer_input=0,
    enable_moe_block=True, num_experts=4, top_k_experts=2, moe_intermediate_size=16,
    rope_parameters={S: {"rope_type": "default", "rope_theta": 10_000.0},
                     F: {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1_000_000.0}},
    max_position_embeddings=256, pad_token_id=0, eos_token_id=1, bos_token_id=2)
PLE = dict(  # the small dense family: per-layer inputs, KV-shared last layers, double-wide MLP there
    vocab_size=96, vocab_size_per_layer_input=96, hidden_size=32, intermediate_size=40, num_hidden_layers=6,
    num_attention_heads=4, num_key_value_heads=1, head_dim=8, global_head_dim=16, sliding_window=5,
    layer_types=[S, S, F, S, S, F], final_logit_softcapping=5.0, hidden_size_per_layer_input=8,
    num_kv_shared_layers=2, use_double_wide_mlp=True, enable_moe_block=False,
    max_position_embeddings=256, pad_token_id=0, eos_token_id=1, bos_token_id=2)
PROMPT = [2, 17, 5, 88, 41, 3, 64, 9, 30, 77, 12, 50, 23, 71, 6]       # 15 tokens: longer than both windows
TOL = dict(rtol=1e-4, atol=1e-4)


def _randomize(model, seed: int) -> None:
    """Every stored tensor random and non-trivial (norm weights, scalars and router scales included)."""
    import torch
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("norm.weight", "router.scale", "per_expert_scale")):
                p.copy_(1.0 + 0.25 * torch.randn(p.shape, generator=g))
            elif p.dim() >= 2:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5)
            else:
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
        for name, b in model.named_buffers():
            if name.endswith("layer_scalar"):
                b.copy_(0.6 + 0.8 * torch.rand(b.shape, generator=g))


def _build(kind: str, out: Path):
    """Save a random model; return (dir, the fp32 eager reference read back from it)."""
    import torch
    import transformers as tf
    if kind == "moe":       # as the shipped checkpoint: the multimodal wrapper, text weights only
        cfg = tf.Gemma4Config(text_config=dict(MOE), vision_config=None, audio_config=None)
        cls = tf.Gemma4ForConditionalGeneration
    else:
        cfg, cls = tf.Gemma4TextConfig(**PLE), tf.Gemma4ForCausalLM
    torch.manual_seed(0)
    model = cls(cfg)
    _randomize(model, seed=1 if kind == "moe" else 2)
    model.save_pretrained(out)
    return str(out), _load_ref(cls, out)


def _load_ref(cls, path, dtype=None):
    """The reference (fp32 unless `dtype` says otherwise): eager attention and the plain per-expert loop."""
    import torch
    return cls.from_pretrained(path, dtype=dtype or torch.float32, attn_implementation="eager",
                               experts_implementation="eager").eval()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    if not HAS_HF:
        pytest.skip("needs torch and transformers with Gemma 4")
    root = tmp_path_factory.mktemp("gemma4")
    return {k: _build(k, root / k) for k in ("moe", "ple")}


def _ref_logits(hf, ids):
    """HF's last-position logits over the whole sequence, no cache."""
    import torch
    with torch.no_grad():
        return hf(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1].float()


def _hf_layers(hf):
    return hf.model.language_model.layers if hasattr(hf.model, "language_model") else hf.model.layers


def _greedy_against(m, state, prompt, ref, steps=8, tol=TOL):
    """Prefill then `steps` greedy decode steps; each step's logits against ref(ids) (HF over the full ids)."""
    import torch
    got = m.prefill(torch.tensor(prompt), state)
    torch.testing.assert_close(got, ref(prompt), **tol)
    ids, tok = list(prompt), int(got.argmax())
    for _ in range(steps):
        ids.append(tok)
        out = m.decode(torch.tensor([tok]), torch.tensor([len(ids) - 1], dtype=torch.int32), state)
        want = ref(ids)
        torch.testing.assert_close(out, want, **tol)
        tok = int(out.argmax())
        assert tok == int(want.argmax())
    return ids


@needs_hf
@pytest.mark.parametrize("chunk", [4, 64])
@pytest.mark.parametrize("kind", ["moe", "ple"])
def test_prefill_and_greedy_decode_match_hf(tiny, kind, chunk):
    import torch
    from ikengine import engine_api
    from ikengine.models import load_model
    d, hf = tiny[kind]
    m = load_model(d, device="cpu", max_seq=40, prefill_chunk=chunk)
    assert isinstance(m, engine_api.Model)
    assert (m.vocab_size, m.max_seq, m.eos_token_ids) == (96, 40, frozenset({1}))
    assert m.numerics["as_shipped"] and m.numerics["changed"] == []
    state = m.alloc_state(40)
    _greedy_against(m, state, PROMPT, lambda ids: _ref_logits(hf, ids))
    # a second request on the same state: nothing from the first one may leak through the ring buffers
    m.reset(state)
    _greedy_against(m, state, [2, 40, 7], lambda ids: _ref_logits(hf, ids), steps=6)
    assert state.logits.dtype == torch.float32 and state.logits.shape == (96,)


@needs_hf
def test_hub_style_config_matches_hf(tiny, tmp_path):
    """The published checkpoint's text_config is the older hub style (global_head_dim, num_global_key_value_heads,
    no per_layer_config). The same weights under that config must give the same logits."""
    import torch
    import transformers as tf
    from ikengine.models import load_model
    d, hf = tiny["moe"]
    hub = tmp_path / "hub-style"
    shutil.copytree(d, hub)
    cfg = json.loads((hub / "config.json").read_text())
    tc = cfg["text_config"]
    assert tc.pop("per_layer_config")                                 # what transformers 5.19 saved
    tc.update(head_dim=8, global_head_dim=16, num_global_key_value_heads=1)
    (hub / "config.json").write_text(json.dumps(cfg))
    hf_hub = _load_ref(tf.Gemma4ForConditionalGeneration, hub)       # HF reads both styles as one model
    assert torch.equal(_ref_logits(hf_hub, PROMPT), _ref_logits(hf, PROMPT))
    m = load_model(str(hub), device="cpu", max_seq=40, prefill_chunk=4)
    assert m.resolved["head_dim"] == {S: 8, F: 16} and m.resolved["kv_heads"] == {S: 2, F: 1}
    _greedy_against(m, m.alloc_state(40), PROMPT, lambda ids: _ref_logits(hf, ids))


@pytest.fixture(scope="module")
def tiny_bf16(tiny, tmp_path_factory):
    """Each tiny model rounded to bf16 and saved that way (bf16 tensors, dtype bfloat16 in config.json, as the
    shipped checkpoint), with two references read back from it: fp32 compute and HF's own bf16 forward."""
    import torch
    root = tmp_path_factory.mktemp("gemma4-bf16")
    out = {}
    for kind, (d, hf) in tiny.items():
        cls, bdir = type(hf), root / kind
        cls.from_pretrained(d, dtype=torch.bfloat16).save_pretrained(bdir)
        out[kind] = str(bdir), _load_ref(cls, bdir), _load_ref(cls, bdir, torch.bfloat16)
    return out


def _bf16_steps(m, ref32, ref16, prompt, steps=8):
    """[(module, HF fp32, HF bf16) logits] at the prefill and at each decode step along HF fp32's greedy ids."""
    import torch
    state, ids = m.alloc_state(40), list(prompt)
    got = m.prefill(torch.tensor(ids), state)
    rows = []
    for step in range(steps + 1):
        if step:
            ids.append(int(rows[-1][1].argmax()))
            got = m.decode(torch.tensor([ids[-1]]), torch.tensor([len(ids) - 1], dtype=torch.int32), state)
        rows.append((got.clone(), _ref_logits(ref32, ids), _ref_logits(ref16, ids)))
    return rows


def _hf_eager_attention(q, k, v, mask):
    """HF's eager attention on the module's layout (q [C, Hq, D], k/v [S, Hkv, D]): 16-bit scores, the mask added
    as the dtype's minimum, softmax in fp32 rounded back, 16-bit weighted sum."""
    import torch
    c, hq, d = q.shape
    g = hq // k.shape[1]
    kk, vv = (t.repeat_interleave(g, 1).transpose(0, 1) for t in (k, v))
    s = torch.matmul(q.transpose(0, 1), kk.transpose(1, 2))
    s = s + torch.where(mask, 0.0, torch.finfo(s.dtype).min).to(s.dtype)
    p = torch.softmax(s, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(p, vv).transpose(0, 1).reshape(c, hq * d)


@needs_hf
@pytest.mark.parametrize("kind", ["moe", "ple"])
def test_bf16_checkpoint_is_no_worse_than_hf_bf16(tiny_bf16, kind):
    """The shipped dtype path (bf16 activations and cache, torch SDPA), against HF fp32 on the same weights, over
    the prefill and 8 decode steps of PROMPT plus 16 random prompts: the median of (module's mean |logits - HF
    fp32|) / (HF bf16's) is at most 1.25, and no step is worse than 3x HF bf16 at that step or on average.
    A per-step 1.5x bound would fail by chance: two bf16 paths that round differently land 1.5x apart on 15% of
    single prompts, and some random prompts amplify bf16 rounding for both (observed; the module matches HF to
    1.6e-5 in fp32 on the same prompts). Observed medians on 40 prompts: 1.00 dense, 0.94 MoE. The prompt's
    prefill argmax agrees with both references. test_bf16_rounds_as_hf_outside_attention pins the rest exactly."""
    import torch
    from ikengine.models import load_model
    d, ref32, ref16 = tiny_bf16[kind]
    m = load_model(d, device="cpu", max_seq=40, prefill_chunk=4)      # dtype from config.json
    assert m.dtype == torch.bfloat16 and m.kv_dtype == torch.bfloat16 and m.numerics["as_shipped"]
    rows = _bf16_steps(m, ref32, ref16, PROMPT)
    got, want, hf16 = rows[0]
    assert int(got.argmax()) == int(want.argmax()) == int(hf16.argmax())
    g = torch.Generator().manual_seed(7)
    for _ in range(16):
        n = int(torch.randint(8, 30, (1,), generator=g))
        rows += _bf16_steps(m, ref32, ref16, [2] + torch.randint(3, 96, (n,), generator=g).tolist(), steps=0)
    ours = [(a - w).abs().mean().item() for a, w, _ in rows]
    theirs = [(h - w).abs().mean().item() for _, w, h in rows]
    ratios = sorted(a / b for a, b in zip(ours, theirs))
    assert ratios[len(ratios) // 2] <= 1.25, f"median error ratio {ratios[len(ratios) // 2]:.2f}"
    mean = sum(theirs) / len(theirs)
    for i, (a, b) in enumerate(zip(ours, theirs)):
        assert a <= 3 * max(b, mean), f"step {i}: mean |diff| {a:.4f} vs HF bf16 {b:.4f} (mean {mean:.4f})"


@needs_hf
def test_bf16_rounds_as_hf_outside_attention(tiny_bf16, monkeypatch):
    """With HF's eager attention swapped in for SDPA, the dense model in bf16 reproduces HF's bf16 forward: the
    embedding and per-layer-input scales, layer_scalar, norms, RoPE tables and the 16-bit cache all round as HF
    rounds them. The logit softcap (fp32 here, bf16 in HF) is the one difference left. Prefill in chunks of 4 and
    8 decode steps, prompts longer than the window."""
    from ikengine.models import load_model
    monkeypatch.setattr(g4, "_attend", _hf_eager_attention)
    d, ref32, ref16 = tiny_bf16["ple"]
    m = load_model(d, device="cpu", max_seq=40, prefill_chunk=4)
    for step, (got, want, hf16) in enumerate(_bf16_steps(m, ref32, ref16, PROMPT)):
        same, err = (got - hf16).abs().mean().item(), (hf16 - want).abs().mean().item()
        assert same <= 0.25 * err, f"step {step}: mean |module - HF bf16| {same:.4f}, HF bf16 error {err:.4f}"


@needs_torch
def test_declared_dtype_reads_the_top_level_config_first():
    """HF's dtype="auto" uses the top-level dtype and copies it onto the sub-configs; a saved text_config can
    keep a stale one (transformers 5.19 leaves text_config.dtype float32 after model.to(bfloat16))."""
    import torch
    from types import SimpleNamespace
    stale = SimpleNamespace(config={"dtype": "bfloat16"}, text_config={"dtype": "float32"})
    assert g4._declared_dtype(stale) == torch.bfloat16
    assert g4._declared_dtype(SimpleNamespace(config={}, text_config={"torch_dtype": "float16"})) == torch.float16
    assert g4._declared_dtype(SimpleNamespace(config={}, text_config={})) == torch.bfloat16


@needs_hf
def test_resolved_config_and_numerics(tiny):
    from ikengine.models import load_model
    m = load_model(tiny["moe"][0], device="cpu", max_seq=16)
    r = m.resolved
    assert r["architecture"] == "Gemma4ForConditionalGeneration" and r["text_prefix"] == "model.language_model."
    assert r["head_dim"] == {S: 8, F: 16} and r["kv_heads"] == {S: 2, F: 1} and r["k_eq_v_full_layers"]
    assert (r["experts"], r["top_k_experts"], r["final_logit_softcapping"]) == (4, 2, 5.0)
    assert r["expert_input_scales"] is None                           # 16-bit experts: no activation scales
    assert "expert_input_scales" not in m.numerics and json.loads(json.dumps(m.numerics)) == m.numerics
    assert m.numerics["kv"].startswith("float32") and "float32" in m.numerics["activations"]
    assert all(c.shape[0] == 6 for c, a in zip(m.alloc_state(16).k, m.plan) if a.kind == S)   # ring of window slots
    with pytest.raises(ValueError):
        load_model(tiny["moe"][0], device="cpu", max_seq=16, act="fp8")
    with pytest.raises(ValueError):
        load_model(tiny["moe"][0], device="cpu", max_seq=16, kv="int8")


def _fp8_round_trip(x, scale):
    import torch
    return ((x.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale).to(x.dtype)


@needs_hf
@pytest.mark.parametrize("kind", ["moe", "ple"])
def test_fp8_kv_cache_matches_hf_with_the_same_rounding(tiny, kind, tmp_path, monkeypatch):
    """kv="fp8" with per-layer k/v scales stored in the checkpoint: stored = fp8(x / scale), read = stored * scale.
    The reference is HF with that round trip applied to its keys (after RoPE) and values (after their norm)."""
    import torch
    from safetensors.torch import save_file
    from transformers.models.gemma4 import modeling_gemma4 as mg
    from ikengine.models import load_model
    d, hf = tiny[kind]
    n = len(_hf_layers(hf))
    ks, vs = [0.004 * (i + 2) for i in range(n)], [0.003 * (i + 3) for i in range(n)]
    kd = tmp_path / "with-kv-scales"
    shutil.copytree(d, kd)
    prefix = "model.language_model." if kind == "moe" else "model."
    save_file({**{f"{prefix}layers.{i}.self_attn.k_proj.k_scale": torch.tensor(ks[i]) for i in range(n)},
               **{f"{prefix}layers.{i}.self_attn.v_proj.v_scale": torch.tensor(vs[i]) for i in range(n)}},
              str(kd / "kv_scales.safetensors"))
    m = load_model(str(kd), device="cpu", max_seq=40, kv="fp8", prefill_chunk=4)
    assert m.kv_dtype == torch.float8_e4m3fn and not m.numerics["as_shipped"]
    assert m.numerics["changed"][0].startswith("kv: fp8 e4m3, k/v scales from the checkpoint")

    current = {}
    handles = []
    for i, layer in enumerate(_hf_layers(hf)):
        handles.append(layer.self_attn.register_forward_pre_hook(lambda mod, a, i=i: current.update(i=i)))
        if hasattr(layer.self_attn, "v_norm"):
            handles.append(layer.self_attn.v_norm.register_forward_hook(
                lambda mod, a, out, i=i: _fp8_round_trip(out, vs[i])))
    rope = mg.apply_rotary_pos_emb
    heads = hf.config.get_text_config().num_attention_heads

    def rope_then_round(x, cos, sin, unsqueeze_dim=1):
        y = rope(x, cos, sin, unsqueeze_dim)
        return y if x.shape[2] == heads else _fp8_round_trip(y, ks[current["i"]])   # keys have fewer heads
    monkeypatch.setattr(mg, "apply_rotary_pos_emb", rope_then_round)
    try:
        ids = _greedy_against(m, m.alloc_state(40), PROMPT, lambda ids: _ref_logits(hf, ids), tol=dict(rtol=2e-4, atol=2e-4))
    finally:
        for h in handles:
            h.remove()
    plain = load_model(d, device="cpu", max_seq=40)
    st = plain.alloc_state(40)
    assert (plain.prefill(torch.tensor(ids), st) - m.prefill(torch.tensor(ids), m.alloc_state(40))).abs().max() > 1e-3


# --- NVFP4 experts ------------------------------------------------------------------------------------
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _nvfp4(w):
    """Quantize w [N, K] as an NVFP4 export stores it: e2m1 codes two per byte (low nibble first), an e4m3 scale
    per 16 values and an fp32 global scale. Returns (packed, block_scale, scale_2, dequantized)."""
    import torch
    n, k = w.shape
    lut = torch.tensor(E2M1)
    s2 = w.abs().max() / (6.0 * 448.0)
    blocks = w.reshape(n, k // 16, 16)
    bs = (blocks.abs().amax(-1) / 6.0 / s2).to(torch.float8_e4m3fn)
    scale = (bs.float() * s2).clamp(min=1e-30)[..., None]
    mag = (blocks / scale).abs()
    code = (mag[..., None] - lut).abs().argmin(-1) + 8 * (blocks < 0)
    deq = torch.cat([lut, -lut])[code] * scale
    code = code.reshape(n, k).to(torch.uint8)
    return (code[:, 0::2] | (code[:, 1::2] << 4)), bs, s2.float(), deq.reshape(n, k)


def _decode_nvfp4(qw):
    import torch
    lut = torch.cat([torch.tensor(E2M1), -torch.tensor(E2M1)])
    lo, hi = (qw.weight & 15).long(), (qw.weight >> 4).long()
    codes = torch.stack([lo, hi], -1).reshape(qw.shape)
    return lut[codes] * (qw.scale.float() * qw.scale_2.float()).repeat_interleave(16, -1)


@pytest.fixture(scope="module")
def nvfp4(tiny, tmp_path_factory):
    """The MoE model with its experts stored as NVFP4 (one linear per expert, as quantized exports store them),
    plus an HF reference whose experts hold the dequantized values."""
    import torch
    from safetensors.torch import load_file, save_file
    d, _ = tiny["moe"]
    root = tmp_path_factory.mktemp("gemma4-nvfp4")
    qdir, ddir = root / "nvfp4", root / "dequant"
    shutil.copytree(d, qdir)
    shutil.copytree(d, ddir)
    sd = load_file(str(Path(d) / "model.safetensors"))
    q, dq = dict(sd), dict(sd)
    p, inter = "model.language_model.layers.", MOE["moe_intermediate_size"]
    for i in range(MOE["num_hidden_layers"]):
        gu, dn = q.pop(f"{p}{i}.experts.gate_up_proj"), q.pop(f"{p}{i}.experts.down_proj")
        dgu, ddn = torch.empty_like(gu), torch.empty_like(dn)
        for e in range(MOE["num_experts"]):
            for name, w, out in (("gate_proj", gu[e, :inter], dgu[e, :inter]), ("up_proj", gu[e, inter:], dgu[e, inter:]),
                                 ("down_proj", dn[e], ddn[e])):
                packed, bs, s2, deq = _nvfp4(w)
                pre = f"{p}{i}.experts.{e}.{name}."
                q.update({pre + "weight": packed, pre + "weight_scale": bs, pre + "weight_scale_2": s2,
                          pre + "input_scale": torch.tensor(4.0 / (6.0 * 448.0))})
                out.copy_(deq)
        dq[f"{p}{i}.experts.gate_up_proj"], dq[f"{p}{i}.experts.down_proj"] = dgu, ddn
    save_file(q, str(qdir / "model.safetensors"))
    save_file(dq, str(ddir / "model.safetensors"))
    (qdir / "hf_quant_config.json").write_text(json.dumps({"quantization": {
        "quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8", "group_size": 16,
        "exclude_modules": ["lm_head", "model.language_model.layers.*.self_attn*",
                            "model.language_model.layers.*.mlp*", "model.language_model.layers.*.router*"]}}))
    import transformers as tf
    return str(qdir), _load_ref(tf.Gemma4ForConditionalGeneration, ddir)


def _qlinear_supports_nvfp4() -> bool:
    import torch
    from ikengine.loader import QWeight
    from ikengine.qlinear import qlinear
    w = QWeight("nvfp4", torch.zeros(16, 8, dtype=torch.uint8), (16, 16),
                scale=torch.ones(16, 1).to(torch.float8_e4m3fn), scale_2=torch.tensor(1.0),
                input_scale=torch.tensor(1.0), block=(1, 16))
    try:
        qlinear(torch.zeros(1, 16), w)
    except NotImplementedError:
        return False
    return True


def _use_w4a16_stand_in(monkeypatch):
    """Route NVFP4 linears through a test-local W4A16 linear (16-bit x times the decoded weight), so these tests
    run before the skeleton's NVFP4 qlinear exists; every other kind goes through the real qlinear."""
    import torch
    real = g4.qlinear

    def w4a16(x, w, act="as_shipped"):
        if w.kind != "nvfp4":
            return real(x, w, act)
        return torch.nn.functional.linear(x.float(), _decode_nvfp4(w)).to(x.dtype)
    monkeypatch.setattr(g4, "qlinear", w4a16)


@needs_hf
def test_nvfp4_experts_are_stacked_and_gathered_as_stored(nvfp4, monkeypatch):
    """The per-expert NVFP4 layout through the prefill loop and the graph-safe decode gather."""
    import torch
    from ikengine.models import load_model
    _use_w4a16_stand_in(monkeypatch)
    qdir, ref = nvfp4
    m = load_model(qdir, device="cpu", max_seq=40, act="a16", kv="bf16", prefill_chunk=4)
    assert m.layers[0].moe.gate.kind == "nvfp4" and m.layers[0].moe.gate.t["weight"].dtype == torch.uint8
    assert "60 nvfp4 linears" in m.numerics["weights"]
    assert not m.numerics["as_shipped"] and len(m.numerics["changed"]) == 2      # a16 and 16-bit KV, both declared
    assert m.numerics["changed"][0] == f"activations: a16 instead of {W4A4} for 60 linears"
    assert m.resolved["expert_input_scales"] == {"one_value_per_layer": {"gate": True, "up": True, "down": True},
                                                 "gate_equals_up": True}
    assert m.numerics["expert_input_scales"].startswith("unused (a16")
    _greedy_against(m, m.alloc_state(40), PROMPT, lambda ids: _ref_logits(ref, ids))


@needs_hf
def test_weight_only_nvfp4_ships_16_bit_activations(nvfp4, tmp_path, monkeypatch):
    """The same experts exported without input scales (W4A16): 16-bit activations are what the checkpoint ships,
    so act="a16" is no declared change and nothing claims W4A4."""
    from safetensors.torch import load_file, save_file
    from ikengine.models import load_model
    qdir, ref = nvfp4
    wdir = tmp_path / "w4a16"
    shutil.copytree(qdir, wdir)
    sd = load_file(str(wdir / "model.safetensors"))
    save_file({k: v for k, v in sd.items() if not k.endswith(".input_scale")}, str(wdir / "model.safetensors"))
    hq = json.loads((wdir / "hf_quant_config.json").read_text())
    del hq["quantization"]["kv_cache_quant_algo"]                    # a 16-bit cache as shipped: tight compare
    (wdir / "hf_quant_config.json").write_text(json.dumps(hq))
    if not _qlinear_supports_nvfp4():
        _use_w4a16_stand_in(monkeypatch)
    for act in ("as_shipped", "a16"):
        m = load_model(str(wdir), device="cpu", max_seq=40, act=act, prefill_chunk=4)
        assert m.numerics["as_shipped"] and m.numerics["changed"] == []
        assert "60 weight-only nvfp4 linears (W4A16)" in m.numerics["activations"]
        assert "W4A4" not in m.numerics["activations"]
        assert m.resolved["expert_input_scales"] is None and "expert_input_scales" not in m.numerics
    _greedy_against(m, m.alloc_state(40), PROMPT, lambda ids: _ref_logits(ref, ids))


@needs_hf
def test_nvfp4_as_shipped(nvfp4):
    """W4A4 with the stored input scales and the declared FP8 KV cache, through the skeleton's qlinear. Skipped
    until qlinear implements NVFP4."""
    import torch
    from ikengine.models import load_model
    if not _qlinear_supports_nvfp4():
        pytest.skip("qlinear has no NVFP4 path yet (the skeleton core provides it)")
    qdir, ref = nvfp4
    a16 = load_model(qdir, device="cpu", max_seq=40, act="a16", kv="bf16", prefill_chunk=4)
    _greedy_against(a16, a16.alloc_state(40), PROMPT, lambda ids: _ref_logits(ref, ids), tol=dict(rtol=1e-3, atol=1e-3))

    m = load_model(qdir, device="cpu", max_seq=40, prefill_chunk=4)
    assert m.numerics["as_shipped"] and m.kv_dtype == torch.float8_e4m3fn
    assert "W4A4" in m.numerics["activations"] and m.numerics["kv"].startswith("fp8 e4m3, scale 1.0")
    assert m.numerics["expert_input_scales"].endswith("computes the same")     # one stored value everywhere

    def fresh(ids):                       # the same ids prefilled from scratch: decode must agree with it
        return m.prefill(torch.tensor(ids), m.alloc_state(40)).clone()
    state = m.alloc_state(40)
    first = m.prefill(torch.tensor(PROMPT), state).clone()
    ids = _greedy_against(m, state, PROMPT, fresh, tol=dict(rtol=1e-3, atol=1e-3))
    w4a16 = a16.prefill(torch.tensor(PROMPT), a16.alloc_state(40))
    assert torch.nn.functional.cosine_similarity(first, w4a16, dim=0) > 0.95
    assert len(ids) == len(PROMPT) + 8
