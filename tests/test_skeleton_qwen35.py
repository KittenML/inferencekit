"""The engine skeleton's Qwen3.5 / Qwen3.8 hybrid model module (ikengine/models/qwen3_5.py), checked on CPU.

Needs torch; the Hugging Face comparisons also need transformers with qwen3_5. Both are skipped cleanly without
them (CI has no torch). Locally, with an env that has both:
    python -m pytest -q --noconftest tests/test_skeleton_qwen35.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

SKELETON = Path(__file__).resolve().parents[1] / "plugin" / "skills" / "optimizing-kernels" / "skeleton"
sys.path.insert(0, str(SKELETON))

from ikengine import engine_api, models  # noqa: E402
from ikengine.loader import Checkpoint, QWeight  # noqa: E402
from ikengine.models import qwen3_5 as q35  # noqa: E402
from ikengine.qlinear import qlinear  # noqa: E402

LAYERS = ["linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention",
          "full_attention"]
PROMPT = 70     # longer than one prefill chunk (64)
STEPS = 8       # greedy decode steps compared with Hugging Face


def _tr():
    tr = pytest.importorskip("transformers")
    if not hasattr(tr, "Qwen3_5ForCausalLM"):
        pytest.skip("this transformers has no qwen3_5")
    return tr


def text_config(**kw):
    """A tiny Qwen3.5 text config: GDN and full attention mixed, 2 key heads serving 4 value heads, 4 query heads
    over 2 KV heads, partial RoPE (8 of 32 dims)."""
    base = dict(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=len(LAYERS),
                num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2,
                linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=24, linear_conv_kernel_dim=4,
                layer_types=LAYERS, max_position_embeddings=256, tie_word_embeddings=False, eos_token_id=[1, 2],
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
                initializer_range=0.1)
    base.update(kw)
    return _tr().Qwen3_5TextConfig(**base)


def _perturb(model, seed=1):
    """HF initialises the (1 + w) norms to 0 and the gated norm to 1; move every 1-D parameter so the tests see
    the norm forms and dt_bias at work. A_log is redrawn so A = exp(A_log) is in [0.01, 0.3]: HF's init (A up to
    16) forgets the DeltaNet state within a token or two, while these heads keep it for tens of tokens, across
    the 64-token prefill chunks and into decode, as trained models do."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("A_log"):
                p.copy_(torch.empty(p.shape).uniform_(0.01, 0.3, generator=g).log())
            elif p.ndim == 1:
                p.add_(0.2 * torch.randn(p.shape, generator=g))
    return model


def hf_causal_lm(tmp_path, dtype=torch.float32, **cfg):
    torch.manual_seed(0)
    model = _perturb(_tr().Qwen3_5ForCausalLM(text_config(**cfg)).eval()).to(dtype)
    model.save_pretrained(str(tmp_path))
    return model


def prompt(n=PROMPT, seed=2):
    return torch.randint(3, 128, (n,), generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def hf_last_logits(model, ids):
    return model(torch.tensor([list(ids)])).logits[0, -1].float()


def run_greedy(model, ids, steps=STEPS, max_seq=96):
    """Prefill, then `steps` greedy decode steps. Returns (tokens, the logits before each token: steps+1 rows)."""
    st = model.alloc_state(max_seq)
    model.reset(st)
    logits = [model.prefill(ids, st).clone()]
    toks = []
    for _ in range(steps):
        toks.append(int(logits[-1].argmax()))
        pos = torch.tensor([len(ids) + len(toks) - 1], dtype=torch.int32)
        logits.append(model.decode(torch.tensor([toks[-1]]), pos, st).clone())
    return toks, logits


def assert_matches_hf(ours, hf, ids, atol, rtol=0.0, steps=STEPS):
    """Our prefill and every greedy decode step against a fresh HF forward over the whole sequence (no HF cache),
    and the same greedy tokens."""
    toks, logits = run_greedy(ours, ids, steps)
    seq = list(ids.tolist())
    for i, lg in enumerate(logits):
        ref = hf_last_logits(hf, seq)
        torch.testing.assert_close(lg, ref, atol=atol, rtol=rtol, msg=lambda m: f"step {i}: {m}")
        if i < len(toks):
            assert toks[i] == int(ref.argmax()), f"step {i}: greedy token {toks[i]} != HF {int(ref.argmax())}"
            seq.append(toks[i])
    return toks


# --- the delta rule ------------------------------------------------------------------------------------------
def delta_rule_inputs(T, H, dk, dv, decay, seed):
    """Raw q, k (not normalised), v, log-decay in [-decay, 0], beta in [0, 1], and an initial state."""
    g = torch.Generator().manual_seed(seed)
    q, k, v = torch.randn(T, H, dk, generator=g), torch.randn(T, H, dk, generator=g), torch.randn(T, H, dv, generator=g)
    return q, k, v, -decay * torch.rand(T, H, generator=g), torch.rand(T, H, generator=g), \
        torch.randn(H, dk, dv, generator=g)


# decay 3: the state lasts a token or two. decay 0.01: the decay alone would keep it for hundreds of tokens, so
# what one chunk hands the next carries real weight.
@pytest.mark.parametrize("T,chunk,decay", [(1, 64, 3.0), (5, 4, 3.0), (64, 64, 3.0), (130, 64, 3.0), (37, 16, 3.0),
                                           (130, 64, 0.01), (200, 64, 0.01)])
def test_chunked_delta_rule_equals_token_by_token(T, chunk, decay):
    H, dk, dv = 3, 8, 5
    q, k, v, logdecay, beta, S0 = delta_rule_inputs(T, H, dk, dv, decay, seed=T)
    q, k = q35.l2norm(q) * dk ** -0.5, q35.l2norm(k)

    o, S = q35.delta_rule_chunked(q, k, v, logdecay, beta, S0.clone(), chunk=chunk)
    S_ref = S0.clone()
    o_ref = torch.stack([q35.delta_rule_step(q[t], k[t], v[t], logdecay[t], beta[t], S_ref) for t in range(T)])
    torch.testing.assert_close(o, o_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(S, S_ref, atol=1e-5, rtol=1e-5)
    if decay < 1 and T > chunk:             # the long-memory case: 16 tokens past a chunk edge, the state handed
        o_cut, _ = q35.delta_rule_chunked(q[chunk:], k[chunk:], v[chunk:], logdecay[chunk:], beta[chunk:],
                                          torch.zeros_like(S0), chunk=chunk)       # over still changes the output
        assert float((o[chunk + 16:] - o_cut[16:]).abs().max()) > 1e-2


@pytest.mark.parametrize("decay", [1e-3, 0.05, 3.0])
def test_delta_rule_matches_the_hf_torch_kernels(decay):
    """Both forms against transformers' own torch_chunk_gated_delta_rule and torch_recurrent_gated_delta_rule
    (q/k L2-normalised inside, as the model calls them), from a nonzero state, over several chunks."""
    _tr()
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf_mod
    T, H, dk, dv = 200, 4, 16, 8
    q, k, v, logdecay, beta, S0 = delta_rule_inputs(T, H, dk, dv, decay, seed=11)
    args = (q[None], k[None], v[None], logdecay[None], beta[None])
    o_hf, S_hf = hf_mod.torch_chunk_gated_delta_rule(*args, chunk_size=64, initial_state=S0[None].clone(),
                                                     output_final_state=True, use_qk_l2norm_in_kernel=True)
    o_hr, S_hr = hf_mod.torch_recurrent_gated_delta_rule(*args, initial_state=S0[None].clone(),
                                                         output_final_state=True, use_qk_l2norm_in_kernel=True)
    qn, kn = q35.l2norm(q) * dk ** -0.5, q35.l2norm(k)
    o, S = q35.delta_rule_chunked(qn, kn, v, logdecay, beta, S0.clone())
    S_step = S0.clone()
    o_step = torch.stack([q35.delta_rule_step(qn[t], kn[t], v[t], logdecay[t], beta[t], S_step) for t in range(T)])
    for mine, mine_S in ((o, S), (o_step, S_step)):
        for ref, ref_S in ((o_hf[0], S_hf[0]), (o_hr[0], S_hr[0])):
            torch.testing.assert_close(mine, ref, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(mine_S, ref_S, atol=1e-5, rtol=1e-5)


def test_delta_rule_step_is_the_written_recurrence():
    """S <- exp(g) S; S <- S + k (beta (v - S^T k))^T; o = S^T q, for one head written out by hand."""
    g = torch.Generator().manual_seed(0)
    q, k, v = torch.randn(1, 4, generator=g), torch.randn(1, 4, generator=g), torch.randn(1, 3, generator=g)
    S0 = torch.randn(1, 4, 3, generator=g)
    lg, beta = torch.tensor([-0.5]), torch.tensor([0.7])
    S = S0.clone()
    o = q35.delta_rule_step(q, k, v, lg, beta, S)
    Sd = S0[0] * torch.exp(lg)
    S_ref = Sd + torch.outer(k[0], 0.7 * (v[0] - Sd.T @ k[0]))
    torch.testing.assert_close(S[0], S_ref)
    torch.testing.assert_close(o[0], S_ref.T @ q[0])


# --- against Hugging Face ------------------------------------------------------------------------------------
def test_causal_lm_fp32_matches_hf_prefill_and_greedy_decode(tmp_path):
    hf = hf_causal_lm(tmp_path)
    ours = models.load_model(str(tmp_path), device="cpu", max_seq=96)
    assert isinstance(ours, q35.Qwen35Model) and isinstance(ours, engine_api.Model)
    assert ours.vocab_size == 128 and ours.eos_token_ids == frozenset({1, 2})
    assert assert_matches_hf(ours, hf, prompt(), atol=1e-4)


def test_conditional_generation_checkpoint_uses_the_text_path(tmp_path):
    """Qwen3_5ForConditionalGeneration (what the Qwen3.8 checkpoints are): tensors under model.language_model.,
    a vision tower that is not loaded, text positions equal on every M-RoPE axis."""
    tr = _tr()
    vision = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=4,
                  temporal_patch_size=2, spatial_merge_size=2, num_position_embeddings=16)
    tc = text_config(rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                                      "mrope_section": [1, 1, 2], "mrope_interleaved": True})
    cfg = tr.Qwen3_5Config(text_config=tc.to_dict(), vision_config=vision, tie_word_embeddings=False)
    torch.manual_seed(0)
    hf = _perturb(tr.Qwen3_5ForConditionalGeneration(cfg).eval())
    hf.save_pretrained(str(tmp_path))
    ck = Checkpoint(str(tmp_path))
    assert ck.config["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert ck.has("model.language_model.embed_tokens.weight") and any(n.startswith("model.visual.") for n in ck.names())
    ours = models.load_model(str(tmp_path), device="cpu", max_seq=96)
    assert assert_matches_hf(ours, hf, prompt(seed=3), atol=1e-4, steps=4)


@pytest.mark.parametrize("n", [1, 64])
def test_prompt_lengths_at_chunk_edges_match_hf(tmp_path, n):
    """A one-token prompt (conv padding only) and a prompt of exactly one prefill chunk."""
    hf = hf_causal_lm(tmp_path)
    ours = models.load_model(str(tmp_path), device="cpu", max_seq=96)
    assert assert_matches_hf(ours, hf, prompt(n=n, seed=6), atol=1e-4, steps=3)


def test_prefill_attention_in_query_chunks_matches_hf(tmp_path, monkeypatch):
    """Prefill attention Q_CHUNK queries at a time (512 in use; 16 here, so 70 tokens make 4 full chunks and a
    partial one), each against the keys up to its own end."""
    monkeypatch.setattr(q35, "Q_CHUNK", 16)
    hf = hf_causal_lm(tmp_path)
    ours = models.load_model(str(tmp_path), device="cpu", max_seq=96)
    assert assert_matches_hf(ours, hf, prompt(seed=7), atol=1e-4, steps=2)


def test_attend_groups_query_heads_without_repeating_kv():
    """attend() against plain per-head causal attention with K/V repeated to the query heads."""
    g = torch.Generator().manual_seed(0)
    q, k, v = torch.randn(6, 5, 8, generator=g), torch.randn(2, 9, 8, generator=g), torch.randn(2, 9, 8, generator=g)
    qpos, kpos = torch.arange(4, 9), torch.arange(9)               # 5 queries at positions 4..8 over 9 keys
    out = q35.attend(q, k, v, qpos, kpos, 0.3)
    K, V = k.repeat_interleave(3, 0), v.repeat_interleave(3, 0)    # query head h reads KV head h // 3
    s = (q @ K.transpose(1, 2) * 0.3).masked_fill(kpos[None, None, :] > qpos[None, :, None], float("-inf"))
    torch.testing.assert_close(out, s.softmax(-1) @ V, atol=1e-6, rtol=1e-5)


def test_bf16_checkpoint_numerics_and_closeness_to_hf(tmp_path):
    """As the real checkpoints ship: BF16 activations and KV cache, conv state BF16, recurrent state fp32.

    BF16 rounding differs with the order of operations, so two BF16 runs are not compared with each other. Against
    the fp32 reference, our error must be the same size as Hugging Face's own BF16 run on the same tokens."""
    hf32 = hf_causal_lm(tmp_path / "f32")
    hf16 = hf_causal_lm(tmp_path / "b16", dtype=torch.bfloat16)
    ours = models.load_model(str(tmp_path / "b16"), device="cpu", max_seq=96)
    st = ours.alloc_state(96)
    assert (st.k.dtype, st.v.dtype, st.conv.dtype, st.rec.dtype) == (torch.bfloat16,) * 3 + (torch.float32,)
    n = ours.numerics
    assert (n["kv"], n["conv_state"], n["changed"]) == ("bf16", "bf16", [])
    assert n["state"].startswith("fp32") and n["weights"].startswith("as stored: bf16")
    ids = prompt()
    toks, logits = run_greedy(ours, ids, steps=6)
    seq = ids.tolist()
    ours_err, hf_err = [], []
    for i, lg in enumerate(logits):
        ref16, ref32 = hf_last_logits(hf16, seq), hf_last_logits(hf32, seq)
        ours_err.append(float((lg - ref32).abs().mean()))
        hf_err.append(float((ref16 - ref32).abs().mean()))
        seq += toks[i:i + 1]
    assert sum(ours_err) < 2.0 * sum(hf_err), (ours_err, hf_err)


def test_kv_bf16_flag_on_an_fp32_checkpoint_is_declared(tmp_path):
    hf = hf_causal_lm(tmp_path)
    ours = models.load_model(str(tmp_path), device="cpu", max_seq=96, kv="bf16")
    assert ours.alloc_state(8).k.dtype == torch.bfloat16
    assert ours.numerics["kv"] == "bf16" and ours.numerics["changed"]
    ids = prompt()
    _, logits = run_greedy(ours, ids, steps=2)
    torch.testing.assert_close(logits[0], hf_last_logits(hf, ids.tolist()), atol=0.05, rtol=0.0)


def test_declared_fp8_kv_cache_uses_stored_scales(tmp_path):
    """A checkpoint that declares an FP8 KV cache gets e4m3 storage with its per-tensor k/v scales (1.0 if none;
    one stored scale serves both k and v, as vLLM loads them). Each decode step writes K and V at its position
    with those scales: its logits equal an FP8-KV prefill of the same tokens, which quantizes every position the
    same way, up to the chunked-vs-step delta rule (about 1e-6)."""
    from safetensors.torch import save_file
    hf = hf_causal_lm(tmp_path)
    save_file({"model.layers.1.self_attn.k_scale": torch.tensor(0.05), "model.layers.1.self_attn.v_scale":
               torch.tensor(0.04), "model.layers.5.self_attn.k_proj.k_scale": torch.tensor(0.06)},
              str(tmp_path / "kv_scales.safetensors"))
    ck = Checkpoint(str(tmp_path))
    ck.quant["kv"] = "fp8"
    ours = q35.Qwen35Model.from_checkpoint(ck, "cpu", 96)
    at = [L.mixer for L in ours.layers if L.kind == q35.FULL]
    assert [(a.k_scale, a.v_scale) for a in at] == [(pytest.approx(0.05), pytest.approx(0.04)),
                                                    (pytest.approx(0.06), pytest.approx(0.06))]
    assert ours.alloc_state(8).k.dtype == torch.float8_e4m3fn
    assert ours.numerics["kv"] == "fp8 e4m3 with the stored k/v scales" and ours.numerics["changed"] == []
    ids = prompt()
    toks, logits = run_greedy(ours, ids, steps=4)
    ref = hf_last_logits(hf, ids.tolist())
    spread = float(ref.max() - ref.min())
    assert float((logits[0] - ref).abs().max()) < 0.1 * spread                 # e4m3 K/V: close to HF, not equal
    for i in range(1, len(logits)):
        want = ours.prefill(torch.cat([ids, torch.tensor(toks[:i])]), ours.alloc_state(96))
        assert float((logits[i] - want).abs().max()) < 1e-3 * float(want.max() - want.min()), f"decode step {i}"
    ck = Checkpoint(str(tmp_path))
    ck.quant["kv"] = "fp8"
    assert q35.Qwen35Model.from_checkpoint(ck, "cpu", 96, kv="bf16").numerics["changed"]   # opting out is declared


def test_kv_scale_names():
    """k/v scales by each stored name; the older shared kv_scale; one of k or v alone serves both; none -> None."""
    def ck_with(**stored):
        return SimpleNamespace(has=lambda n: n in stored, tensor=lambda n, **kw: torch.tensor(stored[n]))
    a = "model.layers.3.self_attn."
    assert q35._kv_scales(ck_with(), a) == (None, None)
    assert q35._kv_scales(ck_with(**{a + "attn.k_scale": 2.0, a + "attn.v_scale": 3.0}), a) == (2.0, 3.0)
    assert q35._kv_scales(ck_with(**{a + "v_proj.v_scale": 0.5}), a) == (0.5, 0.5)
    assert q35._kv_scales(ck_with(**{a + "k_proj.kv_scale": 0.25}), a) == (0.25, 0.25)
    assert q35._kv_scales(ck_with(**{a + "k_scale": 2.0, a + "kv_scale": 9.0}), a) == (2.0, 2.0)


# --- the engine contract -------------------------------------------------------------------------------------
def test_state_is_static_and_requests_do_not_leak(tmp_path):
    hf_causal_lm(tmp_path)
    ours = models.load_model(str(tmp_path), device="cpu", max_seq=96)
    st = ours.alloc_state(96)
    ptrs = [t.data_ptr() for t in (st.k, st.v, st.conv, st.rec, st.logits)]
    a, b = prompt(seed=4), prompt(n=9, seed=5)

    fresh = ours.alloc_state(96)
    want = ours.prefill(b, fresh).clone()
    ours.prefill(a, st)
    for i in range(3):
        out = ours.decode(torch.tensor([7 + i]), torch.tensor([PROMPT + i], dtype=torch.int32), st)
        assert out.data_ptr() == st.logits.data_ptr() and out.dtype == torch.float32 and out.shape == (128,)
    ours.reset(st)
    torch.testing.assert_close(ours.prefill(b, st), want, atol=0.0, rtol=0.0)
    ours.prefill(a, st)                                            # prefill alone also starts clean
    torch.testing.assert_close(ours.prefill(b, st), want, atol=0.0, rtol=0.0)
    assert [t.data_ptr() for t in (st.k, st.v, st.conv, st.rec, st.logits)] == ptrs
    with pytest.raises(ValueError):
        ours.prefill(torch.zeros(97, dtype=torch.long), st)
    with pytest.raises(ValueError):
        ours.alloc_state(97)


def test_registry_flags_and_tied_head(tmp_path):
    assert models.model_class("Qwen3_5ForCausalLM") is q35.Qwen35Model
    assert models.model_class("Qwen3_5ForConditionalGeneration") is q35.Qwen35Model
    hf = hf_causal_lm(tmp_path, tie_word_embeddings=True)
    ours = models.load_model(str(tmp_path), device="cpu", max_seq=96)
    assert ours.lm_head.weight is ours.embed and "tied" in ours.numerics["lm_head"]
    assert assert_matches_hf(ours, hf, prompt(), atol=1e-4, steps=2)
    with pytest.raises(ValueError):
        models.load_model(str(tmp_path), device="cpu", max_seq=8, act="a8")
    with pytest.raises(ValueError):
        models.load_model(str(tmp_path), device="cpu", max_seq=8, kv="fp8")


def test_activation_labels_follow_the_stored_scales():
    w8, w4 = torch.zeros(4, 4).to(torch.float8_e4m3fn), torch.zeros(4, 2, dtype=torch.uint8)
    s = torch.tensor(0.5)
    cases = [(QWeight("bf16", torch.zeros(4, 4), (4, 4)), ("bf16", False)),
             (QWeight("fp8_block", w8, (4, 4), scale=s, block=(128, 128)),
              ("fp8 e4m3, dynamic per token in groups of 128", True)),
             (QWeight("fp8_tensor", w8, (4, 4), scale=s), ("fp8 e4m3, dynamic per tensor", True)),
             (QWeight("fp8_tensor", w8, (4, 4), scale=s, input_scale=s),
              ("fp8 e4m3, static per-tensor input_scale", True)),
             (QWeight("nvfp4", w4, (4, 4), scale=s, scale_2=s), ("bf16 (weight-only nvfp4)", False)),
             (QWeight("nvfp4", w4, (4, 4), scale=s, scale_2=s, input_scale=s),
              ("nvfp4 e2m1, static input_scale, groups of 16", True))]
    for w, want in cases:
        assert q35.act_scheme(w, "bf16") == want, w.kind


@pytest.mark.parametrize("act", ["as_shipped", "a16"])
def test_numerics_activations_per_stored_layer(tmp_path, act):
    """ENGINE.json's activation labels come from each loaded weight (as nvidia/Qwen3.8-27B-NVFP4 mixes them:
    per-tensor FP8 attention with static input scales, NVFP4 MLP). The weights are stand-ins; nothing runs."""
    hf_causal_lm(tmp_path)
    ck = Checkpoint(str(tmp_path))
    real = ck.linear

    def linear(prefix, bias=False):
        w = real(prefix, bias=bias)
        n, k = w.shape
        one = torch.tensor(1.0)
        if prefix.rsplit(".", 1)[-1] in ("q_proj", "k_proj", "v_proj", "o_proj"):
            return QWeight("fp8_tensor", w.weight.to(torch.float8_e4m3fn), (n, k), scale=one, input_scale=one)
        if prefix.endswith("mlp.down_proj"):                        # one MLP projection W4A4, the others weight-only
            return QWeight("nvfp4", torch.zeros(n, k // 2, dtype=torch.uint8), (n, k), scale=one, scale_2=one,
                           input_scale=one)
        if ".mlp." in prefix:
            return QWeight("nvfp4", torch.zeros(n, k // 2, dtype=torch.uint8), (n, k), scale=one, scale_2=one)
        return w

    ck.linear = linear
    n = q35.Qwen35Model.from_checkpoint(ck, "cpu", 16, act=act).numerics
    assert n["weights"] == "as stored: fp32 x21, fp8_tensor x8, nvfp4 x18"
    if act == "as_shipped":
        assert n["activations"] == {"fp32": "fp32", "fp8_tensor": "fp8 e4m3, static per-tensor input_scale",
                                    "nvfp4": "fp32 (weight-only nvfp4); nvfp4 e2m1, static input_scale, groups of 16"}
        assert n["changed"] == []
    else:
        assert n["activations"] == {"fp32": "fp32", "fp8_tensor": "fp32 (a16)", "nvfp4": "fp32 (a16)"}
        assert n["changed"] == ["activations kept 16-bit (a16) for the fp8_tensor, nvfp4 layers"]


def test_config_defaults_and_refusals():
    tc = {"hidden_size": 64, "num_hidden_layers": 8, "num_attention_heads": 4, "num_key_value_heads": 2,
          "head_dim": 32, "linear_num_key_heads": 2, "linear_num_value_heads": 6}
    c = q35.Qwen35Config.from_checkpoint(SimpleNamespace(config={"text_config": tc}, text_config=tc))
    assert c.layer_types == ("linear_attention",) * 3 + ("full_attention",) + ("linear_attention",) * 3 + \
        ("full_attention",)
    assert (c.rot_dim, c.attn_gate, c.tie, c.conv_kernel) == (8, True, False, 4)    # HF's defaults
    for bad in ({"rope_parameters": {"rope_type": "yarn", "factor": 4.0}}, {"hidden_act": "gelu"},
                {"linear_num_value_heads": 5}, {"layer_types": ["mamba"] * 8}, {"mamba_ssm_dtype": "bfloat16"},
                {"output_gate_type": "sigmoid"}):
        t = {**tc, **bad}
        with pytest.raises(NotImplementedError):
            q35.Qwen35Config.from_checkpoint(SimpleNamespace(config=t, text_config=t))


# The text_config fields of Qwen/Qwen3.8-27B-FP8 (config.json as published; layer_types written out there too).
QWEN38_27B = {"attention_bias": False, "attn_output_gate": True, "eos_token_id": 248044, "full_attention_interval": 4,
              "head_dim": 256, "hidden_act": "silu", "hidden_size": 5120, "intermediate_size": 17408,
              "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
              "linear_num_value_heads": 48, "linear_value_head_dim": 128, "mamba_ssm_dtype": "float32",
              "num_attention_heads": 24, "num_hidden_layers": 64, "num_key_value_heads": 4,
              "output_gate_type": "swish", "partial_rotary_factor": 0.25, "rms_norm_eps": 1e-06,
              "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10],
                                  "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
              "tie_word_embeddings": False, "vocab_size": 248320,
              "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 16}


def test_qwen38_27b_config():
    top = {"architectures": ["Qwen3_5ForConditionalGeneration"], "tie_word_embeddings": False,
           "text_config": QWEN38_27B}
    c = q35.Qwen35Config.from_checkpoint(SimpleNamespace(config=top, text_config=QWEN38_27B))
    assert c.layer_types.count("full_attention") == 16 and c.layer_types[3::4] == ("full_attention",) * 16
    assert (c.heads, c.kv_heads, c.head_dim, c.rot_dim, c.rope_theta) == (24, 4, 256, 64, 1e7)
    assert (c.k_heads, c.v_heads, c.k_dim, c.v_dim, c.conv_dim) == (16, 48, 128, 128, 10240)   # conv1d [10240, 1, 4]
    assert (c.attn_gate, c.tie, c.vocab) == (True, False, 248320)


# --- FP8 block-128 weights (the Qwen/Qwen3.8-27B-FP8 format) -------------------------------------------------
def _fp8_qlinear_available() -> bool:
    w = QWeight("fp8_block", torch.zeros(128, 128).to(torch.float8_e4m3fn), (128, 128),
                scale=torch.ones(1, 1), block=(128, 128))
    try:
        qlinear(torch.zeros(1, 128), w)
    except NotImplementedError:
        return False
    return True


def fp8_block_quantize(w, block=128):
    """[N, K] -> (float8_e4m3fn [N, K], weight_scale_inv [ceil(N/128), ceil(K/128)]), amax / 448 per block. The
    scales are BF16, as Qwen/Qwen3.8-27B-FP8 stores them."""
    N, K = w.shape
    nb, kb = -(-N // block), -(-K // block)
    pad = torch.zeros(nb * block, kb * block)
    pad[:N, :K] = w.float()
    blocks = pad.view(nb, block, kb, block)
    scale = (blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12) / q35.FP8_MAX).to(torch.bfloat16)
    q = (blocks / scale.float()[:, None, :, None]).clamp(-q35.FP8_MAX, q35.FP8_MAX).to(torch.float8_e4m3fn)
    return q.view(nb * block, kb * block)[:N, :K].contiguous(), scale


def fp8_dequantize(q, scale, block=128):
    s = scale.float().repeat_interleave(block, 0).repeat_interleave(block, 1)[:q.shape[0], :q.shape[1]]
    return q.float() * s


QUANTIZED = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj",
             "down_proj")


def fp8_act_fake_quant(x, group=128):
    """The declared dynamic activation scheme, written out: per token, per group of 128 along K, scale amax / 448,
    round to e4m3 and back."""
    g = x.float().reshape(*x.shape[:-1], -1, group)
    s = g.abs().amax(-1, keepdim=True).clamp(min=1e-10) / q35.FP8_MAX
    return ((g / s).clamp(-q35.FP8_MAX, q35.FP8_MAX).to(torch.float8_e4m3fn).float() * s).reshape(x.shape).to(x.dtype)


def fp8_checkpoint(tmp_path, **cfg):
    """A tiny FP8 block-128 checkpoint laid out as Qwen/Qwen3.8-27B-FP8: the projections in e4m3 with BF16 block
    scales; embeddings, lm_head, norms, conv1d, A_log, dt_bias, in_proj_a/b left as they were. Returns (the HF model,
    now holding the dequantized weights, and the checkpoint directory)."""
    from safetensors.torch import save_file
    hf = hf_causal_lm(tmp_path / "ref", **cfg)
    tensors = {}
    with torch.no_grad():
        for name, p in hf.state_dict().items():
            mod = name.rsplit(".", 2)[-2] if name.endswith(".weight") else ""
            if mod in QUANTIZED:
                q, s = fp8_block_quantize(p)
                tensors[name], tensors[name[:-len("weight")] + "weight_scale_inv"] = q, s
                p.copy_(fp8_dequantize(q, s))                      # HF now runs the dequantized weights
            else:
                tensors[name] = p.contiguous()
    out = tmp_path / "fp8"
    out.mkdir()
    save_file(tensors, str(out / "model.safetensors"))
    config = json.loads((tmp_path / "ref" / "config.json").read_text())
    config["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
                                     "weight_block_size": [128, 128],
                                     "modules_to_not_convert": ["lm_head", "model.embed_tokens", "in_proj_a",
                                                                "in_proj_b", "conv1d", "A_log", "dt_bias"]}
    (out / "config.json").write_text(json.dumps(config))
    (out / "generation_config.json").write_text((tmp_path / "ref" / "generation_config.json").read_text())
    return hf, out


# Mean |logit error| / logit spread for the as-shipped FP8 path, against HF with the declared activation scheme.
# Both sides compute the same scheme and differ only in fp32 summation order. That flips a few e4m3 roundings,
# which then spread through the layers and, through the long-memory DeltaNet heads, across tokens; how far depends
# on the BLAS, the thread count and the CPU. Observed with the skeleton core's qlinear (commit 91fc4a4) on
# CPU at 1, 4 and 16 threads: mean 1.0-1.3% per step. The max over the vocabulary (3.4-5.4%) is too chaotic to
# bound tightly, so it is reported, not asserted. act=a16 against the same reference: mean about 4.2%.
FP8_TOL = 2.5e-2
FP8_CFG = dict(hidden_size=256, intermediate_size=384, head_dim=64, linear_key_head_dim=64, linear_value_head_dim=64)


def test_fp8_block_checkpoint_matches_dequantized_hf(tmp_path):
    """act=a16 must match HF run on the dequantized weights. As shipped (dynamic per-token FP8 activations in groups
    of 128) must match the same HF model with that activation scheme applied at the input of every FP8 projection."""
    if not _fp8_qlinear_available():
        pytest.skip("qlinear has no fp8_block path yet (the skeleton core provides it)")
    hf, out = fp8_checkpoint(tmp_path, **FP8_CFG)
    ids = prompt()

    a16 = models.load_model(str(out), device="cpu", max_seq=96, act="a16")
    assert "fp8_block x" in a16.numerics["weights"] and a16.numerics["changed"]
    assert assert_matches_hf(a16, hf, ids, atol=1e-3, steps=4)

    shipped = models.load_model(str(out), device="cpu", max_seq=96)
    assert shipped.numerics["changed"] == [] and "dynamic per token" in shipped.numerics["activations"]["fp8_block"]
    hooks = [m.register_forward_pre_hook(lambda mod, args: (fp8_act_fake_quant(args[0]),) + args[1:])
             for name, m in hf.named_modules() if name.rsplit(".", 1)[-1] in QUANTIZED]
    try:
        toks, logits = run_greedy(shipped, ids, steps=3)
        seq = ids.tolist()
        for i, lg in enumerate(logits):
            ref = hf_last_logits(hf, seq)
            spread = float(ref.max() - ref.min())
            err = (lg - ref).abs()
            assert float(err.mean()) < FP8_TOL * spread, \
                f"step {i}: mean {float(err.mean()) / spread:.4f}, max {float(err.max()) / spread:.4f} of the spread"
            seq += toks[i:i + 1]
        # The scheme is applied: as shipped sits at least twice as close to this reference as a16 does.
        ref = hf_last_logits(hf, ids.tolist())
        plain = a16.prefill(ids, a16.alloc_state(96))
        assert float((logits[0] - ref).abs().mean()) < 0.5 * float((plain - ref).abs().mean())
    finally:
        for h in hooks:
            h.remove()


# --- validate_model against an SGLang reference (the Qwen3.8 GPU validation used SGLang 0.5.21) ------------------
def test_validate_model_sglang_reference_alignment_and_divergence_margins(monkeypatch):
    import math
    from ikengine.tools import validate_model as vm

    def top(d):
        return [[lp, t, None] for t, lp in sorted(d.items(), key=lambda kv: -kv[1])]

    lp = {5: math.log(0.6), 6: math.log(0.35), 7: math.log(0.05)}
    near = {6: math.log(0.5), 5: math.log(0.48), 7: math.log(0.02)}       # the reference's tokens 6 and 5 nearly tie
    lp_tf = {6: math.log(0.7), 5: math.log(0.25), 7: math.log(0.05)}      # step 0 teacher-forced: it prefers 6 there
    forced_on = []                                                           # the tokens each forced request scores

    def fake_post(url, body, key_env):
        assert url.endswith("/generate") and key_env is None
        ids, n0 = body["input_ids"], 3
        if body["logprob_start_len"] == -1:                                  # free run: greedy 5, 6
            return {"meta_info": {"output_token_logprobs": [[lp[5], 5, None], [near[6], 6, None]],
                                  "output_top_logprobs": [top(lp), top(near)]}}
        s = body["logprob_start_len"]                                        # SGLang 0.5.21: from position s on,
        assert s == n0 - 1                                                   # the first entry empty
        forced_on.append(ids[n0:])
        forced = [lp_tf, near][: len(ids) - s - 1]
        return {"meta_info": {"input_token_logprobs": [[None, ids[s], None]] + [[f[t], t, None] for f, t in
                                                                                zip(forced, ids[s + 1:])],
                              "input_top_logprobs": [None] + [top(f) for f in forced]}}

    monkeypatch.setattr(vm, "_post", fake_post)
    args = SimpleNamespace(ref="http://ref:1", ref_kind="sglang", ref_model=None, api_key_env=None, tokens=2)
    mine = {5: math.log(0.5), 6: math.log(0.49), 7: math.log(0.01)}
    eng = {"ids": [5, 5], "tops": [lp, mine]}                                 # the engine picks 5 at step 1
    ref = vm.ref_openai(args, [[1, 2, 3]], [eng])[0]
    assert ref["teacher_forced"] and ref["ids"] == [5, 6] and ref["tops"][1] == pytest.approx(near)
    c = vm.compare(eng["ids"], eng["tops"], ref["ids"], ref["tops"], ref["free_tops"], tie_nats=0.1)
    d = c["divergence"]
    assert c["first_divergence"] == 1 and d["engine_token"] == 5 and d["ref_token"] == 6
    assert d["ref_tf_gap"] == pytest.approx(math.log(0.5 / 0.48), abs=1e-4) and d["near_tie"]
    assert d["eng_gap"] == pytest.approx(math.log(0.5 / 0.49), abs=1e-4)
    assert vm.compare([5, 5], [lp, lp], [5, 5], [lp, lp])["divergence"] is None
    args.ref_aa, args.tie_nats = True, 0.5                                   # A/A: SGLang forced on its own path
    forced_on.clear()
    aa = vm.ref_openai(args, [[1, 2, 3]], [eng])[0]["aa"]
    assert forced_on == [[5, 5], [5, 6]]                                     # the engine's tokens, then its own
    # its decode top-20 in the engine's place, its teacher-forced top-20 as the reference: KL(forced || free), and
    # top-1 is the forced argmax against its own token (6 at step 0, not its 5)
    kl0 = vm.bucket_kl(lp_tf, lp)
    assert abs(kl0 - vm.bucket_kl(lp, lp_tf)) > 0.01                         # the two directions differ here
    assert aa["steps"] == 2 and aa["mean_kl"] == pytest.approx(kl0 / 2) and aa["top1_agreement"] == 0.5
    assert aa["top1_gaps"] == [pytest.approx(math.log(0.7 / 0.25))] and aa["run_to_run_first_divergence"] is None

    def shifted(url, body, key_env):                                         # entries one position off: refused
        r = fake_post(url, body, key_env)
        if body["logprob_start_len"] != -1:
            r["meta_info"]["input_token_logprobs"] = r["meta_info"]["input_token_logprobs"][1:]
        return r
    monkeypatch.setattr(vm, "_post", shifted)
    with pytest.raises(ValueError):
        vm._sglang_forced(args, "http://ref:1", [1, 2, 3], [5, 5])


def test_validate_model_reads_bucketed_chat_jsonl_and_reuses_engine_runs(tmp_path, monkeypatch):
    from ikengine.tools import validate_model as vm
    recs = [{"bucket": b, "id": f"{b}-{i}", "messages": [{"role": "user", "content": f"{b} {i}"}]}
            for b in ("very_short", "short", "long") for i in range(3)]
    p = tmp_path / "dev.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    got, labels = vm.read_prompt_file(str(p), per_bucket=2, buckets=["short", "very_short"])
    assert [m[0]["content"] for m in got] == ["short 0", "short 1", "very_short 0", "very_short 1"]
    assert labels == ["short-0", "short-1", "very_short-0", "very_short-1"]
    assert len(vm.read_prompt_file(str(p), per_bucket=1)[0]) == 3

    class Tok:
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt, **kw):
            assert not tokenize and add_generation_prompt and kw == {"enable_thinking": False}
            return "|".join(m["content"] for m in msgs)

        def encode(self, text, add_special_tokens=True):
            assert not add_special_tokens
            return [len(text)]
    assert vm.tokenize(Tok(), got[:1], {"enable_thinking": False}) == [[len("short 0")]]

    saved = tmp_path / "engine.json"
    runs = [{"ids": [4, 5], "tops": [{4: -0.1}, {5: -0.2}]}]
    monkeypatch.setattr(vm, "run_engine", lambda a, pr: (runs, 7.5, {"engine": "fake"}))
    a = SimpleNamespace(load_engine=None, save_engine=str(saved), tokens=2, model="m")
    assert vm._engine_runs(a, [[1, 2]]) == (runs, 7.5, {"engine": "fake"})
    monkeypatch.setattr(vm, "run_engine", lambda a, pr: pytest.fail("the engine must not be rebuilt"))
    a = SimpleNamespace(load_engine=str(saved), save_engine=None, tokens=2, model="m")
    assert vm._engine_runs(a, [[1, 2]]) == (runs, 7.5, {"engine": "fake"})
    with pytest.raises(ValueError):
        vm._engine_runs(a, [[1, 3]])                                          # other prompts: refused
