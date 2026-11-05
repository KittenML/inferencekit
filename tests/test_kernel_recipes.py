"""Kernel-side skill text: the byte-cut and MoE references, their links, the model-agnostic rule, and the
reference code in byte-cuts.md, executed as written (needs torch; CI has none and skips those tests)."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SKILLS = ROOT / "plugin" / "skills"
KERN = SKILLS / "optimizing-kernels"
BYTE_CUTS = KERN / "references" / "byte-cuts.md"
MOE = KERN / "references" / "moe-decode.md"
LAUNCH = KERN / "references" / "launch-overhead.md"
PROVING = SKILLS / "proving-correctness" / "SKILL.md"
PROFILING = SKILLS / "profiling" / "SKILL.md"

# Recipes written before a pre-registered comparison must stay generic: no names or constants of the models
# it runs on, no comparison wording, no private run paths, no internal job words. Built from parts so this
# file never matches itself.
FORBIDDEN = re.compile("(?i)" + "|".join([
    "gem" + "ma", r"qwen3[._]?[58]", r"262,?144", r"248,?320", r"2,?816", r"\ba4" + r"b\b", r"12\.25", r"4\.77",
    r"1\.62x", "no[- ]?" + "plugin", "without (the )?" + "plugin", "control[- ]" + "arm", "plugin " + "vs",
    "inferencekit-" + "runs", r"\bengines/", r"\b" + "he" + r"ro\b", r"\b" + "ar" + r"ms?\b", r"\b" + "la" + r"nes?\b"]))
CHECKED = [BYTE_CUTS, MOE, LAUNCH, KERN / "SKILL.md", PROVING, PROFILING,
           SKILLS / "proving-correctness" / "references" / "gsm8k-parity.md"]


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def test_recipes_exist_and_are_linked():
    skill = (KERN / "SKILL.md").read_text()
    assert "(references/byte-cuts.md)" in skill and "(references/moe-decode.md)" in skill
    assert "(skeleton/README.md)" in skill
    assert "(../optimizing-kernels/references/byte-cuts.md)" in PROVING.read_text()
    assert "(byte-cuts.md)" in (KERN / "references" / "fp8-gemv.md").read_text()
    assert "### PDL placement" in LAUNCH.read_text()
    assert "(launch-overhead.md#pdl-placement)" in (KERN / "references" / "lessons.md").read_text()


@pytest.mark.parametrize("path", CHECKED, ids=_rel)
def test_kernel_text_is_model_agnostic(path: Path):
    hits = [f"{ln}: {m.group(0)!r}" for ln, line in enumerate(path.read_text().splitlines(), 1)
            for m in FORBIDDEN.finditer(line)]
    assert not hits, f"{_rel(path)}: {hits}"


def test_byte_cut_rules_are_stated():
    text = BYTE_CUTS.read_text()
    for rule in ["raw 16-bit value", "Never store an exception as an fp32 correction", "all 65,536 BF16 bit patterns",
                 "logprobs are requested", "monotone non-decreasing", "ties to the lowest token id",
                 "inferencekit quality sampling", "Recompute the ceiling",
                 # -inf rows sent vLLM's seeded top-k/top-p kernel down its full-vocabulary path
                 "Fill pruned rows with their f(U) or f(L), never -inf", "first 8,192-token tile",
                 "below the k-th largest kept logit", "logprob requests still take the full head"]:
        assert rule in text, rule
    gemv = (KERN / "references" / "fp8-gemv.md").read_text()
    for rule in ["E8M0", "requantised weights and power-of-two scales vLLM hands DeepGEMM", "bit for bit",
                 "`torch.equal` on real layers before timing", "a mismatch makes it a numerics change"]:
        assert rule in gemv, rule
    moe = MOE.read_text()
    for rule in ["top-k order", "Stay inside the CUDA graph", "PDL between the two MoE kernels",
                 "Recomputing top-k in the second kernel", "Speculation reads more experts"]:
        assert rule in moe, rule


def test_fidelity_column_and_ncu_sentence():
    proving = PROVING.read_text()
    header = "| Kind of change | Examples | Fidelity | Pass rule |"
    assert header in proving
    rows = proving.split(header, 1)[1].split("\n\n", 1)[0].strip().splitlines()[1:]
    assert len(rows) >= 7 and all(r.split("|")[3].strip().startswith(("lossless", "lossy")) for r in rows), rows
    assert '"ncu was not possible on this host; profiling used nsys timelines only"' in PROFILING.read_text()


def test_sampled_exactness_names_what_first_token_checks_miss():
    """`quality sampling` sends max_tokens 1, so the first token comes from prefill: speculation and a decode-only
    pruned head need their own checks, and the speculation row says so."""
    proving = PROVING.read_text()
    section = proving.split("## Sampled exactness", 1)[1].split("\n## ", 1)[0]
    for rule in ["checks only the first generated token", "accept/resample step", "pre-registered bound",
                 "first token must also go through the pruned head", "the sampled claim is UNPROVEN"]:
        assert rule in section, rule
    spec_row = next(r for r in proving.splitlines() if r.startswith("| Exact speculation |"))
    assert "accept/resample check" in spec_row and "`spec_exact`" in spec_row
    assert "counts only when the first token also goes through the pruned head" in BYTE_CUTS.read_text()


def test_moe_quantises_the_input_before_gate_up():
    """A checkpoint that quantises activations quantises the MoE input before the gate/up GEMV (not only the
    intermediate); otherwise gate/up silently run with 16-bit activations."""
    step1 = MOE.read_text().split("1. **Routing + gate/up.**", 1)[1].split("\n2. ", 1)[0]
    i_in = step1.index("quantise the MoE input")
    assert i_in < step1.index("before the gate/up GEMV") < step1.index("The epilogue computes")
    assert "re-quantises the intermediate" in step1[step1.index("The epilogue computes"):]


def test_pdl_consumer_reads_activations_coherently():
    text = LAUNCH.read_text().split("### PDL placement", 1)[1]
    for rule in ["Read the producer's output only after the wait", "`__restrict__`/`__ldg`", "keep it on weights",
                 "cuobjdump -sass"]:
        assert rule in text, rule
    assert "(launch-overhead.md#pdl-placement)" in (KERN / "references" / "fp8-gemv.md").read_text()


def test_checks_are_sized_to_what_the_result_is_used_for():
    """The compact loop still separates exploration, promotion and final proof.
    Detailed text-job contents have one owner in state.md; media uses its own checks."""
    skill = (KERN / "SKILL.md").read_text()
    checks = skill.split("## Checks and implementation loop", 1)[1].split("\n## ", 1)[0]
    assert "one fast screen" in checks and "fixed reference" in checks
    assert "registered checks" in checks and "for text" in checks
    assert "ikstep job light" in checks and "ikstep job full" in checks
    assert "full proof once for the final candidate" in checks
    assert "claimed change needs interleaved A/B confidence intervals" in checks
    assert "profile showing the mechanism" in checks
    assert "Exploration screens never become published proof" in checks
    assert "Do not run full gates" in checks and "after every edit" in checks
    assert "state.md" in checks and ">= 1.2x" not in checks  # one owner for triggers
    state = (KERN.parent / "optimizing-inference" / "references" / "state.md").read_text()
    for owner in ("`bench rounds --rounds 2` with `--limit-per-bucket 8`", "`gates run --gates caching --gates token_count --gates rare_tokens`",
                  "`--limit 8`", "GSM8K `--limit 100 --concurrency 0`", "`quality sampling` only if sampling code changed",
                  "`bench rounds --rounds 3` on all 64 dev prompts", "all 1,319 of GSM8K `--paired-with` the baseline's"):
        assert owner in state, owner


# --- the reference code in byte-cuts.md, run as written --------------------------------------------------

def _byte_cut_code() -> dict:
    ns: dict = {}
    for block in re.findall(r"```python\n(.*?)```", BYTE_CUTS.read_text(), re.S):
        exec(compile(block, str(BYTE_CUTS), "exec"), ns)
    for name in ("pack_bf16", "unpack_bf16", "head_proxy", "head_bounds", "candidates", "pruned_argmax"):
        assert callable(ns.get(name)), name
    return ns


def _all_bf16(torch):
    b = torch.arange(65536, dtype=torch.int32)
    return (b - ((b >> 15) << 16)).to(torch.int16)                    # every 16-bit pattern as int16


def test_packer_round_trips_every_bf16_pattern():
    torch = pytest.importorskip("torch")
    ns = _byte_cut_code()
    pack, unpack = ns["pack_bf16"], ns["unpack_bf16"]
    pats = _all_bf16(torch)
    g = torch.Generator().manual_seed(0)
    cases = [pats.view(torch.bfloat16).view(256, 256),                                  # sorted: similar groups
             pats[torch.randperm(65536, generator=g)].view(torch.bfloat16).view(64, 1024),  # mixed groups
             torch.zeros(2, 64, dtype=torch.bfloat16).neg(),                             # -0 only: all escapes
             torch.full((1, 128), float("nan"), dtype=torch.bfloat16)]
    for w in cases:
        for G, d in [(64, 4), (32, 4), (128, 3), (64, 5)]:
            if w.shape[1] % G:
                continue
            out = unpack(*pack(w, G, d), d=d)
            assert torch.equal(out.view(torch.int16), w.view(torch.int16)), (tuple(w.shape), G, d)


def test_packer_escape_rate_and_size_on_gaussian_rows():
    torch = pytest.importorskip("torch")
    ns = _byte_cut_code()
    g = torch.Generator().manual_seed(1)
    w = (torch.randn(512, 4096, generator=g) * 0.02).to(torch.bfloat16)
    base, sm, delta, exc = ns["pack_bf16"](w, 64, 4)
    assert torch.equal(ns["unpack_bf16"](base, sm, delta, exc, 4).view(torch.int16), w.view(torch.int16))
    rate = exc.numel() / w.numel()
    assert rate < 1e-3                                       # the doc: about 1 in 10,000 on Gaussian rows
    bits = 8 + 4 + 8 / 64 + 16 * rate
    assert bits < 12.2                                       # 12.125 bits plus exceptions, against 16
    rate3 = ns["pack_bf16"](w, 64, 3)[3].numel() / w.numel()
    assert 0.005 < rate3 < 0.05                              # the doc: d = 3 escapes about 2 in 100


def test_pruned_head_matches_the_full_head():
    torch = pytest.importorskip("torch")
    ns = _byte_cut_code()
    g = torch.Generator().manual_seed(2)
    V, H, G = 4096, 256, 64
    W = (torch.randn(V, H, generator=g) * 0.03).to(torch.bfloat16)
    W[100] = W[7]
    W[3000] = W[7]                                           # exact ties with row 7
    q, s, e, a = ns["head_proxy"](W, G)
    cap = 30.0
    transforms = [lambda z: z,
                  lambda z: cap * torch.tanh(z / cap),                                  # soft cap
                  lambda z: (cap * torch.tanh(z / cap)).to(torch.bfloat16).float()]     # plus output rounding
    for step in range(40):
        x = torch.randn(H, generator=g) * (1 + step % 5)
        if step % 4 == 0:
            x = W[7].float() * (5 + 200 * (step % 8 == 0))    # row 7 and its copies win; large: saturation
        z = W.float() @ x                                    # the full head's fp32 logits
        L, U = ns["head_bounds"](x, q, s, e, a, G)
        assert bool(((z.double() >= L) & (z.double() <= U)).all())
        for f in transforms:
            fz = f(z)
            full = int((fz == fz.max()).nonzero()[0, 0])     # argmax, ties to the lowest id
            rows = ns["candidates"](L, U, f)
            assert ns["pruned_argmax"](rows, z[rows], f) == full
            for k in (20, 64):
                kept = set(ns["candidates"](L, U, f, k).tolist())
                kth = fz.topk(k).values[-1]
                assert set((fz >= kth).nonzero().squeeze(1).tolist()) <= kept    # top k and its ties


def test_sampling_gaps_keep_every_row_top_p_or_min_p_can_draw():
    """The doc's top-p and min-p rules on the bounds: every row the full sampler could draw is kept."""
    torch = pytest.importorskip("torch")
    ns = _byte_cut_code()
    g = torch.Generator().manual_seed(3)
    V, H, G = 4096, 256, 64
    W = (torch.randn(V, H, generator=g) * 0.05).to(torch.bfloat16)
    q, s, e, a = ns["head_proxy"](W, G)
    pruned = 0
    for step in range(24):
        x = torch.randn(H, generator=g) * (0.5 + 2 * (step % 6))
        z = (W.float() @ x).double()
        L, U = ns["head_bounds"](x, q, s, e, a, G)
        for T, top_p, min_p in [(1.0, 0.95, 0.05), (0.6, 0.9, 0.1), (1.5, 0.5, 0.01)]:
            p = torch.softmax(z / T, 0)
            order = p.argsort(descending=True)
            n = int((p[order].cumsum(0) < top_p).sum()) + 1                 # smallest prefix with mass >= top_p
            nucleus = p >= p[order[n - 1]]                                  # with any ties at the boundary
            kept = U >= L.max() - T * math.log(V / (1 - top_p))
            assert bool(kept[nucleus].all()), (step, T, top_p)
            kept_min_p = U >= L.max() - T * math.log(1 / min_p)
            assert bool(kept_min_p[p >= min_p * p.max()].all()), (step, T, min_p)
            pruned += int((~kept).sum())
    assert pruned > 0                                                       # the rule prunes something here


def _vllm_top_k_top_p(logits, k, p):
    """vLLM's (and Hugging Face's) order on one row: keep values >= the k-th largest (ties kept), then the smallest
    set of the survivors holding at least p of their mass; the probabilities the sampler draws from."""
    import torch
    x = logits.masked_fill(logits < logits.topk(k).values[-1], float("-inf"))
    srt, idx = x.sort()
    drop = srt.softmax(0).cumsum(0) <= 1 - p
    drop[-1] = False
    return x.scatter(0, idx, srt.masked_fill(drop, float("-inf"))).softmax(0)


def test_finite_fill_of_pruned_rows_keeps_sampling_and_top20_exact():
    """Pruned rows filled with their bound f(U) or f(L), not -inf, still give the sampler bit-identical probabilities
    and the same top 20 ids as the full head. Speed, measured on an RTX 5090 with vLLM 0.31.0 (321 real
    decode rows, k 20, p 0.95, 131 rows kept): _topk_topp_kernel 91.0 us per call for full rows and for both
    finite fills, 295.1 us with -inf rows, the same kept set in every row."""
    torch = pytest.importorskip("torch")
    ns = _byte_cut_code()
    g = torch.Generator().manual_seed(4)
    V, H, G = 4096, 256, 64
    W = (torch.randn(V, H, generator=g) * 0.05).to(torch.bfloat16)
    q, s, e, a = ns["head_proxy"](W, G)
    for step in range(24):
        x = torch.randn(H, generator=g) * (0.5 + 2 * (step % 6))
        z = W.float() @ x
        L, U = ns["head_bounds"](x, q, s, e, a, G)
        lo, hi = L.float(), U.float()
        for T, k, p in [(0.6, 20, 0.95), (1.0, 50, 0.9), (1.5, 20, 1.0)]:
            kept = ns["candidates"](L, U, k=k)
            assert kept.numel() < V                                       # it prunes
            full = _vllm_top_k_top_p(z / T, k, p)
            for fill in (hi, lo):
                pruned = fill.clone()
                pruned[kept] = z[kept]                                    # rescored rows: the full head's values
                assert bool((fill[~torch.isin(torch.arange(V), kept)] < z[kept].topk(k).values[-1]).all())
                assert torch.equal(_vllm_top_k_top_p(pruned / T, k, p), full), (step, T, k, p)
                assert torch.equal(pruned.topk(20).indices, z.topk(20).indices)
