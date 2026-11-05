#!/usr/bin/env python3
"""Bytes read per decoded token at batch 1, from a checkpoint's safetensors headers (claim class: derived).

    python3 bytes_per_token.py <model-dir> <context> [--kv-bytes 2] [--topk K] [--n-experts N]
        [--expert-pattern REGEX] [--layer-types full,sliding,...] [--kv-bytes-total N]
        [--pack-bits 12] [--head-proxy-bits 8.5]

Stdlib only; reads the JSON headers, never the tensor data. Copy it next to the checkpoint (the GPU host)
and run it there. It is a skill helper, not part of the harness.

It refuses to guess (exit 2, the reason on stderr) when guessing would OVERSTATE bytes per token: a bytes
model that is too high lowers the ceiling and makes "% of ceiling" look better than it is.
  * The config says MoE but no tensor matches the routed-expert pattern (all expert bytes would be counted
    as read every token). Pass --expert-pattern; sample tensor names per group are printed.
  * The attention layout is not plain full attention (sliding-window, chunked, linear-attention/SSM layers,
    per-layer-type KV shapes, MLA, shared KV) and config.json has no usable layer_types. Pass --layer-types,
    or --kv-bytes-total with a hand-derived figure (record the formula).

It also estimates the exact byte floor: what an engine that changes no output bit could still read per token.
BF16 weights read per token (not the output layer) are re-encoded losslessly at --pack-bits per element; the
output layer is read through a bound-pruned proxy at --head-proxy-bits per element (full rows only for the
candidates that can still win); everything else stays as stored. Both bit widths are assumptions (inferred):
measure what the built engine reads.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import struct
import sys

PACKABLE = ("BF16",)  # F16 has 5 exponent bits: little left to re-encode losslessly
SIXTEEN = ("BF16", "F16")
SKIP = re.compile(r"vision|visual|audio|mm_projector|multi_modal|\bmtp\b", re.I)
EMBED = re.compile(r"embed_tokens|\bwte\b|tok_embeddings|word_embeddings")
EXPERT_COUNT_KEYS = ("num_experts", "n_routed_experts", "num_local_experts")
TOPK_KEYS = ("num_experts_per_tok", "top_k_experts", "moe_top_k", "num_experts_per_token")
# config keys whose presence means the KV term is not "2 x layers x kv_heads x head_dim" over the full context
LAYOUT_KEYS = re.compile(r"^(sliding_window_pattern|full_attention_interval|attention_chunk_size|"
                         r"max_window_layers|layer_types|no_rope_layers)$|^linear_|ssm|mamba|conv_kernel")
SHAPE_KEYS = re.compile(r"(global|full|sliding|local|swa).*(head_dim|key_value_heads)|"
                        r"(head_dim|key_value_heads).*(global|full|sliding|local|swa)|kv_lora_rank|"
                        r"kv_shared|shared_kv")


def fail(msg: str, samples: dict | None = None) -> None:
    print(f"bytes_per_token: {msg}", file=sys.stderr)
    if samples:
        print(json.dumps({"sample_tensors_per_group": samples}, indent=1), file=sys.stderr)
    sys.exit(2)


def first(cfg: dict, keys) -> tuple:
    for k in keys:
        if cfg.get(k) is not None:
            return cfg[k], k
    return None, None


def read_groups(model_dir: str, expert_re: re.Pattern) -> tuple[dict, dict, dict]:
    """Stored bytes per group, sample names per group, and stored bytes per group and dtype."""
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        fail(f"no *.safetensors files in {model_dir}")
    groups = {k: 0 for k in ("every_token", "routed_experts", "embedding", "lm_head", "skipped")}
    samples: dict = {k: [] for k in groups}
    dtypes: dict = {k: {} for k in groups}
    for f in files:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for name, t in header.items():
            if name == "__metadata__":
                continue
            b = t["data_offsets"][1] - t["data_offsets"][0]
            if SKIP.search(name):
                k = "skipped"
            elif EMBED.search(name):
                k = "embedding"
            elif "lm_head" in name:
                k = "lm_head"
            elif expert_re.search(name) and "shared_expert" not in name:
                k = "routed_experts"
            else:
                k = "every_token"
            groups[k] += b
            dtypes[k][t["dtype"]] = dtypes[k].get(t["dtype"], 0) + b
            if len(samples[k]) < 3:
                samples[k].append(name)
    return groups, samples, dtypes


def layer_kind(t: str) -> str:
    s = str(t).lower()
    if any(x in s for x in ("linear", "mamba", "ssm", "conv", "recurrent", "delta")):
        return "state"
    if any(x in s for x in ("sliding", "local", "chunk", "swa", "window")):
        return "window"
    if any(x in s for x in ("full", "global", "attention", "attn")):
        return "full"
    fail(f"unknown layer type {t!r}: pass --layer-types with full/sliding/linear entries")
    return ""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("context", type=int, help="tokens in the KV cache: prompt length + about half of max_tokens")
    ap.add_argument("--kv-bytes", type=float, default=2.0, help="bytes per KV element the engine RESOLVED "
                    "(2 = bf16/fp16, 1 = fp8)")
    ap.add_argument("--topk", type=int, help="experts per token (default: from config.json)")
    ap.add_argument("--n-experts", type=int, help="routed experts (default: from config.json)")
    ap.add_argument("--expert-pattern", default=r"\.experts\.", help="regex for routed-expert tensor names "
                    r"(default '\.experts\.'; shared experts are never routed)")
    ap.add_argument("--layer-types", help="comma list of layer kinds (full | sliding | linear), tiled to "
                    "num_hidden_layers; e.g. sliding,sliding,sliding,sliding,sliding,full. 'full' alone states "
                    "that every layer is full attention.")
    ap.add_argument("--kv-bytes-total", type=float, help="KV bytes read per token at this context, derived by "
                    "hand (overrides the KV term; record the formula)")
    ap.add_argument("--pack-bits", type=float, default=12, help="bits per BF16 element after a lossless "
                    "re-encoding (default 12: 8 raw sign and mantissa bits plus a 4-bit exponent code; group "
                    "overhead and raw exceptions add a little; inferred)")
    ap.add_argument("--head-proxy-bits", type=float, default=8.5, help="bits per element of the 16-bit output "
                    "layer that a bound-pruned head reads (default 8.5: an 8-bit proxy of every row plus scales; "
                    "inferred)")
    a = ap.parse_args(argv)
    for flag, bits in (("--pack-bits", a.pack_bits), ("--head-proxy-bits", a.head_proxy_bits)):
        if not 0 < bits <= 16:
            fail(f"{flag} {bits:g} is outside (0, 16]")

    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
    tc = cfg.get("text_config") or cfg
    notes = []

    groups, samples, dtypes = read_groups(a.model_dir, re.compile(a.expert_pattern))
    if groups["embedding"] == 0:
        fail(f"no embedding tensor matched {EMBED.pattern!r}; check the tensor names below", samples)

    # --- MoE
    n_cfg, n_key = first(tc, EXPERT_COUNT_KEYS)
    k_cfg, k_key = first(tc, TOPK_KEYS)
    nexp, topk = a.n_experts or n_cfg, a.topk or k_cfg
    is_moe = bool(nexp) and int(nexp) > 1
    if is_moe and groups["routed_experts"] == 0:
        fail(f"MoE config ({n_key or '--n-experts'}={nexp}) but no routed-expert tensors matched "
             f"{a.expert_pattern!r}; pass the expert name pattern with --expert-pattern REGEX "
             "(sample tensor names per group below)", samples)
    if groups["routed_experts"] and not is_moe:
        fail(f"tensors match {a.expert_pattern!r} but config.json gives no expert count; pass --n-experts and "
             "--topk from the model card", samples)
    if groups["routed_experts"] and not topk:
        fail(f"MoE with {nexp} experts but no top-k in config.json ({TOPK_KEYS}); pass --topk")
    frac = (int(topk) / int(nexp)) if groups["routed_experts"] else None
    if frac is not None:
        notes.append("MoE at batch 1: expert bytes x topk/n_experts assumes no expert is re-read within a "
                     "step (inferred modelling assumption)")

    # --- embedding and head
    tie = cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings"))
    if groups["lm_head"] == 0 and tie is False:
        fail("tie_word_embeddings is false but no lm_head tensor was found", samples)
    tied = groups["lm_head"] == 0
    vocab = tc["vocab_size"]
    head = groups["embedding"] if tied else groups["lm_head"]
    emb_row = groups["embedding"] / vocab

    # --- attention layout
    layers = tc["num_hidden_layers"]
    sliding_on = tc.get("use_sliding_window", True) is not False
    if a.layer_types:
        pattern = [t.strip() for t in a.layer_types.split(",") if t.strip()]
        if not pattern or layers % len(pattern):
            fail(f"--layer-types has {len(pattern)} entries, which does not tile {layers} layers")
        kinds, source = [layer_kind(t) for t in pattern] * (layers // len(pattern)), "--layer-types"
    elif isinstance(tc.get("layer_types"), list):
        if len(tc["layer_types"]) != layers:
            fail(f"config layer_types has {len(tc['layer_types'])} entries for {layers} layers")
        kinds, source = [layer_kind(t) for t in tc["layer_types"]], "config.json layer_types"
    else:
        hints = [k for k in tc if LAYOUT_KEYS.search(k) and tc[k] not in (None, False, 0, [], {})
                 and (sliding_on or k != "max_window_layers")]
        if tc.get("sliding_window") and sliding_on:
            hints.append("sliding_window")
        if hints and a.kv_bytes_total is None:
            fail(f"config.json has non-full-attention keys {sorted(hints)} but no layer_types. Pass --layer-types "
                 f"(a comma list tiled to {layers} layers, e.g. sliding,sliding,sliding,sliding,sliding,full), or "
                 "--layer-types full if every layer really is full attention (say why), or --kv-bytes-total. "
                 "Never assume full attention silently: it overstates KV bytes.")
        kinds, source = ["full"] * layers, "default: config.json has no sliding-window/linear-attention keys"
    shape_keys = sorted(k for k in tc if SHAPE_KEYS.search(k) and tc[k] not in (None, False, 0))
    if shape_keys and a.kv_bytes_total is None:
        fail(f"config.json has per-layer-type or compressed KV shapes {shape_keys}: derive the KV bytes by hand and "
             "pass --kv-bytes-total (record the formula)")
    window = tc.get("sliding_window") or tc.get("attention_chunk_size")
    if "window" in kinds and not window:
        fail("sliding/chunked layers but no sliding_window or attention_chunk_size in config.json")
    counts = {k: kinds.count(k) for k in ("full", "window", "state")}
    kvh = tc.get("num_key_value_heads") or tc["num_attention_heads"]
    hd = tc.get("head_dim") or tc["hidden_size"] // tc["num_attention_heads"]
    per_token_per_layer = 2 * kvh * hd * a.kv_bytes
    window_tokens = min(a.context, int(window)) if window else None
    if a.kv_bytes_total is not None:
        kv, kv_source = a.kv_bytes_total, "--kv-bytes-total (hand-derived; record the formula)"
    else:
        kv = per_token_per_layer * (counts["full"] * a.context + counts["window"] * (window_tokens or 0))
        kv_source = "2 x kv_heads x head_dim x kv_bytes x (full layers x context + window layers x min(context, window))"
    if counts["state"]:
        notes.append(f"{counts['state']} linear-attention/SSM layers: no per-token KV; their fixed-size state is "
                     "read every step and is NOT counted (understates bytes: the conservative direction)")
    if counts["window"] and "chunk" in str(tc.get("layer_types", "")).lower():
        notes.append("chunked-attention layers counted as min(context, chunk): an upper bound")

    weights = groups["every_token"] + head + emb_row + (groups["routed_experts"] * frac if frac is not None else 0)
    total = weights + kv

    # --- exact byte floor: weight bytes read per token by dtype, then the two lossless cuts
    reads = {"every_token": 1.0, "routed_experts": frac or 0.0, "lm_head": 1.0, "embedding": 1 / vocab + tied,
             "skipped": 0.0}  # the share of each group's stored bytes read per token
    by_dtype: dict = {}
    for g, per in dtypes.items():
        for dt, b in per.items():
            if reads[g]:
                by_dtype[dt] = by_dtype.get(dt, 0) + b * reads[g]
    packable = sum(b * reads[g] for g in ("every_token", "routed_experts") for dt, b in dtypes[g].items()
                   if dt in PACKABLE)
    head_dtypes = dtypes["embedding" if tied else "lm_head"]
    head16 = sum(b for dt, b in head_dtypes.items() if dt in SIXTEEN)
    floor = total - packable * (1 - a.pack_bits / 16) - head16 * (1 - a.head_proxy_bits / 16)
    f16 = sum(b * reads[g] for g in ("every_token", "routed_experts") for dt, b in dtypes[g].items() if dt == "F16")
    if f16:
        notes.append(f"{f16:.0f} F16 weight bytes per token are not counted as packable: 5 exponent bits leave "
                     "little to re-encode losslessly (the floor is higher, the conservative direction)")
    if head16 < head:
        notes.append(f"output layer stored as {sorted(head_dtypes)}: the bound-pruned proxy is counted only for "
                     "its 16-bit part")

    print(json.dumps({
        "claim_class": "derived",
        "context": a.context,
        "bytes_per_token": weights + kv,
        "weights_per_token": weights,
        "kv_bytes": kv,
        "groups_bytes": groups,
        "checkpoint_bytes": sum(groups.values()),
        "sample_tensors_per_group": samples,
        "tied_lm_head": tied,
        "moe": {"n_experts": nexp, "topk": topk, "expert_fraction": frac,
                "source": {"n_experts": "--n-experts" if a.n_experts else n_key,
                           "topk": "--topk" if a.topk else k_key}} if (is_moe or frac is not None) else None,
        "attention": {"layers": counts, "window_tokens": window_tokens, "source": source,
                      "kv_heads": kvh, "head_dim": hd, "kv_bytes_per_elem": a.kv_bytes, "kv_source": kv_source},
        "notes": notes,
        "formula": "every_token + lm_head (or tied embedding) + one embedding row + routed_experts x topk/n_experts"
                   " + KV term (attention.kv_source)",
        "bytes_by_dtype": by_dtype,
        "packable_16bit_bytes": packable,
        "lm_head_bytes": head,
        "lm_head_share": head / total if total else None,
        "floor_bytes_per_token": floor,
        "floor_formula": f"bytes_per_token - packable_16bit_bytes x (1 - {a.pack_bits:g}/16) - 16-bit output layer "
                         f"bytes ({head16:.0f}) x (1 - {a.head_proxy_bits:g}/16). bytes_by_dtype and "
                         "packable_16bit_bytes are weight bytes read per token (BF16, outside the output layer); "
                         "lm_head_share = lm_head_bytes / bytes_per_token. Derived from the headers; both bit widths "
                         "are inferred (--pack-bits, --head-proxy-bits)",
    }, indent=1))


if __name__ == "__main__":
    main()
