"""Bytes read per batch-1 decode token, from the real tensors of a checkpoint.

``roofline --method headers`` (the default whenever safetensors are available). The weight
term is the sum of stored tensor bytes (dtype x shape from the safetensors headers, so packed
FP4 values, FP8 values and every scale tensor count exactly as stored), grouped by the role of
the module each tensor belongs to:

  attention         full / sliding attention projections and their scales
  linear_attention  Gated DeltaNet (or other linear-attention / SSM) projections, conv, A_log...
  mlp_dense         dense MLPs (including the always-on MLP next to an MoE block)
  moe_experts       routed experts, counted x top_k / num_experts (batch 1 reads top_k of them)
  moe_shared        shared experts (always on)
  moe_router        router / gate weights
  norms             norm weights and per-layer scalars
  lm_head           the logits matrix, read in full every token (a tied one is the embedding,
                    counted once here)
  embedding_row     one row of each token-embedding table (the current token's)
  other             any text-decoder tensor not matched above (counted; listed by name pattern)

Excluded from decode (reported, not counted): vision / audio towers and projectors, the MTP /
next-n head (unless include_mtp), and the unread rows of the embedding table.

The per-token state term adds:
  * full-attention KV: context x 2 (K, V) x kv width x kv bytes per layer;
  * sliding-window layers: min(context, window) tokens instead of context;
  * linear-attention (Gated DeltaNet) recurrent state, read and written every token:
    num_value_heads x key_head_dim x value_head_dim per layer, plus the conv state
    (conv channels x (kernel - 1)), also read and written. Architectures whose state layout is
    not known here are marked UNPROVEN rather than guessed.
"""

from __future__ import annotations

import re
from collections import defaultdict

from inferencekit.checkpoint import Checkpoint, Tensor
from inferencekit.ops.roofline import text_config
from inferencekit.registry import OpError

GROUPS = ("attention", "linear_attention", "mlp_dense", "moe_experts", "moe_shared", "moe_router", "norms",
          "lm_head", "embedding_row", "other")
KV_BYTES = {"bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "q8_0": 34 / 32}
STATE_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2}
CONFIG_DTYPE = {"float32": "fp32", "float": "fp32", "bfloat16": "bf16", "float16": "fp16", "half": "fp16"}

N_EXPERTS_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts", "num_routed_experts")
TOP_K_KEYS = ("num_experts_per_tok", "top_k_experts", "moe_top_k", "num_experts_per_token")
MTP_KEYS = ("num_nextn_predict_layers", "mtp_num_hidden_layers", "num_mtp_layers")

_TOWER = {"visual", "vision_tower", "vision_model", "vision_encoder", "embed_vision", "vision_embed",
          "multi_modal_projector", "mm_projector", "audio_tower", "audio_model", "audio_encoder", "embed_audio",
          "image_newline", "speech_encoder"}
_EMBED = {"embed_tokens", "wte", "word_embeddings", "tok_embeddings", "embed_tokens_per_layer"}
_LINEAR_ATTN = {"linear_attn", "linear_attention", "mamba", "ssm", "mixer", "time_mix"}
_ATTN = {"self_attn", "attn", "attention", "self_attention"}
_MLP = {"mlp", "feed_forward", "ffn", "block_sparse_moe", "moe", "channel_mix"}
_LAYER = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)(?:\.|$)")
_EXPERT_INDEX = re.compile(r"\.experts\.(\d+)\.")

MOE_CAVEAT = ("MoE: counted top_k / num_experts of each MoE layer's routed-expert bytes. Batch-1 decode reads "
              "exactly top_k experts per MoE layer per token, so this is exact at batch 1; it assumes no expert "
              "reuse across consecutive tokens from cache (experts are MBs, L2 does not keep them). At batch > 1 "
              "tokens in the same step share any expert they both pick, so bytes per token fall. Shared experts, "
              "the dense MLP and the router are read in full every token.")


def _first(c: dict, keys) -> int | None:
    for k in keys:
        if c.get(k) not in (None, 0):
            return int(c[k])
    return None


def _module(comps: list[str]) -> str:
    """The module a parameter belongs to: 'q_norm' for 'layers.0.self_attn.q_norm.weight'."""
    return comps[-2] if len(comps) > 1 and comps[-1] in ("weight", "bias") else comps[-1]


def _is_norm(comps: list[str]) -> bool:
    m = _module(comps)
    return "norm" in m or m.startswith("ln_") or m in ("ln", "layer_scalar", "layer_scale")


def classify(name: str, c: dict, config: dict) -> tuple[str, int | None]:
    """(group, layer index) of one tensor. Groups outside GROUPS: 'tower', 'mtp', 'extra_layer'."""
    comps = name.split(".")
    if any(x in _TOWER or x.startswith(("vision", "audio")) for x in comps):
        return "tower", None
    if comps[0] in ("mtp", "nextn") or any(x.startswith("mtp") or x == "nextn" or x.startswith("nextn_")
                                           for x in comps):
        return "mtp", None
    m = _LAYER.search(name)
    if m is None:
        # 'lm_head.weight', or nested as in 'language_model.lm_head.weight' (older multimodal saves)
        if "lm_head" in comps or "embed_out" in comps or name == "output.weight":
            return "lm_head", None
        if any(x in _EMBED for x in comps):
            return "embedding", None
        if _is_norm(comps):
            return "norms", None
        return "other", None
    layer = int(m.group(1))
    n_layers = int(c.get("num_hidden_layers") or 0)
    if n_layers and layer >= n_layers:
        has_mtp = _first(c, MTP_KEYS) or _first(config, MTP_KEYS)
        return ("mtp" if has_mtp else "extra_layer"), layer
    if "shared_expert" in name or "shared_experts" in name:
        return "moe_shared", layer
    if "experts" in comps:
        return "moe_experts", layer
    if "router" in comps or "e_score_correction_bias" in comps or ("gate" in comps and _MLP & set(comps)):
        return "moe_router", layer
    if _is_norm(comps):
        return "norms", layer
    if _LINEAR_ATTN & set(comps):
        return "linear_attention", layer
    if _ATTN & set(comps):
        return "attention", layer
    if _MLP & set(comps):
        return "mlp_dense", layer
    return "other", layer


def _pattern(name: str) -> str:
    return re.sub(r"\.\d+\.", ".N.", name)


# --------------------------------------------------------------------------- weight formats

_FLOAT = {"BF16": "bf16", "F16": "fp16", "F32": "fp32", "F64": "fp64"}
# leaves of a module's quantization tensors ('<module>.weight_scale' belongs to '<module>.weight')
_SCALE_LEAVES = {"weight_scale", "weight_scale_2", "weight_scale_inv", "input_scale", "weight_global_scale",
                 "input_global_scale", "weight_zero_point", "scales", "qzeros", "g_idx"}
_SCALE_SUFFIXES = ("_scale_2", "_scale_inv", "_scale", "_scales")


def _format(w: Tensor, scale: Tensor | None) -> str:
    """The format of one weight from its stored dtype and its scale tensor's dtype and grouping."""
    if w.dtype == "U8" and scale is not None and w.shape and scale.shape:
        if w.name.endswith("_blocks"):  # gpt-oss style MXFP4: [..., blocks, 16 bytes] + [..., blocks] scales
            return "mxfp4"
        group = (w.shape[-1] * 2) / scale.shape[-1]  # two FP4 values per byte
        if scale.dtype == "F8_E4M3" and group == 16:
            return "nvfp4"
        if scale.dtype in ("U8", "F8_E8M0") and group == 32:
            return "mxfp4"
    if w.dtype.startswith("F8_"):
        return "fp8"
    return _FLOAT.get(w.dtype, w.dtype.lower())


def tensor_formats(tensors: dict[str, Tensor]) -> dict[str, str]:
    """Format label per tensor. Scale tensors take their weight's format (NVFP4 values are stored as
    U8 pairs and their per-16 scales as F8_E4M3: both are 'nvfp4'); other tensors their own dtype."""

    def scale_of(name: str) -> Tensor | None:
        stem = name.removesuffix("_blocks")
        return next((tensors[n] for n in (name + "_scale", name + "_scale_inv", stem + "_scales")
                     if n in tensors), None)

    def owner(name: str) -> str | None:
        """The weight a scale tensor belongs to, or None when ``name`` is not a scale tensor."""
        module, _, leaf = name.rpartition(".")
        if leaf in _SCALE_LEAVES and f"{module}.weight" in tensors:
            return f"{module}.weight"
        for suf in _SCALE_SUFFIXES:
            if name.endswith(suf):
                base = name.removesuffix(suf)
                for cand in (base, base + "_blocks"):
                    if cand in tensors and cand != name:
                        return cand
        return None

    out = {n: _format(t, scale_of(n)) for n, t in tensors.items() if owner(n) is None}
    for n, t in tensors.items():
        if n not in out:
            out[n] = out.get(owner(n)) or _format(t, None)
    return out


# --------------------------------------------------------------------------- per-layer kinds

_KIND = {"full_attention": "full", "attention": "full", "global": "full", "global_attention": "full",
         "sliding_attention": "sliding", "sliding": "sliding", "local": "sliding", "local_attention": "sliding",
         "linear_attention": "linear", "gated_deltanet": "linear", "mamba": "linear", "chunked_attention": "chunked"}


def layer_kinds(c: dict, by_layer: dict[int, set[str]], notes: list[str]) -> list[str]:
    """'full' | 'sliding' | 'linear' | 'chunked' | 'unknown' per decoder layer."""
    n = int(c.get("num_hidden_layers") or (max(by_layer) + 1 if by_layer else 0))
    types = c.get("layer_types")
    if isinstance(types, list) and len(types) == n:
        return [_KIND.get(str(t), "unknown") for t in types]
    kinds = []
    for i in range(n):
        groups = by_layer.get(i, set())
        if "linear_attention" in groups:
            kinds.append("linear")
        elif "attention" in groups:
            kinds.append("full")
        else:
            kinds.append("unknown")
    window = c.get("sliding_window")
    if window and c.get("use_sliding_window") is not False and "full" in kinds:
        notes.append(f"config sets sliding_window={window} without layer_types: every attention layer is counted "
                     f"as full attention (an upper bound on KV reads; exact when context <= {window})")
    return kinds


# --------------------------------------------------------------------------- KV cache

def _rows(t: Tensor | None) -> int | None:
    return t.shape[0] if t is not None and t.shape else None


def kv_width(i: int, kind: str, c: dict, layer_tensors: dict[str, Tensor], notes: list[str]) -> tuple[int, str]:
    """Elements cached per token in layer i (K and V together), and where the number came from."""
    if c.get("kv_lora_rank"):  # MLA: one latent (+ the rope part of K) per token, no separate V
        return int(c["kv_lora_rank"]) + int(c.get("qk_rope_head_dim") or 0), "config (MLA latent)"
    heads = int(c.get("num_attention_heads") or 1)
    if kind == "full" and c.get("num_global_key_value_heads"):
        kvh = int(c["num_global_key_value_heads"])
    else:
        kvh = int(c.get("num_key_value_heads") or heads)
    if kind == "full" and c.get("global_head_dim"):
        hd = int(c["global_head_dim"])
    else:
        hd = int(c.get("head_dim") or int(c["hidden_size"]) // heads)
    cfg_k = kvh * hd

    def find(suffixes):
        return next((t for n, t in layer_tensors.items() if n.endswith(suffixes)), None)

    k_rows = _rows(find(("k_proj.weight", ".wk.weight", ".key.weight")))
    v_rows = _rows(find(("v_proj.weight", ".wv.weight", ".value.weight")))
    if k_rows is None:
        return 2 * cfg_k, "config"
    if v_rows is None:
        if c.get("attention_k_eq_v"):
            notes.append("attention_k_eq_v: full-attention layers have no v_proj (V comes from the K projection); "
                         "K and V are still counted as two cached tensors, as engines store both")
        v_rows = k_rows
    if k_rows != cfg_k:
        notes.append(f"layer {i}: k_proj has {k_rows} output rows, config gives {kvh} kv heads x {hd} = {cfg_k}; "
                     f"used the checkpoint's {k_rows}")
    return k_rows + v_rows, "checkpoint k_proj/v_proj rows"


# --------------------------------------------------------------------------- recurrent state

GDN_KEYS = ("linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim", "linear_num_key_heads")


def gdn_state(c: dict, config: dict, layer_tensors: dict[str, Tensor], state_dtype: str | None,
              conv_dtype: str) -> dict:
    """Gated DeltaNet state of one layer (Qwen3-Next / Qwen3.5-style config names)."""
    if not all(c.get(k) for k in GDN_KEYS):
        keys = sorted(k for k in c if any(s in k for s in ("linear", "ssm", "mamba", "state", "conv")))
        return {"status": "UNPROVEN", "reason": "linear-attention layers, but the config does not use the Gated "
                "DeltaNet names " + ", ".join(GDN_KEYS) + f"; state layout unknown (config keys seen: {keys})"}
    nv, nk = int(c["linear_num_value_heads"]), int(c["linear_num_key_heads"])
    dk, dv = int(c["linear_key_head_dim"]), int(c["linear_value_head_dim"])
    kernel = int(c.get("linear_conv_kernel_dim") or 4)
    conv_dim = 2 * nk * dk + nv * dv  # q, k and v channels go through the causal conv
    # Cross-check against the tensors: conv1d [conv_dim, 1, kernel], in_proj_z [nv*dv, hidden], A_log [nv].
    conv = next((t for n, t in layer_tensors.items() if n.endswith("conv1d.weight")), None)
    z = next((t for n, t in layer_tensors.items() if n.endswith("in_proj_z.weight")), None)
    a_log = next((t for n, t in layer_tensors.items() if n.endswith("A_log")), None)
    problems = []
    if conv is not None and (conv.shape[0] != conv_dim or conv.shape[-1] != kernel):
        problems.append(f"conv1d is {list(conv.shape)}, config implies [{conv_dim}, 1, {kernel}]")
    if z is not None and z.shape[0] != nv * dv:
        problems.append(f"in_proj_z has {z.shape[0]} rows, config implies {nv * dv}")
    if a_log is not None and a_log.shape and a_log.shape[0] != nv:
        problems.append(f"A_log has {a_log.shape[0]} heads, config implies {nv}")
    if problems:
        return {"status": "UNPROVEN", "reason": "Gated DeltaNet config does not match the checkpoint: "
                + "; ".join(problems)}
    source = "flag"
    if state_dtype is None:
        cfg = CONFIG_DTYPE.get(str(c.get("mamba_ssm_dtype") or config.get("mamba_ssm_dtype") or ""))
        state_dtype, source = (cfg, "config mamba_ssm_dtype") if cfg else ("bf16", "default")
    recurrent = nv * dk * dv * STATE_BYTES[state_dtype]
    conv_state = conv_dim * (kernel - 1) * STATE_BYTES[conv_dtype]
    return {"status": "ok", "kind": "gated_deltanet", "num_value_heads": nv, "num_key_heads": nk,
            "key_head_dim": dk, "value_head_dim": dv, "conv_dim": conv_dim, "conv_kernel": kernel,
            "state_dtype": state_dtype, "state_dtype_source": source, "conv_dtype": conv_dtype,
            "recurrent_bytes": recurrent, "conv_bytes": conv_state,
            "read_write_bytes_per_token": 2 * (recurrent + conv_state)}


# --------------------------------------------------------------------------- the count

def header_decode_bytes(ckpt: Checkpoint, config: dict, context: int, kv_dtype: str = "bf16",
                        state_dtype: str | None = None, include_mtp: bool = False) -> dict:
    """Bytes read from DRAM to decode one token at batch 1, from real tensor sizes. No I/O."""
    if kv_dtype not in KV_BYTES:
        raise OpError(f"unsupported kv dtype {kv_dtype}; use one of {sorted(KV_BYTES)}")
    if state_dtype is not None and state_dtype not in STATE_BYTES:
        raise OpError(f"unsupported state dtype {state_dtype}; use one of {sorted(STATE_BYTES)}")
    c = text_config(config)
    notes: list[str] = []
    unproven: list[str] = []
    stored = defaultdict(int)
    read = defaultdict(float)
    count = defaultdict(int)
    by_dtype = defaultdict(float)
    by_format = defaultdict(float)
    formats = tensor_formats(ckpt.tensors)
    other_patterns: set[str] = set()
    by_layer: dict[int, set[str]] = defaultdict(set)
    layer_tensors: dict[int, dict[str, Tensor]] = defaultdict(dict)
    embeddings: list[Tensor] = []
    experts: list[tuple[Tensor, int]] = []

    def add(group: str, t: Tensor, nbytes: float):
        stored[group] += t.nbytes
        read[group] += nbytes
        count[group] += 1
        by_dtype[t.dtype] += nbytes
        by_format[formats[t.name]] += nbytes

    for t in ckpt.tensors.values():
        group, layer = classify(t.name, c, config)
        if layer is not None and group not in ("mtp", "extra_layer"):
            by_layer[layer].add(group)
            layer_tensors[layer][t.name] = t
        if group == "tower":
            stored["excluded_towers"] += t.nbytes
            count["excluded_towers"] += 1
        elif group == "mtp":
            if include_mtp:
                add("mtp", t, t.nbytes)
            else:
                stored["excluded_mtp"] += t.nbytes
                count["excluded_mtp"] += 1
        elif group == "embedding":
            embeddings.append(t)
        elif group == "moe_experts":
            experts.append((t, layer))
        else:
            if group in ("other", "extra_layer"):
                other_patterns.add(_pattern(t.name))
                group = "other"
            add(group, t, t.nbytes)

    # lm_head: a tied one is the embedding matrix, read in full for the logits and counted once.
    vocab = int(c.get("vocab_size") or 0)
    tied = bool(c.get("tie_word_embeddings", config.get("tie_word_embeddings", False)))
    main_embed = next((t for t in embeddings if t.name.endswith(("embed_tokens.weight", "wte.weight",
                                                                 "word_embeddings.weight", "tok_embeddings.weight"))),
                      None)
    tied_used = False
    if count["lm_head"] == 0:
        if tied and main_embed is not None:
            add("lm_head", main_embed, main_embed.nbytes)
            tied_used = True
            notes.append(f"tied lm_head: {main_embed.name} is read in full for the logits; counted once, as lm_head")
        else:
            unproven.append("lm_head: no lm_head tensor and tie_word_embeddings is false; the logits matrix "
                            "is missing from the count")
    for t in embeddings:
        if tied_used and t is main_embed:
            continue  # already read in full as the lm_head
        rows = t.shape[0] if t.shape else 1
        add("embedding_row", t, t.nbytes / rows)  # stored = the whole table, read = one row
    if vocab and main_embed is not None and main_embed.shape and main_embed.shape[0] != vocab:
        notes.append(f"{main_embed.name} has {main_embed.shape[0]} rows, config vocab_size is {vocab}")

    # MoE: top_k of num_experts routed experts per MoE layer.
    moe = None
    if experts:
        n_exp = _first(c, N_EXPERTS_KEYS)
        top_k = _first(c, TOP_K_KEYS)
        if not n_exp or not top_k:
            raise OpError("the checkpoint has routed-expert tensors but the config gives no expert count / top-k "
                          f"(looked for {N_EXPERTS_KEYS} and {TOP_K_KEYS})")
        frac = top_k / n_exp
        seen: dict[int, set[int]] = defaultdict(set)
        fused_lead: set[int] = set()
        for t, layer in experts:
            add("moe_experts", t, t.nbytes * frac)
            m = _EXPERT_INDEX.search(t.name)
            if m:
                seen[layer].add(int(m.group(1)))
            elif t.shape and t.name.endswith(("proj", "weight", "_blocks", "_scales", "gate_up_proj", "down_proj")):
                fused_lead.add(t.shape[0])
        per_layer_counts = sorted({len(v) for v in seen.values()})
        if per_layer_counts and per_layer_counts != [n_exp]:
            notes.append(f"expert tensors per MoE layer: {per_layer_counts}, config num_experts = {n_exp}")
        if fused_lead and fused_lead != {n_exp}:
            notes.append(f"fused expert tensors lead with {sorted(fused_lead)}, config num_experts = {n_exp}")
        moe_layers = sorted({layer for _, layer in experts})
        moe = {"num_experts": n_exp, "top_k": top_k, "fraction": frac, "moe_layers": len(moe_layers),
               "experts_stored_bytes": stored["moe_experts"], "experts_read_bytes": read["moe_experts"],
               "shared_expert_bytes": read["moe_shared"], "caveat": MOE_CAVEAT}

    weights = sum(read[g] for g in (*GROUPS, "mtp"))

    # KV cache and recurrent state.
    kinds = layer_kinds(c, by_layer, notes)
    window = int(c.get("sliding_window") or 0)
    kv_elem = KV_BYTES[kv_dtype]
    kv_layers = []
    kv_total = 0.0
    slope = 0.0  # d(kv bytes) / d(context) at this context, for the report's context check
    state_layers = []
    state_total: float | None = 0.0
    model_dtype = CONFIG_DTYPE.get(str(c.get("dtype") or c.get("torch_dtype") or config.get("torch_dtype")
                                       or config.get("dtype") or "bfloat16"), "bf16")
    for i, kind in enumerate(kinds):
        lt = layer_tensors.get(i, {})
        if kind in ("full", "sliding"):
            width, source = kv_width(i, kind, c, lt, notes)
            if kind == "sliding":
                if not window:
                    unproven.append(f"layer {i}: sliding attention but the config has no sliding_window")
                    continue
                tokens = min(context, window)
            else:
                tokens = context
            per_ctx = width * kv_elem
            kv_layers.append({"layer": i, "kind": kind, "elements_per_token": width, "source": source,
                              "tokens_read": tokens, "bytes": per_ctx * tokens})
            kv_total += per_ctx * tokens
            if kind == "full" or context < window:
                slope += per_ctx
        elif kind == "linear":
            st = gdn_state(c, config, lt, state_dtype, model_dtype)
            st["layer"] = i
            state_layers.append(st)
            if st["status"] != "ok":
                state_total = None
                unproven.append(f"layer {i}: {st['reason']}")
            elif state_total is not None:
                state_total += st["read_write_bytes_per_token"]
        else:
            unproven.append(f"layer {i}: layer type '{kind}' has no KV/state model here")
    unproven = _dedupe_layers(unproven)
    notes = list(dict.fromkeys(notes))
    kind_counts = {k: kinds.count(k) for k in dict.fromkeys(kinds)}
    state_summary = None
    if state_layers:
        ok = [s for s in state_layers if s["status"] == "ok"]
        state_summary = {"status": "ok" if state_total is not None else "UNPROVEN", "layers": len(state_layers),
                         "per_layer": {k: v for k, v in (ok[0] if ok else state_layers[0]).items() if k != "layer"},
                         "bytes": state_total}
    total = weights + kv_total + (state_total or 0.0)
    groups = {g: {"read_bytes": read[g], "stored_bytes": stored[g], "tensors": count[g]}
              for g in (*GROUPS, "mtp") if count[g]}
    excluded = {k.removeprefix("excluded_"): {"stored_bytes": stored[k], "tensors": count[k]}
                for k in ("excluded_towers", "excluded_mtp") if count[k]}
    return {
        "method": "headers",
        "checkpoint": {"source": ckpt.source, "commit": ckpt.commit, "files": ckpt.files,
                       "headers_digest": ckpt.headers_digest, "tensors": len(ckpt.tensors),
                       "stored_bytes": ckpt.stored_bytes},
        "groups": groups,
        "excluded": excluded,
        "other_patterns": sorted(other_patterns),
        "read_bytes_by_dtype": dict(sorted(by_dtype.items())),
        "read_bytes_by_format": dict(sorted(by_format.items())),
        "moe": moe,
        "layer_kinds": kind_counts,
        "sliding_window": window or None,
        "kv": {"dtype": kv_dtype, "bytes_per_element": kv_elem, "layers": _compress(kv_layers), "bytes": kv_total},
        "state": state_summary,
        "include_mtp": include_mtp,
        "lm_head_bytes": read["lm_head"],
        "weight_bytes": weights,
        "kv_bytes_per_context_token": slope,
        "context": context,
        "kv_bytes": kv_total,
        "state_bytes": state_total,
        "bytes_per_token": total,
        "notes": notes,
        "unproven": unproven,
    }


def _compress(kv_layers: list[dict]) -> list[dict]:
    """Group identical per-layer KV rows: [{kind, elements_per_token, tokens_read, layers: [...]}]."""
    out: dict[tuple, dict] = {}
    for row in kv_layers:
        key = (row["kind"], row["elements_per_token"], row["source"], row["tokens_read"])
        g = out.setdefault(key, {"kind": row["kind"], "elements_per_token": row["elements_per_token"],
                                 "source": row["source"], "tokens_read": row["tokens_read"], "layers": [],
                                 "bytes": 0.0})
        g["layers"].append(row["layer"])
        g["bytes"] += row["bytes"]
    return list(out.values())


def _dedupe_layers(lines: list[str]) -> list[str]:
    """'layer 3: X', 'layer 7: X' -> 'layers 3, 7: X'."""
    by_reason: dict[str, list[str]] = {}
    for line in lines:
        m = re.match(r"layer (\d+): (.*)", line)
        if m:
            by_reason.setdefault(m.group(2), []).append(m.group(1))
        else:
            by_reason.setdefault(line, [])
    return [(f"layers {', '.join(ls)}: {r}" if ls else r) for r, ls in by_reason.items()]
