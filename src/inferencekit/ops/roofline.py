"""roofline: the speed limit for batch-1 decode.

Two methods. ``--method headers`` (the default whenever safetensors are available) sums the
real stored bytes of every tensor from the checkpoint's safetensors headers, grouped by module
role, with MoE top-k, sliding windows and linear-attention state handled exactly; see
``inferencekit.roofline_headers``. ``--method config`` is the original dense estimate from
config.json, described below.

At batch 1, generating one token means reading every weight once (each weight is
used for a single multiply-add per token, so the GPU spends its time waiting on
memory, not computing) plus the KV cache of the context so far. So

    ceiling tok/s = DRAM bandwidth (bytes/s) / bytes read per token

No kernel can beat this on a given GPU; a measured tok/s as a percentage of it says
how much headroom the engine has left. Counted per token, for a dense
Llama/Qwen-style decoder:

  * per layer: q, k, v, o projections, gate / up / down MLP projections, the two
    RMSNorm weights (+ q/k norms for Qwen3), optional q/k/v biases;
  * the final norm and the full lm_head (vocab x hidden). A tied lm_head is the
    embedding matrix, still read in full every token to compute the logits;
  * embeddings: only the one row of the current token is read, not the matrix;
  * KV cache: 2 (K and V) x kv_heads x head_dim x bytes x layers x context.

Weight formats:
  bf16   2 bytes / weight
  fp8    1 byte / weight (e4m3) + one fp32 scale per 128x128 block = 4/16384 bytes per
         weight (+0.024%, kept for honesty, negligible); norms stay bf16
  q8_0   blocks of 32 int8 + one fp16 scale = 34 bytes / 32 weights = 8.5 bits;
         norms are fp32 in GGUF files
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from inferencekit import evidence as ev
from inferencekit.registry import Arg, OpError, operation

GB = 1e9  # decimal gigabytes: bandwidths are quoted in GB/s = 1e9 bytes/s

# bytes per weight of the large matrices, and of small vectors (norms, biases)
WEIGHT_BYTES = {"bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "q8_0": 34 / 32}
VECTOR_BYTES = {"bf16": 2.0, "fp16": 2.0, "fp8": 2.0, "q8_0": 4.0}
DEFAULT_LM_HEAD = {"bf16": "bf16", "fp16": "fp16", "fp8": "bf16", "q8_0": "q8_0"}
KV_BYTES = {"bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "q8_0": 34 / 32}
FP8_BLOCK = 128


def matrix_bytes(rows: int, cols: int, dtype: str) -> float:
    """Bytes to read one weight matrix of shape rows x cols in ``dtype``."""
    n = rows * cols
    if dtype == "fp8":
        blocks = math.ceil(rows / FP8_BLOCK) * math.ceil(cols / FP8_BLOCK)
        return n * 1.0 + blocks * 4.0  # e4m3 values + one fp32 scale per 128x128 block
    return n * WEIGHT_BYTES[dtype]


def text_config(config: dict) -> dict:
    """Vision-language configs nest the language model under text_config / llm_config."""
    if "hidden_size" in config:
        return config
    for key in ("text_config", "llm_config", "language_config"):
        if isinstance(config.get(key), dict) and "hidden_size" in config[key]:
            return config[key]
    raise OpError("config has no hidden_size (is this a decoder-only LLM config?)")


def decode_bytes(config: dict, dtype: str, context: int, kv_dtype: str = "bf16",
                 lm_head_dtype: str | None = None) -> dict:
    """Bytes read from DRAM to decode one token at batch 1. Pure function: no I/O."""
    if dtype not in WEIGHT_BYTES:
        raise OpError(f"unsupported dtype {dtype}; use one of {sorted(WEIGHT_BYTES)}")
    c = text_config(config)
    for moe_key in ("num_experts", "num_local_experts", "n_routed_experts"):
        if c.get(moe_key):
            raise OpError("mixture-of-experts configs need the real tensor sizes: use --method headers "
                          "(the config-only method covers dense models)")
    hidden = int(c["hidden_size"])
    layers = int(c["num_hidden_layers"])
    inter = int(c["intermediate_size"])
    heads = int(c["num_attention_heads"])
    kv_heads = int(c.get("num_key_value_heads") or heads)
    head_dim = int(c.get("head_dim") or hidden // heads)
    vocab = int(c["vocab_size"])
    tied = bool(c.get("tie_word_embeddings", config.get("tie_word_embeddings", False)))
    model_type = str(c.get("model_type") or config.get("model_type") or "")
    qkv_bias = bool(c.get("attention_bias", model_type.startswith("qwen2")))
    qk_norm = model_type.startswith("qwen3")
    lm_dtype = lm_head_dtype or DEFAULT_LM_HEAD[dtype]
    vec = VECTOR_BYTES[dtype]

    q_out, kv_out = heads * head_dim, kv_heads * head_dim
    per_layer = {
        "q_proj": matrix_bytes(q_out, hidden, dtype),
        "k_proj": matrix_bytes(kv_out, hidden, dtype),
        "v_proj": matrix_bytes(kv_out, hidden, dtype),
        "o_proj": matrix_bytes(hidden, q_out, dtype),
        "gate_proj": matrix_bytes(inter, hidden, dtype),
        "up_proj": matrix_bytes(inter, hidden, dtype),
        "down_proj": matrix_bytes(hidden, inter, dtype),
        "norms": 2 * hidden * vec + (2 * head_dim * vec if qk_norm else 0),
        "qkv_bias": (q_out + 2 * kv_out) * vec if qkv_bias else 0.0,
    }
    layer_total = sum(per_layer.values())
    linear_params = (q_out * hidden + 2 * kv_out * hidden + hidden * q_out + 3 * inter * hidden) * layers
    lm_head = matrix_bytes(vocab, hidden, lm_dtype)
    # One embedding row per token. FP8 checkpoints keep embeddings in bf16; GGUF quantises them.
    embed_dtype = lm_dtype if tied else ("bf16" if dtype == "fp8" else dtype)
    embed_row = hidden * WEIGHT_BYTES[embed_dtype]
    final_norm = hidden * vec
    weights = layer_total * layers + lm_head + embed_row + final_norm
    kv_per_token_of_context = 2 * kv_out * KV_BYTES[kv_dtype] * layers
    kv = kv_per_token_of_context * context
    return {
        "shape": {"hidden": hidden, "layers": layers, "intermediate": inter, "heads": heads, "kv_heads": kv_heads,
                  "head_dim": head_dim, "vocab": vocab, "tied_lm_head": tied, "qk_norm": qk_norm,
                  "qkv_bias": qkv_bias, "linear_params": linear_params},
        "dtypes": {"weights": dtype, "lm_head": lm_dtype, "kv": kv_dtype},
        "per_layer_bytes": per_layer,
        "layers_bytes": layer_total * layers,
        "lm_head_bytes": lm_head,
        "embedding_row_bytes": embed_row,
        "final_norm_bytes": final_norm,
        "weight_bytes": weights,
        "kv_bytes_per_context_token": kv_per_token_of_context,
        "context": context,
        "kv_bytes": kv,
        "bytes_per_token": weights + kv,
    }


def load_config(model: str, revision: str | None = None) -> tuple[dict, str, str | None]:
    """config.json from a file, a model directory, or the Hugging Face Hub.
    Returns (config, source, Hub commit when known)."""
    from inferencekit import checkpoint

    p = Path(model)
    if p.is_file():
        return json.loads(p.read_text()), "file", None
    if p.is_dir() and (p / "config.json").is_file():
        return json.loads((p / "config.json").read_text()), "directory", None
    try:  # use the hub client's cache, offline mode and auth when it is installed
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model, "config.json", revision=revision)
        return json.loads(Path(path).read_text()), "hub", checkpoint.snapshot_commit(Path(path))
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001 - surface hub errors as a user-facing message
        raise OpError(f"could not fetch config.json for '{model}': {e}") from None
    cached = checkpoint.cached_file(model, "config.json", revision)
    if cached is not None:
        return json.loads(cached.read_text()), "hub-cache", checkpoint.snapshot_commit(cached)
    raw, commit = checkpoint.hub_small_file(model, "config.json", revision)  # HF_TOKEN used, never recorded
    if raw is None:
        raise OpError(f"'{model}' has no config.json at revision {revision or 'main'}")
    return json.loads(raw.decode()), "hub", commit


def resolve_bandwidth(gpu_bw: str | None) -> tuple[float | None, dict, list[str]]:
    """GB/s from a number, an evidence id of kind=ceiling, or 'measure' (runs measure_bandwidth)."""
    if gpu_bw in (None, ""):
        return None, {"source": "none"}, []
    if gpu_bw == "measure":
        from inferencekit import registry

        rec = registry.call("measure_bandwidth", {})
        return _bw_from_record(rec)
    try:
        value = float(gpu_bw)
        return value, {"source": "user", "value_gbs": value}, [
            "bandwidth is a user-supplied figure; a datasheet number is usually 5-15% above what kernels can "
            "sustain, so the ceiling is optimistic (use --gpu-bw measure)"]
    except ValueError:
        pass
    return _bw_from_record(ev.Store().load(gpu_bw))


def _bw_from_record(rec: dict) -> tuple[float | None, dict, list[str]]:
    if rec.get("kind") != "ceiling":
        raise OpError(f"{rec.get('evidence_id')} is a '{rec.get('kind')}' record, expected kind=ceiling")
    problem = ev.integrity_problem(rec)
    if problem:
        raise OpError(f"{problem}; refusing to use it as the bandwidth")
    best = next((m for m in rec["metrics"] if m["name"] == "dram_read_gbs" and m["stat"] == "best_sustained"), None)
    if best is None or best["value"] is None:
        raise OpError(f"{rec['evidence_id']} has no best sustained read bandwidth")
    return float(best["value"]), {"source": "evidence", "evidence_id": rec["evidence_id"],
                                  "value_gbs": best["value"]}, []


def _precision_label(config: dict, by_format: dict) -> tuple[str, str | None]:
    """(dtype, quant) for the record identity. dtype: the weight formats (nvfp4, mxfp4, fp8, bf16...,
    each with its own scale tensors) that carry >= 1% of the bytes read, largest first. quant: the
    quant config's algorithm; a mixed-precision one names the algorithms of its quantized layers."""
    total = sum(by_format.values()) or 1.0
    main = [f for f, b in sorted(by_format.items(), key=lambda kv: -kv[1]) if b / total >= 0.01]
    q = config.get("quantization_config") or {}
    quant = str(q.get("quant_algo") or q.get("quant_method") or "").lower() or None
    layers = q.get("quantized_layers")
    if quant == "mixed_precision" and isinstance(layers, dict):
        algos: dict[str, int] = {}
        for v in layers.values():
            a = str((v or {}).get("quant_algo") or "").lower() if isinstance(v, dict) else ""
            if a:
                algos[a] = algos.get(a, 0) + 1
        if algos:
            quant += "(" + "+".join(a for a, _ in sorted(algos.items(), key=lambda kv: (-kv[1], kv[0]))) + ")"
    return ("+".join(main) or "unknown"), quant


def _config_weight_dtype(config: dict) -> tuple[str | None, str | None]:
    """(--dtype for the config method, problem) from the config's quantization_config, for the
    auto fallback: fp8 checkpoints are fp8, unquantized ones bf16; other formats the dense config
    method cannot represent are counted as bf16 and flagged."""
    q = config.get("quantization_config") or text_config(config).get("quantization_config") or {}
    if not q:
        return None, None
    names = {str(q.get(k) or "").lower() for k in ("quant_method", "quant_algo")} - {""}
    if names & {"fp8"} and not names - {"fp8", "modelopt", "compressed-tensors"}:
        return "fp8", None
    return None, (f"the config declares {'/'.join(sorted(names)) or 'a'} quantization, which the config-only "
                  "method does not model; weights counted as bf16 (bytes overstated, ceiling understated)")


def _declared_kv_dtype(config: dict) -> str | None:
    """The KV-cache format the quant config declares (no KV scales need to ship in the checkpoint)."""
    q = config.get("quantization_config") or {}
    scheme = q.get("kv_cache_scheme") or {}
    if isinstance(scheme, dict) and scheme.get("num_bits") == 8 and scheme.get("type") == "float":
        return "fp8"
    algo = str(q.get("kv_cache_quant_algo") or "").lower()
    return "fp8" if algo.startswith("fp8") else None


def _headers_roofline(model, revision, ckpt, config, context, kv_dtype, state_dtype, include_mtp, dtype_given):
    """(counts, summary prefix, limitations, missing, identity model dict) for --method headers."""
    from inferencekit.roofline_headers import header_decode_bytes

    counts = header_decode_bytes(ckpt, config, context, kv_dtype, state_dtype, include_mtp)
    limitations = list(counts["notes"])
    missing = [f"UNPROVEN: {u}" for u in counts["unproven"]]
    if dtype_given:
        limitations.append(f"--dtype {dtype_given} ignored: with --method headers every tensor's dtype comes "
                           "from the checkpoint")
    declared = _declared_kv_dtype(config)
    if declared and declared != kv_dtype:
        limitations.append(f"the checkpoint's quant config declares an {declared} KV cache but KV was counted as "
                           f"{kv_dtype}; pass --kv-dtype {declared} if the engine ran an {declared} cache")
    limitations.append(f"KV cache counted as {kv_dtype} (--kv-dtype); use the cache dtype the engine logged, "
                       "not the config's")
    st = counts["state"]
    if st and st["status"] == "ok":
        per = st["per_layer"]
        limitations.append(f"linear-attention state counted as {per['state_dtype']} ({per['state_dtype_source']}), "
                           f"conv state as {per['conv_dtype']}, each read and written once per token; engines pick "
                           "their own state dtype (e.g. vLLM --mamba-ssm-cache-dtype), so pass --state-dtype to "
                           "match what the engine logged")
    if counts["moe"]:
        limitations.append(counts["moe"]["caveat"])
    ex = counts["excluded"]
    if "towers" in ex:
        limitations.append(f"vision/audio towers and projectors ({ex['towers']['stored_bytes'] / GB:.3f} GB stored) "
                           "excluded: text decode does not run them")
    if "mtp" in ex:
        limitations.append(f"MTP / next-n head ({ex['mtp']['stored_bytes'] / GB:.3f} GB stored) excluded; with "
                           "MTP speculative decoding on, rerun with --include-mtp and read the result per step, "
                           "not per token")
    if counts["other_patterns"]:
        limitations.append(f"{len(counts['other_patterns'])} tensor pattern(s) not matched to a module role were "
                           "counted as read every token (group 'other'): "
                           + ", ".join(counts["other_patterns"][:8]))
    dtype_label, quant = _precision_label(config, counts["read_bytes_by_format"])
    revision_ = ckpt.commit or revision
    model_id = {"id": model, "revision": revision_, "dtype": dtype_label, "quant": quant}
    kv, state = counts["kv_bytes"], counts["state_bytes"]
    terms = f"weights {counts['weight_bytes'] / GB:.3f} + KV {kv / GB:.3f}"
    if counts["state"] is not None:
        terms += " + state " + ("UNPROVEN" if state is None else f"{state / GB:.3f}")
    prefix = (f"{model} ({dtype_label}): {counts['bytes_per_token'] / GB:.3f} GB/token at context {context} "
              f"[safetensors headers: {terms} GB]")
    return counts, prefix, limitations, missing, model_id


@operation("roofline", ("roofline",), "Bytes read per decode token (weights + KV cache / recurrent state at a "
           "context length) and the bandwidth ceiling in tok/s. Default: exact tensor sizes from the safetensors "
           "headers (NVFP4/FP8/MoE/hybrid/sliding-window aware, no weights downloaded); --method config for the "
           "config-only dense estimate.",
           [Arg("model", "string", "Hugging Face id, a model directory, or a config.json path.", required=True),
            Arg("dtype", "string", "Weight format of the linear layers (--method config only; headers read the "
                "real dtypes). Default bf16.", choices=("fp8", "bf16", "q8_0")),
            Arg("gpu_bw", "string", "DRAM bandwidth: GB/s number, 'measure', or an evidence id of kind=ceiling."),
            Arg("context", "integer", "Tokens already in the KV cache when the token is decoded.", default=1024),
            Arg("kv_dtype", "string", "KV cache format.", default="bf16", choices=("bf16", "fp16", "fp8", "q8_0")),
            Arg("lm_head_dtype", "string", "lm_head format for --method config (default: bf16 for fp8 "
                "checkpoints, else the weight dtype).", choices=("bf16", "fp16", "fp8", "q8_0")),
            Arg("revision", "string", "Hub revision (branch, tag or commit)."),
            Arg("method", "string", "headers: sum real tensor bytes from the safetensors headers; config: dense "
                "estimate from config.json; auto: headers whenever safetensors are available.", default="auto",
                choices=("auto", "headers", "config")),
            Arg("state_dtype", "string", "Linear-attention (Gated DeltaNet) recurrent-state format (default: the "
                "config's mamba_ssm_dtype, else bf16).", choices=("fp32", "bf16", "fp16")),
            Arg("include_mtp", "boolean", "Count the MTP / next-n head (excluded by default: plain decode does "
                "not run it).", default=False)])
def roofline(call, model, dtype, gpu_bw, context, kv_dtype, lm_head_dtype, revision, method="auto",
             state_dtype=None, include_mtp=False):
    from inferencekit import checkpoint

    ckpt = None
    no_headers = "no safetensors headers available"
    if method in ("auto", "headers"):
        try:
            ckpt = checkpoint.load(model, revision)
        except checkpoint.HubUnavailable as e:
            if method == "headers":
                raise
            no_headers = f"safetensors headers unavailable ({e})"
        if ckpt is None and method == "headers":
            raise OpError(f"no safetensors files found for '{model}'; use --method config")
    if ckpt is not None and ckpt.local_dir is not None and (ckpt.local_dir / "config.json").is_file():
        # headers from the local HF cache: the config of the same snapshot
        config, source, commit = json.loads((ckpt.local_dir / "config.json").read_text()), "hub-cache", ckpt.commit
    else:
        # pin the config to the commit the headers came from, so both describe the same checkpoint
        config, source, commit = load_config(model, (ckpt.commit if ckpt else None) or revision)
    limitations: list[str] = []
    missing: list[str] = []
    if ckpt is not None:
        counts, summary, limitations, missing, model_id = _headers_roofline(
            model, revision, ckpt, config, context, kv_dtype, state_dtype, include_mtp, dtype)
        if counts["unproven"]:
            summary += f" ({len(counts['unproven'])} UNPROVEN term(s): ceiling is an upper bound)"
        if ckpt.source == "hub-cache":
            limitations.append(f"headers read from the local Hugging Face cache (snapshot {ckpt.commit}), not "
                               "fetched from the Hub")
    else:
        if method == "auto" and dtype is None:  # a fallback: take the format the config declares
            dtype, why = _config_weight_dtype(config)
            if why:
                limitations.append(why)
                missing.append(f"UNPROVEN: {why}")
        dtype = dtype or "bf16"
        counts = decode_bytes(config, dtype, context, kv_dtype, lm_head_dtype)
        counts["method"] = "config"
        summary = f"{model} {dtype}: {counts['bytes_per_token'] / GB:.3f} GB/token at context {context}"
        model_id = {"id": model, "revision": commit or revision, "dtype": dtype,
                    "quant": dtype if dtype in ("fp8", "q8_0") else None}
        if method == "auto":
            limitations.append(f"{no_headers}: config-only estimate (--method config)")
    # The stored request pins what was resolved (the method that ran, the commit read), so the
    # reproduce command rebuilds this record rather than whatever 'auto' / 'main' resolve to later.
    request = call.request()
    requested = {k: request["args"].get(k) for k in ("method", "revision", "dtype")}
    request["args"]["method"] = counts["method"]
    if model_id["revision"] and checkpoint.is_hub_id(model):
        request["args"]["revision"] = model_id["revision"]
    if counts["method"] == "config":
        request["args"]["dtype"] = dtype  # the format counted, also when the auto fallback inferred it
    if requested != {k: request["args"].get(k) for k in requested}:
        request["requested"] = requested
    bw, bw_info, bw_limits = resolve_bandwidth(gpu_bw)
    per_token = counts["bytes_per_token"]
    metrics = [
        ev.metric("bytes_per_token", per_token, "B", "derived"),
        ev.metric("weight_bytes", counts["weight_bytes"], "B", "derived"),
        ev.metric("kv_bytes", counts["kv_bytes"], "B", "derived"),
    ]
    if counts["method"] == "headers":
        if counts["state"] is not None:
            metrics.append(ev.metric("state_bytes", counts["state_bytes"], "B", "derived"))
        metrics += [ev.metric(f"bytes_{g}", v["read_bytes"], "B", "derived") for g, v in counts["groups"].items()]
    if bw:
        ceiling = bw * GB / per_token
        metrics += [ev.metric("ceiling_tok_s", ceiling, "tok/s", "ceiling"),
                    ev.metric("min_token_latency_ms", 1e3 / ceiling, "ms", "ceiling"),
                    ev.metric("bandwidth_gbs", bw, "GB/s", "input")]
        summary += f"; ceiling {ceiling:.1f} tok/s at {bw:.0f} GB/s"
    else:
        missing.append("bandwidth (pass --gpu-bw)")
    limitations = bw_limits + limitations
    limitations += ["assumes every weight is read once per token from DRAM (true at batch 1; L2 reuse across "
                    "tokens is negligible for multi-GB models)",
                    "ignores activations, logits and sampling traffic (small next to the weights)"]
    ident = ev.identity(model=model_id, workload={"prompt_len": context, "gen_len": 1, "batch": 1})
    record = ev.build(call, kind="roofline", claim_class="derived", summary=summary, identity_=ident,
                      metrics=metrics, missing=missing, limitations=limitations,
                      data={"counts": counts, "bandwidth": bw_info, "config_source": source, "request": request})
    return ev.Store().save(record)
