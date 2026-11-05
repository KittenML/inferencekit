"""Hard gates: configurations whose decode numbers must not be presented as an engine's speed.

A benchmark that silently ran without CUDA graphs, or on a fallback attention
kernel, measures a misconfiguration rather than the engine. Batch-1 decode is
dominated by per-kernel launch overhead without CUDA graphs (each token launches
hundreds of small kernels; a graph replays them with one launch), and fallback
attention kernels can be several times slower. Such runs stay recorded, but their
violations go into ``limitations`` and the claim is marked invalid.
"""

from __future__ import annotations

import re

# Attention implementations that engines use only when the fast path is unavailable.
FALLBACK_ATTENTION = {"TORCH_SDPA", "XFORMERS", "FLEX_ATTENTION", "TORCH_NATIVE"}
# Legitimate on some hardware, a fallback on others: recorded as a warning.
SUSPECT_ATTENTION = {"TRITON_ATTN", "TRITON"}

# Keyed by the name with separators removed, so "Flash Attention", "FLASH_ATTN" and
# "flash-attn" all map to one canonical spelling.
_ALIASES = {"FLASHATTENTION": "FLASH_ATTN", "FLASHATTN": "FLASH_ATTN", "FLASHINFER": "FLASHINFER",
            "TRITONATTENTION": "TRITON_ATTN", "TRITONATTN": "TRITON_ATTN", "TORCHSDPA": "TORCH_SDPA",
            "FLEXATTENTION": "FLEX_ATTENTION", "XFORMERS": "XFORMERS", "TORCHNATIVE": "TORCH_NATIVE",
            "FA3": "FA3", "FLASHMLA": "FLASHMLA", "CUTLASSMLA": "CUTLASS_MLA"}


def normalize_backend(name: str | None) -> str | None:
    if not name:
        return None
    n = name.strip().upper().replace("ATTENTIONBACKENDENUM.", "").replace(" ", "_").replace("-", "_")
    return _ALIASES.get(n.replace("_", ""), n)


def _is_attention(name: str) -> bool:
    # "Using X backend" also appears for MoE, sampling and communication; keep attention only.
    return "ATTN" in name or "ATTENTION" in name or "MLA" in name or name in _ALIASES.values()


def attention_backend_from_log(log_text: str) -> str | None:
    """vLLM logs e.g. 'Using Flash Attention backend' or 'Using FLASHINFER backend'. Last mention wins."""
    hits = re.findall(r"Using (?:AttentionBackendEnum\.)?([A-Za-z0-9_][A-Za-z0-9_ ]*?) (?:attention )?backend",
                      log_text)
    names = [normalize_backend(h) for h in hits]
    names = [n for n in names if n and _is_attention(n)]
    return names[-1] if names else None


def check(engine: str, flags: dict, attention_backend: str | None) -> dict:
    """Return {'valid': bool | None, 'violations': [...], 'warnings': [...], 'checked': [...]}.

    valid is None when the adapter could not see enough to check (e.g. a ``cmd``
    benchmark that reported no flags): unknown is not the same as fine.
    """
    violations, warnings, checked = [], [], []
    if engine == "vllm":
        checked.append("cuda_graphs")
        if flags.get("enforce_eager") is True:
            violations.append("CUDA graphs off (enforce_eager=True)")
        mode = str(flags.get("cudagraph_mode") or "")
        if mode.upper().endswith("NONE"):
            violations.append(f"CUDA graphs off (cudagraph_mode={mode})")
        sizes = flags.get("cudagraph_capture_sizes")
        if isinstance(sizes, dict) and sizes.get("count") == 0 and flags.get("enforce_eager") is not True:
            violations.append("CUDA graphs off (no capture sizes)")
    elif engine == "sglang":
        checked.append("cuda_graphs")
        if flags.get("disable_cuda_graph") is True:
            violations.append("CUDA graphs off (disable_cuda_graph=True)")
    elif engine == "llamacpp":
        checked.append("gpu_offload")
        ngl = flags.get("n_gpu_layers")
        if ngl == 0:
            violations.append("no layers on the GPU (n_gpu_layers=0)")
        elif isinstance(ngl, int) and ngl < 99:
            warnings.append(f"n_gpu_layers={ngl}: check that every layer is offloaded")
        backends = str(flags.get("backends") or "")
        if backends and not re.search(r"CUDA|Vulkan|Metal|HIP|SYCL|MUSA", backends, re.IGNORECASE):
            violations.append(f"no GPU backend in this llama.cpp build (backends={backends})")
    elif engine == "cmd":
        if "cuda_graphs" in flags:
            checked.append("cuda_graphs")
            if flags["cuda_graphs"] is False:
                violations.append("CUDA graphs off (reported by the command)")
    backend = normalize_backend(attention_backend or flags.get("attention_backend"))
    if backend:
        checked.append("attention_backend")
        if backend in FALLBACK_ATTENTION:
            violations.append(f"fallback attention backend ({backend})")
        elif backend in SUSPECT_ATTENTION:
            warnings.append(f"attention backend {backend} is a fallback on many NVIDIA GPUs: confirm it is intended")
    valid = (not violations) if checked else None
    return {"valid": valid, "violations": violations, "warnings": warnings, "checked": checked,
            "attention_backend": backend}
