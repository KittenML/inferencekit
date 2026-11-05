"""Mixture-of-experts building blocks for model modules: the routing step and an expert MLP over stored weights.

    router logits [T, E] --route--> (weights fp32 [T, k], ids int64 [T, k]) --expert_mlp--> [T, H]

route():
- scoring "softmax": p = softmax(logits) in fp32 over all experts, then the top k of p (Mixtral, Qwen MoE,
  Gemma 4). scoring "sigmoid": s = sigmoid(logits) in fp32; the top k are chosen on s + correction_bias when
  the model has one (DeepSeek-V3 style), and the weights come from s without the bias.
- renormalize: the k weights are divided by their sum (Qwen's norm_topk_prob, Gemma 4).
- then weights *= per_expert_scale[ids] when the model stores one (Gemma 4), and *= routed_scaling.
- Exact ties at the k-th place go to torch.topk's choice; a fused kernel may break them differently.

ExpertStack holds one projection of every expert as stored (QWeight tensors with a leading [E] dimension), so
NVFP4 and FP8 experts keep their packed values and scales. `gather(ids)` picks one expert per row with
index_select: static shapes, no host synchronisation, so it can sit inside the decode CUDA graph, and
qlinear applies row b's own weight and scales to row b.

expert_mlp() computes, for every (token, slot), down(act(gate(x)) * up(x)) with qlinear (rounded to x's dtype
like a dense MLP), then sums the k outputs times their fp32 weights in fp32 and rounds once. Small batches
(decode) use the gather path; long prompts loop over the experts that were picked (that path synchronises
with the host, which is fine in eager prefill). Both paths give the same values up to fp32 summation order.
Engines differ in where they round (Hugging Face rounds the weights to the activation dtype and sums in it,
vLLM's fused kernels multiply the weights into fp32 accumulators): a model module that needs one engine's
exact order says so and implements it.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Callable

from .loader import QWeight
from .qlinear import ACT_AS_SHIPPED, qlinear

GATHER_MAX_ROWS = 64        # T * k up to this uses the gather path (static shapes); above it, the expert loop
_TENSORS = ("weight", "scale", "scale_2", "input_scale", "bias")


def route(logits, top_k: int, scoring: str = "softmax", renormalize: bool = True, per_expert_scale=None,
          correction_bias=None, routed_scaling: float = 1.0):
    """Router logits [T, E] -> (weights fp32 [T, k], ids int64 [T, k]), in descending order of the selection
    score. Graph safe."""
    import torch
    lf = logits.float()
    if scoring == "softmax":
        scores = lf.softmax(dim=-1)
        sel = scores
    elif scoring == "sigmoid":
        scores = lf.sigmoid()
        sel = scores if correction_bias is None else scores + correction_bias.float()
    else:
        raise ValueError(f"scoring must be 'softmax' or 'sigmoid', got {scoring!r}")
    _, ids = torch.topk(sel, top_k, dim=-1)
    w = scores.gather(-1, ids)
    if renormalize:
        w = w / w.sum(dim=-1, keepdim=True)
    if per_expert_scale is not None:
        w = w * per_expert_scale.float()[ids]
    if routed_scaling != 1.0:
        w = w * float(routed_scaling)
    return w, ids


@dataclass
class ExpertStack:
    """One projection for all E experts, as stored. Every tensor of `stored` has a leading [E] dimension."""
    stored: QWeight

    @classmethod
    def from_linears(cls, linears: list[QWeight]) -> "ExpertStack":
        """Stack per-expert linears (as Checkpoint.linear returns them). All must share kind, shape and scales."""
        import torch
        first = linears[0]
        for e, w in enumerate(linears):
            if (w.kind, w.shape, w.block) != (first.kind, first.shape, first.block) or \
                    any((getattr(w, f) is None) != (getattr(first, f) is None) for f in _TENSORS):
                raise ValueError(f"expert {e} is stored differently from expert 0")
        stacked = {f: torch.stack([getattr(w, f) for w in linears]) for f in _TENSORS if getattr(first, f) is not None}
        return cls(replace(first, **stacked))

    @classmethod
    def from_dense(cls, weight, kind: str | None = None) -> "ExpertStack":
        """A 16-bit stack [E, N, K] (e.g. one half of a Hugging Face gate_up_proj)."""
        kind = kind or {"torch.bfloat16": "bf16", "torch.float16": "fp16"}.get(str(weight.dtype), "fp32")
        return cls(QWeight(kind, weight, (int(weight.shape[1]), int(weight.shape[2]))))

    @property
    def num_experts(self) -> int:
        return int(self.stored.weight.shape[0])

    def expert(self, e: int) -> QWeight:
        """Expert e's weight as an ordinary 2-D QWeight."""
        return replace(self.stored, **{f.name: getattr(self.stored, f.name)[e] for f in fields(QWeight)
                                       if f.name in _TENSORS and getattr(self.stored, f.name) is not None})

    def gather(self, ids) -> QWeight:
        """One expert per row: ids int64 [B] -> a QWeight whose tensors are [B, ...] (graph safe)."""
        flat = ids.reshape(-1)
        return replace(self.stored, **{f: getattr(self.stored, f).index_select(0, flat) for f in _TENSORS
                                       if getattr(self.stored, f) is not None})


def _mlp(x, gate: QWeight, up: QWeight, down: QWeight, act_fn: Callable, act: str):
    return qlinear(act_fn(qlinear(x, gate, act)) * qlinear(x, up, act), down, act)


def expert_mlp(x, weights, ids, gate: ExpertStack, up: ExpertStack, down: ExpertStack, act_fn: Callable,
               act: str = ACT_AS_SHIPPED, gather_max_rows: int = GATHER_MAX_ROWS) -> Any:
    """x [T, H]; weights fp32 [T, k]; ids int64 [T, k] -> [T, H] in x's dtype (see the module docstring)."""
    import torch
    t, k = ids.shape
    flat = ids.reshape(-1)
    if t * k <= gather_max_rows:
        xr = x.repeat_interleave(k, dim=0)                                         # row t*k + j is token t
        y = _mlp(xr, gate.gather(flat), up.gather(flat), down.gather(flat), act_fn, act)
        per_slot = y.float().reshape(t, k, -1)
    else:
        per_slot = torch.zeros(t * k, down.stored.shape[0], dtype=torch.float32, device=x.device)
        for e in torch.unique(flat).tolist():                                       # host sync: eager only
            rows = (flat == e).nonzero().squeeze(1)
            y = _mlp(x.index_select(0, rows // k), gate.expert(e), up.expert(e), down.expert(e), act_fn, act)
            per_slot.index_copy_(0, rows, y.float())
        per_slot = per_slot.reshape(t, k, -1)
    return (per_slot * weights.float().unsqueeze(-1)).sum(dim=1).to(x.dtype)
