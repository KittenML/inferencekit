"""Next-token choice and logprobs, on the device, with static shapes (CUDA-graph safe).

Order (as Hugging Face's logits warpers, and the contract in engine_api.SamplingParams): temperature, top-k,
top-p, min-p, then a draw. vLLM V1 applies min-p right after temperature, before top-k/top-p; the two orders
give the same kept set whenever min_p is 0 (the default of every request the harness sends).

- Greedy (temperature below 1e-5, as vLLM): argmax of the raw logits; ties go to the lowest token id.
- Top-k keeps every token whose logit is >= the k-th largest (ties at the boundary are kept, as HF and vLLM).
- Top-p sorts ascending, removes the tokens whose cumulative probability is <= 1 - top_p and always keeps the
  most likely one (as HF's TopPLogitsWarper and vLLM's apply_top_k_top_p).
- Min-p removes tokens whose probability is below min_p x the largest probability.
- The draw is argmax(p / q) with q ~ Exponential(1) i.i.d. per token (an exact categorical sample, as vLLM's
  random_sample). The noise `q` comes from a per-request torch.Generator seeded with the request's seed, and is
  drawn OUTSIDE any CUDA graph into a static buffer, so the same seed gives the same tokens with or without
  graphs, and the graph needs no Philox offset bookkeeping.

Every parameter is a device tensor, so `filter_logits` and `draw` contain no host synchronisation and no
value-dependent Python control flow: they can live inside the decode graph (graph_loop.py puts them there).
A faster sampler (top-k first, then top-p on the k survivors) is a valid optimisation only if it keeps the
boundary ties and the same noise per token id; prove it against this one with fixed noise.

Logprobs are the log-softmax of the raw fp32 logits over the full vocabulary, before temperature or any filter
(vLLM's default "raw_logprobs").
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

SAMPLING_EPS = 1e-5     # temperature below this is greedy (vLLM's _SAMPLING_EPS)


def is_greedy(temperature: float) -> bool:
    return temperature < SAMPLING_EPS


def greedy(logits):
    """argmax over the last dim; the first (lowest-id) maximum wins (torch.argmax's documented tie rule)."""
    return logits.argmax(dim=-1)


def filter_logits(logits, temperature, top_k, top_p, min_p):
    """logits [B, V] fp32; parameters are device tensors [B] (top_k int64, the rest fp32).

    Returns the processed logits [B, V] fp32 in vocabulary order, -inf where a token was removed.
    top_k <= 0 or >= V is off; top_p >= 1 is off; min_p <= 0 is off.
    """
    import torch
    v = logits.shape[-1]
    x = logits.float() / temperature.float().unsqueeze(-1)
    srt, idx = x.sort(dim=-1)                                                  # ascending
    k = top_k.long()
    k = torch.where((k <= 0) | (k > v), torch.full_like(k, v), k)
    kth = srt.gather(-1, (v - k).unsqueeze(-1))                                # k-th largest value
    srt = srt.masked_fill(srt < kth, float("-inf"))
    cum = srt.softmax(dim=-1).cumsum(dim=-1)
    drop = cum <= (1.0 - top_p.float()).unsqueeze(-1)
    drop[..., -1] = False                                                      # the most likely token stays
    srt = srt.masked_fill(drop, float("-inf"))
    p = srt.softmax(dim=-1)
    srt = srt.masked_fill(p < min_p.float().unsqueeze(-1) * p[..., -1:], float("-inf"))
    return torch.full_like(x, float("-inf")).scatter(-1, idx, srt)


def draw(processed, noise):
    """One categorical sample per row: argmax(softmax(processed) / noise). noise [B, V] ~ Exponential(1)."""
    return (processed.softmax(dim=-1) / noise).argmax(dim=-1)


def logprobs(logits, token, k: int):
    """Full-vocab log-softmax of the raw logits [V]: (logprob of `token` [1], top-k ids [k], top-k logprobs [k]).

    k 0 returns empty top-k tensors. The top-k are in descending order (torch.topk, sorted).
    """
    import torch
    lp = torch.log_softmax(logits.float().reshape(-1), dim=-1)
    chosen = lp.gather(0, token.reshape(-1).long())
    if k <= 0:
        return chosen, lp.new_zeros(0, dtype=torch.long), lp.new_zeros(0)
    vals, ids = torch.topk(lp, k)
    return chosen, ids, vals


@dataclass
class DeviceParams:
    """The sampling parameters of the current request as static device tensors (one row)."""
    temperature: Any
    top_k: Any
    top_p: Any
    min_p: Any
    noise: Any

    @classmethod
    def alloc(cls, vocab_size: int, device) -> "DeviceParams":
        import torch
        f = dict(dtype=torch.float32, device=device)
        return cls(torch.ones(1, **f), torch.zeros(1, dtype=torch.long, device=device), torch.ones(1, **f),
                   torch.zeros(1, **f), torch.ones(1, vocab_size, **f))

    def set(self, sampling) -> None:
        """Load a request's SamplingParams (fill_ launches a kernel; no host synchronisation)."""
        self.temperature.fill_(max(float(sampling.temperature), SAMPLING_EPS))
        self.top_k.fill_(int(sampling.top_k or 0))
        self.top_p.fill_(float(sampling.top_p if sampling.top_p is not None else 1.0))
        self.min_p.fill_(float(sampling.min_p or 0.0))

    def refill_noise(self, generator) -> None:
        """New Exponential(1) noise for the next draw, from the request's generator (outside any graph)."""
        self.noise.exponential_(generator=generator)

    def choose(self, logits):
        """The sampled token id [1] for logits [V] (graph safe)."""
        processed = filter_logits(logits.reshape(1, -1), self.temperature, self.top_k, self.top_p, self.min_p)
        return draw(processed, self.noise)


def make_generator(device, seed: int | None):
    """A request's generator: seeded when the request gives a seed (reproducible), else from os.urandom."""
    import torch
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) if seed is not None else int.from_bytes(os.urandom(8), "little") & ((1 << 63) - 1))
    return g
