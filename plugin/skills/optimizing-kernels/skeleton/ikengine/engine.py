"""The generic engine: any engine_api.Model, one request at a time (batch 1), streamed token by token.

    prefill (eager)  ->  first token  ->  DecodeLoop.step() per token (CUDA graph on a GPU)  ->  StepOut

- Sampling and greedy run on the device (sampler.py); a request with a seed is reproducible.
- Logprobs are computed only when the request asks for them (top_logprobs not None): the full-vocab log-softmax
  of the raw logits, the chosen token's logprob and the top k (k <= 20). A request without logprobs pays nothing.
- Up to `lookahead` steps are queued ahead of the host (2 on a GPU), so the GPU does not wait while the server
  detokenizes. The first token (prefill's) is sent before any decode step is queued, so time to first token and
  `prefill_ms` measure prefill alone. When the server stops early it closes the generator; the steps in flight
  finish and are dropped before the next request starts (GeneratorExit-safe).

`build_engine(model_dir, device, max_seq, act, kv)` loads the model module that config.json names
(ikengine.models.load_model), captures the decode graphs and runs one short warm-up request.
"""
from __future__ import annotations

import threading
import time
from typing import Iterator, Sequence

from .engine_api import MAX_TOP_LOGPROBS, SamplingParams, StepOut, describe_numerics
from .graph_loop import DecodeLoop, pipelined
from .sampler import is_greedy, logprobs, make_generator

ENGINE_NAME = "ikengine"


class Engine:
    """engine_api.Engine over a Model. Not re-entrant: the server runs one request at a time."""

    logprobs_supported = True

    def __init__(self, model, device: str = "cuda", max_seq: int | None = None, graphs: bool | None = None,
                 lookahead: int | None = None):
        import torch
        self.model = model
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.vocab_size = int(model.vocab_size)
        self.max_seq = int(min(max_seq or model.max_seq, model.max_seq))
        self.eos_token_ids = frozenset(int(t) for t in (getattr(model, "eos_token_ids", None) or ()))
        self.numerics = describe_numerics(model)
        with torch.no_grad():
            self.state = model.alloc_state(self.max_seq)
        self.loop = DecodeLoop(model, self.state, self.device, use_graphs=graphs)
        self.lookahead = max(0, int(lookahead if lookahead is not None else (2 if self.cuda else 0)))
        ring = self.lookahead + 2                    # > lookahead: a slot is reused only after it was collected
        self._tok_h = torch.zeros(ring, dtype=torch.long, pin_memory=self.cuda)
        self._lp_h = torch.zeros(ring, 1 + MAX_TOP_LOGPROBS, dtype=torch.float32, pin_memory=self.cuda)
        self._id_h = torch.zeros(ring, MAX_TOP_LOGPROBS, dtype=torch.long, pin_memory=self.cuda)
        self._events = [torch.cuda.Event() for _ in range(ring)] if self.cuda else None
        self._busy = threading.Lock()
        self._captured = False

    # --- startup ----------------------------------------------------------------------------------------
    def capture(self) -> None:
        """Capture the greedy and the sampled decode graphs (a no-op on a CPU or once done)."""
        import torch
        if self._captured:
            return
        with torch.no_grad():
            for mode in (True, False):
                self.loop.capture(mode)
        self._captured = True

    def warmup(self, prompt_len: int = 8, tokens: int = 4) -> None:
        """Capture the graphs and run one greedy and one sampled request, so the first real request is warm."""
        self.capture()
        ids = [i % self.vocab_size for i in range(1, prompt_len + 1)]
        for sp in (SamplingParams(temperature=0.0), SamplingParams(temperature=1.0, top_k=20, top_p=0.95, seed=0)):
            for _ in self.stream(ids, tokens, sp, MAX_TOP_LOGPROBS, {}):
                pass

    def config(self) -> dict:
        """Facts for ENGINE.json and the server's resolved-config line."""
        return {
            "engine": ENGINE_NAME, "model_class": type(self.model).__name__, "device": str(self.device),
            "max_seq": self.max_seq, "vocab_size": self.vocab_size, "batch": 1,
            "cuda_graphs": bool(self.loop.graphs_captured), "graphs": self.loop.graphs_captured,
            "graph_error": self.loop.graph_error, "lookahead": self.lookahead,
            "sampling": "temperature, top-k, top-p, min-p on the device; seeded per request; greedy ties to the lowest id",
            "logprobs": f"full-vocab log-softmax of the raw logits, chosen token plus up to {MAX_TOP_LOGPROBS} alternatives",
            "numerics": self.numerics,
        }

    # --- a request ----------------------------------------------------------------------------------------
    def stream(self, prompt_ids: Sequence[int], max_tokens: int, sampling: SamplingParams,
               top_logprobs: int | None, info: dict) -> Iterator[StepOut]:
        import torch
        ids = [int(t) for t in prompt_ids]
        n = len(ids)
        if n == 0:
            raise ValueError("empty prompt")
        if n >= self.max_seq:
            raise ValueError(f"the prompt has {n} tokens; this engine holds at most {self.max_seq} (--max-seq)")
        steps = max(0, min(int(max_tokens), self.max_seq - n))
        if steps < int(max_tokens):
            info["max_seq_capped"] = steps
        k = None if top_logprobs is None else max(0, min(int(top_logprobs), MAX_TOP_LOGPROBS))
        if steps == 0:
            return
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("Engine.stream is already running: the engine serves one request at a time")
        try:
            self.capture()
            gen = None if is_greedy(sampling.temperature) else make_generator(self.device, sampling.seed)
            info["decode"] = "cuda_graph" if self.loop.graphs_captured else "eager"
            ring = len(self._tok_h)
            t0 = time.perf_counter()

            def launch(i: int) -> None:
                with torch.no_grad():
                    if i == 0:
                        self.model.reset(self.state)
                        inp = torch.tensor(ids, dtype=torch.long, device=self.device)
                        logits = self.model.prefill(inp, self.state)
                        self.loop.start(logits, n, sampling, gen)
                    else:
                        logits = self.loop.step()
                    slot = i % ring
                    self._tok_h[slot:slot + 1].copy_(self.loop.tok, non_blocking=self.cuda)
                    if k is not None:                          # only logprob requests pay for the log-softmax
                        chosen, top_ids, top_lps = logprobs(logits, self.loop.tok, k)
                        self._lp_h[slot, :1].copy_(chosen, non_blocking=self.cuda)
                        if k:
                            self._lp_h[slot, 1:1 + k].copy_(top_lps, non_blocking=self.cuda)
                            self._id_h[slot, :k].copy_(top_ids, non_blocking=self.cuda)
                    if self._events is not None:
                        self._events[slot].record()

            def collect(i: int) -> StepOut:
                slot = i % ring
                if self._events is not None:
                    self._events[slot].synchronize()
                if i == 0:
                    info["prefill_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
                tok = int(self._tok_h[slot])
                if k is None:
                    return StepOut(tok)
                top = [(int(self._id_h[slot, j]), float(self._lp_h[slot, 1 + j])) for j in range(k)]
                return StepOut(tok, float(self._lp_h[slot, 0]), top)

            yield from pipelined(steps, self.lookahead, launch, collect)
        finally:
            if self.cuda:
                torch.cuda.current_stream(self.device).synchronize()   # steps queued past a stop finish here
            self._busy.release()


def build_engine(model_dir: str, device: str = "cuda", max_seq: int = 4096, act: str = "as_shipped",
                 kv: str = "as_shipped", graphs: bool | None = None, lookahead: int | None = None,
                 warmup: bool = True) -> Engine:
    """Load the model module for config.json's architecture and wrap it in an Engine (graphs captured)."""
    from .models import load_model
    model = load_model(model_dir, device=device, max_seq=max_seq, act=act, kv=kv)
    eng = Engine(model, device=device, max_seq=max_seq, graphs=graphs, lookahead=lookahead)
    if warmup:
        eng.warmup()
    return eng
