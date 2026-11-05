"""The decode step on the device: Model.decode + next-token choice + position advance, captured in a CUDA graph.

One step reads the static token and position buffers, runs `model.decode`, adds 1 to the position and writes
the chosen next token back into the token buffer. Nothing in it goes back to the host, so steps can be queued
back to back (see `pipelined`) and the host only copies each token out to send it.

- On CUDA, each mode (greedy, sampled) is captured once into its own graph (sharing one memory pool) and then
  replayed; capture happens between requests (`DecodeLoop.capture`), never in the middle of one, because the
  warm-up steps before a capture write to the KV cache.
- On a CPU, or when capture fails, the same step runs eagerly; `graph_error` keeps the reason.
- The sampled mode draws from noise refilled before each replay, outside the graph (see sampler.py), so the
  graph holds no random-number state.

To make decoding fast, change the model's decode (kernels, fusions, a pruned output layer for plain decoding)
and keep this loop: the server, the logprob path and the proofs stay valid.
"""
from __future__ import annotations

from typing import Any, Callable, Iterator

from .sampler import DeviceParams, greedy, is_greedy


class DecodeLoop:
    def __init__(self, model, state, device, use_graphs: bool | None = None, warmup_steps: int = 2):
        import torch
        self.model, self.state = model, state
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.use_graphs = self.cuda if use_graphs is None else (bool(use_graphs) and self.cuda)
        self.warmup_steps = warmup_steps
        self.tok = torch.zeros(1, dtype=torch.long, device=self.device)      # input of the next step
        self.pos = torch.zeros(1, dtype=torch.int32, device=self.device)     # its position
        self.params = DeviceParams.alloc(model.vocab_size, self.device)
        self.greedy_mode = True
        self.generator = None
        self.graph_error: str | None = None
        self._graphs: dict[bool, tuple[Any, Any]] = {}
        self._pool = None

    # --- one step, as captured ----------------------------------------------------------------------
    def _choose(self, logits, greedy_mode: bool) -> None:
        nxt = greedy(logits.reshape(1, -1)) if greedy_mode else self.params.choose(logits)
        self.tok.copy_(nxt)

    def _step(self, greedy_mode: bool):
        logits = self.model.decode(self.tok, self.pos, self.state)
        self.pos.add_(1)
        self._choose(logits, greedy_mode)
        return logits

    @property
    def graphs_captured(self) -> list[str]:
        return [("greedy" if g else "sampled") for g in sorted(self._graphs, reverse=True)]

    def capture(self, greedy_mode: bool) -> bool:
        """Capture the graph for one mode, between requests. False (eager from now on) if capture fails."""
        if not self.use_graphs:
            return False
        if greedy_mode in self._graphs:
            return True
        import torch
        try:
            self.pos.zero_()
            self.tok.zero_()
            side = torch.cuda.Stream(self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):                       # warm-up: lazy inits and allocator state
                for _ in range(self.warmup_steps):
                    self._step(greedy_mode)
            torch.cuda.current_stream(self.device).wait_stream(side)
            torch.cuda.synchronize(self.device)
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._pool):
                out = self._step(greedy_mode)
            torch.cuda.synchronize(self.device)
            self._graphs[greedy_mode] = (g, out)
            return True
        except Exception as e:  # noqa: BLE001  (any capture failure falls back to eager, with the reason kept)
            self.graph_error = f"{type(e).__name__}: {e}"[:500]
            self.use_graphs = False
            self._graphs.clear()
            return False

    # --- a request ------------------------------------------------------------------------------------
    def start(self, logits, prompt_len: int, sampling, generator) -> None:
        """Choose the first token from the prefill logits; the next step decodes it at position prompt_len."""
        self.greedy_mode = is_greedy(sampling.temperature)
        self.generator = generator
        if not self.greedy_mode:
            self.params.set(sampling)
            self.params.refill_noise(generator)
        self.pos.fill_(int(prompt_len))
        self._choose(logits, self.greedy_mode)

    def step(self):
        """Decode the current token and choose the next one (into `tok`). Returns the logits it was chosen
        from (the graph's static output buffer when replaying: read it before the next step)."""
        if not self.greedy_mode:
            self.params.refill_noise(self.generator)
        entry = self._graphs.get(self.greedy_mode) if self.use_graphs else None
        if entry is None:
            return self._step(self.greedy_mode)
        graph, out = entry
        graph.replay()
        return out


def pipelined(n: int, lookahead: int, launch: Callable[[int], None], collect: Callable[[int], Any]) -> Iterator[Any]:
    """Yield collect(0..n-1) in order while up to `lookahead` further steps are already queued.

    launch(i) enqueues step i without waiting; collect(i) waits for step i's result on the host. For i >= 1,
    step i + lookahead is launched before step i is yielded, so the GPU keeps working while the server handles
    a token. Step 0 (prefill's token) is launched alone and yielded as soon as it is ready: time to first
    token and the engine's prefill_ms hold no decode launches, and nothing is queued when token 0 arrives
    (a decode timer started then measures decode only). Closing the generator stops further launches (the
    caller drains what is in flight).
    """
    launched = 0
    for i in range(n):
        while launched < (min(n, i + 1 + max(0, lookahead)) if i else 1):
            launch(launched)
            launched += 1
        yield collect(i)
