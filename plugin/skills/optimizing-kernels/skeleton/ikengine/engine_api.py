"""The contract between the server, the engine and a model module.

    serve.py (HTTP, tokenizer, queue, stop rules)  ->  Engine.stream(...)  ->  Model.prefill / Model.decode

- The server owns: the OpenAI API, chat templates, tokenization, the request queue, stop conditions
  (EOS, stop ids, stop strings, ignore_eos, max_tokens) and the response format. It closes the
  stream generator early when it stops; an engine must handle GeneratorExit cleanly.
- The engine owns: prefill, the CUDA-graph decode loop, sampling and logprobs.
- A model module owns: weights and the forward pass. It is correct first; making it fast is your job.

Numerics rule: a model computes what the checkpoint declares ("as shipped"): stored weights and scales
used as stored, the declared activation scheme, the declared KV-cache dtype. A model that changes any of
these says so in `numerics`, which the engine writes to ENGINE.json.

Logprobs are the log-softmax of the raw logits over the full vocabulary (before temperature, top-k or
top-p), as vLLM returns by default. A logprob request always takes the full output-layer path, even when
the engine has a faster pruned path for plain decoding.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Protocol, Sequence, runtime_checkable

MAX_TOP_LOGPROBS = 20


@dataclass(frozen=True)
class SamplingParams:
    """Per-request sampling. Fields the request leaves out come from the model's generation_config.json.

    temperature 0 means greedy (argmax; ties go to the lowest token id). top_k 0 or -1 means off.
    top_p 1.0 means off. min_p 0.0 means off. seed None means not reproducible.
    Order of application (as vLLM and Hugging Face): temperature, then top-k, then top-p, then min-p.
    """
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0

    @classmethod
    def from_generation_config(cls, gen: Mapping[str, Any] | None) -> "SamplingParams":
        """Defaults from generation_config.json. do_sample false (or absent with no temperature) is greedy."""
        gen = dict(gen or {})
        if gen.get("do_sample") is False:
            return cls(temperature=0.0)
        return cls(
            temperature=float(gen.get("temperature", 1.0)),
            top_k=int(gen.get("top_k", 0) or 0),
            top_p=float(gen.get("top_p", 1.0) if gen.get("top_p") is not None else 1.0),
            min_p=float(gen.get("min_p", 0.0) or 0.0),
        )

    def merged(self, body: Mapping[str, Any]) -> "SamplingParams":
        """Request fields override these defaults; absent (or null) fields keep them."""
        def pick(key, cast, default):
            v = body.get(key)
            return default if v is None else cast(v)
        return SamplingParams(
            temperature=pick("temperature", float, self.temperature),
            top_k=pick("top_k", int, self.top_k),
            top_p=pick("top_p", float, self.top_p),
            min_p=pick("min_p", float, self.min_p),
            seed=pick("seed", int, None),
        )


@dataclass
class StepOut:
    """One generated token."""
    token_id: int
    logprob: float | None = None                                      # chosen token, full-vocab log-softmax
    top: list[tuple[int, float]] = field(default_factory=list)        # up to k alternatives, descending


@runtime_checkable
class Model(Protocol):
    """What a model module provides (ikengine/models/*.py, or ikengine/ref_dense.py).

    Tensors are torch tensors on the model's device. `decode` must be CUDA-graph capturable: static
    shapes, no host synchronisation, no Python control flow that depends on tensor values, all outputs
    written to buffers that `alloc_state` created.
    """
    vocab_size: int
    max_seq: int
    eos_token_ids: frozenset[int]
    numerics: dict          # e.g. {"weights": "nvfp4 as stored", "activations": "nvfp4 (input_scale)", "kv": "fp8 e4m3", "state": "fp32"}

    def alloc_state(self, max_seq: int) -> Any:
        """Allocate KV cache, recurrent state and static input/output buffers for one sequence."""

    def reset(self, state: Any) -> None:
        """Make `state` empty for a new request (no reallocation)."""

    def prefill(self, input_ids: Any, state: Any) -> Any:
        """Run the prompt (int64 [T] on device) from position 0; fill the caches; return the fp32 logits
        of the last position, shape [vocab_size]."""

    def decode(self, token: Any, pos: Any, state: Any) -> Any:
        """One step. token: int64 [1] device tensor; pos: int32 [1] device tensor (the position of `token`).
        Returns fp32 logits [vocab_size] in a static buffer owned by `state`."""


@runtime_checkable
class Engine(Protocol):
    """What the server calls."""
    vocab_size: int
    max_seq: int
    logprobs_supported: bool

    def stream(self, prompt_ids: Sequence[int], max_tokens: int, sampling: SamplingParams,
               top_logprobs: int | None, info: dict) -> Iterator[StepOut]:
        """Generate up to max_tokens tokens. top_logprobs None: no logprobs (fast path). 0: the chosen
        token's logprob only. k (1..MAX_TOP_LOGPROBS): also the k most likely tokens. The engine may write
        timing facts into `info` (e.g. "prefill_ms"). The server stops early by closing the generator."""


def describe_numerics(model: Model) -> dict:
    """The numerics block written to ENGINE.json next to the launch command."""
    return dict(getattr(model, "numerics", {}) or {})
