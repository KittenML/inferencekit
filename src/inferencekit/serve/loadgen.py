"""Open-loop load: seeded Poisson arrivals, the same schedule for every engine.

Open loop means a request is sent at its scheduled arrival time whether or not earlier requests have
finished, like real users. A closed loop (N workers, each sending its next request when the previous
one ends) slows its own arrival rate as the server slows down, which hides overload; it is offered
only as a labelled stress test.

Levels:
  * a rate grid (req/s): arrivals at exactly that Poisson rate;
  * a concurrency grid (N streams): Poisson arrivals at rate N / e2e_ref, where e2e_ref is the
    unloaded end-to-end latency (the median of dedicated reference requests sent one at a time after
    the warm-up, or given with --ref-e2e-ms so every engine gets the identical schedule). By Little's
    law that offers about N requests in flight on average when latency stays at e2e_ref. It is an
    offered load, not a guarantee: a short window, Poisson noise in the arrival count, or an e2e_ref
    that is too high (slow reference requests) all leave fewer than N in flight. So every level
    reports its measured mean_inflight, and the capacity record flags an N* whose measured in-flight
    count is below 0.9 x N*.

Arrival times use ``random.Random(seed).random()`` only (reproducible across Python versions), with
one seed per level, so engine A and engine B see the same arrival offsets and prompt order.

Prompts are unique inside a run: each request takes the next prompt of a seeded permutation of the
pool; when the pool runs out, a reused prompt gets a short unique prefix ("[ik-<level>-<n>] ") so no
two requests send identical text and none can be served from another's prefix cache entry.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from typing import Awaitable, Callable

from inferencekit.serve.client import RequestResult
from inferencekit.serve.prompts_io import Prompt

Send = Callable[[Prompt, "str | None", "int | None"], Awaitable[RequestResult]]


def level_seed(seed: int, index: int) -> int:
    return seed * 1_000_003 + index


def poisson_schedule(rate_rps: float, duration_s: float, seed: int, max_requests: int | None = None) -> list[float]:
    """Arrival offsets in seconds (ascending) of a Poisson process with this rate, within duration_s."""
    if rate_rps <= 0 or duration_s <= 0:
        return []
    rng = random.Random(seed)
    t, out = 0.0, []
    while True:
        t += -math.log(1.0 - rng.random()) / rate_rps
        if t >= duration_s or (max_requests and len(out) >= max_requests):
            return out
        out.append(t)


class PromptPool:
    """Hands out prompts so that no two requests of a run send the same text."""

    def __init__(self, prompts: list[Prompt]):
        self.prompts = list(prompts)
        self.used: set[str] = set()
        self.salted = 0

    def order(self, seed: int) -> list[Prompt]:
        idx = list(range(len(self.prompts)))
        rng = random.Random(seed)
        for i in range(len(idx) - 1, 0, -1):  # Fisher-Yates with rng.random() only
            j = min(int(rng.random() * (i + 1)), i)
            idx[i], idx[j] = idx[j], idx[i]
        return [self.prompts[i] for i in idx]

    def stream(self, seed: int, tag: str):
        """Endless (prompt, salt) pairs for one level: unused prompts first (seeded order), then reused
        ones, each with a unique salt."""
        ordered = self.order(seed)
        fresh = [p for p in ordered if p.id not in self.used]
        reused = [p for p in ordered if p.id in self.used]
        k = 0
        while True:
            if k < len(fresh):
                p, salt = fresh[k], None
            else:
                pool = reused + fresh
                p, salt = pool[(k - len(fresh)) % len(pool)], f"ik-{tag}-{k}"
                self.salted += 1
            self.used.add(p.id)
            yield p, salt
            k += 1


def salted_messages(prompt: Prompt, salt: str | None) -> list[dict]:
    msgs = [dict(m) for m in prompt.messages]
    if salt:
        for m in msgs:
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                m["content"] = f"[{salt}] " + m["content"]
                break
        else:
            msgs[-1]["content"] = f"[{salt}] " + str(msgs[-1].get("content", ""))
    return msgs


async def open_loop(send: Send, items: list[tuple[Prompt, str | None]], offsets_s: list[float]) -> list[RequestResult]:
    """Send items[i] at offsets_s[i] after the start, never waiting for earlier requests."""
    start = time.perf_counter_ns() + 5_000_000  # 5 ms head start for task creation
    tasks = []
    for (prompt, salt), off in zip(items, offsets_s):
        target = start + int(off * 1e9)
        delay = (target - time.perf_counter_ns()) / 1e9
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(send(prompt, salt, target)))
    return list(await asyncio.gather(*tasks))


async def closed_loop(send: Send, items, workers: int, duration_s: float,
                      max_requests: int | None = None) -> list[RequestResult]:
    """STRESS TEST ONLY: ``workers`` users each send their next request as soon as the last one ends.

    ``items`` is an iterator of (prompt, salt), e.g. ``PromptPool.stream``.
    """
    deadline = time.perf_counter_ns() + int(duration_s * 1e9)
    results: list[RequestResult] = []
    sent = 0

    async def worker():
        nonlocal sent
        while time.perf_counter_ns() < deadline and (not max_requests or sent < max_requests):
            prompt, salt = next(items)
            sent += 1
            results.append(await send(prompt, salt, None))

    await asyncio.gather(*(worker() for _ in range(max(1, workers))))
    return results
