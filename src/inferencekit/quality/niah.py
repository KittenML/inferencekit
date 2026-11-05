"""Needle in a haystack: can the engine still retrieve one fact from a long context?

Long-context paths (chunked prefill, paged/sliding attention, quantised KV cache, RoPE
scaling) can break retrieval while short-prompt quality looks fine. This builds, for a
target context length and a depth (0 = start, 1 = end), a haystack of synthetic filler
sentences with one "needle" sentence holding a random 7-digit code, then asks for the
code. Correct = the code's digits appear in the reply.

Everything is generated from a seed (no third-party text, nothing to license): the
filler is grammatical nonsense about rooms, boxes and colours, so it never contains a
competing code. Lengths are hit approximately: characters per token are measured once
with the server's ``/tokenize`` when it has one (else 4.0 is assumed), and the actual
prompt length reported by the server (``usage.prompt_tokens``) is recorded per item.
"""

from __future__ import annotations

import random
import re

_ADJ = ("quiet", "narrow", "amber", "distant", "hollow", "crowded", "frozen", "silver", "patient", "brittle")
_NOUN = ("archive", "harbour", "orchard", "workshop", "station", "library", "market", "garden", "tower", "bridge")
_PLACE = ("the northern valley", "the old quarter", "the river delta", "the high plateau", "the east coast",
          "the lower town", "the island chain", "the western ridge")
_VERB = ("stores", "counts", "repairs", "ships", "sorts", "paints", "weighs", "labels")
_THING = ("wooden crates", "paper maps", "glass jars", "copper wires", "woollen blankets", "clay tiles",
          "brass keys", "linen sacks")
_COLOR = ("red", "green", "blue", "violet", "orange", "teal", "golden", "grey")
_ANIMAL = ("falcon", "otter", "lynx", "heron", "badger", "salmon", "gecko", "raven")

QUESTION = "What is the access code for the {color} {animal} vault? Reply with the number only."


def filler_sentence(rng: random.Random) -> str:
    # Small numbers in words only, so the filler never contains a digit string like the code.
    return (f"Every morning the {rng.choice(_ADJ)} {rng.choice(_NOUN)} of {rng.choice(_PLACE)} "
            f"{rng.choice(_VERB)} {rng.choice(('two', 'five', 'nine', 'twelve', 'forty'))} "
            f"{rng.choice(_THING)}.")


def build(target_tokens: int, depth: float, seed: int, chars_per_token: float = 4.0) -> dict:
    """{messages, code, depth, target_tokens, needle} for one haystack item."""
    rng = random.Random(f"niah:{seed}:{target_tokens}:{depth}")
    color, animal = rng.choice(_COLOR), rng.choice(_ANIMAL)
    code = str(rng.randrange(1_000_000, 10_000_000))
    needle = f"The access code for the {color} {animal} vault is {code}."
    question = QUESTION.format(color=color, animal=animal)
    budget = max(0, int(target_tokens * chars_per_token) - len(needle) - len(question) - 200)
    sentences, size = [], 0
    while size < budget:
        s = filler_sentence(rng)
        sentences.append(s)
        size += len(s) + 1
    pos = round(min(max(depth, 0.0), 1.0) * len(sentences))
    sentences.insert(pos, needle)
    context = " ".join(sentences)
    content = (f"Read the following document.\n\n<document>\n{context}\n</document>\n\n{question}")
    return {"messages": [{"role": "user", "content": content}], "code": code, "depth": depth,
            "target_tokens": target_tokens, "needle_sentence_index": pos, "sentences": len(sentences)}


def found(reply: str | None, code: str) -> bool:
    return bool(reply) and code in re.sub(r"[\s,._'-]", "", reply)
