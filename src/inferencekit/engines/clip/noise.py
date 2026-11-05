"""The initial noise a clip starts from: drawn on the CPU from one seed, the same for every engine.

The recipe (an engine can follow it without inferencekit)::

    g = torch.Generator(device="cpu").manual_seed(noise_seed)
    for name, shape in shapes:          # in order: "video" first, then "audio" when the model makes audio
        noise[name] = torch.randn(shape, generator=g, dtype=torch.float32)

The numbers are drawn on the CPU and only then moved to the GPU. CUDA's generator gives other numbers for
the same seed, so noise drawn on the GPU would differ between engines (and is not promised to stay the
same across GPU models or library versions). ``noise_seed`` is the request's ``noise_seed``, or its
``seed`` when the request has none.

For LTX-2.x the shapes are the unpacked latents (diffusers_ltx.noise_shapes): video
[1, C, (frames - 1) // 8 + 1, height // 32, width // 32] and audio [1, Ca, L, M]. These are also the first
two draws a Diffusers LTX-2 pipeline makes when it is given a CPU generator seeded with ``noise_seed``.

``digest`` hashes names, shapes and the float32 bytes in order: equal digests mean identical noise. The
server reports it as ``effective.noise_sha256``, so two engines' records show whether they started from
the same tensors.
"""

from __future__ import annotations

import hashlib

RECIPE = "cpu-randn-float32/1"  # bump if the recipe above ever changes


def initial_noise(noise_seed: int, shapes: dict) -> dict:
    """{name: CPU float32 tensor} drawn in the order of ``shapes`` from one CPU generator."""
    import torch

    g = torch.Generator(device="cpu").manual_seed(int(noise_seed))
    return {name: torch.randn(tuple(int(d) for d in shape), generator=g, dtype=torch.float32)
            for name, shape in shapes.items()}


def digest(noise: dict) -> str:
    """sha256 over (name, shape, float32 bytes) for each tensor, in order."""
    h = hashlib.sha256(RECIPE.encode())
    for name, t in noise.items():
        t = t.detach().to("cpu").contiguous()
        h.update(f"|{name}:{'x'.join(str(d) for d in t.shape)}|".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()
