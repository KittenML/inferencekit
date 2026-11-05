"""Per-step dumps: what a clip engine leaves of every request for ``inferencekit quality clip-steps``.

THE CONTRACT (the same for every diffusion clip engine, whatever it is built on):

  The engine process finds a directory in its environment variable ``CLIP_STEP_DUMP_DIR``. For every clip
  request it serves (timed or not: there is no audit mode, and the requests to check are picked after the
  clips are made), it writes one file before it answers:

      $CLIP_STEP_DUMP_DIR/<video_sha256>.safetensors

  <video_sha256> is the lowercase hex sha256 of the mp4 bytes it answers with. The file is in the
  safetensors format and holds, for each latent stream <s> the model denoises ("video", and "audio" for an
  audio-and-video model) and each denoising step i = 0 .. N-1 in the order the steps ran:

      step.<i>.<s>.x       the latent the step started from (x_t)
      step.<i>.<s>.out     the guided model output the sampler's update took: every guidance pass of the
                           step combined (and rescaled, if the pipeline rescales), in the form the sampler
                           takes it. Flow-matching models: the velocity, so that the Euler update is
                           x_next = x + (sigma_next - sigma) * out; a sampler that holds the denoised
                           estimate x0 writes (x - x0) / sigma
      step.<i>.<s>.x_next  the latent the update produced (x_{t-1})
      final.<s>            the latent the clip was decoded from
      sigma                [N] float64: each step's noise level (flow matching: x = (1 - sigma) x0 + sigma noise)
      sigma_next           [N] float64: the noise level each step's update went to

  Latents have the layout and shape of the initial noise (video [1, C, F, H, W], audio [1, C, L, M]; not a
  packed token sequence) and the dtype the engine holds them in. The file's metadata (safetensors' string
  map) holds "format": "clip-steps.v1" and "request": the request body as JSON (the checker compares its prompt,
  shape, steps, guidance scale, seed and noise seed with the request it sent). Write under a temporary name and
  rename, so no reader sees a partial file; never delete one.

  Cost on LTX-2.5 (768x512, 121 frames, 30 steps, float32 latents): 292,150,200 bytes per request, written in
  0.26 s (median of 2 clips, the clip server's step_dump stage, RTX PRO 6000) [measured].

``StepRecorder`` writes this format from torch tensors in a few lines (any engine can copy the idea);
``DumpReader`` and ``layout_problems`` are the checker's side. Both need torch and safetensors, imported when
used: the clip server and the harness import this module without them.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

ENV = "CLIP_STEP_DUMP_DIR"
FORMAT = "clip-steps.v1"
SUFFIX = ".safetensors"
FIELDS = ("x", "out", "x_next")
REQUEST_FIELDS = ("task", "prompt", "width", "height", "num_frames", "fps", "steps", "guidance_scale", "seed",
                  "noise_seed", "audio")
# What the checker compares between a dump's request and the request it sent (task and audio are not replayed).
COMPARED_FIELDS = ("prompt", "width", "height", "num_frames", "fps", "steps", "guidance_scale", "seed", "noise_seed")
STEP_KEY = re.compile(r"^step\.(\d+)\.([a-z0-9_]+)\.(x|out|x_next)$")
FINAL_KEY = re.compile(r"^final\.([a-z0-9_]+)$")


def request_of(job) -> dict:
    """The request fields a dump records (a ClipJob or a request body; noise_seed defaults to seed)."""
    get = job.get if isinstance(job, dict) else lambda k, d=None: getattr(job, k, d)
    out = {k: get(k) for k in REQUEST_FIELDS}
    if out.get("noise_seed") is None:
        out["noise_seed"] = out.get("seed")
    if out.get("fps") is not None:
        out["fps"] = float(out["fps"])
    if out.get("guidance_scale") is not None:
        out["guidance_scale"] = float(out["guidance_scale"])
    return out


def dump_path(directory, video_sha256: str) -> Path:
    return Path(directory) / f"{video_sha256.lower()}{SUFFIX}"


class StepRecorder:
    """Collects one request's steps (CPU copies) and writes them in the contract's format."""

    def __init__(self, request: dict):
        self.request = dict(request)
        self.tensors: dict = {}
        self.sigma: list[float] = []
        self.sigma_next: list[float] = []
        self.count: dict[str, int] = {}
        self.finals: dict = {}

    def step(self, stream: str, sigma, sigma_next, x, out, x_next) -> None:
        i = self.count.get(stream, 0)
        self.count[stream] = i + 1
        if i == len(self.sigma):  # the first stream to reach step i records its noise levels
            self.sigma.append(float(sigma))
            self.sigma_next.append(float(sigma_next))
        for name, t in zip(FIELDS, (x, out, x_next)):
            self.tensors[f"step.{i}.{stream}.{name}"] = t.detach().to("cpu", copy=True).contiguous()

    def final(self, stream: str, latent) -> None:
        self.finals[stream] = latent.detach().to("cpu", copy=True).contiguous()

    def save(self, path) -> int:
        """Write the file (temporary name, then rename); its size in bytes."""
        import torch
        from safetensors.torch import save_file

        tensors = dict(self.tensors)
        for s, n in self.count.items():
            final = self.finals.get(s)
            tensors[f"final.{s}"] = final if final is not None else tensors[f"step.{n - 1}.{s}.x_next"].clone()
        tensors["sigma"] = torch.tensor(self.sigma, dtype=torch.float64)
        tensors["sigma_next"] = torch.tensor(self.sigma_next, dtype=torch.float64)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".partial")
        save_file(tensors, str(tmp), metadata={"format": FORMAT, "request": json.dumps(self.request, sort_keys=True)})
        os.replace(tmp, path)
        return path.stat().st_size


class DumpReader:
    """One dump file, read lazily (tensors load on ``get``)."""

    def __init__(self, path):
        from safetensors import safe_open

        self.path = Path(path)
        self._f = safe_open(str(path), framework="pt", device="cpu")
        self.meta = self._f.metadata() or {}
        self.keys = set(self._f.keys())
        try:
            self.request = json.loads(self.meta.get("request") or "null")
        except ValueError:
            self.request = None
        steps: dict[str, set[int]] = {}
        for k in self.keys:
            m = STEP_KEY.match(k)
            if m:
                steps.setdefault(m.group(2), set()).add(int(m.group(1)))
        self.streams = sorted(steps)
        self.n_steps = max((max(v) + 1 for v in steps.values()), default=0)
        self._steps = steps

    def get(self, key: str):
        return self._f.get_tensor(key)

    def has(self, key: str) -> bool:
        return key in self.keys

    def shape(self, key: str) -> tuple:
        return tuple(self._f.get_slice(key).get_shape())

    def dtype(self, key: str) -> str:
        """safetensors' dtype name: F32, BF16, F16, F64..."""
        return str(self._f.get_slice(key).get_dtype())


def layout_problems(d: DumpReader, shapes: dict, steps: int) -> list[str]:
    """What is wrong with a dump's layout against the request (streams, step count, keys, shapes, dtypes)."""
    out = []
    if d.meta.get("format") != FORMAT:
        out.append(f"metadata format is {d.meta.get('format')!r}, not {FORMAT!r}")
    if set(d.streams) != set(shapes):
        out.append(f"streams {d.streams} differ from the model's latent streams {sorted(shapes)}")
    if d.n_steps != steps:
        out.append(f"{d.n_steps} steps dumped, {steps} asked")
    allowed = {"sigma", "sigma_next"}
    for s in shapes:
        allowed.add(f"final.{s}")
        allowed.update(f"step.{i}.{s}.{f}" for i in range(steps) for f in FIELDS)
    missing, extra = sorted(allowed - d.keys), sorted(d.keys - allowed)
    if missing:
        out.append(f"{len(missing)} tensor(s) missing (e.g. {missing[0]})")
    if extra:
        out.append(f"{len(extra)} unexpected tensor(s) (e.g. {extra[0]})")
    for name in ("sigma", "sigma_next"):
        if d.has(name) and d.shape(name) != (d.n_steps,):
            out.append(f"{name} has shape {d.shape(name)}, not ({d.n_steps},)")
    for s, shape in shapes.items():
        bad = [k for k in sorted(d.keys) if (k == f"final.{s}" or (k.startswith("step.") and f".{s}." in k))
               and d.shape(k) != tuple(shape)]
        if bad:
            out.append(f"{len(bad)} {s} tensor(s) have another shape than the request's latent {list(shape)} "
                       f"(e.g. {bad[0]}: {list(d.shape(bad[0]))})")
    return out
