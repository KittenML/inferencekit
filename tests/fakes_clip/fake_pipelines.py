"""Fake pipelines for the clip server: one with no torch at all, and a torch stand-in for Diffusers' LTX2Pipeline.

FakeClipPipeline implements the server's ClipPipeline interface directly. Knobs: ``skip_steps`` (runs
fewer steps than asked), ``fail`` (raise this message from generate), ``gate`` (a threading.Event that
generate waits on), ``step_s``. ``make_pipeline`` is the factory for ``--pipeline
fakes_clip.fake_pipelines:make_pipeline`` (its --pipeline-arg values become knobs).

``fake_ltx2(...)`` returns an object shaped like a Diffusers LTX-2 pipeline (the attributes and methods
diffusers_ltx uses), drawing its noise like the real one: video then audio, float32, from the generator.
"""

from __future__ import annotations

import hashlib
import time
from types import SimpleNamespace

from inferencekit.engines.clip.server import BadRequest, Generated


def fake_noise(noise_seed: int, shapes: dict) -> tuple[dict, str]:
    """Stands in for noise.initial_noise without torch: the 'tensors' are just (seed, shape)."""
    tensors = {k: (noise_seed, tuple(v)) for k, v in shapes.items()}
    return tensors, hashlib.sha256(repr(sorted(tensors.items())).encode()).hexdigest()


class FakeClipPipeline:
    def __init__(self, skip_steps: int = 0, fail: str | None = None, gate=None, step_s: float = 0.001):
        self.skip_steps, self.fail, self.gate, self.step_s = skip_steps, fail, gate, step_s
        self.calls: list = []  # (job, noise) per generate

    def describe(self) -> dict:
        return {"pipeline": "fake", "dtype": "bf16", "attention": "fake-sdpa"}

    def make_noise(self, noise_seed: int, shapes: dict):
        return fake_noise(noise_seed, shapes)

    def noise_shapes(self, job) -> dict:
        if job.width % 32 or job.height % 32:
            raise BadRequest("width and height must be multiples of 32")
        return {"video": (1, 4, (job.num_frames - 1) // 8 + 1, job.height // 32, job.width // 32)}

    def generate(self, job, noise, clock) -> Generated:
        self.calls.append((job, noise))
        if self.gate is not None:
            self.gate.wait(5)
        if self.fail:
            raise RuntimeError(self.fail)
        with clock.span("text_encode"):
            with clock.span("inner"):  # folded into text_encode
                time.sleep(self.step_s)
        clock.denoise_start()
        for _ in range(job.steps - self.skip_steps):
            time.sleep(self.step_s)
            clock.step()
        with clock.span("vae_decode"):
            time.sleep(self.step_s)
        payload = hashlib.sha256(f"{job.prompt}|{job.seed}|{job.noise_seed}|{job.steps}|{job.task}|"
                                 f"{job.width}x{job.height}x{job.num_frames}".encode()).digest()
        return Generated(video=b"\x00\x00\x00\x18ftypmp42" + payload * 4, width=job.width, height=job.height,
                         frames=job.num_frames, fps=job.fps, effective={"fake": True})

    def write(self, out: Generated, path, clock) -> None:
        path.write_bytes(out.video)


def make_pipeline(opts) -> FakeClipPipeline:
    return FakeClipPipeline(**opts.extra)


# --------------------------------------------------------------------------- a torch stand-in for LTX2Pipeline


def fake_ltx2(i2v: bool = False, wrong_noise: bool = False, with_audio: bool = True):
    import numpy as np
    import torch

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(in_channels=4)
            self.calls = 0

        @property
        def dtype(self):
            return torch.bfloat16

        def set_attention_backend(self, name):
            self.attention_backend = name

        def forward(self, hidden_states, audio_hidden_states=None):
            self.calls += 1
            return hidden_states * 0.1, None if audio_hidden_states is None else audio_hidden_states * 0.1

    class Passthrough(torch.nn.Module):
        def __init__(self, **config):
            super().__init__()
            self.config = SimpleNamespace(**config)

        def set_attention_backend(self, name):
            self.attention_backend = name

        def forward(self, x):
            return x if x.ndim != 4 else x.mean(dim=(1, 3)).unsqueeze(1).repeat(1, 2, 1)  # mel -> [B, 2, T]

    class VAE:
        def decode(self, latents, timestep=None, return_dict=False):
            b, _, f, h, w = latents.shape
            return (torch.zeros(b, 3, (f - 1) * 8 + 1, h * 32, w * 32),)

    class AudioVAE:
        def __init__(self):
            self.config = SimpleNamespace(latent_channels=2, mel_bins=16)
            self.latents_mean, self.latents_std = torch.full((8,), 0.5), torch.full((8,), 2.0)

        def decode(self, latents, return_dict=False):
            return (latents,)

    class VideoProcessor:
        def postprocess_video(self, video, output_type="np"):
            return np.clip(video.permute(0, 2, 3, 4, 1).float().numpy() + 0.5, 0, 1)

    class FakeLTX2:
        vae_spatial_compression_ratio = 32
        vae_temporal_compression_ratio = 8
        transformer_spatial_patch_size = 1
        transformer_temporal_patch_size = 1
        audio_sampling_rate = 16000
        audio_hop_length = 160
        audio_vae_temporal_compression_ratio = 4
        audio_vae_mel_compression_ratio = 4

        def __init__(self):
            self.transformer = Transformer()
            self.connectors = Passthrough()
            self.vae = VAE()
            self.audio_vae = AudioVAE() if with_audio else None
            self.vocoder = Passthrough(output_sampling_rate=24000) if with_audio else None
            self.video_processor = VideoProcessor()
            self.last_kwargs: dict = {}
            self.placed = None  # what configure did: ("to", device) or ("model_offload", device)

        def to(self, device):
            self.placed = ("to", device)
            return self

        def enable_model_cpu_offload(self, device=None):
            self.placed = ("model_offload", device)

        @staticmethod
        def _pack_latents(latents, patch_size=1, patch_size_t=1):
            return latents.permute(0, 2, 3, 4, 1).flatten(1, 3)  # [B, F*H*W, C] for patch size 1

        @staticmethod
        def _pack_audio_latents(latents):
            return latents.transpose(1, 2).flatten(2, 3)  # [B, L, C*M]

        def encode_prompt(self, prompt, negative_prompt=None):
            return torch.ones(2 if negative_prompt is not None else 1, 3, 8)

        def prepare_latents(self, shape, generator, image=None):
            if wrong_noise:
                generator = torch.Generator("cpu").manual_seed(12345)
            noise = torch.randn(shape, generator=generator, dtype=torch.float32)
            if image is None:
                return self._pack_latents(noise)
            mask = torch.zeros((shape[0], 1, *shape[2:]))
            mask[:, :, 0] = 1.0
            latents = torch.full(shape, 7.0) * mask + noise * (1 - mask)
            return self._pack_latents(latents), self._pack_latents(mask).squeeze(-1)

        def prepare_audio_latents(self, shape, generator):
            return self._pack_audio_latents(torch.randn(shape, generator=generator, dtype=torch.float32))

        def __call__(self, prompt, negative_prompt, height, width, num_frames, frame_rate, num_inference_steps,
                     guidance_scale, generator, output_type, return_dict, callback_on_step_end, image=None, **extra):
            self.last_kwargs = dict(prompt=prompt, negative_prompt=negative_prompt, height=height, width=width,
                                    num_frames=num_frames, frame_rate=frame_rate, steps=num_inference_steps,
                                    guidance_scale=guidance_scale, generator=generator, image=image, **extra)
            self.connectors(self.encode_prompt(prompt, negative_prompt))
            f, h, w = (num_frames - 1) // 8 + 1, height // 32, width // 32
            out = self.prepare_latents((1, 4, f, h, w), generator, image=image)
            lat = out[0] if isinstance(out, tuple) else out
            aud = None
            if self.audio_vae is not None:
                length = round(num_frames / frame_rate * self.audio_sampling_rate / self.audio_hop_length / 4)
                aud = self.prepare_audio_latents((1, 2, length, 4), generator)
            for i in range(num_inference_steps):
                v, _ = self.transformer(lat, aud)
                lat = lat - v
                callback_on_step_end(self, i, i, {})
            video = self.vae.decode(lat.reshape(1, f, h, w, 4).permute(0, 4, 1, 2, 3), None, return_dict=False)[0]
            frames = self.video_processor.postprocess_video(video, output_type=output_type)
            wave = None
            if self.audio_vae is not None:
                mel = self.audio_vae.decode(aud.reshape(1, -1, 2, 4).transpose(1, 2), return_dict=False)[0]
                wave = self.vocoder(mel)
            return frames, wave

    return FakeLTX2()
