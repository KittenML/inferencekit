"""A tiny LTX-2-shaped pipeline with a real denoising loop, for the per-step dump and replay tests (torch, CPU).

Like Diffusers' LTX2Pipeline in what diffusers_ltx touches: video and audio latents packed as token sequences, a
flow-matching Euler scheduler stepped with ``scheduler.step(...)`` for video and with a deep copy of it for audio,
CFG (conditional and unconditional in one batch) plus a perturbed pass (STG: attention skipped) at every step,
prepare_latents / prepare_audio_latents drawing float32 noise video then audio from the generator, a decoder to
frames in [0, 1] and an audio decoder plus vocoder. Attention backends: ``native`` float32 scores, the other exact
ones (``_native_efficient``, ``_native_cudnn``) float64 scores rounded to float32 (another exact kernel: the floor),
``lowp`` bfloat16 scores (a lower-precision engine).

Knobs (``--pipeline-arg``), the shortcuts the per-step check must catch: ``fewer_steps=N`` runs N fewer steps,
``skip_guidance_steps=[i, ...]`` drops CFG and STG at those steps, ``reuse_steps=[i, ...]`` reuses the previous
step's guided output, ``fewer_frames=N`` renders N fewer frames, ``other_sigmas=true`` shifts the schedule
differently. ``make_pipeline`` is the factory for
``--pipeline fakes_clip.fake_steps:make_pipeline``.
"""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

KNOBS = ("fewer_steps", "skip_guidance_steps", "reuse_steps", "fewer_frames", "other_sigmas")


def fake_loop_pipeline(**knobs):
    import torch

    cv, ca, d = 4, 2, 8

    class Scheduler:
        def __init__(self):
            self.sigmas, self.timesteps, self._step_index, self._begin_index = None, None, None, None

        @property
        def step_index(self):
            return self._step_index

        def set_timesteps(self, n):
            s = torch.linspace(1.0, 1.0 / n, n, dtype=torch.float64)
            k = 2.0 if knobs.get("other_sigmas") else 3.0
            s = k * s / (1 + (k - 1) * s)  # a shifted schedule
            self.sigmas = torch.cat([s, torch.zeros(1, dtype=torch.float64)]).float()
            self.timesteps = self.sigmas[:-1] * 1000
            self._step_index = None

        def set_begin_index(self, i):
            self._begin_index = i

        def step(self, model_output, timestep, sample, return_dict=False):
            if self._step_index is None:
                self._step_index = self._begin_index or 0
            s, sn = self.sigmas[self._step_index], self.sigmas[self._step_index + 1]
            prev = (sample.float() + (sn - s) * model_output.float()).to(model_output.dtype)
            self._step_index += 1
            return (prev,)

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(1)
            self.config = SimpleNamespace(in_channels=cv)
            self.win = torch.randn(cv, d, generator=g) / 2
            self.wout = torch.randn(d, cv, generator=g) / 3
            self.ain = torch.randn(ca * 4, d, generator=g) / 2
            self.aout = torch.randn(d, ca * 4, generator=g) / 3
            self.wtext = torch.randn(8, d, generator=g) / 4
            self.backend = "native"

        @property
        def dtype(self):
            return torch.bfloat16

        def set_attention_backend(self, name):
            self.backend = name

        def _attn(self, h):
            dt = {"native": torch.float32, "lowp": torch.bfloat16}.get(self.backend, torch.float64)
            q = h.to(dt)
            w = torch.softmax(q @ q.transpose(-1, -2) / math.sqrt(d), dim=-1)
            return (w @ q).float()

        def forward(self, hidden_states, audio_hidden_states, text, timestep, perturb=False):
            t = (timestep.float() / 1000).view(-1, 1, 1)
            hv = hidden_states.float() @ self.win + text.mean(1, keepdim=True) @ self.wtext + t
            ha = audio_hidden_states.float() @ self.ain + t
            if not perturb:
                hv, ha = hv + self._attn(hv), ha + self._attn(ha)
            hv, ha = hv + ha.mean(1, keepdim=True), ha + hv.mean(1, keepdim=True)
            return torch.tanh(hv) @ self.wout, torch.tanh(ha) @ self.aout

    class Passthrough(torch.nn.Module):
        def __init__(self, **config):
            super().__init__()
            self.config = SimpleNamespace(**config)

        def set_attention_backend(self, name):
            self.backend = name

        def forward(self, x):
            return x if x.ndim != 4 else x.mean(dim=(1, 3)).unsqueeze(1).repeat(1, 2, 1)

    class VAE:
        def decode(self, latents, timestep=None, return_dict=False):
            b, c, f, h, w = latents.shape
            rgb = torch.sigmoid(latents[:, :3] * 1.5)  # [B, 3, F, H, W]
            rgb = rgb.repeat_interleave(8, dim=2)[:, :, : (f - 1) * 8 + 1]
            return (rgb.repeat_interleave(32, dim=3).repeat_interleave(32, dim=4),)

    class AudioVAE:
        def __init__(self):
            self.config = SimpleNamespace(latent_channels=ca, mel_bins=16)

        def decode(self, latents, return_dict=False):
            return (latents,)

    class VideoProcessor:
        def postprocess_video(self, video, output_type="np"):
            return video.permute(0, 2, 3, 4, 1).float().numpy()

    class FakeLoop:
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
            self.audio_vae = AudioVAE()
            self.vocoder = Passthrough(output_sampling_rate=16000)
            self.video_processor = VideoProcessor()
            self.scheduler = Scheduler()

        def to(self, device):
            return self

        @staticmethod
        def _pack_latents(latents, patch_size=1, patch_size_t=1):
            return latents.permute(0, 2, 3, 4, 1).flatten(1, 3)

        @staticmethod
        def _unpack_latents(latents, num_frames, height, width, patch_size=1, patch_size_t=1):
            b = latents.size(0)
            return latents.reshape(b, num_frames, height, width, -1).permute(0, 4, 1, 2, 3)

        @staticmethod
        def _pack_audio_latents(latents):
            return latents.transpose(1, 2).flatten(2, 3)

        @staticmethod
        def _unpack_audio_latents(latents, latent_length, num_mel_bins):
            b = latents.size(0)
            return latents.reshape(b, latent_length, -1, num_mel_bins).transpose(1, 2)

        def encode_prompt(self, prompt, negative_prompt=None):
            def emb(text):
                g = torch.Generator().manual_seed(sum(map(ord, text or "")) % (2 ** 31))
                return torch.randn(1, 3, 8, generator=g)
            return torch.cat([emb(negative_prompt), emb(prompt)])

        def prepare_latents(self, shape, generator):
            return self._pack_latents(torch.randn(shape, generator=generator, dtype=torch.float32))

        def prepare_audio_latents(self, shape, generator):
            return self._pack_audio_latents(torch.randn(shape, generator=generator, dtype=torch.float32))

        def __call__(self, prompt, negative_prompt, height, width, num_frames, frame_rate, num_inference_steps,
                     guidance_scale, generator, output_type, return_dict, callback_on_step_end, **extra):
            num_frames -= int(knobs.get("fewer_frames") or 0)
            steps = num_inference_steps - int(knobs.get("fewer_steps") or 0)
            skip = set(knobs.get("skip_guidance_steps") or ())
            reuse = set(knobs.get("reuse_steps") or ())
            text = self.connectors(self.encode_prompt(prompt, negative_prompt))
            f, h, w = (num_frames - 1) // 8 + 1, height // 32, width // 32
            lat = self.prepare_latents((1, cv, f, h, w), generator)
            length = round(num_frames / frame_rate * self.audio_sampling_rate / self.audio_hop_length / 4)
            aud = self.prepare_audio_latents((1, ca, length, 4), generator)
            self.scheduler.set_timesteps(steps)
            audio_scheduler = copy.deepcopy(self.scheduler)
            self.scheduler.set_begin_index(0)
            audio_scheduler.set_begin_index(0)
            last = None
            for i, t in enumerate(self.scheduler.timesteps):
                if i in reuse and last is not None:
                    pv, pa = last
                else:
                    tt = t.expand(2)
                    (vu, vc), (au, ac) = (o.chunk(2) for o in self.transformer(torch.cat([lat] * 2),
                                                                               torch.cat([aud] * 2), text, tt))
                    if i in skip:
                        pv, pa = vc, ac
                    else:
                        vp, ap = self.transformer(lat, aud, text[1:], t.expand(1), perturb=True)
                        pv = vc + (guidance_scale - 1) * (vc - vu) + (vc - vp)
                        pa = ac + (guidance_scale - 1) * (ac - au) + (ac - ap)
                last = (pv, pa)
                lat = self.scheduler.step(pv, t, lat, return_dict=False)[0]
                aud = audio_scheduler.step(pa, t, aud, return_dict=False)[0]
                callback_on_step_end(self, i, t, {})
            video = self.vae.decode(self._unpack_latents(lat, f, h, w), None, return_dict=False)[0]
            frames = self.video_processor.postprocess_video(video, output_type=output_type)
            mel = self.audio_vae.decode(self._unpack_audio_latents(aud, length, 4), return_dict=False)[0]
            wave = self.vocoder(mel)
            return frames, wave

    return FakeLoop()


def make_pipeline(opts):
    import dataclasses

    import torch

    from inferencekit.engines.clip import diffusers_ltx

    knobs = {k: opts.extra[k] for k in KNOBS if k in opts.extra}
    opts = dataclasses.replace(opts, extra={k: v for k, v in opts.extra.items() if k not in KNOBS},
                               attention=opts.attention or "native")
    pipe = fake_loop_pipeline(**knobs)
    diffusers_ltx.configure(pipe, opts, reference=False)
    return diffusers_ltx.LTXDiffusers(pipe, opts, torch, {"pipeline": "fake-loop", "knobs": knobs})
