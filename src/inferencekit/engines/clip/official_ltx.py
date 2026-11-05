"""The official LTX-2 inference pipelines (the ``ltx_pipelines`` package) behind the clip server.

    python -m inferencekit.engines.clip.server --pipeline official_ltx --model DIR --task t2v \\
        --pipeline-arg builder=MODULE:CALLABLE [--pipeline-arg builder.NAME=VALUE ...] \\
        [--pipeline-arg step_module=ATTR.PATH --pipeline-arg forwards_per_step=N]

Run on a GPU host with ltx-pipelines v1.4.2 (15-16 Oct 2026: the LTX-2 monolith checkpoint, and LTX-2.5's split
files through ``builder.transformer/text_encoder/video_vae/audio_vae``, with the one_stage builder below; on
LTX-2.5 bench rounds and gates clip PASS for t2v and i2v). Written before the package was installed: Diffusers'
encode_video cites
``packages/ltx-pipelines/src/ltx_pipelines/utils/media_io.py`` in the LTX-2 repository, so the import name
is ``ltx_pipelines``. Everything else is looked up when the server starts or a clip runs, and refused when
it is not there, rather than guessed:

  * The pipeline: ``builder=MODULE:CALLABLE`` (required; e.g. a pipeline class or a small function of
    your own in that venv) is called as ``builder(model=..., device=..., dtype=..., **builder.* args)``;
    the object it returns makes one clip per call (``generate`` if it has one, else ``__call__``).
  * Call arguments are matched by name (ALIASES, e.g. ``num_inference_steps`` or ``steps``). A request
    field with no matching argument is refused (400), never dropped.
  * Noise: if the call takes ``generator``, it gets ``torch.Generator("cpu").manual_seed(noise_seed)``;
    else, if it takes ``seed``, it gets ``noise_seed``. Whether the package draws video then audio noise
    from it the way noise.py does is unchecked, so ``effective.noise_check`` is false: do not use this
    engine as one side of a shared-noise quality comparison until a check shows it.
  * Steps: ``step_module=ATTR.PATH`` names the denoiser module on the pipeline object; its forward
    calls are counted and divided by ``forwards_per_step`` (default 1; 2 when guidance runs the
    conditional and unconditional passes as separate calls). Without it steps_run is null (not counted)
    and bench clip says the steps were not checked.
  * Output: if the call takes ``output_path`` the package writes the mp4 itself; otherwise it must return
    (video, audio) or an object with ``video``/``frames`` and ``audio``, frames shaped [F, H, W, 3]
    (uint8, or float in [0, 1]), which are muxed with ``diffusers.utils.encode_video`` (needs Diffusers
    and PyAV in that venv). Audio needs ``--pipeline-arg audio_sample_rate=N``.

Stages: only ``generate`` (the whole call) and ``mux``; the package's inner stages are not hooked.

Per-step dumps (stepdump.py, for quality clip-steps): NOT written. The package's denoising loop is not hooked (it
was not read when this wrapper was written), so an engine served through this wrapper has no dumps and cannot pass
quality clip-steps; a hook belongs where the package calls its sampler step, recording x, the guided velocity and
the next latent with stepdump.StepRecorder and returning it as Generated.step_dump.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import io

from inferencekit.engines.clip.server import BadRequest, ClipJob, Generated, PipelineOptions, StageClock

# request field -> call argument names to try, in order
ALIASES = {
    "prompt": ("prompt",),
    "width": ("width",),
    "height": ("height",),
    "num_frames": ("num_frames", "frames", "length"),
    "fps": ("frame_rate", "fps"),
    "steps": ("num_inference_steps", "steps", "num_steps"),
    "guidance_scale": ("guidance_scale", "cfg_guidance_scale", "cfg_scale", "cfg"),
    "image": ("image", "images", "conditioning_image"),
}
OPTIONS = {"builder", "step_module", "forwards_per_step", "audio_sample_rate", "negative_prompt"}


def _resolve(spec: str):
    mod, _, attr = str(spec).partition(":")
    if not attr:
        raise ValueError(f"builder must be MODULE:CALLABLE, got {spec!r}")
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def make_pipeline(opts: PipelineOptions) -> OfficialLTX:
    try:
        import ltx_pipelines  # noqa: F401 - the guard: refuse early when the package is missing
    except ImportError as e:
        raise ImportError("official_ltx needs the ltx_pipelines package (LTX-2 repository) in this "
                          f"interpreter: {e}") from None
    import torch

    unknown = sorted(k for k in opts.extra if k not in OPTIONS and not k.startswith("builder."))
    if unknown:
        raise ValueError(f"unknown --pipeline-arg {unknown}; known: {sorted(OPTIONS)} and builder.NAME")
    if not opts.extra.get("builder"):
        raise ValueError("official_ltx needs --pipeline-arg builder=MODULE:CALLABLE (see the module docstring)")
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if opts.dtype not in dtypes:
        raise ValueError(f"--dtype must be one of {sorted(dtypes)}")
    if opts.offload != "none" or opts.attention:
        raise ValueError("official_ltx takes no --offload or --attention (pass the package's own options "
                         "as builder.NAME=VALUE)")
    build = _resolve(opts.extra["builder"])
    bkw = {k[8:]: v for k, v in opts.extra.items() if k.startswith("builder.")}
    pipe = build(model=opts.model, device=opts.device, dtype=dtypes[opts.dtype], **bkw)
    version = getattr(importlib.import_module("ltx_pipelines"), "__version__", None)
    return OfficialLTX(pipe, opts, torch, {"pipeline": f"ltx_pipelines:{opts.extra['builder']}",
                                           "ltx_pipelines_version": version, "torch_version": torch.__version__})


class OfficialLTX:
    def __init__(self, pipe, opts: PipelineOptions, torch_mod, facts: dict | None = None):
        self.pipe, self.opts, self.torch = pipe, opts, torch_mod
        self.facts = dict(facts or {})
        self.fn = getattr(pipe, "generate", None) or pipe
        self.params = set(inspect.signature(self.fn).parameters)
        self.forwards = 0
        self.per_step = int(opts.extra.get("forwards_per_step") or 1)
        self.counting = False
        self._out_path: str | None = None
        if opts.extra.get("step_module"):
            mod = pipe
            for part in str(opts.extra["step_module"]).split("."):
                mod = getattr(mod, part)
            mod.register_forward_pre_hook(self._count)
            self.counting = True

    def _count(self, module, args):
        self.forwards += 1

    def _arg(self, field: str) -> str:
        for name in ALIASES[field]:
            if name in self.params:
                return name
        raise BadRequest(f"the pipeline call takes none of {list(ALIASES[field])} for {field}")

    def describe(self) -> dict:
        neg = self.opts.extra.get("negative_prompt")
        own = self.pipe.describe() if callable(getattr(self.pipe, "describe", None)) else {}
        return {**self.facts, "model": self.opts.model, "task": self.opts.task, "dtype": self.opts.dtype,
                "attention": "package default", "device": self.opts.device,
                "steps_counted": self.counting or hasattr(self.pipe, "steps_run"),
                "negative_prompt_sha256": hashlib.sha256(str(neg).encode()).hexdigest() if neg is not None else None,
                **own}

    def noise_shapes(self, job: ClipJob) -> dict:
        if job.task != self.opts.task:
            raise BadRequest(f"this server loaded the {self.opts.task} pipeline; start another for {job.task}")
        if "output_path" not in self.params and float(job.fps) != int(job.fps):
            raise BadRequest(f"fps must be a whole number for this muxer, got {job.fps:g}")
        return {}  # the package draws its own noise (see the module docstring)

    def generate(self, job: ClipJob, noise: dict, clock: StageClock) -> Generated:
        kw = {self._arg(f): getattr(job, f) for f in ("prompt", "width", "height", "num_frames", "fps", "steps",
                                                         "guidance_scale")}
        if job.task == "i2v":
            from PIL import Image

            kw[self._arg("image")] = Image.open(io.BytesIO(job.image)).convert("RGB")
        if "generator" in self.params:
            kw["generator"], how = self.torch.Generator(device="cpu").manual_seed(job.noise_seed), "cpu generator"
        elif "seed" in self.params:
            kw["seed"], how = job.noise_seed, "package seed"
        else:
            raise BadRequest("the pipeline call takes neither generator nor seed: the noise cannot be fixed")
        if self.opts.extra.get("negative_prompt") is not None:
            if "negative_prompt" not in self.params:
                raise BadRequest("negative_prompt was set but the pipeline call does not take it")
            kw["negative_prompt"] = self.opts.extra["negative_prompt"]
        self._out_path = None
        if "output_path" in self.params:
            import os
            import tempfile

            fd, self._out_path = tempfile.mkstemp(suffix=".mp4", dir=os.getcwd())
            os.close(fd)
            kw["output_path"] = self._out_path
        self.forwards = 0
        with clock.span("generate"):
            ret = self.fn(**kw)
        if self.counting:  # else nothing counted them: the server reports steps_run null, not 0
            clock.set_steps(self.forwards // self.per_step)
        elif isinstance(getattr(self.pipe, "steps_run", None), int):  # a builder that counts its own loop
            clock.set_steps(self.pipe.steps_run)
        video, audio = (ret if isinstance(ret, tuple) else (getattr(ret, "video", getattr(ret, "frames", None)),
                                                             getattr(ret, "audio", None)))
        eff = {"noise_check": False, "generator": how}
        if self._out_path:  # the package wrote the file: audio is what the builder says it muxed, else unknown
            audio_written = getattr(self.pipe, "audio_written", None)
            return Generated(video=None, width=job.width, height=job.height, frames=job.num_frames, fps=job.fps,
                             effective={**eff, "shape_source": "request (the package wrote the file)",
                                        "audio": audio_written if isinstance(audio_written, bool) else None})
        if video is None:
            raise RuntimeError("the pipeline returned no frames and took no output_path")
        frames = video[0] if getattr(video, "ndim", 4) == 5 else video
        n, h, w = (int(d) for d in frames.shape[:3])
        rate = self.opts.extra.get("audio_sample_rate")
        wave = None
        if audio is not None and job.audio is not False and rate:
            wave = audio[0] if getattr(audio, "ndim", 2) == 3 else audio
        return Generated(video=frames, width=w, height=h, frames=n, fps=job.fps, audio=wave,
                         audio_sample_rate=int(rate) if wave is not None else None, effective=eff)

    def write(self, out: Generated, path, clock: StageClock) -> None:
        if out.video is None and self._out_path:
            import shutil

            shutil.move(self._out_path, path)
            return
        from diffusers.utils import encode_video

        encode_video(out.video, fps=int(out.fps), output_path=str(path), audio=out.audio,
                     audio_sample_rate=out.audio_sample_rate)


# --------------------------------------------------------------------------- a builder for the one-stage pipeline


def one_stage(model: str, device: str, dtype, gemma_root: str | None = None, transformer: str | None = None,
              text_encoder: str | None = None, video_vae: str | None = None, audio_vae: str | None = None,
              negative_prompt: str | None = None) -> OneStage:
    """``builder=inferencekit.engines.clip.official_ltx:one_stage``: the official text/image-to-video one-stage
    pipeline (``ltx_pipelines.ti2vid_one_stage.TI2VidOneStagePipeline``, the full model with guidance), with
    the package's own defaults for everything the request does not set.

    ``--model`` is the monolith checkpoint (e.g. ltx-2-19b-dev.safetensors) with ``builder.gemma_root=DIR``
    (the Gemma text encoder directory), or a split pack: ``builder.transformer=``, ``builder.text_encoder=``,
    ``builder.video_vae=``, ``builder.audio_vae=`` (``--model`` is then only recorded). Guidance: the request's
    guidance_scale is the video CFG scale; every other guider setting (STG, rescale, modality, the audio
    guider) is the package's default for the checkpoint (``detect_params``), recorded in describe(). The
    negative prompt is the package's DEFAULT_NEGATIVE_PROMPT unless ``builder.negative_prompt`` is given. The
    package draws its own noise on the GPU from the seed (no shared-noise recipe), writes the mp4 itself and
    builds the transformer on every call, as it ships. Steps are counted by the denoise loop's calls.
    """
    import torch
    from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
    from ltx_pipelines.utils.constants import detect_params
    from ltx_pipelines.utils.model_paths import ModelPaths

    if dtype != torch.bfloat16:
        raise ValueError("the official one-stage pipeline runs in BF16 only (--dtype bf16)")
    split = {k: v for k, v in (("transformer_path", transformer), ("text_encoder_path", text_encoder),
                               ("video_vae_path", video_vae), ("audio_vae_path", audio_vae)) if v}
    if split:
        paths = ModelPaths.from_split(**split)
        checkpoint = transformer or model
    else:
        if not gemma_root:
            raise ValueError("a monolith checkpoint needs --pipeline-arg builder.gemma_root=DIR (the text encoder)")
        paths, checkpoint = ModelPaths.from_monolith(model, gemma_root), model
    params = detect_params(checkpoint)
    pipe = TI2VidOneStagePipeline(model_paths=paths, loras=[], device=torch.device(device))
    if negative_prompt is None:
        from ltx_pipelines.utils.args import DEFAULT_NEGATIVE_PROMPT
        negative_prompt = DEFAULT_NEGATIVE_PROMPT
    return OneStage(pipe, params, negative_prompt)


class _CountingStage:
    """The pipeline's diffusion stage, with its denoiser wrapped so each loop step is counted."""

    def __init__(self, stage, owner: OneStage):
        self._stage, self._owner = stage, owner

    def __getattr__(self, name):
        return getattr(self._stage, name)

    def __call__(self, *args, denoiser, **kwargs):
        owner = self._owner

        def counted(*a, **k):
            owner.steps_run += 1
            return denoiser(*a, **k)

        return self._stage(*args, denoiser=counted, **kwargs)


class OneStage:
    """What ``one_stage`` returns: ``generate(...)`` makes one clip and writes it to ``output_path``."""

    def __init__(self, pipe, params, negative_prompt: str):
        self.pipe, self.params, self.negative_prompt = pipe, params, negative_prompt
        self.steps_run: int | None = None
        self.audio_written: bool | None = None  # whether the last clip's file got an audio track
        pipe.stage = _CountingStage(pipe.stage, self)

    def describe(self) -> dict:
        vg, ag = self.params.video_guider_params, self.params.audio_guider_params
        return {"video_guider": {k: getattr(vg, k) for k in ("stg_scale", "rescale_scale", "modality_scale",
                                                               "skip_step", "stg_blocks")},
                "audio_guider": {k: getattr(ag, k) for k in ("cfg_scale", "stg_scale", "rescale_scale",
                                                               "modality_scale", "skip_step", "stg_blocks")},
                "negative_prompt_sha256": hashlib.sha256(self.negative_prompt.encode()).hexdigest()}

    def generate(self, prompt, width, height, num_frames, frame_rate, num_inference_steps, guidance_scale, seed,
                 output_path, image=None):
        import dataclasses
        import os
        import tempfile

        import torch
        from ltx_core.model.video_vae import get_video_chunks_number
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_pipelines.utils.types import ImageConditioningInput

        images, tmp = [], None
        if image is not None:  # the package conditions on an image file at frame 0, strength 1.0
            fd, tmp = tempfile.mkstemp(suffix=".png", dir=os.path.dirname(os.path.abspath(output_path)))
            os.close(fd)
            image.save(tmp)
            images = [ImageConditioningInput(tmp, 0, 1.0)]
        self.steps_run, self.audio_written = 0, None
        try:
            with torch.inference_mode():
                out = self.pipe(prompt=prompt, negative_prompt=self.negative_prompt, seed=int(seed), height=int(height),
                                width=int(width), frame_rate=float(frame_rate), num_frames=int(num_frames),
                                num_inference_steps=int(num_inference_steps),
                                video_guider_params=dataclasses.replace(self.params.video_guider_params,
                                                                        cfg_scale=float(guidance_scale)),
                                audio_guider_params=self.params.audio_guider_params, images=images)
                encode_video(video=out.video, fps=int(frame_rate), audio=out.audio, output_path=str(output_path),
                             video_chunks_number=get_video_chunks_number(out.num_frames, out.tiling_config))
            self.audio_written = out.audio is not None
        finally:
            if tmp:
                os.unlink(tmp)
