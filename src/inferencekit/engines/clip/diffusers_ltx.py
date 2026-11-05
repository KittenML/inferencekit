"""LTX-2.x through Diffusers behind the clip server: text-to-video and image-to-video.

    python -m inferencekit.engines.clip.server --pipeline diffusers_ltx --model Lightricks/LTX-2.5 \\
        --task t2v --dtype bf16 [--offload model] [--attention native] \\
        [--pipeline-arg reference=true --pipeline-arg default_negative_prompt=true]

What was checked, and how (Diffusers 0.41.0.dev0 from the main branch, read on 15 Oct 2026; run on a GPU host on
LTX-2 (15 Oct) and LTX-2.5 (16 Oct)):
  * Classes: ``diffusers.LTX2Pipeline`` (t2v) and ``diffusers.LTX2ImageToVideoPipeline`` (i2v), both in
    diffusers/pipelines/ltx2. Both take prompt, negative_prompt, height, width, num_frames, frame_rate,
    num_inference_steps, guidance_scale, generator, output_type, return_dict, callback_on_step_end; i2v
    also takes image. With return_dict=False the call returns (video, audio); output_type="np" gives
    float frames in [0, 1], shape [batch, frames, height, width, 3]; audio is a waveform
    [batch, channels, samples] at ``pipe.vocoder.config.output_sampling_rate``.
  * Sizes: height and width must be multiples of 32 (check_inputs); frames are 8k+1 (the VAE's temporal
    ratio); the latent video is [1, transformer.config.in_channels, (frames-1)//8+1, height//32, width//32].
  * Noise: randn_tensor draws on the generator's device, so a CPU generator gives the same numbers on any
    GPU. With ``generator=torch.Generator("cpu").manual_seed(noise_seed)`` the pipeline's first draw is
    the video noise and the second the audio noise, both float32: the recipe in noise.py. The pipeline
    draws the noise itself rather than taking ``latents`` because in i2v a ``latents`` argument
    replaces the image-conditioned latents (the image would be lost), and audio ``latents`` would be
    renormalised. Hooks on prepare_latents / prepare_audio_latents compare what the pipeline drew with
    the server's CPU tensors on every clip: ``effective.noise_check`` is true only when they are equal.
  * Muxing: ``diffusers.utils.encode_video`` (PyAV, libx264 yuv420p, AAC audio). It writes an integer
    frame rate (``rate=int(fps)``), so a fractional fps is refused instead of being silently rounded.

BF16 reference mode (``--pipeline-arg reference=true``): every component in BF16, PyTorch
scaled-dot-product attention (backend ``native``, set on the transformer and the text connectors, so a
``DIFFUSERS_ATTN_BACKEND`` in the environment cannot change it), the default pipeline class, no
``call.*`` overrides, no VAE tiling (tiles are blended, which changes the frames) and no prompt
enhancement. It is the pipeline as shipped, the side that quality clip compares the others against.
Offloading moves weights but changes no numbers, so it is allowed in reference mode. Reference mode also
needs an explicit negative prompt (``default_negative_prompt=true`` or ``negative_prompt=TEXT``; an empty
TEXT for none): without one Diffusers encodes an empty one, and nothing would say which the clips used.

Other options (``--pipeline-arg``): ``pipeline_class=NAME`` (another diffusers class with the same call
arguments), ``negative_prompt=TEXT`` or ``default_negative_prompt=true`` (Diffusers' LTX-2 DEFAULT_NEGATIVE_PROMPT,
the one the model cards use), ``revision=REV``, ``vae_tiling=true``, ``call.NAME=VALUE`` (an extra __call__
argument, e.g. ``call.decode_timestep=0.05``; never one the request sets), ``transformer_subfolder=NAME`` (load
the transformer from another folder of the checkpoint) and ``scheduler=JSON`` (scheduler config overrides).

LTX-2.5-Diffusers (from its model card, then run on its weights on 16 Oct 2026): ``transformer/`` is the DISTILLED
DiT and model_index.json loads it by default, with a scheduler set up for distilled sigma schedules. The full
BF16 model is ``transformer_subfolder=transformer_full`` with ``scheduler={"use_dynamic_shifting": true,
"shift_terminal": 0.1}``. Reference mode refuses a local folder that has transformer_full/ unless
transformer_subfolder is given, so a reference never runs the distilled DiT by accident: a hub id is looked up
in the Hugging Face cache (offline, at ``revision``), and reference mode refuses a hub id it cannot find there.
In reference mode a transformer_subfolder needs the scheduler overrides that go with it (use_dynamic_shifting
and shift_terminal: the checkpoint's own scheduler is set up for the distilled sigmas). These options are
allowed in reference mode (they pick the checkpoint and its schedule; they are recorded in describe(), and
a re-run of the manifest can compare that with the reference settings it expects).

Loading (first run on real LTX-2.5 weights, RTX PRO 6000, 16 Oct 2026): a hub id with ``revision`` set to a full
40-character commit sha (or with HF_HUB_OFFLINE set) is loaded from its Hugging Face cache snapshot folder when it
is there, because Diffusers' sharded-checkpoint loader asks the Hub for the file list even offline and fails
(OfflineModeIsEnabled). A branch or tag name online is left to from_pretrained: the cache's ref for it may point
at an older snapshot. describe() records the revision asked for and the snapshot's commit (``snapshot_commit``), so
a reference manifest's expect_config can pin the commit that was loaded. LTX-2.5-Diffusers' model_index.json also lists optional components a clip never uses:
the Gemma-4 prompt enhancer (about 10 GB) with its processor, and the duration head (only used when num_frames
is left out). They are passed as None (``skipped_components`` in describe()): loaded, they would only take GPU
memory, and on a host with only the folders a clip loads, loading them fails.

Per-step dumps (stepdump.py) and replay: the scheduler class's ``step`` is wrapped for the length of each pipeline call
(the pipeline steps the video latents with ``pipe.scheduler`` and the audio latents with a deep copy of it, so an
instance attribute would be copied into the audio scheduler still bound to the video one; the class-level wrapper
tells them apart by identity). Every clip records, per step and stream, the latent the step started from, the
guided velocity the scheduler took and the latent it returned (unpacked to the noise layout) and the scheduler's
sigmas. ``replay`` is the checker's side (quality clip-steps): the reference pipeline runs a request with an engine's
dump forced in: prepare_latents returns the dump's first latents, and every scheduler step returns the dump's next
latent, so each step's transformer passes see the engine's x_t; it reports the reference's own guided output at
every step, the reference scheduler's update of the engine's output (the chain check), and the frames its own
decode path makes from the dump's final latent.

Stages (self-reported, see server.py): text_encode (encode_prompt + connectors), prepare_latents (in i2v
this includes encoding the image), denoise (+ denoise_steps), vae_decode, postprocess (to frames),
audio_decode (audio VAE + vocoder), mux.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import re
from contextlib import contextmanager

from inferencekit.engines.clip import stepdump
from inferencekit.engines.clip.server import (BadRequest, ClipJob, Generated, PipelineOptions, StageClock,
                                              hook_method, timed)

DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
CLASSES = {"t2v": "LTX2Pipeline", "i2v": "LTX2ImageToVideoPipeline"}
# __call__ arguments the request sets: a call.* override may not touch them.
REQUEST_ARGS = {"prompt", "negative_prompt", "image", "height", "width", "num_frames", "frame_rate",
                "num_inference_steps", "guidance_scale", "generator", "latents", "audio_latents", "output_type",
                "return_dict", "callback_on_step_end", "callback_on_step_end_tensor_inputs"}
OPTIONS = {"reference", "pipeline_class", "negative_prompt", "default_negative_prompt", "revision", "vae_tiling",
           "transformer_subfolder", "scheduler"}
# Optional pipeline components a clip never uses (prompt enhancement is off; num_frames is always given), passed as
# None so from_pretrained does not load them (see the module docstring).
UNUSED_COMPONENTS = ("prompt_enhancer", "processor", "duration_head")
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")  # a Hub revision that names one commit (a branch or tag can move)


def _dtype_name(dtype) -> str:
    s = str(dtype).replace("torch.", "")
    return {v: k for k, v in DTYPES.items()}.get(s, s)


def check_options(opts: PipelineOptions) -> bool:
    """ValueError for options this factory does not take or reference mode forbids; True in reference mode."""
    unknown = sorted(k for k in opts.extra if k not in OPTIONS and not k.startswith("call."))
    if unknown:
        raise ValueError(f"unknown --pipeline-arg {unknown}; known: {sorted(OPTIONS)} and call.NAME")
    if opts.dtype not in DTYPES:
        raise ValueError(f"--dtype must be one of {sorted(DTYPES)}")
    if opts.extra.get("default_negative_prompt") and opts.extra.get("negative_prompt") is not None:
        raise ValueError("give negative_prompt or default_negative_prompt=true, not both")
    sched = opts.extra.get("scheduler")
    if sched is not None and not isinstance(sched, dict):
        raise ValueError("scheduler must be a JSON object of scheduler config overrides, e.g. "
                         '{"use_dynamic_shifting": true, "shift_terminal": 0.1}')
    reference = bool(opts.extra.get("reference"))
    if not reference:
        return False
    bad = [k for k in opts.extra if k.startswith("call.") or k in ("vae_tiling", "pipeline_class")]
    if opts.dtype != "bf16" or opts.attention not in (None, "native") or bad:
        raise ValueError("reference mode is BF16 with native attention, the default pipeline class and "
                         "no call.* overrides or VAE tiling")
    if opts.extra.get("negative_prompt") is None and not opts.extra.get("default_negative_prompt"):
        raise ValueError("reference mode needs an explicit negative prompt: --pipeline-arg default_negative_prompt=true "
                         "(Diffusers' LTX-2 DEFAULT_NEGATIVE_PROMPT) or --pipeline-arg negative_prompt=TEXT (empty "
                         "for none)")
    sub = opts.extra.get("transformer_subfolder")
    if sub:
        missing = [k for k in ("use_dynamic_shifting", "shift_terminal") if k not in (sched or {})]
        if missing:
            raise ValueError(f"reference mode with transformer_subfolder={sub} needs the scheduler overrides that go "
                             f"with it (missing {', '.join(missing)}): the checkpoint's own scheduler is set up for "
                             "its default (distilled) transformer, e.g. --pipeline-arg 'scheduler="
                             "{\"use_dynamic_shifting\": true, \"shift_terminal\": 0.1}'")
        return reference
    full = _has_full_transformer(opts.model, opts.extra.get("revision"))
    if full is None:
        raise ValueError(f"reference mode cannot tell whether {opts.model} ships a transformer_full/ folder (its default "
                         "transformer/ would then be a distilled one): it is not a local folder and not in the Hugging "
                         "Face cache at this revision. Download it first, or give the folder")
    if full:
        raise ValueError(f"{opts.model} has a transformer_full/ folder: its default transformer/ is the distilled "
                         "one (LTX-2.5-Diffusers), not the BF16 full model a reference needs. Pass "
                         "--pipeline-arg transformer_subfolder=transformer_full and the scheduler the full model "
                         "needs, e.g. --pipeline-arg 'scheduler={\"use_dynamic_shifting\": true, "
                         "\"shift_terminal\": 0.1}'")
    return reference


def _offline() -> bool:
    import os

    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")


def _cached_snapshot(model: str, revision: str | None = None) -> str | None:
    """The folder to load ``model`` from without calling the Hub: ``model`` itself when it is a local folder; for a hub
    id whose ``revision`` is a full commit sha (or any hub id with HF_HUB_OFFLINE set), its Hugging Face cache snapshot
    folder when it is there and holds a model_index.json. None otherwise (from_pretrained then resolves the hub id
    itself, online: a branch or tag name such as ``main`` may have moved since the cache last saw it)."""
    import os

    if os.path.isdir(str(model)):
        return str(model)
    if not (revision and COMMIT_SHA.match(str(revision))) and not _offline():
        return None  # unpinned, or a branch or tag name, online: the cache's ref may point at an older snapshot
    try:
        from huggingface_hub import snapshot_download

        folder = snapshot_download(str(model), revision=revision, local_files_only=True)
    except Exception:  # noqa: BLE001 - not installed, not a valid id, or not in the cache
        return None
    if not os.path.isfile(os.path.join(str(folder), "model_index.json")):
        return None  # not a Diffusers pipeline snapshot (e.g. only some files of it were fetched)
    return str(folder)


def _snapshot_commit(folder: str | None) -> str | None:
    """The commit a Hugging Face cache snapshot folder holds (its name, .../snapshots/<sha>), else None."""
    import os

    if not folder:
        return None
    name = os.path.basename(os.path.normpath(str(folder)))
    parent = os.path.basename(os.path.dirname(os.path.normpath(str(folder))))
    return name if parent == "snapshots" and COMMIT_SHA.match(name) else None


def _has_full_transformer(model: str, revision: str | None = None) -> bool | None:
    """Whether a Diffusers checkpoint ships a second, full transformer beside its default one (LTX-2.5-Diffusers:
    transformer/ is the distilled DiT, transformer_full/ the full one). A local folder is looked at directly; a hub
    id in its Hugging Face cache snapshot at ``revision`` (offline: local_files_only). None when neither is there.
    A snapshot cached without transformer_full/ reads as False: a re-run that checks the expected settings catches that."""
    import os

    if os.path.isdir(str(model)):
        return os.path.isdir(os.path.join(str(model), "transformer_full"))
    try:
        from huggingface_hub import snapshot_download

        folder = snapshot_download(str(model), revision=str(revision) if revision else None, local_files_only=True)
    except Exception:  # noqa: BLE001 - not installed, not a valid id, or not in the cache
        return None
    return os.path.isdir(os.path.join(str(folder), "transformer_full"))


def configure(pipe, opts: PipelineOptions, reference: bool) -> PipelineOptions:
    """Place a loaded pipeline (device, offload), set its attention backend and VAE tiling.

    Returns ``opts`` with the attention backend that was set (None: Diffusers' default, which
    ``DIFFUSERS_ATTN_BACKEND`` can change). Reference mode sets ``native`` explicitly.
    """
    if opts.offload == "model":
        pipe.enable_model_cpu_offload(device=opts.device)
    elif opts.offload == "sequential":
        pipe.enable_sequential_cpu_offload(device=opts.device)
    else:
        pipe.to(opts.device)
    attention = opts.attention or ("native" if reference else None)
    if attention:
        pipe.transformer.set_attention_backend(attention)
        connectors = getattr(pipe, "connectors", None)  # the text connectors attend too (LTX2Attention)
        if hasattr(connectors, "set_attention_backend"):
            connectors.set_attention_backend(attention)
    if opts.extra.get("vae_tiling"):
        pipe.vae.enable_tiling()
    return dataclasses.replace(opts, attention=attention)


def make_pipeline(opts: PipelineOptions) -> LTXDiffusers:
    reference = check_options(opts)  # before any import or load: a refused option costs nothing
    import diffusers
    import torch

    cls_name = opts.extra.get("pipeline_class") or CLASSES[opts.task]
    cls = getattr(diffusers, cls_name)
    revision = str(opts.extra["revision"]) if opts.extra.get("revision") else None
    source = _cached_snapshot(opts.model, revision)  # a local folder: no Hub call while loading
    kw = {"torch_dtype": getattr(torch, DTYPES[opts.dtype])}
    if revision and source is None:  # a hub id not in the cache: from_pretrained fetches that revision
        kw["revision"] = revision
    load_from = source or opts.model
    sub = opts.extra.get("transformer_subfolder")
    if sub:  # e.g. LTX-2.5-Diffusers' transformer_full/ (its model_index.json names the distilled transformer/)
        tkw = {k: v for k, v in kw.items()}
        kw["transformer"] = diffusers.LTX2VideoTransformer3DModel.from_pretrained(load_from, subfolder=str(sub),
                                                                                  **tkw)
    skipped = [c for c in UNUSED_COMPONENTS if c in (getattr(cls, "_optional_components", None) or ())]
    pipe = cls.from_pretrained(load_from, **kw, **{c: None for c in skipped})
    if opts.extra.get("scheduler"):
        pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config, **opts.extra["scheduler"])
    if opts.extra.get("default_negative_prompt"):
        from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT

        opts = dataclasses.replace(opts, extra={**opts.extra, "negative_prompt": DEFAULT_NEGATIVE_PROMPT})
    opts = configure(pipe, opts, reference)
    facts = {"pipeline": f"diffusers:{cls_name}", "diffusers_version": diffusers.__version__,
             "torch_version": torch.__version__, "reference": reference,
             "transformer_subfolder": sub or "transformer", "skipped_components": skipped,
             "loaded_from": "hf_cache_snapshot" if source and source != opts.model else
             ("folder" if source else "hub"),
             "revision": revision, "snapshot_commit": _snapshot_commit(source) if source != opts.model else None,
             "scheduler": {"class": type(pipe.scheduler).__name__,
                           **{k: pipe.scheduler.config.get(k) for k in ("use_dynamic_shifting", "shift_terminal",
                                                                       "shift") if k in pipe.scheduler.config}}}
    if str(opts.device).startswith("cuda") and torch.cuda.is_available():
        facts["gpu"] = torch.cuda.get_device_name(torch.device(opts.device))
    return LTXDiffusers(pipe, opts, torch, facts)


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _torch_dtype(t) -> str:
    return str(t.dtype).replace("torch.", "")


def _prev(res):
    """The new latent in a scheduler step's result (a tuple with return_dict=False, else an output object)."""
    return res[0] if isinstance(res, tuple) else res.prev_sample


def _with_prev(res, latent):
    if isinstance(res, tuple):
        return (latent, *res[1:])
    res.prev_sample = latent
    return res


def _active_attention() -> str | None:
    try:  # private in Diffusers: the backend used when a model sets none
        from diffusers.models.attention_dispatch import _AttentionBackendRegistry

        backend = _AttentionBackendRegistry.get_active_backend()[0]
        return getattr(backend, "value", str(backend))
    except Exception:  # noqa: BLE001
        return None


class LTXDiffusers:
    """The clip server's pipeline object for a loaded Diffusers LTX-2.x pipeline (``pipe``)."""

    def __init__(self, pipe, opts: PipelineOptions, torch_mod, facts: dict | None = None):
        self.pipe, self.opts, self.torch = pipe, opts, torch_mod
        self.facts = dict(facts or {})
        self.negative_prompt = opts.extra.get("negative_prompt")
        self.call_kwargs = {k[5:]: v for k, v in opts.extra.items() if k.startswith("call.")}
        clash = sorted(set(self.call_kwargs) & REQUEST_ARGS)
        if clash:
            raise ValueError(f"call.* may not set what the request sets: {clash}")
        self.clock: StageClock | None = None
        self._noise: dict = {}
        self._noise_seen: dict = {}
        self._install_hooks()

    # Hooks are installed once; they time on self.clock while a clip runs and call straight through otherwise.
    # Modules get torch forward hooks, not a replaced ``forward``: with --offload model, Diffusers re-applies
    # the offload hooks after every call (maybe_free_model_hooks), and removing them restores ``forward``.
    def _install_hooks(self) -> None:
        p, cur = self.pipe, lambda: self.clock
        hook_method(p, "encode_prompt", "text_encode", cur)
        if getattr(p, "connectors", None) is not None:
            self._hook_module(p.connectors, "text_encode")
        for attr, check in (("prepare_latents", self._check_video), ("prepare_audio_latents", self._check_audio)):
            if hasattr(p, attr):
                setattr(p, attr, self._capturing(timed(getattr(p, attr), "prepare_latents", cur), check))

        def loop_start(module, args):  # returns None: a pre-hook's return value would replace the inputs
            if self.clock is not None:
                self.clock.denoise_start()

        p.transformer.register_forward_pre_hook(loop_start)
        hook_method(p.vae, "decode", "vae_decode", cur)
        hook_method(p.video_processor, "postprocess_video", "postprocess", cur)
        if getattr(p, "audio_vae", None) is not None:
            hook_method(p.audio_vae, "decode", "audio_decode", cur)
        if getattr(p, "vocoder", None) is not None:
            self._hook_module(p.vocoder, "audio_decode")

    def _hook_module(self, module, stage: str) -> None:
        def pre(mod, args):
            if self.clock is not None:
                self.clock.begin()
                mod._ik_clock = self.clock  # close the stage on the clock it was opened on

        def post(mod, args, output):
            clock = mod.__dict__.pop("_ik_clock", None)
            if clock is not None:
                clock.end(stage)

        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)

    def _capturing(self, fn, check):
        def wrapper(*args, **kwargs):
            out = fn(*args, **kwargs)
            if self.clock is not None:
                check(out)
            return out

        return wrapper

    def _check_video(self, out) -> None:
        latents, mask = out if isinstance(out, tuple) else (out, None)
        ref = self._noise.get("video")
        if ref is None:
            return
        p = self.pipe
        ref = p._pack_latents(ref, p.transformer_spatial_patch_size, p.transformer_temporal_patch_size)
        ref = ref.to(latents.device, latents.dtype)
        if ref.shape != latents.shape:
            self._noise_seen["video"] = False
            return
        if mask is not None:  # i2v: the image's tokens (mask 1) are not noise
            free = mask[0] == 0
            latents, ref = latents[:, free], ref[:, free]
        self._noise_seen["video"] = bool(self.torch.equal(latents, ref))

    def _check_audio(self, out) -> None:
        ref = self._noise.get("audio")
        if ref is None:
            return
        ref = self.pipe._pack_audio_latents(ref).to(out.device, out.dtype)
        self._noise_seen["audio"] = ref.shape == out.shape and bool(self.torch.equal(out, ref))

    # ------------------------------------------------------------------ the ClipPipeline interface

    def describe(self) -> dict:
        dtype = _dtype_name(getattr(self.pipe.transformer, "dtype", self.opts.dtype))
        attention = self.opts.attention or _active_attention() or "default"
        neg = self.negative_prompt
        return {**self.facts, "model": self.opts.model, "task": self.opts.task, "dtype": dtype,
                "attention": attention, "device": self.opts.device, "offload": self.opts.offload,
                "negative_prompt_sha256": hashlib.sha256(str(neg).encode()).hexdigest() if neg is not None else None,
                "call_kwargs": self.call_kwargs}

    def noise_shapes(self, job: ClipJob) -> dict:
        p = self.pipe
        if job.task != self.opts.task:
            raise BadRequest(f"this server loaded the {self.opts.task} pipeline; start another for {job.task}")
        if float(job.fps) != int(job.fps):
            raise BadRequest(f"fps must be a whole number for this muxer, got {job.fps:g}")
        s, t = p.vae_spatial_compression_ratio, p.vae_temporal_compression_ratio
        if job.width % 32 or job.height % 32:
            raise BadRequest(f"width and height must be multiples of 32, got {job.width}x{job.height}")
        if (job.num_frames - 1) % t:
            raise BadRequest(f"num_frames must be {t}k+1, got {job.num_frames}")
        shapes = {"video": (1, p.transformer.config.in_channels, (job.num_frames - 1) // t + 1, job.height // s,
                            job.width // s)}
        avae = getattr(p, "audio_vae", None)
        if avae is not None:
            per_s = p.audio_sampling_rate / p.audio_hop_length / float(p.audio_vae_temporal_compression_ratio)
            length = round(job.num_frames / job.fps * per_s)
            shapes["audio"] = (1, avae.config.latent_channels, length,
                               avae.config.mel_bins // p.audio_vae_mel_compression_ratio)
        return shapes

    def _call_kwargs(self, job: ClipJob) -> dict:
        torch = self.torch
        kwargs = dict(prompt=job.prompt, negative_prompt=self.negative_prompt, height=job.height, width=job.width,
                      num_frames=job.num_frames, frame_rate=float(job.fps), num_inference_steps=job.steps,
                      guidance_scale=job.guidance_scale,
                      generator=torch.Generator(device="cpu").manual_seed(job.noise_seed),
                      output_type="np", return_dict=False, callback_on_step_end=self._on_step,
                      **self.call_kwargs)
        if job.task == "i2v":
            from PIL import Image

            kwargs["image"] = Image.open(io.BytesIO(job.image)).convert("RGB")
        return kwargs

    def generate(self, job: ClipJob, noise: dict, clock: StageClock) -> Generated:
        kwargs = self._call_kwargs(job)
        rec, shapes = stepdump.StepRecorder(stepdump.request_of(job)), self.noise_shapes(job)

        def record(stream, sched, step, model_output, timestep, sample, args, kw):
            res = step(model_output, timestep, sample, *args, **kw)
            i = sched.step_index - 1
            rec.step(stream, sched.sigmas[i], sched.sigmas[i + 1], self._unpack(stream, sample, shapes),
                     self._unpack(stream, model_output, shapes), self._unpack(stream, _prev(res), shapes))
            return res

        self.clock, self._noise, self._noise_seen = clock, noise, {}
        clock.set_steps(0)  # the step callback counts from here: a pipeline that never calls it ran 0 steps
        try:
            with self._scheduler_steps(record):
                video, audio = self.pipe(**kwargs)
        finally:
            self.clock = None
        frames = video[0]
        n, h, w = (int(d) for d in frames.shape[:3])
        wave, rate = None, None
        if audio is not None and job.audio is not False:
            wave = audio[0].float().cpu()
            rate = int(self.pipe.vocoder.config.output_sampling_rate)
        expected = set(noise)
        check = all(self._noise_seen.get(k) is True for k in expected) and set(self._noise_seen) >= expected
        return Generated(video=frames, width=w, height=h, frames=n, fps=float(job.fps), audio=wave,
                         audio_sample_rate=rate,
                         effective={"noise_check": check, "noise_checked": sorted(self._noise_seen),
                                    "generator": "cpu"}, step_dump=rec if rec.count else None)

    # ------------------------------------------------------------------ per-step hooks (dumps and replay)

    @contextmanager
    def _scheduler_steps(self, handler):
        """For one pipeline call, every scheduler.step(model_output, timestep, sample, ...) goes to
        ``handler(stream, scheduler, step, model_output, timestep, sample, args, kwargs)``; ``step`` is the real
        bound method. stream: "video" for pipe.scheduler, "audio" for any other instance (LTX-2's audio copy)."""
        video = getattr(self.pipe, "scheduler", None)
        if video is None:  # nothing to hook (a stand-in pipeline without a scheduler)
            yield
            return
        cls = type(video)
        had, orig = "step" in cls.__dict__, cls.step

        def step(sched, model_output, timestep, sample, *args, **kw):
            bound = orig.__get__(sched, cls)
            return handler("video" if sched is video else "audio", sched, bound, model_output, timestep, sample,
                           args, kw)

        cls.step = step
        try:
            yield
        finally:
            if had:
                cls.step = orig
            else:
                del cls.step

    def _unpack(self, stream: str, t, shapes: dict):
        """A packed latent in the noise layout of ``shapes``; as it is when its size is not that shape's (a pipeline
        that ran another shape than asked: the dump then shows that shape, which the checker refuses)."""
        p, shape = self.pipe, shapes[stream]
        if t.numel() != _numel(shape):
            return t
        if stream == "video":
            return p._unpack_latents(t, shape[2], shape[3], shape[4], p.transformer_spatial_patch_size,
                                     p.transformer_temporal_patch_size)
        return p._unpack_audio_latents(t, shape[2], shape[3])

    def _pack(self, stream: str, t):
        p = self.pipe
        if stream == "video":
            return p._pack_latents(t, p.transformer_spatial_patch_size, p.transformer_temporal_patch_size)
        return p._pack_audio_latents(t)

    def _set_attention(self, name) -> None:
        self.pipe.transformer.set_attention_backend(name)
        connectors = getattr(self.pipe, "connectors", None)
        if hasattr(connectors, "set_attention_backend"):
            connectors.set_attention_backend(name)

    def replay(self, job: ClipJob, dump, on_step, on_schedule=None, attention: str | None = None,
               decode: bool = True) -> dict:
        """Run ``job`` through this pipeline with ``dump``'s latents forced in (see the module docstring).

        ``on_step(i, stream, sigma, sigma_next, ref_out, chain)``: the reference's guided output from the dump's x
        at step i and the reference scheduler's update of the dump's output, both unpacked (on the device).
        ``on_schedule(sigmas)`` gets the reference's sigmas (N + 1 values) before the first update and may raise
        to stop the replay. ``attention``: another backend for this replay only (the floor run); ``decode=False``
        skips the decoders (frames None). Returns
        {"frames": decoded frames [F, H, W, 3] in [0, 1], "initial": {stream: the reference's own initial latent},
        "dtypes": {stream: {x, out, x_next: dtype name}}}."""
        if attention and not self.opts.attention:
            raise ValueError("a replay with another attention backend needs the pipeline's own backend set "
                             "(--attention, or reference mode), to switch back to it")
        p, shapes, n = self.pipe, self.noise_shapes(job), dump.n_steps
        initial, dtypes, counts = {}, {}, {}

        def forced(stream, i, like):
            key = f"step.{i}.{stream}.x" if i < n else f"final.{stream}"
            return self._pack(stream, dump.get(key).to(like.device)).to(like.dtype)

        def prepare(stream, fn):
            def wrapper(*a, **kw):
                out = fn(*a, **kw)
                lat, rest = (out[0], out[1:]) if isinstance(out, tuple) else (out, None)
                initial[stream] = self._unpack(stream, lat, shapes).float().cpu()
                x = forced(stream, 0, lat)
                return (x, *rest) if rest is not None else x
            return wrapper

        def step(stream, sched, real_step, model_output, timestep, sample, args, kw):
            i = counts.get(stream, 0)
            counts[stream] = i + 1
            if i == 0 and stream == "video" and on_schedule is not None:
                on_schedule([float(v) for v in sched.sigmas])
            if i >= n:  # the dump has fewer steps than the reference (on_schedule normally stopped it)
                return real_step(model_output, timestep, sample, *args, **kw)
            if stream not in dtypes:
                dtypes[stream] = {"x": _torch_dtype(sample), "out": _torch_dtype(model_output)}
            eng = self._pack(stream, dump.get(f"step.{i}.{stream}.out").to(sample.device)).to(model_output.dtype)
            res = real_step(eng, timestep, sample, *args, **kw)
            chain = _prev(res)
            dtypes[stream].setdefault("x_next", _torch_dtype(chain))
            j = sched.step_index - 1
            on_step(i, stream, float(sched.sigmas[j]), float(sched.sigmas[j + 1]),
                    self._unpack(stream, model_output, shapes), self._unpack(stream, chain, shapes))
            return _with_prev(res, forced(stream, i + 1, chain))

        saved = {a: getattr(p, a) for a in ("prepare_latents", "prepare_audio_latents") if hasattr(p, a)}
        for attr, stream in (("prepare_latents", "video"), ("prepare_audio_latents", "audio")):
            if attr in saved and stream in shapes:
                setattr(p, attr, prepare(stream, saved[attr]))
        if attention:
            self._set_attention(attention)
        try:
            kwargs = self._call_kwargs(job)
            if not decode:
                kwargs["output_type"] = "latent"
            with self._scheduler_steps(step):
                video, _audio = p(**kwargs)
        finally:
            for attr, fn in saved.items():
                setattr(p, attr, fn)
            if attention:
                self._set_attention(self.opts.attention)
        return {"frames": video[0] if decode else None, "initial": initial, "dtypes": dtypes}

    def _on_step(self, pipe, i, t, callback_kwargs):
        if self.clock is not None:
            self.clock.step()
        return {}

    def write(self, out: Generated, path, clock: StageClock) -> None:
        from diffusers.utils import encode_video

        encode_video(out.video, fps=int(out.fps), output_path=str(path), audio=out.audio,
                     audio_sample_rate=out.audio_sample_rate)
