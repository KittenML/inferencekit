"""Local speech scorers, isolated from inferencekit's lightweight environment.

This file is also the worker entry point and intentionally imports only stdlib
at module scope. Optional model libraries belong in the configured Python env.
Backend APIs: SYSTRAN/faster-whisper; sarulab-speech/UTMOSv2 (create_model,
predict); speechbrain.inference.classifiers.EncoderClassifier (encode_batch).
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

CONFIG_FORMAT = "inferencekit.quality.audio.scorers.v1"
DESCRIPTORS_FORMAT = "inferencekit.quality.audio.scorer-descriptors.v1"
WORKER_FORMAT = "inferencekit.quality.audio.scorer-worker.v1"
BACKENDS = {
    "faster-whisper": ("faster_whisper", ("faster-whisper", "ctranslate2", "av", "tokenizers", "onnxruntime", "numpy", "scipy", "huggingface-hub")),
    "utmosv2": ("utmosv2", ("utmosv2", "torch", "torchaudio", "torchvision", "timm", "transformers", "librosa", "numpy", "scipy")),
    "speechbrain-ecapa": ("speechbrain", ("speechbrain", "torch", "torchaudio", "hyperpyyaml", "numpy", "scipy", "soundfile")),
}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _file_hash(path):
    h = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def tree_identity(path):
    """Bind bytes and relative names, including HF snapshot symlink targets.

    Cache metadata, Git metadata and Python bytecode are not model/source files.
    Absolute paths never enter the public descriptor.
    """
    root = Path(path)
    if not root.exists() or not (root.is_file() or root.is_dir()):
        raise ValueError("local_checkpoint_missing")
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    entries = []
    for item in paths:
        relative = item.relative_to(root).as_posix() if root.is_dir() else "checkpoint"
        if any(p in {".git", ".cache", "__pycache__"} for p in Path(relative).parts) or item.suffix == ".pyc":
            continue
        if item.is_symlink() and item.is_dir():
            raise ValueError("checkpoint_directory_symlinks_unsupported")
        if item.is_file():
            digest, size = _file_hash(item)
            entries.append({"file": relative, "sha256": digest, "bytes": size})
    if not entries:
        raise ValueError("local_checkpoint_empty")
    return {"sha256": _digest(entries), "files": len(entries), "bytes": sum(x["bytes"] for x in entries)}


def _positive(value, name, maximum=None, zero=False):
    if type(value) is not int or value < (0 if zero else 1) or maximum is not None and value > maximum:
        raise ValueError("invalid_" + name)


def normalize_config(doc, base=None):
    """Strict config; resolve local paths before crossing the worker boundary."""
    if not isinstance(doc, dict) or set(doc) - {"format", "python", "threads", "seed", "batch_size", "timeout_s", "scorers"}:
        raise ValueError("invalid_scorer_config_fields")
    if doc.get("format") != CONFIG_FORMAT:
        raise ValueError("invalid_scorer_config_format")
    out = {"python": sys.executable, "threads": 1, "seed": 0, "batch_size": 4, "timeout_s": 3600, **doc}
    for key in ("threads", "seed", "batch_size"):
        _positive(out[key], key, maximum=2**32 - 1 if key == "seed" else None, zero=key == "seed")
    if type(out["timeout_s"]) not in (int, float) or not math.isfinite(out["timeout_s"]) or out["timeout_s"] <= 0:
        raise ValueError("invalid_timeout_s")
    if not isinstance(out.get("scorers"), dict) or not out["scorers"] or set(out["scorers"]) - {"asr", "asr_retry", "naturalness", "speaker"}:
        raise ValueError("invalid_scorer_roles")
    if "asr_retry" in out["scorers"] and "asr" not in out["scorers"]:
        raise ValueError("asr_retry_requires_asr")
    base = Path(base or Path.cwd())

    def local(value):
        if not isinstance(value, str) or not value or "://" in value:
            raise ValueError("local_path_required")
        return str((base / Path(value).expanduser()).absolute())

    out["python"] = local(out["python"])
    if not Path(out["python"]).is_file():
        raise ValueError("scorer_python_missing")
    normalized = {}
    for role, spec in out["scorers"].items():
        if not isinstance(spec, dict) or set(spec) - {"backend", "model", "settings", "assets"}:
            raise ValueError("invalid_scorer_fields")
        backend = {"asr": "faster-whisper", "asr_retry": "faster-whisper", "naturalness": "utmosv2", "speaker": "speechbrain-ecapa"}[role]
        if spec.get("backend") != backend or not isinstance(spec.get("settings", {}), dict):
            raise ValueError("invalid_scorer_backend_or_settings")
        if role.startswith("asr"):
            defaults = {"language": "en", "beam_size": 5, "compute_type": "int8", "vad_filter": role == "asr_retry", "condition_on_previous_text": False}
        elif role == "naturalness":
            defaults = {"model_name": "fusion_stage3", "window_seconds": 3, "fold": 0, "checkpoint_seed": 42,
                        "predict_dataset": "sarulab", "num_repetitions": 1}
        else:
            defaults = {}
        if set(spec.get("settings", {})) - set(defaults):
            raise ValueError("unknown_scorer_setting")
        settings = {**defaults, **spec.get("settings", {})}
        if role.startswith("asr"):
            _positive(settings["beam_size"], "beam_size")
            if (not isinstance(settings["language"], str) or not settings["language"].isalpha()
                    or settings["compute_type"] not in {"int8", "float32", "int8_float32"}
                    or type(settings["vad_filter"]) is not bool or settings["condition_on_previous_text"] is not False
                    or role == "asr_retry" and settings["vad_filter"] is not True):
                raise ValueError("invalid_asr_settings")
        if role == "naturalness":
            if settings["model_name"] != "fusion_stage3":
                raise ValueError("utmos_requires_fusion_stage3")
            seconds = settings["window_seconds"]
            if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 3:
                raise ValueError("utmos_window_seconds_must_be_positive_at_most_three")
            _positive(settings["fold"], "fold", 4, zero=True)
            _positive(settings["checkpoint_seed"], "checkpoint_seed", zero=True)
            _positive(settings["num_repetitions"], "num_repetitions")
            if settings["predict_dataset"] not in {"sarulab", "bvcc", "somos", "blizzard2008", "blizzard2009", "blizzard2010-EH1", "blizzard2010-EH2", "blizzard2010-ES1", "blizzard2010-ES3", "blizzard2011"}:
                raise ValueError("invalid_utmos_predict_dataset")
        assets = spec.get("assets", {})
        expected = {"facebook/wav2vec2-base", "tf_efficientnetv2_s.in21k_ft_in1k"} if role == "naturalness" else set()
        if not isinstance(assets, dict) or set(assets) != expected:
            raise ValueError("utmos_requires_local_backbone_assets" if expected else "unexpected_scorer_assets")
        normalized[role] = {"backend": backend, "model": local(spec.get("model")), "settings": settings,
                            "assets": {k: local(v) for k, v in assets.items()}}
    out["scorers"] = normalized
    return out


def describe_local(config):
    """Executed by the scorer Python, never by the core Python environment."""
    descriptors = {}
    package_cache, checkpoint_cache = {}, {}
    for role, spec in config["scorers"].items():
        backend = spec["backend"]
        if backend not in package_cache:
            module, packages = BACKENDS[backend]
            versions = {p: importlib.metadata.version(p) for p in packages}
            found = importlib.util.find_spec(module)
            if found is None or not found.origin:
                raise ValueError("scorer_backend_missing")
            source = tree_identity(Path(found.origin).parent)
            package_cache[backend] = (versions, source)
        versions, source = package_cache[backend]
        identities = {}
        for key, path in {"checkpoint": spec["model"], **spec["assets"]}.items():
            if path not in checkpoint_cache:
                checkpoint_cache[path] = tree_identity(path)
            identities[key] = checkpoint_cache[path]
        settings = {**spec["settings"], "device": "cpu", "threads": config["threads"], "seed": config["seed"],
                    "batch_size": config["batch_size"], "backend_source": source, "packages": versions,
                    "checkpoints": identities, "adapter_sha256": _file_hash(__file__)[0],
                    "python_version": sys.version.split()[0], "mono": "arithmetic_channel_mean",
                    "resample": "scipy.signal.resample_poly_kaiser5", "sample_rate": 16000}
        if role.startswith("asr"):
            settings.update(temperature=0, best_of=1, num_workers=1, local_files_only=True, batch_size=1)
        elif role == "naturalness":
            settings.update(remove_silent_section=False, num_workers=0, short_input="model_native_tiling",
                            windows="contiguous_original_pcm_frames_including_tail", spectrogram_views="model_native_seeded",
                            identical_windows="score_once", batch_order="sorted_pcm_hash_equal_length")
        else:
            settings.update(normalize_embeddings=False, similarity="cosine", batching="equal_length",
                            identical_pcm="embed_once", batch_order="sorted_pcm_hash", cosine_roundoff_atol=8 * sys.float_info.epsilon)
        descriptor = {"name": backend, "version": versions[BACKENDS[backend][1][0]],
                      "model": "fusion_stage3" if role == "naturalness" else "ecapa-tdnn" if role == "speaker" else "local-whisper",
                      "revision": "sha256:" + _digest(identities), "config": settings}
        if role in {"naturalness", "speaker"}:
            descriptor["scale"] = [1, 5] if role == "naturalness" else [-1, 1]
        descriptors[role] = descriptor
    return {"format": DESCRIPTORS_FORMAT, "scorers": descriptors}


def _worker_environment(config, home):
    # Never inherit API keys, user import paths, download caches or proxy settings.
    return {"PATH": os.defpath, "HOME": str(home), "TMPDIR": str(home), "LANG": "C.UTF-8",
            "PYTHONHASHSEED": str(config["seed"]), "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HOME": str(home / "hf"), "TORCH_HOME": str(home / "torch"),
            "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(config["threads"]),
            "MKL_NUM_THREADS": str(config["threads"]), "OPENBLAS_NUM_THREADS": str(config["threads"]),
            "NUMBA_NUM_THREADS": str(config["threads"]), "NUMBA_CACHE_DIR": str(home / "numba")}


def run_worker(config, action, payload=None):
    """Return only structured output; backend stdout/stderr may contain speech."""
    from inferencekit.registry import OpError

    with tempfile.TemporaryDirectory(prefix="inferencekit-audio-worker-") as directory:
        root = Path(directory)
        request = {"format": WORKER_FORMAT, "action": action, "config": config, "payload": payload}
        input_path, output_path = root / "request.json", root / "response.json"
        input_path.write_text(_json(request))
        try:
            process = subprocess.run([config["python"], "-s", str(Path(__file__).resolve()), "--worker", str(input_path), str(output_path)],
                                     cwd=root, env=_worker_environment(config, root), stdin=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=config["timeout_s"], check=False)
        except subprocess.TimeoutExpired:
            raise OpError("audio scorer worker timed out; no scores written") from None
        except OSError:
            raise OpError("cannot start configured audio scorer Python") from None
        try:
            result = json.loads(output_path.read_text())
            _json(result)
        except (OSError, ValueError, TypeError):
            raise OpError("audio scorer worker returned missing or invalid output; no scores written") from None
        if not isinstance(result, dict) or result.get("request_sha256") != _digest(request):
            raise OpError("audio scorer worker output does not bind its request")
        if process.returncode or result.get("error"):
            error = result.get("error", {})
            if not isinstance(error, dict):
                error = {}
            stage = error.get("stage", "worker")
            kind = error.get("type", "Error")
            # Never relay arbitrary model exception messages, filenames or text.
            if stage not in {"describe", "asr", "asr_retry", "naturalness", "speaker", "worker", "verify"}:
                stage = "worker"
            if not isinstance(kind, str) or not kind.isidentifier() or len(kind) > 80:
                kind = "Error"
            raise OpError(f"audio scorer {stage} failed ({kind}); check installed dependencies, local checkpoints and backend API; no scores written")
        if set(result) != {"request_sha256", "result"}:
            raise OpError("invalid audio scorer worker response")
        return result["result"]


def load_config(path):
    from inferencekit.quality import audio as qa
    from inferencekit.registry import OpError

    doc, _ = qa.read_json(path)
    try:
        return normalize_config(doc, Path(path).resolve().parent)
    except (ValueError, TypeError, OSError) as error:
        code = str(error) if isinstance(error, ValueError) else "invalid_config"
        raise OpError("audio scorer configuration: " + code) from None


def describe(config):
    from inferencekit.quality import audio as qa
    from inferencekit.registry import OpError

    result = run_worker(config, "describe")
    if (not isinstance(result, dict) or set(result) != {"format", "scorers"} or result["format"] != DESCRIPTORS_FORMAT
            or not isinstance(result["scorers"], dict) or set(result["scorers"]) != set(config["scorers"])):
        raise OpError("invalid audio scorer descriptors")
    qa.validate_policy({"acceptance": "exact", "scorers": result["scorers"]})
    return result


def windows(frames, rate, seconds):
    step = max(1, int(rate * seconds))
    return [(start, min(start + step, frames)) for start in range(0, frames, step)]


def _audio(raw, rate, channels):
    import numpy as np
    from scipy.signal import resample_poly

    values = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).astype(np.float32).mean(axis=1) / 32768.0
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        values = resample_poly(values, 16000 // divisor, rate // divisor, window=("kaiser", 5.0)).astype(np.float32)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("invalid_audio_samples")
    return values


def _seed(seed, torch=None):
    import numpy as np

    random.seed(seed)
    np.random.seed(seed % (2**32))
    if torch is not None:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)


def _scalar(value, low, high):
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError("nonfinite_or_out_of_scale_score")
    return value


def _cosine_value(value):
    # Double-precision dot/norm arithmetic can exceed unity by a few ULP for
    # identical vectors. Clamp only that roundoff, never arbitrary model values.
    value = _scalar(value, -1 - 8 * sys.float_info.epsilon, 1 + 8 * sys.float_info.epsilon)
    return max(-1.0, min(1.0, value))


def _asr(spec, config, jobs, rows, cache):
    from faster_whisper import WhisperModel

    settings = spec["settings"]
    key = (spec["model"], settings["compute_type"])
    if key not in cache:
        cache[key] = WhisperModel(spec["model"], device="cpu", compute_type=settings["compute_type"],
                                  cpu_threads=config["threads"], num_workers=1, local_files_only=True)
    dest = "retry_text" if spec is config["scorers"].get("asr_retry") else "text"
    for job, row in zip(jobs, rows):
        audio = _audio(Path(job["pcm_path"]).read_bytes(), job["sample_rate"], job["channels"])
        segments, _ = cache[key].transcribe(audio, language=settings["language"], beam_size=settings["beam_size"],
                                           best_of=1, temperature=0, vad_filter=settings["vad_filter"],
                                           condition_on_previous_text=False)
        texts = [segment.text for segment in segments]
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("invalid_asr_transcript")
        row.setdefault("asr", {})[dest] = "".join(texts).strip()


@contextlib.contextmanager
def _utmos_local_assets(assets):
    """Keep upstream architecture IDs while redirecting pretrained assets locally."""
    import timm
    from transformers import AutoFeatureExtractor, AutoModel
    from unittest.mock import patch

    extractor, model, create = AutoFeatureExtractor.from_pretrained, AutoModel.from_pretrained, timm.create_model

    def local_loader(original):
        def load(name, *args, **kwargs):
            if name != "facebook/wav2vec2-base":
                raise ValueError("unexpected_utmos_backbone")
            return original(assets[name], *args, **{**kwargs, "local_files_only": True})
        return load

    def local_timm(name, *args, **kwargs):
        if name != "tf_efficientnetv2_s.in21k_ft_in1k":
            raise ValueError("unexpected_utmos_backbone")
        return create(name, *args, **{**kwargs, "pretrained_cfg_overlay": {"file": assets[name]}})

    with patch.object(AutoFeatureExtractor, "from_pretrained", side_effect=local_loader(extractor)), \
            patch.object(AutoModel, "from_pretrained", side_effect=local_loader(model)), \
            patch.object(timm, "create_model", side_effect=local_timm):
        yield


def _naturalness(spec, config, jobs, rows):
    import numpy as np
    import torch
    import utmosv2

    settings = spec["settings"]
    torch.set_num_threads(config["threads"])
    _seed(config["seed"], torch)
    with _utmos_local_assets(spec["assets"]):
        model = utmosv2.create_model(pretrained=True, config=settings["model_name"], fold=settings["fold"],
                                     checkpoint_path=spec["model"], seed=settings["checkpoint_seed"], device="cpu")
    model.eval()
    # Equal-length batches avoid adding silence to short tail windows.
    unique = {}
    for job, row in zip(jobs, rows):
        raw = Path(job["pcm_path"]).read_bytes()
        row["naturalness"] = {"windows": []}
        for start, end in windows(job["sample_count"], job["sample_rate"], settings["window_seconds"]):
            window = {"start_frame": start, "end_frame": end}
            row["naturalness"]["windows"].append(window)
            chunk = raw[start * job["channels"] * 2:end * job["channels"] * 2]
            identity = _digest([hashlib.sha256(chunk).hexdigest(), job["sample_rate"], job["channels"]])
            if identity not in unique:
                unique[identity] = ([], _audio(chunk, job["sample_rate"], job["channels"]))
            unique[identity][0].append(window)
    batches = {}
    for identity in sorted(unique):
        targets, audio = unique[identity]
        batches.setdefault(len(audio), []).append((targets, audio))
    _seed(config["seed"], torch)  # create_model can reset the upstream RNG.
    for items in batches.values():
        for offset in range(0, len(items), config["batch_size"]):
            batch = items[offset:offset + config["batch_size"]]
            scores = model.predict(data=np.stack([x[1] for x in batch]), sr=16000, predict_dataset=settings["predict_dataset"],
                                   device="cpu", num_workers=0, batch_size=len(batch), num_repetitions=settings["num_repetitions"],
                                   remove_silent_section=False, verbose=False)
            scores = np.asarray(scores).reshape(-1)
            if len(scores) != len(batch):
                raise ValueError("missing_naturalness_scores")
            for (targets, _), score in zip(batch, scores):
                for window in targets:
                    window["score"] = _scalar(score, 1, 5)


def _speaker(spec, config, jobs, rows, scratch):
    import numpy as np
    import soundfile
    import torch
    from scipy.signal import resample_poly
    from speechbrain.inference.classifiers import EncoderClassifier

    torch.set_num_threads(config["threads"])
    _seed(config["seed"], torch)
    # Official ECAPA YAML refers to pretrained_path; bind it to the same local
    # directory so collect_files cannot fall back to a Hub repository.
    model = EncoderClassifier.from_hparams(source=spec["model"], savedir=str(scratch / "ecapa"),
                                           overrides={"pretrained_path": spec["model"]}, run_opts={"device": "cpu"})
    inputs, ids = {}, {}
    for job in jobs:
        if not job.get("reference_path"):
            continue  # missing fixture references remain missing quality observations
        reference_id = ("reference", job["reference_sha256"])
        if reference_id not in inputs:
            values, rate = soundfile.read(job["reference_path"], dtype="float32", always_2d=True)
            values = values.mean(axis=1)
            if rate != 16000:
                divisor = math.gcd(rate, 16000)
                values = resample_poly(values, 16000 // divisor, rate // divisor, window=("kaiser", 5.0)).astype(np.float32)
            if not values.size or not np.isfinite(values).all():
                raise ValueError("invalid_voice_reference")
            inputs[reference_id] = values
        identity = ("pcm", _digest([job["pcm_sha256"], job["sample_rate"], job["channels"]]))
        if identity not in inputs:
            inputs[identity] = _audio(Path(job["pcm_path"]).read_bytes(), job["sample_rate"], job["channels"])
        ids[(job["arm"], job["id"], job["round"])] = identity
    embeddings = {}
    groups = {}
    for identity, values in sorted(inputs.items()):
        groups.setdefault(len(values), []).append((identity, values))
    for group in groups.values():
        for offset in range(0, len(group), config["batch_size"]):
            batch = group[offset:offset + config["batch_size"]]
            waves = torch.from_numpy(np.stack([x[1] for x in batch]))
            with torch.inference_mode():
                values = model.encode_batch(waves, wav_lens=torch.ones(len(batch)), normalize=False).reshape(len(batch), -1)
            if not torch.isfinite(values).all() or torch.any(torch.linalg.vector_norm(values, dim=1) == 0):
                raise ValueError("invalid_speaker_embedding")
            embeddings.update(zip([x[0] for x in batch], values))
    for job, row in zip(jobs, rows):
        if not job.get("reference_path"):
            continue
        left = embeddings[ids[(job["arm"], job["id"], job["round"])]].double()
        right = embeddings[("reference", job["reference_sha256"])].double()
        value = torch.nn.functional.cosine_similarity(left, right, dim=0).item()
        row["speaker"] = {"score": _cosine_value(value), "reference_sha256": job["reference_sha256"]}


def _offline_audit(event, args):
    if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.sendto", "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"}:
        raise PermissionError("scorer_worker_is_offline")


def _worker(request_path, response_path):
    request = json.loads(Path(request_path).read_text())
    stage = "worker"
    result = {"request_sha256": _digest(request)}
    sys.addaudithook(_offline_audit)
    try:
        if request["format"] != WORKER_FORMAT:
            raise ValueError("invalid_worker_request")
        config = normalize_config(request["config"])
        stage = "describe"
        description = describe_local(config)
        if request["action"] == "describe":
            result["result"] = description
        elif request["action"] == "score":
            payload = request["payload"]
            if payload["descriptors"] != description:
                raise ValueError("scorer_identity_changed")
            jobs = payload["jobs"]
            stage = "verify"
            for job in jobs:
                digest, size = _file_hash(job["pcm_path"])
                if digest != job["pcm_sha256"] or size != job["sample_count"] * job["channels"] * 2:
                    raise ValueError("pcm_changed")
                if job.get("reference_path") and _file_hash(job["reference_path"])[0] != job["reference_sha256"]:
                    raise ValueError("reference_changed")
            rows = [{key: job[key] for key in ("arm", "id", "round", "seed", "pcm_sha256", "text_sha256")} for job in jobs]
            cache = {}
            for role, spec in config["scorers"].items():
                stage = role
                if role.startswith("asr"):
                    _asr(spec, config, jobs, rows, cache)
                elif role == "naturalness":
                    _naturalness(spec, config, jobs, rows)
                else:
                    _speaker(spec, config, jobs, rows, Path(response_path).parent)
            stage = "verify"
            if describe_local(config) != description:
                raise ValueError("scorer_identity_changed_during_execution")
            result["result"] = {"descriptors": description, "rows": rows}
        else:
            raise ValueError("invalid_worker_action")
        _json(result)
    except Exception as error:
        result = {"request_sha256": _digest(request), "error": {"stage": stage, "type": type(error).__name__}}
    with Path(response_path).open("x") as stream:
        stream.write(_json(result))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit("Use inferencekit quality audio-scorers or audio-score")
    raise SystemExit(_worker(sys.argv[2], sys.argv[3]))
