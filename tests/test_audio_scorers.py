"""Core tests need no ML packages; external worker and backend contracts are faked."""

import copy
import contextlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from inferencekit.ops import audio_scorers as op
from inferencekit.quality import audio as qa, audio_scorers as sc
from inferencekit.registry import OpError
from test_quality_audio import bench, fixture_file, invoke


def config(tmp_path, **updates):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "model.bin").write_bytes(b"public checkpoint fixture")
    return {"format": sc.CONFIG_FORMAT, "python": sys.executable,
            "scorers": {"asr": {"backend": "faster-whisper", "model": str(model)}}, **updates}


def test_config_is_local_strict_cpu_and_deterministic(tmp_path):
    parsed = sc.normalize_config(config(tmp_path))
    assert parsed["threads"] == 1
    assert parsed["scorers"]["asr"]["settings"] == {
        "language": "en", "beam_size": 5, "compute_type": "int8", "vad_filter": False, "condition_on_previous_text": False}
    for mutation in [lambda d: d.update(device="cuda"), lambda d: d.update(seed=2**32),
                     lambda d: d.pop("scorers"), lambda d: d.update(threads=True),
                     lambda d: d["scorers"]["asr"].update(revision="declared-pin"),
                     lambda d: d["scorers"]["asr"].update(model="https://example.com/model"),
                     lambda d: d["scorers"]["asr"].update(settings={"token": "secret"}),
                     lambda d: d.update(timeout_s=float("inf"))]:
        doc = config(tmp_path)
        mutation(doc)
        with pytest.raises(ValueError):
            sc.normalize_config(doc)


def test_retry_and_utmos_configuration(tmp_path):
    doc = config(tmp_path)
    doc["scorers"]["asr_retry"] = copy.deepcopy(doc["scorers"]["asr"])
    assert sc.normalize_config(doc)["scorers"]["asr_retry"]["settings"]["vad_filter"]
    doc["scorers"]["asr_retry"]["settings"] = {"vad_filter": False}
    with pytest.raises(ValueError, match="asr_settings"):
        sc.normalize_config(doc)
    doc = config(tmp_path)
    doc["scorers"]["naturalness"] = {"backend": "utmosv2", "model": "checkpoint.pth"}
    with pytest.raises(ValueError, match="local_backbone"):
        sc.normalize_config(doc)
    spec = doc["scorers"]["naturalness"]
    spec["assets"] = {"facebook/wav2vec2-base": "ssl", "tf_efficientnetv2_s.in21k_ft_in1k": "timm.pth"}
    assert sc.normalize_config(doc)["scorers"]["naturalness"]["settings"]["window_seconds"] == 3
    spec["settings"] = {"window_seconds": 10}
    with pytest.raises(ValueError, match="at_most_three"):
        sc.normalize_config(doc)


def test_tree_hash_binds_content_and_names_but_not_absolute_paths(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    for path in (left, right):
        path.mkdir()
        (path / "weights").write_bytes(b"same")
    assert sc.tree_identity(left) == sc.tree_identity(right)
    (right / "__pycache__").mkdir()
    (right / "__pycache__" / "cache.pyc").write_bytes(b"ignored")
    assert sc.tree_identity(left) == sc.tree_identity(right)
    (right / "weights").write_bytes(b"changed")
    assert sc.tree_identity(left) != sc.tree_identity(right)
    (right / "weights").write_bytes(b"same")
    (right / "weights").rename(right / "other")
    assert sc.tree_identity(left) != sc.tree_identity(right)
    # Hugging Face snapshots use file symlinks outside the snapshot directory.
    (right / "other").unlink()
    (right / "weights").symlink_to(left / "weights")
    assert sc.tree_identity(left) == sc.tree_identity(right)
    (left / "weights").write_bytes(b"updated")
    assert sc.tree_identity(right)["bytes"] == 7
    with pytest.raises(ValueError, match="missing"):
        sc.tree_identity(tmp_path / "absent")


def test_descriptor_hashes_installed_source_version_and_assets(tmp_path, monkeypatch):
    doc = sc.normalize_config(config(tmp_path))
    package = tmp_path / "backend"
    package.mkdir()
    (package / "__init__.py").write_text("# backend")
    monkeypatch.setattr(sc.importlib.metadata, "version", lambda name: "1.2.3")
    monkeypatch.setattr(sc.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(package / "__init__.py")))
    first = sc.describe_local(doc)
    descriptor = first["scorers"]["asr"]
    assert descriptor["version"] == "1.2.3"
    assert descriptor["revision"].startswith("sha256:")
    assert str(tmp_path) not in json.dumps(first)
    (package / "__init__.py").write_text("# modified backend")
    changed = sc.describe_local(doc)
    assert changed != first
    assert changed["scorers"]["asr"]["revision"] == descriptor["revision"]
    (Path(doc["scorers"]["asr"]["model"]) / "model.bin").write_bytes(b"other checkpoint")
    assert sc.describe_local(doc)["scorers"]["asr"]["revision"] != descriptor["revision"]


def test_windows_cover_every_frame_and_tail():
    intervals = sc.windows(101, 10, 3)
    assert intervals == [(0, 30), (30, 60), (60, 90), (90, 101)]
    assert sum(end - start for start, end in intervals) == 101
    assert sc.windows(1, 44100, 3) == [(0, 1)]


def test_cosine_only_clamps_float64_roundoff():
    assert sc._cosine_value(1.0000000000000004) == 1
    assert sc._cosine_value(-1.0000000000000004) == -1
    assert sc._cosine_value(0.8) == 0.8
    for value in [float("nan"), float("inf"), 1.0000000001, -1.0000000001]:
        with pytest.raises(ValueError, match="out_of_scale"):
            sc._cosine_value(value)


def test_child_environment_drops_tokens_paths_and_proxies(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "private")
    monkeypatch.setenv("OPENAI_API_KEY", "private")
    monkeypatch.setenv("PYTHONPATH", "/private")
    monkeypatch.setenv("HTTPS_PROXY", "private")
    environment = sc._worker_environment(sc.normalize_config(config(tmp_path)), tmp_path)
    assert not {"HF_TOKEN", "OPENAI_API_KEY", "PYTHONPATH", "HTTPS_PROXY"}.intersection(environment)
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["PYTHONHASHSEED"] == "0"
    for event in ["socket.connect", "socket.getaddrinfo", "socket.sendto", "subprocess.Popen", "os.system"]:
        with pytest.raises(PermissionError):
            sc._offline_audit(event, ())


def fake_python(tmp_path, mode="ok"):
    """An independent process exercises the actual transport and result checks."""
    executable = tmp_path / ("worker-" + mode)
    executable.write_text("#!" + sys.executable + "\n" + '''import hashlib,json,os,sys
from pathlib import Path
request=json.loads(Path(sys.argv[-2]).read_text())
digest=hashlib.sha256(json.dumps(request,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
mode=''' + repr(mode) + '''
if mode=='missing': sys.exit(0)
if mode=='invalid':
 Path(sys.argv[-1]).write_text('not JSON'); sys.exit(0)
if mode=='failure':
 Path(sys.argv[-1]).write_text(json.dumps({'request_sha256':digest,'error':{'stage':'asr','type':'RuntimeError','message':'private transcript secret'}}));sys.exit(1)
if mode=='tamper': digest='0'*64
description={'format':'inferencekit.quality.audio.scorer-descriptors.v1','scorers':{}}
for role,spec in request['config']['scorers'].items():
 description['scorers'][role]={'name':spec['backend'],'version':'test-1','model':'fake-local','revision':'sha256:'+'a'*64,'config':{'device':'cpu','vad_filter':role=='asr_retry','condition_on_previous_text':False}}
 if role in ['naturalness','speaker']: description['scorers'][role]['scale']=[1,5] if role=='naturalness' else [-1,1]
if request['action']=='describe': result=description
else:
 rows=[]
 for job in request['payload']['jobs']:
  row={k:job[k] for k in ['arm','id','round','seed','pcm_sha256','text_sha256']}
  row['asr']={'text':'The meeting starts tomorrow morning.'}
  if 'asr_retry' in description['scorers']: row['asr']['retry_text']=row['asr']['text']
  if 'naturalness' in description['scorers']: row['naturalness']={'windows':[{'start_frame':0,'end_frame':job['sample_count'],'score':4.0}]}
  rows.append(row)
 if mode=='missing_row': rows.pop()
 if mode=='missing_asr': rows[0].pop('asr')
 if mode=='missing_retry': rows[0]['asr'].pop('retry_text')
 if mode=='identity': rows[0]['pcm_sha256']='0'*64
 if mode=='duplicate': rows[-1]=rows[0]
 if mode=='listening': rows[0]['listening']={'ratings':[]}
 if mode=='window_gap': rows[0]['naturalness']['windows'][0]['start_frame']=1
 if mode=='nonfinite': rows[0]['naturalness']['windows'][0]['score']=float('nan')
 result={'descriptors':description,'rows':rows}
Path(sys.argv[-1]).write_text(json.dumps({'request_sha256':digest,'result':result}))
''')
    executable.chmod(0o700)
    return executable


@pytest.mark.parametrize("mode,match", [("missing", "missing or invalid"), ("invalid", "missing or invalid"),
                                       ("tamper", "bind its request"), ("failure", "asr failed")])
def test_worker_transport_rejects_missing_tampered_failed_outputs(tmp_path, mode, match):
    settings = sc.normalize_config(config(tmp_path, python=str(fake_python(tmp_path, mode))))
    with pytest.raises(OpError, match=match) as error:
        sc.run_worker(settings, "describe")
    assert "private" not in str(error.value)
    assert "secret" not in str(error.value)


def test_worker_timeout_is_clear_and_no_raw_stderr(tmp_path, monkeypatch):
    settings = sc.normalize_config(config(tmp_path))
    def timeout(*args, **kwargs):
        assert kwargs["stderr"] is subprocess.DEVNULL
        assert kwargs["stdout"] is subprocess.DEVNULL
        raise subprocess.TimeoutExpired(args[0], 1)
    monkeypatch.setattr(sc.subprocess, "run", timeout)
    with pytest.raises(OpError, match="timed out"):
        sc.run_worker(settings, "describe")


@pytest.mark.parametrize("value", [None, [], {"format": sc.DESCRIPTORS_FORMAT, "scorers": None},
                                     {"format": sc.DESCRIPTORS_FORMAT, "scorers": {"asr": {}}}])
def test_malformed_worker_descriptors_fail_cleanly(tmp_path, monkeypatch, value):
    monkeypatch.setattr(sc, "run_worker", lambda *args: value)
    with pytest.raises(OpError):
        sc.describe(sc.normalize_config(config(tmp_path)))


def scoring_setup(tmp_path, store, mode="ok", retry=False):
    doc = config(tmp_path, python=str(fake_python(tmp_path, mode)))
    doc["scorers"]["naturalness"] = {"backend": "utmosv2", "model": "utmos.pth",
        "assets": {"facebook/wav2vec2-base": "ssl", "tf_efficientnetv2_s.in21k_ft_in1k": "timm.pth"}}
    if retry:
        doc["scorers"]["asr_retry"] = copy.deepcopy(doc["scorers"]["asr"])
    config_path = tmp_path / "scorers.json"
    config_path.write_text(json.dumps(doc))
    description = op.quality_audio_scorers(None, str(config_path))
    fixtures = fixture_file(tmp_path)
    policy = {"acceptance": "task_noninferiority", "resamples": 100,
              "metrics": {"wer": {"margin": .02, "outlier_wer": 1}, "naturalness": {"margin": .1}},
              "scorers": description["scorers"]}
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    registered = invoke("quality_audio_policy", policy=str(policy_path), fixtures=str(fixtures))
    a, b = bench(store, fixtures, "baseline"), bench(store, fixtures, "candidate")
    return dict(baseline=a["evidence_id"], candidate=b["evidence_id"], fixtures=str(fixtures),
                protocol=registered["evidence_id"], config=str(config_path), out=str(tmp_path / "scores.json"))


def test_fake_worker_end_to_end_hash_bound_import_and_no_overwrite(tmp_path, store):
    args = scoring_setup(tmp_path, store, retry=True)
    result = op.quality_audio_score(None, **args)
    assert result["rows"] == 12
    assert result["listening"] == "NOT RUN"
    raw = Path(args["out"]).read_bytes()
    assert result["sha256"] == hashlib.sha256(raw).hexdigest()
    quality = invoke("quality_audio", **{k: v for k, v in args.items() if k in {"baseline", "candidate", "fixtures", "protocol"}}, scores=args["out"])
    assert quality["data"]["verdict"] == "PASS"
    with pytest.raises(OpError, match="already exists"):
        op.quality_audio_score(None, **args)
    assert Path(args["out"]).read_bytes() == raw


@pytest.mark.parametrize("mode", ["missing_row", "missing_asr", "identity", "duplicate", "listening", "window_gap", "nonfinite", "missing_retry"])
def test_score_rejects_incomplete_or_tampered_rows_atomically(tmp_path, store, mode):
    args = scoring_setup(tmp_path, store, mode, retry=True)
    with pytest.raises(OpError):
        op.quality_audio_score(None, **args)
    assert not Path(args["out"]).exists()


def test_score_rejects_changed_descriptors_before_execution(tmp_path, store, monkeypatch):
    args = scoring_setup(tmp_path, store)
    old = sc.describe
    def different(settings):
        description = old(settings)
        description["scorers"]["asr"]["version"] = "changed"
        return description
    monkeypatch.setattr(sc, "describe", different)
    with pytest.raises(OpError, match="differ from the registered"):
        op.quality_audio_score(None, **args)
    assert not Path(args["out"]).exists()


def test_faster_whisper_api_is_local_deterministic_and_model_reused(tmp_path, monkeypatch):
    calls = []
    class Whisper:
        def __init__(self, path, **kwargs):
            calls.append(("load", kwargs))
        def transcribe(self, audio, **kwargs):
            calls.append(("transcribe", kwargs))
            return iter([SimpleNamespace(text=" Hello"), SimpleNamespace(text=" world")]), None
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Whisper))
    monkeypatch.setattr(sc, "_audio", lambda *args: [0.0])
    settings = sc.normalize_config(config(tmp_path))
    spec = settings["scorers"]["asr"]
    settings["scorers"]["asr_retry"] = {**spec, "settings": {**spec["settings"], "vad_filter": True}}
    pcm = tmp_path / "input.pcm"
    pcm.write_bytes(b"\0\0")
    jobs = [{"pcm_path": str(pcm), "sample_rate": 16000, "channels": 1}] * 2
    rows, cache = [{}, {}], {}
    sc._asr(spec, settings, jobs, rows, cache)
    sc._asr(settings["scorers"]["asr_retry"], settings, jobs, rows, cache)
    assert [x[0] for x in calls].count("load") == 1
    assert calls[0][1]["local_files_only"] is True
    assert calls[0][1]["device"] == "cpu"
    assert all(row["asr"] == {"text": "Hello world", "retry_text": "Hello world"} for row in rows)
    assert all(c[1]["temperature"] == 0 and c[1]["condition_on_previous_text"] is False for c in calls if c[0] == "transcribe")
    assert calls[-1][1]["vad_filter"] is True


def test_utmos_backbone_api_redirects_to_bound_local_assets(monkeypatch):
    calls = []
    class Extractor:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls.append((name, kwargs))
    class Model(Extractor):
        pass
    def timm_create(name, **kwargs):
        calls.append((name, kwargs))
    timm = SimpleNamespace(create_model=timm_create)
    monkeypatch.setitem(sys.modules, "timm", timm)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoFeatureExtractor=Extractor, AutoModel=Model))
    assets = {"facebook/wav2vec2-base": "/local/ssl", "tf_efficientnetv2_s.in21k_ft_in1k": "/local/image.safetensors"}
    with sc._utmos_local_assets(assets):
        Extractor.from_pretrained("facebook/wav2vec2-base")
        Model.from_pretrained("facebook/wav2vec2-base")
        timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=True, num_classes=0)
        with pytest.raises(ValueError, match="unexpected"):
            Model.from_pretrained("unbound/model")
    assert calls[:2] == [("/local/ssl", {"local_files_only": True})] * 2
    assert calls[2][1]["pretrained_cfg_overlay"] == {"file": "/local/image.safetensors"}
    assert timm.create_model is timm_create


def test_utmos_predict_api_covers_tail_reuses_identical_windows_and_loads_once(tmp_path, monkeypatch):
    calls = []
    class Predictor:
        def eval(self):
            calls.append(("eval", {}))
        def predict(self, **kwargs):
            calls.append(("predict", kwargs))
            return [4.0] * len(kwargs["data"])
    def create(**kwargs):
        calls.append(("create", kwargs))
        return Predictor()
    monkeypatch.setitem(sys.modules, "utmosv2", SimpleNamespace(create_model=create))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda n: None))
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(stack=lambda seq: seq,
        asarray=lambda seq: SimpleNamespace(reshape=lambda shape: seq)))
    monkeypatch.setattr(sc, "_seed", lambda *args: None)
    monkeypatch.setattr(sc, "_utmos_local_assets", lambda assets: contextlib.nullcontext())
    monkeypatch.setattr(sc, "_audio", lambda raw, rate, channels: [0] * (len(raw) // 2))
    doc = config(tmp_path)
    doc["scorers"]["naturalness"] = {"backend": "utmosv2", "model": "utmos.pth",
        "assets": {"facebook/wav2vec2-base": "ssl", "tf_efficientnetv2_s.in21k_ft_in1k": "timm.pth"}}
    settings = sc.normalize_config(doc)
    pcm = tmp_path / "window.pcm"
    pcm.write_bytes(b"\x01\x00" * 35)
    jobs = [{"pcm_path": str(pcm), "sample_rate": 10, "channels": 1, "sample_count": 35}] * 2
    rows = [{}, {}]
    sc._naturalness(settings["scorers"]["naturalness"], settings, jobs, rows)
    assert rows[0] == rows[1]
    assert rows[0]["naturalness"] == {"windows": [
        {"start_frame": 0, "end_frame": 30, "score": 4.0}, {"start_frame": 30, "end_frame": 35, "score": 4.0}]}
    assert [name for name, _ in calls].count("create") == 1
    predictions = [kwargs for name, kwargs in calls if name == "predict"]
    assert len(predictions) == 2
    assert sum(len(kwargs["data"]) for kwargs in predictions) == 2  # four windows, two unique inputs
    assert all(kwargs["remove_silent_section"] is False and kwargs["device"] == "cpu" and kwargs["num_workers"] == 0
               and kwargs["num_repetitions"] == 1 for kwargs in predictions)
