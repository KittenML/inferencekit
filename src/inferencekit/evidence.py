"""Evidence records: build, validate, store, load.

A record is the unit of truth in inferencekit. Its id is ``"ev_" + sha256(canonical JSON
of the record without evidence_id)``, so:
  * the same id always means the same content (a report citing ev_ab12... cites
    exactly those numbers), and
  * tampering is detectable: ``verify`` recomputes the id.

Schemas: new records are written as ``inferencekit.evidence.v2`` (``schemas/evidence.v2.json``);
v1 records (``inferkit.evidence.v1``) still load and validate against their own schema.
v2 keeps the v1 envelope and adds serving workloads and the kinds of bench serve, gates and quality,
and clip workloads and the kinds of bench clip, gates clip and quality clip.

Storage is ``<runs>/<evidence_id>/evidence.json`` plus any artifact files (traces,
logs). ``<runs>`` is ``./runs`` unless ``--runs-dir`` or ``INFERENCEKIT_RUNS_DIR`` says
otherwise (``INFERKIT_RUNS_DIR``, the 0.1 name, is still read as a deprecated fallback).
A record made with a ``run_dir`` argument is also copied to ``<run_dir>/evidence/<evidence_id>.json``,
so one copy of a run directory (e.g. an rsync from a GPU host) carries every record of that run.

Secret safety: records never contain environment variable values, and commands are
stored only as digests. As a second line of defence, ``redact`` runs before hashing and
replaces (1) any text that contains the value of a secret-looking environment variable
(``*TOKEN*``, ``*KEY*``...) and (2) the value of any secret-looking *key* in a dict or in
JSON-like text, such as ``{"hf_token": "..."}`` typed into ``--engine-args``.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import shutil
import uuid
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

from inferencekit import __version__
from inferencekit.canonical import canonical_json, file_sha256, sha256_hex
from inferencekit.registry import Call, OpError, utc_now

# New records are written as v2; v1 records (written by early pre-release versions) still read and validate.
SCHEMA_V1 = "inferkit.evidence.v1"
SCHEMA_V2 = "inferencekit.evidence.v2"
SCHEMA_ID = SCHEMA_V2
SCHEMA_FILES = {SCHEMA_V1: "evidence.v1.json", SCHEMA_V2: "evidence.v2.json"}
KINDS_V1 = ("environment", "roofline", "ceiling", "benchmark", "nsys_capture", "nsys_timeline",
            "ncu_capture", "ncu_kernels", "correctness", "comparison")
KINDS = KINDS_V1 + ("serve_benchmark", "capacity", "gates", "quality_trajectory", "quality_task", "manifest")
# Video clips: bench clip, gates clip, quality clip, quality clip-steps.
KINDS += ("clip_benchmark", "clip_gates", "clip_quality", "clip_steps")
# Client-timed images and speech. Structural validity never establishes semantic quality.
KINDS += ("image_benchmark", "audio_benchmark", "rocprof_capture", "rocprof_summary")
KINDS += ("quality_audio",)
# identity.workload fields: the v1 four are always present (null when unknown); the v2 serving fields
# are written only when the operation knows them.
WORKLOAD_V1 = ("prompt_len", "gen_len", "batch", "seed")
WORKLOAD_V2 = ("mode", "bucket", "concurrency", "rate_rps", "duration_s", "prompt_set", "slo", "endpoint_kind",
               "prefix_cache", "request_set", "image_task_set", "concurrency_source")
# Clip workloads: the requested shape, the seed policy and the clip set; written only when known.
WORKLOAD_V2 += ("task", "width", "height", "num_frames", "fps", "steps", "guidance_scale", "seed_policy", "seeds",
                "clip_set", "audio")
CLAIM_CLASSES = ("observed", "derived", "inferred")

_default_root: Path | None = None


def set_default_root(path: str | os.PathLike | None) -> None:
    global _default_root
    _default_root = Path(path) if path else None


def default_root() -> Path:
    if _default_root is not None:
        return _default_root
    return Path(os.environ.get("INFERENCEKIT_RUNS_DIR") or os.environ.get("INFERKIT_RUNS_DIR") or "runs")


# --------------------------------------------------------------------------- building

def metric(name: str, value: float | int | None, unit: str, stat: str, n: int | None = None,
           ci95: Iterable[float | None] | None = None, subject: str | None = None) -> dict:
    """One number with its unit and the statistic it is (median, max, ceiling...)."""
    m = {"name": name, "value": _finite(value), "unit": unit, "stat": stat, "n": n,
         "ci95": [_finite(x) for x in ci95] if ci95 is not None else None}
    if subject is not None:
        m["subject"] = subject
    return m


def identity(gpu: dict | None = None, engine: dict | None = None, model: dict | None = None,
             workload: dict | None = None, cpu: dict | None = None) -> dict:
    """The four things a number depends on. Unknown fields are null, never guessed."""
    g = {"name": None, "count": None, "driver": None, "cuda": None, "clocks": None}
    e = {"name": None, "version": None, "commit": None, "flags": {}}
    m = {"id": None, "revision": None, "dtype": None, "quant": None}
    w = {k: None for k in WORKLOAD_V1}
    for base, extra in ((g, gpu), (e, engine), (m, model), (w, workload)):
        for k, v in (extra or {}).items():
            if k in base:
                base[k] = v
    for k in WORKLOAD_V2:  # serving workloads (v2): mode, concurrency grid, prompt set, SLO...
        if (workload or {}).get(k) is not None:
            w[k] = workload[k]
    result = {"gpu": g, "engine": e, "model": m, "workload": w}
    if cpu is not None:
        result["cpu"] = {k: cpu.get(k) for k in ("model", "architecture", "logical_cores", "physical_cores",
                                               "allowed_cpus", "topology_sha256")}
    return result


def build(call: Call, *, kind: str, claim_class: str, summary: str, identity_: dict | None = None,
          metrics: list[dict] | None = None, artifacts: list[dict] | None = None,
          missing: list[str] | None = None, limitations: list[str] | None = None,
          data: dict | None = None) -> dict:
    """Assemble a record (without its id). ``missing`` lists what could not be measured."""
    if kind not in KINDS:
        raise ValueError(f"unknown evidence kind {kind}")
    if claim_class not in CLAIM_CLASSES:
        raise ValueError(f"unknown claim class {claim_class}")
    missing = sorted(set(missing or []))
    body_data = dict(data or {})
    body_data.setdefault("request", call.request())
    return {
        "schema": SCHEMA_ID,
        "kind": kind,
        "claim_class": claim_class,
        "summary": summary[:600],
        "identity": identity_ or identity(),
        "provenance": {"tool": "inferencekit", "tool_version": __version__, "argv_digest": call.argv_digest(),
                       "started": call.started, "ended": utc_now()},
        "metrics": metrics or [],
        "artifacts": artifacts or [],
        "coverage": {"complete": not missing, "missing": missing},
        "limitations": _dedupe(limitations or []),
        "data": body_data,
    }


def artifact(path: str | os.PathLike, run_dir: str | os.PathLike | None, role: str) -> dict:
    """Describe a file by content hash. Paths inside the run dir are stored relative to it."""
    p = Path(path)
    sha, size = file_sha256(p)
    if run_dir is not None:
        try:
            rel = p.resolve().relative_to(Path(run_dir).resolve())
            return {"path": rel.as_posix(), "sha256": sha, "bytes": size, "role": role}
        except ValueError:
            pass
    # An input that lives elsewhere (e.g. a trace the user passed in): keep the path
    # relative to the working directory when possible, so home directories do not leak.
    try:
        shown = p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        shown = str(p)
    return {"path": shown, "sha256": sha, "bytes": size, "role": role}


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for s in items:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _finite(x):
    """NaN/inf cannot be stored in canonical JSON; they mean 'undefined', i.e. null."""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float):
        return _finite(value)
    return value


# --------------------------------------------------------------------------- secrets

# Environment variable NAMES that usually hold a secret (HF_TOKEN, OPENAI_API_KEY, AWS_SECRET_ACCESS_KEY...).
_SECRET_NAME = re.compile(r"TOKEN|SECRET|PASSW|(^|_)KEY($|_)|API_?KEY|ACCESS_?KEY|PRIVATE|CREDENTIAL|AUTH|COOKIE",
                          re.IGNORECASE)
# Dict / JSON KEYS that hold a secret. Stricter than _SECRET_NAME on purpose: the key must *end* in the
# secret word, so "hf_token" and "api_key" match but "max_num_batched_tokens" and "tokenizer" do not.
_SECRET_KEY = re.compile(r"(^|[_-])(token|api_?key|access_?key|secret(_?key)?|password|passwd|credentials?|auth)$",
                         re.IGNORECASE)
# "key": "value", 'key': 'value' (JSON or a Python dict repr) or key="value". Group 2 is the key; the
# lookbehind and the length cap keep matching linear on long runs of word characters (hex dumps, logs).
_KEY_VALUE = re.compile(r"""(["']?)(?<![A-Za-z0-9_.-])([A-Za-z0-9_.-]{1,64})\1(\s*[:=]\s*)(["'])(.*?)(?<!\\)\4""")
REDACTED = "<redacted>"


def looks_secret_name(name: str) -> bool:
    """True for environment variable names such as HF_TOKEN or MY_API_KEY."""
    return bool(_SECRET_NAME.search(name))


def looks_secret_key(key: str) -> bool:
    """True for dict / JSON keys such as hf_token, api_key or password."""
    return bool(_SECRET_KEY.search(str(key)))


def _secret_values() -> list[tuple[str, str]]:
    """(name, value) of environment variables whose *names* look secret. Values stay in memory only."""
    out = []
    for name, value in os.environ.items():
        if looks_secret_name(name) and value and len(value) >= 8:  # short values would redact common words
            out.append((name, value))
    return sorted(out, key=lambda nv: -len(nv[1]))


def _redact_key_value(m: re.Match) -> str:
    if not looks_secret_key(m.group(2)):
        return m.group(0)
    q = m.group(4)
    return f"{m.group(1)}{m.group(2)}{m.group(1)}{m.group(3)}{q}{REDACTED}{q}"


def redact_text(text: str) -> str:
    for name, value in _secret_values():
        if value in text:
            text = text.replace(value, f"<redacted:{name}>")
    # A secret typed straight into a JSON argument or echoed in a config dump: '"hf_token": "hf_..."'.
    text = _KEY_VALUE.sub(_redact_key_value, text)
    # Paths under the home directory name the user; write them as ~/... instead.
    home = os.path.expanduser("~").rstrip("/")
    if len(home) > 1 and home + "/" in text:
        text = text.replace(home + "/", "~/")
    return text


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: REDACTED if looks_secret_key(k) and isinstance(v, str) else redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


# --------------------------------------------------------------------------- ids and schema

def compute_id(record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "evidence_id"}
    return "ev_" + sha256_hex(canonical_json(body))


def finalize(record: dict) -> dict:
    """Clean numbers, redact secrets, then stamp the content hash as the id."""
    body = redact(_clean({k: v for k, v in record.items() if k != "evidence_id"}))
    eid = compute_id(body)
    ordered = {"schema": body["schema"], "evidence_id": eid}
    ordered.update({k: v for k, v in body.items() if k != "schema"})
    return ordered


def schema_path(schema_id: str = SCHEMA_ID) -> Path:
    """The schema file for ``schema_id``: the copy shipped in the wheel, or the repo copy in a checkout."""
    name = SCHEMA_FILES.get(schema_id)
    if name is None:
        raise ValueError(f"unknown evidence schema {schema_id!r}; known: {sorted(SCHEMA_FILES)}")
    try:
        packaged = resources.files("inferencekit").joinpath(f"schemas/{name}")
        if packaged.is_file():
            return Path(str(packaged))
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    return Path(__file__).resolve().parents[2] / "schemas" / name


_schema_cache: dict[str, dict] = {}


def load_schema(schema_id: str = SCHEMA_ID) -> dict:
    if schema_id not in _schema_cache:
        _schema_cache[schema_id] = json.loads(schema_path(schema_id).read_text())
    return _schema_cache[schema_id]


def validate(record: dict) -> list[str]:
    """Schema errors as readable strings (empty list = valid). The record's own ``schema`` picks v1 or v2."""
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - jsonschema is a declared dependency
        return ["jsonschema not installed: record not validated"]
    schema_id = record.get("schema") if isinstance(record, dict) else None
    if schema_id not in SCHEMA_FILES:
        return [f"schema: {schema_id!r} is not a known evidence schema ({', '.join(sorted(SCHEMA_FILES))})"]
    validator = jsonschema.Draft202012Validator(load_schema(schema_id))
    errors = []
    for err in sorted(validator.iter_errors(record), key=lambda e: list(e.path)):
        where = "/".join(str(p) for p in err.path) or "(root)"
        errors.append(f"{where}: {err.message}")
    return errors


def verify(record: dict, *, include_error_count: bool = False) -> dict:
    """Integrity check: does the id match the content, and does the record fit the schema?"""
    errors = validate(record)
    result = {"id_ok": compute_id(record) == record.get("evidence_id"), "schema_ok": not errors,
              "schema_errors": errors[:20]}
    if include_error_count and errors:
        result["schema_error_count"] = len(errors)
    return result


def integrity_problem(record: dict) -> str | None:
    """None when a record's id matches its content; otherwise a one-line reason.

    Operations that build a new record from stored ones (compare, correctness, roofline
    with a measured ceiling...) call this on each input. Without it, a hand-edited input
    would be laundered into a fresh record whose own id verifies.
    """
    if compute_id(record) != record.get("evidence_id"):
        return f"{record.get('evidence_id')} does not match its content (edited after capture?)"
    return None


# --------------------------------------------------------------------------- the store

def _copy_to_run_dir(rec: dict, text: str) -> None:
    """Also write the record to ``<run_dir>/evidence/<id>.json`` when it was made with a ``run_dir``
    argument (bench serve, bench rounds, gates, quality...), so the run directory alone carries its
    evidence: review build and ledger trailers work on a synced copy of ``runs/<run-id>/`` with no store.
    Best effort: the store copy is the one that counts."""
    args = ((rec.get("data") or {}).get("request") or {}).get("args")
    rd = args.get("run_dir") if isinstance(args, dict) else None
    if not isinstance(rd, str) or not rd:
        return
    try:
        run = Path(rd).expanduser()
        if not run.is_dir():
            return
        d = run / "evidence"
        d.mkdir(exist_ok=True)
        (d / f"{rec['evidence_id']}.json").write_text(text)
    except OSError:
        return


class Store:
    """Files under ``<root>/<evidence_id>/``. Staging dirs hold artifacts until the id is known."""

    def __init__(self, root: str | os.PathLike | None = None):
        self.root = Path(root) if root else default_root()

    def staging(self) -> Path:
        """A scratch dir for an operation's artifacts; ``save`` moves it to the run dir."""
        d = self.root / ".staging" / uuid.uuid4().hex
        d.mkdir(parents=True, exist_ok=True)
        return d

    @contextlib.contextmanager
    def staging_dir(self):
        """``with store.staging_dir() as staging:`` -- a staging dir that is deleted if the block fails.

        On success ``save`` has already moved its files into the run dir. On any error
        (including Ctrl-C) the directory and everything in it is removed, so raw program
        output that was never redacted does not stay behind in ``runs/.staging``.
        """
        d = self.staging()
        try:
            yield d
        except BaseException:
            shutil.rmtree(d, ignore_errors=True)
            raise

    def run_dir(self, evidence_id: str) -> Path:
        return self.root / evidence_id

    def save(self, record: dict, staging: Path | None = None) -> dict:
        rec = finalize(record)
        errors = validate(rec)
        if errors:
            raise OpError("evidence record failed schema validation: " + "; ".join(errors[:5]))
        target = self.run_dir(rec["evidence_id"])
        target.mkdir(parents=True, exist_ok=True)
        if staging is not None and staging.exists():
            for item in staging.iterdir():
                dest = target / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))
            shutil.rmtree(staging, ignore_errors=True)
        text = json.dumps(rec, indent=2, ensure_ascii=False) + "\n"
        (target / "evidence.json").write_text(text)
        _copy_to_run_dir(rec, text)
        return rec

    def paths(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(self.root.glob("ev_*/evidence.json"))

    def load(self, ref: str) -> dict:
        """Load by full id, unique id prefix (like git), or a path to an evidence.json."""
        p = Path(ref)
        if p.suffix == ".json" and p.is_file():
            return json.loads(p.read_text())
        if p.is_dir() and (p / "evidence.json").is_file():
            return json.loads((p / "evidence.json").read_text())
        ref = ref if ref.startswith("ev_") else "ev_" + ref
        exact = self.run_dir(ref) / "evidence.json"
        if exact.is_file():
            return json.loads(exact.read_text())
        if len(ref) < 3 + 6:
            raise OpError(f"evidence prefix '{ref}' is too short (use at least 6 hex characters)")
        matches = [q for q in self.paths() if q.parent.name.startswith(ref)]
        if not matches:
            raise OpError(f"no evidence matching '{ref}' under {self.root}")
        if len(matches) > 1:
            raise OpError(f"evidence prefix '{ref}' is ambiguous ({len(matches)} matches)")
        return json.loads(matches[0].read_text())

    def records(self) -> list[dict]:
        out = []
        for q in self.paths():
            try:
                out.append(json.loads(q.read_text()))
            except (OSError, ValueError):
                continue
        return out
