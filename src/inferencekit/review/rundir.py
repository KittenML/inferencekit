"""Read a run directory (``runs/<run-id>/``) into the rows and columns of the review page.

Layout read:

    inputs/<bucket>/<id>.json | .txt | .wav      exact request bodies, texts or audio
    outputs/<engine>/<bucket>/<id>.json | .txt | .wav
    metrics/<engine>.json | .jsonl               speed + quality per input and per bucket
    manifest.json, intake.yaml                   optional: engine order, roles, modality

Video clips (bench clip's layout) use the round as the bucket: ``outputs/<engine>/r<k>/<id>.mp4``,
the exact request body each engine was sent at ``inputs/<engine>/r<k>/<id>.json`` (rows come from
those when there is no ``inputs/r<k>/<id>.json``), ``metrics/<engine>.clip.json`` for the speed
summary, and per-clip times and self-reported stages from the bench clip record's ``requests`` rows.

Everything is read defensively, by field presence: a missing file is a NOT RUN cell, an
unreadable one is a FAILED cell with the reason, and unknown fields are ignored. The
``metrics/<engine>.json`` fields this module understands are listed in ``load_metrics``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

BUCKET_ORDER = ("very_short", "short", "medium", "long")
AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg", ".opus")
VIDEO_EXT = (".mp4", ".webm", ".mov", ".mkv", ".m4v")
IMAGE_EXT = (".png",)
MODALITIES = ("llm", "tts", "stt", "video", "image")
ROUND_DIR = re.compile(r"^r\d+$")  # bench clip: outputs/<label>/r<round>/<id>.<fmt>
# What a clip request body says about the clip (bench clip's inputs/<label>/r<k>/<id>.json), shown with the input.
CLIP_FIELDS = ("task", "width", "height", "num_frames", "fps", "steps", "guidance_scale", "seed", "noise_seed",
               "audio", "negative_prompt", "sampler", "shift")
GROUND_TRUTH_SUFFIXES = ("gt.txt", "ref.txt", "truth.txt", "transcript.txt")
REFERENCE_DEFAULT = "reference"
_NATURAL = re.compile(r"(\d+)")


def natural_key(s: str):
    return [int(p) if p.isdigit() else p.lower() for p in _NATURAL.split(s)]


def bucket_key(b: str):
    # Known buckets in their order, then the rest in natural order (bench clip's rounds: r2 before r10).
    return (BUCKET_ORDER.index(b), []) if b in BUCKET_ORDER else (len(BUCKET_ORDER), natural_key(b))


@dataclass
class Item:
    """One input (a row): what was sent, and the ground truth when there is one."""

    id: str
    bucket: str
    text: str | None = None            # the prompt / the text to speak, flattened for display
    messages: list | None = None       # chat messages, when the input is a chat request body
    audio: str | None = None           # run-dir-relative path of an input audio file
    ground_truth: str | None = None    # expected transcript (STT) or the text itself (TTS)
    clip: dict | None = None           # video: what the clip request asked for (task, shape, steps, seed, image)
    files: list[str] = field(default_factory=list)


@dataclass
class Cell:
    """One engine's output for one input."""

    engine: str
    status: str = "not_run"            # ok | not_run | failed
    reason: str = ""
    text: str | None = None
    token_ids: list | None = None
    pieces: list[str] | None = None    # per-token text, when the output carries it (logprobs)
    audio: str | None = None
    video: str | None = None           # run-dir-relative path of the generated clip
    image: str | None = None
    pcm: str | None = None             # raw PCM needs verified index metadata before playback
    clip_row: dict | None = None       # video: bench clip's row for this clip (client time, stages, effective)
    transcript: str | None = None      # ASR of generated audio (TTS) or the transcript itself (STT)
    finish_reason: str | None = None
    reasoning: str | None = None       # a reasoning model's separate thinking text, when the output has it
    metrics: dict = field(default_factory=dict)
    files: list[str] = field(default_factory=list)


@dataclass
class EngineInfo:
    label: str
    role: str | None = None
    metrics: dict = field(default_factory=dict)        # top-level fields of metrics/<engine>.json
    per_input: dict = field(default_factory=dict)      # input id -> {metric: value}
    per_bucket: dict = field(default_factory=dict)     # bucket -> {metric: value}
    metrics_file: str | None = None
    # From evidence records about this label (run-dir evidence/ or --evidence):
    ev_quality: list = field(default_factory=list)     # [{name, verdict, value, unit, evidence}]
    ev_gates: dict | None = None                       # {verdict, violations, evidence}
    capacity: dict | None = None                       # {n_star, reason, level_kind, slo, usd_per_1m, price, ...}
    evidence: list = field(default_factory=list)       # evidence ids that mention this label
    clip_rows: dict = field(default_factory=dict)      # video: "r<k>/<id>" -> bench clip request row
    stages: dict = field(default_factory=dict)         # video: stage -> median ms over clips (self-reported)
    media_rows: dict = field(default_factory=dict)
    media_kind: str | None = None
    metric_classes: dict = field(default_factory=dict)


@dataclass
class Run:
    root: Path
    modality: str
    engines: list[EngineInfo]
    reference: str | None
    items: list[Item]
    cells: dict[tuple[str, str], Cell]
    warnings: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)   # run-relative paths read to build the page
    evidence_attached: int = 0                         # evidence records read (evidence/, --evidence, the store)

    def buckets(self) -> list[str]:
        return sorted({i.bucket for i in self.items}, key=bucket_key)

    def engine(self, label: str) -> EngineInfo | None:
        return next((e for e in self.engines if e.label == label), None)

    def key(self, item: Item) -> str:
        return f"{item.bucket}/{item.id}"


# --------------------------------------------------------------------------- helpers

def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _split_name(name: str) -> tuple[str, str]:
    """'short-0007.gt.txt' -> ('short-0007', 'gt.txt'). A clip ('scene.01.mp4' -> ('scene.01', 'mp4')) keeps
    every dot but the last: bench clip ids may contain dots, and a clip file is always <id>.<video ext>."""
    if Path(name).suffix.lower() in VIDEO_EXT + IMAGE_EXT + (".pcm",):
        return name[:-len(Path(name).suffix)], Path(name).suffix.lower()[1:]
    stem, _, suffix = name.partition(".")
    return stem, suffix.lower()


def flatten_messages(messages) -> str:
    parts = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, list):  # OpenAI content parts
            content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
        parts.append(f"{m.get('role', '?')}: {content if content is not None else ''}")
    return "\n\n".join(parts)


def token_pieces(obj: dict) -> list[str] | None:
    """Per-token text of an output, from explicit fields or OpenAI-style logprobs."""
    for key in ("token_texts", "tokens"):
        v = obj.get(key)
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            return v
    lp = obj.get("logprobs")
    if isinstance(lp, dict):
        content = lp.get("content")
        if isinstance(content, list) and content and all(isinstance(c, dict) and "token" in c for c in content):
            return [str(c["token"]) for c in content]
        toks = lp.get("tokens")
        if isinstance(toks, list) and toks and all(isinstance(x, str) for x in toks):
            return toks
    return None


def _output_text(obj: dict) -> str | None:
    if isinstance(obj.get("text"), str):
        return obj["text"]
    choices = obj.get("choices")  # a raw OpenAI response saved as-is
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        c = choices[0]
        if isinstance(c.get("message"), dict) and isinstance(c["message"].get("content"), str):
            return c["message"]["content"]
        if isinstance(c.get("text"), str):
            return c["text"]
    return None


def _ids(value) -> list | None:
    return value if isinstance(value, list) and all(isinstance(x, int) for x in value) else None


def detect_modality(root: Path, items: list[Item], cells: dict) -> str:
    for name in ("manifest.json", "run.json"):
        p = root / name
        if p.is_file():
            try:
                m = _read_json(p)
                mod = m.get("modality") or (m.get("model") or {}).get("modality")
                if mod in MODALITIES:
                    return mod
            except (OSError, ValueError, AttributeError):
                pass
    intake = root / "intake.yaml"
    if intake.is_file():  # one top-level key; no YAML parser needed for it
        found = re.search(r"(?m)^modality:\s*['\"]?(llm|tts|stt|video|image)\b", intake.read_text(errors="replace"))
        if found:
            return found.group(1)
    if any(c.video for c in cells.values()):
        return "video"
    if any(c.image for c in cells.values()) or any((root / "metrics").glob("*.image.json")):
        return "image"
    if any(c.audio or c.pcm for c in cells.values()) or any((root / "metrics").glob("*.audio.json")):
        return "tts"
    if any(i.audio for i in items):
        return "stt"
    return "llm"


# --------------------------------------------------------------------------- loading

def load_prompt_set(path: Path, items: dict[str, Item], run: Run) -> None:
    """Fill rows from a prompt-set JSONL ({"id", "bucket", "messages" | "prompt" | "text"} per line)."""
    from inferencekit import prompts as sets

    try:
        lines = sets.read_text(sets.existing(path)).splitlines()  # .jsonl or .jsonl.gz
    except (OSError, ValueError) as e:  # missing, corrupt .jsonl.gz, or not UTF-8
        run.warnings.append(f"prompt set {path.name}: unreadable ({e})")
        return
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            run.warnings.append(f"prompt set {path.name}:{n}: not JSON")
            continue
        if not isinstance(obj, dict) or not obj.get("id"):
            continue
        # Unbucketed media fixtures apply to every observed round of that id.
        targets = [i for i in items.values() if i.id == str(obj["id"]) and ROUND_DIR.match(i.bucket)] \
            if not obj.get("bucket") else []
        if not targets:
            targets = [items.setdefault(f"{obj.get('bucket') or 'unbucketed'}/{obj['id']}",
                                       Item(id=str(obj["id"]), bucket=str(obj.get("bucket") or "unbucketed")))]
        for item in targets:
            if item.text is None:
                if isinstance(obj.get("messages"), list):
                    item.messages = obj["messages"]
                    item.text = flatten_messages(obj["messages"])
                elif isinstance(obj.get("prompt") or obj.get("text"), str):
                    item.text = obj.get("prompt") or obj.get("text")
            if item.ground_truth is None and isinstance(obj.get("ground_truth"), str):
                item.ground_truth = obj["ground_truth"]


def load_inputs(root: Path, run: Run) -> dict[str, Item]:
    items: dict[str, Item] = {}
    base = root / "inputs"
    if not base.is_dir():
        return items
    # bench serve also keeps each engine's exact request bodies at inputs/<label>/<bucket>/<id>.json (and
    # inputs/<label>/load-<level>/...); those are per-engine copies, not inputs: only inputs/<bucket>/<id>
    # (the shared path) and inputs/<id> are rows.
    out_dir = root / "outputs"
    labels = {d.name for d in out_dir.iterdir() if d.is_dir()} if out_dir.is_dir() else set()
    for ending in (".clip.json", ".image.json", ".audio.json"):
        labels.update(p.name[:-len(ending)] for p in (root / "metrics").glob(f"*{ending}"))
    clip_copies: dict[str, list[tuple[str, dict]]] = {}  # "r<k>/<id>" -> [(path, clip request body)] per engine
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        rel = path.relative_to(base)
        if (len(rel.parts) == 3 and rel.parts[0] in labels and ROUND_DIR.match(rel.parts[1])
                and path.suffix == ".json"):
            body = _clip_body(path)  # bench clip: the exact body this engine was sent for this clip and round
            if body is not None:  # <id>.json, and the id may contain dots
                clip_copies.setdefault(f"{rel.parts[1]}/{path.name[:-len('.json')]}", []).append(
                    (path.relative_to(root).as_posix(), body))
            continue
        if len(rel.parts) > 2 or (len(rel.parts) == 2 and rel.parts[0] in labels):
            continue
        bucket = rel.parts[0] if len(rel.parts) > 1 else "unbucketed"
        stem, suffix = _split_name(path.name)
        if ROUND_DIR.match(bucket) and path.suffix == ".json":  # a clip request r<k>/<id>.json: the id may hold dots
            stem, suffix = path.name[:-len(".json")], "json"
        item = items.setdefault(f"{bucket}/{stem}", Item(id=stem, bucket=bucket))
        rel_root = path.relative_to(root).as_posix()
        item.files.append(rel_root)
        run.sources.append(rel_root)
        try:
            if suffix in GROUND_TRUTH_SUFFIXES:
                item.ground_truth = path.read_text(encoding="utf-8", errors="replace").strip()
            elif suffix == "json":
                body = _read_json(path)
                if isinstance(body, dict):
                    if isinstance(body.get("messages"), list):
                        item.messages = body["messages"]
                        item.text = flatten_messages(body["messages"])
                    elif isinstance(body.get("prompt"), str):
                        item.text = body["prompt"]
                    elif isinstance(body.get("text"), str) or isinstance(body.get("input"), str):
                        item.text = body.get("text") if isinstance(body.get("text"), str) else body["input"]
                    if "num_frames" in body or ("width" in body and "height" in body):
                        item.clip = clip_spec(body)
                    for k in ("ground_truth", "reference_text", "transcript"):
                        if isinstance(body.get(k), str):
                            item.ground_truth = body[k]
                            break
            elif suffix == "txt":
                item.text = path.read_text(encoding="utf-8", errors="replace")
            elif path.suffix.lower() in AUDIO_EXT and not item.audio:
                item.audio = rel_root
        except (OSError, ValueError) as e:
            run.warnings.append(f"{rel_root}: unreadable ({e})")
    for key, copies in sorted(clip_copies.items()):
        if key in items:  # the shared path inputs/r<k>/<id>.json wins over the per-engine copies
            continue
        bucket, _, cid = key.partition("/")
        first_path, body = copies[0]
        item = items[key] = Item(id=cid, bucket=bucket, text=body["prompt"] if isinstance(body.get("prompt"), str)
                                 else None, clip=clip_spec(body), files=[p for p, _ in copies])
        run.sources += item.files
        differ = [p for p, b in copies[1:] if clip_spec(b) != item.clip or b.get("prompt") != body.get("prompt")]
        if differ:
            run.warnings.append(f"{key}: the engines were not sent the same clip request ({first_path} differs from "
                                f"{', '.join(differ[:3])}): their clips are not comparable")
    for item in items.values():
        if item.audio and item.ground_truth is None and item.text is not None and item.messages is None:
            item.ground_truth = item.text.strip()  # STT: the .txt next to the audio is its transcript
    return items


def _clip_body(path: Path) -> dict | None:
    """A saved video/image request body, or None for e.g. a chat request copy."""
    try:
        body = _read_json(path)
    except (OSError, ValueError):
        return None
    return body if isinstance(body, dict) and ("num_frames" in body or
                                              ("width" in body and "height" in body)) else None


def clip_spec(body: dict) -> dict:
    """The fields of a clip request shown with the input (the image as its sha256 and size, never its bytes)."""
    spec = {k: body[k] for k in CLIP_FIELDS if k in body and isinstance(body[k], (str, int, float, bool))}
    image = body.get("image")
    if isinstance(image, dict) and isinstance(image.get("sha256"), str):
        spec["image_sha256"] = image["sha256"]
        if isinstance(image.get("bytes"), int):
            spec["image_bytes"] = image["bytes"]
    return spec


def load_metrics(root: Path, label: str, run: Run) -> EngineInfo:
    """metrics/<label>.json or .jsonl.

    Understood fields (all optional):
      per_input | inputs   {id: {metric: value}} or [{"id": ..., metric: value}]
      per_bucket | buckets {bucket: {metric: value}}
      quality              {name: verdict | {"verdict", "value", "unit"}} or a list of those
      gates                "PASS" or {"verdict", "violations"}
      cost                 {"usd_per_1m_output_tokens" | "usd_per_1m": x, "gpu_price_usd_h": y} or a number
      ceiling_tok_s, pct_of_ceiling, role, evidence (list of ids)
    A value is a number or {"value"|"median": x, "ci95": [lo, hi]}. A .jsonl file holds one
    {"id": ..., metric: value} line per input. Video: ``metrics/<label>.clip.json`` (bench clip:
    {"speed": {"all": {clip_ms, clips_per_hour}, "stages_self_reported": ...}, "rounds": [...]}) is
    read as well, for the summary row and the per-round values.
    """
    info = EngineInfo(label=label)
    load_clip_metrics(root, label, info, run)
    jpath, lpath = root / "metrics" / f"{label}.json", root / "metrics" / f"{label}.jsonl"
    try:
        if jpath.is_file():
            data = _read_json(jpath)
            info.metrics_file = jpath.relative_to(root).as_posix()
            if not isinstance(data, dict):
                raise ValueError("not a JSON object")
            per_input = data.get("per_input", data.get("inputs"))
            per_bucket = data.get("per_bucket", data.get("buckets"))
            speed = data.get("speed")  # bench serve: {"speed": {"per_bucket": ..., "per_input_median": ...}}
            if isinstance(speed, dict):
                per_input = per_input if per_input is not None else speed.get("per_input_median",
                                                                              speed.get("per_input"))
                per_bucket = per_bucket if per_bucket is not None else speed.get("per_bucket")
            info.metrics = {k: v for k, v in data.items()
                            if k not in ("per_input", "inputs", "per_bucket", "buckets", "speed")}
        elif lpath.is_file():
            info.metrics_file = lpath.relative_to(root).as_posix()
            per_input = [json.loads(line) for line in lpath.read_text(encoding="utf-8").splitlines() if line.strip()]
            per_bucket = None
        else:
            return info
    except (OSError, ValueError) as e:
        run.warnings.append(f"metrics/{label}: unreadable ({e})")
        return info
    run.sources.append(info.metrics_file)
    clip_bucket = dict(info.per_bucket)  # what the clip summary filled; the metrics file adds to it
    if isinstance(per_input, dict):
        info.per_input = {str(k): v for k, v in per_input.items() if isinstance(v, dict)}
    elif isinstance(per_input, list):
        for row in per_input:
            if isinstance(row, dict) and (row.get("id") or row.get("input_id")):
                rid = str(row.get("id") or row.get("input_id"))
                info.per_input[rid] = {k: v for k, v in row.items() if k not in ("id", "input_id")}
    if isinstance(per_bucket, dict):
        info.per_bucket = {**clip_bucket, **{str(k): v for k, v in per_bucket.items() if isinstance(v, dict)}}
    if isinstance(info.metrics.get("role"), str):
        info.role = info.metrics["role"]
    return info


def load_clip_metrics(root: Path, label: str, info: EngineInfo, run: Run) -> None:
    """metrics/<label>.clip.json (bench clip --run-dir): the 'all' speed (clip time in s, clips per hour), each
    round's values as that round's bucket, and the self-reported stage medians. All observed or derived by bench
    clip; the page converts ms to s only."""
    path = root / "metrics" / f"{label}.clip.json"
    if not path.is_file():
        return
    try:
        data = _read_json(path)
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as e:
        run.warnings.append(f"metrics/{label}.clip.json: unreadable ({e})")
        return
    rel = path.relative_to(root).as_posix()
    run.sources.append(rel)
    info.metrics_file = info.metrics_file or rel
    speed = data.get("speed") if isinstance(data.get("speed"), dict) else {}
    overall = speed.get("all") if isinstance(speed.get("all"), dict) else {}
    row = {}
    cm = overall.get("clip_ms")
    if isinstance(cm, dict) and isinstance(cm.get("median"), (int, float)):
        ci = cm.get("ci95_median")
        ok = isinstance(ci, list) and len(ci) == 2 and all(isinstance(c, (int, float)) for c in ci)
        row["clip_s"] = {"value": cm["median"] / 1000, "ci95": [ci[0] / 1000, ci[1] / 1000] if ok else None}
    cph = overall.get("clips_per_hour")
    if isinstance(cph, dict) and isinstance(cph.get("median"), (int, float)):
        ci = cph.get("ci95_median")
        row["clips_per_hour"] = {"value": cph["median"], "ci95": ci if isinstance(ci, list) and len(ci) == 2
                                 and all(isinstance(c, (int, float)) for c in ci) else None}
    if row:
        info.per_bucket["all"] = row
    for r in data.get("rounds") or []:
        if isinstance(r, dict) and isinstance(r.get("index"), int):
            vals = {}
            if isinstance(r.get("clip_ms"), (int, float)):
                vals["clip_s"] = r["clip_ms"] / 1000
            if isinstance(r.get("clips_per_hour"), (int, float)):
                vals["clips_per_hour"] = r["clips_per_hour"]
            if vals:
                info.per_bucket[f"r{r['index']}"] = vals
    stages = speed.get("stages_self_reported")
    if isinstance(stages, dict):
        info.stages = {str(k): float(v["median"]) for k, v in stages.items()
                       if isinstance(v, dict) and isinstance(v.get("median"), (int, float))}


def load_cell(root: Path, engine: str, item: Item, run: Run) -> Cell:
    cell = Cell(engine=engine)
    folder = root / "outputs" / engine / item.bucket
    if item.bucket == "unbucketed":
        folder = root / "outputs" / engine
    found = sorted(p for p in folder.glob(f"{item.id}.*") if p.is_file()) if folder.is_dir() else []
    found = [p for p in found if _split_name(p.name)[0] == item.id or p.name == f"{item.id}.json"]
    for path in found:
        rel = path.relative_to(root).as_posix()
        cell.files.append(rel)
        run.sources.append(rel)
        _, suffix = _split_name(path.name)
        if path.name == f"{item.id}.json":
            suffix = "json"
        try:
            if suffix == "json":
                obj = _read_json(path)
                if not isinstance(obj, dict):
                    raise ValueError("not a JSON object")
                if obj.get("error"):
                    cell.status, cell.reason = "failed", str(obj["error"])
                cell.text = _output_text(obj) if cell.text is None else cell.text
                cell.token_ids = _ids(obj.get("token_ids")) or cell.token_ids
                cell.pieces = token_pieces(obj) or cell.pieces
                cell.finish_reason = obj.get("finish_reason") if isinstance(obj.get("finish_reason"), str) else None
                for k in ("reasoning", "reasoning_content"):
                    if isinstance(obj.get(k), str) and obj[k]:
                        cell.reasoning = obj[k]
                for k in ("transcript", "asr_text"):
                    if isinstance(obj.get(k), str):
                        cell.transcript = obj[k]
            elif suffix in ("asr.txt", "transcript.txt"):
                cell.transcript = path.read_text(encoding="utf-8", errors="replace")
            elif suffix in GROUND_TRUTH_SUFFIXES:
                continue
            elif suffix == "txt":
                if cell.text is None:
                    cell.text = path.read_text(encoding="utf-8", errors="replace")
            elif path.suffix.lower() in AUDIO_EXT and not cell.audio:
                cell.audio = rel
            elif path.suffix.lower() in VIDEO_EXT and not cell.video:
                cell.video = rel
            elif path.suffix.lower() in IMAGE_EXT and not cell.image:
                cell.image = rel
            elif path.suffix.lower() == ".pcm" and not cell.pcm:
                cell.pcm = rel
        except (OSError, ValueError) as e:
            cell.status, cell.reason = "failed", f"{rel}: unreadable ({e})"
    info = run.engine(engine)
    if info is not None:
        cell.metrics = dict(info.per_input.get(item.id) or info.per_input.get(f"{item.bucket}/{item.id}") or {})
        for k in ("transcript", "asr_text"):
            if cell.transcript is None and isinstance(cell.metrics.get(k), str):
                cell.transcript = cell.metrics[k]
        if cell.metrics.get("error") and cell.status != "failed":
            cell.status, cell.reason = "failed", str(cell.metrics["error"])
    if cell.status == "failed":
        return cell
    if cell.text is not None or cell.audio or cell.video or cell.image or cell.pcm or cell.token_ids is not None:
        cell.status = "ok"
    elif found:
        cell.status, cell.reason = "failed", "output file has no text, token ids, audio, video or image"
    else:
        cell.status, cell.reason = "not_run", "no output file for this input"
    return cell


def _manifest_engines(root: Path) -> list[dict]:
    p = root / "manifest.json"
    if not p.is_file():
        return []
    try:
        engines = _read_json(p).get("engines")
    except (OSError, ValueError, AttributeError):
        return []
    return [e for e in engines if isinstance(e, dict)] if isinstance(engines, list) else []


def _label_of(rec: dict) -> str | None:
    data = rec.get("data") or {}
    label = data.get("label") or ((rec.get("identity") or {}).get("engine") or {}).get("name")
    return str(label) if label else None


def _metric(rec: dict, names, subject=None) -> float | None:
    for name in names:
        for m in rec.get("metrics") or []:
            if (isinstance(m, dict) and m.get("name") == name and isinstance(m.get("value"), (int, float))
                    and (subject is None or m.get("subject") in (subject, None))):
                return float(m["value"])
    return None


def attach_evidence(run: Run, records: list[tuple[str, dict]]) -> None:
    """Quality, gate and capacity verdicts per engine label, from evidence records (v1 or v2, by field presence)."""
    from inferencekit.evidence import integrity_problem

    seen = set()
    for source, rec in records:
        if not isinstance(rec, dict) or not str(rec.get("evidence_id", "")).startswith("ev_"):
            continue
        if rec["evidence_id"] in seen:  # the same record from evidence/ and --evidence counts once
            continue
        seen.add(rec["evidence_id"])
        run.evidence_attached += 1
        data = rec.get("data") or {}
        kind = data.get("v2_kind") or ("gates" if data.get("gates_format") else rec.get("kind"))
        eid = rec["evidence_id"]
        problem = integrity_problem(rec)
        if problem:
            run.warnings.append(f"{source}: {problem}: its verdicts are shown as UNPROVEN")
        touched = []
        if kind == "quality_trajectory":
            for label, verdict in (data.get("verdicts") or {}).items():
                info = run.engine(str(label))
                if info is not None:
                    info.ev_quality.append({"name": "top-1 agreement", "unit": "%", "evidence": eid,
                                            "verdict": "UNPROVEN" if problem else str(verdict),
                                            "value": _metric(rec, ("top1_agreement_pct",), str(label))})
                    touched.append(info)
        elif kind == "clip_quality":
            touched += clip_quality(run, rec, source, problem)
        else:
            info = run.engine(_label_of(rec) or "")
            if info is None:
                continue
            touched.append(info)
            if kind == "clip_benchmark" and data.get("source") == "bench clip":
                for key, row in clip_rows(run, rec, source).items():  # merged: quality clip may have come first
                    info.clip_rows.setdefault(key, {}).update(row)
            elif kind in ("image_benchmark", "audio_benchmark"):
                from inferencekit.review.media import attach_rows, attach_verdicts

                media_kind = "image" if kind == "image_benchmark" else "audio"
                if data.get("source") == f"bench {media_kind}":
                    attach_verdicts(info, media_kind, data, eid, problem)
                    if not problem:
                        attach_rows(run, info, rec, source)
            elif kind == "quality_task":
                name = str(data.get("task") or "task")
                if name == "sampling":
                    value, unit = _metric(rec, ("first_token_tv_excess",), info.label), "probability"
                else:
                    value, unit = _metric(rec, (f"{name}_accuracy_pct", "accuracy_pct", "niah_accuracy_pct"),
                                          info.label), "%"
                info.ev_quality.append({"name": name, "unit": unit, "evidence": eid, "value": value,
                                        "verdict": "UNPROVEN" if problem else data.get("verdict")})
            elif kind in ("gates", "clip_gates"):
                gates = data.get("gates") or {}
                info.ev_gates = {"verdict": "UNPROVEN" if problem else str(data.get("verdict") or "UNPROVEN"),
                                 "violations": [f"{g}: {v.get('verdict')}" for g, v in gates.items()
                                                if isinstance(v, dict) and v.get("verdict") != "PASS"],
                                 "evidence": eid}
            elif kind == "capacity":
                # bench serve writes n_star=None for several reasons (SLO missed at the first level, no SLO,
                # closed loop, no levels); data.reason says which, data.level_kind whether it is streams or req/s.
                star = data["n_star"] if "n_star" in data else _metric(rec, ("n_star", "rate_star_rps"))
                level_kind = data.get("level_kind") or ("rate" if _metric(rec, ("rate_star_rps",)) is not None
                                                        else "concurrency")
                info.capacity = {"n_star": star, "reason": str(data["reason"]) if data.get("reason") else None,
                                 "level_kind": level_kind, "slo": data.get("slo"), "evidence": eid,
                                 "usd_per_1m": _metric(rec, ("usd_per_1m_output_tokens",)),
                                 "price": data.get("gpu_price_usd_h"), "edited": bool(problem)}
        for info in touched:
            if eid not in info.evidence:
                info.evidence.append(eid)


# Frame-quality metrics of a clip_quality record (quality clip): (name, unit, names it may carry), mean over a
# clip's frames, then over clips; psnr_db is the alias other writers use.
CLIP_QUALITY = (("psnr", "dB", ("psnr", "psnr_db")), ("ssim", "", ("ssim",)), ("lpips", "", ("lpips",)))
CLIP_ROW_KEYS = ("ok", "error", "seed", "e2e_ms", "stage_ms", "stage_sum_ms", "stage_sum_ok", "steps_run", "w", "h",
                 "frames", "fps", "dtype", "attention", "effective_mismatch", "cache", "out_sha256")
_QUALITY_KEY = re.compile(r"^engine:(.+)\|(r\d+/[^/]+)$")  # quality clip's clip_metrics rows: engine:<label>|r<k>/<id>


def artifact_lines(run: Run, rec: dict, source: str, role: str, what: str) -> list[dict] | None:
    """The JSON lines of a record's artifact with ``role``, found in the evidence store or, for a run directory
    inside the store root (``runs/<run-id>``), next to it. The file must have the sha256 the record lists;
    otherwise nothing is read and the page says why (``what`` names what is then not shown)."""
    from inferencekit.canonical import file_sha256
    from inferencekit.evidence import Store

    art = next((a for a in rec.get("artifacts") or [] if isinstance(a, dict) and a.get("role") == role), None)
    eid = rec["evidence_id"]
    if not art or not isinstance(art.get("path"), str) or ".." in Path(art["path"]).parts:
        return None
    for base in (Store().run_dir(eid), run.root.parent / eid):
        path = base / art["path"]
        if not path.is_file():
            continue
        if file_sha256(path)[0] != art.get("sha256"):
            run.warnings.append(f"{source}: its {role} file at {path.name} does not match the record's sha256: "
                                f"{what} not shown")
            return None
        out = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line) if line.strip() else None
            except ValueError:
                continue
            if isinstance(row, dict):
                out.append(row)
        return out
    run.warnings.append(f"{source}: record {eid[:15]}: its {role} file is not in the evidence store (pass "
                        f"--runs-dir for that store): {what} not shown")
    return None


def clip_rows(run: Run, rec: dict, source: str) -> dict:
    """``"r<k>/<id>" -> row`` from a bench clip record's ``requests`` artifact (requests.jsonl)."""
    out = {}
    for row in artifact_lines(run, rec, source, "requests", "per-clip times") or []:
        if isinstance(row.get("id"), str) and isinstance(row.get("round"), int):
            out[f"r{row['round']}/{row['id']}"] = {k: row.get(k) for k in CLIP_ROW_KEYS if k in row}
    return out


def clip_quality(run: Run, rec: dict, source: str, problem: str | None) -> list:
    """A quality clip record covers every engine it was given: ``data.engines[label]`` holds each one's verdict,
    its metrics are ``psnr`` / ``ssim`` / ``lpips`` with ``subject`` = the label, and the per-clip scores are the
    ``clip_metrics`` artifact's rows (key ``engine:<label>|r<k>/<id>``, ``mean`` per metric). Returns the engines
    it touched."""
    data, eid = rec.get("data") or {}, rec["evidence_id"]
    engines = data.get("engines") if isinstance(data.get("engines"), dict) else {}
    touched = []
    for label, block in engines.items():
        info = run.engine(str(label))
        if info is None or not isinstance(block, dict):
            continue
        touched.append(info)
        verdict = "UNPROVEN" if problem else str(block.get("verdict") or "UNPROVEN")
        for name, unit, names in CLIP_QUALITY:
            value = _metric(rec, names, info.label)
            if value is not None:
                info.ev_quality.append({"name": name, "unit": unit, "evidence": eid, "value": value,
                                        "verdict": verdict})
        if not any(q["evidence"] == eid for q in info.ev_quality):
            info.ev_quality.append({"name": "clip quality", "unit": None, "evidence": eid, "value": None,
                                    "verdict": verdict})
    if not touched:
        return touched
    for row in artifact_lines(run, rec, source, "clip_metrics", "per-clip quality") or []:
        found = _QUALITY_KEY.match(str(row.get("key") or ""))
        info = run.engine(found.group(1)) if found else None
        if info is None or info not in touched or row.get("ok") is False:
            continue
        mean = row.get("mean") if isinstance(row.get("mean"), dict) else {}
        got = {name: float(mean[name]) for name, _, _ in CLIP_QUALITY
               if isinstance(mean.get(name), (int, float)) and not isinstance(mean.get(name), bool)}
        if got:
            info.clip_rows.setdefault(found.group(2), {}).update(got)
    return touched


def merge_clip_rows(run: Run) -> None:
    """Video: each cell gets its bench clip row (client-timed clip_s, self-reported stages, effective shape) and
    quality clip's per-clip numbers. A clip the engine failed is FAILED with bench clip's error, not NOT RUN."""
    for info in run.engines:
        for item in run.items:
            row = info.clip_rows.get(run.key(item))
            cell = run.cells.get((info.label, run.key(item)))
            if not row or cell is None:
                continue
            cell.clip_row = row
            if isinstance(row.get("e2e_ms"), (int, float)):
                cell.metrics.setdefault("clip_s", row["e2e_ms"] / 1000)
            for name, _, _ in CLIP_QUALITY:
                if isinstance(row.get(name), (int, float)):
                    cell.metrics.setdefault(name, row[name])
            if row.get("ok") is False and cell.status != "failed":
                cell.status, cell.reason = "failed", f"bench clip: {row.get('error') or 'the clip failed'}"


def run_evidence(root: Path, run: Run) -> list[tuple[str, dict]]:
    out = []
    d = root / "evidence"
    if d.is_dir():
        for p in sorted(d.glob("*.json")):
            try:
                out.append((p.relative_to(root).as_posix(), _read_json(p)))
                run.sources.append(p.relative_to(root).as_posix())
            except (OSError, ValueError) as e:
                run.warnings.append(f"evidence/{p.name}: unreadable ({e})")
    return out


def load_run(root: str | Path, reference: str | None = None, prompts: str | Path | None = None,
             extra_evidence: list[dict] | None = None) -> Run:
    """Read a run directory. ``reference`` names the outputs/<label> column the others are diffed against;
    ``prompts`` (a prompt-set JSONL) fills the input column when inputs/ lacks them; ``extra_evidence``
    adds records (e.g. gates) beyond the run's own evidence/ directory."""
    root = Path(root)
    run = Run(root=root, modality="llm", engines=[], reference=None, items=[], cells={})
    items = load_inputs(root, run)
    if not items and not (root / "inputs").is_dir() and not prompts:
        run.warnings.append("no inputs/ directory (and no --prompts): rows come from the outputs, without prompts")

    out_dir = root / "outputs"
    labels = sorted((p.name for p in out_dir.iterdir() if p.is_dir()), key=natural_key) if out_dir.is_dir() else []
    met_dir = root / "metrics"
    if met_dir.is_dir():
        for p in sorted(met_dir.iterdir(), key=lambda q: natural_key(q.name)):
            if p.name.endswith(".load.json") or p.suffix not in (".json", ".jsonl"):
                continue  # load-mode levels belong to the engine of the same label, not a column of their own
            ending = next((s for s in (".clip.json", ".image.json", ".audio.json") if p.name.endswith(s)), None)
            label = p.name[:-len(ending)] if ending else p.name.rsplit(".", 1)[0]
            if label not in labels:
                labels.append(label)
    # Manifest order first (baseline recipe, tuned, custom), then anything else found.
    manifest = _manifest_engines(root)
    roles = {}
    order = []
    for e in manifest:
        label = str(e.get("label") or e.get("name") or "")
        if label:
            roles[label] = e.get("role")
            if label in labels:
                order.append(label)
    order += [lb for lb in labels if lb not in order]

    if reference is not None:
        if reference not in order:
            from inferencekit.registry import OpError

            raise OpError(f"reference '{reference}' has no outputs/{reference}/ or metrics/{reference}.json in "
                          f"{root}; found: {order or 'nothing'}")
        run.reference = reference
    elif REFERENCE_DEFAULT in order:
        run.reference = REFERENCE_DEFAULT
    if run.reference:
        order = [run.reference] + [lb for lb in order if lb != run.reference]

    for label in order:
        info = load_metrics(root, label, run)
        from inferencekit.review.media import load_media_metrics

        load_media_metrics(root, label, info, run, items)
        info.role = roles.get(label) or info.role or ("reference" if label == run.reference else None)
        run.engines.append(info)

    # Rows: every input, plus anything an engine produced for an input that is not in inputs/.
    for label in order:
        base = out_dir / label
        if not base.is_dir():
            continue
        deeper = 0
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(base)
            if len(rel.parts) != 2:  # index.json at the root; load-level dirs (<level>/<bucket>/<id>) below
                deeper += len(rel.parts) > 2
                continue
            bucket = rel.parts[0]
            stem, _ = _split_name(path.name)
            if f"{bucket}/{stem}" not in items:
                items[f"{bucket}/{stem}"] = Item(id=stem, bucket=bucket)
                if (root / "inputs").is_dir():
                    run.warnings.append(f"outputs/{label}/{rel.as_posix()}: no matching input under "
                                        f"inputs/{bucket}/")
        if deeper:
            run.warnings.append(f"outputs/{label}/: {deeper} file(s) in load-level directories are not shown "
                                "(the page shows the single-user outputs, outputs/<label>/<bucket>/<id>)")
    if prompts:
        load_prompt_set(Path(prompts), items, run)
    run.items = sorted(items.values(), key=lambda i: (bucket_key(i.bucket), natural_key(i.id)))
    for item in run.items:
        for info in run.engines:
            run.cells[(info.label, run.key(item))] = load_cell(root, info.label, item, run)
    attach_evidence(run, run_evidence(root, run) + [(f"evidence {str(r.get('evidence_id', '?'))[:15]}", r)
                                                    for r in extra_evidence or [] if isinstance(r, dict)])
    from inferencekit.review.media import merge_rows

    merge_rows(run)
    run.modality = detect_modality(root, run.items, run.cells)
    if run.modality == "llm" and not run.reference:
        run.warnings.insert(0, "no reference outputs (pass --reference LABEL or add outputs/reference/): "
                               "divergence is not marked")
    if run.modality == "video":
        merge_clip_rows(run)
        if not run.reference:
            run.warnings.insert(0, "no reference clips (pass --reference LABEL or add outputs/reference/): there is "
                                   "no column to compare the clips with by eye")
    if run.modality == "tts":
        for item in run.items:
            if item.ground_truth is None and item.text:
                item.ground_truth = item.text.strip()
    run.sources = sorted(set(run.sources))
    return run
