"""bench clip: time a video clip engine from the client side, one clip at a time, over a clip set.

The engine runs as its own server (the clip HTTP contract in serve/clip_client.py, or ComfyUI through
its adapter); inferencekit is only the client, so nothing the engine does can touch the timer.

  clip_ms          end to end per clip on the client's clock: from just before the request is sent to
                   the moment the client holds the whole clip (observed). The only published timer. Each
                   round's value is the MEAN over the round's clips (every round runs the same clips, so it
                   is the set's time per clip; a median over clips of different tasks or shapes would be one
                   clip's time and hide changes to the others); the record's value is the median over
                   rounds. ``clip_ms_<task>`` is the same over one task's clips (rounds carry both).
  clips_per_hour   3,600,000 x clips / (sum of their clip_ms) per round, one clip at a time (derived; per
                   task too); divided by ``--gpu-count`` for clips_per_hour_per_gpu.
  stage_ms         the engine's own per-stage times (text encoding, denoising, decode...): SELF-REPORTED,
                   never published as a timer. Each clip's stages are checked to sum to no more than its
                   client-timed end to end; a clip whose stages sum to more is flagged (overlapping
                   stages, or wrong self-reports).

Clip set: JSONL (or .jsonl.gz), one clip per line:
  {"id", "task": "t2v" | "i2v", "prompt", "image" (i2v only: image file, relative to the clip set),
   "width", "height", "num_frames", "fps", "steps", "guidance_scale", "audio"? (true/false)}

Seeds (``--seed-policy``) are fixed before the run, so every engine gets the same seed in the same round:
  per_round (default)  seed(r) = first 31 bits of sha256("inferencekit-clip-seed:<base>:<r>"): the same for
                       every engine and clip in round r and different across rounds, so no timed clip
                       repeats an earlier request (the counterpart of bench serve's cache salt).
  fixed                seed(r) = base in every round: later rounds repeat the first round's requests
                       (for quality checks; the record says an output cache could serve them).
  list                 seed(r) = --seeds[r].
A later re-run on another machine should pick a new ``--seed-base``, so no seed an engine may have seen before
repeats. The warm-up (``--warmup``, default 1, untimed) runs that many clips for EVERY distinct request
shape in the set (task, width, height, frames, fps, steps, guidance, audio, image or not), each at the
first clip of that shape with seeds of their own, all before the first timed clip: an engine compiles,
captures graphs and loads the image path per shape, and none of that may land in a timed clip. A warm-up
clip's prompt is its clip's prompt plus " (warm-up)", never a timed clip's prompt, so no timed clip can
reuse a text encoding the warm-up left behind (ComfyUI's node cache would).

Order: every round runs the clips in one order fixed by the clip set: the set's own order when no two
neighbours (the last clip and the first counted as neighbours too, for the next round) share a prompt or
an i2v image, else the first order found that has the fewest such neighbours (``data.order``). An engine
that caches encodings by their input would reuse them between such neighbours; the ones left are named in
the limitations.

Checks on every timed clip (``data.claim_validity``; compare refuses clip records that fail them):
  * the engine's reported ``effective`` steps_run, width, height, frames and fps equal what was asked
    (self-reported; gates clip decodes the outputs to check the real shape);
  * the engine reports no stage served from a cache, except the encoding of its fixed negative prompt
    (``negative_text_encode``: computed once and reused, allowed for every engine and stated as a limitation);
  * different requests never return byte-identical clips (a cached or seed-ignoring engine would).

Files, in the record's directory and copied to ``--run-dir`` when given:
  requests.jsonl                         one row per timed clip (role ``requests``)
  outputs/<label>/r<round>/<id>.<fmt>    every timed clip as returned (``--save-outputs hash_only`` keeps only
                                         the hashes); outputs/<label>/index.json has each file's sha256
                                         (role ``outputs_index``)
  inputs/<label>/r<round>/<id>.json      the exact request body (the image as its sha256 and size)
To interleave engines, alternate ``--rounds 1 --extend <previous ev>`` calls, or use bench rounds' clip
mode (serve/clip_rounds.py); ``compare --metric clip_ms`` takes two such records.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from inferencekit import evidence as ev
from inferencekit import prompts as sets
from inferencekit import stats
from inferencekit.canonical import canonical_json, file_sha256
from inferencekit.cpuinfo import cpu_identity
from inferencekit.gpuinfo import gpu_identity
from inferencekit.ops.bench_serve import _record_call
from inferencekit.registry import Arg, OpError, operation
from inferencekit.serve import clip_client as cc
from inferencekit.serve.client import Endpoint, api_key_from_env
from inferencekit.serve.metrics import pct

KIND = "clip_benchmark"
SEED_POLICIES = ("per_round", "fixed", "list")
SAVE_OUTPUTS = ("all", "hash_only")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}$")
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}$")
_STAGE = re.compile(r"[^a-z0-9_]+")
CLIP_KEYS = ("id", "task", "prompt", "image", "width", "height", "num_frames", "fps", "steps", "guidance_scale",
             "audio")
SHAPE_KEYS = ("task", "width", "height", "num_frames", "fps", "steps", "guidance_scale")
# effective key -> clip-set key it must equal
EFFECTIVE_CHECKS = (("steps_run", "steps"), ("width", "width"), ("height", "height"), ("frames", "num_frames"),
                    ("fps", "fps"))
# What a warm-up must cover besides SHAPE_KEYS: an engine compiles or captures graphs per request shape.
WARM_KEYS = SHAPE_KEYS + ("audio",)
ROUND_METRICS = ("clip_ms", "clips_per_hour")  # per round, for 'all' and as <metric>_<task> per task
# Arguments that must match for --extend (plus the clip set's and the workflow's sha256).
SAME_CONFIG = ("base_url", "endpoint_kind", "label", "seed_policy", "seed_base", "seeds", "engine_version", "model")
MS_PER_HOUR = 3_600_000.0
WARMUP_SUFFIX = " (warm-up)"

ARGS = [
    Arg("base_url", "string", "Engine root, e.g. http://127.0.0.1:8000. Never stored: records keep only an HMAC "
        "of it under a per-machine key (as bench serve does).", required=True, sensitive=True),
    Arg("clips", "string", "Clip set JSONL: {id, task t2v|i2v, prompt, image? (i2v), width, height, num_frames, "
        "fps, steps, guidance_scale, audio?} per line.", required=True),
    Arg("endpoint_kind", "string", "clip_http = the clip HTTP contract (POST /v1/clips, GET /health); comfyui = "
        "a ComfyUI server through its adapter (needs --workflow).", default="clip_http", choices=cc.ENDPOINT_KINDS),
    Arg("label", "string", "Name for this engine configuration in outputs/ (default: the endpoint kind)."),
    Arg("model", "string", "Model id for the record's identity, e.g. the checkpoint's repo id."),
    Arg("rounds", "integer", "Timed passes over the clip set.", default=3),
    Arg("first_round", "integer", "Index of the first timed round (picks its seed). Default 0, or the previous "
        "record's last round + 1 with --extend. bench rounds passes the round number so every engine gets the "
        "same seed in the same round."),
    Arg("warmup", "integer", "Untimed warm-up clips per distinct request shape (task, size, frames, fps, steps, "
        "guidance, audio, image), at that shape's first clip with seeds of their own, before any timed clip.",
        default=1),
    Arg("seed_policy", "string", "per_round = a seed derived from --seed-base and the round (same for every "
                                 "engine in a round, new every round); fixed = --seed-base every round; list = "
                                 "--seeds in round order.", default="per_round", choices=SEED_POLICIES),
    Arg("seed_base", "integer", "Base of the seed policy. Pick a new one when you re-run on another machine.", default=0),
    Arg("seeds", "array", "--seed-policy list: the seed of each round, round 0 first.", items="integer"),
    Arg("save_outputs", "string", "all = keep every timed clip; hash_only = record each clip's sha256 and size "
        "but not its bytes.", default="all", choices=SAVE_OUTPUTS),
    Arg("workflow", "string", "comfyui: the API-format workflow JSON with its node mapping (its sha256 is "
        "recorded)."),
    Arg("run_dir", "string", "Run directory: outputs/<label>/..., inputs/<label>/... and metrics/<label>.clip.json "
        "are also written there."),
    Arg("api_key_env", "string", "Name of the environment variable holding the API key (never stored)."),
    Arg("engine_version", "string", "Engine version, recorded in the identity."),
    Arg("gpu_name", "string", "GPU model, when the engine is not on this machine (otherwise read locally)."),
    Arg("gpu_count", "integer", "GPUs behind the engine (clips_per_hour_per_gpu divides by it).", default=1),
    Arg("timeout_s", "number", "Per-clip timeout, seconds.", default=cc.DEFAULT_TIMEOUT_S),
    Arg("resamples", "integer", "Bootstrap resamples for CIs.", default=stats.DEFAULT_RESAMPLES),
    Arg("extend", "string", "Evidence id of an earlier run of the same configuration: its rounds are kept and new "
        "rounds appended (alternate engines this way)."),
]


# --------------------------------------------------------------------------- the clip set

@dataclass
class ClipSet:
    path: Path
    name: str
    sha256: str
    clips: list[dict]
    images: dict  # clip id -> (bytes, sha256)
    ignored_keys: list[str]

    def identity(self) -> dict:
        return {"name": self.name, "sha256": self.sha256}

    def uniform(self, key: str):
        """The value every clip has for ``key``, or None when clips differ."""
        values = {json.dumps(c.get(key)) for c in self.clips}
        return self.clips[0].get(key) if len(values) == 1 else None


def _positive_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 1


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def load_clip_set(path: str | os.PathLike) -> ClipSet:
    """Read and check a clip set; read the i2v images (paths relative to the clip set file)."""
    p = Path(path)
    try:
        raw = sets.read_bytes(p)
    except (OSError, ValueError) as e:
        raise OpError(f"cannot read clip set {path}: {e}") from None
    clips, problems, seen, ignored = [], [], set(), set()
    for n, line in enumerate(raw.decode(errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            problems.append(f"line {n} is not JSON")
            continue
        if not isinstance(row, dict):
            problems.append(f"line {n} is not an object")
            continue
        cid = row.get("id")
        where = f"line {n} ({cid})" if isinstance(cid, str) else f"line {n}"
        if not isinstance(cid, str) or not _ID.match(cid):
            problems.append(f"{where}: id must be 1-128 letters, digits and _ . + - (it names a file)")
        elif cid in seen:
            problems.append(f"{where}: id appears twice")
        seen.add(cid)
        if row.get("task") not in cc.TASKS:
            problems.append(f"{where}: task must be one of {list(cc.TASKS)}")
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            problems.append(f"{where}: prompt must be non-empty text")
        for key in ("width", "height", "num_frames", "steps"):
            if not _positive_int(row.get(key)):
                problems.append(f"{where}: {key} must be an integer >= 1")
        if not _number(row.get("fps")) or row["fps"] <= 0:
            problems.append(f"{where}: fps must be a number > 0")
        if not _number(row.get("guidance_scale")) or row["guidance_scale"] < 0:
            problems.append(f"{where}: guidance_scale must be a number >= 0")
        if row.get("task") == "i2v" and not isinstance(row.get("image"), str):
            problems.append(f"{where}: an i2v clip needs an image (a file path)")
        if row.get("task") == "t2v" and row.get("image") is not None:
            problems.append(f"{where}: a t2v clip takes no image")
        if row.get("audio") is not None and not isinstance(row["audio"], bool):
            problems.append(f"{where}: audio must be true or false")
        ignored.update(k for k in row if k not in CLIP_KEYS)
        clips.append(row)
    if not clips and not problems:
        problems.append("no clips")
    images = {}
    for c in clips:
        if isinstance(c.get("image"), str) and isinstance(c.get("id"), str):
            img = Path(c["image"]) if Path(c["image"]).is_absolute() else p.parent / c["image"]
            try:
                data = img.read_bytes()
            except OSError as e:
                problems.append(f"{c['id']}: image {c['image']} not readable ({type(e).__name__})")
                continue
            images[c["id"]] = (data, hashlib.sha256(data).hexdigest())
    if problems:
        raise OpError(f"clip set {path}: " + "; ".join(problems[:10]))
    return ClipSet(p, sets.set_stem(p), hashlib.sha256(raw).hexdigest(), clips, images, sorted(ignored))


# --------------------------------------------------------------------------- seeds

def round_seed(policy: str, base: int, seeds: list[int] | None, r: int) -> int:
    """The seed of round ``r`` under a seed policy (see the module docstring)."""
    if policy == "fixed":
        return base
    if policy == "list":
        return int(seeds[r])
    digest = hashlib.sha256(f"inferencekit-clip-seed:{base}:{r}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def warmup_seed(base: int, i: int) -> int:
    digest = hashlib.sha256(f"inferencekit-clip-warmup:{base}:{i}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def seed_policy_doc(policy: str, base: int, seeds: list[int] | None) -> dict:
    rule = {"per_round": "seed(r) = first 31 bits of sha256('inferencekit-clip-seed:<base>:<r>'), every clip of "
                         "round r",
            "fixed": "seed(r) = base, every clip and round",
            "list": f"seed(r) = seeds[r], every clip of round r ({len(seeds or [])} seeds given)"}[policy]
    return {"name": policy, "base": None if policy == "list" else base, "rule": rule}


def request_set(clipset: ClipSet, policy: dict, seeds: list[int] | None) -> dict:
    """identity.workload.request_set: sha256 over what every clip asks for (its shape, prompt, image sha256) and
    the seed policy, so compare can refuse two records that sent different requests."""
    items = [{**{k: c.get(k) for k in CLIP_KEYS if k != "image"},
              "image_sha256": (clipset.images.get(c["id"]) or (None, None))[1]} for c in clipset.clips]
    settings = {"seed_policy": {k: policy.get(k) for k in ("name", "base")},
                "seeds": list(seeds) if policy["name"] == "list" else None}
    return {"sha256": hashlib.sha256(canonical_json({"items": items, "settings": settings}).encode()).hexdigest(),
            "n": len(items)}


def warm_shapes(clipset: ClipSet) -> list[tuple[dict, dict]]:
    """(shape, its first clip) for every distinct request shape of the set, in clip-set order."""
    seen, out = set(), []
    for c in clipset.clips:
        shape = {**{k: c.get(k) for k in WARM_KEYS}, "image": c.get("image") is not None}
        key = canonical_json(shape)
        if key not in seen:
            seen.add(key)
            out.append((shape, c))
    return out


def warmup_prompt(prompt: str, taken: set[str]) -> str:
    """A warm-up prompt: ``prompt`` plus " (warm-up)" (numbered if needed), never one of the ``taken`` (timed)
    prompts, so a timed clip never reuses a text encoding made by the warm-up."""
    out, n = prompt + WARMUP_SUFFIX, 2
    while out in taken:
        out, n = f"{prompt} (warm-up {n})", n + 1
    return out


def _clash(a: dict, b: dict, clipset: ClipSet) -> bool:
    """Neighbouring clips that share a prompt or an i2v image (an encoding an engine could reuse)."""
    ia, ib = (clipset.images.get(c["id"], (None, None))[1] for c in (a, b))
    return a["prompt"] == b["prompt"] or (ia is not None and ia == ib)


def clashes(order: list[dict], clipset: ClipSet, wrap: bool = True) -> list[tuple[str, str]]:
    """(id, next id) of neighbouring clips sharing a prompt or an image; ``wrap``: the last clip and the first
    (the next round) count as neighbours."""
    pairs = list(zip(order, order[1:])) + ([(order[-1], order[0])] if wrap and len(order) > 2 else [])
    return [(a["id"], b["id"]) for a, b in pairs if _clash(a, b, clipset)]


def timed_order(clipset: ClipSet) -> list[dict]:
    """The order every round runs: the clip set's own when no neighbours clash (prompt or image, the last and
    the first included), else the order with the fewest clashes found by a greedy pass from each start (most
    clashing clips first; deterministic, the same for every engine and call)."""
    clips = clipset.clips
    if not clashes(clips, clipset):
        return list(clips)
    n = len(clips)
    clash = [[i != j and _clash(clips[i], clips[j], clipset) for j in range(n)] for i in range(n)]
    best, best_bad = list(clips), len(clashes(clips, clipset))
    for start in range(n):
        seq, left = [start], set(range(n)) - {start}
        deg = [sum(clash[j][k] for k in left) for j in range(n)]  # clashing partners still to place
        while left:
            prev = seq[-1]
            nxt = min(left, key=lambda j: (clash[prev][j], -deg[j], j))
            seq.append(nxt)
            left.discard(nxt)
            for k in range(n):
                deg[k] -= clash[k][nxt]
        order = [clips[i] for i in seq]
        bad = len(clashes(order, clipset))
        if bad < best_bad:
            best, best_bad = order, bad
    return best


def warm_plan(shapes: list[tuple[dict, dict]], order: list[dict], clipset: ClipSet) -> list[tuple[dict, dict]]:
    """warm_shapes, re-ordered (and, if needed, another clip of the last shape) so that the last warm-up clip's
    i2v image is not the first timed clip's: the image encoding left by the warm-up must not serve it."""
    first = clipset.images.get(order[0]["id"], (None, None))[1] if order else None
    img = lambda c: clipset.images.get(c["id"], (None, None))[1]  # noqa: E731
    if first is None or not shapes or img(shapes[-1][1]) != first:
        return shapes
    ok = [k for k, (_, c) in enumerate(shapes) if img(c) != first]
    if ok:
        k = ok[-1]
        return shapes[:k] + shapes[k + 1:] + [shapes[k]]
    shape, _ = shapes[-1]
    other = next((c for c in clipset.clips if img(c) not in (None, first)
                  and {**{k: c.get(k) for k in WARM_KEYS}, "image": True} == shape), None)
    return shapes[:-1] + [(shape, other)] if other else shapes


def make_request(clip: dict, seed: int, clipset: ClipSet, prompt: str | None = None) -> cc.ClipRequest:
    """The request for ``clip`` at ``seed``; ``prompt`` replaces its prompt (the warm-up's own prompt)."""
    image = clipset.images.get(clip["id"])
    return cc.ClipRequest(id=clip["id"], task=clip["task"], prompt=prompt or clip["prompt"], width=clip["width"],
                          height=clip["height"], num_frames=clip["num_frames"], fps=clip["fps"], steps=clip["steps"],
                          guidance_scale=clip["guidance_scale"], seed=seed, image=image[0] if image else None,
                          audio=clip.get("audio"))


# --------------------------------------------------------------------------- per clip

def _stage_name(name: str) -> str:
    return _STAGE.sub("_", str(name).lower()).strip("_")[:48] or "stage"


def check_effective(effective: dict | None, clip: dict) -> tuple[list[str], int]:
    """(mismatches, number of fields checked) between what the engine says it ran and what was asked."""
    if not isinstance(effective, dict):
        return [], 0
    bad, checked = [], 0
    for eff_key, asked_key in EFFECTIVE_CHECKS:
        got = effective.get(eff_key)
        if not _number(got):
            continue
        checked += 1
        want = clip[asked_key]
        if abs(float(got) - float(want)) > 1e-6:
            bad.append(f"{eff_key} {got:g} (asked {want:g})")
    return bad, checked


def clip_row(res: cc.ClipResult, clip: dict, r: int, seed: int, t_wall: float, out_sha: str | None) -> dict:
    """One timed clip's numbers (no bytes): a row of requests.jsonl."""
    e2e = res.e2e_ms
    stage_sum = cc.stage_sum_ms(res.stage_ms) if res.ok else None
    eff = res.effective or {}
    mismatch, checked = check_effective(res.effective, clip) if res.ok else ([], 0)
    return {"id": clip["id"], "task": clip["task"], "round": r, "seed": seed, "ok": res.ok, "error": res.error,
            "status": res.status, "e2e_ms": e2e, "connect_ms": res.connect_ms,
            "stage_ms": res.stage_ms if res.ok else None, "stage_sum_ms": stage_sum,
            "stage_sum_ok": None if stage_sum is None or e2e is None else stage_sum <= e2e,
            "steps_run": eff.get("steps_run"), "w": eff.get("width"), "h": eff.get("height"),
            "frames": eff.get("frames"), "fps": eff.get("fps"), "dtype": eff.get("dtype"),
            "attention": eff.get("attention"), "audio": eff.get("audio"),
            "effective_checked": checked, "effective_mismatch": mismatch, "cache": res.cache,
            "cache_allowed": res.cache_allowed or None,
            "out_sha256": out_sha, "out_bytes": len(res.video) if res.video is not None else None,
            "format": res.format if res.ok else None, "via": res.via,
            "asked": {k: clip[k] for k in SHAPE_KEYS}, "t_start": t_wall,
            "t_send_ns": res.t_send_ns, "t_end_ns": res.t_end_ns}


def _place(data: bytes, rel: str, roots: list[Path]) -> None:
    """Write ``data`` at ``<root>/<rel>`` for each root (a hard link after the first when possible).

    ``rel`` is built from checked ids and labels and a sanitised format; a path that would still land
    outside a root is refused before anything is created or removed there."""
    if Path(rel).is_absolute() or ".." in Path(rel).parts:
        raise OpError(f"refusing to write {rel!r}: not a plain relative path")
    first = None
    for root in roots:
        target = root / rel
        if not target.resolve().is_relative_to(root.resolve()):
            raise OpError(f"refusing to write {rel!r}: it resolves outside the output directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        if first is not None:
            try:
                os.link(first, target)
                continue
            except OSError:
                shutil.copyfile(first, target)
                continue
        target.write_bytes(data)
        first = target


def _input_doc(req: cc.ClipRequest, clipset: ClipSet) -> dict:
    body = req.body()
    if "image" in body:
        data, sha = clipset.images[req.id]
        body["image"] = {"sha256": sha, "bytes": len(data)}
    return body


# --------------------------------------------------------------------------- summaries

def round_stats(rows: list[dict]) -> dict[str, dict]:
    """One round's values per task and 'all': clip_ms = mean client-timed ms over its ok clips, clips_per_hour
    = 3,600,000 x clips / their sum (one clip at a time), n ok clips, n_failed. A mean, not a median: with one
    t2v and one i2v clip a median is just the faster clip, and a change to the other would not show."""
    out = {}
    for subject in sorted({str(r["task"]) for r in rows if r.get("task")}) + ["all"]:
        sel = [r for r in rows if subject == "all" or r.get("task") == subject]
        ok = [float(r["e2e_ms"]) for r in sel if r.get("ok") and _number(r.get("e2e_ms"))]
        total = sum(ok)
        out[subject] = {"clip_ms": total / len(ok) if ok and total > 0 else None,
                        "clips_per_hour": MS_PER_HOUR * len(ok) / total if ok and total > 0 else None,
                        "n": len(ok), "n_failed": len(sel) - len(ok)}
    return out


def round_key(metric: str, subject: str) -> str:
    """The data.rounds key of a per-round value: clip_ms for every clip, clip_ms_i2v for one task."""
    return metric if subject == "all" else f"{metric}_{subject}"


def round_rows(rows: list[dict], walls: dict[int, tuple[float, float]], seeds: dict[int, int]) -> list[dict]:
    """One row per round in the shape compare reads: index, t_start, clip_ms and clips_per_hour (every clip),
    clip_ms_<task> and clips_per_hour_<task> (one task)."""
    out = []
    for r, (t0, t1) in sorted(walls.items()):
        mine = [x for x in rows if x["round"] == r]
        vals = round_stats(mine)
        row = {"index": r, "t_start": t0, "t_end": t1, "seed": seeds.get(r), "n_requests": vals["all"]["n"],
               "n_failed": vals["all"]["n_failed"]}
        for subject, v in vals.items():
            row.update({round_key(m, subject): v[m] for m in ROUND_METRICS})
        out.append(row)
    return out


def clip_metrics(rows: list[dict], tasks: list[str], gpu_count: int, resamples: int) -> tuple[list[dict], dict]:
    """Evidence metrics per task and 'all': clip_ms and clips_per_hour, each the median over rounds of the
    round's value (round_stats) with a bootstrap CI over rounds, and the self-reported stage medians.

    The median over rounds is the usual one (statistics.median: with an even number of rounds, the mean of the two
    middle rounds), as in bench rounds' summary. It used to be the nearest-rank value, which with an even number
    of rounds is the lower of the two: the faster round for clip_ms but the slower one for clips_per_hour, so one
    record's two "medians" came from different rounds. The CIs are unchanged."""
    out, table = [], {}
    by_round = {k: round_stats([r for r in rows if r["round"] == k]) for k in sorted({r["round"] for r in rows})}
    for subject in list(tasks) + ["all"]:
        sel = [r for r in rows if r["ok"] and r["e2e_ms"] is not None and (subject == "all" or r["task"] == subject)]
        values = [r["e2e_ms"] for r in sel]
        done = [v[subject] for v in by_round.values() if (v.get(subject) or {}).get("clip_ms") is not None]
        means, rates = [v["clip_ms"] for v in done], [v["clips_per_hour"] for v in done]
        entry = {"n_clips": len(sel), "rounds": len(means)}
        if means:
            ci = stats.bootstrap_ci(means, ps=(50,), resamples=resamples, seed=0)[50]
            entry["clip_ms"] = {"median": statistics.median(means), "ci95_median": list(ci), "n_rounds": len(means),
                                "per_clip": {"median": pct(values, 50), "max": max(values), "n": len(values)}}
            out.append(ev.metric("clip_ms", statistics.median(means), "ms", "median", n=len(means), ci95=ci,
                                 subject=subject))
            rci = stats.bootstrap_ci(rates, ps=(50,), resamples=resamples, seed=0)[50]
            rate = statistics.median(rates)
            entry["clips_per_hour"] = {"median": rate, "ci95_median": list(rci), "n_rounds": len(rates)}
            out.append(ev.metric("clips_per_hour", rate, "clips/h", "median_over_rounds", n=len(rates), ci95=rci,
                                 subject=subject))
            if gpu_count > 1:
                out.append(ev.metric("clips_per_hour_per_gpu", rate / gpu_count, "clips/h/GPU",
                                     "median_over_rounds", n=len(rates),
                                     ci95=[x / gpu_count if x is not None else None for x in rci], subject=subject))
        else:
            out.append(ev.metric("clip_ms", None, "ms", "median", n=0, subject=subject))
        table[subject] = entry
    stages: dict[str, list[float]] = {}
    for r in rows:
        if r["ok"] and isinstance(r.get("stage_ms"), dict):
            for k, v in r["stage_ms"].items():
                if _number(v):
                    stages.setdefault(_stage_name(k), []).append(float(v))
    for name, vals in sorted(stages.items()):
        out.append(ev.metric(f"stage_{name}_ms", statistics.median(vals), "ms", "median_self_reported", n=len(vals),
                             subject="all"))
    table["stages_self_reported"] = {k: {"median": statistics.median(v), "n": len(v)}
                                     for k, v in sorted(stages.items())}
    return out, table


def validity(rows: list[dict], policy: str) -> tuple[dict, list[str]]:
    """claim_validity for the record (violations make compare refuse it) and plain limitations."""
    violations, warnings, lim = [], [], []
    ok = [r for r in rows if r["ok"]]
    for r in ok:
        if r["effective_mismatch"]:
            violations.append(f"{r['id']} r{r['round']}: engine reports {', '.join(r['effective_mismatch'])}")
        if r.get("cache"):
            violations.append(f"{r['id']} r{r['round']}: engine reports cached stage(s) {r['cache'][:5]}")
    by_sha: dict[str, set] = {}
    for r in ok:
        if r.get("out_sha256"):
            by_sha.setdefault(r["out_sha256"], set()).add((r["id"], r["seed"]))
    for sha, reqs in sorted(by_sha.items()):
        if len(reqs) > 1:
            shown = ", ".join(f"{i} seed {s}" for i, s in sorted(reqs)[:4])
            violations.append(f"byte-identical clips for {len(reqs)} different requests ({shown}): cached outputs "
                              "or an ignored seed/prompt")
    unchecked = [r for r in ok if not r["effective_checked"]]
    if unchecked:
        lim.append(f"{len(unchecked)} of {len(ok)} clip(s): the engine reported no effective steps/width/height/"
                   "frames/fps, so this record cannot say they were honoured (gates clip decodes the outputs)")
    over = [r for r in ok if r.get("stage_sum_ok") is False]
    if over:
        warnings.append(f"{len(over)} clip(s): self-reported stage times sum to more than the client-timed end to "
                        f"end (e.g. {over[0]['id']} r{over[0]['round']}: {over[0]['stage_sum_ms']:.1f} > "
                        f"{over[0]['e2e_ms']:.1f} ms); overlapping stages or wrong self-reports, so the stage "
                        "breakdown is not trustworthy for them")
    no_stage = [r for r in ok if r.get("stage_sum_ms") is None]
    if no_stage:
        lim.append(f"{len(no_stage)} clip(s) without self-reported stage times")
    reused = [r for r in ok if r.get("cache_allowed")]
    if reused:
        lim.append(f"{len(reused)} of {len(ok)} clip(s): the engine reused the encoding of its fixed negative prompt "
                   f"({cc.NEGATIVE_PROMPT_CACHE}: computed once, not per clip). Allowed for every engine (a clip "
                   "HTTP engine reports it in cache; a ComfyUI workflow lists that encoder in allow_cached) and "
                   "stated here")
    if policy == "fixed" and len({r["round"] for r in rows}) > 1:
        lim.append("--seed-policy fixed: every round repeats the first round's requests, so an engine output cache "
                   "could serve rounds after the first (unique-output checks cannot see a cache that returns the "
                   "same bytes for the same request)")
    lim += [f"claim invalid: {v}" for v in violations[:10]] + warnings
    return {"valid": not violations, "violations": violations[:50], "warnings": warnings,
            "cache_allowed_clips": len(reused),
            "checked": ["effective_shape (self-reported)", "engine_cache (self-reported)", "unique_outputs",
                        "stage_sum <= e2e (self-reported stages)"]}, lim


# --------------------------------------------------------------------------- the operation

def _load_previous(store, extend, rcall, clipset, workflow_sha):
    prev = store.load(extend)
    if prev.get("kind") != KIND or (prev.get("data") or {}).get("source") != "bench clip":
        raise OpError(f"--extend {extend}: not a bench clip record")
    problem = ev.integrity_problem(prev)
    if problem:
        raise OpError(f"--extend: {problem}; refusing to build on it")
    prev_args = prev["data"].get("request", {}).get("args", {})
    mine = ev.redact(rcall.request()["args"])
    diff = [k for k in SAME_CONFIG if prev_args.get(k) != mine.get(k)]
    if (prev["identity"]["workload"].get("clip_set") or {}).get("sha256") != clipset.sha256:
        diff.append("clip set")
    if (prev["data"].get("backend") or {}).get("workflow_sha256") != workflow_sha:
        diff.append("workflow")
    if diff:
        raise OpError(f"--extend needs the same configuration; these differ: {diff}")
    art = next((a for a in prev["artifacts"] if a.get("role") == "requests"), None)
    path = store.run_dir(prev["evidence_id"]) / art["path"] if art else None
    if path is None or not path.is_file() or file_sha256(path)[0] != art["sha256"]:
        raise OpError(f"--extend: the per-request rows of {prev['evidence_id'][:15]} are missing or changed")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return prev, rows


def _fmt_s(ms) -> str:
    return "n/a" if ms is None else f"{ms / 1000:.2f} s"


@operation("bench_clip", ("bench", "clip"), "Client-timed benchmark of a video clip engine (clip HTTP contract or "
           "ComfyUI): end-to-end seconds per clip with CIs over rounds, clips/hour, self-reported stage times, "
           "pre-registered seeds per round, and checks on steps, shape, caching and unique outputs.", ARGS)
def bench_clip(call, base_url, clips, endpoint_kind, label, model, rounds, first_round, warmup, seed_policy,
               seed_base, seeds, save_outputs, workflow, run_dir, api_key_env, engine_version, gpu_name, gpu_count,
               timeout_s, resamples, extend):
    if rounds < 1:
        raise OpError("rounds must be >= 1")
    if warmup < 0:
        raise OpError("warmup must be >= 0")
    if gpu_count < 1:
        raise OpError("gpu_count must be >= 1")
    if first_round is not None and first_round < 0:
        raise OpError("first_round must be >= 0")
    if seeds and seed_policy != "list":
        raise OpError("--seeds is for --seed-policy list")
    if seed_policy == "list" and not seeds:
        raise OpError("--seed-policy list needs --seeds (one per round, round 0 first)")
    label = label or endpoint_kind
    if not _LABEL.match(label):
        raise OpError("label must be 1-64 characters of letters, digits and _ . + - (it names a directory)")
    if endpoint_kind == "comfyui" and not workflow:
        raise OpError("--endpoint-kind comfyui needs --workflow (the API-format workflow JSON)")
    workflow_sha = file_sha256(Path(workflow))[0] if workflow and Path(workflow).is_file() else None
    if workflow and workflow_sha is None:
        raise OpError(f"--workflow {workflow}: not a readable file")
    clipset = load_clip_set(clips)
    try:
        ep = Endpoint.parse(base_url, api_key_from_env(api_key_env))
    except ValueError as e:
        raise OpError(str(e)) from None
    if ep.is_local() and os.environ.get("INFERENCEKIT_CPU_LOCK_HELD") and (gpu_name or gpu_count != 1):
        raise OpError("CPU rounds cannot use --gpu-name or a non-default --gpu-count")
    rcall, key_persistent = _record_call(call, ep)
    store = ev.Store()
    prev, prev_rows = (None, [])
    if extend:
        prev, prev_rows = _load_previous(store, extend, rcall, clipset, workflow_sha)
    done = max((r["round"] for r in prev_rows), default=-1)
    start = done + 1 if first_round is None else first_round
    if start <= done:
        raise OpError(f"--first-round {start}: the extended record already has rounds up to {done}")
    last = start + rounds - 1
    if seed_policy == "list" and len(seeds) <= last:
        raise OpError(f"--seeds gives {len(seeds)} seed(s) but round {last} needs seed index {last}")
    policy = seed_policy_doc(seed_policy, seed_base, seeds)
    round_seeds = {r: round_seed(seed_policy, seed_base, seeds, r) for r in range(last + 1)}

    try:
        backend = cc.make_backend(endpoint_kind, ep, timeout_s=timeout_s, workflow=workflow)
    except (ValueError, LookupError) as e:
        raise OpError(str(e)) from None
    status, detail = backend.health()
    if status != 200:
        backend.close()
        raise OpError(f"engine not ready: GET {cc.HEALTH_PATH} answered {status or 'nothing'} ({detail}); start it "
                      "and wait for /health")
    run = Path(run_dir) if run_dir else None
    order = timed_order(clipset)
    shapes = warm_plan(warm_shapes(clipset), order, clipset)
    taken = {c["prompt"] for c in clipset.clips}
    try:
        with store.staging_dir() as staging:
            # Untimed, before the first timed clip: every shape once compiled/captured, seeds and prompts of
            # their own (a timed clip never reuses an encoding the warm-up made).
            warm = {"per_shape": warmup, "shapes": [{**shape, "clip": c["id"]} for shape, c in shapes], "clips": [],
                    "prompt": f"the clip's prompt + {WARMUP_SUFFIX.strip()!r}, never a timed clip's prompt"}
            first_img = clipset.images.get(order[0]["id"], (None, None))[1]
            warm["last_image_is_first_clips"] = bool(warmup and shapes and first_img is not None
                                                     and clipset.images.get(shapes[-1][1]["id"], (None, None))[1]
                                                     == first_img)
            for s, (_, clip) in enumerate(shapes):
                for i in range(warmup):
                    seed = warmup_seed(seed_base, s * warmup + i)
                    res = backend.generate(make_request(clip, seed, clipset, warmup_prompt(clip["prompt"], taken)))
                    warm["clips"].append({"id": clip["id"], "shape": s, "seed": seed, "ok": res.ok,
                                          "e2e_ms": res.e2e_ms, "error": res.error})
            rows, walls, index = [], {}, {}
            roots = [staging] + ([run] if run else [])
            for r in range(start, last + 1):
                t0 = time.time()
                for clip in order:
                    req = make_request(clip, round_seeds[r], clipset)
                    t_wall = time.time()
                    res = backend.generate(req)
                    out_sha = hashlib.sha256(res.video).hexdigest() if res.ok and res.video is not None else None
                    base = f"r{r}/{clip['id']}"
                    _place(json.dumps(_input_doc(req, clipset), indent=1).encode() + b"\n",
                           f"inputs/{label}/{base}.json", roots)
                    if out_sha and save_outputs == "all":
                        rel = f"{base}.{res.format}"
                        _place(res.video, f"outputs/{label}/{rel}", roots)
                        index[rel] = out_sha
                    rows.append(clip_row(res, clip, r, round_seeds[r], t_wall, out_sha))
                    res.video = None  # the bytes are on disk; do not hold every clip in memory
                walls[r] = (t0, time.time())
            return _record(rcall=rcall, store=store, staging=staging, run=run, backend=backend, ep=ep,
                           clipset=clipset, order=order, rounds=rounds, label=label, model=model,
                           endpoint_kind=endpoint_kind, policy=policy, seeds=seeds, round_seeds=round_seeds,
                           warm=warm, rows=rows, walls=walls, index=index,
                           prev=prev, prev_rows=prev_rows, save_outputs=save_outputs, workflow_sha=workflow_sha,
                           engine_version=engine_version, gpu_name=gpu_name, gpu_count=gpu_count,
                           resamples=resamples, key_persistent=key_persistent)
    finally:
        backend.close()


def _record(*, rcall, store, staging, run, backend, ep, clipset, order, rounds, label, model, endpoint_kind, policy,
            seeds, round_seeds, warm, rows, walls, index, prev, prev_rows, save_outputs, workflow_sha, engine_version,
            gpu_name, gpu_count, resamples, key_persistent):
    all_rows = prev_rows + rows
    req_path = staging / "requests.jsonl"
    req_path.write_text("".join(json.dumps(ev.redact(r), ensure_ascii=False) + "\n" for r in all_rows))
    artifacts = [ev.artifact(req_path, staging, "requests")]
    if index:
        idx = staging / "outputs" / label / "index.json"
        idx.parent.mkdir(parents=True, exist_ok=True)
        idx.write_text(json.dumps({"label": label, "files": dict(sorted(index.items()))}, indent=1) + "\n")
        artifacts.append(ev.artifact(idx, staging, "outputs_index"))
    tasks = sorted({c["task"] for c in clipset.clips})
    mets, table = clip_metrics(all_rows, tasks, gpu_count, resamples)
    prev_walls = {int(x["index"]): (x["t_start"], x["t_end"]) for x in (prev or {}).get("data", {}).get("rounds", [])}
    rounds_table = round_rows(all_rows, {**prev_walls, **walls}, round_seeds)
    claim, limitations = validity(all_rows, policy["name"])
    failed = [r for r in all_rows if not r["ok"]]
    if failed:
        limitations.insert(0, f"{len(failed)} of {len(all_rows)} clip(s) failed (e.g. {failed[0]['id']} "
                              f"r{failed[0]['round']}: {failed[0]['error']})")
    bad_warm = [w for w in warm["clips"] if not w["ok"]]
    if bad_warm:
        limitations.append(f"{len(bad_warm)} of {len(warm['clips'])} warm-up clip(s) failed (e.g. "
                           f"{bad_warm[0]['id']}: {bad_warm[0]['error']}): the first timed clip of that shape may "
                           "include compilation")
    if not warm["clips"]:
        limitations.append("--warmup 0: the first timed clip of each shape may include compilation or graph capture")
    if warm.get("last_image_is_first_clips"):
        limitations.append("the last warm-up clip used the first timed clip's image (the set has no other image of "
                           "that shape): an engine that caches image encodings could reuse it for that clip")
    if len(rounds_table) < 2:
        limitations.append("fewer than 2 rounds: no confidence interval")
    if not ep.is_local():
        limitations.append("engine is not on this host: clip_ms includes the network transfer of the clip")
    if not key_persistent:
        limitations.append("no per-machine key file could be written: the endpoint tag in this record is valid for "
                           "this process only, so --extend from another process sees a different base_url")
    if clipset.ignored_keys:
        limitations.append(f"clip set keys not sent to the engine: {clipset.ignored_keys[:8]}")
    if save_outputs == "hash_only":
        limitations.append("--save-outputs hash_only: clip bytes were not kept, only their sha256 and size")
    near = clashes(order, clipset, wrap=rounds > 1) + (
        [(order[0]["id"], order[0]["id"])] if len(order) == 1 and rounds > 1 else [])
    if near:
        limitations.append(f"{len(near)} pair(s) of neighbouring clips share a prompt or an image (e.g. "
                           f"{near[0][0]} -> {near[0][1]}): an engine that caches text or image encodings reuses "
                           "them there (ComfyUI does, and reports it: the claim is then invalid); give the clip set "
                           "other prompts and images")
    cache_mode = (backend.describe() or {}).get("cache_mode")
    if cache_mode in ("ram", "lru") and rounds > 1:  # classic keeps the last prompt only; none keeps nothing
        limitations.append(f"ComfyUI runs with its {cache_mode} cache, which keeps the outputs of earlier prompts: "
                           "each clip's prompt repeats in every round, so later rounds can reuse its encoding "
                           "(reported, and the claim is then invalid); time ComfyUI launched with --cache-classic")
    missing = [] if any(r["ok"] for r in all_rows) else ["every clip failed"]

    audio = clipset.uniform("audio")
    reported = {r.get("audio") for r in all_rows if r["ok"]}
    if audio is None and len(reported) == 1:  # not asked: take it from the engine when every clip says the same
        only = next(iter(reported))
        audio = only if isinstance(only, bool) else None
    workload = {"mode": "single", "batch": 1, "endpoint_kind": endpoint_kind, "clip_set": clipset.identity(),
                "seed_policy": policy, "seeds": [round_seeds[r] for r in sorted(round_seeds)], "audio": audio,
                "request_set": request_set(clipset, policy, seeds),
                **{k: clipset.uniform(k) for k in SHAPE_KEYS}}
    cpu = cpu_identity() if ep.is_local() and os.environ.get("INFERENCEKIT_CPU_LOCK_HELD") else None
    if cpu is not None:
        gpu = {}
    elif gpu_name:
        gpu = {"name": gpu_name, "count": gpu_count}
    elif ep.is_local():
        gpu = {**(gpu_identity() or {}), "count": gpu_count}
    else:
        gpu = {"count": gpu_count}
    gpu_missing = (["cpu identity"] if not cpu.get("model") else []) if cpu is not None else \
        ([] if gpu.get("name") else ["gpu identity (remote engine or no nvidia-smi: pass --gpu-name)"])
    ident = ev.identity(gpu=gpu, cpu=cpu, engine={"name": label, "version": engine_version}, model={"id": model},
                        workload=workload)
    a = table["all"]
    cm, cph = a.get("clip_ms") or {}, a.get("clips_per_hour") or {}
    ci = cm.get("ci95_median") or [None, None]
    summary = (f"{label} ({endpoint_kind}, one clip at a time, {len(rounds_table)} round(s) x {len(clipset.clips)} "
               f"clips): {_fmt_s(cm.get('median'))} per clip, mean over the set [observed, client timer]")
    if ci[0] is not None:
        summary += f" (95% CI {_fmt_s(ci[0])}-{_fmt_s(ci[1])})"
    if cph.get("median") is not None:
        summary += f", {cph['median']:.1f} clips/h [derived]"
    if failed:
        summary += f"; {len(failed)} failed"
    if not claim["valid"]:
        summary += f"; CLAIM INVALID ({len(claim['violations'])} violation(s))"
    data = {"source": "bench clip", "label": label, "base_url_kind": "local" if ep.is_local() else "remote",
            "backend": {**backend.describe(), "workflow_sha256": workflow_sha},
            "clip_set": {"name": clipset.name, "sha256": clipset.sha256, "n": len(clipset.clips),
                         "images": {k: v[1] for k, v in sorted(clipset.images.items())}},
            "per_task": table, "rounds": rounds_table, "warmup": warm, "claim_validity": claim,
            "order": {"ids": [c["id"] for c in order], "reordered": [c["id"] for c in order] != [
                c["id"] for c in clipset.clips], "rule": "the clip set's order unless neighbours (last and first "
                      "included) share a prompt or an i2v image; then the order with the fewest such neighbours"},
            "save_outputs": save_outputs, "extends": prev["evidence_id"] if prev else None,
            "outputs_from": ([prev["evidence_id"]] + list(prev["data"].get("outputs_from") or [])) if prev else [],
            "timer": "client perf_counter_ns",
            "metric_claim_classes": {"clip_ms": "observed (client timer)", "clips_per_hour": "derived",
                                     "clips_per_hour_per_gpu": "derived", "stage_*_ms": "self-reported by the engine "
                                     "(not verified; never a published timer)"},
            "definitions": {"clip_ms": "per clip: t_done - t_send on the client (t_send just before the connection "
                                       "opens, t_done once the whole clip is in the client's hands). Per round: "
                                       "the mean over the round's clips (data.rounds clip_ms; clip_ms_<task> over "
                                       "one task's). The record's value: the median over rounds, 95% CI by "
                                       "bootstrap over rounds; per_clip has the median and max over single clips",
                            "clips_per_hour": "3,600,000 x clips / sum of their clip_ms, per round (one clip at a "
                                              "time; clips_per_hour_<task> per task); the record's value is the "
                                              "median over rounds",
                            "warmup": "per_shape untimed clips for each distinct request shape, all before the "
                                      "first timed clip, with prompts no timed clip has",
                            "stage_sum_ok": "the engine's numeric stage_ms (totals and per-step lists left out) "
                                            "sum to no more than clip_ms"}}
    if run:
        p = run / "metrics" / f"{label}.clip.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"label": label, "speed": table, "rounds": rounds_table}, indent=1) + "\n")
    record = ev.build(rcall, kind=KIND, claim_class="observed", summary=summary, identity_=ident, metrics=mets,
                      artifacts=artifacts, missing=missing + gpu_missing, limitations=limitations, data=data)
    return store.save(record, staging)
