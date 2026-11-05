"""gates clip: anti-cheat gates for a video clip speed claim.

A clip time is only worth publishing if the engine made the clip it was asked for. These gates probe a
live clip engine (the clip HTTP contract, or ComfyUI through its adapter: serve/clip_client.py) and,
optionally, check a ``bench clip`` record's timed clips. Each gate ends PASS, FAIL, UNPROVEN or NOT RUN
with the probes and numbers that decided it:

  steps_honoured      the engine's own step counter (effective.steps_run) equals the steps asked on every
                      probe; a clip at N steps differs from the same request at N/2 steps; and N steps take
                      measurably longer than N/2 (median e2e ratio >= 1.10 in >= 2/3 of the pairs; FAIL below
                      1.03: the step count changes nothing). An engine that runs a fixed fraction of whatever
                      it is asked scales the same way; quality clip (outputs vs the reference at the same
                      steps) catches that, not this gate.
  cfg_honoured        the same request with another guidance scale (1.0, or +3 when the clip set's is <= 1.5)
                      gives a different clip. The time ratio is recorded, not judged (an engine may batch the
                      two guidance branches).
  resolution_frames   every clip is decoded (PyAV in the video venv with --python, else ffprobe): the decoded
                      frames' width and height, the decoded frame count, the fps from the decoded timestamps
                      and, when audio was asked for, the decoded audio duration (within 0.1 s + one frame of
                      num_frames / fps) must match the request. What the container header claims is recorded
                      beside it but never trusted.
  seed_honoured       a different seed (same prompt) gives a different clip, and the same seed repeated stays
                      within the A/A floor: FAIL when the different seed is no farther from the clip than the
                      repeat is, or when the repeat is more than half as far (the seed is not reproduced).
  prompt_sensitivity  a one-word change in the prompt (same seed) gives a different clip.
  no_cache            replaying a request is not > 10% faster than its first time (median over pairs and in
                      >= 2/3 of the pairs, as gates run's caching gate); different requests never get
                      byte-identical clips; no probe reports a stage served from a cache. With
                      --bench-evidence, the same holds for every timed clip of that bench clip run.

--bench-evidence: the record must verify (its id matches its content, its requests.jsonl matches the
sha256 it lists). Its timed clips are found in its own store directory, in those of the records it
extends (``data.outputs_from``: bench clip --extend and bench rounds' clip mode keep each round's clips
with the record that timed it) and in the run directory (``outputs/<label>/r<round>/<id>.<fmt>``), and
each file must have the sha256 its row records: one that does not is not the clip that was timed (FAIL);
a timed clip with no file anywhere leaves resolution_frames UNPROVEN. ``--max-decode N`` decodes a random
N of them, drawn from the probe seed.

"Different clip" is judged on decoded pixels when --python names a venv with PyAV and numpy: the mean
absolute difference (MAD, 0-255 levels) between the clips must exceed max(3 x the same-seed repeat's MAD,
0.5). Byte-identical clips are always a FAIL; without --python, clips whose bytes differ leave the gate
UNPROVEN (bytes can differ while the pictures do not).

The probes are one randomly interleaved sequence after an untimed warm-up: each repeat follows its first
request with other probes between them, seeds are drawn fresh for every run (and recorded), and the
one-word change is drawn from a pool, so no probe is predictable from this code. Probes use one clip of
the clip set (``--clip-id``, default the first): its task, shape, steps and guidance. A refused or failed
probe is evidence: it leaves its gates UNPROVEN, never PASS. Only when the engine answers no probe at all
are the live gates NOT RUN.

No probe follows a request with the same prompt or image file, so an engine never has a legitimate
encoding to reuse (ComfyUI's node cache keeps the last run's outputs: a text encoder whose prompt did not
change is skipped, and no_cache fails on any costly cached stage but the fixed negative prompt's). Each
request/replay pair is a group with its own prompt (the clip's for the first, then the clip's plus
" (probe N)") and, for i2v, its own image file (the same picture with a comment chunk: image_variant); the
probes compared with each other are in one group (N/2 steps with its pair, the other seed with the first,
the other guidance with the last), the one-word change has an image file of its own, and the warm-up has
a prompt and image file no probe has. With --pairs 1 the first group is too big to keep its probes apart:
those neighbours are recorded and named in no_cache.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import secrets
import time
import zlib
from pathlib import Path

from inferencekit import evidence as ev
from inferencekit.canonical import file_sha256
from inferencekit.ops import bench_clip as bc
from inferencekit.ops.bench_serve import _record_call
from inferencekit.ops.gates import _gate, _median, overall
from inferencekit.ops.quality_clip import probe_clips, run_worker
from inferencekit.proc import which
from inferencekit.registry import Arg, OpError, operation
from inferencekit.serve import clip_client as cc
from inferencekit.serve.client import Endpoint, api_key_from_env

GATES = ("steps_honoured", "cfg_honoured", "resolution_frames", "seed_honoured", "prompt_sensitivity", "no_cache")
LIVE = {"steps_honoured", "cfg_honoured", "seed_honoured", "prompt_sensitivity"}
# "clip_gates" is a newer evidence kind; on a store without it the record falls back to "gates".
KIND = "clip_gates" if "clip_gates" in ev.KINDS else "gates"
FORMAT = "inferencekit.gates.clip.v1"
STEP_RATIO_PASS = 1.10  # N steps must take >= 1.10x the time of N/2 ...
STEP_RATIO_FAIL = 1.03  # ... and below 1.03x the step count plainly changes nothing
SENSITIVITY = 3.0  # a changed input must move the clip > 3x as far as the same request repeated
MIN_MAD = 0.5  # ... and by more than half a grey level on average
REPRO_SHARE = 0.5  # the repeat may be at most half as far as a different seed
FPS_TOL = 0.01  # relative
AUDIO_TOL_S = 0.1  # plus one frame
SWAP_WORDS = ("crimson", "turquoise", "golden", "silver", "violet", "emerald", "amber", "cobalt", "scarlet",
              "ivory", "copper", "indigo", "lavender", "charcoal", "saffron")


# --------------------------------------------------------------------------- probes

@dataclasses.dataclass
class Probe:
    key: str
    kind: str  # base | replay | half_steps | other_seed | prompt | cfg
    req: cc.ClipRequest
    pair: int | None = None
    meta: dict = dataclasses.field(default_factory=dict)
    vs: str = "base0"  # other_seed, prompt, cfg: the probe this one is compared with
    res: cc.ClipResult | None = None
    sha: str | None = None
    path: Path | None = None
    t_start: float | None = None

    @property
    def ok(self) -> bool:
        return self.res is not None and self.res.ok and self.sha is not None

    @property
    def e2e_ms(self) -> float | None:
        return self.res.e2e_ms if self.ok else None


def one_word_change(prompt: str, rng: random.Random) -> tuple[str, dict]:
    """The prompt with one word replaced by a word from the pool (or one word put in front)."""
    words = prompt.split(" ")
    idx = [i for i, w in enumerate(words) if w.isalpha() and len(w) >= 4]
    if idx:
        i = rng.choice(idx)
        old = words[i]
        new = rng.choice([w for w in SWAP_WORDS if w != old.lower()])
        words[i] = new.capitalize() if old[:1].isupper() else new
        return " ".join(words), {"replaced": old, "with": words[i], "word_index": i}
    new = rng.choice(SWAP_WORDS)
    return f"{new} {prompt}", {"inserted": new}


def alt_guidance(g: float) -> float:
    return 1.0 if g > 1.5 else round(g + 3.0, 6)


PROBE_SUFFIX = " (probe {n})"  # the prompts of probe groups after the first
PNG_SIG = b"\x89PNG\r\n\x1a\n"


def image_variant(data: bytes, tag: str) -> bytes | None:
    """The same picture as another file: ``tag`` in a comment that decoders skip (a PNG tEXt chunk after IHDR,
    a JPEG COM segment after SOI and APP0). None for any other format."""
    text = tag.encode("latin-1")
    if data.startswith(PNG_SIG) and data[12:16] == b"IHDR":
        end = 8 + 12 + int.from_bytes(data[8:12], "big")  # the signature, then IHDR: length, type, data, crc
        body = b"Comment\x00" + text
        chunk = len(body).to_bytes(4, "big") + b"tEXt" + body + zlib.crc32(b"tEXt" + body).to_bytes(4, "big")
        return data[:end] + chunk + data[end:]
    if data.startswith(b"\xff\xd8\xff"):
        at = 4 + int.from_bytes(data[4:6], "big") if data[2:4] == b"\xff\xe0" else 2  # JFIF's APP0 stays first
        return data[:at] + b"\xff\xfe" + (len(text) + 2).to_bytes(2, "big") + text + data[at:]
    return None


def clash(a: cc.ClipRequest, b: cc.ClipRequest) -> bool:
    """Requests in a row that share a prompt or an image file: the second could reuse an encoding of the first
    (ComfyUI's node cache keeps the last run's outputs)."""
    return a.prompt == b.prompt or (a.image is not None and a.image == b.image)


def plan_probes(clip: dict, clipset, rng: random.Random, pairs: int) -> list[Probe]:
    """Probes in their first-seen roles; ``order`` interleaves them.

    Each request/replay pair is a group with an input of its own: the clip's prompt for group 0, the prompt
    plus " (probe N)" for the others, and for i2v the clip's image file for group 0 and the same picture as
    another file (image_variant) for the others. The probes compared with each other share their group's
    inputs: N/2 steps with its pair, the other seed with group 0, the other guidance with the last group. The
    one-word change gets a variant image of its own. So ``order`` can keep every request from following one
    with the same prompt or image file, and no engine can serve a probe's encoding from the probe before it."""
    seeds: list[int] = []
    while len(seeds) < pairs + 1:
        s = rng.getrandbits(31)
        if s not in seeds:
            seeds.append(s)
    probes = []
    half = clip["steps"] // 2
    first = bc.make_request(clip, seeds[0], clipset)
    prompts = [first.prompt] + [first.prompt + PROBE_SUFFIX.format(n=i + 1) for i in range(1, pairs)]
    for i in range(pairs):
        image = first.image
        if image is not None and i:
            image = image_variant(first.image, f"inferencekit probe group {i}") or first.image
        base = dataclasses.replace(first, id=f"base{i}", seed=seeds[i], prompt=prompts[i], image=image)
        probes.append(Probe(f"base{i}", "base", base, pair=i))
        probes.append(Probe(f"replay{i}", "replay", dataclasses.replace(base, id=f"replay{i}"), pair=i))
        if half >= 1:
            probes.append(Probe(f"half{i}", "half_steps", dataclasses.replace(base, id=f"half{i}", steps=half),
                                pair=i, meta={"steps": half}))
    b0, last = probes[0].req, next(p.req for p in probes if p.key == f"base{pairs - 1}")
    probes.append(Probe("seed", "other_seed", dataclasses.replace(b0, id="seed", seed=seeds[pairs])))
    prompt, how = one_word_change(b0.prompt, rng)
    image = (image_variant(b0.image, "inferencekit probe prompt") or b0.image) if b0.image is not None else None
    probes.append(Probe("prompt", "prompt", dataclasses.replace(b0, id="prompt", prompt=prompt, image=image),
                        meta=how))
    g = alt_guidance(last.guidance_scale)
    probes.append(Probe("cfg", "cfg", dataclasses.replace(last, id="cfg", guidance_scale=g),
                        meta={"guidance_scale": g}, vs=f"base{pairs - 1}"))
    return probes


def warmup_request(b0: cc.ClipRequest, i: int, seed: int, probes: list[Probe]) -> cc.ClipRequest:
    """An untimed warm-up at the probe shape with a prompt (and an image file) no probe has."""
    image = (image_variant(b0.image, "inferencekit warm-up") or b0.image) if b0.image is not None else None
    return dataclasses.replace(b0, id=f"warmup{i}", seed=seed, image=image,
                               prompt=bc.warmup_prompt(b0.prompt, {p.req.prompt for p in probes}))


def neighbour_clashes(seq: list[Probe]) -> list[list[str]]:
    """[key, next key] of probes in a row that share a prompt or an image file."""
    return [[a.key, b.key] for a, b in zip(seq, seq[1:]) if clash(a.req, b.req)]


def order(probes: list[Probe], rng: random.Random, tries: int = 200) -> list[Probe]:
    """A random interleave that keeps each repeat after its first request with others between them, and no
    probe right after one with the same prompt or image file; when no such order is found (one pair has
    too few other probes to go between its own), the one with the fewest such neighbours."""
    by = {p.key: p for p in probes}
    best = None
    for _ in range(tries):
        chains = [[by[f"base{p.pair}"], p] for p in probes if p.kind == "replay"]
        chains += [[p] for p in probes if p.kind not in ("base", "replay")]
        seq: list[Probe] = []
        while chains:  # a random chain head each step, one that does not clash with the last probe when possible
            free = [c for c in chains if not (seq and clash(seq[-1].req, c[0].req))]
            c = rng.choice(free or chains)
            seq.append(c.pop(0))
            chains = [x for x in chains if x]
        adjacent = sum(1 for a, b in zip(seq, seq[1:]) if b.kind == "replay" and a.key == f"base{b.pair}")
        score = (adjacent, len(neighbour_clashes(seq)))
        if best is None or score < best[0]:
            best = (score, seq)
        if score == (0, 0):
            break
    return best[1]


def run_probes(backend, seq: list[Probe], staging: Path, label: str) -> None:
    for p in seq:
        p.t_start = time.time()
        p.res = backend.generate(p.req)
        if p.res.ok and p.res.video is not None:
            p.sha = hashlib.sha256(p.res.video).hexdigest()
            p.path = staging / "outputs" / label / "probes" / f"{p.key}.{p.res.format}"
            p.path.parent.mkdir(parents=True, exist_ok=True)
            p.path.write_bytes(p.res.video)
            p.res.video = None  # on disk now


def probe_row(p: Probe, decoded: dict | None) -> dict:
    body = p.req.body()
    if "image" in body:
        body["image"] = {"sha256": hashlib.sha256(p.req.image).hexdigest(), "bytes": len(p.req.image)}
    r = p.res
    return {"key": p.key, "kind": p.kind, "pair": p.pair, "meta": p.meta, "request": body, "ok": p.ok,
            "status": r.status if r else None, "error": (r.error if r else "not sent"),
            "e2e_ms": p.e2e_ms, "out_sha256": p.sha, "format": r.format if r and r.ok else None,
            "effective": r.effective if r else None, "stage_ms": r.stage_ms if r else None,
            "cache": r.cache if r else None, "decoded": decoded, "t_start": p.t_start}


# --------------------------------------------------------------------------- distances

DISTANCE_PAIRS = (("same", "base", "replay"), ("steps", "base", "half"))


def distance_plan(probes: dict[str, Probe], pairs: int) -> list[tuple[str, str, str]]:
    """(name, a, b) clip pairs whose pixel distance the gates need."""
    out = []
    for i in range(pairs):
        out.append((f"same{i}", f"base{i}", f"replay{i}"))
        if f"half{i}" in probes:
            out.append((f"steps{i}", f"base{i}", f"half{i}"))
    out += [(k, probes[k].vs, k) for k in ("seed", "prompt", "cfg")]
    return [x for x in out if probes[x[1]].ok and probes[x[2]].ok and probes[x[1]].sha != probes[x[2]].sha]


def distances(probes: dict[str, Probe], plan: list[tuple[str, str, str]], python: str | None, work: Path,
              timeout_s: float) -> tuple[dict | None, str | None]:
    """{name: {"mad", "psnr", "frames_compared", "ok", "error"}} on decoded pixels, or (None, why) without the
    worker."""
    if not python:
        return None, "no --python: clips were compared by bytes only"
    if not plan:
        return {}, None
    doc = run_worker(python, {"task": "compare", "metrics": ["mad", "psnr"], "frame_stride": 1,
                              "pairs": [{"key": n, "a": str(probes[a].path), "b": str(probes[b].path)}
                                        for n, a, b in plan]}, work, "distances", timeout_s)
    out = {}
    for it in doc["items"]:
        mean = it.get("mean") or {}
        out[it["key"]] = {"ok": bool(it.get("ok")) and mean.get("mad") is not None, "mad": mean.get("mad"),
                          "psnr": mean.get("psnr"), "frames_compared": it.get("frames_compared"),
                          "frames": [it.get("frames_a"), it.get("frames_b")], "error": it.get("error")}
    return out, None


class Judge:
    """Shared facts for the gates: probes by key, pixel distances, the same-seed A/A floor, and the probes run
    right after one with the same prompt or image file (``clashes``)."""

    def __init__(self, probes: dict[str, Probe], dist: dict | None, pairs: int, clashes: list | None = None):
        self.p, self.dist, self.pairs, self.clashes = probes, dist, pairs, list(clashes or [])
        self.floor_rows = []
        for i in range(pairs):
            a, b = probes[f"base{i}"], probes[f"replay{i}"]
            if not (a.ok and b.ok):
                continue
            if a.sha == b.sha:
                self.floor_rows.append({"pair": i, "identical_bytes": True, "mad": 0.0})
            elif dist is not None and (dist.get(f"same{i}") or {}).get("ok"):
                self.floor_rows.append({"pair": i, "identical_bytes": False, "mad": dist[f"same{i}"]["mad"],
                                        "psnr": dist[f"same{i}"]["psnr"]})
            else:
                self.floor_rows.append({"pair": i, "identical_bytes": False, "mad": None})
        known = [r["mad"] for r in self.floor_rows]
        # The largest repeat distance; None when no repeat was judged or one could not be compared.
        self.floor = max(known) if known and None not in known else None

    def differs(self, name: str, a: str, b: str) -> dict:
        """How clip b differs from clip a: same_bytes | within_floor | differs | small | bytes_only | missing."""
        pa, pb = self.p[a], self.p[b]
        row = {"name": name, "a": a, "b": b}
        if not (pa.ok and pb.ok):
            return {**row, "state": "missing"}
        if pa.sha == pb.sha:
            return {**row, "state": "same_bytes", "mad": 0.0}
        if self.dist is None:
            return {**row, "state": "bytes_only"}
        d = self.dist.get(name) or {}
        if not d.get("ok"):
            return {**row, "state": "missing", "error": d.get("error") or "not compared"}
        row.update({"mad": d["mad"], "psnr": d["psnr"]})
        if self.floor is None:
            return {**row, "state": "small" if d["mad"] <= MIN_MAD else "no_floor"}
        if d["mad"] <= self.floor:
            return {**row, "state": "within_floor"}
        if d["mad"] > max(SENSITIVITY * self.floor, MIN_MAD):
            return {**row, "state": "differs"}
        return {**row, "state": "small"}


def _verdict_of(rows: list[dict]) -> tuple[str, list[dict]]:
    """FAIL when any changed input left the clip the same; PASS when every one moved it clearly."""
    bad = [r for r in rows if r["state"] in ("same_bytes", "within_floor")]
    if bad:
        return "FAIL", bad
    if rows and all(r["state"] == "differs" for r in rows):
        return "PASS", []
    return "UNPROVEN", [r for r in rows if r["state"] != "differs"]


STATE_TEXT = {"same_bytes": "byte-identical clips", "within_floor": "no farther apart than the same request repeated",
              "small": "the clips barely differ (not above 3x the repeat floor and 0.5 grey levels)",
              "bytes_only": "bytes differ but pixels were not compared (pass --python)",
              "no_floor": "no same-seed repeat to compare against", "missing": "a probe failed or was not compared"}


def _explain(rows: list[dict]) -> str:
    return "; ".join(f"{r['b']} vs {r['a']}: {STATE_TEXT.get(r['state'], r['state'])}"
                     + (f" (MAD {r['mad']:.3g})" if r.get("mad") is not None else "") for r in rows[:4])


def _failed_note(probes: list[Probe]) -> str | None:
    bad = [p for p in probes if not p.ok]
    if not bad:
        return None
    first = next((p.res.error for p in bad if p.res is not None and p.res.error), "not sent")
    return f"{len(bad)} probe(s) failed (first: {str(first)[:160]})"


# --------------------------------------------------------------------------- the gates

def gate_steps(j: Judge, steps: int) -> dict:
    if steps < 2:
        return _gate("steps_honoured", "NOT RUN", "the probe clip asks for fewer than 2 steps: no N/2 to compare")
    probes = list(j.p.values())
    counter = [{"key": p.key, "asked": p.req.steps, "steps_run": (p.res.effective or {}).get("steps_run")}
               for p in probes if p.ok]
    reported = [c for c in counter if c["steps_run"] is not None]
    wrong = [c for c in reported if c["steps_run"] != c["asked"]]
    rows = [j.differs(f"steps{i}", f"base{i}", f"half{i}") for i in range(j.pairs)]
    out_v, out_bad = _verdict_of(rows)
    ratios = []
    for i in range(j.pairs):
        a, h = j.p[f"base{i}"], j.p[f"half{i}"]
        if a.e2e_ms and h.e2e_ms:
            ratios.append({"pair": i, "e2e_ms": a.e2e_ms, "half_e2e_ms": h.e2e_ms, "ratio": a.e2e_ms / h.e2e_ms,
                           "denoise_ratio": _stage_ratio(a, h, "denoise")})
    rs = [r["ratio"] for r in ratios]
    med = _median(rs)
    typical = bool(rs) and 3 * sum(1 for x in rs if x >= STEP_RATIO_PASS) >= 2 * len(rs)
    metrics = {"step_time_ratio_median": med, "pairs": len(rs), "steps_reported": len(reported),
               "steps_mismatch": len(wrong), "half_steps": steps // 2}
    details = {"counter": counter, "outputs": rows, "timing": ratios}
    lim = ["an engine that runs a fixed fraction of the steps it is asked for scales like an honest one here; "
           "quality clip against the reference at the same steps catches it"]
    note = _failed_note([j.p[k] for k in j.p if k.startswith(("base", "half"))])
    if note:
        lim.append(note)
    if wrong:
        return _gate("steps_honoured", "FAIL", f"the engine reports steps_run != asked on {len(wrong)} probe(s) "
                     f"(e.g. {wrong[0]['key']}: asked {wrong[0]['asked']}, ran {wrong[0]['steps_run']})",
                     metrics, details, lim)
    if out_v == "FAIL":
        return _gate("steps_honoured", "FAIL", f"N/2 steps gave the same clip as N steps: {_explain(out_bad)}",
                     metrics, details, lim)
    if med is not None and med < STEP_RATIO_FAIL:
        return _gate("steps_honoured", "FAIL", f"{steps} steps take {med:.3f}x the time of {steps // 2} (median "
                     f"of {len(rs)} pair(s)): the step count does not change the work done", metrics, details, lim)
    problems = []
    if not reported:
        problems.append("the engine reports no steps_run (effective) to check")
    if out_v != "PASS":
        problems.append(f"outputs: {_explain(out_bad)}")
    if med is None:
        problems.append("no N vs N/2 pair was timed")
    elif not (med >= STEP_RATIO_PASS and typical):
        problems.append(f"time ratio N vs N/2 {med:.3f}x (median of {len(rs)}) is below {STEP_RATIO_PASS}x or not "
                        "in 2/3 of the pairs")
    if problems:
        return _gate("steps_honoured", "UNPROVEN", "; ".join(problems), metrics, details, lim)
    return _gate("steps_honoured", "PASS", f"steps_run == asked on {len(reported)} probe(s); N/2 steps changes the "
                 f"clip; {steps} steps take {med:.2f}x the time of {steps // 2}", metrics, details, lim)


def _stage_ratio(a: Probe, b: Probe, stage: str) -> float | None:
    sa, sb = (a.res.stage_ms or {}).get(stage), (b.res.stage_ms or {}).get(stage)
    ok = all(isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0 for x in (sa, sb))
    return sa / sb if ok else None


def gate_changed(j: Judge, name: str, probe: str, what: str) -> dict:
    """cfg_honoured / prompt_sensitivity: one changed input, same seed, must change the clip."""
    a, b = j.p[j.p[probe].vs], j.p[probe]
    rows = [j.differs(probe, a.key, probe)]
    v, bad = _verdict_of(rows)
    metrics = {"mad": rows[0].get("mad"), "psnr": rows[0].get("psnr"), "repeat_floor_mad": j.floor}
    details = {"change": b.meta, "outputs": rows, "floor": j.floor_rows}
    if name == "cfg_honoured" and a.e2e_ms and b.e2e_ms:
        metrics["time_ratio"] = a.e2e_ms / b.e2e_ms
    lim = [x for x in [_failed_note([a, b])] if x]
    if v == "PASS":
        return _gate(name, "PASS", f"{what} changed the clip (MAD {rows[0]['mad']:.3g} vs the repeat's "
                     f"{j.floor:.3g})", metrics, details, lim)
    return _gate(name, v, f"{what}: {_explain(bad)}", metrics, details, lim)


def gate_seed(j: Judge) -> dict:
    rows = [j.differs("seed", j.p["seed"].vs, "seed")]
    v, bad = _verdict_of(rows)
    mads = [r["mad"] for r in rows if r.get("mad") is not None and r["state"] != "same_bytes"]
    metrics = {"repeat_floor_mad": j.floor, "different_seed_mad_min": min(mads) if mads else None,
               "repeats": len(j.floor_rows), "different_seed_pairs": len(rows)}
    details = {"floor": j.floor_rows, "different_seeds": rows}
    lim = [x for x in [_failed_note([p for k, p in j.p.items() if k.startswith(("base", "replay")) or k == "seed"])]
           if x]
    same = [r for r in rows if r["state"] == "same_bytes"]
    if same:  # checked first: definitive whatever the floor is
        return _gate("seed_honoured", "FAIL", f"a different seed did not change the clip: {_explain(same)}",
                     metrics, details, lim)
    if j.floor is not None and mads and j.floor > REPRO_SHARE * min(mads):
        return _gate("seed_honoured", "FAIL", f"the same seed does not reproduce the clip: the repeat is MAD "
                     f"{j.floor:.3g} from the first, a different seed {min(mads):.3g}", metrics, details, lim)
    if v == "FAIL":
        return _gate("seed_honoured", "FAIL", f"a different seed did not change the clip: {_explain(bad)}",
                     metrics, details, lim)
    if v == "PASS" and j.floor is not None:
        return _gate("seed_honoured", "PASS", f"different seeds change the clip (MAD >= {min(mads):.3g}); the "
                     f"same seed repeats within MAD {j.floor:.3g}", metrics, details, lim)
    why = _explain(bad) if bad else "no same-seed repeat could be compared"
    return _gate("seed_honoured", "UNPROVEN", why, metrics, details, lim)


def check_decoded(d: dict | None, asked: dict, audio: bool | None) -> list[str]:
    """What the decoded clip gets wrong against the request (empty = all good)."""
    if not d or not d.get("ok"):
        return []
    v, problems = d.get("video") or {}, []
    if not v.get("frames"):
        return ["no video frames decoded"]
    if (v.get("width"), v.get("height")) != (asked["width"], asked["height"]):
        problems.append(f"decoded {v.get('width')}x{v.get('height')}, asked {asked['width']}x{asked['height']}")
    if len(v.get("sizes") or []) > 1:
        problems.append(f"frame size changes inside the clip: {v['sizes'][:4]}")
    if v["frames"] != asked["num_frames"]:
        problems.append(f"decoded {v['frames']} frames, asked {asked['num_frames']}")
    if v.get("fps") is not None and abs(v["fps"] - asked["fps"]) > FPS_TOL * asked["fps"]:
        problems.append(f"decoded timestamps give {v['fps']:.4g} fps, asked {asked['fps']:g}")
    if audio:
        a = d.get("audio") or {}
        want = asked["num_frames"] / asked["fps"]
        if not a.get("duration_s"):
            problems.append("audio was asked for but none was decoded")
        elif abs(a["duration_s"] - want) > AUDIO_TOL_S + 1 / asked["fps"]:
            problems.append(f"decoded audio is {a['duration_s']:.3f} s, the clip is {want:.3f} s")
    return problems


def header_claims(d: dict | None) -> list[str]:
    """Where the container header disagrees with what decoding gave (recorded, never trusted)."""
    if not d or not d.get("ok"):
        return []
    hv, v, out = (d.get("header") or {}).get("video") or {}, d.get("video") or {}, []
    for k in ("width", "height", "frames"):
        if hv.get(k) is not None and v.get(k) is not None and hv[k] != v[k]:
            out.append(f"header {k} {hv[k]} vs decoded {v[k]}")
    return out


def gate_resolution(items: list[dict], decoder: str | None, why_not: str | None, left_out: int = 0) -> dict:
    """items: {key, asked, audio, decoded, source, problems?, error?}: ``problems`` found before decoding (a
    file that is not the timed clip) fail the clip; ``error`` says why a clip has no decode (no file)."""
    if not items:
        return _gate("resolution_frames", "NOT RUN", why_not or "no clip to decode")
    rows, bad, undecoded = [], [], []
    for it in items:
        d = it["decoded"]
        probs = list(it.get("problems") or []) + check_decoded(d, it["asked"], it["audio"])
        row = {"key": it["key"], "source": it["source"], "problems": probs, "header_mismatch": header_claims(d),
               "decoded": {k: (d.get("video") or {}).get(k) for k in ("width", "height", "frames", "fps")}
               if d and d.get("ok") else None,
               "audio_s": ((d or {}).get("audio") or {}).get("duration_s") if d and d.get("ok") else None}
        if probs:
            bad.append(row)
        elif not d or not d.get("ok"):
            why = (d or {}).get("error") or it.get("error") or (f"not decoded: {why_not}" if why_not else "not decoded")
            undecoded.append({"key": it["key"], "error": why})
        rows.append(row)
    decoded = sum(1 for r in rows if r["decoded"] is not None)
    metrics = {"clips_decoded": decoded, "clips_wrong": len(bad), "clips_undecoded": len(undecoded),
               "clips_left_out": left_out}
    lim = [f"decoded with {decoder}"] if decoder else []
    if left_out:
        lim.append(f"--max-decode: {left_out} timed clip(s) of the bench clip run were not decoded (a random subset "
                   "was, drawn from the probe seed)")
    if bad:
        return _gate("resolution_frames", "FAIL", f"{len(bad)} clip(s) are not what was asked (e.g. {bad[0]['key']}: "
                     f"{'; '.join(bad[0]['problems'])})", metrics, rows, lim)
    if undecoded:
        return _gate("resolution_frames", "UNPROVEN", f"{len(undecoded)} clip(s) could not be decoded (e.g. "
                     f"{undecoded[0]['key']}: {undecoded[0]['error'][:160]})", metrics, rows, lim)
    return _gate("resolution_frames", "PASS", f"{decoded} decoded clip(s) match the asked size, frame count, fps"
                 + (" and audio length" if any(it["audio"] for it in items) else ""), metrics, rows, lim)


def gate_no_cache(j: Judge | None, threshold: float, static: dict | None) -> dict:
    problems, unproven, rows, metrics, lim = [], [], [], {}, []
    if j is not None:
        live = list(j.p.values())
        cached = [{"key": p.key, "cache": p.res.cache} for p in live if p.res is not None and p.res.cache]
        if cached:
            after = {b for _, b in j.clashes} & {c["key"] for c in cached}
            problems.append(f"{len(cached)} probe(s) report stages served from a cache (e.g. {cached[0]['key']}: "
                            f"{cached[0]['cache'][:4]})" + (
                                f"; {len(after)} of them ran right after a probe with the same prompt or image file "
                                "(no order without such neighbours was found: use --pairs 2 or more)" if after else ""))
        reused = [p.key for p in live if p.res is not None and p.res.cache_allowed]
        if reused:
            lim.append(f"{len(reused)} probe(s): the engine reused the encoding of its fixed negative prompt "
                       f"({cc.NEGATIVE_PROMPT_CACHE}: computed once, not per clip; allowed for every engine)")
        sp = []
        for i in range(j.pairs):
            a, b = j.p[f"base{i}"], j.p[f"replay{i}"]
            if a.e2e_ms and b.e2e_ms:
                sp.append(a.e2e_ms / b.e2e_ms)
                rows.append({"pair": i, "first_e2e_ms": a.e2e_ms, "replay_e2e_ms": b.e2e_ms,
                             "speedup": a.e2e_ms / b.e2e_ms, "same_bytes": a.sha == b.sha})
        med = _median(sp)
        most = bool(sp) and 3 * sum(1 for x in sp if x > 1 + threshold) >= 2 * len(sp)
        metrics.update({"replay_speedup_median": med, "replay_pairs": len(sp), "reported_cached": len(cached),
                        "neighbour_clashes": len(j.clashes)})
        if j.clashes:
            lim.append(f"{len(j.clashes)} probe(s) ran right after one with the same prompt or image file (e.g. "
                       f"{j.clashes[0][0]} -> {j.clashes[0][1]}): an engine that keeps the last run's encodings "
                       "(ComfyUI) reuses them there; use --pairs 2 or more")
        if med is not None and med > 1 + threshold and most:
            problems.append(f"replays are {med:.2f}x faster than their first request (median of {len(sp)}; limit "
                            f"{1 + threshold:.2f}x)")
        elif not sp:
            unproven.append("no request and its replay both succeeded")
        distinct = [p for p in live if p.kind != "replay" and p.ok]
        dup = _duplicates({p.key: p.sha for p in distinct})
        metrics["distinct_probe_clips"] = len(distinct)
        if dup:
            problems.append(f"different requests got byte-identical clips: {dup[:3]} (an output cache, or an input "
                            "the engine ignores)")
        note = _failed_note(live)
        if note:
            unproven.append(note)
    if static is not None:
        metrics.update({"bench_clips": static["n"], "bench_duplicates": len(static["duplicates"]),
                        "bench_cached": len(static["cached"])})
        if static["duplicates"]:
            problems.append(f"bench clip: different requests got byte-identical clips: {static['duplicates'][:3]}")
        if static["cached"]:
            problems.append(f"bench clip: {len(static['cached'])} timed clip(s) report cached stages (e.g. "
                            f"{static['cached'][0]})")
    details = {"replays": rows, "static": static, "neighbour_clashes": j.clashes if j is not None else None}
    if j is None and static is None:
        return _gate("no_cache", "NOT RUN", "no live probes and no --bench-evidence")
    if problems:
        return _gate("no_cache", "FAIL", "; ".join(problems), metrics, details, lim)
    if unproven:
        return _gate("no_cache", "UNPROVEN", "; ".join(unproven), metrics, details, lim)
    parts = []
    if j is not None:
        parts.append(f"replays {metrics['replay_speedup_median']:.2f}x (limit {1 + threshold:.2f}x); "
                     f"{metrics['distinct_probe_clips']} distinct requests, distinct clips")
    if static is not None:
        parts.append(f"bench clip: {static['n']} timed clip(s), no duplicate or cached clip")
    return _gate("no_cache", "PASS", "; ".join(parts), metrics, details, lim)


def _duplicates(shas: dict[str, str]) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for k, s in shas.items():
        groups.setdefault(s, []).append(k)
    return [sorted(v) for v in groups.values() if len(v) > 1]


# --------------------------------------------------------------------------- a bench clip record

def load_bench(store: ev.Store, ref: str, run_dir: str | None = None) -> dict:
    """Rows, the directories its clips may be in, and the asked audio of a bench clip record.

    Refused when the record or its requests.jsonl was edited after capture: a gates record built from an
    edited input would carry a fresh id that verifies."""
    rec = store.load(ref)
    data = rec.get("data") or {}
    if data.get("source") != "bench clip":
        raise OpError(f"--bench-evidence {ref}: not a bench clip record")
    problem = ev.integrity_problem(rec)
    if problem:
        raise OpError(f"--bench-evidence: {problem}; refusing to check it")
    root = store.run_dir(rec["evidence_id"])
    art = next((a for a in rec.get("artifacts", []) if a.get("role") == "requests"), None)
    path = root / art["path"] if art and isinstance(art.get("path"), str) else None
    if path is None or not path.is_file():
        raise OpError(f"--bench-evidence {ref}: its requests.jsonl is not in the store ({root})")
    if file_sha256(path)[0] != art.get("sha256"):
        raise OpError(f"--bench-evidence {ref}: its requests.jsonl does not match the sha256 the record lists "
                      "(edited after capture?); refusing to check it")
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    # Where the timed clips may be: this record's directory, those of the records it extends (each keeps the
    # rounds it timed), and the run directories (bench clip's and this call's --run-dir).
    roots = [root] + [store.run_dir(e) for e in data.get("outputs_from") or [] if isinstance(e, str)
                      and e.startswith("ev_") and "/" not in e and ".." not in e]
    bench_run = ((data.get("request") or {}).get("args") or {}).get("run_dir")
    for rd in (run_dir, bench_run):  # records keep paths under the home directory as ~/...
        d = Path(rd).expanduser() if isinstance(rd, str) and rd else None
        if d is not None and d.is_dir() and d not in roots:
            roots.append(d)
    audio = (rec.get("identity") or {}).get("workload", {}).get("audio")
    return {"record": rec, "root": root, "roots": roots, "rows": rows, "label": data.get("label"),
            "hash_only": data.get("save_outputs") == "hash_only", "audio": audio if isinstance(audio, bool) else None}


def static_cache(bench: dict) -> dict:
    """Different requests (clip id, seed) must never share a clip; no timed clip may report a cached stage."""
    shas: dict[str, set] = {}
    for r in bench["rows"]:
        if r.get("ok") and r.get("out_sha256"):
            shas.setdefault(r["out_sha256"], set()).add(f"{r['id']}@seed{r['seed']}")
    dup = [sorted(v) for v in shas.values() if len(v) > 1]
    cached = [f"{r['id']} r{r['round']}" for r in bench["rows"] if r.get("ok") and r.get("cache")]
    return {"n": sum(1 for r in bench["rows"] if r.get("ok")), "duplicates": dup, "cached": cached}


def bench_clips(bench: dict, limit: int, rng: random.Random) -> tuple[list[dict], int]:
    """The timed clips of a bench clip record to decode: [{key, path, asked, state, error}] with state ok (the
    file has the row's sha256), changed (files found, none with it) or lost (no file), and how many timed
    clips ``limit`` left out (0 = decode all; else a random ``limit`` of them)."""
    timed = [r for r in bench["rows"] if r.get("ok") and r.get("out_sha256")]
    chosen = timed
    if limit and len(timed) > limit:
        chosen = [timed[i] for i in sorted(rng.sample(range(len(timed)), limit))]
    out = []
    for r in chosen:
        key = f"bench:r{r.get('round')}/{r.get('id')}"
        rel = Path("outputs", str(bench["label"]), f"r{r.get('round')}", f"{r.get('id')}.{r.get('format')}")
        it = {"key": key, "path": None, "asked": r.get("asked"), "state": "lost", "error": None}
        if not (r.get("format") and isinstance(r.get("asked"), dict)) or rel.is_absolute() or ".." in rel.parts:
            out.append({**it, "error": "the row names no usable clip file"})
            continue
        found = [root / rel for root in bench["roots"] if (root / rel).is_file()]
        hit = next((f for f in found if file_sha256(f)[0] == r["out_sha256"]), None)
        if hit is not None:
            out.append({**it, "path": hit, "state": "ok"})
        elif found:
            out.append({**it, "state": "changed", "error": "clip file is not the clip that was timed (its sha256 "
                        "differs from the row's out_sha256)"})
        else:
            out.append({**it, "error": "no clip file kept (--save-outputs hash_only)" if bench["hash_only"] else
                        "no clip file in the record's store directory, the records it extends or the run directory"})
    return out, len(timed) - len(chosen)


# --------------------------------------------------------------------------- the operation

@operation("gates_clip", ("gates", "clip"), "Anti-cheat gates for a video clip engine: steps and guidance honoured, "
           "decoded resolution / frame count / fps / audio length, seeds and prompts change the clip, no cached "
           "outputs (replay timing, unique clips); optionally the timed clips of a bench clip record.",
           [Arg("base_url", "string", "Engine root, e.g. http://127.0.0.1:8000 (the live gates need it). Never "
                "stored: records keep an HMAC of it (as bench clip does).", sensitive=True),
            Arg("clips", "string", "Clip set JSONL (as bench clip): the probes use one of its clips."),
            Arg("clip_id", "string", "Clip of the set the probes are built from (default: the first)."),
            Arg("endpoint_kind", "string", "clip_http (POST /v1/clips) or comfyui (needs --workflow).",
                default="clip_http", choices=cc.ENDPOINT_KINDS),
            Arg("workflow", "string", "comfyui: the API-format workflow JSON."),
            Arg("label", "string", "Engine label for the record (default: the endpoint kind)."),
            Arg("model", "string", "Model id for the record's identity."),
            Arg("engine_version", "string", "Engine version, recorded in the identity."),
            Arg("bench_evidence", "string", "Evidence id of a bench clip run (with --extend, its last record): "
                "its timed clips, every round, are hash-checked, decoded and checked for duplicates and cached "
                "stages."),
            Arg("max_decode", "integer", "Decode at most this many of the bench clip run's clips, a random subset "
                "drawn from the probe seed (0 = all).", default=0),
            Arg("python", "string", "Python of the video pipeline's venv (PyAV + numpy): decodes clips and "
                "compares their pixels. Without it, ffprobe decodes and clips are compared by bytes only."),
            Arg("pairs", "integer", "Request/replay pairs (each also gets an N/2-steps probe).", default=2),
            Arg("warmup", "integer", "Untimed warm-up clips at the probe shape before the probes.", default=1),
            Arg("threshold", "number", "no_cache: largest allowed replay speedup - 1.", default=0.10),
            Arg("seed", "integer", "Probe seed (default: drawn at random for every run and recorded; pass it only "
                "to reproduce a run)."),
            Arg("api_key_env", "string", "Name of the environment variable holding the API key (never stored)."),
            Arg("timeout_s", "number", "Per-clip timeout, seconds.", default=cc.DEFAULT_TIMEOUT_S),
            Arg("run_dir", "string", "Run directory: the record is also copied to <run-dir>/evidence/.")])
def gates_clip(call, base_url, clips, clip_id, endpoint_kind, label, model, engine_version, bench_evidence,
               max_decode, python, pairs, warmup, threshold, seed, api_key_env, timeout_s, workflow, run_dir):
    if not base_url and not bench_evidence:
        raise OpError("give --base-url (live probes, with --clips) and/or --bench-evidence (a bench clip run)")
    if base_url and not clips:
        raise OpError("--base-url needs --clips (the clip set the probes are built from)")
    if pairs < 1:
        raise OpError("pairs must be >= 1")
    if warmup < 0 or max_decode < 0:
        raise OpError("warmup and max_decode must be >= 0")
    if endpoint_kind == "comfyui" and base_url and not workflow:
        raise OpError("--endpoint-kind comfyui needs --workflow (the API-format workflow JSON)")
    label = label or endpoint_kind
    store = ev.Store()
    bench = load_bench(store, bench_evidence, run_dir) if bench_evidence else None
    rcall, ep, clipset, clip = call, None, None, None
    if base_url:
        clipset = bc.load_clip_set(clips)
        clip = next((c for c in clipset.clips if c["id"] == clip_id), None) if clip_id else clipset.clips[0]
        if clip is None:
            raise OpError(f"--clip-id {clip_id!r} is not in the clip set")
        try:
            ep = Endpoint.parse(base_url, api_key_from_env(api_key_env))
        except ValueError as e:
            raise OpError(str(e)) from None
        rcall, _ = _record_call(call, ep)
    probe_seed = seed if seed is not None else secrets.randbits(31)
    rng = random.Random(probe_seed)

    with store.staging_dir() as staging:
        probes: dict[str, Probe] = {}
        warm, judge, dist, why_bytes, seq = [], None, None, None, []
        if ep is not None:
            try:
                backend = cc.make_backend(endpoint_kind, ep, timeout_s=timeout_s, workflow=workflow)
            except (ValueError, LookupError) as e:
                raise OpError(str(e)) from None
            try:
                status, detail = backend.health()
                if status != 200:
                    raise OpError(f"engine not ready: GET {cc.HEALTH_PATH} answered {status or 'nothing'} ({detail})")
                planned = plan_probes(clip, clipset, rng, pairs)
                probes = {p.key: p for p in planned}
                for i in range(warmup):
                    wseed = rng.getrandbits(31)
                    wreq = warmup_request(planned[0].req, i, wseed, planned)
                    res = backend.generate(wreq)
                    warm.append({"seed": wseed, "prompt": wreq.prompt, "ok": res.ok, "e2e_ms": res.e2e_ms,
                                 "error": res.error})
                seq = order(planned, rng)
                run_probes(backend, seq, staging, label)
            finally:
                backend.close()
        live_ok = any(p.ok for p in probes.values())
        files, asked = [], {}
        for p in probes.values():
            if p.ok:
                files.append((p.key, p.path))
                asked[p.key] = ({"width": p.req.width, "height": p.req.height, "num_frames": p.req.num_frames,
                                 "fps": p.req.fps}, p.req.audio, "probe")
        # Its own generator, so the live probes do not depend on whether a bench record was given.
        bench_items, left_out = bench_clips(bench, max_decode, random.Random(f"bench:{probe_seed}")) if bench \
            else ([], 0)
        for it in bench_items:
            if it["state"] == "ok":
                files.append((it["key"], it["path"]))
                asked[it["key"]] = (it["asked"], bench["audio"], "bench clip")
        decoded, decoder, why_not = {}, None, None
        if files:  # the probes are done: a decoder that fails must not throw them away
            try:
                decoded, decoder = probe_clips(files, python, staging / "worker", timeout_s)
            except OpError as e:
                why_not = str(e)
                if python and which("ffprobe"):
                    decoded, decoder = probe_clips(files, None, staging / "worker", timeout_s)
                    decoder += f" (the --python worker failed: {why_not[:200]})"
        if live_ok:
            try:
                dist, why_bytes = distances(probes, distance_plan(probes, pairs), python, staging / "worker",
                                            timeout_s)
            except OpError as e:
                dist, why_bytes = None, f"pixel comparison failed, clips compared by bytes only: {str(e)[:300]}"
            judge = Judge(probes, dist, pairs, neighbour_clashes(seq))

        gates = []

        def live(g: dict) -> dict:
            g["applicable"] = ep is not None
            return g

        not_run = _failed_note(list(probes.values())) or "no --base-url: live probes not run"
        if judge is None:
            for name in ("steps_honoured", "cfg_honoured"):
                gates.append(live(_gate(name, "NOT RUN", not_run)))
        else:
            gates.append(live(gate_steps(judge, clip["steps"])))
            gates.append(live(gate_changed(judge, "cfg_honoured", "cfg", f"guidance {clip['guidance_scale']:g} -> "
                                           f"{probes['cfg'].meta['guidance_scale']:g}")))
        res_items = [{"key": k, "asked": asked[k][0], "audio": asked[k][1], "source": asked[k][2],
                      "decoded": decoded.get(k)} for k, _ in files]
        res_items += [{"key": it["key"], "asked": it["asked"] or {}, "audio": bench["audio"], "source": "bench clip",
                       "decoded": None, "error": it["error"],
                       "problems": [it["error"]] if it["state"] == "changed" else []}
                      for it in bench_items if it["state"] != "ok"]
        if not res_items and bench is not None and why_not is None:
            why_not = "the bench clip record has no timed clip (every clip failed?), so none was decoded"
        g = gate_resolution(res_items, decoder, why_not, left_out)
        g["applicable"] = bool(files) or bench is not None  # a bench record's clips are expected to be decoded
        gates.append(g)
        if judge is None:
            for name in ("seed_honoured", "prompt_sensitivity"):
                gates.append(live(_gate(name, "NOT RUN", not_run)))
        else:
            gates.append(live(gate_seed(judge)))
            gates.append(live(gate_changed(judge, "prompt_sensitivity", "prompt", "a one-word change")))
        g = gate_no_cache(judge, threshold, static_cache(bench) if bench else None)
        g["applicable"] = ep is not None or bench is not None
        gates.append(g)
        if why_bytes and judge is not None:
            for g in gates:
                if g["gate"] in LIVE:
                    g["limitations"].append(why_bytes)

        log = [probe_row(p, decoded.get(p.key)) for p in probes.values()]
        return _record(rcall, store, staging, gates, log=log, warm=warm, label=label, model=model,
                       engine_version=engine_version, endpoint_kind=endpoint_kind if ep else None, clipset=clipset,
                       clip=clip, probe_seed=probe_seed, decoder=decoder, dist=dist, judge=judge, bench=bench,
                       threshold=threshold, python_given=bool(python), seq=[p.key for p in seq])


def _metrics_of(gates: list[dict]) -> list[dict]:
    out = []
    for g in gates:
        for k, v in g["metrics"].items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            unit = ("x" if "ratio" in k or "speedup" in k else "levels" if "mad" in k else "dB" if "psnr" in k
                    else "count")
            out.append(ev.metric(f"{g['gate']}.{k}", v, unit, "median" if "median" in k else
                                 "min" if k.endswith("_min") else "value" if unit != "count" else "count"))
    return out


def _record(rcall, store, staging, gates, *, log, warm, label, model, engine_version, endpoint_kind, clipset, clip,
            probe_seed, decoder, dist, judge, bench, threshold, python_given, seq=()) -> dict:
    verdict = overall(gates)
    artifacts = []
    if log:
        p = staging / "probes.jsonl"
        p.write_text("".join(json.dumps(ev.redact(x), ensure_ascii=False) + "\n" for x in log), encoding="utf-8")
        artifacts.append(ev.artifact(p, staging, "probe requests and replies"))
        files = {f"probes/{x['key']}.{x['format']}": x["out_sha256"] for x in log if x["ok"]}
        idx = staging / "outputs" / label / "index.json"
        idx.parent.mkdir(parents=True, exist_ok=True)
        idx.write_text(json.dumps({"label": label, "files": dict(sorted(files.items()))}, indent=1) + "\n")
        artifacts.append(ev.artifact(idx, staging, "outputs_index"))
    for f in (staging / "worker").glob("*.results.json") if (staging / "worker").is_dir() else []:
        f.unlink()
    lim = [f"{g['gate']}: {x}" for g in gates for x in g["limitations"]]
    if clip is not None:
        lim.append(f"probes use one clip of the set ({clip['id']}, {clip['task']}); run again with --clip-id for "
                   "another task or shape")
    if judge is not None and judge.pairs > 1:
        lim.append(f"probe groups after the first use the clip's prompt plus '{PROBE_SUFFIX.format(n='N').strip()}'"
                   + (" and the clip's picture as another file" if clip and clip.get("task") == "i2v" else "")
                   + ", so no probe follows one with the same prompt or image file")
    if clip is not None and clip.get("task") == "i2v" and log and \
            image_variant(clipset.images[clip["id"]][0], "check") is None:
        lim.append("the clip's image is not a PNG or JPEG file, so every probe sends the same file: an engine that "
                   "keeps the last run's image encoding (ComfyUI) reuses it and fails no_cache")
    bad_warm = [w for w in warm if not w["ok"]]
    if bad_warm:
        lim.append(f"{len(bad_warm)} warm-up clip(s) failed ({bad_warm[0]['error']})")
    missing = [g["gate"] for g in gates if g["verdict"] == "NOT RUN"]
    summary = f"gates clip {verdict}: " + ", ".join(f"{g['gate']} {g['verdict']}" for g in gates)
    workload = {"mode": "single", "batch": 1, "endpoint_kind": endpoint_kind}
    if clip is not None:
        workload.update({k: clip.get(k) for k in bc.SHAPE_KEYS})
        workload["clip_set"] = clipset.identity()
        workload["audio"] = clip.get("audio")
    ident = ev.identity(engine={"name": label, "version": engine_version}, model={"id": model}, workload=workload)
    data = {"verdict": verdict, "gates": {g["gate"]: g for g in gates}, "gates_format": FORMAT, "label": label,
            "not_run_but_expected": [g["gate"] for g in gates if g["verdict"] == "NOT RUN" and g.get("applicable")],
            "probe_seed": probe_seed, "probe_clip": clip["id"] if clip else None, "warmup": warm,
            "probe_order": list(seq), "neighbour_clashes": judge.clashes if judge else None,
            "decoder": decoder, "pixels_compared": dist is not None, "python_given": python_given,
            "repeat_floor_mad": judge.floor if judge else None,
            "bench_evidence": bench["record"]["evidence_id"] if bench else None,
            "thresholds": {"replay_speedup_max": 1 + threshold, "step_ratio_pass": STEP_RATIO_PASS,
                           "step_ratio_fail": STEP_RATIO_FAIL, "sensitivity_x_floor": SENSITIVITY,
                           "min_mad": MIN_MAD, "repro_share": REPRO_SHARE, "fps_tol": FPS_TOL,
                           "audio_tol_s": AUDIO_TOL_S},
            "definitions": {"mad": "mean absolute difference of decoded RGB24 pixels, 0-255, over the frames of both "
                                   "clips", "repeat_floor_mad": "largest MAD between a request and its replay"}}
    rec = ev.build(rcall, kind=KIND, claim_class="observed", summary=summary, identity_=ident,
                   metrics=_metrics_of(gates), artifacts=artifacts, missing=missing, limitations=lim, data=data)
    return store.save(rec, staging)
