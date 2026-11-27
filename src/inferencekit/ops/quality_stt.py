"""Pre-registered paired transcript quality against reference text."""
from __future__ import annotations

import json
import random
import time
import unicodedata
from functools import lru_cache
from pathlib import Path

from inferencekit import evidence as ev, stats
from inferencekit.canonical import canonical_json
from inferencekit.ops.bench_audio import round_seed
from inferencekit.ops.bench_stt import FIXTURE_ARG, KIND, load_fixtures
from inferencekit.quality import audio as qa
from inferencekit.registry import Arg, OpError, operation

QUALITY_KIND = "quality_stt"
NORMALIZATIONS = (*qa.NORMALIZATIONS, "whisper_english_v1", "indic_cer_v1")
MARGINS = {"wer_noninferiority": "wer_increase_max", "cer_noninferiority": "cer_increase_max"}
RUN_ARG = Arg("run_dir", "string", "Also write evidence and review artifacts into this run directory.")


@lru_cache(maxsize=1)
def _english_normalizer():
    from inferencekit.quality._whisper_normalizers.english import EnglishTextNormalizer

    return EnglishTextNormalizer()


def normalize(text: str, method: str) -> str:
    if method == "whisper_english_v1":
        return " ".join(_english_normalizer()(text).split())
    if method == "indic_cer_v1":
        text = unicodedata.normalize("NFKD", text)
        # Nukta combining marks in major Indic blocks, plus the joiners.
        skip = "\u093c\u09bc\u0a3c\u0abc\u0b3c\u0c3c\u0cbc\u200c\u200d"
        text = unicodedata.normalize("NFKC", text.translate(str.maketrans("", "", skip)))
        method = "nfkc_casefold_punctuation_v1"
    return " ".join(qa.words(text, method))


def policy_document(doc):
    if not isinstance(doc, dict):
        raise OpError("STT policy must be a JSON object")
    acceptance = doc.get("acceptance")
    if not isinstance(acceptance, str) or acceptance not in ("exact", *MARGINS):
        raise OpError("STT policy acceptance must be exact, wer_noninferiority or cer_noninferiority")
    if set(doc) - {"acceptance", "normalization", "min_fixtures", "min_rounds", "resamples", "seed",
                   "allow_numerics_change", "completion_failure_increase_max", *MARGINS.values()}:
        raise OpError("unknown STT policy fields")
    doc = {"normalization": "nfkc_casefold_punctuation_v1", "min_fixtures": 20, "min_rounds": 1,
           "resamples": 2000, "seed": 0, "allow_numerics_change": False,
           "completion_failure_increase_max": 0.0, **doc}
    if doc["normalization"] not in NORMALIZATIONS or type(doc["allow_numerics_change"]) is not bool:
        raise OpError("invalid STT normalization or numerical-change permission")
    for key, lo, hi in (("min_fixtures", 2, 100000), ("min_rounds", 1, 10000),
                        ("resamples", 100, 10000), ("seed", -(1 << 63), (1 << 63) - 1)):
        if type(doc[key]) is not int or not lo <= doc[key] <= hi:
            raise OpError(f"invalid STT policy {key}")
    for name in ("completion_failure_increase_max", *MARGINS.values()):
        if name == MARGINS.get(acceptance) and name not in doc:
            raise OpError(f"STT policy requires {name}")
        if name in doc and (not qa.finite(doc[name]) or doc[name] < 0 or
                            name == "completion_failure_increase_max" and doc[name] > 1):
            raise OpError(f"invalid STT policy {name}")
    return doc


def _save(call, store, data, metrics, filename, payload, run_dir):
    with store.staging_dir() as stage:
        path = stage / filename
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        record = ev.build(call, kind=QUALITY_KIND, claim_class="derived",
                          summary=f"STT quality {data['verdict']}: {data['source']}",
                          metrics=metrics, artifacts=[ev.artifact(path, stage, "stt_quality")],
                          data=data)
        saved = store.save(record, stage)
        if run_dir:
            run = Path(run_dir)
            target = run / "evidence" / f"{saved['evidence_id']}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
            copy = run / "artifacts" / saved["evidence_id"] / filename
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes((store.run_dir(saved['evidence_id']) / filename).read_bytes())
        return saved


@operation("quality_stt_policy", ("quality", "stt-policy"), "Register transcript accuracy before capture.", (
    Arg("policy", "string", "JSON file with STT acceptance, normalization, margins and coverage.", required=True),
    FIXTURE_ARG, RUN_ARG,
))
def quality_stt_policy(call, policy, fixtures, run_dir):
    raw_policy, _ = qa.read_json(policy)
    doc = policy_document(raw_policy)
    rows, fixture_sha = load_fixtures(fixtures)
    for fixture in rows:
        if not normalize(fixture.reference, doc["normalization"]):
            raise OpError(f"STT fixture {fixture.id}: empty normalized reference")
    payload = {"policy": doc, "fixture_sha256": fixture_sha, "registered_at": time.time()}
    data = {"source": "quality stt-policy", "verdict": "NOT RUN", **payload}
    return _save(call, ev.Store(), data, [], "stt_policy.json", payload, run_dir)


def evidence_root(record, root):
    # A scheduler run contains independent arm bundles in artifacts/<evidence-id>.
    bundled = root / "artifacts" / record["evidence_id"]
    return bundled if bundled.is_dir() else root


def capture(store, ref, fixtures, fixture_sha):
    record, root = qa.load_record(store, ref, KIND)
    data = record["data"]
    ids = {f.id: f.identity() for f in fixtures}
    if data.get("source") not in ("bench stt", "import stt") or data.get("fixture_set", {}).get("sha256") != fixture_sha:
        raise OpError("STT quality requires fixture-matched benchmark or import evidence")
    root = evidence_root(record, root)
    raw = qa.artifact_bytes(root, qa.one_artifact(record, "transcripts"))
    try:
        rounds = {r["index"]: r for r in data["rounds"]}
        base = record["identity"]["workload"]["seed_policy"]["base"]
        if len(rounds) != len(data["rounds"]) or not rounds or any(
                type(i) is not int or not 0 <= i <= 100000 or row["seed"] != round_seed(base, i)
                for i, row in rounds.items()):
            raise ValueError
        output = {}
        for line in raw.splitlines():
            row = json.loads(line)
            fid, index = row["id"], row["round"]
            if ((fid, index) in output or index not in rounds or any(row[k] != value for k, value in ids[fid].items())
                    or row["seed"] != rounds[index]["seed"] or type(row["completed"]) is not bool):
                raise ValueError
            text = row["transcript"]
            if text is not None and (not isinstance(text, str) or qa.sha(text.encode()) != row["transcript_sha256"]):
                if text != "" or row["transcript_sha256"] is not None:
                    raise ValueError
            output[fid, index] = row
        if len(output) != len(ids) * len(rounds) or len(output) != data["total_requests"]:
            raise ValueError
    except (KeyError, TypeError, ValueError, UnicodeError):
        raise OpError("STT transcript/request identities are incomplete or inconsistent") from None
    return record, output, rounds


def _accuracy(reference, text):
    word = qa.word_error(reference, text, "whitespace_v1")
    char = qa.character_error(reference, text, "whitespace_v1")
    return word["errors"], char["errors"], word["words"], char["characters"]


def _check(fixture_counts, limit, policy, coverage):
    totals = [tuple(sum(row[i] for row in group) for i in range(3)) for group in fixture_counts]
    if not totals:
        return {"baseline": None, "candidate": None, "difference": None, "ci95": [None, None],
                "margin": limit, "verdict": "UNPROVEN"}
    rng = random.Random(policy["seed"])
    a, b, n = (sum(row[i] for row in totals) for i in range(3))
    replicas = []
    if len(totals) >= 2:
        for _ in range(policy["resamples"]):
            draw = [totals[int(rng.random() * len(totals))] for _ in totals]
            replicas.append(sum(b - a for a, b, _ in draw) / sum(n for _, _, n in draw))
    lo, hi = stats.percentile_ci(replicas)
    verdict = "UNPROVEN"
    if limit is not None and coverage and lo is not None:
        if hi <= limit:
            verdict = "PASS"
        elif lo > limit:
            verdict = "FAIL"
    return {"baseline": a / n, "candidate": b / n, "difference": (b - a) / n,
            "ci95": [lo, hi], "margin": limit, "verdict": verdict}


@operation("quality_stt", ("quality", "stt"), "Compare paired transcripts against pinned references.", (
    Arg("baseline", "string", "Baseline STT benchmark evidence id or JSON.", required=True),
    Arg("candidate", "string", "Candidate STT benchmark evidence id or JSON.", required=True),
    FIXTURE_ARG,
    Arg("protocol", "string", "Pre-capture quality stt-policy evidence id or JSON.", required=True),
    RUN_ARG,
))
def quality_stt(call, baseline, candidate, fixtures, protocol, run_dir):
    store = ev.Store()
    registration, root = qa.load_record(store, protocol, QUALITY_KIND)
    registered = registration["data"]
    if registered.get("source") != "quality stt-policy":
        raise OpError("STT protocol must be a quality stt-policy registration")
    raw = qa.artifact_bytes(evidence_root(registration, root), qa.one_artifact(registration, "stt_quality"))
    payload = json.loads(raw)
    if any(payload.get(k) != registered.get(k) for k in ("policy", "fixture_sha256", "registered_at")):
        raise OpError("STT policy artifact and evidence disagree")
    policy = policy_document(registered["policy"])
    fixture_rows, fixture_sha = load_fixtures(fixtures)
    if registered.get("fixture_sha256") != fixture_sha:
        raise OpError("STT policy was registered for a different fixture set")
    a, ar, ai = capture(store, baseline, fixture_rows, fixture_sha)
    b, br, bi = capture(store, candidate, fixture_rows, fixture_sha)
    model_a, model_b = a["identity"]["model"], b["identity"]["model"]
    if (a["data"]["stt_config"] != b["data"]["stt_config"]
            or any(model_a[k] != model_b[k] for k in ("id", "revision"))):
        raise OpError("STT quality arms must match checkpoint, streaming mode and decoding settings")
    if not policy["allow_numerics_change"] and any(model_a[k] != model_b[k] for k in ("dtype", "quant")):
        raise OpError("STT numerical changes require pre-registered allow_numerics_change")
    if any(ai[i]["seed"] != bi[i]["seed"] for i in ai.keys() & bi.keys()):
        raise OpError("paired STT round seeds differ")

    early = all(qa.finite(r.get("t_start")) and r["t_start"] > registered["registered_at"]
                for r in (*ai.values(), *bi.values()))
    shared = sorted(ai.keys() & bi.keys())
    coverage = (early and ai.keys() == bi.keys() and len(shared) >= policy["min_rounds"] and
                len(fixture_rows) >= policy["min_fixtures"] and all(
                    (e["data"].get("round_plan") or {}).get("expected_indices", shared) == shared for e in (a, b)))
    counts = {"wer": [], "cer": [], "completion_failure": []}
    exact, scored, observations = 0, 0, []
    for fixture in fixture_rows:
        ref = normalize(fixture.reference, policy["normalization"])
        groups = {name: [] for name in counts}
        for index in shared:
            pair = [rows[fixture.id, index] for rows in (ar, br)]
            values = [r["transcript"] for r in pair]
            completed = [r["completed"] for r in pair]
            groups["completion_failure"].append((int(not completed[0]), int(not completed[1]), 1))
            if any(ok and text is None for ok, text in zip(completed, values)):
                continue
            hypotheses = [normalize(text or "", policy["normalization"]) if ok else ""
                          for ok, text in zip(completed, values)]
            # An empty normalized hypothesis has no scored text even if the HTTP reply completed.
            groups["completion_failure"][-1] = tuple(int(not text) for text in hypotheses) + (1,)
            matched = all(hypotheses) and hypotheses[0] == hypotheses[1]
            exact += int(bool(matched))
            scored += 1
            wa, ca, nw, nc = _accuracy(ref, hypotheses[0])
            wb, cb, _, _ = _accuracy(ref, hypotheses[1])
            groups["wer"].append((wa, wb, nw))
            groups["cer"].append((ca, cb, nc))
            observations.append({"id": fixture.id, "round": index, "ground_truth": fixture.reference,
                                 "baseline": values[0], "candidate": values[1]})
        for name in groups:
            if groups[name]:
                counts[name].append(groups[name])

    full = coverage and scored == len(fixture_rows) * len(shared)
    primary = policy["acceptance"].partition("_")[0]
    checks = {}
    for name in ("wer", "cer", "completion_failure"):
        limit = policy.get(f"{name}_increase_max")
        checks[name] = _check(counts[name], limit, policy, full)
    if primary == "exact":
        matches = "FAIL" if early and scored > exact else "PASS" if full else "UNPROVEN"
        checks["exact"] = {"matched_pairs": exact, "scored_pairs": scored, "verdict": matches}
    else:
        matches = checks[primary]["verdict"]
    completion = checks["completion_failure"]["verdict"]
    if early and scored and any(checks["completion_failure"][arm] == 1 for arm in ("baseline", "candidate")):
        completion = checks["completion_failure"]["verdict"] = "FAIL"
    verdict = stats.combine_claims([matches, completion])
    data = {"source": "quality stt", "verdict": verdict, "protocol": registration["evidence_id"],
            "baseline": a["evidence_id"], "candidate": b["evidence_id"],
            "arms": {"baseline": a["data"]["label"], "candidate": b["data"]["label"]},
            "policy": policy, "checks": checks, "policy_before_capture": early,
            "coverage_complete": full, "paired_rounds": shared}
    metrics = [ev.metric(name, checks[name][arm], "fraction", "pooled", subject=arm)
               for name in ("wer", "cer", "completion_failure") for arm in ("baseline", "candidate")]
    return _save(call, store, data, metrics, "stt_quality.json",
                 {"checks": checks, "transcripts": ev.redact(observations)}, run_dir)
