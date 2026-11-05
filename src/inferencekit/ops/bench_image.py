"""Client-timed image benchmark with round-level evidence and strict request identity.

Fixtures are JSONL objects {id,prompt,width,height,steps,guidance_scale,settings?}.
settings carries all other generation choices (negative_prompt, scheduler,
flow_shift, uncond_skip_layers, precision, etc.). Unknown top-level keys fail
closed. Timed images use fresh round seeds. Warm-ups cover every shape/settings
combination using separate prompts and seeds. No semantic quality pass is inferred
from a valid PNG or from the server's effective-settings self-report.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import time
from pathlib import Path

from inferencekit import evidence as ev, stats
from inferencekit.canonical import canonical_json, file_sha256
from inferencekit.ops.bench_serve import _record_call
from inferencekit.registry import Arg, OpError, operation, utc_now
from inferencekit.serve import image_client as ic
from inferencekit.serve.client import Endpoint, api_key_from_env

KIND = "image_benchmark"
SEED_POLICIES = ("per_round", "fixed", "list")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}$")
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}$")
_ROW_KEYS = frozenset({"id", "task", "prompt", "width", "height", "steps", "guidance_scale", "settings"})
_RESERVED = frozenset({"id", "task", "model", "prompt", "size", "width", "height", "steps", "seed", "guidance_scale"})
_PIN_KEYS = ("model_revision", "dtype", "quant")
MAX_FIXTURE_BYTES = 4 * 1024 * 1024
MAX_FIXTURES = 1000
MAX_REQUESTS = 10000

ARGS = [
    Arg("base_url", "string", "Image HTTP server root (never stored as an address).", required=True, sensitive=True),
    Arg("prompts", "string", "Image fixtures JSONL: id,prompt,width,height,steps,guidance_scale,settings?.", required=True),
    Arg("model", "string", "Pinned model identifier sent to the engine.", required=True),
    Arg("model_revision", "string", "Checkpoint revision, sent and required in effective settings."),
    Arg("dtype", "string", "Weight/compute precision label, sent and required in effective settings."),
    Arg("quant", "string", "Quantization label, sent and required in effective settings."),
    Arg("label", "string", "Engine configuration label.", default="image_http"),
    Arg("engine_version", "string", "Pinned engine version or commit."),
    Arg("gpu_name", "string", "Exact GPU model (required; image service may be remote).", required=True),
    Arg("gpu_count", "integer", "Number of GPUs serving images.", default=1),
    Arg("rounds", "integer", "Timed passes over the image fixtures.", default=3),
    Arg("first_round", "integer", "First round index; defaults to zero, or previous last + 1 with extend."),
    Arg("warmup", "integer", "Untimed images per distinct request shape/settings combination.", default=1),
    Arg("allow_cache", "string", "Permit only reuse of the fixed empty-negative embedding; every fixture must "
        "pin negative_prompt to an empty string. Positive-prompt and generated-output reuse remain forbidden.",
        default="none", choices=("none", "fixed_negative_embedding")),
    Arg("seed_policy", "string", "per_round=fresh paired round seeds; fixed=quality checks; list=explicit seeds.",
        default="per_round", choices=SEED_POLICIES),
    Arg("seed_base", "integer", "Base seed for the deterministic round-seed policy.", default=0),
    Arg("seeds", "array", "Explicit list of seeds indexed by round, for seed_policy=list.", items="integer"),
    Arg("timeout_s", "number", "Finite whole-response deadline per image, seconds.", default=ic.DEFAULT_TIMEOUT_S),
    Arg("resamples", "integer", "Bootstrap resamples over complete rounds.", default=stats.DEFAULT_RESAMPLES),
    Arg("save_outputs", "string", "all saves PNGs; hash_only saves output hashes.", default="all",
        choices=("all", "hash_only")),
    Arg("api_key_env", "string", "Name of the environment variable holding the API key (never its value)."),
    Arg("run_dir", "string", "Optional run directory receiving inputs, outputs and metrics."),
    Arg("extend", "string", "Previous evidence id for the same exact configuration; append new rounds."),
]


def _finite_json(value, depth=0) -> bool:
    if depth > 32:
        return False
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _finite_json(v, depth + 1) for k, v in value.items())
    if isinstance(value, list):
        return all(_finite_json(v, depth + 1) for v in value)
    return value is None or isinstance(value, (str, int, bool))


def load_image_set(path) -> list[dict]:
    try:
        with Path(path).open("rb") as source:
            raw = source.read(MAX_FIXTURE_BYTES + 1)
    except OSError:
        raise OpError("image fixture file is not readable") from None
    if len(raw) > MAX_FIXTURE_BYTES:
        raise OpError("image fixture file exceeds byte bound")
    rows, seen = [], set()
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        raise OpError("image fixture file must be UTF-8 JSONL") from None
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (ValueError, RecursionError):
            raise OpError(f"image fixture line {line_no} is not JSON") from None
        prefix = f"image fixture line {line_no}: "
        if not isinstance(row, dict) or not _finite_json(row):
            raise OpError(prefix + "must be an object with finite JSON values")
        if set(row) - _ROW_KEYS:
            raise OpError(prefix + "unknown top-level settings; put generation settings in settings")
        if not isinstance(row.get("id"), str) or not _ID.fullmatch(row["id"]) or row["id"] in seen:
            raise OpError(prefix + "id must be unique and safe as a filename")
        seen.add(row["id"])
        if row.get("task", "t2i") != "t2i":
            raise OpError(prefix + "task must be t2i")
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise OpError(prefix + "prompt must be non-empty text")
        for key in ("width", "height", "steps"):
            if type(row.get(key)) is not int or not 1 <= row[key] <= 16384:
                raise OpError(prefix + key + " must be an integer in [1,16384]")
        if row["width"] * row["height"] > ic.MAX_PIXELS:
            raise OpError(prefix + "image exceeds pixel bound")
        scale = row.get("guidance_scale")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not 0 <= scale <= 1000:
            raise OpError(prefix + "guidance_scale must be finite and in [0,1000]")
        settings = row.get("settings", {})
        if not isinstance(settings, dict) or set(settings) & _RESERVED:
            raise OpError(prefix + "settings must be an object without required request field overrides")
        if ev.redact(settings) != settings:
            raise OpError(prefix + "settings must not contain credentials or secret-looking fields")
        rows.append({**row, "task": "t2i", "settings": settings})
        if len(rows) > MAX_FIXTURES:
            raise OpError("too many image fixtures")
    if not rows:
        raise OpError("image fixture file contains no requests")
    return rows


def round_seed(policy, base, seeds, index):
    if policy == "fixed":
        return base
    if policy == "list":
        return seeds[index]
    return int.from_bytes(hashlib.sha256(f"inferencekit-image-seed:{base}:{index}".encode()).digest()[:4], "big") & 0x7FFFFFFF


def request_set(rows, model, pins, policy, base, seeds, allow_cache="none"):
    doc = {"items": rows, "model": model, "settings": pins,
           "seed_policy": {"name": policy, "base": None if policy == "list" else base,
                           "seeds": seeds if policy == "list" else None}}
    if allow_cache != "none":
        doc["cache_policy"] = allow_cache
    return {"sha256": hashlib.sha256(canonical_json(doc).encode()).hexdigest(), "n": len(rows)}


def image_task_set(rows, model, revision, policy, base, seeds, allow_cache="none"):
    """Bind an image task while allowing steps and numerical format to vary.

    All other fixture fields (including prompts, dimensions and settings) remain
    part of the identity. This is only for explicitly labelled tradeoff timing.
    """
    items = []
    for row in rows:
        item = {k: v for k, v in row.items() if k != "steps"}
        item["settings"] = {k: v for k, v in row.get("settings", {}).items()
                            if k not in ("dtype", "quant")}
        items.append(item)
    doc = {"items": items, "model": model, "revision": revision,
           "seed_policy": {"name": policy, "base": None if policy == "list" else base,
                           "seeds": seeds if policy == "list" else None},
           "cache_policy": allow_cache}
    return {"sha256": hashlib.sha256(canonical_json(doc).encode()).hexdigest(), "n": len(items)}


def _request(row, model, seed, pins, prompt=None):
    settings = {**row["settings"]}
    for key, value in pins.items():
        if key in settings and settings[key] != value:
            raise OpError(f"fixture setting {key} conflicts with the pinned operation argument")
        settings[key] = value
    return ic.ImageRequest(row["id"], model, prompt if prompt is not None else row["prompt"],
                           row["width"], row["height"], row["steps"], row["guidance_scale"], seed, settings)


def _put(staging, run, relative, data):
    for root in [staging] + ([run] if run else []):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _previous(store, ref, rcall, ident):
    previous = store.load(ref)
    if previous.get("kind") != KIND or ev.integrity_problem(previous):
        raise OpError("extend needs an intact image benchmark record")
    if previous["identity"] != ident:
        raise OpError("extend needs identical workload, model, engine version and hardware")
    old_args = previous.get("data", {}).get("request", {}).get("args", {})
    for key in ("base_url", "save_outputs", "seed_policy", "seed_base", "seeds", "timeout_s"):
        if old_args.get(key) != ev.redact(rcall.request()["args"]).get(key):
            raise OpError(f"extend configuration differs: {key}")
    artifact = next((a for a in previous.get("artifacts", []) if a.get("role") == "requests"), None)
    if not artifact:
        raise OpError("extend needs the previous request artifact")
    path = store.run_dir(previous["evidence_id"]) / artifact["path"]
    if not path.is_file() or file_sha256(path)[0] != artifact["sha256"]:
        raise OpError("extend request artifact is missing or changed")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return previous, rows


@operation("bench_image", ("bench", "image"), "Client-timed image generation with PNG validation, paired round "
           "seeds, effective-setting/cache gates and image artifacts; semantic quality remains UNPROVEN.", ARGS)
def bench_image(call, base_url, prompts, model, model_revision, dtype, quant, label, engine_version, gpu_name,
                gpu_count, rounds, first_round, warmup, seed_policy, seed_base, seeds, timeout_s, resamples,
                save_outputs, api_key_env, run_dir, extend, allow_cache="none"):
    if not _LABEL.fullmatch(label or ""):
        raise OpError("image label must be a safe 1-64 character filename")
    if not model.strip() or not gpu_name.strip():
        raise OpError("model and gpu_name must be non-empty")
    if not 1 <= rounds <= 10000 or not 0 <= warmup <= 100 or not 1 <= gpu_count <= 1024:
        raise OpError("rounds, warmup or gpu_count is outside its finite bound")
    if not 1 <= resamples <= 100000 or first_round is not None and not 0 <= first_round <= 1000000:
        raise OpError("resamples or first_round is outside its finite bound")
    if not 0 <= seed_base <= 0x7FFFFFFF:
        raise OpError("seed_base must be a 31-bit nonnegative integer")
    if seed_policy == "list":
        if not seeds or any(type(s) is not int or not 0 <= s <= 0x7FFFFFFF for s in seeds):
            raise OpError("list seed policy requires 31-bit nonnegative seeds")
    elif seeds is not None:
        raise OpError("seeds is only valid with seed_policy=list")
    fixtures = load_image_set(prompts)
    if allow_cache != "none" and any(row["settings"].get("negative_prompt") != "" for row in fixtures):
        raise OpError("fixed_negative_embedding reuse requires negative_prompt='' in every fixture's settings")
    if len(fixtures) * (rounds + warmup) > MAX_REQUESTS:
        raise OpError("image run exceeds request count bound")
    pins = {key: value for key, value in zip(_PIN_KEYS, (model_revision, dtype, quant)) if value is not None}
    # Validate conflicts before any network operation, including warm-ups.
    for row in fixtures:
        _request(row, model, seed_base, pins)
    policy = {"name": seed_policy, "base": None if seed_policy == "list" else seed_base}
    rset = request_set(fixtures, model, pins, seed_policy, seed_base, seeds, allow_cache)
    uniform = {key: fixtures[0][key] if all(row[key] == fixtures[0][key] for row in fixtures) else None
               for key in ("width", "height", "steps", "guidance_scale")}
    ident = ev.identity(gpu={"name": gpu_name, "count": gpu_count},
                        engine={"name": label, "version": engine_version},
                        model={"id": model, "revision": model_revision, "dtype": dtype, "quant": quant},
                        workload={"mode": "single", "batch": 1, "task": "t2i", "endpoint_kind": "image_http",
                                  "prompt_set": {"name": "image fixtures", "sha256": hashlib.sha256(
                                      canonical_json(fixtures).encode()).hexdigest()},
                                  "image_task_set": image_task_set(fixtures, model, model_revision, seed_policy,
                                                                   seed_base, seeds, allow_cache),
                                  "request_set": rset, "seed_policy": policy, **uniform})
    try:
        ep = Endpoint.parse(base_url, api_key_from_env(api_key_env))
        backend = ic.ImageHTTP(ep, timeout_s=timeout_s)
    except ValueError:
        raise OpError("invalid image endpoint or finite timeout") from None
    rcall, key_persistent = _record_call(call, ep)
    store, run = ev.Store(), Path(run_dir) if run_dir else None
    previous, old_rows = _previous(store, extend, rcall, ident) if extend else (None, [])
    old_rounds = (previous or {}).get("data", {}).get("rounds", [])
    done = max((r["index"] for r in old_rounds), default=-1)
    start = done + 1 if first_round is None else first_round
    if extend and start != done + 1:
        raise OpError("extend first_round must be the previous last round + 1, without overlaps or gaps")
    last = start + rounds - 1
    if seed_policy == "list" and len(seeds) <= last:
        raise OpError("seed list does not cover the requested round indices")
    round_seeds = [round_seed(seed_policy, seed_base, seeds, r) for r in range(start, last + 1)]
    if seed_policy != "fixed" and len(set(round_seeds + [r["seed"] for r in old_rounds])) != len(round_seeds) + len(old_rounds):
        raise OpError("timed round seeds must be unique (use fixed only for labelled quality checks)")
    with store.staging_dir() as staging:
        warm_rows, timed, rounds_table, output_index = [], list(old_rows), list(old_rounds), {}
        shapes, taken = set(), {row["prompt"] for row in fixtures}
        for row in fixtures:
            shape = canonical_json({k: v for k, v in row.items() if k not in ("id", "prompt")})
            if shape in shapes:
                continue
            shapes.add(shape)
            for number in range(warmup):
                warm_seed = int.from_bytes(hashlib.sha256(f"image-warm:{seed_base}:{len(shapes)}:{number}".encode()).digest()[:4], "big") & 0x7FFFFFFF
                # Avoid both seed and prompt collisions with the timed workload.
                while warm_seed in round_seeds:
                    warm_seed = (warm_seed + 1) & 0x7FFFFFFF
                prompt = row["prompt"] + " (inferencekit warm-up)"
                while prompt in taken:
                    prompt += " warm-up"
                res = backend.generate(_request(row, model, warm_seed, pins, prompt))
                warm_rows.append({"id": row["id"], "ok": res.ok, "error": res.error,
                                  "e2e_ms": res.e2e_ms, "seed": warm_seed})
        for index, seed in zip(range(start, last + 1), round_seeds):
            t_start, started_at, current = time.time(), utc_now(), []
            for row in fixtures:
                req = _request(row, model, seed, pins)
                res = backend.generate(req)
                key = f"r{index}/{row['id']}"
                body = req.body()
                _put(staging, run, f"inputs/{label}/{key}.json", (json.dumps(ev.redact(body), ensure_ascii=False) + "\n").encode())
                sha = hashlib.sha256(res.image).hexdigest() if res.image is not None else None
                pixel_sha = res.pixel_sha256
                if res.image is not None and save_outputs == "all":
                    path = f"outputs/{label}/{key}.png"
                    _put(staging, run, path, res.image)
                    output_index[path] = sha
                entry = {"id": row["id"], "round": index, "seed": seed, "ok": res.ok, "error": res.error,
                         "e2e_ms": res.e2e_ms, "t_send_ns": res.t_send_ns, "t_done_ns": res.t_done_ns,
                         "t_body_ns": res.t_body_ns,
                         "request_sha256": hashlib.sha256(canonical_json(body).encode()).hexdigest(),
                         "out_sha256": sha, "pixel_sha256": pixel_sha,
                         "image_bytes": len(res.image) if res.image is not None else None,
                         "effective": res.effective, "effective_mismatch": res.effective_mismatch,
                         "cache": res.cache, "width": res.width, "height": res.height}
                current.append(entry)
                timed.append(entry)
            failures = sum(not row["ok"] for row in current)
            mean = statistics.mean(row["e2e_ms"] for row in current) if not failures else None
            rounds_table.append({"index": index, "round": index, "started_at": started_at, "seed": seed,
                                 "t_start": t_start, "t_end": time.time(), "n_failed": failures,
                                 "n_images": len(current), "image_ms": mean,
                                 "images_per_hour": 3_600_000 / mean if mean and mean > 0 else None})
        violations = []
        for row in timed:
            if not row["ok"]:
                violations.append(f"{row['id']} r{row['round']}: image request failed")
            if row["effective_mismatch"]:
                violations.append(f"{row['id']} r{row['round']}: effective settings differ or are missing")
            forbidden_cache = [name for name in row["cache"] or []
                               if allow_cache == "none" or name != "fixed_negative_embedding"]
            if forbidden_cache:
                violations.append(f"{row['id']} r{row['round']}: engine reports forbidden cached stages")
        hashes = {}
        for row in timed:
            if row["ok"] and row["pixel_sha256"]:
                hashes.setdefault(row["pixel_sha256"], set()).add(row["request_sha256"])
        if any(len(requests) > 1 for requests in hashes.values()):
            violations.append("identical decoded pixel data for different requests: cached output or ignored settings")
        if any(not row["ok"] for row in warm_rows):
            violations.append("one or more warm-up requests failed")
        request_path = staging / "requests.jsonl"
        request_path.write_text("".join(json.dumps(ev.redact(row), ensure_ascii=False) + "\n" for row in timed))
        artifacts = [ev.artifact(request_path, staging, "requests")]
        if output_index:
            output_path = staging / "outputs" / label / "index.json"
            output_path.write_text(json.dumps({"files": output_index}, indent=2) + "\n")
            artifacts.append(ev.artifact(output_path, staging, "outputs_index"))
        metrics = []
        for name, unit in (("image_ms", "ms"), ("images_per_hour", "images/h")):
            values = [row[name] for row in rounds_table if row[name] is not None]
            ci = stats.bootstrap_ci(values, resamples=resamples)[50]
            metrics.append(ev.metric(name, statistics.median(values) if values else None,
                                     unit, "median", n=len(values), ci95=ci))
        limitations = ["semantic image quality UNPROVEN: PNG structure and server self-reports do not assess quality",
                       "effective settings and cache reports are engine self-reports; independent execution proof is required"]
        if allow_cache != "none":
            limitations.append("Fixed empty-negative embedding reuse is permitted for every compared engine; "
                               "verify its fixed model/settings and actual reused stage independently. "
                               "Positive conditioning, denoising, VAE and output reuse remain forbidden.")
        if not warmup:
            limitations.append("warmup=0: timed requests may include compilation or graph capture")
        if seed_policy == "fixed":
            limitations.append("fixed seed policy repeats requests: output-cache absence is UNPROVEN")
        if not ep.is_local():
            limitations.append("client timer includes network transfer from a remote image engine")
        if not key_persistent:
            limitations.append("endpoint tag persists only for this process; later extend may be unavailable")
        if save_outputs == "hash_only":
            limitations.append("only output hashes saved; independent image quality checks need saved image bytes")
        if not model_revision or not dtype or not engine_version:
            limitations.append("model revision, precision or engine version is not fully pinned")
        if previous:
            violations = list(dict.fromkeys(previous["data"].get("claim_validity", {}).get("violations", []) + violations))
        validity = {"valid": not violations and seed_policy != "fixed", "violations": violations,
                    "cache_policy": allow_cache,
                    "fixed_negative_reused_requests": sum("fixed_negative_embedding" in (row["cache"] or [])
                                                          for row in timed),
                    "warnings": [], "checked": ["decoded PNG dimensions/structure", "effective settings (self-reported)",
                                                  "cache (self-reported)", "unique decoded pixel data"]}
        data = {"source": "bench image", "label": label, "rounds": rounds_table,
                "cache_policy": allow_cache,
                "warmup": {"per_shape": warmup, "requests": warm_rows}, "claim_validity": validity,
                "quality": {"verdict": "UNPROVEN", "reason": "no semantic image quality evaluation supplied"},
                "timer": "client perf_counter_ns: before connection through HTTP response, PNG decoding and validation",
                "metric_claim_classes": {"image_ms": "observed (client timer)", "images_per_hour": "derived"},
                "definitions": {"image_ms": "mean per-image latency within each complete round; median across rounds",
                                "images_per_hour": "3,600,000 / round mean image_ms (one request at a time)"},
                "extends": previous["evidence_id"] if previous else None,
                "outputs_from": ([previous["evidence_id"]] + previous["data"].get("outputs_from", [])) if previous else [],
                "save_outputs": save_outputs}
        if run:
            _put(staging, run, f"metrics/{label}.image.json", (json.dumps(data, indent=2) + "\n").encode())
        record = ev.build(rcall, kind=KIND, claim_class="observed",
                          summary=f"{label}: {len(rounds_table)} image rounds x {len(fixtures)} fixtures, client-timed; semantic quality UNPROVEN",
                          identity_=ident, metrics=metrics, artifacts=artifacts,
                          missing=["semantic image quality"] + (["complete successful image rounds"] if not any(
                              row["image_ms"] is not None for row in rounds_table) else []),
                          limitations=limitations, data=data)
        return store.save(record, staging)
