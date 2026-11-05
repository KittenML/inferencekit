"""Round runner: every engine of a run manifest, R rounds, in a balanced order, one server at a time.

Why rounds, and why this order
------------------------------
A GPU's speed drifts during a session (temperature, clocks, other tenants on a rented
host). If engine A ran all its requests before engine B, that drift would be mixed into
"A vs B". So each round runs every engine once, and the order inside each round follows a
Williams design: a Latin square in which every engine runs once in every position, and
every ordered pair "X right after Y" occurs equally often. Position effects (the first
server after an idle GPU) and carry-over effects (a server starting on a GPU the previous
one just heated) then hit every engine equally. With n engines one balanced block takes
n rounds when n is even and 2n when n is odd; ``rounds`` that are not a multiple of that
still run, with ``balanced: false`` recorded. The default number of rounds is the smallest
multiple of that period that is at least 3 (2 engines: 4, 3 engines: 6, 4 engines: 4).

What one (round, engine) cell does: see ``servers.py`` (launch under a clean environment,
readiness, warm-up, /metrics before and after, stop of the whole process group, GPU
memory back to baseline). The measurement itself is ``inferencekit bench serve``, run as a
subprocess against the live server: the client-side timer is the only timer, and the
coupling to it is only its command line.

Summary: per engine x bucket x metric, the median over rounds with a bootstrap 95% CI
over rounds (rounds are the unit of repetition), and the ratio to the best engine with a
paired bootstrap CI (round k of an engine paired with round k of the best engine; both
ran within the same round, minutes apart). An engine with failed requests in a bucket is
never named best there and gets no ratio: its latencies cover only the requests it
answered, and a fast error is not a fast answer.

Load mode gives every engine the same arrival schedule: bench serve turns a concurrency
level N into a Poisson rate N / e2e_ref, so the runner passes one shared ``--ref-e2e-ms``
(given, or measured by the first load cell's warm-up) to every cell, unless ``--serve-args``
already fixes the schedule (``--rates``, ``--ref-e2e-ms``, ``--closed-loop``).

Sampling: ``extra_body`` (``--extra-body``, else the manifest's ``workload.extra_body``) goes to
every bench serve call as its ``--extra-body``, so every engine is timed with the same sampling
settings (e.g. the model's default temperature / top_k / top_p and a seed) instead of greedy.

Adaptive stop (``ci_target``, off by default): ``rounds`` becomes a maximum. After each whole
Williams period, once at least ``min_rounds`` rounds ran, the runner stops if every engine's
decode_tok_s ratio to the best engine (bucket ``all``) has a 95% CI whose half-width is at most
``ci_target`` times the ratio. Stopping only at whole periods keeps the order balanced. Stopping
when the CI looks narrow enough is optional stopping: the reported CIs come out slightly too
narrow, so the rule (target, minimum, maximum) is recorded in the plan and should be fixed before
the run. Without ``ci_target`` every round runs and nothing about the run changes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import statistics
import sys
import time
from pathlib import Path

from inferencekit import evidence as ev
from inferencekit import proc, stats
from inferencekit.registry import OpError
from inferencekit.serve import servers

SERVE_CMD_ENV = "INFERENCEKIT_BENCH_SERVE_CMD"
STATUSES_OK = ("ok",)
# The adaptive stop reads this metric in this bucket (single mode: client-timed decode speed over all prompts).
ADAPTIVE_METRIC, ADAPTIVE_BUCKET = "decode_tok_s", "all"
ADAPTIVE_MAX_ROUNDS = 12  # default maximum with --ci-target, rounded up to whole Williams periods


# --------------------------------------------------------------------------- order design

def williams_square(n: int) -> list[list[int]]:
    """Rows of a Williams design for ``n`` treatments (n rows if n is even, 2n if odd).

    First row 0, 1, n-1, 2, n-2, 3, ...; row i adds i (mod n). For odd n the mirrored
    rows are appended, because a single square cannot balance carry-over then.
    """
    if n <= 0:
        return []
    if n == 1:
        return [[0]]
    first, lo, hi = [0], 1, n - 1
    take_lo = True
    while len(first) < n:
        if take_lo:
            first.append(lo)
            lo += 1
        else:
            first.append(hi)
            hi -= 1
        take_lo = not take_lo
    rows = [[(x + i) % n for x in first] for i in range(n)]
    if n % 2:
        rows += [list(reversed(r)) for r in rows]
    return rows


def round_orders(labels: list[str], rounds: int) -> dict:
    """The engine order for each round, the design's period, and whether R balances it."""
    square = williams_square(len(labels))
    period = len(square)
    orders = [[labels[i] for i in square[r % period]] for r in range(rounds)] if period else []
    return {"design": "williams", "orders": orders, "period": period,
            "balanced": bool(period) and rounds % period == 0}


def default_rounds(n_engines: int, minimum: int = 3) -> int:
    """The smallest multiple of the Williams period for ``n_engines`` that is at least ``minimum``."""
    period = len(williams_square(n_engines)) or 1
    return period * max(1, -(-minimum // period))


def carryover_counts(orders: list[list[str]]) -> dict[tuple[str, str], int]:
    """How often engine b ran right after engine a, inside rounds."""
    counts: dict[tuple[str, str], int] = {}
    for row in orders:
        for a, b in zip(row, row[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


# --------------------------------------------------------------------------- manifest and prompts

def load_manifest(path: str | os.PathLike) -> dict:
    p = Path(path)
    try:
        manifest = json.loads(p.read_text())
    except OSError as e:
        raise OpError(f"cannot read manifest {path}: {e}") from None
    except ValueError as e:
        raise OpError(f"manifest {path} is not valid JSON: {e}") from None
    if not isinstance(manifest, dict) or not isinstance(manifest.get("engines"), list) or not manifest["engines"]:
        raise OpError(f"manifest {path} has no 'engines' list")
    return manifest


def engine_specs(manifest: dict, only: str | list[str] | None) -> list[dict]:
    """The manifest's engines (optionally only the given labels), checked for what a launch needs."""
    specs, problems, labels = [], [], set()
    for i, e in enumerate(manifest["engines"]):
        if not isinstance(e, dict):
            problems.append(f"engines[{i}] is not an object")
            continue
        label = str(e.get("label") or e.get("name") or "")
        launch = e.get("launch")
        if not label:
            problems.append(f"engines[{i}] has no label")
        elif label in labels:
            problems.append(f"engine label '{label}' appears twice")
        elif not all(c.isalnum() or c in "-_." for c in label):
            problems.append(f"engine label '{label}' may only use letters, digits, '-', '_' and '.'")
        labels.add(label)
        if not isinstance(launch, list) or not launch or not all(isinstance(a, str) for a in launch):
            problems.append(f"engine '{label}': 'launch' must be a non-empty list of strings")
        if not isinstance(e.get("port"), int):
            problems.append(f"engine '{label}': 'port' must be an integer")
        if e.get("env") is not None and not isinstance(e.get("env"), dict):
            problems.append(f"engine '{label}': 'env' must be an object")
        specs.append({"label": label, "name": e.get("name"), "kind": servers.engine_kind(e.get("name"), launch),
                      "version": e.get("version"), "launch": launch, "env": e.get("env") or {},
                      "port": e.get("port"), "host": e.get("host") or "127.0.0.1",
                      "ready_check": "/" + str(e.get("ready_check") or "/v1/models").lstrip("/"),
                      "role": e.get("role"),
                      "cwd": e.get("cwd")})
    if problems:
        raise OpError("manifest engines: " + "; ".join(problems))
    if only:
        wanted = [w.strip() for w in (only.split(",") if isinstance(only, str) else only) if w.strip()]
        unknown = [w for w in wanted if w not in labels]
        if unknown:
            raise OpError(f"--engines: unknown label(s) {unknown}; manifest has {sorted(labels)}")
        specs = [s for s in specs if s["label"] in wanted]
    return specs


def read_prompts(path: str | os.PathLike) -> dict:
    """sha256, count and bucket counts of a prompt set (JSONL or .jsonl.gz; the sha256 is of the JSONL
    bytes), plus its first rows for warm-up."""
    from inferencekit import prompts as sets

    p = sets.existing(path)
    try:
        raw = sets.read_bytes(p)
    except (OSError, ValueError) as e:  # missing, or a truncated / corrupt .jsonl.gz
        raise OpError(f"cannot read prompts {path}: {e}") from None
    rows, buckets = [], {}
    for n, line in enumerate(raw.decode(errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            raise OpError(f"prompts {path}: line {n} is not JSON") from None
        rows.append(row)
        b = str(row.get("bucket") or "all") if isinstance(row, dict) else "all"
        buckets[b] = buckets.get(b, 0) + 1
    if not rows:
        raise OpError(f"prompts {path} is empty")
    name = sets.set_stem(p)
    return {"name": name, "sha256": hashlib.sha256(raw).hexdigest(), "n": len(rows), "buckets": buckets,
            "head": [r for r in rows[:8] if isinstance(r, dict)]}


# --------------------------------------------------------------------------- bench serve

def _drop_flag(words: list[str], flag: str) -> list[str]:
    """``words`` without ``flag VALUE`` / ``flag=VALUE``."""
    out, skip = [], False
    for w in words:
        if skip:
            skip = False
        elif w == flag:
            skip = True
        elif not w.startswith(flag + "="):
            out.append(w)
    return out


def _flag_value(words: list[str], flag: str) -> str | None:
    """The last value given to ``flag`` in ``words`` (``flag VALUE`` or ``flag=VALUE``), else None."""
    found = None
    for i, w in enumerate(words):
        name, eq, val = w.partition("=")
        if name == flag:
            found = val if eq else (words[i + 1] if i + 1 < len(words) else "")
    return found


def serve_argv(template: str | None, ctx: dict, extra: str | None) -> list[str]:
    """The ``bench serve`` command for one cell (the flags ``bench serve`` accepts).

    Default: ``<python> -m inferencekit bench serve --base-url ... --model ... --prompts ... --mode ...
    --engine-kind ... --label ... --run-dir ...``, plus in single mode ``--rounds 1`` and, from the
    second round of an engine on, ``--extend <its previous record>`` (so the engine's last record
    holds all its rounds, interleaved with the other engines), plus ``--engine-version`` when the
    manifest gives one, then ``extra``.
    A template (``--serve-cmd`` or $INFERENCEKIT_BENCH_SERVE_CMD) containing ``{placeholders}`` is
    used as written, with {base_url} {model} {prompts} {mode} {engine_kind} {label} {round} {run_dir}
    filled in; a template without placeholders replaces only the program part.
    In load mode with a shared reference latency (``ctx["ref_e2e_ms"]``), ``--ref-e2e-ms`` is
    appended in both cases, so every engine gets the same arrival schedule.
    """
    template = template or os.environ.get(SERVE_CMD_ENV) or ""
    words = shlex.split(template) if template else [sys.executable, "-m", "inferencekit", "bench", "serve"]
    if any("{" in w for w in words):
        try:
            argv = [w.format(**ctx) for w in words]
        except (KeyError, IndexError, ValueError) as e:
            raise OpError(f"--serve-cmd template: unknown placeholder {e}") from None
    else:
        argv = words + ["--base-url", ctx["base_url"], "--model", ctx["model"], "--prompts", ctx["prompts"],
                        "--mode", ctx["mode"], "--engine-kind", ctx["engine_kind"], "--label", ctx["label"],
                        "--run-dir", ctx["run_dir"]]
        if ctx["mode"] == "single":
            argv += ["--rounds", "1"]
            if ctx.get("extend"):
                argv += ["--extend", ctx["extend"]]
        if ctx.get("engine_version"):
            argv += ["--engine-version", str(ctx["engine_version"])]
    if ctx.get("mode") == "load" and ctx.get("ref_e2e_ms") not in (None, ""):
        argv += ["--ref-e2e-ms", str(ctx["ref_e2e_ms"])]  # repr of the float: the exact same schedule
    extra_words = shlex.split(extra) if extra else []
    body = ctx.get("extra_body")
    if body:
        # The one resolved value (--extra-body, the manifest's workload.extra_body, or one in --serve-args; checked
        # to agree by run_rounds) for every engine and round; a copy in --serve-args is dropped.
        extra_words = _drop_flag(extra_words, "--extra-body")
        argv += ["--extra-body", json.dumps(body, sort_keys=True, separators=(",", ":"))]
    ctk = ctx.get("chat_template_kwargs")
    if ctk:
        # The same template settings for every engine and round (from --chat-template-kwargs, the manifest's
        # workload.chat_template_kwargs, one in --serve-args, plus the intake pin; run_rounds has checked they
        # agree). A copy in --serve-args is dropped so the one resolved value is the only one sent; a prompt
        # record's own field still wins per key.
        extra_words = _drop_flag(extra_words, "--chat-template-kwargs")
        argv += ["--chat-template-kwargs", json.dumps(ctk, sort_keys=True, separators=(",", ":"))]
    return argv + extra_words


def template_kwargs(flag: str | dict | None, manifest: dict) -> tuple[dict, str]:
    """chat_template_kwargs for every cell: the flag (JSON object), else the manifest's
    ``workload.chat_template_kwargs``, else none. Returns (value, where it came from)."""
    if flag not in (None, ""):
        try:
            value = json.loads(flag) if isinstance(flag, str) else flag
        except ValueError as e:
            raise OpError(f"--chat-template-kwargs is not valid JSON: {e}") from None
        if not isinstance(value, dict):
            raise OpError("--chat-template-kwargs must be a JSON object")
        return value, "--chat-template-kwargs"
    value = (manifest.get("workload") or {}).get("chat_template_kwargs")
    if isinstance(value, dict) and value:
        return value, "manifest workload.chat_template_kwargs"
    return {}, "none"


def _serve_args_template_kwargs(serve_args: str | None, ctk: dict, source: str) -> tuple[dict, str]:
    """A --chat-template-kwargs inside --serve-args is the value every cell sends: use it for the warm-up and
    plan.json too, and refuse it when it contradicts the flag or the manifest."""
    found = _flag_value(shlex.split(serve_args) if serve_args else [], "--chat-template-kwargs")
    if found is None:
        return ctk, source
    value, _ = template_kwargs(found, {})
    if ctk and value != ctk:
        raise OpError(f"--serve-args sets --chat-template-kwargs {json.dumps(value, sort_keys=True)} but {source} "
                      f"says {json.dumps(ctk, sort_keys=True)}: give one value")
    return value, (source if ctk else "--chat-template-kwargs in --serve-args")


def _pinned_template_kwargs(ctk: dict, source: str, dirs: list[Path]) -> tuple[dict, str]:
    """Apply the intake's pinned chat_template_kwargs (intake.normalized.json in the run dir or next to the
    manifest): pinned keys are added, a contradicting value is refused."""
    from inferencekit.ops.intake import apply_pin, intake_pin

    for d in dirs:
        if intake_pin(d):
            merged = apply_pin(ctk, d, "bench rounds")
            if merged != ctk:
                source = f"{source} + intake pin" if ctk else "intake pin (intake.normalized.json)"
            return merged, source
    return ctk, source


def _extra_body_value(text, what: str) -> tuple[dict, list[str]]:
    """The parsed extra body and its warnings; an OpError lists every error (manifest.extra_body_problems)."""
    from inferencekit.ops.manifest import extra_body_problems

    try:
        value = json.loads(text) if isinstance(text, str) else text
    except ValueError as e:
        raise OpError(f"{what} is not valid JSON: {e}") from None
    if not isinstance(value, dict):
        raise OpError(f"{what} must be a JSON object")
    errors, warnings = extra_body_problems(value, what)
    if errors:
        raise OpError("; ".join(errors))
    return value, warnings


def extra_body_for(flag: str | dict | None, manifest: dict,
                   serve_args: str | None = None) -> tuple[dict, str, list[str]]:
    """The JSON object every bench serve call gets as ``--extra-body``, where it came from, and the warnings
    about it (recorded as limitations): the flag, else the manifest's ``workload.extra_body``; one inside
    ``--serve-args`` is used when neither is set and refused when it contradicts them."""
    if flag not in (None, ""):
        (value, warnings), source = _extra_body_value(flag, "--extra-body"), "--extra-body"
    else:
        found = (manifest.get("workload") or {}).get("extra_body") if isinstance(manifest.get("workload"),
                                                                                dict) else None
        source = "none" if found in (None, {}) else "manifest workload.extra_body"
        value, warnings = ({}, []) if found in (None, {}) else _extra_body_value(found, source)
    inner = _flag_value(shlex.split(serve_args) if serve_args else [], "--extra-body")
    if inner is None:
        return value, source, warnings
    given, given_warnings = _extra_body_value(inner, "--extra-body in --serve-args")
    if value and given != value:
        raise OpError(f"--serve-args sets --extra-body {json.dumps(given, sort_keys=True)} but {source} says "
                      f"{json.dumps(value, sort_keys=True)}: give one value")
    return (given, source, warnings) if value else (given, "--extra-body in --serve-args", given_warnings)


SCHEDULE_FLAGS = ("--rates", "--ref-e2e-ms", "--closed-loop")


def schedule_fixed_by(serve_args: str | None) -> str | None:
    """The ``--serve-args`` flag that already fixes the load schedule for every engine, if any."""
    words = shlex.split(serve_args) if serve_args else []
    return next((w.split("=", 1)[0] for w in words if w.split("=", 1)[0] in SCHEDULE_FLAGS), None)


def reference_e2e_ms(record: dict | None, store_root: Path) -> float | None:
    """The unloaded e2e latency a load-mode ``bench serve`` call used to turn concurrency into rates.

    It is ``data.e2e_ref_ms`` of the measured ``serve_benchmark`` record, which the returned
    ``capacity`` record cites as ``data.measurement``.
    """
    if not isinstance(record, dict):
        return None
    data = record.get("data") if isinstance(record.get("data"), dict) else {}
    x = data.get("e2e_ref_ms")
    if x is None and data.get("measurement"):
        try:
            measured = json.loads((store_root / str(data["measurement"]) / "evidence.json").read_text())
            x = (measured.get("data") or {}).get("e2e_ref_ms")
        except (OSError, ValueError, AttributeError):
            x = None
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0 else None


def parse_serve_output(stdout: str) -> dict | None:
    """The JSON object ``bench serve`` printed (an evidence record), or None."""
    text = (stdout or "").strip()
    if not text:
        return None
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except ValueError:
        pass
    i = text.find("{")  # tolerate a log line before the JSON
    while i >= 0:
        try:
            value = json.loads(text[i:])
            return value if isinstance(value, dict) else None
        except ValueError:
            i = text.find("\n{", i + 1)
            i = i + 1 if i >= 0 else -1
    return None


def _num(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict):
        for k in ("value", "median", "p50"):
            if isinstance(x.get(k), (int, float)) and not isinstance(x.get(k), bool):
                return float(x[k])
    return None


def _bucket_of(metric: dict) -> str:
    if metric.get("bucket") is not None:
        return str(metric["bucket"])
    subject = str(metric.get("subject") or "")
    if not subject:
        return "all"
    subject = subject.rsplit("/", 1)[-1]  # "<engine>/short" -> "short"
    for sep in ("=", ":"):
        head, found, tail = subject.partition(sep)
        if found and head.strip().lower() == "bucket":
            return tail.strip()
    return subject  # "short", or a load level such as "concurrency=8"


def bucket_metrics(record: dict | None) -> dict:
    """``{bucket: {metric: {"value", "unit", "stat"}}}`` from a ``bench serve`` record.

    Accepted shapes (``bench serve`` owns the format; this reads it tolerantly):
      * ``metrics: [{"name", "value", "unit", "stat", "bucket"? , "subject"?}]`` (a stat other than
        the median is appended to the name: ``ttft_ms_p95``); the bucket is
        ``bucket``, else ``subject`` (``"short"``, ``"bucket=short"``, ``"bucket:short"``,
        ``"<engine>/short"``), else ``"all"``;
      * ``data.per_bucket`` or ``data.buckets``: ``{bucket: {metric: number | {"value"|"median": x}}}``.
    """
    out: dict[str, dict] = {}
    if not isinstance(record, dict):
        return out

    def put(bucket, name, value, unit=None, stat=None):
        v = _num(value)
        if v is None or not name:
            return
        # "ttft_ms" with stat p95 becomes "ttft_ms_p95"; a median (or unlabelled value) keeps its name.
        key = str(name) if stat in (None, "", "median", "p50", "value", "threshold") else f"{name}_{stat}"
        out.setdefault(str(bucket), {})[key] = {"value": v, "unit": unit, "stat": stat}

    for m in record.get("metrics") or []:
        if isinstance(m, dict):
            put(_bucket_of(m), m.get("name"), m.get("value"), m.get("unit"), m.get("stat"))
    data = record.get("data") if isinstance(record.get("data"), dict) else {}
    for key in ("per_bucket", "buckets"):
        block = data.get(key)
        if isinstance(block, dict):
            for b, mets in block.items():
                if isinstance(mets, dict):
                    for name, value in mets.items():
                        if name not in out.get(str(b), {}):
                            unit = value.get("unit") if isinstance(value, dict) else None
                            put(b, name, value, unit, "median")
    return out


ROW_METRICS = (("ttft_ms", "ms"), ("tpot_ms", "ms"), ("decode_tok_s", "tok/s"), ("e2e_ms", "ms"))


def request_rows(record: dict | None, store_root: Path) -> list[dict] | None:
    """The per-request rows of a ``bench serve`` record (its ``requests`` artifact), or None."""
    if not isinstance(record, dict) or not record.get("evidence_id"):
        return None
    art = next((a for a in record.get("artifacts") or [] if isinstance(a, dict) and a.get("role") == "requests"),
               None)
    if not art:
        return None
    path = store_root / record["evidence_id"] / art["path"]
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError):
        return None


def round_metrics_from_rows(rows: list[dict]) -> tuple[int | None, dict]:
    """``(round, {bucket: {metric: ...}})`` for the latest round in ``rows``: medians over its requests.

    A single-mode record extended with ``--extend`` holds every earlier round too; its summary
    metrics are cumulative, so the value of *this* round comes from its own requests. Buckets
    are the prompts' buckets plus ``all``; ``failed_requests`` counts the round's failures.
    """
    rounds = [r.get("round") for r in rows if isinstance(r.get("round"), int)]
    if not rounds:
        return None, {}
    k = max(rounds)
    mine = [r for r in rows if r.get("round") == k]
    out: dict[str, dict] = {}
    for bucket in sorted({str(r.get("bucket") or "all") for r in mine}) + ["all"]:
        sel = [r for r in mine if bucket == "all" or str(r.get("bucket") or "all") == bucket]
        ok = [r for r in sel if r.get("ok")]
        mets = {}
        for name, unit in ROW_METRICS:
            vals = [float(r[name]) for r in ok if isinstance(r.get(name), (int, float))]
            if vals:
                mets[name] = {"value": statistics.median(vals), "unit": unit, "stat": "median", "n": len(vals)}
        mets["failed_requests"] = {"value": float(len(sel) - len(ok)), "unit": "requests", "stat": "count",
                                   "n": len(sel)}
        out[bucket] = mets
    return k, out


# Stat suffixes bucket_metrics() appends to a metric name ("ttft_ms" + "p95_at_n_star").
_STAT_SUFFIXES = ("_at_n_star", "_threshold", "_median", "_mean", "_p50", "_p90", "_p95", "_p99")
COUNT_METRICS = ("failed_requests",)  # reported per engine, never ranked


def base_metric(metric: str) -> str:
    """The metric without its stat suffixes: ``ttft_ms_p95_at_n_star`` -> ``ttft_ms``."""
    m = metric.lower()
    stripped = True
    while stripped:
        stripped = False
        for suffix in _STAT_SUFFIXES:
            if m.endswith(suffix) and len(m) > len(suffix):
                m, stripped = m[: -len(suffix)], True
    return m


def higher_is_better(metric: str) -> bool | None:
    """True for rates and capacity, False for latencies, costs and failures, None when unknown.

    The base name decides (stat suffixes such as ``_p95`` or ``_at_n_star`` are stripped first),
    and latency markers are checked before rate markers, so ``ttft_ms_p95_at_n_star`` is a
    latency (lower is better) although it mentions N*.
    """
    m = base_metric(metric)
    if any(k in m for k in ("usd", "cost", "fail", "error")):
        return False
    if any(k in m for k in ("ttft", "tpot", "itl", "e2e", "latency")) or m.endswith(("_ms", "_seconds")):
        return False
    if m in ("n_star", "rate_star") or m.startswith("rate_star") or \
            any(k in m for k in ("tok_s", "tokens_per_s", "throughput", "goodput", "rps", "per_s", "per_hour")):
        return True
    if m.endswith("_s"):
        return False
    return None


def cell_status(returncode: int, record: dict | None, metrics: dict) -> tuple[str, str | None]:
    """``(status, detail)`` of a measured cell: ok, serve_failed or serve_no_metrics.

    A record whose requests all failed is ``serve_failed`` even though bench serve exited 0
    and the record has a ``failed_requests`` count: there is no latency to compare.
    """
    if returncode != 0 or record is None:
        return "serve_failed", None
    if not metrics:
        return "serve_no_metrics", None
    failed = (metrics.get("all") or {}).get("failed_requests") or {}
    n = failed.get("n")
    if isinstance(n, (int, float)) and n > 0 and failed.get("value", 0) >= n:
        return "serve_failed", f"all {int(n)} requests failed"
    return "ok", None


# --------------------------------------------------------------------------- one cell

def _tail(path: Path, n: int = 4000) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return data[-n:].decode(errors="replace")


def run_cell(spec: dict, k: int, position: int, cfg: dict, prompts: dict) -> dict:
    """Launch, check, measure and stop one engine for round ``k``. Never raises for engine failures."""
    run_dir: Path = cfg["run_dir"]
    label = spec["label"]
    base = f"http://{spec['host']}:{spec['port']}"
    log_rel = f"logs/{label}.r{k}.log"
    log_path = run_dir / log_rel
    cell: dict = {"round": k, "position": position, "label": label, "kind": spec["kind"], "status": None,
                  "log": log_rel, "t_start": time.time()}
    if log_path.exists():
        log_path.unlink()  # a rerun of the same round must not append to an old log
    env, env_info = servers.server_env(spec["env"])
    cell["env"] = env_info
    if servers.port_in_use(spec["host"], spec["port"]):
        cell.update(status="port_busy", detail=f"something already listens on {spec['host']}:{spec['port']}; "
                                               "refusing to benchmark it in place of the engine")
        cell["t_end"] = time.time()
        return cell
    gpu_before = cfg["gpu_reader"]()
    server = None
    try:
        try:
            server = servers.launch(spec["launch"], env, log_path, cwd=spec.get("cwd"))
        except OSError as e:
            cell.update(status="launch_failed", detail=str(e)[:300])
            return cell
        ready = servers.wait_ready(server, base + spec["ready_check"], cfg["startup_timeout_s"], cfg["poll_s"])
        cell["ready"] = ready
        if ready["status"] != "ready":
            cell["status"] = "startup_timeout" if ready["status"] == "timeout" else "exited_early"
            return cell
        model = servers.served_model(base)
        cell["served_model"] = model
        model = model or cfg.get("model_id")
        if not model:
            cell.update(status="serve_failed", detail="no model name: /v1/models listed none and the manifest "
                                                      "has no model.id")
            return cell
        if cfg["warmup"] > 0:
            cell["warmup"] = servers.warmup(base, model, prompts["head"], cfg["warmup"], cfg["warmup_max_tokens"],
                                            salt=f"ik-rounds-warmup-r{k}",
                                            chat_template_kwargs=cfg.get("chat_template_kwargs"))
            if cell["warmup"]["ok"] == 0:
                cell["status"] = "warmup_failed"
                return cell
        before = servers.scrape_metrics(base)
        extend = cfg["last_ev"].get(label) if cfg["mode"] == "single" else None
        ctx = {"base_url": base, "prompts": str(cfg["prompts_path"]), "mode": cfg["mode"], "label": label,
               "round": k, "run_dir": str(run_dir), "model": model, "engine_kind": spec["kind"],
               "engine_version": spec.get("version") or "", "extend": extend or "",
               "ref_e2e_ms": cfg.get("ref_e2e_ms") if cfg["mode"] == "load" else None,
               "chat_template_kwargs": cfg.get("chat_template_kwargs"), "extra_body": cfg.get("extra_body")}
        if cfg["mode"] == "load":
            sched = cfg["schedule"]
            shared = sched.get("fixed_by") or ctx["ref_e2e_ms"] is not None
            cell["load_schedule"] = {"ref_e2e_ms": ctx["ref_e2e_ms"],
                                     "source": sched["source"] if shared else "this engine's own warm-up"}
        argv = serve_argv(cfg["serve_cmd"], ctx, cfg["serve_args"])
        child_env = dict(os.environ)
        # Same evidence store as this run; and tell a lock-aware client that the GPU lock is already held
        # (by this process) so it must not try to take it again.
        root = str(ev.default_root().resolve())
        # The child's record is read from its stdout (parse_serve_output): full JSON, never the terse view.
        child_env.update({"INFERKIT_RUNS_DIR": root, "INFERENCEKIT_RUNS_DIR": root,
                          "INFERENCEKIT_GPU_LOCK_HELD": str(cfg.get("gpu_lock_path") or ""),
                          "INFERENCEKIT_OUTPUT": "json"})
        t0 = time.monotonic()
        res = proc.run(argv, timeout=cfg["serve_timeout_s"], env=child_env)  # a timeout kills its process group
        rc, so, se = res.returncode, res.stdout, res.stderr
        after = servers.scrape_metrics(base)
        record = parse_serve_output(so)
        cell["serve"] = {"returncode": rc, "seconds": round(time.monotonic() - t0, 3),
                         "evidence_id": record.get("evidence_id") if record else None,
                         "stderr_tail": ev.redact_text(se[-1500:]) if rc != 0 else None}
        prom_dir = run_dir / "prometheus"
        if before is not None and after is not None:
            prom_dir.mkdir(parents=True, exist_ok=True)
            (prom_dir / f"{label}.r{k}.before.txt").write_text(before)
            (prom_dir / f"{label}.r{k}.after.txt").write_text(after)
        cell["prometheus"] = servers.metrics_delta(before, after)
        rows = request_rows(record, ev.default_root())
        if rows:
            serve_round, cell["metrics"] = round_metrics_from_rows(rows)
            cell["metrics_source"] = f"requests artifact, round {serve_round} (median over its requests)"
        elif record is not None and (record.get("data") or {}).get("extends"):
            cell["metrics"] = {}  # cumulative over earlier rounds: not this round's value
            cell["metrics_source"] = "extended record without per-request rows: not used"
        else:
            cell["metrics"] = bucket_metrics(record)
            cell["metrics_source"] = "record metrics"
        if record is not None and rc == 0 and record.get("evidence_id") and cfg["mode"] == "single":
            cfg["last_ev"][label] = record["evidence_id"]
        cell["status"], detail = cell_status(rc, record, cell["metrics"])
        if detail:
            cell["detail"] = detail
        if cfg["mode"] == "load" and cell["status"] == "ok":
            used = reference_e2e_ms(record, ev.default_root())
            cell["load_schedule"]["ref_e2e_ms_used"] = used
            sched = cfg["schedule"]
            if not sched.get("fixed_by") and cfg.get("ref_e2e_ms") is None and used is not None:
                # The first load cell measured it with its warm-up; every later cell gets the same value.
                cfg["ref_e2e_ms"] = used
                sched.update(ref_e2e_ms=used, source=f"warm-up of {label} r{k} (the first load cell), passed as "
                                                     "--ref-e2e-ms to every later cell", pilot=f"{label} r{k}")
        return cell
    finally:
        if server is not None:
            cell["stop"] = servers.stop(server, cfg["stop_timeout_s"])
            if not cell["stop"]["group_gone"] and cell["status"] == "ok":
                cell["status"] = "stop_failed"
            cell["gpu_memory"] = servers.wait_gpu_memory(gpu_before, cfg["gpu_mem_tolerance_mib"],
                                                         cfg["gpu_mem_wait_s"], reader=cfg["gpu_reader"])
            resolved = servers.parse_resolved_config(spec["kind"], log_path.read_text(errors="replace")
                                                     if log_path.exists() else "")
            rc_dir = run_dir / "resolved_config"
            rc_dir.mkdir(parents=True, exist_ok=True)
            (rc_dir / f"{label}.r{k}.json").write_text(json.dumps(ev.redact(resolved), indent=2) + "\n")
            cell["resolved_config"] = {"status": resolved["status"], "version": resolved.get("version"),
                                       "validity": resolved.get("validity"),
                                       "warnings": len(resolved["warnings"]),
                                       "unknown_settings": resolved["unknown_settings"],
                                       "self_reported": resolved.get("self_reported") or [],
                                       "file": f"resolved_config/{label}.r{k}.json"}
            if cell["status"] in ("startup_timeout", "exited_early"):
                cell["log_tail"] = ev.redact_text(_tail(log_path, 1500))
        cell["t_end"] = time.time()


# --------------------------------------------------------------------------- summary

def summarize(cells: list[dict], labels: list[str], resamples: int = stats.DEFAULT_RESAMPLES,
              seed: int = 0, mode: str = "single") -> dict:
    """Medians over rounds with bootstrap CIs, and ratios to the best engine with paired CIs.

    An engine with any failed request in a bucket (over all its rounds) is not eligible as
    the best engine in that bucket and gets no ratio there: its latency medians cover only
    the requests it answered. ``failed_requests`` itself is reported (median and total over
    rounds) but never ranked.
    """
    values: dict[tuple[str, str, str], dict[int, float]] = {}
    units: dict[str, str | None] = {}
    requests: dict[tuple[str, str], float] = {}
    for c in cells:
        if c.get("status") not in STATUSES_OK:
            continue
        for bucket, mets in (c.get("metrics") or {}).items():
            for name, m in mets.items():
                values.setdefault((c["label"], bucket, name), {})[c["round"]] = m["value"]
                units.setdefault(name, m.get("unit"))
                if name == "failed_requests" and isinstance(m.get("n"), (int, float)):
                    requests[(c["label"], bucket)] = requests.get((c["label"], bucket), 0) + m["n"]
    failures = {(lb, b): sum(v.values()) for (lb, b, n), v in values.items() if n == "failed_requests"}
    rows = []
    groups = sorted({(b, n) for (_, b, n) in values})
    for bucket, name in groups:
        direction = None if name in COUNT_METRICS else higher_is_better(name)
        per = {}
        for label in labels:
            v = values.get((label, bucket, name))
            if not v:
                continue
            xs = [v[r] for r in sorted(v)]
            ci = stats.bootstrap_ci(xs, ps=(50,), resamples=resamples, seed=seed)[50]
            per[label] = {"median": statistics.median(xs), "ci95": list(ci), "n": len(xs), "by_round": v}
        if not per:
            continue
        eligible = [lb for lb in per if not failures.get((lb, bucket))]
        best = None
        if direction is not None and eligible:
            pick = max if direction else min
            best = pick(eligible, key=lambda lb: per[lb]["median"])
        for label, info in per.items():
            failed = failures.get((label, bucket), 0.0)
            row = {"engine": label, "bucket": bucket, "metric": name, "unit": units.get(name),
                   "higher_is_better": direction, "median": info["median"], "ci95": info["ci95"], "n": info["n"],
                   "best": best, "ratio_to_best": None, "ratio_ci95": [None, None], "n_pairs": 0,
                   "failed_requests": failed}
            if name in COUNT_METRICS:
                row["total"] = sum(info["by_round"].values())
            if failed and direction is not None:
                row["excluded_from_best"] = (f"{failed:g} failed request(s) over {info['n']} round(s): latencies "
                                             "cover only the answered requests")
            elif best is not None:
                b, c = per[best]["by_round"], info["by_round"]
                paired = sorted(set(b) & set(c))
                if label == best:
                    row.update(ratio_to_best=1.0, n_pairs=len(paired))
                elif paired:
                    r = stats.paired_ratio_ci([b[i] for i in paired], [c[i] for i in paired], resamples=resamples,
                                              seed=seed)
                    row.update(ratio_to_best=r["ratio"], ratio_ci95=list(r["ci95"]), n_pairs=r["n_pairs"])
            rows.append(row)
    fail_list = [{"engine": lb, "bucket": b, "failed_requests": f, "requests": requests.get((lb, b)),
                  "rounds": len(values.get((lb, b, "failed_requests")) or {})}
                 for (lb, b), f in sorted(failures.items()) if f]
    if mode == "load":
        median_def = ("median over rounds of the per-round value from bench serve's capacity record (N*, and "
                      "goodput, tok/s, p95 latencies and cost at N*) (derived)")
    else:
        median_def = ("median over rounds of the per-round value (median over that round's requests in the "
                      "bucket, from bench serve's requests artifact) (observed)")
    return {"rows": rows, "failures": fail_list, "definitions": {
        "median": median_def,
        "ci95": f"bootstrap 95% CI of that median, rounds resampled ({'derived' if mode == 'load' else 'observed'})",
        "ratio_to_best": "median(engine) / median(best engine) over rounds both completed, with a paired "
                         "bootstrap 95% CI resampling rounds (derived); best = lowest for latencies, highest "
                         "for rates, among engines with no failed request in the bucket",
        "failed_requests": "requests that failed in a round (median over rounds; 'total' sums them); an engine "
                           "with any failure in a bucket is never best there and gets no ratio"}}


# --------------------------------------------------------------------------- adaptive stop

def ci_half_widths(summary: dict, labels: list[str], metric: str = ADAPTIVE_METRIC,
                   bucket: str = ADAPTIVE_BUCKET) -> dict:
    """Per engine, the relative half-width of the 95% CI the adaptive stop reads: (hi - lo) / 2 / value.

    The value is the engine's ratio to the best engine (paired bootstrap CI over rounds). The best
    engine is the reference of every ratio and is not checked (``counted`` False). An engine with no
    ratio (failed requests, or no round shared with the best) and the only engine of a run use the CI
    of their median instead. ``width`` None: fewer than 2 rounds, so no CI yet.
    """
    rows = {r["engine"]: r for r in summary["rows"] if r["metric"] == metric and r["bucket"] == bucket}
    out = {}
    for label in labels:
        r = rows.get(label)
        if r is None:
            out[label] = {"width": None, "basis": "no successful round", "counted": True}
            continue
        if len(labels) > 1 and r["best"] == label:
            out[label] = {"width": None, "basis": "best engine (the reference of every ratio)", "counted": False}
            continue
        lo, hi = r["ratio_ci95"]
        if r["ratio_to_best"] and lo is not None and hi is not None and r["n_pairs"] >= 2:
            out[label] = {"width": (hi - lo) / 2 / r["ratio_to_best"], "basis": "ratio_ci95", "n": r["n_pairs"],
                          "counted": True}
            continue
        lo, hi = r["ci95"]
        if r["median"] and lo is not None and hi is not None and r["n"] >= 2:
            out[label] = {"width": (hi - lo) / 2 / abs(r["median"]), "basis": "ci95 of the median", "n": r["n"],
                          "counted": True}
        else:
            out[label] = {"width": None, "basis": "fewer than 2 rounds", "counted": True}
    return out


def adaptive_check(cells: list[dict], labels: list[str], k: int, ci_target: float) -> dict:
    """After round ``k``: the CI half-widths (``ci_half_widths``) and whether every checked engine meets
    ``ci_target``. Only the primary metric is summarized (same seed as the final summary, so the same CIs)."""
    primary = []
    for c in cells:
        mets = (c.get("metrics") or {}).get(ADAPTIVE_BUCKET) or {}
        keep = {n: m for n, m in mets.items() if n in (ADAPTIVE_METRIC, "failed_requests")}
        primary.append({**c, "metrics": {ADAPTIVE_BUCKET: keep} if keep else {}})
    widths = ci_half_widths(summarize(primary, labels), labels)
    checked = [w for w in widths.values() if w["counted"]]
    met = bool(checked) and all(w["width"] is not None and w["width"] <= ci_target for w in checked)
    return {"round": k, "met": met, "widths": widths}


def _adaptive_plan(n_engines: int, rounds: int | None, min_rounds: int | None, ci_target: float,
                   mode: str) -> tuple[dict, int]:
    """The adaptive-stop settings recorded in plan.json, and the maximum number of rounds."""
    if mode != "single":
        raise OpError("--ci-target (adaptive stop) is for single mode: it reads each round's decode_tok_s")
    if not 0 < ci_target < 1:
        raise OpError("--ci-target is a relative CI half-width between 0 and 1, e.g. 0.01 for +/-1%")
    period = len(williams_square(n_engines)) or 1
    rounds_source = "--rounds"
    if rounds is None:
        rounds, rounds_source = default_rounds(n_engines, ADAPTIVE_MAX_ROUNDS), \
            f"default: {ADAPTIVE_MAX_ROUNDS} rounded up to whole Williams periods"
    if min_rounds is None:
        min_rounds = min(default_rounds(n_engines), rounds)
    if min_rounds < 2:
        raise OpError("--min-rounds must be at least 2: a confidence interval needs 2 rounds")
    first = period * -(-min_rounds // period)
    if first >= rounds:
        raise OpError(f"--ci-target cannot stop anything: the first check comes after round {first} (whole Williams "
                      f"periods of {period} for {n_engines} engine(s), from --min-rounds {min_rounds} on) and "
                      f"--rounds is {rounds}; raise --rounds")
    rule = (f"stop after a whole Williams period (every {period} round(s)) once at least {min_rounds} rounds ran, "
            f"when every engine's {ADAPTIVE_METRIC} [{ADAPTIVE_BUCKET}] ratio to the best engine has a 95% CI "
            f"half-width <= {ci_target:g} x the ratio (the best engine is the reference and is not checked; an "
            f"engine without a ratio, or the only engine, uses the CI of its median); at most {rounds} rounds")
    return {"ci_target": ci_target, "min_rounds": min_rounds, "max_rounds": rounds, "max_rounds_source": rounds_source,
            "period": period, "first_check_round": first, "metric": ADAPTIVE_METRIC, "bucket": ADAPTIVE_BUCKET,
            "rule": rule, "checks": []}, rounds


def _fmt(x, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{digits}g}" if abs(x) < 1000 else f"{x:.0f}"
    return str(x)


def markdown(summary: dict, plan: dict, cells: list[dict], resolved: dict) -> str:
    load = plan["mode"] == "load"
    cls = "derived" if load else "observed"
    lines = [f"# bench rounds: {plan['mode']} mode, {plan['rounds']} rounds, {len(plan['engines'])} engines", "",
             f"Order: {plan['design']['design']} design, period {plan['design']['period']}, "
             f"balanced: {'yes' if plan['design']['balanced'] else 'NO'}. Prompt set {plan['prompts']['name']} "
             f"(sha256 {plan['prompts']['sha256'][:12]}..., n={plan['prompts']['n']}).", ""]
    adaptive = plan.get("adaptive")
    if adaptive:
        if plan.get("stopped_early"):
            how = (f"stopped early after round {plan['stopped_early']['round']} of at most "
                   f"{adaptive['max_rounds']}: the CI target was met")
        else:
            how = f"ran all {plan['rounds']} rounds; the CI target was {'met' if adaptive.get('met') else 'NOT met'}"
        lines += [f"Adaptive stop: {how}. Rule: {adaptive['rule']}.", ""]
    if plan.get("extra_body"):
        lines += [f"Every request carried --extra-body {json.dumps(plan['extra_body']['value'], sort_keys=True)} "
                  f"({plan['extra_body']['source']}).", ""]
    if load:
        sched = plan.get("load_schedule") or {}
        same = "the same for every engine" if sched.get("shared") else "NOT the same for every engine"
        ref = f" (--ref-e2e-ms {_fmt(sched['ref_e2e_ms'])} ms)" if sched.get("ref_e2e_ms") else ""
        lines += ["Medians and CIs are [derived]: each round's value comes from bench serve's capacity record "
                  "(N* and the values at N*, derived from the client-side timings of the load levels); ratios "
                  "are [derived].", "",
                  f"Arrival schedule: {same}{ref}; {sched.get('source') or 'unknown'}.", ""]
    else:
        lines += ["Medians and CIs are [observed] (client-side timer in `bench serve`); ratios are [derived].", ""]
    lines += ["An engine marked [FAILURES] had failed requests in that bucket: its latencies cover only the "
              "requests it answered, so it is never named best there and gets no ratio.", "",
              f"| engine | bucket | metric | median [{cls}] | 95% CI | rounds | ratio to best [derived] | "
              "ratio 95% CI |", "|---|---|---|---|---|---|---|---|"]
    for r in summary["rows"]:
        unit = f" {r['unit']}" if r.get("unit") else ""
        best = " (best)" if r["engine"] == r["best"] else ""
        if r.get("failed_requests"):
            best += " [FAILURES]"
        if (resolved.get(r["engine"]) or {}).get("violations"):
            best += " [HARD GATE]"
        if (resolved.get(r["engine"]) or {}).get("status") == "SELF_REPORTED":
            best += " [SELF-REPORTED CONFIG]"
        ratio_ci = "-" if r["engine"] == r["best"] or r["ratio_to_best"] is None else \
            f"{_fmt(r['ratio_ci95'][0])} - {_fmt(r['ratio_ci95'][1])}"
        lines.append(f"| {r['engine']}{best} | {r['bucket']} | {r['metric']} | {_fmt(r['median'])}{unit} | "
                     f"{_fmt(r['ci95'][0])} - {_fmt(r['ci95'][1])} | {r['n']} | {_fmt(r['ratio_to_best'])} | "
                     f"{ratio_ci} |")
    if not summary["rows"]:
        lines.append("| (no successful rounds) | | | | | | | |")
    if summary.get("failures"):
        lines += ["", "## Failed requests", "", "| engine | bucket | failed | requests | rounds |",
                  "|---|---|---|---|---|"]
        for f in summary["failures"]:
            lines.append(f"| {f['engine']} | {f['bucket']} | {_fmt(f['failed_requests'])} | "
                         f"{_fmt(f['requests'])} | {f['rounds']} |")
    lines += ["", "## Resolved configuration (from server logs)", "",
              "SELF_REPORTED: the settings come only from the engine process's own `inferencekit_resolved_config` "
              "line, which nothing verifies; for vllm, sglang, llamacpp and trtllm such a line is ignored.", "",
              "| engine | status | version | warnings | unknown settings | hard-gate violations | "
              "self-reported (unverified) |", "|---|---|---|---|---|---|---|"]
    for label, info in resolved.items():
        lines.append(f"| {label} | {info['status']} | {info.get('version') or 'n/a'} | {info['warnings']} | "
                     f"{', '.join(info['unknown_settings']) or '-'} | {'; '.join(info['violations']) or '-'} | "
                     f"{', '.join(info.get('self_reported') or []) or '-'} |")
    failed = [c for c in cells if c.get("status") != "ok"]
    if failed:
        lines += ["", "## Failed cells", "", "| round | engine | status | detail |", "|---|---|---|---|"]
        for c in failed:
            detail = c.get("detail") or (c.get("ready") or {}).get("last") or (c.get("serve") or {}).get("returncode")
            lines.append(f"| {c['round']} | {c['label']} | {c['status']} | {str(detail)[:120]} |")
    return "\n".join(lines) + "\n"


def resolved_summary(cells: list[dict], labels: list[str]) -> dict:
    """Per engine: the weakest resolved-config status over rounds, and what varied."""
    rank = servers.STATUS_RANK
    out = {}
    for label in labels:
        mine = [c["resolved_config"] for c in cells if c["label"] == label and c.get("resolved_config")]
        if not mine:
            out[label] = {"status": "UNPROVEN", "version": None, "warnings": 0, "unknown_settings": [],
                          "violations": [], "attention_backends": [], "self_reported": [], "rounds": 0}
            continue
        worst = max(mine, key=lambda r: rank.get(r["status"], rank["UNPROVEN"]))
        violations = sorted({v for r in mine for v in ((r.get("validity") or {}).get("violations") or [])})
        backends = sorted({str((r.get("validity") or {}).get("attention_backend")) for r in mine
                           if (r.get("validity") or {}).get("attention_backend")})
        out[label] = {"status": worst["status"], "version": next((r["version"] for r in mine if r.get("version")),
                                                                 None),
                      "warnings": max(r["warnings"] for r in mine),
                      "unknown_settings": sorted({u for r in mine for u in r["unknown_settings"]}),
                      "violations": violations, "attention_backends": backends, "rounds": len(mine),
                      "self_reported": sorted({k for r in mine for k in (r.get("self_reported") or [])}),
                      "files": [r["file"] for r in mine]}
    return out


# --------------------------------------------------------------------------- the run

def run_rounds(manifest_path: str, prompts_path: str, rounds: int | None, mode: str, engines: str | None = None,
               run_dir: str | None = None, gpu_lock_path: str | None = None, startup_timeout_s: float = 1200,
               stop_timeout_s: float = 30, serve_timeout_s: float = 7200, warmup: int = 2,
               warmup_max_tokens: int = 16, serve_cmd: str | None = None, serve_args: str | None = None,
               gpu_mem_tolerance_mib: int = 512, gpu_mem_wait_s: float = 60, poll_s: float = 0.5,
               gpu_reader=None, ref_e2e_ms: float | None = None, chat_template_kwargs: str | dict | None = None,
               ci_target: float | None = None, min_rounds: int | None = None,
               extra_body: str | dict | None = None) -> dict:
    """Run every (round, engine) cell under the GPU lock; return plan, cells, summary and paths.

    ``rounds`` None: the smallest multiple of the Williams period that is at least 3.
    ``ref_e2e_ms`` (load mode): passed as ``--ref-e2e-ms`` to every cell; None: taken from the
    first load cell's warm-up and passed to every later one (unless ``serve_args`` fixes the schedule).
    ``ci_target`` (single mode): adaptive stop, see the module docstring; ``rounds`` is then the maximum
    (None: 12 rounded up to whole periods) and ``min_rounds`` the minimum (None: the default rounds).
    ``extra_body``: JSON object for every bench serve call's ``--extra-body`` (None: the manifest's
    ``workload.extra_body``).
    """
    if mode not in ("single", "load"):
        raise OpError("mode must be 'single' or 'load'")
    manifest = load_manifest(manifest_path)
    specs = engine_specs(manifest, engines)
    if not specs:
        raise OpError("no engines selected")
    if min_rounds is not None and ci_target is None:
        raise OpError("--min-rounds applies only with --ci-target (the adaptive stop)")
    if rounds is not None and rounds < 1:
        raise OpError("rounds must be >= 1")
    adaptive = None
    if ci_target is not None:
        adaptive, rounds = _adaptive_plan(len(specs), rounds, min_rounds, ci_target, mode)
    if rounds is None:
        rounds = default_rounds(len(specs))
    if ref_e2e_ms is not None and ref_e2e_ms <= 0:
        raise OpError("--ref-e2e-ms must be > 0")
    fixed_by = schedule_fixed_by(serve_args) if mode == "load" else None
    if mode != "load":
        schedule = {}
    elif fixed_by:
        schedule = {"fixed_by": fixed_by, "ref_e2e_ms": None,
                    "source": f"{fixed_by} in --serve-args (the same for every engine)"}
        ref_e2e_ms = None
    elif ref_e2e_ms is not None:
        schedule = {"ref_e2e_ms": ref_e2e_ms, "source": "--ref-e2e-ms of bench rounds, passed to every cell"}
    else:
        schedule = {"ref_e2e_ms": None, "source": "not yet measured (the first load cell's warm-up sets it)"}
    ctk, ctk_source = template_kwargs(chat_template_kwargs, manifest)
    ctk, ctk_source = _serve_args_template_kwargs(serve_args, ctk, ctk_source)
    body, body_source, body_warnings = extra_body_for(extra_body, manifest, serve_args)
    prompts = read_prompts(prompts_path)
    rdir = Path(run_dir) if run_dir else Path(manifest_path).resolve().parent
    ctk, ctk_source = _pinned_template_kwargs(ctk, ctk_source, [rdir, Path(manifest_path).resolve().parent])
    rdir.mkdir(parents=True, exist_ok=True)
    labels = [s["label"] for s in specs]
    design = round_orders(labels, rounds)
    plan = {"mode": mode, "rounds": rounds, "engines": [{k: s[k] for k in ("label", "name", "kind", "version",
                                                                           "port", "ready_check", "role")}
                                                        for s in specs],
            "design": design, "prompts": {k: prompts[k] for k in ("name", "sha256", "n", "buckets")},
            "settings": {"startup_timeout_s": startup_timeout_s, "stop_timeout_s": stop_timeout_s,
                         "serve_timeout_s": serve_timeout_s, "warmup": warmup, "warmup_max_tokens": warmup_max_tokens,
                         "gpu_mem_tolerance_mib": gpu_mem_tolerance_mib, "gpu_mem_wait_s": gpu_mem_wait_s,
                         "serve_args": serve_args, "serve_cmd_custom": bool(serve_cmd or os.environ.get(SERVE_CMD_ENV)),
                         "warmup_kind": servers.WARMUP_KIND},
            "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()}
    if mode == "load":
        plan["load_schedule"] = schedule
    plan["chat_template_kwargs"] = {"value": ctk or None, "source": ctk_source}
    if body:
        plan["extra_body"] = {"value": body, "source": body_source}
        if body_warnings:
            plan["extra_body"]["warnings"] = body_warnings
    if adaptive is not None:
        plan["adaptive"] = adaptive
    cfg = {"run_dir": rdir, "prompts_path": Path(prompts_path).resolve(), "mode": mode,
           "startup_timeout_s": startup_timeout_s, "stop_timeout_s": stop_timeout_s,
           "serve_timeout_s": serve_timeout_s, "warmup": warmup, "warmup_max_tokens": warmup_max_tokens,
           "serve_cmd": serve_cmd, "serve_args": serve_args, "gpu_mem_tolerance_mib": gpu_mem_tolerance_mib,
           "gpu_mem_wait_s": gpu_mem_wait_s, "poll_s": poll_s,
           "gpu_reader": gpu_reader or servers.gpu_memory_used_mib,
           "model_id": (manifest.get("model") or {}).get("id"), "last_ev": {},
           "ref_e2e_ms": ref_e2e_ms, "schedule": schedule, "chat_template_kwargs": ctk or None,
           "extra_body": body or None}
    out_dir = rdir / "rounds"
    out_dir.mkdir(parents=True, exist_ok=True)
    cells: list[dict] = []
    with servers.gpu_lock(gpu_lock_path, owner=f"inferencekit bench rounds {rdir.name}") as lock:
        plan["gpu_lock"] = lock["path"]
        cfg["gpu_lock_path"] = lock["path"]
        plan["started"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        (out_dir / "plan.json").write_text(json.dumps(ev.redact(plan), indent=2) + "\n")
        by_label = {s["label"]: s for s in specs}
        for k, order in enumerate(design["orders"], 1):
            for pos, label in enumerate(order):
                cell = run_cell(by_label[label], k, pos, cfg, prompts)
                cells.append(cell)
                (out_dir / f"{label}.r{k}.json").write_text(json.dumps(ev.redact(cell), indent=2) + "\n")
            if adaptive is None:
                continue
            whole = k % adaptive["period"] == 0 and k >= adaptive["first_check_round"]
            if whole or k == rounds:  # the last round is checked too, to record whether the target was met
                check = adaptive_check(cells, labels, k, ci_target)
                adaptive["checks"].append(check)
                adaptive["met"] = check["met"]
                if check["met"] and whole and k < rounds:
                    plan["stopped_early"] = {"round": k, "rule": adaptive["rule"], "widths": check["widths"]}
                    plan["rounds"] = k
                    plan["design"] = design = round_orders(labels, k)
                    break
        if adaptive is not None:
            adaptive["stopped_early"] = "stopped_early" in plan
            adaptive.setdefault("met", False)
        plan["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if mode == "load":
        refs = {c["load_schedule"].get("ref_e2e_ms_used") for c in cells
                if c.get("status") == "ok" and c.get("load_schedule")}
        schedule["shared"] = bool(schedule.get("fixed_by")) or ref_e2e_ms is not None or \
            (len(refs) == 1 and None not in refs)
        if not schedule.get("fixed_by") and schedule.get("ref_e2e_ms") is None:
            schedule["source"] = ("no reference latency reported by bench serve: each engine's concurrency levels "
                                  "were turned into rates with its own warm-up e2e")
        plan["load_schedule"] = schedule
    (out_dir / "plan.json").write_text(json.dumps(ev.redact(plan), indent=2) + "\n")
    summary = summarize(cells, labels, mode=mode)
    resolved = resolved_summary(cells, labels)
    plan["serve_evidence_final"] = dict(cfg["last_ev"])  # single mode: each engine's record with all its rounds
    (out_dir / "summary.json").write_text(json.dumps(ev.redact({"plan": plan, "summary": summary,
                                                                "resolved_config": resolved}), indent=2) + "\n")
    md = markdown(summary, plan, cells, resolved)
    (out_dir / "summary.md").write_text(md)
    return {"plan": plan, "cells": cells, "summary": summary, "resolved_config": resolved, "markdown": md,
            "run_dir": rdir, "manifest": manifest, "prompts": prompts}
