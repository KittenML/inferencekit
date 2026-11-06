"""intake template / intake validate / intake budget: the input of the optimizing-inference flow.

An intake file (``intake.yaml``) says what to make fast (model + pinned revision +
allowed precisions), on what hardware and how to reach it, for which goal
(single-user speed or capacity, with optional targets), with which input/output
modes and buckets, and inside which approved budget. The orchestrator skill reads it
once at stage 1 and never changes it afterwards: the targets in it are the
pre-registered claims the final report is judged against.

* ``intake template`` prints an annotated YAML file to start from.
* ``intake validate <file>`` checks it against ``schemas/intake.v1.json`` plus the
  rules a schema cannot express (an unattended run needs an approved budget, a
  capacity goal needs an SLO, a lossy run needs a quality budget, ...), fills defaults and
  returns the normalized intake with warnings and the derived plan (stop rule, fidelity rule
  and quality budget, speculation and route, budget cap, per-stage time boxes).
  With ``--run-dir`` it also starts (or resumes) ``runs/<run-id>/``: a redacted copy of
  the intake (access details removed), ``intake.normalized.json``, ``state.json`` and
  ``QUESTIONS.md``. A run whose recorded intake_sha256 differs is refused: the targets
  are pre-registered, so a changed intake needs a new run id.
* ``intake budget <run-dir>`` reads ``state.json`` and says how much of the budget and
  of the current stage's time box is spent. Read-only; the skill runs it before every
  long step.

No numbers here are measurements: the budget and time-box figures are arithmetic on
the intake and the clock (claim class ``derived``), so no evidence record is written.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import re
from importlib import resources
from pathlib import Path

from inferencekit.canonical import sha256_hex
from inferencekit.registry import Arg, OpError, operation, utc_now

BUCKETS = ("very_short", "short", "medium", "long")

# Stage keys of state.json, in order, with the share of the run's hour cap each one gets.
# The report keeps a fixed reserve so a run that hits its budget still ends with a report.
STAGES = (
    ("1_intake", 0.02),
    ("2_hardware", 0.05),
    ("3_candidates", 0.18),
    ("4_tuning", 0.22),
    ("5_benchmark", 0.18),
    ("6_stop_point", 0.02),
    ("7_custom", 0.23),
    ("8_report", 0.10),
)
REPORT_RESERVE = dict(STAGES)["8_report"]

# name -> (better direction, unit)
TARGETS = {
    "decode_tok_s": ("higher", "tok/s"),
    "ttft_ms": ("lower", "ms"),
    "n_star": ("higher", "streams/GPU"),
    "decode_ceiling_fraction": ("higher", "fraction of roofline ceiling"),
    "ttfa_ms": ("lower", "ms"),
    "ttfa_p95_ms": ("lower", "ms"),
    "image_ms": ("lower", "ms per image"),
    "images_per_hour": ("higher", "images/h"),
    "rtf": ("lower", "x real time"),
    "usd_per_1m": ("lower", "USD per 1M units"),
    "clip_ms": ("lower", "ms per clip"),
    "clips_per_hour": ("higher", "clips/h"),
}
MODALITY_TARGETS = {
    "llm": {"decode_tok_s", "ttft_ms", "n_star", "decode_ceiling_fraction", "usd_per_1m"},
    "tts": {"ttfa_ms", "ttfa_p95_ms", "rtf", "n_star", "usd_per_1m"},
    "stt": {"rtf", "n_star", "usd_per_1m"},
    "video": {"clip_ms", "clips_per_hour"},
    "image": {"image_ms", "images_per_hour"},
}
DEFAULT_CONCURRENCY = [1, 2, 4, 8, 16, 32, 64]
ROOFLINE_DTYPES = {"bf16", "fp16", "fp8", "q8_0"}
# Run rules a constraint string can set ("after_target: keep_climbing"): name -> allowed values. after_target and
# fidelity are also schema fields (goals.after_target, fidelity); a field and a constraint must agree.
RULES = {"after_target": ("stop", "keep_climbing"), "fidelity": ("lossless", "lossy"),
         "speculation": ("off", "auto"), "route": ("build", "patch")}
_RULE_NAME = r"(after[\s_-]?target|fidelity|speculation|route)"
_RULE = re.compile(rf"^\s*{_RULE_NAME}\s*:\s*(.*)$", re.IGNORECASE | re.DOTALL)
_RULE_INSIDE = re.compile(rf"(?<![\w-]){_RULE_NAME}\s*:", re.IGNORECASE)
# Budget keys that fix how long a run may go (with keep_climbing they are where it ends).
CAP_KEYS = ("gpu_hours", "usd", "wall_clock_h")
QUALITY_BUDGET_KEYS = ("gsm8k_drop_pts_max", "top1_agreement_min", "kl_max", "nll_increase_max", "tv_max")
# Video (quality clip --fidelity lossy --budget): full-reference bounds on the engine's mean vs the BF16 reference.
VIDEO_BUDGET_KEYS = ("psnr_db_min", "ssim_min", "lpips_max")
SPEECH_BUDGET_KEYS = ("wer_max", "cer_max", "speaker_similarity_min")
# Degradation limits evaluated by a hash-bound external modality protocol, not by
# the built-in absolute scorers. Positive increase/drop is always worse.
RELATIVE_BUDGET_KEYS = {
    "image": ("lpips_excess_max", "ssim_loss_max", "ocr_cer_increase_max",
              "semantic_score_drop_max", "new_severe_regressions_max"),
    "tts": ("wer_increase_max", "cer_increase_max", "completion_failure_increase_max",
            "speaker_similarity_drop_max", "perceptual_score_drop_max", "new_severe_regressions_max"),
    "stt": ("wer_increase_max", "cer_increase_max", "completion_failure_increase_max",
            "new_severe_regressions_max"),
}
PERCEPTUAL_ACCEPTANCE = "perceptual_and_task_noninferiority"
PERCEPTUAL_BUDGET_KEYS = {
    "image": ("ocr_cer_increase_max", "semantic_score_drop_max", "new_severe_regressions_max"),
    "tts": RELATIVE_BUDGET_KEYS["tts"],
}
_DRIFT_DIAGNOSTICS = ("cross_engine_byte_identity", "cross_engine_tensor_equality", "rounding_drift",
                      "eager_compiled_drift", "repeated_run_variation")
DIAGNOSTIC_QUALITY_METRICS = {
    "image": ("lpips", "ssim", "psnr", *_DRIFT_DIAGNOSTICS),
    "tts": ("cross_engine_pcm_equality", "cross_engine_generated_code_equality", *_DRIFT_DIAGNOSTICS),
}
# KV cache dtypes below 16 bits. Under fidelity lossless they change numerics, unless the checkpoint itself specifies
# one (then the intake says checkpoint_default).
_LOSSY_KV = re.compile(r"^(u?int[248]|fp[468]|float[48]|f8|e[2-5]m[0-9]|nvfp4|mxfp[468]|i?q[2-8])(?![0-9])",
                       re.IGNORECASE)
# <angle>, CHANGEME, TODO, and __UPPER_SNAKE__ fill-in markers (case-sensitive, so __init__ is not one).
_PLACEHOLDER = re.compile(r"<[^<>]+>|\bCHANGE[_-]?ME\b|\bTODO\b|(?-i:(?<![A-Za-z0-9])__[A-Z][A-Z0-9_]*__(?![A-Za-z0-9]))",
                          re.IGNORECASE)
REDACTED = "<redacted>"
_SHA = re.compile(r"^[0-9a-f]{40}$")


# --------------------------------------------------------------------------- loading

def schema_path() -> Path:
    """The schema shipped in the wheel, or the repo copy in a source checkout."""
    try:
        packaged = resources.files("inferencekit").joinpath("schemas/intake.v1.json")
        if packaged.is_file():
            return Path(str(packaged))
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    return Path(__file__).resolve().parents[3] / "schemas" / "intake.v1.json"


def load_schema() -> dict:
    p = schema_path()
    if not p.is_file():
        raise OpError(f"intake schema not found at {p}: the wheel must ship schemas/intake.v1.json")
    return json.loads(p.read_text())


def parse_text(text: str, name: str = "intake") -> dict:
    """YAML (or JSON, which is valid YAML) to a dict."""
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            data = json.loads(text)
        except ValueError as e:
            raise OpError(f"{name}: invalid JSON: {e}") from None
    else:
        try:
            import yaml
        except ImportError:
            raise OpError("reading YAML needs PyYAML (pip install pyyaml); a JSON intake works without it") from None
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise OpError(f"{name}: invalid YAML: {e}") from None
    if not isinstance(data, dict):
        raise OpError(f"{name}: the top level must be a mapping (key: value), got {type(data).__name__}")
    return data


def load_file(path: str | os.PathLike) -> tuple[dict, bytes]:
    p = Path(path)
    if not p.is_file():
        raise OpError(f"intake file not found: {p}")
    raw = p.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise OpError(f"{p}: not UTF-8 text") from None
    return parse_text(text, p.name), raw


# --------------------------------------------------------------------------- checking

def _dotted(path) -> str:
    return ".".join(str(p) for p in path) or "(top level)"


def schema_problems(data: dict) -> tuple[list[str], list[str]]:
    """(errors, warnings). Unknown keys are warnings: probably a typo, never silently used."""
    import jsonschema

    errors, warnings = [], []
    validator = jsonschema.Draft202012Validator(load_schema())
    for err in sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path)):
        where = _dotted(err.absolute_path)
        if err.validator == "additionalProperties":
            known = sorted((err.schema.get("properties") or {}).keys())
            extra = sorted(set(err.instance) - set(known)) if isinstance(err.instance, dict) else []
            if list(err.absolute_path)[:3] == ["quality", "budget", "relative"]:
                errors.append(f"{where}: unknown relative budget key(s) {extra}; known keys: {known}")
            else:
                warnings.append(f"{where}: unknown key(s) {extra} ignored (typo?); known keys: {known}")
        elif err.validator == "oneOf":
            errors.append(f"{where}: {err.instance!r} is not a valid value "
                          f"({err.schema.get('description') or 'see schemas/intake.v1.json'})")
        else:
            errors.append(f"{where}: {err.message}")
    return errors, warnings


def _drop_unknown(data, schema):
    """Remove keys the schema does not define (they were reported as warnings)."""
    if not isinstance(data, dict) or not isinstance(schema, dict):
        return data
    props = schema.get("properties")
    if props is None or schema.get("additionalProperties", True) is not False:
        return data
    return {k: _drop_unknown(v, props[k]) for k, v in data.items() if k in props}


def placeholders(node, path=()) -> list[str]:
    """Dotted paths of string values that still look like template placeholders."""
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            out += placeholders(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out += placeholders(v, path + (i,))
    elif isinstance(node, str) and _PLACEHOLDER.search(node):
        out.append(_dotted(path))
    return out


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "x"


def constraint_rules(constraints) -> tuple[dict, list[str]]:
    """Run rules set by constraint strings, e.g. ``after_target: keep_climbing`` or ``speculation: off``.

    Returns ({name: value}, errors). An entry may hold several rules separated by ``;``. The name may be written
    with ``_``, ``-``, a space or nothing between its words ("after target"); the value is read at the start of
    the text after the colon, the same way ("keep climbing"), so a reason may follow it ("speculation: off, the
    task forbids it"). A recognised name with an unknown value, a second rule inside a reason, or two entries that
    disagree, is an error: a typo here would otherwise be ignored silently.
    """
    found, errors = {}, []
    for text in constraints or []:
        for part in text.split(";") if isinstance(text, str) else []:
            m = _RULE.match(part)
            if not m:
                continue
            name = "after_target" if m.group(1).lower().startswith("after") else m.group(1).lower()
            rest = m.group(2).strip()
            if _RULE_INSIDE.search(rest):
                errors.append(f"constraints: {part.strip()!r} holds more than one rule: give each rule its own "
                              "entry, or separate them with ';'")
                continue
            value = next((v for v in RULES[name]
                          if re.match(v.replace("_", r"[\s_-]+") + r"(?![\w-])", rest, re.IGNORECASE)), "")
            if value not in RULES[name]:
                errors.append(f"constraints: {part.strip()!r}: {name} must be one of {list(RULES[name])}")
            elif found.setdefault(name, value) != value:
                errors.append(f"constraints: {name} is given twice with different values "
                              f"({found[name]!r} and {value!r}); keep one")
    return found, errors


def normalize(data: dict) -> tuple[dict, list[str], list[str]]:
    """Fill defaults and apply the cross-field rules. Returns (intake, errors, warnings)."""
    errors, warnings = [], []
    d = json.loads(json.dumps(data))  # deep copy, JSON types only

    model = d["model"]
    if "precision_allowed" not in model:
        if "precision" in model:
            model["precision_allowed"] = [model["precision"]]
        else:
            errors.append("model: give precision (the checkpoint's own) and/or precision_allowed")
            model["precision_allowed"] = []
    if "precision" in model and model["precision_allowed"] and model["precision"] not in model["precision_allowed"]:
        warnings.append(f"model.precision {model['precision']!r} is not in model.precision_allowed "
                        f"{model['precision_allowed']}: every ranked engine converts the checkpoint; the "
                        "conversion itself must pass the quality gates")
    if len(model.get("precision_allowed", [])) > 1:
        warnings.append(f"model.precision_allowed has {len(model['precision_allowed'])} entries: each precision "
                        "is its own track; rows are never ranked across precisions")
    model.setdefault("kv_cache_dtype_allowed", ["checkpoint_default"])
    model.setdefault("tokenizer", model["id"])
    rev = model.get("revision")
    if rev is None:
        warnings.append("model.revision missing: stage 1 resolves the current commit sha and records it; "
                        "every engine must load that exact revision")
    elif not _SHA.match(rev) and not _PLACEHOLDER.search(rev):
        warnings.append(f"model.revision {rev!r} is not a 40-hex commit sha: stage 1 resolves it to one and "
                        "records it")

    hw = d["hardware"]
    hw.setdefault("count", 1)
    access = hw.setdefault("access", {"kind": "local"})
    if access.get("kind") == "ssh":
        access.setdefault("port", 22)
        key = access.get("key_path")
        if key and not _PLACEHOLDER.search(key) and not Path(os.path.expanduser(key)).exists():
            warnings.append("hardware.access.key_path does not exist on this machine")

    modality = d["modality"]
    goals = d.setdefault("goals", {})
    goals.setdefault("lead", "single_user")
    affordability = goals.get("affordability")
    if goals["lead"] == "affordability" and not affordability:
        errors.append("goals.lead affordability requires goals.affordability memory limits")
    if affordability:
        affordability.setdefault("offload_allowed", False)
        stretch = affordability.get("stretch_memory_gb_max")
        if stretch is not None and stretch >= affordability["memory_gb_max"]:
            errors.append("goals.affordability.stretch_memory_gb_max must be smaller than memory_gb_max")
    targets = goals.setdefault("targets", {})
    slo = goals.setdefault("slo", {})
    slo.setdefault("max_fail", 0)
    goals.setdefault("traffic", {})
    bad = sorted(set(targets) - {"buckets"} - MODALITY_TARGETS[modality])
    if bad:
        warnings.append(f"goals.targets {bad} do not apply to modality {modality} "
                        f"(use {sorted(MODALITY_TARGETS[modality])}); they are reported NOT RUN")
    needs_load = goals["lead"] == "capacity" or "n_star" in targets
    if needs_load and "ttft_p95_ms" not in slo:
        errors.append("goals.slo.ttft_p95_ms is required when the lead goal is capacity or a target is n_star "
                      "(N* is defined by the SLO)")
    if needs_load and modality == "llm" and "tpot_p95_ms" not in slo:
        warnings.append("goals.slo.tpot_p95_ms missing: N* will hold only the first-token SLO and failures")
    if "n_star" in targets and goals["lead"] != "capacity":
        warnings.append("goals.targets.n_star is set but lead is single_user: the capacity (load) path runs too")
    frac = targets.get("decode_ceiling_fraction")
    if frac is not None and frac > 0.9:
        warnings.append(f"goals.targets.decode_ceiling_fraction {frac} is above 0.9 of a measured-bandwidth "
                        "ceiling; few engines get there at batch 1")
    if frac is not None and not set(model["precision_allowed"]) <= ROOFLINE_DTYPES:
        warnings.append("goals.targets.decode_ceiling_fraction: 'inferencekit roofline' covers dense "
                        f"{sorted(ROOFLINE_DTYPES)} full-attention models only; for {model['precision_allowed']} (or MoE, "
                        "sliding-window or hybrid attention) stage 2 derives bytes/token from the checkpoint with the "
                        "skill's scripts/bytes_per_token.py and records the formula")

    io = d.setdefault("io_modes", {})
    io.setdefault("input", "complete")
    io.setdefault("output", "streaming")
    if io["input"] == "streaming":
        warnings.append("io_modes.input is streaming: engines without incremental input are NOT SUPPORTED for "
                        "it; any client-side emulation is labelled as emulation")
        if "input_rate" not in io:
            warnings.append("io_modes.input_rate missing: the replay pace of streaming input is unknown")

    wl = d.setdefault("workload", {})
    tts = wl.get("tts")
    if tts is not None and modality != "tts":
        errors.append("workload.tts applies only to modality tts")
    text_buckets = (tts or {}).get("text_length_buckets", [])
    names = [b["name"] for b in text_buckets]
    if len(names) != len(set(names)):
        errors.append("workload.tts.text_length_buckets names must be unique")
    if any(b["max_chars"] < b["min_chars"] for b in text_buckets):
        errors.append("workload.tts.text_length_buckets max_chars must be >= min_chars")
    d.setdefault("buckets", names or list(BUCKETS))
    d["buckets"] = [b for b in BUCKETS if b in d["buckets"]]
    if text_buckets and set(d["buckets"]) - set(names):
        errors.append("buckets must be defined in workload.tts.text_length_buckets when character bounds are supplied")
    if modality in ("image", "video"):  # modality workload set, not text length buckets
        if targets.pop("buckets", None) is not None:
            warnings.append(f"goals.targets.buckets do not apply to modality {modality}: a target holds over the "
                            "whole modality workload set")
    elif "buckets" in targets:
        outside = sorted(set(targets["buckets"]) - set(d["buckets"]))
        if outside:
            errors.append(f"goals.targets.buckets {outside} are not in buckets {d['buckets']}")
    elif any(k != "buckets" for k in targets):
        targets["buckets"] = list(d["buckets"])

    if modality == "llm":
        wl.setdefault("max_tokens", 256)
        wl.setdefault("prompt_set", "dev")
        if wl["max_tokens"] < 2:
            errors.append("workload.max_tokens must be >= 2 for modality llm")
    elif "max_tokens" in wl:
        warnings.append(f"workload.max_tokens does not apply to modality {modality}; retained for compatibility, "
                        "but unused by its timing client")
    wl.setdefault("rounds", 5)
    client = wl.get("client")
    expected_client = {"llm": "bench_serve", "tts": "bench_audio", "image": "bench_image", "video": "bench_clip"}
    if client is not None and client not in ("external", expected_client.get(modality)):
        errors.append(f"workload.client {client!r} does not support modality {modality}")
    if needs_load:
        wl.setdefault("concurrency", list(DEFAULT_CONCURRENCY))
    if "concurrency" in wl:
        wl["concurrency"] = sorted(set(wl["concurrency"]))

    budget = d.setdefault("budget", {})
    budget.setdefault("approved", False)
    uncapped = budget.get("spend_cap") == "none"
    if uncapped:
        if budget.get("usd") is not None or budget.get("gpu_hours") is not None:
            errors.append("budget.spend_cap: none conflicts with budget.usd or budget.gpu_hours")
        if not budget["approved"]:
            errors.append("budget.spend_cap: none requires budget.approved: true")
        if not budget.get("deadline") and not budget.get("wall_clock_h"):
            errors.append("budget.spend_cap: none requires a finite budget.deadline or budget.wall_clock_h")
    if budget.get("deadline"):
        try:
            deadline = _dt.datetime.fromisoformat(budget["deadline"].replace("Z", "+00:00"))
            if deadline.tzinfo is None:
                raise ValueError("timezone required")
        except ValueError:
            errors.append("budget.deadline must be an ISO8601 time with a timezone")
    price = hw.get("price_usd_h")
    has_cap = "gpu_hours" in budget or ("usd" in budget and price)
    if "usd" in budget and not price:
        (errors if d.get("non_interactive") else warnings).append(
            "budget.usd is set but hardware.price_usd_h is not: dollars cannot be turned into hours")
    if d.get("non_interactive"):
        if not budget["approved"]:
            errors.append("non_interactive: true needs budget.approved: true (an unattended run may only spend an "
                          "approved budget)")
        if not has_cap and not uncapped:
            errors.append("non_interactive: true needs a spend cap: budget.gpu_hours, or budget.usd with "
                          "hardware.price_usd_h")
    elif not budget["approved"]:
        warnings.append("budget.approved is false: every GPU-occupying step asks first")

    q = d.setdefault("quality", {})
    ref = q.get("reference", "none")
    if ref == "none" or (isinstance(ref, dict) and ref.get("kind") == "none"):
        ref = {"kind": "none"}
        warnings.append("quality.reference is none: agreement is measured against the baseline engine at the same "
                        "precision, not against a higher-precision reference")
    q["reference"] = ref
    if goals["lead"] == "affordability":
        if not q.get("protocol"):
            errors.append("goals.lead affordability requires quality.protocol with modality quality gates")
        if not any(TARGETS[k][0] == "lower" and k != "usd_per_1m" for k in targets if k in MODALITY_TARGETS[modality]):
            errors.append("goals.lead affordability requires an explicit latency target in goals.targets")
    q.setdefault("noise_floor", "measure")

    repo = d.get("repo")
    if repo:
        if not repo.get("path"):
            warnings.append("repo.path missing: interactive runs ask which repo to use; non_interactive runs skip "
                            "the ledger (NOT RUN)")
        repo.setdefault("push", "ask")
        repo.setdefault("branch", f"perf/{_slug(model['id'].split('/')[-1])}-{_slug(hw['gpu'])}")
    else:
        d["repo"] = None
        warnings.append("repo missing: kept changes are not committed; the commit ledger is NOT RUN")

    cons = d.get("constraints")
    d["constraints"] = [cons] if isinstance(cons, str) else list(cons or [])
    d.setdefault("non_interactive", False)

    # Run rules: a schema field wins over a constraint string, and the two must agree when both are given. The
    # defaults are written into the intake; rule_sources records where each rule came from (see rule_values).
    rules, errs = constraint_rules(d["constraints"])
    errors += errs
    resolved = rule_values(d)
    preference = q.get("preference")
    if preference is not None:
        expected = "lossless" if preference == "exact" else "lossy"
        if resolved["fidelity"]["value"] != expected:
            errors.append(f"quality.preference {preference!r} requires fidelity {expected!r}; "
                          "resolve the conflicting field or constraint")
        if preference == "exact":
            warnings.append("quality.preference exact requires equality of the declared deterministic/seeded "
                            "outputs; legacy lossless distribution checks alone do not prove identical outputs")
    interaction = d.get("interaction", {})
    if d["non_interactive"] and interaction.get("checkpoints", "required") != "required":
        errors.append("non_interactive: true cannot promise human quality or milestone checkpoints")
    for name, where in (("fidelity", "fidelity"), ("after_target", "goals.after_target")):
        given = resolved[name]["value"]
        if resolved[name]["source"] == where and name in rules and rules[name] != given:
            errors.append(f"{where} is {given!r} but constraints say '{name}: {rules[name]}': keep one of them")
    after = goals["after_target"] = resolved["after_target"]["value"]
    fidelity = d["fidelity"] = resolved["fidelity"]["value"]
    d["rule_sources"] = {name: r["source"] for name, r in resolved.items()}
    if after == "keep_climbing" and hour_cap(d)["host_hours_cap"] is None and not budget.get("deadline"):
        warnings.append("goals.after_target is keep_climbing but no budget cap is set"
                        + (" (budget.usd needs hardware.price_usd_h to become one)" if budget.get("usd") else "")
                        + ": the run climbs until you stop it")
    qbudget = q.get("budget")
    relative = (qbudget or {}).get("relative")
    acceptance = q.get("acceptance")
    if relative is not None:
        if fidelity != "lossy":
            errors.append("quality.budget.relative requires fidelity: lossy")
        if relative["modality"] != modality or modality not in RELATIVE_BUDGET_KEYS:
            errors.append("quality.budget.relative.modality must match the intake modality (image, tts or stt)")
        if not q.get("protocol", "").strip() or not re.fullmatch(r"[0-9a-f]{64}", q.get("protocol_sha256", "")):
            errors.append("quality.budget.relative requires quality.protocol and quality.protocol_sha256 "
                          "(64 lowercase hexadecimal characters)")
        allowed = RELATIVE_BUDGET_KEYS[relative["modality"]]
        wrong = sorted(set(relative) - {"modality", *allowed})
        if wrong:
            errors.append(f"quality.budget.relative {wrong} do not apply to modality {relative['modality']}")
        for key, value in relative.items():
            if key != "modality" and isinstance(value, float) and not math.isfinite(value):
                errors.append(f"quality.budget.relative.{key} must be finite")
    if acceptance == PERCEPTUAL_ACCEPTANCE:
        if modality not in PERCEPTUAL_BUDGET_KEYS or relative is None:
            errors.append("quality.acceptance perceptual_and_task_noninferiority requires a typed relative "
                          "image or tts budget and hash-bound quality.protocol")
        else:
            missing = sorted(set(PERCEPTUAL_BUDGET_KEYS[modality]) - set(relative))
            if missing:
                errors.append(f"perceptual/task acceptance requires relative bounds {missing}")
            if modality == "image" and ({"lpips_excess_max", "ssim_loss_max"} & set(relative)
                                        or {"lpips_max", "ssim_min", "psnr_db_min"} & set(qbudget)):
                errors.append("perceptual/task acceptance treats image numerical similarity as diagnostic only; "
                              "remove LPIPS/SSIM/PSNR acceptance thresholds")
        wrong = sorted(set(q.get("diagnostic_only", [])) - set(DIAGNOSTIC_QUALITY_METRICS.get(modality, ())))
        if wrong:
            errors.append(f"quality.diagnostic_only {wrong} do not apply to modality {modality}")
    elif "diagnostic_only" in q:
        errors.append("quality.diagnostic_only requires explicit quality.acceptance "
                      "perceptual_and_task_noninferiority; it cannot disable existing quality gates")
    if fidelity == "lossless":
        prec, allowed = model.get("precision"), model["precision_allowed"]
        if prec and allowed and allowed != [prec]:
            warnings.append(f"fidelity is lossless but model.precision_allowed {allowed} is not [{prec!r}]: rows at "
                            "another precision change numerics; drop them, or set fidelity: lossy with a "
                            "quality.budget")
        lossy_kv = [k for k in model["kv_cache_dtype_allowed"] if _LOSSY_KV.match(k)]
        if lossy_kv:
            warnings.append(f"fidelity is lossless but model.kv_cache_dtype_allowed has {lossy_kv}: a low-precision "
                            "KV cache changes numerics unless the checkpoint itself specifies it (then write "
                            "checkpoint_default); otherwise set fidelity: lossy with a quality.budget")
        if qbudget:
            warnings.append("quality.budget is set but fidelity is lossless: the budget is ignored; set "
                            "fidelity: lossy if speed-ups may change numerics")
    elif not qbudget:
        msg = ("fidelity is lossy but quality.budget is missing: declare its bounds "
               f"({', '.join(budget_keys(modality))}) before any measurement")
        if d["non_interactive"]:
            errors.append(msg)
        else:
            warnings.append(msg + "; the agent agrees them with you before ranking a lossy change")
    elif _unbounded(qbudget, modality, acceptance):
        warnings.append(f"fidelity is lossy but quality.budget does not bound {_unbounded(qbudget, modality, acceptance)}: those "
                        "metrics have no limit; agree them (add them to quality.budget) before ranking a lossy change")
    other = sorted(set(qbudget or {}) - set(budget_keys(modality)) - {"relative"})
    if fidelity == "lossy" and other:
        warnings.append(f"quality.budget {other} do not apply to modality {modality} (use "
                        f"{list(budget_keys(modality))}); they bound nothing")
    d.setdefault("harness", {})
    if access.get("kind") == "ssh" and not d["harness"].get("wheel") and not d["harness"].get("requirements"):
        warnings.append("harness: no wheel or requirements given for the ssh host: install the same inferencekit "
                        "version there as here, and record its sha256")
    cands = d.setdefault("candidates", {})
    cands.setdefault("include", [])
    cands.setdefault("exclude", [])
    cands.setdefault("max_engines", 5)
    if modality in ("tts", "stt") and client is None:
        hint = "configure workload.client: bench_audio for speech synthesis, or external" if modality == "tts" else "configure workload.client: external for speech recognition"
        warnings.append(f"modality {modality}: no audio timing client is configured; {hint}. "
                        "Audio latency claims without a matching client are UNPROVEN")
    if modality == "tts" and client == "bench_audio":
        warnings.append("bench audio supplies client-timed speech screening; quality and fresh-process "
                        "interleaved comparison remain UNPROVEN without their separate evidence")
    if modality == "image":
        warnings.append("modality image: bench serve token metrics and bench clip video metrics do not prove image "
                        "latency or quality; a client-timed image adapter and modality quality protocol are required; "
                        "unsupported checks are NOT RUN, and claims without them are UNPROVEN")
    if modality == "video":
        warnings.append("modality video: time clips with bench clip (clip HTTP contract or ComfyUI) and check them "
                        "with gates clip and quality clip; bench serve's token metrics and buckets do not apply")
    d["intake_version"] = 1
    if not d.get("run_id"):
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M")
        d["run_id"] = f"{_slug(model['id'].split('/')[-1])}-{_slug(hw['gpu'])}-{stamp}"
    return d, errors, warnings


# --------------------------------------------------------------------------- the derived plan

def hour_cap(intake: dict) -> dict:
    """The binding cap on host hours: the smallest of gpu_hours/count, usd/price, wall_clock_h."""
    b, hw = intake["budget"], intake["hardware"]
    caps = {}
    if b.get("gpu_hours"):
        caps["gpu_hours"] = b["gpu_hours"] / hw["count"]
    if b.get("usd") and hw.get("price_usd_h"):
        caps["usd"] = b["usd"] / hw["price_usd_h"]
    if b.get("wall_clock_h"):
        caps["wall_clock_h"] = b["wall_clock_h"]
    if not caps:
        return {"host_hours_cap": None, "binding": None, "caps_h": {}, "usd_cap": b.get("usd"),
                "formula": "no cap given", "spend_cap": b.get("spend_cap", "bounded"),
                "deadline": b.get("deadline")}
    binding = min(caps, key=caps.get)
    usd_cap = b.get("usd")
    if usd_cap is None and b.get("spend_cap") != "none" and hw.get("price_usd_h"):
        usd_cap = round(caps[binding] * hw["price_usd_h"], 2)
    return {"host_hours_cap": round(caps[binding], 3), "binding": binding,
            "caps_h": {k: round(v, 3) for k, v in caps.items()}, "usd_cap": usd_cap,
            "formula": "min(gpu_hours / count, usd / price_usd_h, wall_clock_h)",
            "spend_cap": b.get("spend_cap", "bounded"), "deadline": b.get("deadline")}


def rule_values(intake: dict) -> dict:
    """{name: {"value", "source"}} for every run rule in RULES: its schema field when written, else a constraint
    string, else the default. ``source`` is the field name, ``constraints`` or ``default: ...``.

    normalize writes the defaults into the intake and records each source in ``rule_sources``. A field value that
    ``rule_sources`` says came from a constraint or a default, and still equals what that would give, keeps that
    source, so checking a normalized intake again gives the same plan.
    """
    found, _ = constraint_rules(intake.get("constraints"))
    goals = intake.get("goals") if isinstance(intake.get("goals"), dict) else {}
    budget = intake.get("budget") if isinstance(intake.get("budget"), dict) else {}
    prior = intake.get("rule_sources") if isinstance(intake.get("rule_sources"), dict) else {}
    lead = goals.get("lead", "single_user")
    if intake.get("non_interactive"):
        after_default = ("keep_climbing", "default: non_interactive")
    elif budget.get("deadline"):
        after_default = ("keep_climbing", "default: budget.deadline is set")
    elif any(budget.get(k) for k in CAP_KEYS):
        after_default = ("keep_climbing", "default: budget.gpu_hours, usd or wall_clock_h is set")
    else:
        after_default = ("stop", "default: interactive, no budget cap")
    preference = (intake.get("quality") or {}).get("preference")
    fidelity_default = (("lossless" if preference == "exact" else "lossy", "quality.preference")
                        if preference is not None else ("lossless", "default"))
    rules = {"after_target": ("goals.after_target", goals.get("after_target"), after_default),
             "fidelity": ("fidelity", intake.get("fidelity"), fidelity_default),
             "speculation": (None, None, ("auto", "default")),
             "route": (None, None, ("build" if lead == "single_user" else "patch", f"default: goals.lead {lead}"))}
    out = {}
    for name, (field, written, (default, default_source)) in rules.items():
        fallback = ({"value": found[name], "source": "constraints"} if name in found
                    else {"value": default, "source": default_source})
        filled_in = prior.get(name, field) != field and written == fallback["value"]
        out[name] = {"value": written, "source": field} if written is not None and not filled_in else fallback
    return out


def _stop_rule(named: bool, after: str, ni: bool, capped: bool, explicit: bool, set_keys: list[str],
               where: str = "in every listed bucket") -> dict:
    """``capped``: the budget gives an hour cap (hour_cap), so a climbing run has an end. ``explicit``: after_target
    was written (field or constraint), not defaulted. ``set_keys``: the budget keys that made keep_climbing the
    default. ``where``: where a target must hold (video: over the whole clip set). Without a target, an
    interactive run still asks at the stop point unless the person wrote keep_climbing."""
    until = "until the budget ends" if capped else "until you stop it (no budget cap is set)"
    if named and after == "keep_climbing":
        return {"kind": "target", "text": f"keep climbing {until}; record the best result over time (screened or verified). The "
                f"targets are milestones: record when each is first met by its conservative CI bound {where}, "
                "then continue (the report reserve is kept)"}
    if named:
        return {"kind": "target", "text": f"stop when every target is met by its conservative CI bound {where}; "
                "when one is missed, continue to stage 7 within the approved budget"}
    if after == "keep_climbing" and (ni or explicit):
        return {"kind": "budget", "text": f"no target: keep climbing {until}, stage 7 included, keeping the report "
                "reserve; record the best result over time (screened or verified)"}
    if ni:
        return {"kind": "stop_point", "text": "no target and after_target stop: stop after the stage 6 ranking and "
                "write the report (stage 7 NOT RUN)"}
    text = "no target: show the ranked table at stage 6 and ask whether to stop or continue"
    if after == "keep_climbing":
        text += (f"; the default to offer is keep climbing {until}, recording the best result over time (screened or verified) "
                 f"(after_target defaults to keep_climbing because the intake sets "
                 f"{', '.join('budget.' + k for k in set_keys)})")
    return {"kind": "ask", "text": text}


def budget_keys(modality: str) -> tuple[str, ...]:
    """The quality-budget bounds applicable to a modality."""
    if modality in ("image", "video"):
        return VIDEO_BUDGET_KEYS
    if modality in ("tts", "stt"):
        return SPEECH_BUDGET_KEYS
    return QUALITY_BUDGET_KEYS


def _unbounded(qbudget, modality: str = "llm", acceptance: str | None = None) -> list[str]:
    """The quality-budget bounds a lossy run leaves out."""
    if (qbudget or {}).get("relative"):
        expected = PERCEPTUAL_BUDGET_KEYS if acceptance == PERCEPTUAL_ACCEPTANCE else RELATIVE_BUDGET_KEYS
        return ["relative." + k for k in expected.get(modality, ())
                if k not in qbudget["relative"]]
    return [k for k in budget_keys(modality) if k not in (qbudget or {})]


def _fidelity_text(fidelity: str, qbudget, modality: str = "llm") -> str:
    if modality in ("image", "tts", "stt"):
        return (f"{fidelity}: validate the exact speed configuration using quality.protocol and the declared "
                "modality quality budget against the reference and A/A noise floor; no LLM token/GSM8K proxy. "
                "Numerics changes are declared; missing modality evidence is UNPROVEN")
    if fidelity == "lossless" and modality == "video":
        return ("lossless: numerics stay as shipped (the checkpoint's own precision, the same steps, shape, guidance "
                "and initial noise). Only changes that leave the clips unchanged count: refactors, equivalent "
                "kernels, compilation, overlapped stages. Proof: quality clip (PSNR, SSIM, LPIPS per frame against "
                "the BF16 reference, within the run-to-run and alternative-backend floors); without it a change is "
                "UNPROVEN")
    if fidelity == "lossless":
        return ("lossless: numerics stay as shipped (weights and activations in the checkpoint's own precision; no "
                "KV cache below 16 bits unless the checkpoint ships one). Only changes that leave the output "
                "distribution unchanged count: refactors, equivalent kernels, re-encodings that decode to identical "
                "values, exact speculative decoding. Proof: next-token agreement and KL against the reference within "
                "the noise floor; without it a change is UNPROVEN")
    text = ("lossy: changes that alter numerics are allowed when every bound in quality_budget holds against the "
            "reference; they are a separate track, reported as speed at a stated quality cost and never ranked "
            "against lossless results")
    if not qbudget:
        return text + ". quality_budget is not set yet: agree it before ranking a lossy change"
    missing = _unbounded(qbudget, modality)
    return text + (f". Not bounded: {missing}; agree them before ranking a lossy change" if missing else "")


def _precision_rule(model: dict, fidelity: str, modality: str = "llm") -> str:
    """Which numerics a ranked engine may use. Under lossless only the as-shipped ones: model.precision (when given)
    and KV dtypes of 16 bits or more; the other listed entries are named as levers that are not run or ranked.
    A video model has no KV cache: only the precision part applies."""
    allowed, kv = model["precision_allowed"], model["kv_cache_dtype_allowed"]
    if modality in ("image", "video"):
        kv = []
    prec, kv_ranked = allowed, kv
    if fidelity == "lossless":
        prec = [model["precision"]] if model.get("precision") else allowed
        kv_ranked = [k for k in kv if not _LOSSY_KV.match(k)] or ([] if modality in ("image", "video") else ["checkpoint_default"])
    text = (f"every ranked engine runs weights/activations in one of {prec}"
            + (f" and KV cache in one of {kv_ranked}" if modality not in ("image", "video") else "")
            + "; different precisions are separate tracks")
    extra = [f"{what} {left}" for what, left in (("precision", [p for p in allowed if p not in prec]),
                                                  ("KV cache", [k for k in kv if k not in kv_ranked])) if left]
    if extra:
        text += (f". Listed but outside fidelity lossless (they change numerics): {', '.join(extra)}; they are "
                 "reported as levers, not run or ranked, unless fidelity is lossy with a quality.budget")
    return text


def _speculation_text(value: str, fidelity: str, modality: str = "llm") -> str:
    if modality in ("image", "video"):
        return f"does not apply to modality {modality} (no token decoding)"
    if value == "off":
        return "off: no speculative decoding in any ranked engine (no draft model, MTP head or n-gram lookup)"
    return ("auto: speculative decoding is an ordinary lever where it pays (single user, low concurrency), kept only "
            "if faster on this workload; " + ("only exact speculation counts: greedy output identical and the "
                                               "sampling distribution unchanged" if fidelity == "lossless" else
                                               "inexact acceptance counts only within quality_budget"))


ROUTE_TEXT = {"build": "build: write a single-purpose engine for this model and workload; the best tuned existing "
                       "engine stays the baseline",
              "patch": "patch: tune and patch the best existing engine"}


def derive(intake: dict) -> dict:
    """The derived plan (claim class ``derived``) of a normalized intake. Where each run rule came from is read
    from the intake's ``rule_sources`` (see rule_values), so deriving twice gives the same plan."""
    goals, targets = intake["goals"], intake["goals"]["targets"]
    named = [k for k in targets if k != "buckets"]
    video = intake["modality"] in ("image", "video")
    target_rows = [{"name": k, "value": targets[k], "better": TARGETS[k][0], "unit": TARGETS[k][1],
                    "buckets": ["all"] if video else targets.get("buckets", intake["buckets"]),
                    "applies": k in MODALITY_TARGETS[intake["modality"]]} for k in named]
    ni = intake["non_interactive"]
    rules = rule_values(intake)
    after = rules["after_target"]["value"]
    after_source = rules["after_target"]["source"]
    cap = hour_cap(intake)
    hours = cap["host_hours_cap"]
    set_keys = [k for k in CAP_KEYS if intake["budget"].get(k)]
    target_scope = ("over the whole clip set" if intake["modality"] == "video" else "over the whole image set") if video else "in every listed bucket"
    stop = {**_stop_rule(bool(named), after, ni, hours is not None or bool(intake["budget"].get("deadline")),
                         not after_source.startswith("default"), set_keys, target_scope),
            "after_target": after, "after_target_source": after_source}
    if intake["budget"].get("report_reserve_h") == 0:
        stop["text"] = stop["text"].replace("the report reserve is kept", "no early reserve; validate/report continuously")
        stop["text"] = stop["text"].replace("keeping the report reserve", "validating/reporting continuously")
    interaction = intake.get("interaction") or {}
    if interaction.get("checkpoints") == "required" and not named and stop["kind"] == "ask":
        if after == "keep_climbing" and (hours is not None or intake["budget"].get("deadline")):
            stop.update(kind="budget", text="no target: continue within the authorized budget; pause only "
                        "for a new required decision")
        else:
            stop.update(kind="stop_point", text="no target: finish the ranking and report; further scope "
                        "needs a new decision")
    fidelity = rules["fidelity"]["value"]
    qbudget = (intake.get("quality") or {}).get("budget") if fidelity == "lossy" else None
    boxes = {k: (round(hours * share, 3) if hours and intake["budget"].get("stage_timeboxes", True) else None)
             for k, share in STAGES}
    modes = ["single"]
    if goals["lead"] == "capacity" or "n_star" in targets:
        modes.append("load")
    if intake["modality"] == "image":
        modes = ["image"]
    if intake["modality"] == "tts" and intake["workload"].get("client") == "bench_audio":
        modes = ["audio"]
    if intake["modality"] == "video":
        modes = ["clip"]  # bench rounds --mode clip: one clip at a time
    questions = []
    if ni:
        if not named:
            questions.append({"question": "No target was given. Stop after stage 6 or continue?",
                              "choice": ("keep climbing while budget remains (report reserve kept)"
                                         if after == "keep_climbing" else "stop after stage 6 (after_target: stop)")})
        elif rules["after_target"]["source"].startswith("default"):
            questions.append({"question": "Once every target is met: stop, or keep climbing?",
                              "choice": f"{after} ({rules['after_target']['source']})"})
        if rules["fidelity"]["source"] == "default":
            questions.append({"question": "May speed-ups change numerics (lossy), and inside which quality budget?",
                              "choice": "no: fidelity lossless (the default), numerics stay as shipped"})
        if rules["route"]["source"].startswith("default"):
            questions.append({"question": "Build a new engine, or patch the best existing one?",
                              "choice": f"{rules['route']['value']} ({rules['route']['source']})"})
        if intake["repo"] and intake["repo"]["push"] == "ask":
            questions.append({"question": "Push the perf branch?", "choice": "no push (non_interactive never "
                              "pushes)"})
        if intake["quality"]["reference"]["kind"] == "none":
            questions.append({"question": "Which reference should outputs be compared with?",
                              "choice": "the baseline engine at the checkpoint's precision"})
        if not intake["model"].get("revision"):
            questions.append({"question": "Which model revision?", "choice": "the commit sha current at stage 1"})
    return {
        "claim_class": "derived",
        "run_id": intake["run_id"],
        "modes": modes,
        "stop_rule": stop,
        "targets": target_rows,
        "budget": cap,
        "time_boxes_h": boxes,
        "report_reserve_h": intake["budget"].get("report_reserve_h",
                            round(hours * REPORT_RESERVE, 3) if hours else None),
        "stage_timeboxes": intake["budget"].get("stage_timeboxes", True),
        "affordability": goals.get("affordability"),
        "precision_rule": _precision_rule(intake["model"], fidelity, intake["modality"]),
        "fidelity_rule": {"value": fidelity, "fidelity": fidelity, "source": rules["fidelity"]["source"],
                          "text": ("exact: equality of the declared deterministic/seeded outputs is required; "
                                   "legacy lossless distribution checks alone are insufficient"
                                   if intake["quality"].get("preference") == "exact"
                                   else _fidelity_text(fidelity, qbudget, intake["modality"]))},
        "quality_budget": qbudget or None,
        "quality_preference": intake["quality"].get("preference"),
        "architecture_changes": intake["model"].get("architecture_changes", "forbid"),
        "interaction": {"checkpoints": interaction.get("checkpoints", "required"),
                        "show_plan": interaction.get("show_plan", False)},
        **({"external_quality_proof": {
            "required": True, "quality_verdict": "UNPROVEN",
            "protocol": intake["quality"]["protocol"],
            "protocol_sha256": intake["quality"]["protocol_sha256"],
            "budget_basis": "relative degradation; positive increase/drop is worse",
            "text": "External protocol quality proof required on the exact speed configuration, including "
                    "its reference/A/A, cohort and CI rules. Intake validation is not a quality PASS; "
                    "no absolute scorer or bitwise output test evaluates this relative budget.",
            **({"acceptance": PERCEPTUAL_ACCEPTANCE,
                "diagnostic_only": intake["quality"].get("diagnostic_only", []),
                "blinded_perceptual_review_required": True,
                "perceptual_verdict": "UNPROVEN",
                "text": "External blinded perceptual and task-quality proof required on the exact speed "
                        "configuration under the hash-bound protocol. Numerical similarity is diagnostic, "
                        "not an independent rejection gate. Proxy metrics alone do not establish perceptual "
                        "quality; missing viewing/listening stays UNPROVEN. Intake validation is not quality PASS."}
               if intake["quality"].get("acceptance") == PERCEPTUAL_ACCEPTANCE else {}),
        }} if qbudget and "relative" in qbudget else {}),
        "speculation": {**rules["speculation"], "text": _speculation_text(rules["speculation"]["value"], fidelity,
                                                                          intake["modality"])},
        "route": {**rules["route"], "text": ROUTE_TEXT[rules["route"]["value"]]},
        "questions": questions,
        "chat_template_kwargs": (intake.get("workload") or {}).get("chat_template_kwargs") or None,
    }


def _drop_nulls(node, path=()):
    """Legacy empty YAML blocks are absent; explicit typed quality declarations must validate."""
    if isinstance(node, dict):
        return {k: _drop_nulls(v, path + (k,)) for k, v in node.items()
                if v is not None or path == ("quality", "budget", "relative")
                or (path == ("quality", "budget") and k == "relative")
                or (path == ("quality",) and k in ("acceptance", "diagnostic_only"))}
    if isinstance(node, list):
        return [_drop_nulls(v, path + (i,)) for i, v in enumerate(node)]
    return node


def check(data: dict, allow_placeholders: bool = False) -> dict:
    """Validate a parsed intake. Raises OpError listing every error; returns the normalized result."""
    data = _drop_nulls(data)
    errors, warnings = schema_problems(data)
    if errors:
        raise OpError("intake is invalid:\n  - " + "\n  - ".join(errors + [f"(warning) {w}" for w in warnings]))
    data = _drop_unknown(data, load_schema())
    holes = placeholders(data)
    intake, errs, warns = normalize(data)
    errors += errs
    warnings += warns
    if holes:
        msg = f"placeholder values still present at {holes}: replace them before a run"
        (warnings if allow_placeholders else errors).append(msg)
    if errors:
        raise OpError("intake is invalid:\n  - " + "\n  - ".join(errors + [f"(warning) {w}" for w in warnings]))
    return {"ok": True, "intake": intake, "warnings": warnings, "placeholders": holes, "derived": derive(intake)}


# --------------------------------------------------------------------------- run directory

def initial_state(result: dict, intake_sha: str) -> dict:
    now = utc_now()
    derived = result["derived"]
    stages = {k: {"status": "pending", "started": None, "ended": None, "time_box_h": derived["time_boxes_h"][k],
                  "evidence": [], "notes": []} for k, _ in STAGES}
    stages["1_intake"].update(status="in_progress", started=now)
    state = {
        "state_version": 1,
        "run_id": derived["run_id"],
        "intake_sha256": intake_sha,
        "created": now,
        "updated": now,
        "current_stage": "1_intake",
        "stages": stages,
        "budget": {"host_hours_cap": derived["budget"]["host_hours_cap"], "usd_cap": derived["budget"]["usd_cap"],
                   "price_usd_h": result["intake"]["hardware"].get("price_usd_h"), "run_started": now,
                   "meter": [], "spend_cap": derived["budget"].get("spend_cap", "bounded"),
                   "deadline": derived["budget"].get("deadline"),
                   "stage_timeboxes": derived["stage_timeboxes"],
                   "report_reserve_h": derived["report_reserve_h"]},
        "model": {"id": result["intake"]["model"]["id"], "revision": result["intake"]["model"].get("revision"),
                  "revision_resolved": None},
        "candidates": {},
        "best": None,
        "decisions": [],
        "next_action": "stage 1: export INFERENCEKIT_RUNS_DIR=<abs path>/runs/<run-id>/evidence (here and on the GPU "
                       "host), resolve the model revision to a commit sha, write QUESTIONS.md, then stage 2",
    }
    pin = (result["intake"].get("workload") or {}).get("chat_template_kwargs")
    if pin:
        text = json.dumps(pin, sort_keys=True, separators=(",", ":"))
        state["workload_pin"] = {"chat_template_kwargs": pin}
        state["next_action"] += (f"; the intake pins chat_template_kwargs {text}: put it in manifest.json "
                                 "workload.chat_template_kwargs and pass --chat-template-kwargs '" + text + "' to "
                                 "bench serve, gates run and quality (with --run-dir runs/<run-id> they apply it and "
                                 "refuse a different value)")
    return state


# --------------------------------------------------------------------------- the workload pin

PIN_FILE = "intake.normalized.json"


def intake_pin(run_dir: str | os.PathLike | None) -> dict:
    """The intake's ``workload.chat_template_kwargs`` recorded in ``<run_dir>/intake.normalized.json``
    ({} when there is no run dir, no normalized intake or no pin). The pin is binding: every command that
    generates with that run dir sends it (see apply_pin), and ``manifest validate`` / ``bench rounds`` refuse
    a manifest that contradicts it."""
    if not run_dir:
        return {}
    path = Path(run_dir) / PIN_FILE
    try:
        norm = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    pin = (((norm.get("intake") or {}).get("workload") or {}).get("chat_template_kwargs")
           if isinstance(norm, dict) else None)
    return dict(pin) if isinstance(pin, dict) else {}


def pin_conflicts(value: dict | None, pin: dict) -> list[str]:
    """Pinned keys that ``value`` sets to something else (a missing key is not a conflict)."""
    value = value or {}
    return [f"{k}: {json.dumps(value[k])} (pinned {json.dumps(v)})" for k, v in sorted(pin.items())
            if k in value and value[k] != v]


def apply_pin(value: dict | None, run_dir, where: str) -> dict:
    """``value`` with the run dir's intake pin applied: pinned keys are added when missing; a different value
    for a pinned key is refused (OpError naming ``where``)."""
    pin = intake_pin(run_dir)
    if not pin:
        return dict(value or {})
    bad = pin_conflicts(value, pin)
    if bad:
        raise OpError(f"{where}: chat_template_kwargs contradicts the intake's pinned "
                      f"workload.chat_template_kwargs in {Path(run_dir) / PIN_FILE}: {'; '.join(bad)}. The intake "
                      "is pre-registered: every engine runs with the pinned template settings")
    return {**(value or {}), **pin}


def _home() -> str | None:
    home = os.path.expanduser("~")
    return home if home not in ("", "~", os.sep) else None


def display_path(path: str | os.PathLike) -> str:
    """A path for artifacts: relative to the working directory when under it, else with the home directory as ~."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(Path.cwd().resolve()))
    except ValueError:
        pass
    home = _home()
    s = str(p)
    if home and (s == home or s.startswith(home.rstrip(os.sep) + os.sep)):
        return "~" + s[len(home.rstrip(os.sep)):]
    return s


def redact(intake: dict, *, home: str | None = None) -> dict:
    """A copy of an intake that is safe to keep in the run directory (which is synced, reviewed and published).

    * every ``hardware.access`` field except ``kind`` becomes ``<redacted>`` (host, user, port, key_path);
    * the host anywhere else (e.g. inside ``quality.reference.base_url``) and ``<user>@`` become ``<redacted>``;
    * the home directory at the start of a path becomes ``~`` (it names the local user).

    ``home`` defaults to this account's home; an empty string disables home substitution.
    Resume validation can supply the original prefix when checking a copied run.
    """
    d = json.loads(json.dumps(intake, default=str))
    access = (d.get("hardware") or {}).get("access") if isinstance(d.get("hardware"), dict) else None
    host = user = None
    if isinstance(access, dict):
        host = access.get("host") if isinstance(access.get("host"), str) and access.get("host") != REDACTED else None
        user = access.get("user") if isinstance(access.get("user"), str) and access.get("user") != REDACTED else None
    host_re = re.compile(r"(?<![\w.-])" + re.escape(host) + r"(?![\w-])") if host else None
    user_re = re.compile(r"(?<![\w.-])" + re.escape(user) + r"@") if user else None
    home = _home() if home is None else home
    home_prefix = home.rstrip(os.sep) + os.sep if home else None

    def walk(node):
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, str):
            if user_re:
                node = user_re.sub(REDACTED + "@", node)
            if host_re:
                node = host_re.sub(REDACTED, node)
            if home_prefix and node.startswith(home_prefix):
                node = "~" + os.sep + node[len(home_prefix):]
        return node

    d = walk(d)
    if isinstance(access, dict):
        d["hardware"]["access"] = {k: (v if k == "kind" else REDACTED) for k, v in access.items()}
    return d


def _copy_text(raw: bytes, as_json: bool, sha: str) -> str:
    """The redacted copy of the intake file as written into the run directory."""
    original = redact(parse_text(raw.decode("utf-8")))
    if as_json:
        return json.dumps(original, indent=2, ensure_ascii=False) + "\n"
    import yaml  # parse_text already needed it for a YAML intake

    head = ("# Redacted copy of this run's intake: hardware.access details are removed and the home directory\n"
            f"# is shown as ~. intake_sha256 of the original file: {sha}\n"
            "# To resume, validate the ORIGINAL intake file (state.json: intake_source), not this copy.\n")
    return head + yaml.safe_dump(original, sort_keys=False, allow_unicode=True, default_flow_style=False)


def _recorded_shas(run_dir: Path) -> dict:
    """intake_sha256 values already recorded in the run directory, by file. Raises on unreadable JSON."""
    out = {}
    for name in ("state.json", "intake.normalized.json"):
        p = run_dir / name
        if p.exists():
            try:
                out[name] = json.loads(p.read_text()).get("intake_sha256")
            except (ValueError, AttributeError):
                raise OpError(f"{p} is not valid JSON; fix or move it before resuming") from None
    return out


def start_run(run_dir: Path, src: Path, raw: bytes, result: dict) -> dict:
    """Create or resume runs/<run-id>/.

    The intake is pre-registered: if the run directory already records a different intake_sha256, this raises and
    writes nothing (a changed intake is a new run id). Existing files are never overwritten: the intake copy,
    intake.normalized.json, state.json, experiments.jsonl and QUESTIONS.md are only created when missing.
    Access details (host, user, port, key path) never reach the run directory: the copies are redacted.
    """
    sha = sha256_hex(raw)
    recorded = _recorded_shas(run_dir)
    changed = sorted(name for name, old in recorded.items() if old != sha)
    if changed:
        raise OpError(f"the intake changed since this run started: {', '.join(changed)} in {run_dir} record(s) "
                      f"intake_sha256 {recorded[changed[0]]!r}, this file is {sha!r}. Targets are pre-registered, "
                      "so a changed intake is a new run: start a new run id (--run-dir runs/<new-run-id>). "
                      "Nothing was written.")
    resumed = "state.json" in recorded
    run_dir.mkdir(parents=True, exist_ok=True)
    written = []
    as_json = src.suffix.lower() == ".json"
    copy_name = "intake.json" if as_json else "intake.yaml"
    dst = run_dir / copy_name
    if not dst.exists():
        dst.write_text(_copy_text(raw, as_json, sha))
        written.append(copy_name)
    source = display_path(src)
    norm_path = run_dir / "intake.normalized.json"
    if not norm_path.exists():
        norm = {"intake_sha256": sha, "intake_source": source, "validated": utc_now(),
                **{k: v for k, v in result.items() if k not in ("intake_sha256", "intake")},
                "intake": redact(result["intake"])}
        norm_path.write_text(json.dumps(norm, indent=2, ensure_ascii=False) + "\n")
        written.append("intake.normalized.json")
    if not resumed:
        state = initial_state(result, sha)
        state["intake_source"] = source
        (run_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n")
        written.append("state.json")
    exp = run_dir / "experiments.jsonl"
    if not exp.exists():
        exp.touch()
        written.append("experiments.jsonl")
    qs = result["derived"]["questions"]
    qpath = run_dir / "QUESTIONS.md"
    if qs and not qpath.exists():
        lines = ["# Open questions (non_interactive run: the conservative choice was taken)", ""]
        for i, q in enumerate(qs, 1):
            lines += [f"{i}. **{q['question']}**", f"   - Chosen: {q['choice']}", "   - Stage: 1_intake", ""]
        qpath.write_text("\n".join(lines))
        written.append("QUESTIONS.md")
    return {"run_dir": str(run_dir), "resumed": resumed, "written": written, "intake_sha256": sha,
            "intake_source": source, "evidence_root": str(run_dir / "evidence")}


# --------------------------------------------------------------------------- operations

TEMPLATE = """\
# inferencekit intake v1. Validate with: inferencekit intake validate intake.yaml
# Values in <angle brackets> (or __LIKE_THIS__) are placeholders; validate refuses them unless --allow-placeholders.
intake_version: 1
# run_id: my-model-5090-20261013     # default: <model>-<gpu>-<UTC date>; becomes runs/<run_id>/

model:
  id: <org/model-name>                # Hugging Face id, or a local path
  revision: "<40-hex commit sha>"     # pin it (quoted); a branch name is resolved to a sha at stage 1
  precision: <nvfp4|fp8|bf16|q8_0>    # what the checkpoint is stored in
  # precision_allowed: [nvfp4]        # default [precision]; engines may only run these (equal precision)
  kv_cache_dtype_allowed: [checkpoint_default]   # adding fp8 (when the checkpoint does not ship it) is lossy
  # tokenizer: <org/model-name>       # default: model.id

modality: llm                         # llm | image | tts | stt | video

hardware:
  gpu: RTX 5090
  count: 1
  price_usd_h: 0.60                   # hourly price of the whole host (for $ caps and cost per 1M)
  access:
    kind: ssh                         # ssh: inferencekit and every engine run ON the GPU host
    host: <gpu-host>
    port: 22
    user: <user>
    key_path: ~/.ssh/<key>
  # access: {kind: local}             # the GPU is on this machine

goals:
  lead: single_user                   # single_user | capacity (N* = streams per GPU inside the SLO)
  targets:                            # optional; omit the block for "as fast as possible"
    decode_tok_s: 150                 # absolute, measured by bench serve (client-side timer)
    # ttft_ms: 120
    # decode_ceiling_fraction: 0.70   # relative to the measured-bandwidth roofline ceiling
    # n_star: 32                      # capacity target; needs slo.ttft_p95_ms
    # usd_per_1m: 0.50
    # clip_ms: 60000                  # modality video: client-timed ms per clip (or clips_per_hour)
    # buckets: [short, medium]        # where the target must hold (default: every bucket)
  # after_target: stop                # once every target is met: stop | keep_climbing (keep improving until the
  #                                   # budget ends). Default keep_climbing when non_interactive or a cap is set
  slo:                                # required for capacity / n_star
    ttft_p95_ms: 500
    tpot_p95_ms: 50
    max_fail: 0
  # traffic: {arrival_rate_rps: 2.0, burstiness: poisson, session_len: "1 turn"}

io_modes:
  input: complete                     # complete | streaming (prompt arrives incrementally)
  output: streaming                   # complete | streaming
  # input_rate: "30 tok/s"            # streaming input pace
  # latency_focus: "time to first token"

buckets: [very_short, short, medium, long]   # 1-5 words / ~20-50 tok / ~300 tok / 2-4k tok

workload:
  max_tokens: 256                     # LLM only: generated per request, identical for every engine (ignore_eos)
  rounds: 5                           # interleaved rounds across engines (>= 3)
  # concurrency: [1, 2, 4, 8, 16, 32, 64]   # load-mode grid (capacity)
  # chat_template_kwargs: {enable_thinking: false}   # binding pin: the manifest and every command must use it
  # client: bench_audio               # TTS: omit max_tokens; bench_audio | external
  # tts: {voice: default, sample_rate: 24000, lead_metric: first_audio_ms}
  # TTS text_length_buckets: [{name: short, min_chars: 1, max_chars: 80}] inside tts
  # io_modes.output selects streaming or complete utterance

budget:
  gpu_hours: 6                        # GPU-hours = host hours x count
  usd: 10
  wall_clock_h: 6
  approved: true                      # required for non_interactive runs

fidelity: lossless                    # lossless: numerics as shipped | lossy: numeric changes inside quality.budget

quality:
  reference: none                     # none | {kind: endpoint, base_url: http://<host>:<port>/v1}
  noise_floor: measure                # measure (A/A run) | a fixed agreement floor in [0, 1]
  # budget:                           # required when fidelity is lossy; every bound must hold vs the reference
  #   gsm8k_drop_pts_max: 1.0         # GSM8K accuracy drop, percentage points
  #   top1_agreement_min: 0.95        # next-token agreement, fraction
  #   kl_max: 0.01                    # mean KL per token, nats
  #   nll_increase_max: 0.02          # reference NLL increase per token, nats
  #   tv_max: 0.05                    # mean total variation of the next-token distribution
  #   (modality video: psnr_db_min, ssim_min, lpips_max instead, judged by quality clip)

repo:                                 # the commit ledger; omit to skip it
  path: <path to a git repo on the machine running the agent>
  # branch: perf/<model>-<gpu>        # default
  push: ask                           # ask | never (non_interactive never pushes)

constraints:                          # free text; honoured literally and quoted in the report
  - engines must be open source
  # - "speculation: off"              # read by validate (quote it): speculation: off | auto (default auto),
  # - "route: build"                  # route: build | patch (default: build for single_user, patch for capacity)

non_interactive: false                # true: never block; QUESTIONS.md + the conservative choice

# harness:                            # how to install inferencekit on an ssh GPU host
#   wheel: <path to inferencekit_cli-<version>-py3-none-any.whl>
#   wheel_sha256: <sha256>
# candidates:
#   include: []                       # engines to add (e.g. a custom engine with an OpenAI-compatible endpoint)
#   exclude: []
#   max_engines: 5
"""


@operation("intake_template", ("intake", "template"), "Print an annotated intake.yaml to start an "
           "optimizing-inference run from.",
           [Arg("out", "string", "Also write it to this file (refuses to overwrite)."),
            Arg("force", "boolean", "Overwrite --out if it exists.", default=False)])
def intake_template(call, out, force):
    path = "stdout"
    if out:
        p = Path(out)
        if p.exists() and not force:
            raise OpError(f"{p} exists; pass --force to overwrite")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(TEMPLATE)
        path = str(p)
    return {"format": "md", "markdown": TEMPLATE.rstrip("\n"), "path": path}


@operation("intake_validate", ("intake", "validate"), "Validate an intake.yaml: schema + cross-field rules, "
           "defaults filled, warnings, and the derived plan (stop rule, fidelity rule and quality budget, "
           "speculation, route, budget cap, stage time boxes). With "
           "--run-dir, also start or resume the run directory (state.json, QUESTIONS.md).",
           [Arg("file", "string", "Path to intake.yaml (or .json).", required=True, positional=True),
            Arg("run_dir", "string", "Start or resume this run directory (e.g. runs/<run-id>): writes a "
                "redacted copy of the intake (no host/user/port/key) and intake.normalized.json, and creates "
                "state.json / experiments.jsonl / QUESTIONS.md; never overwrites them. Refuses (writes nothing) "
                "if the run was started from a different intake: start a new run id."),
            Arg("allow_placeholders", "boolean", "Accept <placeholder> values (for checking templates).",
                default=False)])
def intake_validate(call, file, run_dir, allow_placeholders):
    data, raw = load_file(file)
    access = (data.get("hardware") or {}).get("access") if isinstance(data.get("hardware"), dict) else None
    if isinstance(access, dict) and REDACTED in access.values():
        raise OpError(f"{file} is a redacted run-directory copy of an intake (access details removed); validate "
                      "the original intake file instead (state.json: intake_source)")
    if run_dir and not data.get("run_id"):
        data["run_id"] = Path(run_dir).name  # the run directory names the run
    result = check(data, allow_placeholders=allow_placeholders)
    result["intake_sha256"] = sha256_hex(raw)
    if run_dir and Path(run_dir).name != result["intake"]["run_id"]:
        result["warnings"].append(f"run_id {result['intake']['run_id']!r} differs from the run directory name "
                                  f"{Path(run_dir).name!r}")
    if run_dir:
        if result["placeholders"]:
            raise OpError("refusing to start a run with placeholder values: " + ", ".join(result["placeholders"]))
        result["run"] = start_run(Path(run_dir), Path(file), raw, result)
    return result


def _parse_time(s: str) -> _dt.datetime:
    t = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=_dt.timezone.utc)


def _hours(a: _dt.datetime, b: _dt.datetime) -> float:
    return max(0.0, (b - a).total_seconds() / 3600.0)


def budget_status(state: dict, now: _dt.datetime) -> dict:
    b = state.get("budget") or {}
    price_default = b.get("price_usd_h")
    host_h, usd, open_segments = 0.0, 0.0, 0
    for seg in b.get("meter") or []:
        start = _parse_time(seg["start"])
        end = _parse_time(seg["end"]) if seg.get("end") else now
        open_segments += 0 if seg.get("end") else 1
        h = _hours(start, end)
        host_h += h
        price = seg.get("price_usd_h", price_default)
        usd += h * price if price else 0.0
    wall_h = _hours(_parse_time(b["run_started"]), now) if b.get("run_started") else None
    cap_h, cap_usd = b.get("host_hours_cap"), b.get("usd_cap")
    reserve_h = b.get("report_reserve_h", cap_h * REPORT_RESERVE if cap_h else None)
    left_h = cap_h - max(host_h, wall_h or 0.0) if cap_h else None
    left_usd = cap_usd - usd if cap_usd else None
    deadline = _parse_time(b["deadline"]) if b.get("deadline") else None
    deadline_left_h = (deadline - now).total_seconds() / 3600 if deadline else None
    if deadline is not None and reserve_h is None:
        duration = _hours(_parse_time(b["run_started"]), deadline) if b.get("run_started") else 0
        reserve_h = max(1 / 6, duration * REPORT_RESERVE)
    stage_key = state.get("current_stage")
    stage = (state.get("stages") or {}).get(stage_key) or {}
    stage_used = _hours(_parse_time(stage["started"]), now) if stage.get("started") else None
    box = stage.get("time_box_h") if b.get("stage_timeboxes", True) else None
    if ((left_h is not None and left_h <= 0) or (left_usd is not None and left_usd <= 0)
            or (deadline_left_h is not None and deadline_left_h <= 0)):
        verdict, action = "over_budget", "stop GPU work now; release the host; write the report (stage 8)"
    elif (stage_key != "8_report" and reserve_h is not None and
          ((left_h is not None and left_h <= reserve_h) or
           (deadline_left_h is not None and deadline_left_h <= reserve_h))):
        verdict, action = "reserve_only", "only the report reserve is left: go to stage 8 now"
    elif box is not None and stage_used is not None and stage_used > box:
        verdict, action = "stage_over", (f"stage {stage_key} is past its {box} h time box: record what is "
                                         "unfinished as NOT RUN and move to the next stage")
    else:
        verdict, action = "ok", "continue"
    r3 = (lambda x: None if x is None else round(x, 3))
    return {"claim_class": "derived", "now": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "verdict": verdict, "action": action,
            "spent": {"host_hours": r3(host_h), "usd": round(usd, 2), "wall_clock_h": r3(wall_h),
                      "open_meter_segments": open_segments},
            "cap": {"host_hours": cap_h, "usd": cap_usd, "report_reserve_h": r3(reserve_h),
                    "spend_cap": b.get("spend_cap", "bounded"), "deadline": b.get("deadline")},
            "left": {"host_hours": r3(left_h), "usd": None if left_usd is None else round(left_usd, 2),
                     "deadline_h": r3(deadline_left_h)},
            "stage": {"key": stage_key, "used_h": r3(stage_used), "time_box_h": box},
            "formula": "spent host hours = sum of meter segments (open ones up to now); usd = hours x price; "
                       "left = cap - max(host hours, wall clock)"}


@operation("intake_budget", ("intake", "budget"), "Budget and time-box check for a run directory: spent and left "
           "(host hours, USD, wall clock) and the current stage's time box, with a verdict "
           "(ok | stage_over | reserve_only | over_budget). Read-only.",
           [Arg("run_dir", "string", "The run directory (runs/<run-id>) holding state.json.", required=True,
                positional=True),
            Arg("now", "string", "Evaluate at this ISO-8601 UTC time instead of the current time.")])
def intake_budget(call, run_dir, now):
    path = Path(run_dir) / "state.json"
    if not path.is_file():
        raise OpError(f"{path} not found: start the run with 'inferencekit intake validate <file> --run-dir "
                      f"{run_dir}'")
    try:
        state = json.loads(path.read_text())
    except ValueError:
        raise OpError(f"{path} is not valid JSON") from None
    try:
        t = _parse_time(now) if now else _dt.datetime.now(_dt.timezone.utc)
        return {"run_dir": str(run_dir), **budget_status(state, t)}
    except (KeyError, TypeError, ValueError) as e:
        raise OpError(f"{path}: malformed budget/stage fields ({e})") from None
