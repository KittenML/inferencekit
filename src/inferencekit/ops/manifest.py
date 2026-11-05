"""manifest validate: check a run manifest before anyone replays it.

The run manifest (``runs/<run-id>/manifest.json``, schema ``schemas/run-manifest.v1.json``)
is what a re-run of the manifest replays on a clean host: the model at a pinned
revision, each engine's install and launch commands, the workload and the hash of the
harness wheel. A manifest that cannot be replayed exactly is useless as evidence, so
validation has two layers:

  1. JSON Schema (types, required fields, allowed values);
  2. semantic rules the schema cannot express:
     * engine labels are unique (they name directories and result columns);
     * every engine has a non-empty install and launch command;
     * the model revision is a full 40-character commit sha (a branch or tag can move);
     * engine ports are distinct (engines may run side by side);
     * no secret values in an engine's env (pass secrets by name to a re-run);
     * an engine's ``host``, when given, is a loopback address (a re-run runs every
       engine on the host it measures);
     * hashes, when given, are 64 hex characters;
     * ``workload.extra_body`` (merged into every timed request, e.g. sampling settings) sets
       nothing that changes what is timed or how it is read (what the harness sends itself, the
       exact-length and token-counting fields, early stops, extra sequences; EXTRA_BODY_RESERVED),
       keeps chat template settings in ``workload.chat_template_kwargs`` and holds no secret.

Warnings flag things a re-run can still run but a reviewer should see, e.g. a
launch command that names neither the model revision nor a local model path, or a
local ``models/...`` path without ``model.local_dir`` (a re-run would not put the
weights there).

Paths: ``model.local_dir`` and an engine's ``cwd`` are relative to the re-run's working
directory, which is the uploaded copy of the agent's run directory; install commands and
servers run there.

An invalid manifest is an error (exit 1, every problem in the message) so that
``manifest validate m.json && ...`` stops; ``--no-strict`` returns the result with
``valid: false`` instead. Nothing is measured here, so no evidence record is written; the
manifest's sha256 is returned so a report can cite exactly what was checked.
"""

from __future__ import annotations

import json
import re
from importlib import resources
from pathlib import Path

from inferencekit.canonical import file_sha256
from inferencekit.evidence import looks_secret_key, looks_secret_name
from inferencekit.registry import Arg, OpError, operation

SCHEMA_FILE = "run-manifest.v1.json"
FULL_SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
LOOPBACK = ("127.0.0.1", "localhost", "::1", "[::1]")


def _is_local_path(arg: str, local_dir: str | None) -> bool:
    """A launch argument that names a path on the host rather than a Hub repo id."""
    if arg.startswith(("/", "./", "../", "~", "models/")):
        return True
    if local_dir:
        d = local_dir.rstrip("/")
        return arg == d or arg.startswith(d + "/") or arg in ("./" + d, "./" + d + "/")
    return False


def schema_path() -> Path:
    """The schema shipped in the wheel, or the repo copy in a source checkout."""
    try:
        packaged = resources.files("inferencekit").joinpath(f"schemas/{SCHEMA_FILE}")
        if packaged.is_file():
            return Path(str(packaged))
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    return Path(__file__).resolve().parents[3] / "schemas" / SCHEMA_FILE


_schema_cache: dict | None = None


def load_schema() -> dict:
    global _schema_cache
    if _schema_cache is None:
        _schema_cache = json.loads(schema_path().read_text())
    return _schema_cache


def schema_errors(manifest) -> list[str]:
    import jsonschema

    validator = jsonschema.Draft202012Validator(load_schema())
    out = []
    for err in sorted(validator.iter_errors(manifest), key=lambda e: [str(p) for p in e.path]):
        where = "/".join(str(p) for p in err.path) or "(root)"
        out.append(f"{where}: {err.message}")
    return out


# Request fields an extra body must not set, because they change what is timed or how it is read: what the harness
# sends itself (prompt, length, streaming), the exact-length fields bench serve sets (serve/profiles.py), fields that
# end or steer generation early or add sequences, the token-id and logprob fields bench serve counts tokens with, and
# fields that change the prompt or the reply text. An extra body is for sampling settings. A re-run refuses the
# same fields when it re-runs a manifest.
EXTRA_BODY_RESERVED = {
    "model": "the harness sets model itself (from the manifest)",
    "messages": "the harness sets messages itself (from the prompts)",
    "prompt": "the harness sets prompt itself (from the prompts)",
    "max_tokens": "the harness sets max_tokens itself (from the prompts and workload.max_tokens)",
    "max_completion_tokens": "the harness sets max_completion_tokens itself (as max_tokens)",
    "stream": "the harness sets stream itself (every timed request streams)",
    "stream_options": "the harness sets stream_options itself (usage in the last chunk)",
    "n": "the harness sets n itself (one sequence per request)",
    "best_of": "best_of generates extra sequences per request, so the timing is no longer one sequence's",
    "ignore_eos": "bench serve sets ignore_eos for exact-length decoding (bench serve --no-exact-length turns it off)",
    "min_tokens": "bench serve sets min_tokens for exact-length decoding (bench serve --no-exact-length turns it off)",
    "stop": "stop ends generations early, so engines would not decode the same number of tokens",
    "stop_token_ids": "stop_token_ids ends generations early, so engines would not decode the same number of tokens",
    "logit_bias": "logit_bias steers which tokens are chosen (including the end of the reply)",
    "return_token_ids": "bench serve sets return_token_ids to count streamed tokens",
    "return_tokens_as_token_ids": "bench serve sets return_tokens_as_token_ids to count streamed tokens",
    "logprobs": "bench serve sets logprobs when it counts tokens through them (and logprobs cost decode time)",
    "top_logprobs": "bench serve sets top_logprobs when it counts tokens through logprobs",
    "echo": "echo returns the prompt with the reply, which changes what is streamed and counted",
    "suffix": "suffix changes the prompt",
}
# Settings that only matter when sampling: with none of them and no temperature, bench serve stays greedy.
SAMPLING_ONLY = ("top_k", "top_p", "min_p", "seed")


def _extra_body_schema() -> dict:
    return load_schema()["properties"]["workload"]["properties"]["extra_body"]


def extra_body_problems(body, where: str = "workload/extra_body", typed: bool = True) -> tuple[list[str], list[str]]:
    """(errors, warnings) for an extra request body (``workload.extra_body`` or ``bench rounds --extra-body``).

    ``typed``: also check the sampling fields' types and ranges against the run-manifest schema (temperature >= 0,
    0 < top_p <= 1, ...); manifest validate passes False because its schema check already does.
    """
    if not isinstance(body, dict):
        return [f"{where}: must be a JSON object"], []
    errors, warnings = [], []
    for key in body:
        k = str(key)
        if k in EXTRA_BODY_RESERVED:
            errors.append(f"{where}/{k}: {EXTRA_BODY_RESERVED[k]}; an extra body must not set it")
        elif k == "chat_template_kwargs":
            errors.append(f"{where}/chat_template_kwargs: put chat template settings in "
                          "workload.chat_template_kwargs (they are merged per prompt there)")
        elif looks_secret_key(k):  # the JSON-key rule: skip_special_tokens and min_tokens are request fields
            errors.append(f"{where}/{k}: looks like a secret; an extra body is stored with the evidence")
    if typed:
        import jsonschema

        validator = jsonschema.Draft202012Validator(_extra_body_schema())
        for err in sorted(validator.iter_errors(body), key=lambda e: [str(p) for p in e.path]):
            errors.append(f"{'/'.join([where, *(str(p) for p in err.path)])}: {err.message}")
    used = [k for k in SAMPLING_ONLY if k in body]
    if used and not body.get("temperature"):
        warnings.append(f"{where}: sets {', '.join(used)} but no temperature above 0; bench serve sends temperature 0 "
                        "(greedy decoding), so these settings change nothing on most engines")
    return errors, warnings


def semantic_checks(m: dict) -> tuple[list[str], list[str]]:
    """(errors, warnings) for rules JSON Schema cannot express. Tolerates a schema-invalid manifest."""
    errors, warnings = [], []
    model = m.get("model") if isinstance(m.get("model"), dict) else {}
    rev = model.get("revision")
    if isinstance(rev, str) and not FULL_SHA1.match(rev):
        errors.append(f"model/revision: {rev!r} is not a full 40-character lowercase commit sha "
                      "(a branch, tag or short sha can point at different weights later)")
    tok_rev = model.get("tokenizer_revision")
    if isinstance(tok_rev, str) and tok_rev and not FULL_SHA1.match(tok_rev):
        errors.append(f"model/tokenizer_revision: {tok_rev!r} is not a full 40-character commit sha")

    local_dir = model.get("local_dir") if isinstance(model.get("local_dir"), str) else None
    engines = m.get("engines") if isinstance(m.get("engines"), list) else []
    labels: dict[str, int] = {}
    ports: dict[int, str] = {}
    for i, e in enumerate(engines):
        if not isinstance(e, dict):
            continue
        where = f"engines/{i}"
        label = e.get("label")
        if isinstance(label, str):
            if label in labels:
                errors.append(f"{where}/label: {label!r} is already used by engines/{labels[label]}; "
                              "labels must be unique")
            else:
                labels[label] = i
        install = e.get("install")
        if not isinstance(install, list) or not [c for c in install if isinstance(c, str) and c.strip()]:
            errors.append(f"{where}/install: needs at least one non-empty command")
        elif any(isinstance(c, str) and not c.strip() for c in install):
            errors.append(f"{where}/install: contains an empty command")
        launch = e.get("launch")
        if not isinstance(launch, list) or not launch or not (isinstance(launch[0], str) and launch[0].strip()):
            errors.append(f"{where}/launch: needs a non-empty argv (program first)")
        port = e.get("port")
        if isinstance(port, int) and not isinstance(port, bool):
            if port in ports:
                errors.append(f"{where}/port: {port} is already used by {ports[port]}; ports must be distinct")
            else:
                ports[port] = label if isinstance(label, str) else where
        for k, v in (e.get("env") or {}).items() if isinstance(e.get("env"), dict) else ():
            if (looks_secret_name(k) or looks_secret_key(k)) and v:
                errors.append(f"{where}/env/{k}: looks like a secret; never put secret values in a manifest "
                              "(a re-run passes secrets by environment variable name)")
        host = e.get("host")
        if isinstance(host, str) and host.strip().lower() not in LOOPBACK and not host.startswith("127."):
            errors.append(f"{where}/host: {host!r} is not a loopback address; a re-run runs every engine on the "
                          "host it measures (use 127.0.0.1)")
        ready = e.get("ready_check")
        if isinstance(ready, str) and ready and not ready.startswith("/"):
            errors.append(f"{where}/ready_check: {ready!r} must be an HTTP path starting with '/'")
        if not e.get("ready_check"):
            warnings.append(f"{where}: no ready_check; the harness must guess when the server is up")
        if isinstance(launch, list) and isinstance(rev, str) and FULL_SHA1.match(rev):
            text = " ".join(str(a) for a in launch)
            if rev not in text and not any(_is_local_path(str(a), local_dir) for a in launch[1:]):
                warnings.append(f"{where}/launch: does not mention model/revision {rev[:12]}... or a local model "
                                "path; the engine may load a different revision (a re-run runs engines "
                                "offline, so an unpinned repo id fails to load there)")
        if isinstance(launch, list) and not local_dir and any(str(a).startswith(("models/", "./models/"))
                                                              for a in launch[1:]):
            warnings.append(f"{where}/launch: uses a models/... path but model.local_dir is not set; a re-run "
                            "downloads the model to the HF cache, so that path will not exist unless an install "
                            "command creates it")
        if not e.get("resolved_config"):
            warnings.append(f"{where}: no resolved_config (what the engine logged it used); knob claims "
                            "cannot be checked against it")

    harness = m.get("harness") if isinstance(m.get("harness"), dict) else {}
    for key in ("wheel_sha256", "dev_prompt_set_sha256"):
        value = harness.get(key)
        if isinstance(value, str) and value and not SHA256.match(value):
            errors.append(f"harness/{key}: {value!r} is not a 64-character lowercase sha256")
    if harness.get("wheel_sha256") in (None, ""):
        warnings.append("harness/wheel_sha256 is empty: a re-run refuses a manifest without the wheel hash")

    workload = m.get("workload") if isinstance(m.get("workload"), dict) else {}
    modes = workload.get("modes") if isinstance(workload.get("modes"), list) else []
    slo = workload.get("slo") if isinstance(workload.get("slo"), dict) else {}
    if "load" in modes and not (slo.get("ttft_p95_ms") and slo.get("tpot_p95_ms")):
        warnings.append("workload: load mode without slo.ttft_p95_ms and slo.tpot_p95_ms; N* cannot be computed")
    if isinstance(workload.get("extra_body"), dict):
        body_errors, body_warnings = extra_body_problems(workload["extra_body"], typed=False)  # schema: types
        errors += body_errors
        warnings += body_warnings
    return errors, warnings


def pin_errors(manifest, run_dir) -> list[str]:
    """The intake's pinned workload.chat_template_kwargs (``<run_dir>/intake.normalized.json``) against the
    manifest's: every pinned key must be there with the pinned value."""
    from inferencekit.ops.intake import PIN_FILE, intake_pin, pin_conflicts

    pin = intake_pin(run_dir)
    if not pin or not isinstance(manifest, dict):
        return []
    have = (manifest.get("workload") or {}).get("chat_template_kwargs") if isinstance(
        manifest.get("workload"), dict) else None
    have = have if isinstance(have, dict) else {}
    bad = pin_conflicts(have, pin) + [f"{k}: missing (pinned {json.dumps(v)})" for k, v in sorted(pin.items())
                                      if k not in have]
    if not bad:
        return []
    return [f"workload/chat_template_kwargs: does not match the intake's pin in {PIN_FILE} ({'; '.join(bad)}); "
            f"set workload.chat_template_kwargs to {json.dumps(pin, sort_keys=True)}"]


def check_manifest(manifest, run_dir=None) -> dict:
    """Validate a parsed manifest: {valid, errors, warnings}. Used by a re-run script too. With
    ``run_dir``, the intake's pinned chat_template_kwargs there (intake.normalized.json) must be honoured."""
    errors = schema_errors(manifest)
    sem_errors, warnings = semantic_checks(manifest) if isinstance(manifest, dict) else ([], [])
    errors += [e for e in sem_errors if e not in errors]
    errors += pin_errors(manifest, run_dir)
    return {"valid": not errors, "errors": errors, "warnings": warnings}


def load_manifest(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise OpError(f"manifest not found: {path}")
    try:
        return json.loads(p.read_text())
    except ValueError as e:
        raise OpError(f"{path} is not valid JSON: {e}") from None


@operation("validate_manifest", ("manifest", "validate"), "Check a run manifest (schema + semantic rules: unique "
           "engine labels, non-empty install/launch, full commit sha revision, distinct ports, no secrets, and the "
           "intake's pinned chat_template_kwargs when intake.normalized.json sits next to it) before anyone "
           "re-runs it on a clean machine.",
           [Arg("file", "string", "Path to runs/<run-id>/manifest.json.", required=True, positional=True),
            Arg("strict", "boolean", "Exit non-zero (an error listing every problem) when the manifest is invalid; "
                "--no-strict returns the result with valid: false instead.", default=True)])
def validate_manifest(call, file, strict):  # noqa: A002
    manifest = load_manifest(file)
    result = check_manifest(manifest, Path(file).resolve().parent)  # runs/<run-id>/: the intake pin, if any
    sha, size = file_sha256(file)
    out = {"file": str(file), "sha256": sha, "bytes": size, "schema": "run-manifest.v1", **result,
           "engines": [e.get("label") for e in manifest.get("engines", []) if isinstance(e, dict)]
           if isinstance(manifest, dict) else []}
    if strict is not False and not result["valid"]:
        raise OpError(f"{file}: invalid run manifest ({len(result['errors'])} problem(s)): "
                      + "; ".join(result["errors"]))
    return out
