"""quality trajectory / score / gsm8k / niah / sampling: does a faster engine still produce the model's output?

``quality trajectory``  each engine's own greedy text over the OpenAI chat API, with the
    top-k logprobs and token ids at every position, one JSON file per prompt under
    ``<run-dir>/quality/<label>/`` plus an ``index.json`` of their sha256 hashes. An engine
    that returns no logprobs (or refuses the field) gets a text-only trajectory instead.
``quality score``  teacher-forces every trajectory through a higher-precision reference
    (vLLM ``/v1/completions`` with a token-id prompt, ``max_tokens=1`` and the vLLM
    extension ``prompt_logprobs``), then reports per engine and per position bucket:
    top-1 agreement with the reference's argmax, KL(reference || engine) and total variation
    on the top-k supports (approximations documented in ``quality/metrics.py``), the reference's NLL of
    the engine's tokens (and its perplexity), and the first divergence from the
    reference's own greedy trajectory (``--baseline``). The A/A noise floor is the
    reference's second greedy run (``--baseline-repeat``, else the first one) scored the
    same way: it measures how much the reference's decode path and its teacher-forced
    prefill path (and, with a repeat, two runs of it) already disagree. Each engine gets
    PASS / FAIL / UNPROVEN from paired bootstrap CIs (over prompts) of "engine minus
    floor" against pre-registered margins. KL is compared at the engine's k: both lists
    are cut to the k alternatives the engine returned (see ``_score_one``). A PASS also
    needs clean inputs: every prompt the baseline covers scored for the engine and the
    floor, the same prompt file, the same chat template, and every index.json listed by
    an intact ``quality_trajectory`` evidence record; otherwise it becomes UNPROVEN.
``quality gsm8k``  GSM8K test split, zero-shot chain of thought, temperature 0, exact
    numeric match; accuracy with a Wilson CI; ``--paired-with`` adds McNemar's exact test
    (refused when the two runs differ in dataset, prompt, max_tokens or chat template).
    ``--concurrency 0`` sizes the request pool to the server; ``--time-budget-s`` stops early.
``quality sampling``  first-token histograms under given sampling settings, candidate vs
    baseline, against the baseline's own A/A floor (a second seed range).

Sources for the vLLM request fields (checked against vLLM main, commit 0fae39a, 13 Oct 2026):
``vllm/entrypoints/openai/completion/protocol.py`` (``prompt_logprobs``,
``return_tokens_as_token_ids``, ``return_token_ids``; ``CompletionResponseChoice.prompt_logprobs:
list[dict[int, Logprob] | None]``) and ``vllm/entrypoints/serve/tokenize/protocol.py``
(``TokenizeChatRequest``: ``messages``, ``add_generation_prompt`` (default true),
``add_special_tokens`` (default false), ``chat_template_kwargs``; response ``tokens``).
"""

from __future__ import annotations

import json
import math
import re
import shutil
import statistics
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from inferencekit import evidence as ev
from inferencekit import stats
from inferencekit.canonical import digest, file_sha256, sha256_hex
from inferencekit.ops.correctness import first_divergence
from inferencekit.ops.intake import apply_pin
from inferencekit.quality import gsm8k as g8
from inferencekit.quality import metrics as qm
from inferencekit.quality import sampling as sm
from inferencekit.quality.client import Client, HTTPError
from inferencekit.quality.logprobs import argmax_token, parse_chat_logprobs, parse_prompt_logprobs
from inferencekit.registry import Arg, OpError, operation

ENGINE_KINDS = ("vllm", "sglang", "llamacpp", "openai")
TRAJ_FORMAT = "inferencekit.quality.trajectory.v1"
INDEX_FORMAT = "inferencekit.quality.index.v1"
SCORE_FORMAT = "inferencekit.quality.score.v1"
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_ALLOWED = re.compile(r"max(?:imum)?(?: allowed)?\D{0,20}?(\d+)", re.IGNORECASE)


# --------------------------------------------------------------------------- shared helpers

def _kind(v2_kind: str) -> tuple[str, dict]:
    """Evidence kind: the v2 kind once the v2 schema is merged; until then v1 ``correctness`` + data.v2_kind."""
    if v2_kind in ev.KINDS:
        return v2_kind, {}
    return "correctness", {"v2_kind": v2_kind}


def _save(call, run_dir: Path | None, kind: str, **kw) -> dict:
    """Build, store and (if a run dir is given) copy the record to ``<run-dir>/evidence/<id>.json``."""
    real_kind, extra = _kind(kind)
    kw["data"] = {**extra, **kw.get("data", {})}
    rec = ev.Store().save(ev.build(call, kind=real_kind, **kw))
    if run_dir is not None:
        out = Path(run_dir) / "evidence"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{rec['evidence_id']}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    return rec


def _json_arg(name: str, text: str | None) -> dict | None:
    if not text:
        return None
    try:
        value = json.loads(text)
    except ValueError as e:
        raise OpError(f"{name} is not valid JSON: {e}") from None
    if not isinstance(value, dict):
        raise OpError(f"{name} must be a JSON object")
    return value


def _safe(name: str, what: str) -> str:
    if not _SAFE.match(str(name)):
        raise OpError(f"{what} {name!r} must match {_SAFE.pattern} (it becomes a file name)")
    return str(name)


def load_prompts(path: str) -> tuple[list[dict], str]:
    """Prompt-set JSONL or .jsonl.gz (contract format: id, bucket, messages, max_tokens) -> items, sha256 of
    the JSONL bytes."""
    from inferencekit import prompts as sets

    p = sets.existing(path)
    if not p.is_file():
        raise OpError(f"prompts file not found: {path}")
    try:
        raw = sets.read_bytes(p)
        text = raw.decode("utf-8")
    except (OSError, ValueError) as e:  # missing, a truncated / corrupt .jsonl.gz, or not UTF-8
        raise OpError(f"prompts file {p}: cannot read ({e})") from None
    items, seen = [], set()
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            raise OpError(f"{p}:{n}: not JSON") from None
        if "messages" not in obj and "prompt" in obj:
            obj["messages"] = [{"role": "user", "content": obj["prompt"]}]
        if not isinstance(obj.get("messages"), list) or "id" not in obj:
            raise OpError(f"{p}:{n}: need 'id' and 'messages' (OpenAI chat format)")
        pid = _safe(obj["id"], f"{p}:{n}: prompt id")
        if pid in seen:
            raise OpError(f"{p}:{n}: duplicate prompt id {pid}")
        seen.add(pid)
        items.append(obj)
    if not items:
        raise OpError(f"{p}: no prompts")
    return items, sha256_hex(raw)


def _post_capped(client: Client, path: str, body: dict, key: str, state: dict, lock: threading.Lock) -> tuple:
    """POST with ``body[key] = state['k']``; on a "too many logprobs" 400, lower k (shared) and retry.

    Engines cap top-k logprobs (vLLM: ``--max-logprobs``, default 20; the OpenAI API: 20).
    The cap is parsed from the error ("... max allowed: 5") or k is halved.
    """
    while True:
        with lock:
            k = state["k"]
        body = {**body, key: k}
        try:
            return client.post(path, body), body
        except HTTPError as e:
            msg = e.message.lower()
            if e.status in (400, 422) and "logprob" in msg:
                m = _MAX_ALLOWED.search(e.message)
                new = int(m.group(1)) if m and 0 < int(m.group(1)) < k else k // 2
                if new < 1 or new >= k:
                    raise
                with lock:
                    state["k"] = min(state["k"], new)
                    state.setdefault("cap_errors", []).append(e.message[:200])
                continue
            raise


def _run_pool(fn, items, concurrency: int, serial_first: bool = False) -> list:
    """``fn`` over ``items`` with up to ``concurrency`` in flight, results in order.

    ``serial_first`` runs item 0 alone first: a logprob cap that request discovers
    (``_post_capped``) then holds for every later request, so all results share one k.
    """
    items = list(items)
    head = [fn(items[0])] if serial_first and items else []
    rest = items[len(head):]
    if concurrency <= 1:
        return head + [fn(x) for x in rest]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return head + list(pool.map(fn, rest))


# --------------------------------------------------------------------------- quality trajectory

@operation("quality_trajectory", ("quality", "trajectory"),
           "Greedy generation per prompt over the OpenAI chat API with top-k logprobs and token ids at every "
           "position; one file per prompt under <run-dir>/quality/<label>/ (input to quality score).",
           [Arg("base_url", "string", "Engine server, e.g. http://127.0.0.1:8000 (with or without /v1).",
                required=True),
            Arg("model", "string", "Model name the server expects.", required=True),
            Arg("prompts", "string", "Prompt-set JSONL (id, bucket, messages, max_tokens).", required=True),
            Arg("label", "string", "Name of this engine/configuration (directory name).", required=True),
            Arg("run_dir", "string", "Run directory (runs/<run-id>).", required=True),
            Arg("max_tokens", "integer", "Tokens to generate per prompt (default: each item's max_tokens, else 256)."),
            Arg("top_logprobs", "integer", "Alternatives per position (lowered automatically to the engine's cap).",
                default=20),
            Arg("engine_kind", "string", "vllm adds return_tokens_as_token_ids / return_token_ids so token ids "
                "are exact; others rely on id fields in the logprobs.", default="openai", choices=ENGINE_KINDS),
            Arg("chat_template_kwargs", "string", "JSON object passed as chat_template_kwargs, e.g. "
                "'{\"enable_thinking\": false}' (also used by quality score to tokenize the prompt)."),
            Arg("seed", "integer", "Request seed.", default=0),
            Arg("limit", "integer", "Only the first N prompts."),
            Arg("concurrency", "integer", "Requests in flight (1 = sequential; batching can change numerics).",
                default=1),
            Arg("api_key_env", "string", "Name of the env var holding the API key (value never recorded).",
                default="OPENAI_API_KEY"),
            Arg("timeout", "integer", "Seconds per request.", default=600)])
def quality_trajectory(call, base_url, model, prompts, label, run_dir, max_tokens, top_logprobs, engine_kind,
                       chat_template_kwargs, seed, limit, concurrency, api_key_env, timeout):
    label = _safe(label, "label")
    items, prompts_sha = load_prompts(prompts)
    if limit is not None:
        items = items[:max(0, limit)]
    if not items:
        raise OpError("no prompts to run (limit 0?)")
    base_kwargs = apply_pin(_json_arg("chat_template_kwargs", chat_template_kwargs) or {}, run_dir,
                            "quality trajectory")
    if top_logprobs < 1:
        raise OpError("top_logprobs must be >= 1")
    run = Path(run_dir)
    out_dir = run / "quality" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    client = Client(base_url, api_key_env, timeout)
    state, lock = {"k": top_logprobs}, threading.Lock()
    tokenize_state = {"ok": True}

    def one(item: dict) -> dict:
        n = max_tokens or item.get("max_tokens") or 256
        body = {"model": model, "messages": item["messages"], "max_tokens": int(n), "temperature": 0,
                "seed": seed, "logprobs": True, "stream": False}
        # The record's own chat_template_kwargs field is merged over the flag (per key), as bench serve does.
        own = item.get("chat_template_kwargs")
        kwargs = {**base_kwargs, **own} if isinstance(own, dict) else base_kwargs
        if kwargs:
            body["chat_template_kwargs"] = kwargs
        if engine_kind == "vllm":
            body.update(return_tokens_as_token_ids=True, return_token_ids=True)
        plain = {k: v for k, v in body.items() if k not in ("logprobs", "top_logprobs")}
        try:
            if state.get("refused"):  # the engine already refused logprobs once: ask for text only
                resp, sent = client.post("/v1/chat/completions", plain), plain
            else:
                try:
                    resp, sent = _post_capped(client, "/v1/chat/completions", body, "top_logprobs", state, lock)
                except HTTPError as e:
                    # An engine without logprob support may refuse the field outright: keep its text.
                    if e.status not in (400, 422) or "logprob" not in e.message.lower():
                        raise
                    resp, sent = client.post("/v1/chat/completions", plain), plain
                    with lock:
                        state["refused"] = e.message[:200]
                        # _post_capped first read the refusal as a cap and lowered k: those were not cap errors
                        state["cap_errors"] = [m for m in state.get("cap_errors", []) if m != state["refused"]]
        except OpError as e:
            return {"id": item["id"], "error": str(e)[:300]}
        parsed = parse_chat_logprobs(resp)
        if parsed["method"] == "none":
            # No logprobs: keep a text-only trajectory. quality score re-tokenizes the text on the reference
            # and still reports agreement and NLL; KL and TV need the engine's own top-k lists.
            if not parsed["text"]:
                return {"id": item["id"], "error": "response has neither logprobs.content nor text"}
            parsed.update(method="text", token_ids=None, tokens=[], logprobs=[], top=[], top_k_returned=0,
                          id_mismatch_positions=[])
        prompt_ids, prompt_src = resp.get("prompt_token_ids"), "response.prompt_token_ids"
        if not prompt_ids and tokenize_state["ok"]:
            tb = {"model": model, "messages": item["messages"], "add_generation_prompt": True,
                  "add_special_tokens": False}
            if kwargs:
                tb["chat_template_kwargs"] = kwargs
            try:
                prompt_ids, prompt_src = client.post("/tokenize", tb).get("tokens"), "engine_tokenize"
            except OpError:
                tokenize_state["ok"] = False
        if not prompt_ids:
            prompt_ids, prompt_src = None, None
        doc = {"format": TRAJ_FORMAT, "id": item["id"], "bucket": item.get("bucket"), "label": label,
               "engine_kind": engine_kind, "model": model, "base_url": client.base, "messages": item["messages"],
               "chat_template_kwargs": kwargs or None, "request": sent,
               "top_logprobs_requested": sent.get("top_logprobs"), "top_logprobs_returned": parsed["top_k_returned"],
               "token_id_method": parsed["method"], "token_ids": parsed["token_ids"], "tokens": parsed["tokens"],
               "logprobs": parsed["logprobs"],
               "top_logprobs": [sorted(([k, v] for k, v in t.items()), key=lambda kv: -(kv[1] if kv[1] is not None else -1e30))
                                for t in parsed["top"]],
               "id_mismatch_positions": parsed["id_mismatch_positions"],
               "prompt_token_ids": prompt_ids, "prompt_token_ids_source": prompt_src,
               "usage": resp.get("usage"), "finish_reason": parsed["finish_reason"], "text": parsed["text"]}
        path = out_dir / f"{item['id']}.json"
        path.write_text(json.dumps(doc, ensure_ascii=False) + "\n")
        return {"id": item["id"], "file": path.name, "sha256": file_sha256(path)[0], "tokens": len(parsed["tokens"]),
                "chars": len(parsed["text"] or ""), "k": parsed["top_k_returned"], "method": parsed["method"],
                "mismatch": bool(parsed["id_mismatch_positions"]), "finish_reason": parsed["finish_reason"]}

    results = _run_pool(one, items, concurrency, serial_first=True)
    ok = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    index = {"format": INDEX_FORMAT, "label": label, "model": model, "engine_kind": engine_kind,
             "prompts_sha256": prompts_sha, "files": {r["id"]: {"file": r["file"], "sha256": r["sha256"]} for r in ok},
             "failed": {r["id"]: r["error"] for r in failed}}
    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")

    methods = sorted({r["method"] for r in ok})
    text_only = [r for r in ok if r["method"] == "text"]
    k_min = min((r["k"] for r in ok if r["method"] != "text"), default=0)
    limitations, missing = [], []
    if failed:
        missing.append("trajectories")
        limitations.append(f"{len(failed)}/{len(items)} prompts failed (see data.failed)")
    if text_only:
        missing.append("logprobs")  # a required feature: KL and TV cannot be computed for these prompts
        why = f"it refused them: {state['refused']}" if state.get("refused") else "none in the response"
        limitations.append(f"text only for {len(text_only)}/{len(ok)} prompts: the engine returned no logprobs "
                           f"({why}). quality score re-tokenizes their text on the reference and reports top-1 "
                           "agreement and NLL; KL and TV are not computed for them, and re-tokenized text can differ "
                           "from the tokens the engine actually produced")
    if len(text_only) < len(ok) and k_min < top_logprobs:
        limitations.append(f"engine returned only {k_min} alternatives per position (requested {top_logprobs}); "
                           f"quality score compares KL at top-{k_min} on both sides, which is less sensitive")
    if "strings" in methods:
        limitations.append("token ids not recoverable for some prompts (token strings only): quality score "
                           "re-tokenizes their text, and KL is not computed for them")
    if any(r["mismatch"] for r in ok):
        limitations.append("token ids from token_id strings and return_token_ids disagree at some positions")
    truncated = sum(1 for r in ok if r["finish_reason"] == "length")
    tokens = sum(r["tokens"] for r in ok)
    metrics = [ev.metric("prompts_ok", len(ok), "prompts", "count", n=len(items)),
               ev.metric("generated_tokens", tokens, "tokens", "sum", n=len(ok) - len(text_only)),
               ev.metric("top_logprobs_returned", k_min, "tokens", "min", n=len(ok) - len(text_only))]
    if text_only:
        metrics.append(ev.metric("text_only_prompts", len(text_only), "prompts", "count", n=len(ok)))
    ident = ev.identity(engine={"name": engine_kind}, model={"id": model},
                        workload={"gen_len": max_tokens, "batch": concurrency, "seed": seed})
    if len(text_only) == len(ok) and ok:
        what = f"{sum(r['chars'] for r in ok)} characters of text only (no logprobs)"
    else:
        what = (f"{tokens} tokens, top-{k_min} logprobs, token ids via {'/'.join(methods) or 'n/a'}"
                + (f", {len(text_only)} text only" if text_only else ""))
    return _save(call, run, "quality_trajectory", claim_class="observed",
                 summary=f"{label}: greedy trajectories for {len(ok)}/{len(items)} prompts, {what}",
                 identity_=ident, metrics=metrics, artifacts=[ev.artifact(index_path, run, "trajectory_index")],
                 missing=missing, limitations=limitations,
                 data={"label": label, "dir": f"quality/{label}", "prompts_sha256": prompts_sha,
                       "top_logprobs_requested": top_logprobs, "top_logprobs_returned_min": k_min,
                       "token_id_methods": methods, "truncated_at_max_tokens": truncated,
                       "text_only_prompts": len(text_only), "logprobs_refused": state.get("refused"),
                       "chat_template_kwargs": base_kwargs or None, "cap_errors": state.get("cap_errors", [])[:3],
                       "failed": index["failed"], "endpoint_kind": "openai-chat"})


# --------------------------------------------------------------------------- quality score

def _label_dirs(paths: list[str]) -> list[Path]:
    out = []
    for p in paths:
        d = Path(p)
        if (d / "index.json").is_file():
            out.append(d)
        elif d.is_dir():
            out.extend(sorted(c for c in d.iterdir() if (c / "index.json").is_file()))
        else:
            raise OpError(f"trajectories: {p} is not a directory")
    if not out:
        raise OpError(f"no trajectory directories (with index.json) under {paths}")
    return out


def load_trajectories(d: Path) -> tuple[str, dict[str, dict], list[str]]:
    """(label, {prompt id: trajectory}, integrity problems). Files are checked against index.json."""
    index = json.loads((d / "index.json").read_text())
    problems, docs = [], {}
    for pid, meta in sorted(index.get("files", {}).items()):
        f = d / meta["file"]
        if not f.is_file():
            problems.append(f"{d.name}/{meta['file']} is missing")
            continue
        if file_sha256(f)[0] != meta["sha256"]:
            problems.append(f"{d.name}/{meta['file']} does not match index.json (edited after capture?)")
            continue
        docs[pid] = json.loads(f.read_text())
    return index.get("label") or d.name, docs, problems


def _is_kind(rec: dict, v2_kind: str) -> bool:
    """A record of ``v2_kind``, written natively (v2) or as v1 ``correctness`` with data.v2_kind."""
    return rec.get("kind") == v2_kind or (rec.get("data") or {}).get("v2_kind") == v2_kind


def _run_of(d: Path) -> Path | None:
    """``runs/<id>`` for a ``runs/<id>/quality/<label>`` directory."""
    return d.parent.parent if d.parent.name == "quality" else None


class _Records:
    """Evidence records: a run's ``evidence/`` copies first, then the whole store (read once, on first need)."""

    def __init__(self):
        self._store: list[dict] | None = None

    @staticmethod
    def _local(run: Path | None) -> list[dict]:
        out = []
        if run is not None and (run / "evidence").is_dir():
            for p in sorted((run / "evidence").glob("ev_*.json")):
                try:
                    out.append(json.loads(p.read_text()))
                except (OSError, ValueError):
                    continue
        return out

    def find(self, run: Path | None, want) -> tuple[dict | None, bool]:
        """(an intact record for which ``want(record)`` is true or None, whether an edited match was seen)."""
        edited = False
        for source in ("run", "store"):
            if source == "run":
                recs = self._local(run)
            else:
                if self._store is None:
                    self._store = ev.Store().records()
                recs = self._store
            for rec in recs:
                if want(rec):
                    if ev.integrity_problem(rec) is None:
                        return rec, False
                    edited = True
        return None, edited


def _index_problem(d: Path, index_sha: str, records: _Records) -> str | None:
    """None when an intact ``quality_trajectory`` evidence record lists this index.json by sha256.

    The hashes inside index.json catch one trajectory file edited on its own; this catches
    index.json rewritten to match edited files. An evidence id is a content hash, not a
    signature: this makes an edit evident, it cannot stop someone who also forges a record.
    """
    def want(rec: dict) -> bool:
        return _is_kind(rec, "quality_trajectory") and rec.get("claim_class") == "observed" and any(
            a.get("role") == "trajectory_index" and a.get("sha256") == index_sha for a in rec.get("artifacts") or [])

    rec, edited = records.find(_run_of(d), want)
    if rec is not None:
        return None
    if edited:
        return f"{d.name}: the quality_trajectory evidence record for its index.json was edited after capture"
    return (f"{d.name}/index.json: no quality_trajectory evidence record lists it (rewritten after capture, "
            "or produced without quality trajectory)")


def _load_set(d: Path, records: _Records) -> dict:
    """One label dir: its trajectories, the prompt ids its index covers, and integrity problems."""
    label, docs, problems = load_trajectories(d)
    index_path = d / "index.json"
    index = json.loads(index_path.read_text())
    index_sha = file_sha256(index_path)[0]
    problem = _index_problem(d, index_sha, records)
    if problem:
        problems.append(problem)
    failed = dict(index.get("failed") or {})
    return {"dir": d, "label": label, "docs": docs, "problems": problems, "failed": failed,
            "ids": set(index.get("files") or {}) | set(failed), "prompts_sha256": index.get("prompts_sha256"),
            "index_sha256": index_sha}


def _coverage(s: dict, results: list[dict], required: set) -> dict:
    """Which of the required prompt ids (the baseline's) this trajectory set got scored on, and why not."""
    ok = {r["id"] for r in results if "rows" in r}
    errors = {r["id"]: r["error"] for r in results if "error" in r and r["id"] in required}
    missing = sorted(required - ok)
    failed = sorted(set(s["failed"]) & required)
    absent = sorted(required - s["ids"])
    rejected = sorted((required & s["ids"]) - set(s["failed"]) - set(s["docs"]))
    return {"required": len(required), "scored": len(required & ok), "missing": len(missing),
            "missing_ids": missing[:50], "failed_in_trajectory": failed[:50], "absent": absent[:50],
            "rejected_by_integrity_check": rejected[:50], "scoring_errors": dict(sorted(errors.items())[:20]),
            "extra_not_in_baseline": len(ok - required)}


def _coverage_text(label: str, cov: dict) -> str:
    parts = []
    if cov["failed_in_trajectory"]:
        parts.append(f"{len(cov['failed_in_trajectory'])} failed during quality trajectory")
    if cov["absent"]:
        parts.append(f"{len(cov['absent'])} absent from its trajectories")
    if cov["rejected_by_integrity_check"]:
        parts.append(f"{len(cov['rejected_by_integrity_check'])} rejected by the integrity check")
    if cov["scoring_errors"]:
        first = next(iter(cov["scoring_errors"].values()))
        parts.append(f"{len(cov['scoring_errors'])} scoring errors, e.g. {first[:120]}")
    return f"{label}: {cov['missing']} of {cov['required']} prompts not scored ({'; '.join(parts) or 'see coverage'})"


def _doc_k(doc: dict) -> int:
    """Alternatives per position the engine returned for this trajectory (its shortest list)."""
    k = doc.get("top_logprobs_returned")
    if isinstance(k, int):
        return k
    return min((len(t) for t in doc.get("top_logprobs") or []), default=0)


def _top(lp: dict, k: int, keep) -> dict:
    """The ``k`` most likely entries of a {token id: logprob} list (ties broken by id), plus ``keep``."""
    best = sorted((t for t, v in lp.items() if v is not None), key=lambda t: (-lp[t], t))[:k]
    out = {t: lp[t] for t in best}
    if lp.get(keep) is not None:
        out[keep] = lp[keep]
    return out


def _at_k(results: list[dict], k: int) -> list[dict]:
    """Copies of scored results whose per-position ``kl`` / ``tv`` are those at top-``k`` (None where not computed)."""
    return [{**r, "rows": [{**x, "kl": x["kl_at"].get(k), "tv": x["tv_at"].get(k)} for x in r["rows"]]}
            for r in results]


class _PromptIds:
    """Prompt token ids through the reference's chat template: /tokenize, else an HF tokenizer."""

    def __init__(self, client: Client, model: str, tokenizer: str | None):
        self.client, self.model, self.tokenizer_id = client, model, tokenizer
        self.tokenize_ok = True
        self._hf = None
        self.hf_problem: str | None = None
        self._cache: dict = {}
        self.lock = threading.Lock()

    def _hf_ids(self, messages, kwargs):
        if self._hf is None:
            try:
                from transformers import AutoTokenizer  # optional extra
            except ImportError:
                self._hf = False
                self.hf_problem = "transformers not installed (pip install transformers)"
            else:
                try:
                    self._hf = AutoTokenizer.from_pretrained(self.tokenizer_id or self.model)
                except Exception as e:  # network, auth, unknown repo: fall through to the next source
                    self._hf, self.hf_problem = False, f"HF tokenizer failed to load: {str(e)[:200]}"
        if self._hf is False:
            return None
        ids = self._hf.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **(kwargs or {}))
        if not isinstance(ids, list):  # a BatchEncoding in newer transformers
            ids = ids["input_ids"]
        return [int(t) for t in ids]

    def get(self, doc: dict) -> tuple[list[int] | None, str | None]:
        key = digest([doc["messages"], doc.get("chat_template_kwargs")])
        if key in self._cache:
            return self._cache[key]
        ids, src = None, None
        if self.tokenize_ok:
            body = {"model": self.model, "messages": doc["messages"], "add_generation_prompt": True,
                    "add_special_tokens": False}
            if doc.get("chat_template_kwargs"):
                body["chat_template_kwargs"] = doc["chat_template_kwargs"]
            try:
                ids, src = self.client.post("/tokenize", body).get("tokens"), "reference_tokenize"
            except HTTPError as e:
                if e.status in (404, 405, 501):
                    self.tokenize_ok = False
                else:
                    raise
        if ids is None:
            with self.lock:
                ids = self._hf_ids(doc["messages"], doc.get("chat_template_kwargs"))
            src = "hf_tokenizer" if ids is not None else None
        if ids is None and doc.get("prompt_token_ids"):
            ids, src = list(doc["prompt_token_ids"]), "engine_reported"
        self._cache[key] = (ids, src)
        return ids, src


def _retokenize(ref: Client, ref_model: str, doc: dict) -> list[int]:
    """A trajectory's text as the reference tokenizer reads it (for engines without token ids)."""
    return ref.post("/tokenize", {"model": ref_model, "prompt": doc.get("text") or "",
                                  "add_special_tokens": False}).get("tokens") or []


def _as_text(doc: dict) -> dict:
    """A trajectory with its token ids and lists removed: scored from re-tokenized text, as a text-only one is."""
    return {**doc, "token_ids": None, "top_logprobs": [], "top_logprobs_returned": 0}


def _score_one(doc: dict, ref: Client, ref_model: str, pids: _PromptIds, state: dict, lock, cache: dict,
               ks: set[int]) -> dict:
    """Teacher-force one trajectory on the reference and return per-position results.

    KL needs both sides to list the same number of alternatives: a top-20 reference list
    against a top-5 engine list reads as divergence even when the two distributions are
    identical (the reference names 15 tokens the engine never could). So at every position
    both lists are cut to their k most likely tokens plus the trajectory token, with
    k = min(alternatives the engine returned, the reference's k); ``kl`` is that value.
    ``kl_at`` also holds the KL at every other k in ``ks`` (the k of each set being scored),
    so the noise floor can be compared with each engine at the engine's k. ``tv`` and
    ``tv_at`` are the total variation on the same cut lists.
    """
    out = {"id": doc["id"], "bucket": doc.get("bucket")}
    try:
        prompt_ids, src = pids.get(doc)
    except OpError as e:
        return {**out, "error": f"prompt tokenization failed: {e}"}
    if not prompt_ids:
        return {**out, "error": "no prompt token ids (reference has no /tokenize, no HF tokenizer installed, "
                                "and the trajectory has no engine-reported ids)"}
    out["prompt_ids_source"] = src
    engine_prompt = doc.get("prompt_token_ids")
    if engine_prompt and list(engine_prompt) != list(prompt_ids):
        out["template_mismatch"] = "engine prompt ids differ from the reference's"
    elif not engine_prompt and (doc.get("usage") or {}).get("prompt_tokens") not in (None, len(prompt_ids)):
        out["template_mismatch"] = (f"engine counted {doc['usage']['prompt_tokens']} prompt tokens, "
                                    f"reference template gives {len(prompt_ids)}")
    traj, kl_ok = doc.get("token_ids"), True
    if traj is None:  # token strings only: re-tokenize the text with the reference tokenizer
        kl_ok = False
        try:
            traj = _retokenize(ref, ref_model, doc)
        except OpError as e:
            return {**out, "error": f"cannot re-tokenize the engine text: {e}"}
        out["token_ids_source"] = "retokenized_text"
    out["token_ids"] = list(traj)
    if not traj:
        return {**out, "error": "empty trajectory"}
    full = list(prompt_ids) + list(traj)
    key = digest(full)
    try:
        if key in cache:
            plp, k_used = cache[key]
        else:
            body = {"model": ref_model, "prompt": full, "max_tokens": 1, "temperature": 0,
                    "add_special_tokens": False}
            resp, sent = _post_capped(ref, "/v1/completions", body, "prompt_logprobs", state, lock)
            plp, k_used = parse_prompt_logprobs(resp), sent["prompt_logprobs"]
            cache[key] = (plp, k_used)
    except OpError as e:
        return {**out, "error": f"reference request failed: {e}"}
    if len(plp) != len(full):
        return {**out, "error": f"reference returned {len(plp)} prompt_logprobs for {len(full)} tokens"}
    out["ref_k"] = k_used
    out["k_kl"] = k_own = min(_doc_k(doc), k_used)
    kset = sorted({min(k, k_used) for k in ks} | {k_own})
    top = doc.get("top_logprobs") or []
    rows = []
    for t, tok in enumerate(traj):
        pos = plp[len(prompt_ids) + t]
        if not pos or tok not in pos:
            return {**out, "error": f"reference prompt_logprobs lack the token at position {t}"}
        ref_lp = {k: v[0] for k, v in pos.items()}
        best = argmax_token(pos)
        row = {"t": t, "agree": int(tok == best), "nll": -ref_lp[tok], "kl": None, "kl_at": {}, "tv": None,
               "tv_at": {}, "gap": ref_lp[best] - ref_lp[tok] if tok != best else 0.0}
        if kl_ok and t < len(top):
            eng = {(int(k) if isinstance(k, int) else k): v for k, v in top[t] if v is not None}
            if all(isinstance(k, int) for k in eng):
                cut = {k: (_top(ref_lp, k, tok), _top(eng, k, tok)) for k in kset}
                row["kl_at"] = {k: qm.kl_topk(a, b) for k, (a, b) in cut.items()}
                row["tv_at"] = {k: qm.tv_topk(a, b) for k, (a, b) in cut.items()}
                row["kl"], row["tv"] = row["kl_at"][k_own], row["tv_at"][k_own]
        rows.append(row)
    out["rows"] = rows
    return out


def _units(prompts: list[dict], field: str, bucket: str | None) -> list[tuple[float, int]]:
    """Per-prompt (sum, count) of ``field`` over positions in ``bucket`` (None = all positions)."""
    units = []
    for p in prompts:
        vals = [r[field] for r in p.get("rows", []) if r[field] is not None
                and (bucket is None or qm.bucket_of(r["t"]) == bucket)]
        units.append((float(sum(vals)), len(vals)))
    return units


def _summarize(prompts: list[dict], resamples: int, seed: int) -> dict:
    """Pooled metrics with prompt-level bootstrap CIs, overall and per position bucket."""
    def block(bucket):
        out = {}
        for field, name in (("agree", "top1_agreement"), ("kl", "kl_mean"), ("tv", "tv_mean"),
                            ("nll", "ref_nll_mean")):
            u = _units(prompts, field, bucket)
            n = sum(c for _, c in u)
            out[name] = {"value": qm.pooled(u), "ci95": list(qm.cluster_ci(u, resamples, seed)), "n_tokens": n,
                         "n_prompts": sum(1 for _, c in u if c)}
        nll = out["ref_nll_mean"]
        out["ref_ppl"] = {"value": math.exp(nll["value"]) if nll["value"] is not None else None,
                          "ci95": [math.exp(x) if x is not None else None for x in nll["ci95"]]}
        gaps = [r["gap"] for p in prompts for r in p.get("rows", [])
                if not r["agree"] and (bucket is None or qm.bucket_of(r["t"]) == bucket)]
        out["disagreement_gap_median"] = statistics.median(gaps) if gaps else None
        return out

    res = {"all": block(None)}
    for b in qm.BUCKET_LABELS:
        res[b] = block(b)
    return res


def _divergence(a: dict[str, list], b: dict[str, list]) -> dict:
    common = sorted(set(a) & set(b))
    fds = {pid: first_divergence(a[pid], b[pid]) for pid in common}
    diverged = [d for d in fds.values() if d is not None]
    return {"n_prompts": len(common), "identical": len(common) - len(diverged),
            "first_divergence_median": statistics.median(diverged) if diverged else None,
            "first_divergence_min": min(diverged) if diverged else None, "per_prompt": fds}


@operation("quality_score", ("quality", "score"),
           "Teacher-force engine trajectories on a higher-precision reference (vLLM prompt_logprobs): top-1 "
           "agreement, KL, reference NLL, first divergence per position bucket, vs the A/A noise floor.",
           [Arg("reference_url", "string", "Reference vLLM server (e.g. serving BF16).", required=True),
            Arg("reference_model", "string", "Model name on the reference server.", required=True),
            Arg("trajectories", "array", "Trajectory dirs (runs/<id>/quality/<label>) or their parent.",
                required=True),
            Arg("baseline", "string", "Trajectory dir of the reference's own greedy run (quality trajectory "
                "against the reference): the path first divergence is measured from, and the A/A noise floor "
                "when there is no --baseline-repeat."),
            Arg("baseline_repeat", "string", "A second greedy run of the reference (e.g. after a relaunch): "
                "scored like an engine, it is the A/A noise floor, including run-to-run first divergence."),
            Arg("out", "string", "Output JSON (default: <run-dir>/quality/score.json)."),
            Arg("prompt_logprobs", "integer", "Reference top-k per position (vLLM caps at --max-logprobs).",
                default=20),
            Arg("tokenizer", "string", "HF tokenizer id for the fallback when the reference has no /tokenize "
                "(needs transformers). Default: reference_model."),
            Arg("margin_agreement", "number", "Pre-registered margin: max drop in top-1 agreement vs the floor "
                "(absolute fraction).", default=0.02),
            Arg("margin_kl", "number", "Pre-registered margin: max increase in mean KL vs the floor (nats).",
                default=0.02),
            Arg("margin_tv", "number", "Pre-registered margin: max increase in mean total variation per token vs "
                "the floor (probability mass, 0-1). Without it TV is reported but not judged: it is not a stand-in "
                "for the KL margin (by Pinsker's inequality a KL of 0.02 nats allows a TV of up to 0.1)."),
            Arg("margin_nll", "number", "Pre-registered margin: max increase in reference NLL per token vs the "
                "floor (nats).", default=0.05),
            Arg("resamples", "integer", "Bootstrap resamples.", default=2000),
            Arg("seed", "integer", "Bootstrap seed.", default=0),
            Arg("concurrency", "integer", "Reference requests in flight.", default=1),
            Arg("api_key_env", "string", "Name of the env var holding the reference API key.",
                default="OPENAI_API_KEY"),
            Arg("timeout", "integer", "Seconds per request.", default=600)])
def quality_score(call, reference_url, reference_model, trajectories, baseline, baseline_repeat, out,
                  prompt_logprobs, tokenizer, margin_agreement, margin_kl, margin_tv, margin_nll, resamples, seed,
                  concurrency, api_key_env, timeout):
    if baseline_repeat and not baseline:
        raise OpError("--baseline-repeat needs --baseline")
    base_dirs = [Path(p) for p in (baseline, baseline_repeat) if p]
    for b in base_dirs:
        if not (b / "index.json").is_file():
            raise OpError(f"{b}: not a trajectory directory (no index.json)")
    if baseline_repeat and Path(baseline_repeat).resolve() == Path(baseline).resolve():
        raise OpError("--baseline-repeat must be a second greedy run of the reference, not --baseline again")
    ref = Client(reference_url, api_key_env, timeout)
    pids = _PromptIds(ref, reference_model, tokenizer)
    state, lock, cache = {"k": prompt_logprobs}, threading.Lock(), {}
    dirs = [d for d in _label_dirs(trajectories) if d.resolve() not in {b.resolve() for b in base_dirs}]
    if not dirs:
        raise OpError("no engine trajectories to score besides the baseline")
    run = _run_of(dirs[0])
    records = _Records()
    sets = {d: _load_set(d, records) for d in dirs + base_dirs}
    # every k some set returned: the floor is scored at each, so it can be paired with each engine at its k
    # (text-only trajectories have no lists, k 0, and get no KL or TV)
    ks = {min(_doc_k(doc), prompt_logprobs) for s in sets.values() for doc in s["docs"].values() if _doc_k(doc) > 0}
    limitations, missing = [], []

    def score_set(s: dict) -> list[dict]:
        return _run_pool(lambda doc: _score_one(doc, ref, reference_model, pids, state, lock, cache, ks),
                         list(s["docs"].values()), concurrency, serial_first=True)

    scored = {}
    for d in dirs:
        s = sets[d]
        if s["label"] in scored:
            raise OpError(f"two trajectory dirs have the label {s['label']}")
        scored[s["label"]] = (s, score_set(s))
    floor_label, floor_res, floor_set, path_ids = None, None, None, None
    if baseline:
        base_set = sets[base_dirs[0]]
        # The reference's greedy path (first divergence is measured from it) ...
        path_ids = {pid: d["token_ids"] for pid, d in base_set["docs"].items() if d.get("token_ids")}
        # ... and the A/A floor: the reference's second greedy run scored like an engine when there is one
        # (it also gives the run-to-run first-divergence floor), else its first run.
        floor_set = sets[base_dirs[-1]]
        floor_label, floor_res = floor_set["label"], score_set(floor_set)
        # every engine must be scored on every prompt the reference was run on
        required = set(base_set["ids"])
        if not baseline_repeat:
            limitations.append("noise floor = the baseline scored against its own teacher-forced prefill: it holds "
                               "decode-vs-prefill numerics only, no run-to-run variation, and no first-divergence "
                               "floor (pass --baseline-repeat, a second greedy run of the reference)")
    else:
        limitations.append("no --baseline (the reference's own greedy trajectories): no A/A noise floor and no "
                           "first-divergence measurement, so every verdict is UNPROVEN")
        missing.append("noise floor")
        required = set().union(*(s["ids"] for s, _ in scored.values()))
    floor = floor_res is not None
    # A trajectory without token ids is scored on the reference tokenizer's reading of its text. Where that
    # reading differs from the tokens the model generated, agreement and NLL change, so such prompts are
    # compared with the floor read the same way: the floor's text re-tokenized and scored like the engine's.
    retok = {label: {r["id"] for r in res if r.get("token_ids_source") == "retokenized_text"}
             for label, (_, res) in scored.items()}
    floor_text_res, path_text = None, {}
    if floor and any(retok.values()):
        floor_text_res = score_set({"docs": {pid: _as_text(doc) for pid, doc in floor_set["docs"].items()}})
        if not baseline_repeat:  # the floor is the baseline: its re-tokenized ids are the path too
            path_text = {r["id"]: r["token_ids"] for r in floor_text_res if r.get("token_ids")}
        else:  # first divergence of re-tokenized prompts is measured from the baseline's re-tokenized text
            def text_ids(doc: dict) -> tuple:
                try:
                    return doc["id"], _retokenize(ref, reference_model, doc)
                except OpError:
                    return doc["id"], None

            path_text = {pid: ids for pid, ids in _run_pool(text_ids, list(base_set["docs"].values()), concurrency)
                         if ids}
    floor_text_by_id = {r["id"]: r for r in floor_text_res or []}
    base_problems = [p for b in base_dirs for p in sets[b]["problems"]]
    base_shas = {sets[b]["prompts_sha256"] for b in base_dirs}

    report: dict = {"format": SCORE_FORMAT, "reference": {"url": ref.base, "model": reference_model},
                    "buckets": list(qm.BUCKET_LABELS), "noise_floor": floor_label,
                    "margins": {"agreement": margin_agreement, "kl": margin_kl, "tv": margin_tv, "nll": margin_nll},
                    "kl_method": "top-k union support, tail imputation, OTHER bucket (inferencekit.quality.metrics); "
                                 "both lists cut to the engine's k per position; TV on the same support",
                    "required_prompts": len(required), "engines": {}}
    floor_by_id = {r["id"]: r for r in floor_res} if floor else {}
    floor_gaps, floor_mism = [], False
    if floor:
        report["floor"] = _engine_block(floor_label, floor_res, path_ids if baseline_repeat else None,
                                        resamples, seed)
        report["floor"]["source"] = "baseline_repeat" if baseline_repeat else "baseline"
        fcov = report["floor"]["coverage"] = _coverage(floor_set, floor_res, required)
        floor_mism = bool(report["floor"]["template_mismatch"])
        if fcov["missing"]:
            floor_gaps.append(f"the noise floor lacks {fcov['missing']} of {fcov['required']} prompts")
            limitations.append(_coverage_text(f"{floor_label} (noise floor)", fcov))
    verdicts, metrics, short_k = {}, [], {}
    if floor_text_res is not None:
        report["floor_retokenized"] = _engine_block(f"{floor_label} (re-tokenized)", floor_text_res, None,
                                                    resamples, seed)
    for label, (s, results) in scored.items():
        eng_path = path_ids
        if path_ids is not None and retok[label]:
            eng_path = {**{pid: ids for pid, ids in path_ids.items() if pid not in retok[label]},
                        **{pid: path_text[pid] for pid in retok[label] if pid in path_text}}
        block = _engine_block(label, results, eng_path, resamples, seed)
        cov = block["coverage"] = _coverage(s, results, required)
        v = _verdict(results, floor_by_id, floor_text_by_id, floor, margin_agreement, margin_kl, margin_tv, margin_nll,
                     resamples, seed)
        problems = []
        if s["problems"] or base_problems:
            problems.append("trajectory files, index.json or their evidence record failed the integrity check")
        if cov["missing"]:
            problems.append(f"{cov['missing']} of {cov['required']} prompts not scored")
            limitations.append(_coverage_text(label, cov))
        problems.extend(floor_gaps)
        unpaired = [pid for pid in retok[label] if "rows" not in floor_text_by_id.get(pid, {})] if floor else []
        if unpaired:
            problems.append(f"{len(unpaired)} re-tokenized prompts have no noise floor re-tokenized the same way")
        if block["template_mismatch"] or floor_mism:
            problems.append("chat template mismatch between an engine and the reference")
        if baseline and (s["prompts_sha256"] not in base_shas or len(base_shas) > 1):
            problems.append("trajectories come from a different prompt file than the baseline (prompts_sha256)")
        if problems:  # data problems never yield a PASS
            if v["verdict"] == "PASS":
                v["verdict"] = "UNPROVEN"
                v["reason"] += "; downgraded to UNPROVEN: " + "; ".join(problems)
            else:
                v["reason"] += "; data problems: " + "; ".join(problems)
        block["data_problems"] = problems
        block["verdict"] = v
        verdicts[label] = v["verdict"]
        report["engines"][label] = block
        k_check = (v["checks"].get("kl_mean") or {}).get("top_k")
        if k_check is not None and k_check < prompt_logprobs:
            short_k[label] = k_check
        metrics.append(ev.metric("prompts_scored", cov["scored"], "prompts", "count", n=cov["required"],
                                 subject=label))
        metrics.extend(_metrics_for(label, block))
    if floor:
        metrics.extend(_metrics_for(f"{floor_label} (noise floor)", report["floor"]))
    if any(b["coverage"]["missing"] for b in report["engines"].values()) or floor_gaps:
        missing.append("scored prompts")
    integrity = list(dict.fromkeys(p for s in sets.values() for p in s["problems"]))
    if integrity:
        missing.append("input integrity")
        limitations.extend(integrity[:10])
    mism = sum(len(b["template_mismatch"]) for b in report["engines"].values())
    if mism or floor_mism:
        limitations.append(f"{mism} engine prompt(s){' and the noise floor' if floor_mism else ''} where the prompt "
                           "tokens differ from the reference chat template: the engine did not see the same context")
    for label, k in short_k.items():
        limitations.append(f"{label}: KL compared at top-{k} on both sides (the engine returned only {k} "
                           f"alternatives; requested {prompt_logprobs}): fewer alternatives make KL less sensitive")
    for label, b in report["engines"].items():
        if b["retokenized_prompts"]:
            limitations.append(f"{label}: {b['retokenized_prompts']} of {b['n_scored']} prompts scored from "
                               "re-tokenized text (no engine token ids or logprobs): top-1 agreement and NLL only, "
                               "no KL or TV, compared with the noise floor's text re-tokenized the same way "
                               "(report floor_retokenized); the reference's tokenization of the text can differ from "
                               "the engine's")
    # checks that could not be computed (a required input such as logprobs is missing), per engine: the verdict
    # combines the checks that ran, so a reader applying "required feature missing -> no result" reads this
    not_run = {label: [name for name, c in b["verdict"]["checks"].items()
                       if c["verdict"] == "NOT RUN" and "worse_than_floor" not in c]
               for label, b in report["engines"].items()}
    not_run = {label: names for label, names in not_run.items() if names}
    if any("kl_mean" in names for names in not_run.values()):
        missing.append("logprobs")
    if state["k"] < prompt_logprobs:
        limitations.append(f"reference returned top-{state['k']} prompt logprobs (requested {prompt_logprobs})")
    srcs = sorted({r.get("prompt_ids_source") for _, res in scored.values() for r in res
                   if r.get("prompt_ids_source")})
    if "engine_reported" in srcs:
        limitations.append("prompt ids for some prompts came from the engine itself (no reference /tokenize and "
                           "no HF tokenizer): the chat template was not independently checked")
    limitations.append("vLLM returns raw logprobs (before temperature/penalties) by default on both sides; "
                       "KL and TV on top-k lists are approximations (see data.kl_method)")
    if pids.hf_problem and "hf_tokenizer" not in srcs:
        limitations.append(f"HF tokenizer fallback unavailable: {pids.hf_problem}")
    report["verdicts"] = verdicts
    report["not_run"] = not_run
    report["prompt_ids_sources"] = srcs
    out_path = Path(out) if out else ((run / "quality" / "score.json") if run else Path("quality_score.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    overall = stats.combine_claims(verdicts.values())
    summary = f"{overall}: " + ", ".join(
        f"{label} {verdicts[label]} (agreement {_pct(report['engines'][label]['all']['top1_agreement']['value'])}"
        for label in verdicts) + (")" if verdicts else "") + (
        f"; noise floor {_pct(report['floor']['all']['top1_agreement']['value'])}" if floor else "; no noise floor")
    traj_ev = {d.name: {"prompts_sha256": s["prompts_sha256"], "index_sha256": s["index_sha256"]}
               for d, s in sets.items()}
    if len({v["prompts_sha256"] for v in traj_ev.values()}) > 1:
        limitations.append("trajectory dirs come from different prompt files (prompts_sha256 differs): "
                           "engines are compared on their common prompt ids only")
    return _save(call, run, "quality_trajectory", claim_class="derived", summary=summary,
                 identity_=ev.identity(engine={"name": "reference"}, model={"id": reference_model}),
                 metrics=metrics, artifacts=[ev.artifact(out_path, run, "quality_score")], missing=missing,
                 limitations=limitations,
                 data={"verdict": overall, "verdicts": verdicts, "noise_floor": floor_label,
                       "noise_floor_source": (report.get("floor") or {}).get("source"),
                       "margins": report["margins"], "kl_method": report["kl_method"],
                       "prompt_ids_sources": srcs, "reference_top_k": state["k"], "inputs": traj_ev,
                       "required_prompts": len(required), "not_run": not_run,
                       "text_only": {label: b["retokenized_prompts"] for label, b in report["engines"].items()
                                     if b["retokenized_prompts"]},
                       "coverage": {label: {k: b["coverage"][k] for k in ("scored", "missing", "missing_ids")}
                                    for label, b in report["engines"].items()},
                       "score_file": ev.artifact(out_path, run, "quality_score")["path"]})


def _pct(x) -> str:
    return f"{100 * x:.2f}%" if x is not None else "n/a"


def _engine_block(label, results, path_ids, resamples, seed) -> dict:
    ok = [r for r in results if "rows" in r]
    block = {"label": label, "n_prompts": len(results), "n_scored": len(ok),
             "errors": {r["id"]: r["error"] for r in results if "error" in r},
             "template_mismatch": {r["id"]: r["template_mismatch"] for r in results if r.get("template_mismatch")},
             "kl_top_k": sorted({r["k_kl"] for r in ok if any(x["kl"] is not None for x in r["rows"])}),
             "retokenized_prompts": sum(1 for r in ok if r.get("token_ids_source") == "retokenized_text")}
    block.update(_summarize(ok, resamples, seed))
    by_input = {}
    for b in sorted({r.get("bucket") or "none" for r in ok}):
        sub = [r for r in ok if (r.get("bucket") or "none") == b]
        by_input[b] = _summarize(sub, resamples, seed)["all"]
    block["by_input_bucket"] = by_input
    if path_ids is not None:
        mine = {r["id"]: r["token_ids"] for r in ok}
        block["first_divergence"] = _divergence(path_ids, mine)
    block["per_prompt"] = {r["id"]: {"bucket": r.get("bucket"), "tokens": len(r["rows"]),
                                     "agreement": sum(x["agree"] for x in r["rows"]) / len(r["rows"]),
                                     "nll_mean": sum(x["nll"] for x in r["rows"]) / len(r["rows"])}
                           for r in ok}
    return block


def _verdict(results, floor_by_id, floor_text_by_id, have_floor, m_agree, m_kl, m_tv, m_nll, resamples,
             seed) -> dict:
    if not have_floor:
        return {"verdict": "UNPROVEN", "reason": "no A/A noise floor (pass --baseline)", "checks": {}}
    # each prompt is paired with the floor scored through the same path: a re-tokenized engine prompt with the
    # floor's re-tokenized text, one with token ids with the floor's own ids
    pairs = [(r, (floor_text_by_id if r.get("token_ids_source") == "retokenized_text" else floor_by_id).get(r["id"]))
             for r in results if "rows" in r]
    pairs = [(e, f) for e, f in pairs if f and "rows" in f]
    eng, flo = [e for e, _ in pairs], [f for _, f in pairs]
    # KL depends on how many alternatives each side lists: both arms are compared at the smallest k either returned
    with_kl = [min(e["k_kl"], f["k_kl"]) for e, f in zip(eng, flo)
               if any(x["kl"] is not None for x in e["rows"]) and any(x["kl"] is not None for x in f["rows"])]
    k_pair = min(with_kl) if with_kl else None
    arms = {"agree": (eng, flo), "nll": (eng, flo)}
    if k_pair is not None:
        arms["kl"] = arms["tv"] = (_at_k(eng, k_pair), _at_k(flo, k_pair))
    checks = {}
    for field, name, margin, higher_better in (("agree", "top1_agreement", m_agree, True),
                                               ("kl", "kl_mean", m_kl, False),
                                               ("tv", "tv_mean", m_tv, False),
                                               ("nll", "ref_nll_mean", m_nll, False)):
        e_arm, f_arm = arms.get(field, ([], []))
        e_u, f_u = _units(e_arm, field, None), _units(f_arm, field, None)
        if not any(c for _, c in e_u):
            checks[name] = {"verdict": "NOT RUN", "reason": "not computed for this engine"
                            + (" (no token-id top-k lists on both sides)" if field in ("kl", "tv") else "")}
            continue
        d = qm.paired_delta_ci(e_u, f_u, resamples, seed)
        lo, hi = d["ci95"]
        if higher_better:
            worse = (-d["delta"] if d["delta"] is not None else None, -hi if hi is not None else None,
                     -lo if lo is not None else None)
        else:
            worse = (d["delta"], lo, hi)
        if margin is None:  # reported beside the judged checks, not judged (no pre-registered margin)
            v, why = "NOT RUN", "reported, not judged (no --margin-tv)"
        else:
            v, why = qm.margin_verdict(worse[1], worse[2], margin)
        checks[name] = {"verdict": v, "reason": why, "worse_than_floor": worse[0], "ci95": [worse[1], worse[2]],
                        "margin": margin, "n_prompts": d["n_prompts"]}
        if field in ("kl", "tv"):
            checks[name]["top_k"] = k_pair
    verdict = stats.combine_claims(c["verdict"] for c in checks.values())
    if verdict == "NOT RUN":
        verdict = "UNPROVEN"
    return {"verdict": verdict, "reason": "; ".join(f"{k}: {c['reason']}" for k, c in checks.items()),
            "checks": checks}


def _metrics_for(label: str, block: dict) -> list[dict]:
    out = []
    for bucket in ("all",) + qm.BUCKET_LABELS:
        b = block.get(bucket) or {}
        subj = label if bucket == "all" else f"{label} @pos {bucket}"
        agr, kl, nll = b.get("top1_agreement", {}), b.get("kl_mean", {}), b.get("ref_nll_mean", {})
        tv = b.get("tv_mean", {})
        if not agr.get("n_tokens"):
            continue
        out.append(ev.metric("top1_agreement_pct", 100 * agr["value"], "%", "pooled_mean", n=agr["n_prompts"],
                             ci95=[100 * x if x is not None else None for x in agr["ci95"]], subject=subj))
        if kl.get("value") is not None:
            out.append(ev.metric("kl_ref_engine", kl["value"], "nats/token", "pooled_mean", n=kl["n_prompts"],
                                 ci95=kl["ci95"], subject=subj))
        if tv.get("value") is not None:
            out.append(ev.metric("tv_ref_engine", tv["value"], "probability/token", "pooled_mean",
                                 n=tv["n_prompts"], ci95=tv["ci95"], subject=subj))
        out.append(ev.metric("ref_nll", nll["value"], "nats/token", "pooled_mean", n=nll["n_prompts"],
                             ci95=nll["ci95"], subject=subj))
        if bucket == "all":
            out.append(ev.metric("ref_ppl", b["ref_ppl"]["value"], "ppl", "exp_pooled_mean", n=nll["n_prompts"],
                                 ci95=b["ref_ppl"]["ci95"], subject=subj))
    fd = block.get("first_divergence")
    if fd and fd.get("n_prompts"):
        out.append(ev.metric("identical_to_reference_greedy_pct", 100 * fd["identical"] / fd["n_prompts"], "%",
                             "rate", n=fd["n_prompts"], subject=label))
        if fd["first_divergence_median"] is not None:
            out.append(ev.metric("first_divergence_index", fd["first_divergence_median"], "tokens", "median",
                                 n=fd["n_prompts"] - fd["identical"], subject=label))
    return out


# --------------------------------------------------------------------------- quality gsm8k

# Settings two gsm8k runs must share for McNemar's test to compare the same items the same way.
_PAIR_KEYS = ("dataset_sha256", "prompt", "max_tokens", "temperature", "chat_template_kwargs", "backend")


def _gsm8k_record_for(rec: dict, sha: str) -> bool:
    """``rec`` is a builtin ``quality gsm8k`` record whose results artifact has this sha256."""
    data = rec.get("data") or {}
    return _is_kind(rec, "quality_task") and data.get("task") == "gsm8k" and any(
        a.get("role") == "gsm8k_results" and a.get("sha256") == sha for a in rec.get("artifacts") or [])


def _load_gsm8k_run(ref: str) -> dict:
    """An earlier ``quality gsm8k`` run: {rows, path, settings, evidence_id}.

    ``ref`` is its results JSONL or its evidence id/path. Through an evidence id, the results
    file must still match the record's sha256. Through a JSONL path, the record is looked up
    by the file's sha256 (in that run's ``evidence/``, then the store); with none, ``settings``
    is None (the caller cannot check that both runs used the same dataset, prompt and budget).
    """
    p = Path(ref)
    if p.is_file() and p.suffix == ".jsonl":
        path = p
        sha = file_sha256(path)[0]
        run = path.parent.parent.parent if (path.parent.name == "gsm8k"
                                            and path.parent.parent.name == "quality") else None
        rec, _ = _Records().find(run, lambda r: _gsm8k_record_for(r, sha))
    else:
        rec = ev.Store().load(ref)
        problem = ev.integrity_problem(rec)
        if problem:
            raise OpError(f"paired_with: {problem}")
        rows_file = (rec.get("data") or {}).get("results_file")
        if (rec.get("data") or {}).get("task") != "gsm8k" or not rows_file:
            raise OpError(f"paired_with: {ref} is not a builtin quality gsm8k record")
        path = Path(rows_file).expanduser()
        if not path.is_file():
            raise OpError(f"paired_with: results file {rows_file} not found (run from the same directory)")
        if not _gsm8k_record_for(rec, file_sha256(path)[0]):
            raise OpError(f"paired_with: {path} changed since its evidence record {rec['evidence_id']} (edited?)")
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[int(r["idx"])] = r
    settings = {k: (rec.get("data") or {}).get(k) for k in _PAIR_KEYS} if rec else None
    return {"rows": rows, "path": str(path), "settings": settings,
            "evidence_id": rec.get("evidence_id") if rec else None}


def _lm_eval(base: str, model: str, limit: int, concurrency: int, max_tokens: int, out_dir: Path,
             timeout: int) -> dict:
    """Run lm-evaluation-harness's gsm8k task against the chat endpoint; returns its accuracy numbers."""
    from inferencekit.proc import run as run_proc

    exe = shutil.which("lm_eval") or shutil.which("lm-eval")
    if not exe:
        raise OpError("backend lm-eval: 'lm_eval' not found on PATH (pip install 'lm-eval[api]')")
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = [exe, "--model", "local-chat-completions", "--model_args",
            f"model={model},base_url={base}/v1/chat/completions,num_concurrent={concurrency},max_retries=3,"
            "tokenized_requests=False", "--tasks", "gsm8k", "--apply_chat_template", "--num_fewshot", "0",
            "--limit", str(limit), "--gen_kwargs", f"temperature=0,max_gen_toks={max_tokens}",
            "--output_path", str(out_dir), "--log_samples"]
    res = run_proc(argv, timeout=timeout, stdout_path=out_dir / "lm_eval.stdout",
                   stderr_path=out_dir / "lm_eval.stderr")
    if not res.ok:
        raise OpError(f"lm_eval exited {res.returncode}: {res.stderr[-400:]}")
    found = sorted(out_dir.rglob("results*.json"))
    if not found:
        raise OpError(f"lm_eval wrote no results*.json under {out_dir}")
    return parse_lm_eval_results(json.loads(found[-1].read_text())) | {"results_file": str(found[-1])}


def parse_lm_eval_results(obj: dict) -> dict:
    task = (obj.get("results") or {}).get("gsm8k") or {}
    n = ((obj.get("n-samples") or {}).get("gsm8k") or {}).get("effective")
    return {"strict": task.get("exact_match,strict-match"), "flexible": task.get("exact_match,flexible-extract"),
            "n": n, "lm_eval_version": (obj.get("versions") or {}).get("gsm8k")}


# Auto concurrency never sends more requests at once than the fixed default (32) did.
AUTO_MAX_CONCURRENCY = 32
_BURSTS = (8, 32)  # timing-probe burst sizes; the next one runs only when the server ran the last one in parallel
_SATURATED = 0.75  # a burst counts as run in parallel when parallelism >= this share of its size


def _count(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _server_capacity(client: Client) -> tuple[int | None, str | None, dict]:
    """(requests the server runs at once, source, notes) from the server's own info endpoints.

    SGLang: ``GET /get_server_info`` -> ``max_running_requests`` (top level, or in an
    ``internal_states`` entry). llama.cpp's llama-server: ``GET /props`` -> ``total_slots``.
    vLLM has neither (its ``max_num_seqs`` is not exposed), so it falls through to the probe.
    """
    notes: dict = {}
    try:
        info = client.get("/get_server_info")
    except OpError as e:
        notes["get_server_info"] = str(e)[:160]
    else:
        states = info.get("internal_states") if isinstance(info, dict) else None
        for entry in ([info] if isinstance(info, dict) else []) + [x for x in states or [] if isinstance(x, dict)]:
            n = _count(entry.get("max_running_requests"))
            if n:
                return n, "get_server_info", notes
        notes["get_server_info"] = "answered without a max_running_requests value"
    try:
        props = client.get("/props")
    except OpError as e:
        notes["props"] = str(e)[:160]
    else:
        n = _count(props.get("total_slots")) if isinstance(props, dict) else None
        if n:
            return n, "llamacpp_props", notes
        notes["props"] = "answered without total_slots"
    return None, None, notes


def _timing_probe(client: Client, body: dict) -> tuple[int, dict]:
    """Requests the server serves at once, from timing: one request alone, then bursts of 8 (and 32).

    With t1 the time of one request alone and tb the wall time of b identical requests sent
    together, b * t1 / tb is how many the server ran in parallel (b for a batching server,
    1 for one that serves one at a time). A burst run fully in parallel only shows the server
    can do at least b, so the next, larger burst is tried. Failed requests give 1.
    """
    def timed(_=None) -> float:
        t = time.perf_counter()
        client.post("/v1/chat/completions", body)
        return time.perf_counter() - t

    detail: dict = {"max_tokens": body["max_tokens"], "bursts": []}
    try:
        timed()  # warm-up (lazy init, compilation, prefix cache): not timed
        t1 = min(timed(), timed())
    except OpError as e:
        return 1, {**detail, "error": f"probe request failed: {str(e)[:160]}"}
    detail["t1_s"] = round(t1, 4)
    found = 1
    for size in _BURSTS:
        t = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=size) as pool:
                list(pool.map(timed, range(size)))
        except OpError as e:
            detail["error"] = f"burst of {size} failed: {str(e)[:160]}"
            break
        wall = time.perf_counter() - t
        par = size * t1 / wall if wall > 0 else float(size)
        detail["bursts"].append({"size": size, "wall_s": round(wall, 4), "parallelism": round(par, 2)})
        if par < _SATURATED * size:
            found = round(par)
            break
        found = size
    return max(1, min(found, AUTO_MAX_CONCURRENCY)), detail


def _auto_concurrency(base_url: str, api_key_env: str, timeout: int, body: dict) -> dict:
    """``{used, source, detail}``: the server's own capacity when it reports one, else the timing probe."""
    info_client = Client(base_url, api_key_env, min(timeout, 30))
    n, source, notes = _server_capacity(info_client)
    if n:
        return {"used": min(n, AUTO_MAX_CONCURRENCY), "source": source,
                "detail": {"server_reported": n, **({"capped_at": AUTO_MAX_CONCURRENCY}
                                                    if n > AUTO_MAX_CONCURRENCY else {})}}
    used, detail = _timing_probe(Client(base_url, api_key_env, timeout), body)
    return {"used": used, "source": "timing_probe", "detail": {**detail, "server_info": notes}}


def _run_budgeted(fn, items: list, concurrency: int, deadline: float) -> tuple[list, dict]:
    """``fn`` over ``items`` in order with up to ``concurrency`` in flight, stopping at a time budget.

    A slot that frees up gets the next item only while ``now + mean latency so far`` is
    before ``deadline`` (``time.monotonic()``): an item sent now is projected to end one
    mean latency later. Once one item is held back, no later one is sent, so the results are
    always the first n items. Items already in flight finish, so the run can overrun the
    budget by about one request. The full run's projection uses Little's law: with c
    requests in flight and mean latency L, items finish at c / L per second.
    """
    workers = max(1, concurrency)
    results: dict = {}
    latencies: list[float] = []
    nxt, stopped, t0 = 0, False, time.monotonic()

    def timed(item):
        t = time.monotonic()
        out = fn(item)
        return out, time.monotonic() - t

    with ThreadPoolExecutor(max_workers=workers) as pool:
        running: dict = {}
        while True:
            while not stopped and nxt < len(items) and len(running) < workers:
                if latencies and time.monotonic() + statistics.fmean(latencies) > deadline:
                    stopped = True
                    break
                running[pool.submit(timed, items[nxt])] = nxt
                nxt += 1
            if not running:
                break
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for f in done:
                i = running.pop(f)
                results[i], dt = f.result()
                latencies.append(dt)
    mean = statistics.fmean(latencies) if latencies else None
    info = {"issued": nxt, "of": len(items), "stopped_early": stopped, "elapsed_s": round(time.monotonic() - t0, 1),
            "mean_latency_s": round(mean, 3) if mean is not None else None,
            "projected_full_s": round(len(items) * mean / workers, 1) if mean is not None else None}
    return [results[i] for i in range(nxt)], info


@operation("quality_gsm8k", ("quality", "gsm8k"),
           "GSM8K test split (MIT; downloaded once from Hugging Face at a pinned revision, sha256-checked, "
           "cached in ~/.cache/inferencekit), zero-shot chain of thought at temperature 0 over the chat API: "
           "accuracy with a Wilson CI; --paired-with adds McNemar's exact test against another run.",
           [Arg("base_url", "string", "Engine server, e.g. http://127.0.0.1:8000.", required=True),
            Arg("model", "string", "Model name the server expects.", required=True),
            Arg("run_dir", "string", "Run directory (runs/<run-id>).", required=True),
            Arg("label", "string", "Name of this engine/configuration (file name).", default="engine"),
            Arg("limit", "integer", "First N problems of the test split (1319 = all).", default=1319),
            Arg("concurrency", "integer", "Requests in flight; 0 = auto: the server's own capacity (SGLang "
                "/get_server_info max_running_requests, llama.cpp /props total_slots), else a timing probe "
                f"(one request alone, then bursts of 8 and 32); at most {AUTO_MAX_CONCURRENCY}. The value used "
                "and its source are recorded.", default=32),
            Arg("time_budget_s", "integer", "Wall-clock budget in seconds, the auto-concurrency probe included: "
                "once the problems done so far project that the next one would end past it, no new problems are "
                "sent and the record covers the first n (a 'subset' limitation). Default: no budget."),
            Arg("max_tokens", "integer", "Fixed generation budget per problem.", default=1024),
            Arg("chat_template_kwargs", "string", "JSON object passed as chat_template_kwargs."),
            Arg("paired_with", "string", "Results JSONL or evidence id of another gsm8k run on the same items."),
            Arg("backend", "string", "builtin, or lm-eval (lm-evaluation-harness local-chat-completions; must "
                "be installed).", default="builtin", choices=("builtin", "lm-eval")),
            Arg("data", "string", "A different GSM8K-format JSONL (question, answer with '#### n'), or a Parquet "
                "file with those columns, instead of the pinned test split."),
            Arg("seed", "integer", "Request seed.", default=0),
            Arg("api_key_env", "string", "Name of the env var holding the API key.", default="OPENAI_API_KEY"),
            Arg("timeout", "integer", "Seconds per request (lm-eval: for the whole run).", default=600)])
def quality_gsm8k(call, base_url, model, run_dir, label, limit, concurrency, time_budget_s, max_tokens,
                  chat_template_kwargs, paired_with, backend, data, seed, api_key_env, timeout):
    t_start = time.monotonic()
    label = _safe(label, "label")
    if concurrency < 0:
        raise OpError("concurrency must be >= 1, or 0 for auto")
    if time_budget_s is not None and time_budget_s <= 0:
        raise OpError("time_budget_s must be > 0 seconds")
    if time_budget_s is not None and backend == "lm-eval":
        raise OpError("--time-budget-s needs the builtin backend (lm_eval runs its own request loop)")
    kwargs = apply_pin(_json_arg("chat_template_kwargs", chat_template_kwargs), run_dir, "quality gsm8k") or None
    if kwargs and backend == "lm-eval":
        # lm_eval's local-chat-completions has no way to send chat_template_kwargs: the record would claim a
        # template setting the engine never received.
        raise OpError("backend lm-eval cannot send chat_template_kwargs (given, or pinned by the intake); use the "
                      "builtin backend, which sends them on every request")
    run = Path(run_dir)
    out_dir = run / "quality" / "gsm8k"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = Client(base_url, api_key_env, timeout)
    items, data_sha = g8.load(data, limit)
    common = {"label": label, "dataset": g8.DATASET if not data else "custom", "dataset_sha256": data_sha,
              "backend": backend, "prompt": g8.INSTRUCTION, "max_tokens": max_tokens, "temperature": 0,
              "chat_template_kwargs": kwargs}
    other = None
    if paired_with:  # checked before any request, so a mismatch costs nothing
        if backend == "lm-eval":
            raise OpError("--paired-with needs the builtin backend (lm-eval uses its own prompt and extraction)")
        other = _load_gsm8k_run(paired_with)
        if other["settings"] is not None:
            diff = [k for k in _PAIR_KEYS if other["settings"].get(k) != common.get(k)]
            if diff:
                raise OpError("paired_with: not the same evaluation, so McNemar's test would compare different "
                              "items; it differs in " + ", ".join(
                                  f"{k} ({str(other['settings'].get(k))[:60]!r} there, {str(common.get(k))[:60]!r} "
                                  "here)" for k in diff))
    if concurrency == 0 and items:
        probe = {"model": model, "messages": g8.messages(items[0]["question"]), "max_tokens": 32, "temperature": 0,
                 "seed": seed, "stream": False}
        if kwargs:
            probe["chat_template_kwargs"] = kwargs
        conc = {"requested": 0, **_auto_concurrency(base_url, api_key_env, timeout, probe)}
    else:
        conc = {"requested": concurrency, "used": max(1, concurrency), "source": "flag", "detail": {}}
    concurrency = conc["used"]
    # concurrency_source is kept by evidence.identity once the workload schema lists it (data.concurrency has it now)
    ident = ev.identity(engine={"name": label}, model={"id": model},
                        workload={"gen_len": max_tokens, "batch": concurrency, "concurrency": concurrency,
                                  "concurrency_source": conc["source"], "seed": seed})
    if backend == "lm-eval":
        r = _lm_eval(client.base, model, len(items), concurrency, max_tokens, out_dir / f"{label}.lm_eval", timeout)
        acc, n = r["flexible"], r["n"] or len(items)
        ci = qm.wilson_ci(round(acc * n), n) if acc is not None else (None, None)
        return _save(call, run, "quality_task", claim_class="observed",
                     summary=f"{label}: GSM8K (lm-eval) flexible-extract {_pct(acc)}, strict {_pct(r['strict'])}, "
                             f"n={n}",
                     identity_=ident,
                     metrics=[ev.metric("gsm8k_accuracy_pct", 100 * acc if acc is not None else None, "%", "rate",
                                        n=n, ci95=[100 * x if x is not None else None for x in ci], subject=label),
                              ev.metric("gsm8k_strict_accuracy_pct",
                                        100 * r["strict"] if r["strict"] is not None else None, "%", "rate", n=n,
                                        subject=label)],
                     artifacts=[ev.artifact(r["results_file"], run, "lm_eval_results")],
                     limitations=["lm-eval backend: its own prompt and extraction (flexible-extract / strict-match), "
                                  "not comparable with the builtin backend; McNemar pairing not supported"],
                     data={**common, "task": "gsm8k", "lm_eval": r, "extraction": "lm-eval filters",
                           "concurrency": conc})

    def one(item: dict) -> dict:
        body = {"model": model, "messages": g8.messages(item["question"]), "max_tokens": max_tokens,
                "temperature": 0, "seed": seed, "stream": False}
        if kwargs:
            body["chat_template_kwargs"] = kwargs
        row = {"idx": item["idx"], "gold": item["gold"]}
        try:
            resp = client.post("/v1/chat/completions", body)
        except OpError as e:
            return {**row, "error": str(e)[:300], "extracted": None, "rule": "none", "correct": False}
        choice = (resp.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        got, rule = g8.extract(text)
        return {**row, "extracted": got, "rule": rule, "correct": g8.is_correct(got, item["gold"]),
                "finish_reason": choice.get("finish_reason"),
                "completion_tokens": (resp.get("usage") or {}).get("completion_tokens"), "response": text}

    budget = None
    if time_budget_s is not None:
        rows, budget = _run_budgeted(one, items, concurrency, t_start + time_budget_s)
        budget["budget_s"] = time_budget_s
        rows = sorted(rows, key=lambda r: r["idx"])
    else:
        rows = sorted(_run_pool(one, items, concurrency), key=lambda r: r["idx"])
    results_path = out_dir / f"{label}.jsonl"
    results_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    n = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    errors = sum(1 for r in rows if "error" in r)
    truncated = sum(1 for r in rows if r.get("finish_reason") == "length")
    rules = {k: sum(1 for r in rows if r["rule"] == k) for k in ("hash", "boxed", "last", "none")}
    acc = correct / n if n else None
    ci = qm.wilson_ci(correct, n)
    metrics = [ev.metric("gsm8k_accuracy_pct", 100 * acc if acc is not None else None, "%", "rate", n=n,
                         ci95=[100 * x if x is not None else None for x in ci], subject=label),
               ev.metric("truncated_at_max_tokens", truncated, "items", "count", n=n, subject=label)]
    limitations, missing = [], []
    if errors:
        missing.append("responses")
        limitations.append(f"{errors} request(s) failed and count as wrong")
    if truncated:
        limitations.append(f"{truncated} response(s) hit max_tokens={max_tokens} (answer may be cut off)")
    stopped = bool(budget and budget["stopped_early"])
    if stopped:
        limitations.append(f"subset: first {n} of {len(items)} problems{'' if data else ' of the test split'}: "
                           f"--time-budget-s {time_budget_s} stopped new requests (all {len(items)} were projected "
                           f"to take about {budget['projected_full_s']:.0f} s at concurrency {concurrency})")
    elif n < 1319 and not data:
        limitations.append(f"subset: first {n} of 1319 test problems")
    if conc["source"] == "timing_probe" and conc["detail"].get("error"):
        limitations.append(f"auto concurrency: {conc['detail']['error']}; ran with {concurrency}")
    paired = None
    if other is not None:
        other_path, checked = other["path"], other["settings"] is not None
        other = other["rows"]
        common_idx = [r["idx"] for r in rows if r["idx"] in other]
        mine = {r["idx"]: r for r in rows}
        b = sum(1 for i in common_idx if mine[i]["correct"] and not other[i]["correct"])
        c = sum(1 for i in common_idx if not mine[i]["correct"] and other[i]["correct"])
        both = sum(1 for i in common_idx if mine[i]["correct"] and other[i]["correct"])
        m = len(common_idx)
        p = qm.mcnemar_exact(b, c)
        delta = (b - c) / m if m else None
        paired = {"other": other_path, "n_paired": m, "only_this_correct": b, "only_other_correct": c,
                  "both_correct": both, "accuracy_delta": delta, "mcnemar_exact_p": p, "settings_checked": checked}
        if not checked:
            limitations.append(f"paired run {other_path}: its settings (dataset, prompt, max_tokens, chat template) "
                               "could not be checked: no gsm8k evidence record matches that file")
        metrics.append(ev.metric("gsm8k_accuracy_delta_pct", 100 * delta if delta is not None else None, "%",
                                 "paired_difference", n=m, subject=label))
        metrics.append(ev.metric("mcnemar_exact_p", p, "p", "two_sided", n=b + c, subject=label))
        if m < n:
            limitations.append(f"paired on {m} of {n} items (the other run has fewer)")
    summary = (f"{label}: GSM8K {correct}/{n} = {_pct(acc)} (95% Wilson CI {_pct(ci[0])}-{_pct(ci[1])})"
               + (f"; vs other run {paired['accuracy_delta'] * 100:+.2f} pts, McNemar p={paired['mcnemar_exact_p']:.3g}"
                  if paired and paired["accuracy_delta"] is not None else "")
               + (f"; time budget: first {n} of {len(items)}" if stopped else "")
               + (f"; concurrency {concurrency} ({conc['source']})" if conc["source"] != "flag" else ""))
    return _save(call, run, "quality_task", claim_class="observed", summary=summary, identity_=ident,
                 metrics=metrics, artifacts=[ev.artifact(results_path, run, "gsm8k_results")], missing=missing,
                 limitations=limitations,
                 data={**common, "task": "gsm8k", "n": n, "correct": correct, "errors": errors,
                       "extraction": "#### number, else last \\boxed{}, else last number; numeric equality",
                       "extraction_rules_used": rules, "paired": paired, "results_file": str(results_path),
                       "concurrency": conc, "time_budget": budget})


# --------------------------------------------------------------------------- quality niah

def _chars_per_token(client: Client, model: str) -> tuple[float, str]:
    import random as _random

    from inferencekit.quality import niah

    rng = _random.Random(0)
    sample = " ".join(niah.filler_sentence(rng) for _ in range(60))
    try:
        n = len(client.post("/tokenize", {"model": model, "prompt": sample, "add_special_tokens": False})
                .get("tokens") or [])
    except OpError:
        return 4.0, "assumed 4.0 chars/token (no /tokenize)"
    return (len(sample) / n, "measured with /tokenize") if n else (4.0, "assumed 4.0 chars/token")


@operation("quality_niah", ("quality", "niah"),
           "Needle in a haystack: retrieve a 7-digit code from synthetic filler at several context lengths and "
           "depths; accuracy per (length, depth) cell with Wilson CIs.",
           [Arg("base_url", "string", "Engine server, e.g. http://127.0.0.1:8000.", required=True),
            Arg("model", "string", "Model name the server expects.", required=True),
            Arg("run_dir", "string", "Run directory (runs/<run-id>).", required=True),
            Arg("label", "string", "Name of this engine/configuration (file name).", default="engine"),
            Arg("lengths", "array", "Target context lengths in tokens.", items="integer", default=[2048, 4096, 8192]),
            Arg("depths", "array", "Needle depths (0 = start, 1 = end).", items="number", default=[0.1, 0.5, 0.9]),
            Arg("trials", "integer", "Haystacks (different needles) per cell.", default=5),
            Arg("max_tokens", "integer", "Reply budget.", default=32),
            Arg("concurrency", "integer", "Requests in flight.", default=8),
            Arg("chat_template_kwargs", "string", "JSON object passed as chat_template_kwargs."),
            Arg("seed", "integer", "Seed for haystacks and requests.", default=0),
            Arg("api_key_env", "string", "Name of the env var holding the API key.", default="OPENAI_API_KEY"),
            Arg("timeout", "integer", "Seconds per request.", default=600)])
def quality_niah(call, base_url, model, run_dir, label, lengths, depths, trials, max_tokens, concurrency,
                 chat_template_kwargs, seed, api_key_env, timeout):
    from inferencekit.quality import niah

    label = _safe(label, "label")
    kwargs = apply_pin(_json_arg("chat_template_kwargs", chat_template_kwargs), run_dir, "quality niah") or None
    run = Path(run_dir)
    out_dir = run / "quality" / "niah"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = Client(base_url, api_key_env, timeout)
    cpt, cpt_method = _chars_per_token(client, model)
    items = [{**niah.build(n, d, seed * 1000 + t, cpt), "trial": t}
             for n in lengths for d in depths for t in range(trials)]

    def one(item: dict) -> dict:
        body = {"model": model, "messages": item["messages"], "max_tokens": max_tokens, "temperature": 0,
                "seed": seed, "stream": False}
        if kwargs:
            body["chat_template_kwargs"] = kwargs
        row = {k: item[k] for k in ("target_tokens", "depth", "trial", "code", "needle_sentence_index")}
        row["messages_digest"] = digest(item["messages"])
        try:
            resp = client.post("/v1/chat/completions", body)
        except OpError as e:
            return {**row, "error": str(e)[:300], "correct": False}
        text = (((resp.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
        return {**row, "reply": text[:500], "correct": niah.found(text, item["code"]),
                "prompt_tokens": (resp.get("usage") or {}).get("prompt_tokens")}

    rows = _run_pool(one, items, concurrency)
    path = out_dir / f"{label}.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    cells, metrics = [], []
    for n in lengths:
        for d in depths:
            cell = [r for r in rows if r["target_tokens"] == n and r["depth"] == d]
            k = sum(1 for r in cell if r["correct"])
            lo, hi = qm.wilson_ci(k, len(cell))
            actual = [r["prompt_tokens"] for r in cell if r.get("prompt_tokens")]
            cells.append({"target_tokens": n, "depth": d, "n": len(cell), "correct": k,
                          "prompt_tokens_median": statistics.median(actual) if actual else None})
            metrics.append(ev.metric("niah_accuracy_pct", 100 * k / len(cell) if cell else None, "%", "rate",
                                     n=len(cell), ci95=[100 * x if x is not None else None for x in (lo, hi)],
                                     subject=f"{label} @{n} tok, depth {d:g}"))
    total = sum(1 for r in rows if r["correct"])
    lo, hi = qm.wilson_ci(total, len(rows))
    metrics.insert(0, ev.metric("niah_accuracy_pct", 100 * total / len(rows) if rows else None, "%", "rate",
                                n=len(rows), ci95=[100 * x if x is not None else None for x in (lo, hi)],
                                subject=label))
    errors = sum(1 for r in rows if "error" in r)
    limitations = ["synthetic filler: tests retrieval of one planted fact, not long-context reasoning",
                   f"context lengths are approximate ({cpt_method}); see prompt_tokens_median per cell"]
    if errors:
        limitations.append(f"{errors} request(s) failed and count as wrong")
    ident = ev.identity(engine={"name": label}, model={"id": model},
                        workload={"gen_len": max_tokens, "batch": concurrency, "seed": seed})
    return _save(call, run, "quality_task", claim_class="observed",
                 summary=f"{label}: needle retrieval {total}/{len(rows)} = {_pct(total / len(rows) if rows else None)} "
                         f"over lengths {lengths} x depths {depths}",
                 identity_=ident, metrics=metrics, artifacts=[ev.artifact(path, run, "niah_results")],
                 missing=["responses"] if errors else [], limitations=limitations,
                 data={"task": "niah", "label": label, "cells": cells, "chars_per_token": cpt,
                       "chars_per_token_method": cpt_method, "errors": errors, "results_file": str(path),
                       "chat_template_kwargs": kwargs})


# --------------------------------------------------------------------------- quality sampling

SAMPLING_FORMAT = "inferencekit.quality.sampling.v2"
# Request fields the check sets itself; the same keys in --extra-body are dropped (and listed).
_SAMPLING_OWN_KEYS = ("model", "messages", "max_tokens", "max_completion_tokens", "seed", "stream", "n", "logprobs",
                      "top_logprobs", "return_token_ids", "return_tokens_as_token_ids")
# Fields that change the first-token distribution in ways the check does not model: a value other than the
# neutral one makes the verdict UNPROVEN, never PASS or FAIL. presence_penalty and frequency_penalty act on
# generated tokens only: none yet.
_UNMODELLED = {"repetition_penalty": (1, 1.0), "typical_p": (1, 1.0), "min_tokens": (0,), "logit_bias": ({},),
               "allowed_token_ids": ([],), "bad_words": ([],), "logits_processors": ([],), "guided_json": (),
               "guided_regex": (), "guided_choice": ([],), "guided_grammar": (), "structured_outputs": ({},),
               "response_format": ({"type": "text"},)}
_TOP_LOGPROBS = 20
_MIN_PROMPTS = 8  # informative prompts needed: a percentile bootstrap over fewer prompts covers poorly
_NEAR_DETERMINISTIC = 0.98  # implied top-token probability from which a prompt adds nothing to TV (support only)
_OOS_MAX = 0.005  # share of draws outside the kept tokens above which (Wilson 95% lower bound) the sampler FAILs
_SEED_CHECK_PROMPTS = 8  # prompts re-asked with seeds already used, to see if a seed reproduces its draw
_SEED_CHECK_SEEDS = 8
_SEED_MATCH_MIN = 0.9  # share of repeats that must reproduce their first draw
_SEED_INCONCLUSIVE = 0.8  # when draws would repeat this often by chance alone, the repeat check says nothing
_DROPPED = {"incomplete": "a request failed (after a retry)",
            "logprobs_unstable": "the top-20 token ids changed between two identical requests",
            "beyond_top": "the kept tokens may reach past the listed top 20",
            "boundary": "a top-k, top-p or min-p cut within the rounding tolerance",
            "min_p_order": "min-p before and after top-p keep different tokens",
            "top_p_order": "top-p over the top-k tokens (Hugging Face, vLLM) and over the whole distribution "
                           "(SGLang) keep different tokens",
            "bad_logprobs": "no usable log-probabilities",
            "near_deterministic": f"top token at {_NEAR_DETERMINISTIC:.0%} or more (support check only)"}


def _first_token(resp: dict) -> tuple[int | None, str] | None:
    """The first generated token at max_tokens 1: (its id, or None, and its decoded text).

    The id is vLLM's ``choices[0].token_ids`` (asked for with ``return_token_ids``); the text is
    the reply, or the reasoning text when the content is empty. None when the response has no
    ``choices[0].message`` (a malformed answer, counted as an error).
    """
    choice = (resp.get("choices") or [None])[0] if isinstance(resp, dict) else None
    msg = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(msg, dict):
        return None
    ids = choice.get("token_ids")
    tid = ids[0] if isinstance(ids, list) and ids and type(ids[0]) is int else None
    text = msg.get("content") or ""
    if not text:
        for key in ("reasoning_content", "reasoning"):
            if msg.get(key):
                text = "reasoning:" + str(msg[key])
                break
    return tid, str(text)


def _first_top(resp: dict) -> tuple[list | None, str | None]:
    """The first position's listed alternatives [(id or None, text or None, logprob)], or (None, why not)."""
    from inferencekit.quality.logprobs import _entry_id

    choice = (resp.get("choices") or [None])[0] if isinstance(resp, dict) else None
    lps = choice.get("logprobs") if isinstance(choice, dict) else None
    content = lps.get("content") if isinstance(lps, dict) else None
    if not isinstance(content, list) or not content or not isinstance(content[0], dict):
        return None, "the response has no logprobs"
    alts = content[0].get("top_logprobs")
    if not isinstance(alts, list) or not alts or not all(isinstance(a, dict) for a in alts):
        return None, "the response has no top_logprobs at the first position"
    out = []
    for a in alts:
        tid, _ = _entry_id(a)
        tok = a.get("token")
        text = str(tok) if isinstance(tok, str) and not tok.startswith("token_id:") else None
        out.append((tid, text, a.get("logprob")))
    return out, None


def _text_bin(text: str) -> str:
    """The text a first token is binned by when ids are missing.

    Detokenizers differ in a leading space (one-token decode vs decode in context) and in a
    partial UTF-8 character ('' or U+FFFD); both are removed from the draws and the listed tokens
    alike. One map applied to both sides can only merge bins, which can only lower TV.
    """
    return text.replace("�", "").strip()


def _collision(hist: Counter) -> float:
    """Probability that two draws from this histogram's distribution are the same token."""
    n = sum(hist.values())
    return sum((c / n) ** 2 for c in hist.values()) if n else 1.0


def _unmodelled(extra: dict) -> list[str]:
    return sorted(k for k, neutral in _UNMODELLED.items() if k in extra and extra[k] is not None
                  and extra[k] not in neutral)


class _Side:
    """One server's part of the check: per prompt a top-20 logprob request before and after S seeded draws."""

    def __init__(self, name: str, client: Client, model: str, extra: dict, kwargs_for, concurrency: int):
        self.name, self.client, self.model, self.extra = name, client, model, extra
        self.kwargs_for, self.concurrency = kwargs_for, concurrency
        self.lp: dict[str, list] = {}  # prompt id -> [first pass, second pass], each a listed-token list or None
        self.draws: dict[str, dict[int, tuple]] = {}  # prompt id -> {seed: (token id or None, text)}
        self.errors: list[str] = []
        self.aborted: str | None = None
        self.requests = 0
        self.seed_check: dict = {}
        # vLLM's id fields: token ids for the draws (return_token_ids) and the listed tokens
        # (return_tokens_as_token_ids); dropped, in that order of preference, if the first request is refused
        self.fields = {"return_tokens_as_token_ids": True, "return_token_ids": True}
        self.refused: str | None = None
        self.k_state = {"k": _TOP_LOGPROBS}
        self.lock = threading.Lock()
        self.by_id = True  # bins: token ids when every draw and listed token has one (set by bins())
        self.unmatched: str | None = None

    def _body(self, item: dict, seed: int, logprobs: bool) -> dict:
        body = {**self.extra, "model": self.model, "messages": item["messages"], "max_tokens": 1, "seed": seed,
                "stream": False}
        if self.fields["return_token_ids"]:
            body["return_token_ids"] = True
        if logprobs:
            body["logprobs"] = True
            if self.fields["return_tokens_as_token_ids"]:
                body["return_tokens_as_token_ids"] = True
        kwargs = self.kwargs_for(item)
        if kwargs:
            body["chat_template_kwargs"] = kwargs
        return body

    def _post(self, item: dict, seed: int, logprobs: bool) -> tuple:
        """(response, None), or (None, error) after one retry: a transient failure should not cost the prompt."""
        err = None
        for _ in range(2):
            with self.lock:
                self.requests += 1
            try:
                if logprobs:
                    resp, _ = _post_capped(self.client, "/v1/chat/completions", self._body(item, seed, True),
                                           "top_logprobs", self.k_state, self.lock)
                else:
                    resp = self.client.post("/v1/chat/completions", self._body(item, seed, False))
            except OpError as e:
                err = str(e)[:300]
                continue
            return resp, None
        return None, err

    def _logprobs(self, item: dict) -> tuple:
        resp, err = self._post(item, 0, True)
        if err is not None:
            return item["id"], None, err
        top, why = _first_top(resp)
        return item["id"], top, why

    def _draw(self, job) -> tuple:
        item, seed = job
        resp, err = self._post(item, seed, False)
        if err is not None:
            return item["id"], seed, None, err
        tok = _first_token(resp)
        if tok is None:
            return item["id"], seed, None, "response has no choices[0].message"
        return item["id"], seed, tok, None

    def _pass(self, items: list[dict], slot: int) -> None:
        for pid, top, err in _run_pool(self._logprobs, items, self.concurrency):
            self.lp.setdefault(pid, [None, None])[slot] = top
            if err is not None:
                self.errors.append(f"{pid} logprobs: {err}")

    def run(self, items: list[dict], seeds: range) -> None:
        """Logprobs (first pass), the seeded draws, logprobs again (second pass), then the seed check."""
        first = items[0]
        pid, top, err = self._logprobs(first)  # alone first: a refused field, and the logprob cap, settle here
        for drop in ("return_tokens_as_token_ids", "return_token_ids"):
            if top is not None or err is None or "no top_logprobs" in err or "no logprobs" in err:
                break
            self.fields[drop] = False  # perhaps only vLLM's id field is refused: ask once more without it
            pid, top, again = self._logprobs(first)
            if top is not None:
                self.refused = err[:200]
            err = again
        if top is None:
            self.aborted = (err or "no logprobs")[:300]
            self.errors.append(self.aborted)
            return
        self.lp[pid] = [top, None]
        self._pass(items[1:], 0)
        self.draws = {it["id"]: {} for it in items}
        for pid, seed, tok, err in _run_pool(self._draw, [(it, s) for it in items for s in seeds], self.concurrency):
            if err is None:
                self.draws[pid][seed] = tok
            else:
                self.errors.append(f"{pid} seed {seed}: {err}")
        self._pass(items, 1)
        self.bins()
        self.check_seeds(items, seeds)

    def bins(self) -> None:
        """Token ids when every draw and every listed token has one; else text, which every one must have."""
        drawn = [d for got in self.draws.values() for d in got.values()]
        listed = [e for pair in self.lp.values() for top in pair if top for e in top]
        self.by_id = bool(drawn) and all(d[0] is not None for d in drawn) and all(e[0] is not None for e in listed)
        if not self.by_id and any(e[1] is None for e in listed):
            self.unmatched = ("draws and listed tokens cannot be matched: the listed tokens came as ids only "
                              "(return_tokens_as_token_ids) and not every draw had an id")

    def key(self, entry) -> object:
        return entry[0] if self.by_id else _text_bin(entry[1] or "")

    def hist(self, pid: str, seeds) -> Counter:
        got = self.draws.get(pid, {})
        return Counter(self.key(got[s]) for s in seeds if s in got)

    def check_seeds(self, items: list[dict], seeds: range) -> None:
        """Re-ask a few (prompt, seed) pairs already drawn: a server that honours the seed gives the same token.

        Information only (it never blocks a PASS: unseeded draws are still draws from the distribution). The
        prompts with the most varied first token are used, so a chance repeat is unlikely; the repeat rate
        expected if the seed were ignored (the collision probability) is recorded.
        """
        complete = [it for it in items if all(s in self.draws.get(it["id"], {}) for s in seeds)]
        coll = {it["id"]: _collision(self.hist(it["id"], seeds)) for it in complete}
        pick = sorted(complete, key=lambda it: (coll[it["id"]], it["id"]))[:_SEED_CHECK_PROMPTS]
        again = list(seeds)[:_SEED_CHECK_SEEDS]
        results = _run_pool(self._draw, [(it, s) for it in pick for s in again], self.concurrency)
        ok = [(pid, s, tok) for pid, s, tok, err in results if err is None]
        same = sum(1 for pid, s, tok in ok if self.key(tok) == self.key(self.draws[pid][s]))
        chance = statistics.fmean(coll[it["id"]] for it in pick) if pick else 1.0
        if not ok:
            status = "not checked"
        elif chance > _SEED_INCONCLUSIVE:
            status = "inconclusive"
        else:
            status = "honoured" if same >= _SEED_MATCH_MIN * len(ok) else "ignored"
        self.seed_check = {"status": status, "repeats": len(ok), "same": same,
                           "expected_same_if_ignored": round(chance, 4), "prompts": [it["id"] for it in pick],
                           "seeds": again}


def _pairs(side: _Side, top: list) -> list[tuple]:
    return [(side.key(e), e[2]) for e in top]


def _judge(side: _Side, chosen: list[dict], cfg: dict, seeds: int, margin: float, resamples: int,
           bootstrap_seed: int, caps: list[str], unsent: list[str]) -> dict:
    """Per-prompt rows (draws vs the distribution the server's own logprobs imply) and this server's verdict.

    ``caps``: settings the check does not compute; any makes the verdict UNPROVEN (the statistic is still
    reported). ``unsent``: sampling fields read as off because they were not sent, named in a FAIL's reason."""
    rows, dropped = [], Counter()
    for item in chosen:
        pid = item["id"]
        row: dict = {"id": pid, "bucket": item.get("bucket")}
        first, second = side.lp.get(pid) or (None, None)
        got = side.draws.get(pid, {})
        if side.aborted or side.unmatched or first is None or second is None or len(got) < seeds:
            row["status"] = "incomplete"
        else:
            a, b = _pairs(side, first), _pairs(side, second)
            row["listed"] = [[k, v] for k, v in a]
            if Counter(k for k, _ in a) != Counter(k for k, _ in b):
                row["status"] = "logprobs_unstable"
                row["listed_again"] = [[k, v] for k, v in b]
            else:
                va, vb = {}, {}
                for k, v in a:
                    va.setdefault(k, []).append(v)
                for k, v in b:
                    vb.setdefault(k, []).append(v)
                try:
                    delta = max(abs(x - y) for k in va for x, y in zip(sorted(va[k]), sorted(vb[k])))
                except TypeError:  # a logprob that is not a number
                    delta = 0.0
                tol = max(sm.LOGPROB_TOL, delta)
                row.update(logprob_repeat_max_delta=delta, tolerance=tol)
                imp = sm.implied(a, cfg["temperature"], cfg["top_k"], cfg["top_p"], cfg["min_p"], tol)
                if imp["status"] != "ok":
                    row.update(status=imp["status"], detail=imp["detail"])
                else:
                    probs = imp["probs"]
                    hist = side.hist(pid, range(seeds))
                    listed = {k for k, _ in a}
                    tv, expected = sm.tv_to(hist, probs), sm.expected_tv(probs, seeds)
                    if cfg["temperature"] >= sm.GREEDY_T and cfg["temperature"] != 1.0:
                        # diagnosis only: the same logprobs read as already temperature-scaled
                        alt = sm.implied(a, 1.0, cfg["top_k"], cfg["top_p"], cfg["min_p"], tol)
                        if alt["status"] == "ok":
                            row["excess_if_tempered"] = (sm.tv_to(hist, alt["probs"])
                                                         - sm.expected_tv(alt["probs"], seeds))
                    row.update(status="informative" if imp["max_p"] < _NEAR_DETERMINISTIC else "near_deterministic",
                               max_p=imp["max_p"], kept=imp["kept"], unseen_bound=imp["unseen_bound"],
                               tv=tv, tv_expected=expected, excess=tv - expected,
                               out_of_support=sum(n for k, n in hist.items() if k not in probs),
                               outside_listed=sum(n for k, n in hist.items() if k not in listed),
                               implied=[[k, p] for k, p in sorted(probs.items(), key=lambda kv: -kv[1])],
                               histogram=[[k, n] for k, n in hist.most_common()])
        if row["status"] != "informative":
            dropped[row["status"]] += 1
        rows.append(row)
    inf = [r for r in rows if r["status"] == "informative"]
    sup = [r for r in rows if r["status"] in ("informative", "near_deterministic")]
    ex = [r["excess"] for r in inf]
    lo, hi = qm.cluster_ci([(x, 1) for x in ex], resamples, bootstrap_seed) if len(ex) >= 2 else (None, None)
    mean = (lambda key: statistics.fmean(r[key] for r in inf) if inf else None)
    if len(inf) < _MIN_PROMPTS:
        stat_v = "UNPROVEN"
        stat_why = (f"{len(inf)} informative prompt(s), need {_MIN_PROMPTS} (dropped: "
                    + (", ".join(f"{n} {k}" for k, n in sorted(dropped.items())) or "none") + ")")
    else:
        stat_v, stat_why = qm.margin_verdict(lo, hi, margin)
    oos = sum(r["out_of_support"] for r in sup)
    outside = sum(r["outside_listed"] for r in sup)
    of = seeds * len(sup)
    wl, wh = qm.wilson_ci(oos, of)
    oos_fail = wl is not None and wl > _OOS_MAX
    stat = {"mean_tv": mean("tv"), "mean_tv_expected": mean("tv_expected"), "excess": mean("excess"),
            "ci95": [lo, hi], "margin": margin, "verdict": stat_v, "n_prompts": len(inf), "min_prompts": _MIN_PROMPTS,
            "resamples": resamples, "bootstrap_seed": bootstrap_seed, "seeds": seeds,
            "binned_by": "token_id" if side.by_id else "text"}
    support = {"draws": oos, "of": of, "rate": oos / of if of else None, "ci95": [wl, wh], "max_rate": _OOS_MAX,
               "outside_listed": outside, "prompts": len(sup), "fail": oos_fail}
    own = list(caps)
    if side.aborted:
        own.append(f"{side.name}: the first request failed, nothing was sampled: {side.aborted[:200]}")
    elif side.errors:
        own.append(f"{side.name}: {len(side.errors)} of {side.requests} requests failed after a retry "
                   f"(e.g. {side.errors[0][:160]})")
    if side.unmatched:
        own.append(f"{side.name}: {side.unmatched}")
    fails = []
    if stat_v == "FAIL":
        fails.append(f"draws do not follow the server's own log-probabilities: mean excess TV {stat['excess']:+.4f}, "
                     f"{stat_why}")
        alt = [r["excess_if_tempered"] for r in inf if "excess_if_tempered" in r]
        if len(alt) == len(inf):
            alo, ahi = qm.cluster_ci([(x, 1) for x in alt], resamples, bootstrap_seed)
            stat["excess_if_logprobs_were_tempered"] = {"excess": statistics.fmean(alt), "ci95": [alo, ahi]}
            if ahi is not None and ahi <= margin:
                fails.append(f"the draws do fit these log-probabilities read as already temperature-scaled (excess "
                             f"{statistics.fmean(alt):+.4f}, 95% CI {alo:+.4f} to {ahi:+.4f}): the server likely "
                             "returns log-probabilities after temperature, the contract requires raw ones")
    if oos_fail:
        fails.append(f"{oos} of {of} draws ({oos / of:.2%}, 95% CI {wl:.2%} to {wh:.2%}) are tokens these settings "
                     f"remove (limit {_OOS_MAX:.1%})" + (f"; {outside} are not even in the listed top 20, where the "
                                                         "kept set provably ends" if outside else ""))
    if fails and unsent:
        fails.append(f"{', '.join(unsent)} not sent and read as off: a server that applies its own default for them "
                     "(vLLM applies generation_config.json) would also give this")
    if caps:
        verdict = "UNPROVEN"
        reason = ("; ".join(own) + f" (statistic alone: {stat_v}, {stat_why}"
                  + (f"; would FAIL: {'; '.join(fails)}" if fails else "") + ")")
    elif fails:
        verdict, reason = "FAIL", "; ".join(fails) + ("; also: " + "; ".join(own) if own else "")
    elif own:
        verdict, reason = "UNPROVEN", "; ".join(own) + f" (statistic alone: {stat_v}, {stat_why})"
    elif stat_v != "PASS":
        verdict, reason = stat_v, stat_why
    else:
        verdict, reason = "PASS", f"{stat_why}; {oos} of {of} draws outside the kept tokens"
    return {"verdict": verdict, "reason": reason, "problems": own, "statistic": stat, "out_of_support": support,
            "prompts": {"asked": len(chosen), "informative": len(inf), "support_checked": len(sup),
                        "dropped": dict(sorted(dropped.items())),
                        "dropped_why": {k: _DROPPED.get(k, k) for k in sorted(dropped)}},
            "rows": rows}


def _cross(cand: dict, base: dict, same_bins: bool) -> dict | None:
    """Information only: TV between the two servers' implied distributions, on prompts both could build."""
    if not same_bins:
        return {"n_prompts": 0, "note": "the two servers' tokens are binned differently (ids vs text)"}
    by_id = {r["id"]: r for r in base["rows"] if "implied" in r}
    per = []
    for r in cand["rows"]:
        other = by_id.get(r["id"])
        if "implied" in r and other is not None:
            per.append((r["id"], sm.tv_between(dict(map(tuple, r["implied"])), dict(map(tuple, other["implied"])))))
    if not per:
        return {"n_prompts": 0, "mean": None, "max": None, "max_prompt": None, "per_prompt": {}}
    worst = max(per, key=lambda kv: kv[1])
    return {"n_prompts": len(per), "mean": statistics.fmean(v for _, v in per), "max": worst[1],
            "max_prompt": worst[0], "per_prompt": dict(per)}


@operation("quality_sampling", ("quality", "sampling"),
           "Sampler check (self-consistency): per prompt, the server's own top-20 first-token log-probabilities "
           "(asked twice) give the distribution its sampling settings imply; S seeded max_tokens-1 draws are "
           "tested against it (mean TV minus its exact expectation under correct sampling, bootstrap CI over "
           "prompts, against a margin), plus the share of draws outside the kept tokens. One server at a time; "
           "with --baseline-url the baseline is checked the same way and the two implied distributions compared "
           "(information only).",
           [Arg("base_url", "string", "Server to check, e.g. http://127.0.0.1:8000.", required=True),
            Arg("model", "string", "Model name the server expects.", required=True),
            Arg("baseline_url", "string", "Optional second server (e.g. the stock engine on the same checkpoint), "
                "checked the same way, after the first; its verdict is reported, the record's is the first's."),
            Arg("baseline_model", "string", "Model name on the baseline server (default: --model)."),
            Arg("run_dir", "string", "Run directory (runs/<run-id>).", required=True),
            Arg("label", "string", "Name of the server checked (file name).", default="engine"),
            Arg("prompts", "string", "Prompt-set JSONL (default: the packaged dev set)."),
            Arg("buckets", "array", "Prompt buckets to draw from (a file without bucket fields is used whole).",
                default=["very_short", "short"]),
            Arg("n_prompts", "integer", "Prompts: N, taken in turn from those buckets (file order inside each).",
                default=32),
            Arg("seeds", "integer", "Draws per prompt, seeds 0..S-1.", default=256),
            Arg("extra_body", "string", "JSON object of the sampling settings sent with every request; temperature "
                "is required, and top_k, top_p, min_p are read from it, e.g. "
                "'{\"temperature\": 0.6, \"top_k\": 20, \"top_p\": 0.95}'. The check sets max_tokens (1), seed, n, "
                "stream, logprobs, top_logprobs and vLLM's token-id fields itself."),
            Arg("chat_template_kwargs", "string", "JSON object passed as chat_template_kwargs, e.g. "
                "'{\"enable_thinking\": false}' (a model that opens with a fixed thinking token has nothing to "
                "test)."),
            Arg("margin", "number", "Pre-registered margin on the mean excess TV (probability mass).", default=0.02),
            Arg("resamples", "integer", "Bootstrap resamples over prompts.", default=10000),
            Arg("bootstrap_seed", "integer", "Bootstrap seed.", default=0),
            Arg("concurrency", "integer", "Requests in flight per server. 1 by default: a batched engine's logits "
                "change with the batch, so draws taken alongside other requests need not follow the "
                "log-probabilities of a request run alone.", default=1),
            Arg("api_key_env", "string", "Name of the env var holding the server's API key.",
                default="OPENAI_API_KEY"),
            Arg("baseline_api_key_env", "string", "Name of the env var holding the baseline's API key (default: "
                "--api-key-env)."),
            Arg("timeout", "integer", "Seconds per request.", default=600)])
def quality_sampling(call, base_url, model, baseline_url, baseline_model, run_dir, label, prompts, buckets, n_prompts,
                     seeds, extra_body, chat_template_kwargs, margin, resamples, bootstrap_seed, concurrency,
                     api_key_env, baseline_api_key_env, timeout):
    from inferencekit import prompts as sets

    label = _safe(label, "label")
    if n_prompts < 2:
        raise OpError("n_prompts must be >= 2 (the confidence interval resamples prompts)")
    if seeds < 2:
        raise OpError("seeds must be >= 2")
    if concurrency < 1 or resamples < 1:
        raise OpError("concurrency and resamples must be >= 1")
    extra = _json_arg("extra_body", extra_body) or {}
    # chat_template_kwargs layers as in bench serve: --extra-body lowest, then the flag (and the intake pin),
    # then each prompt record's own field
    base_kwargs = apply_pin({**(extra.pop("chat_template_kwargs", None) or {}),
                             **(_json_arg("chat_template_kwargs", chat_template_kwargs) or {})},
                            run_dir, "quality sampling")
    dropped = sorted(k for k in extra if k in _SAMPLING_OWN_KEYS)
    extra = {k: v for k, v in extra.items() if k not in _SAMPLING_OWN_KEYS}
    try:
        cfg, unsent = sm.sampling_settings(extra)
    except ValueError as e:
        raise OpError(f"--extra-body: {e}") from None
    source = prompts or str(sets.resolve("dev"))
    items, prompts_sha = load_prompts(source)
    set_name = sets.set_stem(source) if prompts else "dev"
    wanted = list(dict.fromkeys(buckets or []))
    if wanted and any(it.get("bucket") for it in items):
        # taken in turn from each bucket (file order inside a bucket): 32 prompts of a set with 100 very_short
        # and 100 short ones are 16 + 16, not 32 very_short
        queues = [[it for it in items if it.get("bucket") == b] for b in wanted]
        items = [q[i] for i in range(max(map(len, queues))) for q in queues if i < len(q)]
    chosen = items[:n_prompts]
    if len(chosen) < 2:
        raise OpError(f"{len(chosen)} prompt(s) in buckets {wanted} of {source}: need at least 2")
    bmodel = baseline_model or model

    def kwargs_for(item: dict) -> dict:
        own = item.get("chat_template_kwargs")
        return {**base_kwargs, **own} if isinstance(own, dict) else base_kwargs

    run = Path(run_dir)
    s_range = range(seeds)
    caps = []  # settings this check cannot compute, on both servers: no PASS and no FAIL
    unmodelled = _unmodelled(extra)
    if unmodelled:
        caps.append(f"--extra-body sets {', '.join(unmodelled)}, which change the first-token distribution in a "
                    "way this check does not compute")
    cand = _Side("candidate", Client(base_url, api_key_env, timeout), model, extra, kwargs_for, concurrency)
    cand.run(chosen, s_range)
    res = _judge(cand, chosen, cfg, seeds, margin, resamples, bootstrap_seed, caps, unsent)
    base = bres = cross = None
    if baseline_url:
        base = _Side("baseline", Client(baseline_url, baseline_api_key_env or api_key_env, timeout), bmodel, extra,
                     kwargs_for, concurrency)
        base.run(chosen, s_range)
        bres = _judge(base, chosen, cfg, seeds, margin, resamples, bootstrap_seed, caps, unsent)
        cross = _cross(res, bres, cand.by_id == base.by_id)
    verdict, reason, stat, support = res["verdict"], res["reason"], res["statistic"], res["out_of_support"]

    limitations = []
    if bres is not None and bres["verdict"] != "PASS":
        limitations.append(f"BASELINE {bres['verdict']}: the baseline does not pass this same check, so its sampler "
                           f"is no reference either: {bres['reason'][:300]}")
    limitations += [
        ("self-consistency: shows that the server's draws follow its own top-20 log-probabilities under these "
         "settings; whether those log-probabilities match the baseline's is the teacher-forced check's job "
         "(quality score)"),
        "max_tokens 1: the first generated token only (prefill), not later positions or longer contexts",
        ("the listed log-probabilities must be raw (log-softmax before temperature, top-k and top-p, vLLM's "
         "default); a server that returns processed ones reads as a wrong sampler")]
    sides = [cand] + ([base] if base else [])
    if any(side.by_id for side in sides):
        limitations.append("tokens are matched by id (vLLM's return_token_ids and return_tokens_as_token_ids)")
    if not all(side.by_id for side in sides):
        limitations.append("tokens are matched by their text, surrounding whitespace and U+FFFD removed (not every "
                           "draw or listed token came with an id): tokens with the same text share a bin, which "
                           "can only lower TV")
    for side in sides:
        if side.refused:
            limitations.append(f"{side.name} refused a vLLM token-id field, asked without it: {side.refused}")
        if side.k_state["k"] < _TOP_LOGPROBS:
            limitations.append(f"{side.name} lists at most {side.k_state['k']} log-probabilities (asked for "
                               f"{_TOP_LOGPROBS}): fewer prompts can be tested")
        if side.seed_check.get("status") == "ignored":
            sc = side.seed_check
            limitations.append(f"{side.name} does not honour the request seed ({sc['same']} of {sc['repeats']} "
                               f"repeats gave the same token, about {100 * sc['expected_same_if_ignored']:.0f}% would "
                               "by chance): information only, its draws are still samples; seeding is checked by "
                               "the gates")
    if unsent:
        limitations.append(f"{', '.join(unsent)} not sent: read as off, but a server that applies its own "
                           "default (vLLM applies generation_config.json) then draws from another distribution; "
                           "send every sampling field")
    if concurrency > 1:
        limitations.append(f"concurrency {concurrency}: a batched engine's logits change with the batch, so draws "
                           "taken alongside other requests need not follow the log-probabilities of one alone")
    if dropped:
        limitations.append(f"--extra-body keys set by the check were ignored: {', '.join(dropped)}")
    if len(chosen) < n_prompts:
        limitations.append(f"only {len(chosen)} prompts in buckets {wanted} (asked for {n_prompts})")
    if cfg["min_p"] is not None:
        limitations.append("min_p: engines differ on whether it comes before or after top-p; a prompt counts only "
                           "when both orders keep the same tokens")
    if cfg["top_k"] is not None and cfg["top_p"] is not None:
        limitations.append("top_k with top_p: SGLang takes top-p over the whole distribution, Hugging Face and vLLM "
                           "over the top-k tokens; a prompt counts only when both keep the same tokens")
    limitations.append(f"the margin bounds the mean excess TV, not the sampler's own error: the excess grows "
                       f"about with the square of a small error, so a sampler a few hundredths of probability off "
                       f"can still PASS ({seeds} seeds x {len(chosen)} prompts)")
    missing = ["samples"] if any(side.errors for side in sides) else []

    def side_doc(side: _Side, r: dict, url_model: str) -> dict:
        return {"url": side.client.base, "model": url_model, "verdict": r["verdict"], "reason": r["reason"],
                "problems": r["problems"], "statistic": r["statistic"], "out_of_support": r["out_of_support"],
                "prompts": r["prompts"], "seed_check": side.seed_check, "requests": side.requests,
                "errors": side.errors[:20], "token_id_fields": dict(side.fields), "top_logprobs": side.k_state["k"]}

    out_dir = run / "quality" / "sampling"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{label}.json"
    settings = {"max_tokens": 1, "extra_body": ev.redact(extra), "sampling": cfg, "not_sent": unsent,
                "chat_template_kwargs": base_kwargs or None, "seeds": [0, seeds - 1], "top_logprobs": _TOP_LOGPROBS,
                "concurrency": concurrency, "ignored_extra_body_keys": dropped,
                "near_deterministic": _NEAR_DETERMINISTIC, "logprob_tolerance": sm.LOGPROB_TOL}
    prompts_doc = {"name": set_name, "sha256": prompts_sha, "ids": [it["id"] for it in chosen]}
    doc = {"format": SAMPLING_FORMAT, "method": "self_consistency", "label": label, "verdict": verdict,
           "reason": reason, "settings": settings, "prompts": prompts_doc,
           "candidate": {**side_doc(cand, res, model), "per_prompt": res["rows"]},
           "baseline": {**side_doc(base, bres, bmodel), "per_prompt": bres["rows"]} if base else None,
           "cross_server": cross}
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")

    keep = ("id", "bucket", "status", "max_p", "kept", "tv", "tv_expected", "excess", "out_of_support")
    lo, hi = stat["ci95"]
    metrics = [ev.metric("first_token_tv_excess", stat["excess"], "probability", "mean", n=stat["n_prompts"],
                         ci95=[lo, hi], subject=label),
               ev.metric("first_token_tv", stat["mean_tv"], "probability", "mean", n=stat["n_prompts"], subject=label),
               ev.metric("first_token_tv_expected", stat["mean_tv_expected"], "probability", "mean",
                         n=stat["n_prompts"], subject=label),
               ev.metric("first_token_out_of_support", support["rate"], "fraction", "rate", n=support["of"],
                         ci95=support["ci95"], subject=label)]
    if bres is not None:
        bs = bres["statistic"]
        metrics += [ev.metric("first_token_tv_excess", bs["excess"], "probability", "mean", n=bs["n_prompts"],
                              ci95=bs["ci95"], subject="baseline"),
                    ev.metric("first_token_out_of_support", bres["out_of_support"]["rate"], "fraction", "rate",
                              n=bres["out_of_support"]["of"], ci95=bres["out_of_support"]["ci95"], subject="baseline"),
                    ev.metric("first_token_implied_tv_between_servers", cross.get("mean"), "probability", "mean",
                              n=cross.get("n_prompts"), subject=f"{label} vs baseline")]
    metrics.append(ev.metric("sampling_requests", sum(side.requests for side in sides), "requests", "count"))

    def fmt(x, sign: str = "+") -> str:
        return f"{x:{sign}.4f}" if x is not None else "n/a"

    summary = (f"{label}: {verdict}: sampled first tokens vs the server's own log-probabilities: mean TV "
               f"{fmt(stat['mean_tv'], '')}, expected {fmt(stat['mean_tv_expected'], '')}, excess {fmt(stat['excess'])} "
               f"(95% CI {fmt(lo)} to {fmt(hi)}, margin {margin:g}) over {stat['n_prompts']} prompts x {seeds} seeds; "
               f"{support['draws']} of {support['of']} draws outside the kept tokens")
    if bres is not None:
        summary += f"; baseline {bres['verdict']} on the same check"
    ident = ev.identity(engine={"name": label}, model={"id": model},
                        workload={"gen_len": 1, "batch": concurrency, "concurrency": concurrency,
                                  "prompt_set": {"name": set_name, "sha256": prompts_sha}})
    data = {"task": "sampling", "method": "self_consistency", "label": label, "verdict": verdict, "reason": reason,
            "problems": res["problems"], "statistic": stat, "out_of_support": support, "prompts_used": res["prompts"],
            "settings": settings, "prompts": prompts_doc,
            "seed_checks": {"candidate": cand.seed_check, **({"baseline": base.seed_check} if base else {})},
            "errors": {"candidate": len(cand.errors), **({"baseline": len(base.errors)} if base else {})},
            "per_prompt": [{k: r[k] for k in keep if k in r} for r in res["rows"]],
            "baseline": None, "cross_server": None, "results_file": str(path)}
    if base is not None:
        data["baseline"] = {"url": base.client.base, "model": bmodel, "verdict": bres["verdict"],
                            "reason": bres["reason"], "statistic": bres["statistic"],
                            "out_of_support": bres["out_of_support"], "prompts_used": bres["prompts"]}
        data["cross_server"] = {k: v for k, v in cross.items() if k != "per_prompt"}
    return _save(call, run, "quality_task", claim_class="derived", summary=summary[:600], identity_=ident,
                 metrics=metrics, artifacts=[ev.artifact(path, run, "sampling_histograms")], missing=missing,
                 limitations=limitations, data=data)
