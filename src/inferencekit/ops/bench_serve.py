"""bench serve: time an OpenAI-compatible server from the client side, the one timer for published numbers.

The engine runs as its own server (vllm serve, SGLang launch_server, llama-server, trtllm-serve, a
custom engine); inferencekit is only a client, so nothing it does can flatter the engine's numbers and
nothing the engine does can touch the timer. See serve/client.py for the timestamps and
serve/metrics.py for the definitions.

Modes
  single  sequential requests over the prompt set, ``rounds`` times (plus ``warmup`` untimed requests).
          Per bucket: TTFT, TPOT, decode tok/s, end-to-end; median and p95 with 95% CIs over rounds.
          Writes one ``serve_benchmark`` record (observed). Its ``data.rounds`` has one row per round
          in the row shape of bench decode records (decode_tok_s, ttft_ms, tpot_ms, total_ms, t_start).
          To interleave engines A and B, alternate ``--rounds 1 --extend <previous ev>`` calls (A, B,
          A, B ...), or use ``bench rounds``; ``compare`` takes two such records. The record's
          ``workload.request_set`` hashes the selected prompts and per-request settings, so compare
          can refuse two records that ran different subsets or template settings.
  load    open-loop seeded Poisson arrivals at a concurrency or rate grid (serve/loadgen.py). Per level:
          p50/p95/p99 TTFT, TPOT, ITL, e2e; failures; goodput; output tok/s per GPU. Writes a
          ``serve_benchmark`` record (observed, the levels) and a ``capacity`` record (derived: N*, the
          highest level where the SLO holds there and at every lower level, and $/1M output tokens at
          N*). Returns the capacity record. ``--closed-loop`` runs a labelled stress test instead.

Prefix cache (single mode): the warm-up prompts carry a unique tag, so round 0 is cold. By default
every later round tags its prompts too ("[ik-r<N>] ", a few tokens), so no timed request can be served
from the engine's prefix cache and TTFT is cold prefill in every round. ``--no-cache-salt`` resends
identical prompts on purpose (warm-cache TTFT); the record says which (``workload.prefix_cache``).

Files: every request's text, token ids, finish reason, usage and event timings, under
``outputs/<label>/<bucket>/<id>.json`` (single mode: round 0, the untagged one) or
``outputs/<label>/load-<level>/<bucket>/<id>.json`` (load mode), and the exact request body sent under
``inputs/<label>/...`` with the same layout. Single mode also writes the shared
``inputs/<bucket>/<id>.json`` (engine-independent path read by the gates; the last engine run in a run
dir wins). The files are kept in the evidence record's directory with an ``index.json`` of their
sha256 (the record's ``outputs_index`` artifact), and copied to ``--run-dir`` when given.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import statistics
import time
from pathlib import Path

from inferencekit import evidence as ev
from inferencekit import stats
from inferencekit.canonical import canonical_json, file_sha256
from inferencekit.cpuinfo import cpu_identity
from inferencekit.gpuinfo import gpu_identity
from inferencekit.ops.intake import apply_pin, intake_pin
from inferencekit.registry import Arg, Call, OpError, operation
from inferencekit.serve import client as cl
from inferencekit.serve import loadgen, metrics, profiles, prompts_io

DEFAULT_MAX_TOKENS = 256
# Arguments that must match for --extend: they define "the same configuration".
SAME_CONFIG = ("base_url", "model", "api", "engine_kind", "engine_version", "max_tokens", "buckets",
               "limit_per_bucket", "token_ids", "exact_length", "cache_salt", "extra_body", "label", "seed",
               "chat_template_kwargs")
JSON_ARGS = ("extra_body", "chat_template_kwargs")  # compared and matched as parsed JSON, not as text
INFLIGHT_SHORTFALL = 0.9  # flag an N* whose measured mean in-flight count is below this fraction of N*
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}$")

ARGS = [
    Arg("base_url", "string", "Server root, e.g. http://127.0.0.1:8000 (a trailing /v1 is fine). Never stored: "
        "records keep only an HMAC of it under a per-machine key kept outside the evidence store, so a "
        "published record does not reveal the host (hashing candidate addresses does not recover it).",
        required=True, sensitive=True),
    Arg("model", "string", "Model name the server expects in requests (as listed by /v1/models).", required=True),
    Arg("prompts", "string", "Prompt set JSONL ({id, bucket, messages, max_tokens} per line).", required=True),
    Arg("api", "string", "OpenAI endpoint to call.", default="chat", choices=profiles.APIS),
    Arg("engine_kind", "string", "Server type: picks the documented fields for exact length and token ids.",
        default="generic", choices=profiles.ENGINE_KINDS),
    Arg("mode", "string", "single = one request at a time; load = open-loop arrivals at a grid.",
        default="single", choices=("single", "load")),
    Arg("max_tokens", "integer", "Tokens to generate per request (default: each prompt's max_tokens, else 256)."),
    Arg("rounds", "integer", "Single mode: timed passes over the prompt set.", default=5),
    Arg("warmup", "integer", "Untimed requests first, each with a unique prompt tag (they never seed the prefix "
        "cache of a timed prompt).", default=2),
    Arg("ref_requests", "integer", "Load mode with --concurrency: unloaded requests sent one at a time after the "
        "warm-up; their median e2e is e2e_ref (not sent with --ref-e2e-ms).", default=5),
    Arg("concurrency", "array", "Load mode: concurrency levels (streams); Poisson rate = N / e2e_ref.",
        items="integer"),
    Arg("rates", "array", "Load mode: arrival rates in requests/s (instead of --concurrency).", items="number"),
    Arg("duration_s", "number", "Load mode: arrival window per level, seconds.", default=30.0),
    Arg("max_requests_per_level", "integer", "Load mode: cap on requests per level (0 = no cap).", default=0),
    Arg("ref_e2e_ms", "number", "Load mode: unloaded e2e latency used to turn concurrency into a rate. Pass "
        "the same value for every engine to give them identical schedules (default: the median of "
        "--ref-requests reference requests)."),
    Arg("slo_ttft_p95_ms", "number", "SLO: p95 time to first token, ms."),
    Arg("slo_tpot_p95_ms", "number", "SLO: p95 time per output token, ms."),
    Arg("slo_max_fail", "integer", "SLO: failed requests allowed per level.", default=0),
    Arg("sweep_all", "boolean", "Load mode: keep sweeping after the first level that misses the SLO.",
        default=False),
    Arg("closed_loop", "boolean", "Load mode: N back-to-back workers instead of open-loop arrivals. A STRESS "
        "TEST: it slows its own arrivals when the server slows, so it is never used for N*.", default=False),
    Arg("gpu_price_usd_h", "number", "GPU price per hour (per GPU), for $/1M output tokens at N*."),
    Arg("gpu_count", "integer", "GPUs behind the server (throughput and cost are per GPU).", default=1),
    Arg("gpu_name", "string", "GPU model, when the server is not on this machine (otherwise read locally)."),
    Arg("seed", "integer", "Seed for arrivals and prompt order.", default=0),
    Arg("label", "string", "Name for this engine configuration in outputs/ (default: engine_kind)."),
    Arg("run_dir", "string", "Run directory: outputs/<label>/... and metrics/<label>.json are written there."),
    Arg("buckets", "array", "Only these buckets.", items="string"),
    Arg("limit_per_bucket", "integer", "Only the first N prompts of each bucket (0 = all).", default=0),
    Arg("token_ids", "string", "auto = ask for streamed ids where it is free (vLLM return_token_ids); "
        "logprobs = also through logprobs (can slow decode); off = never.", default="auto",
        choices=profiles.TOKEN_ID_MODES),
    Arg("exact_length", "boolean", "Ask the engine to generate exactly max_tokens (ignore_eos / min_tokens).",
        default=True),
    Arg("cache_salt", "boolean", "Single mode: rounds after the first prefix each prompt with a unique tag, so "
        "no timed request can hit the prefix cache (cold-prefill TTFT in every round; round 0 is sent "
        "untagged and is the one written to outputs/). --no-cache-salt resends identical prompts every "
        "round on purpose (warm-cache TTFT).", default=True),
    Arg("tokenizer", "string", "HF tokenizer to re-count tokens when the server returns neither ids nor usage "
        "(needs transformers)."),
    Arg("extra_body", "string", "JSON object merged into every request body (e.g. sampling settings). "
        "Secret-looking keys are stored as <redacted>.", default="{}"),
    Arg("api_key_env", "string", "Name of the environment variable holding the API key (never stored)."),
    Arg("engine_version", "string", "Engine version, recorded in the identity."),
    Arg("timeout_s", "number", "Per-request timeout, seconds.", default=600.0),
    Arg("resamples", "integer", "Bootstrap resamples for CIs.", default=stats.DEFAULT_RESAMPLES),
    Arg("extend", "string", "Single mode: evidence id of an earlier run of the same configuration; its "
        "rounds are kept and new rounds appended (alternate engines this way)."),
    Arg("chat_template_kwargs", "string", "JSON object sent as chat_template_kwargs on every chat request, e.g. "
        "'{\"enable_thinking\": false}'. A prompt record's own chat_template_kwargs field is merged over it "
        "(per key), so every engine gets the same template settings per request."),
]


def _json_object(text: str, what: str) -> dict:
    try:
        value = json.loads(text or "{}")
    except ValueError as e:
        raise OpError(f"{what} is not valid JSON: {e}") from None
    if not isinstance(value, dict):
        raise OpError(f"{what} must be a JSON object")
    return value


def normalized_json(text) -> str | None:
    """A JSON-object argument in one canonical spelling (sorted keys, compact), so the same setting written
    two ways compares equal ('{"enable_thinking": false}' == '{"enable_thinking":false}'). Text that is not
    JSON is returned unchanged; empty / {} is None."""
    if text in (None, ""):
        return None
    try:
        value = json.loads(text) if isinstance(text, str) else text
    except ValueError:
        return text
    if value in ({}, None):
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def request_set(pset, runner) -> dict:
    """identity.workload.request_set: sha256 over exactly what was asked per prompt, independent of the
    engine: the selected prompt ids and buckets (after --buckets / --limit-per-bucket), each prompt's
    max_tokens and merged chat_template_kwargs, plus the API, exact-length request and --extra-body
    (secret-looking values redacted). Two records with the same prompt-set file but a different selection
    or template settings differ here, so compare can refuse them."""
    items = []
    for p in pset.prompts:
        extra = runner.extra_for(p)
        items.append({"id": p.id, "bucket": p.bucket, "max_tokens": runner.max_for(p),
                      "chat_template_kwargs": extra.get("chat_template_kwargs") if runner.api == "chat" else None})
    settings = {"api": runner.api, "exact_length": bool(runner.exact_length),
                "extra_body": ev.redact({k: v for k, v in runner.extra.items() if k != "chat_template_kwargs"})}
    return {"sha256": hashlib.sha256(canonical_json({"items": items, "settings": settings}).encode()).hexdigest(),
            "n": len(items)}


def chat_template_kwargs_for(prompt, *layers) -> dict:
    """Merged chat_template_kwargs for one prompt: ``layers`` in order (lowest precedence first; None or
    non-dict layers skipped), then the prompt record's own ``chat_template_kwargs`` field, per key.
    ``prompt`` is a prompts_io.Prompt or a prompt-record dict. Shared by bench serve, gates and quality
    trajectory so every op sends an engine the same template settings for the same prompt."""
    own = prompt.get("chat_template_kwargs") if isinstance(prompt, dict) else \
        (getattr(prompt, "extra", None) or {}).get("chat_template_kwargs")
    out: dict = {}
    for layer in (*layers, own):
        if isinstance(layer, dict):
            out.update(layer)
    return out


def _tokenizer(name: str | None):
    if not name:
        return None
    try:
        from transformers import AutoTokenizer  # optional; only for servers without ids or usage
    except ImportError:
        raise OpError("--tokenizer needs the transformers package installed") from None
    tok = AutoTokenizer.from_pretrained(name)
    return lambda text: tok.encode(text, add_special_tokens=False)


def _endpoint_key() -> tuple[bytes, bool]:
    """The per-machine HMAC key for endpoint tags, and whether it persists across processes.

    It lives outside the evidence store ($INFERENCEKIT_KEY_FILE, else $XDG_CONFIG_HOME or ~/.config,
    /inferencekit/endpoint.key, mode 0600), so publishing records never publishes the key.
    """
    cfg = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    path = Path(os.environ.get("INFERENCEKIT_KEY_FILE") or cfg / "inferencekit" / "endpoint.key")
    for _ in range(2):
        try:
            return bytes.fromhex(path.read_text().strip()), True
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            break
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(secrets.token_hex(32) + "\n")
        except FileExistsError:
            continue  # another process created it first: read theirs
        except OSError:
            break
    return _PROCESS_KEY, False


_PROCESS_KEY = secrets.token_bytes(32)


def _record_call(call: Call, ep) -> tuple[Call, bool]:
    """``call`` with base_url replaced by a keyed tag, for everything that is stored.

    The registry stores sensitive arguments as a plain sha256, and the argv digest covers every
    argument; for a URL such as http://<IPv4>:8000 both can be brute-forced over the address space.
    An HMAC under a per-machine key cannot, and it still lets --extend check "same server" on this
    machine.
    """
    key, persistent = _endpoint_key()
    canon = f"{ep.scheme}://{ep.host.lower()}:{ep.port}{ep.prefix}"
    tag = "hmac-sha256:" + hmac.new(key, canon.encode(), hashlib.sha256).hexdigest()
    return Call(call.op, {**call.args, "base_url": tag}, started=call.started), persistent


class Runner:
    """Holds what every request needs: endpoint, body template, token counter, outputs."""

    def __init__(self, ep, api, engine_kind, model, max_tokens, token_ids, exact_length, extra, tokenizer,
                 timeout_s):
        self.ep, self.api, self.engine_kind, self.model = ep, api, engine_kind, model
        self.max_tokens, self.token_ids, self.exact_length, self.extra = max_tokens, token_ids, exact_length, extra
        self.tokenizer, self.timeout_s = tokenizer, timeout_s
        self.chat_template_kwargs: dict = {}
        self.asked = None

    def max_for(self, prompt) -> int:
        return int(self.max_tokens or prompt.max_tokens or DEFAULT_MAX_TOKENS)

    def extra_for(self, prompt) -> dict:
        """--extra-body, plus chat_template_kwargs: --extra-body's, then --chat-template-kwargs, then the
        prompt record's own field, merged per key (chat API only; /v1/completions has no template)."""
        kw = chat_template_kwargs_for(prompt, self.extra.get("chat_template_kwargs"), self.chat_template_kwargs)
        if self.api != "chat" or not kw:
            return self.extra
        return {**self.extra, "chat_template_kwargs": kw}

    def body(self, prompt, salt):
        body, asked = profiles.build_body(self.api, self.engine_kind, self.model,
                                          loadgen.salted_messages(prompt, salt), self.max_for(prompt),
                                          token_ids=self.token_ids, exact_length=self.exact_length,
                                          extra=self.extra_for(prompt))
        self.asked = self.asked or asked
        return body

    async def send(self, prompt, salt=None, t_scheduled_ns=None):
        body = self.body(prompt, salt)
        res = await cl.stream(self.ep, self.api, body, rid=prompt.id, bucket=prompt.bucket,
                              timeout_s=self.timeout_s, t_scheduled_ns=t_scheduled_ns)
        res.salt, res.request_body = salt, body
        cl.count_tokens(res, self.tokenizer)
        return res


def _output_doc(res, prompt, runner, **extra) -> dict:
    doc = {"id": res.id, "bucket": res.bucket, "text": res.text, "token_ids": res.token_ids,
           "finish_reason": res.finish_reason, "usage": res.usage, "n_tokens": res.n_tokens,
           "token_count_method": res.token_count_method, "max_tokens": runner.max_for(prompt),
           "error": res.error, "timings": metrics.timings(res), **extra}
    if res.reasoning:
        doc["reasoning"] = res.reasoning
    return doc


def _dump(doc: dict) -> str:
    return json.dumps(ev.redact(doc), ensure_ascii=False) + "\n"


def _write_run_files(staging: Path, run_dir: Path | None, label: str, outputs: list[tuple[str, dict]],
                     inputs: list[tuple[str, dict]], contract_inputs: bool = False) -> Path | None:
    """Write this run's files and return the path of their index (None when there are none).

    ``outputs/<label>/<rel>`` and ``inputs/<label>/<rel>`` go into the record's directory (``staging``),
    with ``outputs/<label>/index.json`` listing exactly this run's files and their sha256: the record's
    ``outputs_index`` artifact, which later runs cannot change. With ``run_dir`` the same bytes are also
    written there, and the run dir's ``index.json`` is merged across runs as a convenience.
    ``contract_inputs`` also writes ``inputs/<rel>`` (the engine-independent path).
    """
    if not outputs and not inputs:
        return None
    snap: dict = {"label": label, "files": {}, "inputs": {}}
    for root in [staging] + ([Path(run_dir)] if run_dir else []):
        for kind, docs, key in (("outputs", outputs, "files"), ("inputs", inputs, "inputs")):
            for rel, doc in docs:
                data = _dump(doc).encode("utf-8")
                targets = [root / kind / label / rel]
                if kind == "inputs" and contract_inputs:
                    targets.append(root / "inputs" / rel)
                for t in targets:
                    t.parent.mkdir(parents=True, exist_ok=True)
                    t.write_bytes(data)
                if root == staging:
                    snap[key][rel] = hashlib.sha256(data).hexdigest()
    base = staging / "outputs" / label
    base.mkdir(parents=True, exist_ok=True)
    snap = {k: dict(sorted(v.items())) if isinstance(v, dict) else v for k, v in snap.items()}
    (base / "index.json").write_text(json.dumps(snap, indent=1) + "\n")
    if run_dir:
        idx = Path(run_dir) / "outputs" / label / "index.json"
        old = {}
        if idx.exists():
            try:
                old = json.loads(idx.read_text())
            except ValueError:
                old = {}
        merged = {"label": label, "note": "merged across runs of this label; each record keeps its own snapshot"}
        for key in ("files", "inputs"):
            merged[key] = dict(sorted({**(old.get(key) or {}), **snap[key]}.items()))
        idx.parent.mkdir(parents=True, exist_ok=True)
        idx.write_text(json.dumps(merged, indent=1) + "\n")
    return base / "index.json"


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return path


async def _check_server(ep, model) -> tuple[list[str], dict]:
    """GET /v1/models: is the server up, and does it serve ``model``? Returns (limitations, info)."""
    try:
        status, body = await cl.get_json(ep, "/v1/models")
    except (OSError, ConnectionError, asyncio.TimeoutError, ValueError) as e:
        raise OpError(f"server at {ep.scheme}://{ep.host}:{ep.port} not reachable ({type(e).__name__}: {e}); "
                      "start it and wait for /v1/models to answer") from None
    lim, info = [], {"models_status": status}
    if status == 200 and isinstance(body, dict):
        ids = [m.get("id") for m in body.get("data") or [] if isinstance(m, dict)]
        info["served_models"] = ids[:20]
        for m in body.get("data") or []:
            if isinstance(m, dict) and m.get("id") == model and m.get("max_model_len"):
                info["max_model_len"] = m["max_model_len"]
        if ids and model not in ids:
            lim.append(f"model {model!r} is not in the server's /v1/models list {ids[:5]}")
    else:
        lim.append(f"/v1/models answered HTTP {status}: model name not checked")
    return lim, info


def _identity(ep, gpu_name, gpu_count, engine_kind, engine_version, model, workload):
    missing = []
    cpu = cpu_identity() if ep.is_local() and os.environ.get("INFERENCEKIT_CPU_LOCK_HELD") else None
    if cpu is not None:
        gpu = {}
    elif gpu_name:
        gpu = {"name": gpu_name, "count": gpu_count}
    elif ep.is_local():
        gpu = gpu_identity() or {}
    else:
        gpu = {"count": gpu_count}
    if cpu is not None and not cpu.get("model"):
        missing.append("cpu identity")
    elif cpu is None and not gpu.get("name"):
        missing.append("gpu identity (remote server or no nvidia-smi: pass --gpu-name)")
    ident = ev.identity(gpu=gpu, cpu=cpu, engine={"name": engine_kind, "version": engine_version}, model={"id": model},
                        workload=workload)
    return ident, missing


def _count_limitations(rows: list[dict], exact_asked, max_by_id: dict, token_ids_mode: str,
                       engine_kind: str) -> list[str]:
    lim = []
    ok = [r for r in rows if r["ok"]]
    methods = sorted({r["token_count_method"] for r in ok if r["token_count_method"]})
    if any(m == "usage" for m in methods):
        lim.append("tokens counted from usage.completion_tokens (no streamed ids): decode tok/s assumes the first "
                   "token event holds one token")
    if any(m == "sse_events" for m in methods):
        lim.append("tokens counted as SSE events (no ids, no usage): only right if the server sends one token "
                   "per event; pass --tokenizer to re-count")
    mism = [r for r in ok if r.get("count_mismatch")]
    if mism:
        lim.append(f"{len(mism)} request(s): streamed ids disagree with usage ({mism[0]['count_mismatch']})")
    if exact_asked:
        short = [r for r in ok if r["n_tokens"] is not None and r["n_tokens"] != max_by_id.get(r["id"])]
        if short:
            lim.append(f"{len(short)} request(s) did not generate exactly max_tokens although "
                       f"{'/'.join(exact_asked)} was sent (e.g. {short[0]['id']}: {short[0]['n_tokens']} tokens)")
    else:
        lim.append(f"engine_kind={engine_kind}: no exact-length control sent; outputs stop at EOS, so lengths can "
                   "differ between engines (the output-length gate applies)")
    if token_ids_mode == "logprobs":
        lim.append("token ids requested through logprobs where needed: computing logprobs can slow decode")
    multi = [r for r in ok if (r.get("tokens_per_event_max") or 0) > 1]
    if multi:
        lim.append(f"{len(multi)} request(s) streamed more than one token in an SSE event (max "
                   f"{max(r['tokens_per_event_max'] for r in multi)}); counts use ids, ITL is per event")
    return lim


def _fmt(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


@operation("bench_serve", ("bench", "serve"), "Client-side streaming benchmark of an OpenAI-compatible server "
           "(vLLM, SGLang, llama-server, trtllm-serve, custom): single-user TTFT/TPOT/decode tok/s per bucket "
           "with CIs over rounds, or open-loop load with N* at an SLO and $/1M tokens.", ARGS)
def bench_serve(call, base_url, model, prompts, api, engine_kind, mode, max_tokens, rounds, warmup, ref_requests,
                concurrency, rates, duration_s, max_requests_per_level, ref_e2e_ms, slo_ttft_p95_ms, slo_tpot_p95_ms,
                slo_max_fail, sweep_all, closed_loop, gpu_price_usd_h, gpu_count, gpu_name, seed, label, run_dir,
                buckets, limit_per_bucket, token_ids, exact_length, cache_salt, tokenizer, extra_body, api_key_env,
                engine_version, timeout_s, resamples, extend, chat_template_kwargs=None):
    if max_tokens is not None and max_tokens < 2:
        raise OpError("max_tokens must be >= 2 (decode speed needs tokens after the first)")
    if mode == "single" and rounds < 1:
        raise OpError("rounds must be >= 1")
    if mode == "load":
        if bool(concurrency) == bool(rates):
            raise OpError("load mode needs exactly one of --concurrency or --rates")
        grid = concurrency or rates
        if any(x <= 0 for x in grid):
            raise OpError("grid levels must be > 0")
        if concurrency and not closed_loop and ref_e2e_ms is None and ref_requests < 1:
            raise OpError("--concurrency needs --ref-requests >= 1 (to measure the unloaded e2e) or --ref-e2e-ms")
        if extend:
            raise OpError("--extend is for single mode")
    label = label or engine_kind
    if not _LABEL.match(label):
        raise OpError("label must be 1-64 characters of letters, digits and _ . + - (it names a directory)")
    extra = _json_object(extra_body, "extra_body")
    try:
        ep = cl.Endpoint.parse(base_url, cl.api_key_from_env(api_key_env))
    except ValueError as e:
        raise OpError(str(e)) from None
    if ep.is_local() and os.environ.get("INFERENCEKIT_CPU_LOCK_HELD"):
        if mode == "load":
            raise OpError("CPU load mode is not supported: capacity metrics currently use GPU normalization; use --mode single")
        if gpu_name or gpu_count != 1:
            raise OpError("CPU rounds cannot use --gpu-name or a non-default --gpu-count")
    if warmup < 0:
        raise OpError("warmup must be >= 0")
    pset = prompts_io.load(prompts, buckets or None, limit_per_bucket or None)
    tok = _tokenizer(tokenizer)
    rcall, key_persistent = _record_call(call, ep)
    runner = Runner(ep, api, engine_kind, model, max_tokens, token_ids, exact_length, extra, tok, timeout_s)
    runner.chat_template_kwargs = _json_object(chat_template_kwargs, "chat_template_kwargs")
    if run_dir and intake_pin(run_dir):  # the intake's pre-registered template settings: added, never contradicted
        base_ctk = extra.get("chat_template_kwargs") if isinstance(extra.get("chat_template_kwargs"), dict) else {}
        runner.chat_template_kwargs = apply_pin({**base_ctk, **runner.chat_template_kwargs}, run_dir, "bench serve")
    slo = {k: v for k, v in (("ttft_p95_ms", slo_ttft_p95_ms), ("tpot_p95_ms", slo_tpot_p95_ms)) if v is not None}
    if slo:
        slo["max_fail"] = slo_max_fail
    store = ev.Store()
    common = dict(call=rcall, store=store, runner=runner, pset=pset, ep=ep, label=label, seed=seed,
                  run_dir=Path(run_dir) if run_dir else None, gpu_name=gpu_name, gpu_count=gpu_count,
                  engine_version=engine_version, resamples=resamples, warmup=warmup,
                  key_note=None if key_persistent else (
                      "no per-machine key file could be written: the endpoint tag in this record is valid for "
                      "this process only, so --extend from another process sees a different base_url"))
    if mode == "single":
        return _single(rounds=rounds, cache_salt=cache_salt, extend=extend, **common)
    return _load(grid=sorted(concurrency or rates), level_kind="concurrency" if concurrency else "rate",
                 duration_s=duration_s, max_requests=max_requests_per_level or None, ref_e2e_ms=ref_e2e_ms,
                 ref_requests=ref_requests, slo=slo, sweep_all=sweep_all, closed_loop=closed_loop,
                 price=gpu_price_usd_h, **common)


# --------------------------------------------------------------------------- single-user mode

def _config_value(key, value):
    """An argument as --extend compares it: JSON-object arguments by value, not by spelling."""
    return normalized_json(value) if key in JSON_ARGS else value


def _load_previous(store, extend, call):
    prev = store.load(extend)
    if prev.get("kind") != "serve_benchmark" or prev["identity"]["workload"].get("mode") != "single":
        raise OpError(f"--extend {extend}: not a single-mode serve_benchmark record")
    problem = ev.integrity_problem(prev)
    if problem:
        raise OpError(f"--extend: {problem}; refusing to build on it")
    prev_args = prev["data"].get("request", {}).get("args", {})
    mine = ev.redact(call.request()["args"])
    diff = [k for k in SAME_CONFIG if _config_value(k, prev_args.get(k)) != _config_value(k, mine.get(k))]
    if prev["identity"]["workload"].get("prompt_set", {}).get("sha256") != prev["data"].get("prompt_set_sha256"):
        diff.append("prompt_set")
    if diff:
        raise OpError(f"--extend needs the same configuration; these differ: {diff}")
    art = next((a for a in prev["artifacts"] if a.get("role") == "requests"), None)
    path = store.run_dir(prev["evidence_id"]) / art["path"] if art else None
    if path is None or not path.is_file() or file_sha256(path)[0] != art["sha256"]:
        raise OpError(f"--extend: the per-request rows of {prev['evidence_id'][:15]} are missing or changed")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return prev, rows


def _single(*, call, store, runner, pset, ep, label, seed, run_dir, gpu_name, gpu_count, engine_version,
            resamples, warmup, key_note, rounds, cache_salt, extend):
    prev, prev_rows = (None, [])
    if extend:
        prev, prev_rows = _load_previous(store, extend, call)
        if prev["data"].get("prompt_set_sha256") != pset.sha256:
            raise OpError("--extend: the prompt set changed since the earlier run")
    offset = max((r["round"] for r in prev_rows), default=-1) + 1
    prompts = list(pset.prompts)

    def salt_for(r: int) -> str | None:
        # Round 0 goes untagged: it is cold anyway (the warm-up is tagged) and its outputs are comparable
        # with an untagged reference. Later rounds are tagged so they cannot hit the prefix cache.
        return f"ik-r{r}" if cache_salt and r > 0 else None

    async def run():
        lim, info = await _check_server(ep, runner.model)
        for i in range(warmup):  # untimed; tagged so warm-up never seeds the prefix cache of a timed prompt
            await runner.send(prompts[i % len(prompts)], f"ik-warm-{i}")
        timed, walls = [], {}
        for r in range(offset, offset + rounds):
            t0 = time.time()
            for p in prompts:
                timed.append((r, p, salt_for(r), await runner.send(p, salt_for(r))))
            walls[r] = (t0, time.time())
        return lim, info, timed, walls

    lim, info, timed, walls = asyncio.run(run())
    with store.staging_dir() as staging:
        rows, out_docs, in_docs, text_sha = [], [], [], {}
        for r, p, salt, res in timed:
            rows.append(metrics.request_row(res, round=r))
            if res.ok and salt is None:
                text_sha.setdefault(p.id, set()).add(hashlib.sha256(res.text.encode()).hexdigest())
            if r == 0:  # outputs and exact request bodies of the untagged round only
                rel = f"{p.bucket}/{p.id}.json"
                out_docs.append((rel, _output_doc(res, p, runner, round=r, salt=salt)))
                in_docs.append((rel, res.request_body or {}))
        all_rows = prev_rows + rows
        idx = _write_run_files(staging, run_dir, label, out_docs, in_docs, contract_inputs=True)
        outputs_from = None if idx or not prev else (prev["data"].get("outputs_from") or prev["evidence_id"])
        req_path = _write_jsonl(staging / "requests.jsonl", all_rows)
        buckets = pset.buckets()
        mets, table = metrics.bucket_metrics(all_rows, buckets, resamples, seed=0)
        round_walls = {int(x["index"]): (x["t_start"], x["t_end"]) for x in (prev or {}).get("data", {}).get(
            "rounds", [])}
        round_walls.update(walls)
        round_rows = metrics.round_rows(all_rows, round_walls)

        max_by_id = {p.id: runner.max_for(p) for p in prompts}
        limitations = list(lim) + _count_limitations(all_rows, (runner.asked or {}).get("exact_length"), max_by_id,
                                                     runner.token_ids, runner.engine_kind)
        failed = [x for x in all_rows if not x["ok"]]
        if failed:
            limitations.append(f"{len(failed)} of {len(all_rows)} request(s) failed (e.g. {failed[0]['error']})")
        if not cache_salt and len(round_rows) > 1:
            limitations.append("--no-cache-salt: identical prompts were resent every round, so rounds after the first "
                               "can be served from the engine's prefix cache; TTFT here is warm-cache TTFT, not "
                               "cold prefill")
        if not ep.is_local():
            limitations.append("server is not on this host: TTFT and e2e include the network round trip")
        if key_note:
            limitations.append(key_note)
        nondet = sorted(k for k, v in text_sha.items() if len(v) > 1)
        if nondet:
            limitations.append(f"{len(nondet)} prompt(s) produced different text across untagged rounds "
                               f"(non-deterministic decoding?), e.g. {nondet[0]}")
        if len(round_rows) < 2:
            limitations.append("fewer than 2 rounds: no confidence interval")

        maxes = sorted({runner.max_for(p) for p in prompts})
        workload = {"gen_len": maxes[0] if len(maxes) == 1 else None, "batch": 1, "seed": seed, "mode": "single",
                    "bucket": buckets[0] if len(buckets) == 1 else None, "prompt_set": pset.identity(),
                    "endpoint_kind": f"openai_{runner.api}", "prefix_cache": "cold" if cache_salt else "warm",
                    "request_set": request_set(pset, runner)}
        ident, missing = _identity(ep, gpu_name, gpu_count, runner.engine_kind, engine_version, runner.model,
                                   workload)
        if not [x for x in all_rows if x["ok"]]:
            missing.append("every request failed")
        artifacts = [ev.artifact(req_path, staging, "requests")]
        if idx:
            artifacts.append(ev.artifact(idx, staging, "outputs_index"))
        a = table["all"]
        summary = (f"{label} {runner.model} ({runner.api}, single user, {len(round_rows)} round(s) x "
                   f"{len(prompts)} prompts): decode {_fmt((a.get('decode_tok_s') or {}).get('median'))} tok/s")
        ci = (a.get("decode_tok_s") or {}).get("ci95_median") or [None, None]
        if ci[0] is not None:
            summary += f" (95% CI {_fmt(ci[0])}-{_fmt(ci[1])})"
        summary += (f", TTFT {_fmt((a.get('ttft_ms') or {}).get('median'))} ms "
                    f"({'cold prefix cache' if cache_salt else 'prompts repeated: warm prefix cache'}), TPOT "
                    f"{_fmt((a.get('tpot_ms') or {}).get('median'), 2)} ms, all buckets; tokens by "
                    + ",".join(sorted({x["token_count_method"] or "none" for x in all_rows if x["ok"]})))
        if failed:
            summary += f"; {len(failed)} failed"
        data = {"label": label, "base_url_kind": "local" if ep.is_local() else "remote", "server": info,
                "prompt_set_sha256": pset.sha256, "prompt_set_name": pset.name, "buckets": table,
                "rounds": round_rows, "request_accounting": _request_accounting(all_rows), "request_profile": {"engine_kind": runner.engine_kind, "api": runner.api,
                                                          "asked": runner.asked,
                                                          "note": profiles.PROFILES[runner.engine_kind].note},
                "prefix_cache": {"mode": "cold" if cache_salt else "warm",
                                 "how": ("warm-up and every round after round 0 prefix the prompt with a unique "
                                         "tag ([ik-warm-N] / [ik-rN], a few tokens); round 0 is untagged"
                                         if cache_salt else "warm-up tagged; every round resends identical "
                                                            "prompts (--no-cache-salt)")},
                "outputs_from": outputs_from,
                "token_count_methods": _method_counts(all_rows), "nondeterministic_prompts": nondet[:50],
                "extends": prev["evidence_id"] if prev else None, "timer": "client perf_counter_ns",
                "definitions": {"ttft_ms": "first token-bearing SSE event - request send (includes connect)",
                                "tpot_ms": "(t_last - t_first) / (n_tokens - tokens in first event)",
                                "decode_tok_s": "(n_tokens - tokens in first event) / (t_last - t_first)",
                                "e2e_ms": "[DONE] (or end of stream) - request send"}}
        if run_dir:
            _write_metrics_file(run_dir, label, all_rows, table)
        record = ev.build(call, kind="serve_benchmark", claim_class="observed", summary=summary, identity_=ident,
                          metrics=mets, artifacts=artifacts, missing=missing, limitations=limitations, data=data)
        return store.save(record, staging)


def _request_accounting(rows):
    """Summarize timed rows after collection; token totals cover successful requests only."""
    ok = [r for r in rows if r["ok"]]
    counts = [n for r in ok if type(n := r.get("n_tokens")) is int and n >= 0]
    return {"timed_requests": len(rows), "successful_requests": len(ok),
            "failed_requests": len(rows) - len(ok), "selected_token_counts": len(counts),
            "unknown_token_counts": len(ok) - len(counts),
            "potentially_estimated_counts": sum(
                1 for r in ok if type(r.get("n_tokens")) is int and r["n_tokens"] >= 0
                and not (r.get("token_count_method") == "usage"
                         or str(r.get("token_count_method") or "").startswith("stream_token_ids:"))),
            "selected_output_tokens_total": sum(counts) if counts else None,
            "selected_output_tokens_min": min(counts) if counts else None,
            "selected_output_tokens_max": max(counts) if counts else None,
            "successful_count_mismatches": sum(bool(r.get("count_mismatch")) for r in ok)}


def _method_counts(rows):
    out = {}
    for r in rows:
        if r["ok"]:
            out[r["token_count_method"] or "none"] = out.get(r["token_count_method"] or "none", 0) + 1
    return out


def _write_metrics_file(run_dir: Path, label: str, rows: list[dict], table: dict) -> None:
    p = Path(run_dir) / "metrics" / f"{label}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    per_input = {}
    for r in rows:
        if r["ok"]:
            d = per_input.setdefault(r["id"], {"bucket": r["bucket"], "ttft_ms": [], "tpot_ms": [],
                                               "decode_tok_s": [], "e2e_ms": []})
            for k in ("ttft_ms", "tpot_ms", "decode_tok_s", "e2e_ms"):
                if r.get(k) is not None:
                    d[k].append(r[k])
    summary = {k: {"bucket": v["bucket"], **{m: (statistics.median(v[m]) if v[m] else None)
                                             for m in ("ttft_ms", "tpot_ms", "decode_tok_s", "e2e_ms")}}
               for k, v in sorted(per_input.items())}
    p.write_text(json.dumps({"label": label, "speed": {"per_bucket": table, "per_input_median": summary}},
                            indent=1) + "\n")


# --------------------------------------------------------------------------- load mode

def _level_name(kind: str, level: float) -> str:
    return f"c{level:g}" if kind == "concurrency" else f"r{level:g}"


def _load(*, call, store, runner, pset, ep, label, seed, run_dir, gpu_name, gpu_count, engine_version,
          resamples, warmup, key_note, grid, level_kind, duration_s, max_requests, ref_e2e_ms, ref_requests, slo,
          sweep_all, closed_loop, price):
    pool = loadgen.PromptPool(list(pset.prompts))
    need_ref = level_kind == "concurrency" and not closed_loop and ref_e2e_ms is None

    def e2e_of(results):
        return [(r.t_done_ns - r.t_send_ns) / 1e6 for r in results if r.ok and r.t_done_ns]

    async def run():
        lim, info = await _check_server(ep, runner.model)
        # Warm-up, then (for a concurrency grid) dedicated reference requests, one at a time: e2e_ref is
        # their median, so slow first requests (lazy init, JIT, a cold engine) do not lower every
        # level's offered rate. Both are tagged and do not use up the pool's untagged prompts.
        order = pool.order(loadgen.level_seed(seed, 10_000))
        warm = [await runner.send(order[i % len(order)], f"ik-warm-{i}") for i in range(warmup)]
        ref = [await runner.send(order[(warmup + i) % len(order)], f"ik-ref-{i}")
               for i in range(ref_requests if need_ref else 0)]
        e2e_ref = ref_e2e_ms if ref_e2e_ms is not None else (statistics.median(e2e_of(ref)) if e2e_of(ref) else None)
        if level_kind == "concurrency" and not closed_loop and not e2e_ref:
            raise OpError("could not measure the unloaded e2e latency (every reference request failed); "
                          "pass --ref-e2e-ms")
        levels = []
        for i, level in enumerate(grid):
            name = _level_name(level_kind, level)
            items_it = pool.stream(loadgen.level_seed(seed, i), name)
            if closed_loop:
                rate = None
                results = await loadgen.closed_loop(runner.send, items_it, int(level), duration_s, max_requests)
            else:
                rate = level if level_kind == "rate" else level / (e2e_ref / 1000.0)
                offsets = loadgen.poisson_schedule(rate, duration_s, loadgen.level_seed(seed, i), max_requests)
                items = [next(items_it) for _ in offsets]
                results = await loadgen.open_loop(runner.send, items, offsets)
            levels.append({"level": level, "name": name, "rate_rps": rate, "results": results})
            rows = [metrics.request_row(r, level=name) for r in results]
            itls = [x for r in results if r.ok for x in metrics.itl_ms(r)]
            levels[-1]["rows"], levels[-1]["summary"] = rows, metrics.level_summary(
                rows, itls, slo, gpu_count, resamples=min(resamples, 2000), seed=0)
            if levels[-1]["summary"]["slo_ok"] is False and not sweep_all:
                break
        return lim, info, e2e_of(warm), e2e_of(ref), e2e_ref, levels

    lim, info, warm_e2e, ref_e2e, e2e_ref, levels = asyncio.run(run())
    with store.staging_dir() as staging:
        docs, all_rows = [], []
        by_id = {q.id: q for q in pset.prompts}
        for lv in levels:
            for res, row in zip(lv["results"], lv["rows"]):
                docs.append((f"load-{lv['name']}/{res.bucket}/{res.id}",
                             (_output_doc(res, by_id[res.id], runner, level=lv["name"], salt=res.salt),
                              res.request_body or {})))
                all_rows.append(row)
        docs = _unique_paths(docs)
        idx = _write_run_files(staging, run_dir, label, [(rel, d[0]) for rel, d in docs],
                               [(rel, d[1]) for rel, d in docs])
        req_path = _write_jsonl(staging / "requests.jsonl", all_rows)

        max_by_id = {p.id: runner.max_for(p) for p in pset.prompts}
        limitations = list(lim) + _count_limitations(all_rows, (runner.asked or {}).get("exact_length"), max_by_id,
                                                     runner.token_ids, runner.engine_kind)
        if closed_loop:
            limitations.insert(0, "CLOSED-LOOP STRESS TEST: workers wait for their previous request, so the arrival "
                                  "rate falls as the server slows; not a capacity measurement and not used for N*")
        if not ep.is_local():
            limitations.append("server is not on this host: TTFT and e2e include the network round trip")
        if level_kind == "concurrency" and not closed_loop:
            src = ("--ref-e2e-ms" if ref_e2e_ms is not None else
                   f"the median of {len(ref_e2e)} reference request(s) after the warm-up")
            limitations.append(f"concurrency N offered as Poisson rate N / e2e_ref, e2e_ref = {e2e_ref:.1f} ms from "
                               f"{src}; measured mean in-flight is reported per level")
        if key_note:
            limitations.append(key_note)
        small = [lv["name"] for lv in levels if lv["summary"]["n_requests"] < 20]
        if small:
            limitations.append(f"levels with fewer than 20 requests (p95 is rough): {', '.join(small)}")
        lag = [lv["name"] for lv in levels if (lv["summary"]["send_lag_ms"].get("p99") or 0) > 5]
        if lag:
            limitations.append(f"client send lag p99 > 5 ms at {', '.join(lag)}: the load generator fell behind "
                               "its schedule (client CPU bound?)")
        if pool.salted:
            limitations.append(f"{pool.salted} request(s) reused a prompt with a unique prefix tag (the pool is "
                               "smaller than the requests sent); the tag adds a few prompt tokens")
        limitations.append("load-level CIs bootstrap requests as if independent; under load they are not, so "
                           "treat those CIs as optimistic")
        skipped = [g for g in grid if g not in [lv["level"] for lv in levels]]
        if skipped:
            limitations.append(f"levels not run after the SLO failed: {', '.join(f'{g:g}' for g in skipped)} "
                               "(--sweep-all runs them)")

        buckets = pset.buckets()
        maxes = sorted({runner.max_for(p) for p in pset.prompts})
        workload = {"gen_len": maxes[0] if len(maxes) == 1 else None, "seed": seed, "mode": "load",
                    "bucket": buckets[0] if len(buckets) == 1 else None, "duration_s": duration_s,
                    "prompt_set": pset.identity(), "slo": slo or None, "endpoint_kind": f"openai_{runner.api}",
                    "prefix_cache": "cold", "request_set": request_set(pset, runner)}
        workload["concurrency" if level_kind == "concurrency" else "rate_rps"] = (
            [int(g) for g in grid] if level_kind == "concurrency" else [float(g) for g in grid])
        ident, missing = _identity(ep, gpu_name, gpu_count, runner.engine_kind, engine_version, runner.model,
                                   workload)
        mets = []
        for lv in levels:
            s, subj = lv["summary"], lv["name"]
            for name, unit in (("ttft_ms", "ms"), ("tpot_ms", "ms"), ("itl_ms", "ms"), ("e2e_ms", "ms")):
                for pct in ("p50", "p95", "p99"):
                    ci = s[name].get("ci95_p95") if pct == "p95" else None
                    mets.append(ev.metric(name, s[name][pct], unit, pct, n=s[name]["n"], ci95=ci, subject=subj))
            mets += [ev.metric("output_tok_s_per_gpu", s["output_tok_s_per_gpu"], "tok/s", "mean",
                               n=s["n_ok"], subject=subj),
                     ev.metric("goodput_rps", s["goodput_rps"], "req/s", "mean", n=s["n_requests"], subject=subj),
                     ev.metric("failed_requests", s["n_failed"], "requests", "count", n=s["n_requests"],
                               subject=subj),
                     ev.metric("mean_inflight", s["mean_inflight"], "requests", "mean", subject=subj)]
        level_data = [{"level": lv["level"], "name": lv["name"], "offered_rate_rps": lv["rate_rps"],
                       "summary": lv["summary"]} for lv in levels]
        base = {"label": label, "base_url_kind": "local" if ep.is_local() else "remote", "server": info,
                "prompt_set_sha256": pset.sha256, "prompt_set_name": pset.name, "level_kind": level_kind,
                "open_loop": not closed_loop, "e2e_ref_ms": e2e_ref,
                "e2e_ref_source": "ref_e2e_ms" if ref_e2e_ms is not None else (
                    "reference_requests" if need_ref else None),
                "warmup_e2e_ms": warm_e2e, "reference_e2e_ms": ref_e2e,
                "levels": level_data, "salted_requests": pool.salted, "timer": "client perf_counter_ns",
                "request_profile": {"engine_kind": runner.engine_kind, "api": runner.api, "asked": runner.asked,
                                    "note": profiles.PROFILES[runner.engine_kind].note},
                "token_count_methods": _method_counts(all_rows), "request_accounting": _request_accounting(all_rows)}
        if not [r for r in all_rows if r["ok"]]:
            missing.append("every request failed")
        kind_word = "closed-loop stress" if closed_loop else "open-loop"
        top = levels[-1] if levels else None
        summary = (f"{label} {runner.model}: {kind_word} load, {len(levels)} level(s) "
                   f"({', '.join(lv['name'] for lv in levels)})")
        if top:
            summary += (f"; at {top['name']}: p95 TTFT {_fmt(top['summary']['ttft_ms']['p95'])} ms, p95 TPOT "
                        f"{_fmt(top['summary']['tpot_ms']['p95'], 2)} ms, {_fmt(top['summary']['output_tok_s_per_gpu'])}"
                        f" out tok/s/GPU, {top['summary']['n_failed']} failed")
        artifacts = [ev.artifact(req_path, staging, "requests")]
        if idx:
            artifacts.append(ev.artifact(idx, staging, "outputs_index"))
        if run_dir:
            p = Path(run_dir) / "metrics" / f"{label}.load.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"label": label, "levels": level_data}, indent=1) + "\n")
        measured = store.save(ev.build(call, kind="serve_benchmark", claim_class="observed", summary=summary,
                                       identity_=ident, metrics=mets, artifacts=artifacts, missing=missing,
                                       limitations=limitations, data=base), staging)

    # The derived record: N* and cost, citing the measurement.
    star, why = metrics.n_star(levels) if not closed_loop else (None, "closed-loop stress test: N* not computed")
    at = next((lv for lv in levels if lv["level"] == star), None)
    s = at["summary"] if at else {}
    tok_s = s.get("output_tok_s_per_gpu")
    inflight = s.get("mean_inflight")
    cost = metrics.usd_per_1m_tokens(tok_s, price, 1)  # per-GPU throughput x per-GPU price
    cmiss = []
    if not slo:
        cmiss.append("slo (pass --slo-ttft-p95-ms and/or --slo-tpot-p95-ms)")
    if price is None:
        cmiss.append("gpu price (pass --gpu-price-usd-h)")
    unit = "streams" if level_kind == "concurrency" else "req/s"
    star_name = "n_star" if level_kind == "concurrency" else "rate_star_rps"
    cmet = [ev.metric(star_name, star, unit, "threshold", n=len(levels)),
            ev.metric("goodput_rps", s.get("goodput_rps"), "req/s", "at_n_star", n=s.get("n_requests")),
            ev.metric("output_tok_s_per_gpu", tok_s, "tok/s", "at_n_star", n=s.get("n_ok")),
            ev.metric("ttft_ms", (s.get("ttft_ms") or {}).get("p95"), "ms", "p95_at_n_star"),
            ev.metric("tpot_ms", (s.get("tpot_ms") or {}).get("p95"), "ms", "p95_at_n_star"),
            ev.metric("mean_inflight", inflight, "requests", "at_n_star", n=s.get("n_requests")),
            ev.metric("usd_per_1m_output_tokens", cost, "USD", "at_n_star")]
    slo_txt = ", ".join(f"{k} <= {v:g}" for k, v in slo.items()) if slo else "no SLO"
    csum = (f"{label} {runner.model}: {'N*' if level_kind == 'concurrency' else 'max rate'} = "
            f"{'n/a' if star is None else f'{star:g}'} {unit} at {slo_txt} ({why})")
    if star is not None and inflight is not None:
        csum += f"; measured mean in-flight at that level {inflight:.2f}"
    if cost is not None:
        csum += f"; ${cost:.3f} per 1M output tokens at ${price:g}/GPU-h"
    clim = [f"derived from {measured['evidence_id']}: see its limitations"]
    if (level_kind == "concurrency" and not closed_loop and star is not None and inflight is not None
            and inflight < INFLIGHT_SHORTFALL * star):
        clim.insert(0, f"N* = {star:g} is the nominal level, but only {inflight:.2f} requests were in flight on "
                       f"average there (< {INFLIGHT_SHORTFALL:g} x N*): the server was offered less load than N* "
                       "streams (short window, Poisson noise, or a high e2e_ref); read capacity as about "
                       f"{inflight:.1f} streams, or rerun with a longer --duration-s / measured --ref-e2e-ms")
    if closed_loop:
        clim.append("closed-loop stress test: no capacity claim")
    if star is not None and star == grid[-1]:
        clim.append("the SLO held at the top of the grid: N* is at least this; extend the grid")
    cap = ev.build(call, kind="capacity", claim_class="derived", summary=csum, identity_=measured["identity"],
                   metrics=cmet, missing=cmiss, limitations=clim,
                   data={"measurement": measured["evidence_id"], "label": label, "level_kind": level_kind,
                         "n_star": star, "reason": why, "mean_inflight_at_n_star": inflight, "slo": slo,
                         "gpu_price_usd_h": price,
                         "gpu_count": gpu_count, "cost_formula": "price_per_gpu_h / (output_tok_s_per_gpu * 3600) "
                                                                 "* 1e6",
                         "curve": [{"level": lv["level"], "slo_ok": lv["summary"]["slo_ok"],
                                    "ttft_p95_ms": lv["summary"]["ttft_ms"]["p95"],
                                    "tpot_p95_ms": lv["summary"]["tpot_ms"]["p95"],
                                    "output_tok_s_per_gpu": lv["summary"]["output_tok_s_per_gpu"],
                                    "goodput_rps": lv["summary"]["goodput_rps"],
                                    "mean_inflight": lv["summary"]["mean_inflight"]} for lv in levels]})
    return store.save(cap)


def _unique_paths(docs: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """``<level>/<bucket>/<id>.json``; a prompt sent twice in a level gets ``<id>~2.json`` and so on."""
    seen: dict[str, int] = {}
    out = []
    for rel, doc in docs:
        k = seen.get(rel, 0) + 1
        seen[rel] = k
        out.append((f"{rel}.json" if k == 1 else f"{rel}~{k}.json", doc))
    return out
