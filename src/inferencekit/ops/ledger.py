"""ledger trailers / history: every kept commit carries its numbers as git trailers.

``ledger trailers <ev...>`` turns evidence records into a trailer block for a commit message:

    Perf-Workload: Qwen/Qwen3-8B FP8, RTX 5090, vllm 0.11.0, batch 1, prompt 1024, gen 256
    Perf-Decode-Tok-S: 185.3 (95% CI 185.2-185.3) [+2.1% vs 1a2b3c4]
    Perf-TTFT-Ms: 60.7 (95% CI 60.1-61.0)
    Perf-P95-Latency-Ms: 1452
    Perf-Quality: greedy prefix agreement 100.00% (5/5 identical) PASS
    Perf-Cost: $0.93 per 1M output tokens at $0.62/GPU-h (1 stream)
    Perf-Capacity-N: 24 streams (SLO p95 TTFT <= 300 ms, p95 TPOT <= 50 ms, failures <= 0)
    Perf-Goodput: 18.2 req/s at N*=24
    Perf-Evidence: ev_3f2a..., ev_9c1d...
    Perf-Baseline: 1a2b3c4

Rules: a metric the evidence does not contain is omitted (listed under ``omitted`` with the
reason), never written as zero. Values are read by field presence, so v1 ``benchmark`` and
v2 ``serve_benchmark`` / ``capacity`` / ``quality_*`` records both work. The delta against a
baseline comes from a ``comparison`` record when one is given (ratio of medians with its
CI and verdict); otherwise from ``--baseline-ev`` as a plain ratio of medians, labelled
"ratio of medians, no CI" inside the trailer itself. Values never contain a newline, so
evidence text cannot forge extra trailers.

One commit, one engine: speed and capacity records of more than one engine label (or more
than one single-user speed record) are refused unless ``--label`` picks one; the baseline's
record goes in ``--baseline-ev``. Prices are per GPU-hour, as ``bench serve`` takes them: a
capacity record's own cost is printed at its own price, and a different ``--gpu-price-usd-h``
reprices it from the record's per-GPU throughput at N* (derived). In text mode the trailer
block goes to stdout (for ``git commit -m "$(...)"``) and what was omitted or limited goes to
stderr, so it is seen but never lands in the commit message.

``history`` reads the trailers back with ``git log``, prints a table and a sparkline per
metric (oldest to newest), and with ``--tag-best`` points local tags ``perf/best-<metric>``
at the best commit. Tags are never pushed.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

from inferencekit import evidence as ev
from inferencekit.canonical import sha256_hex
from inferencekit.proc import run as run_cmd
from inferencekit.registry import Arg, OpError, operation

ORDER = ("Perf-Workload", "Perf-Decode-Tok-S", "Perf-TTFT-Ms", "Perf-P95-Latency-Ms", "Perf-Quality", "Perf-Cost",
         "Perf-Capacity-N", "Perf-Goodput", "Perf-Evidence", "Perf-Baseline")
SPEED_KINDS = ("benchmark", "serve_benchmark")
# Metric names accepted for each headline (v1 names first; v2 aliases after).
DECODE = ("decode_tok_s", "decode_tok_s_median")
TTFT = ("ttft_ms", "ttft_p50_ms", "ttft_median_ms")
P95_DIRECT = ("e2e_p95_ms", "p95_e2e_ms", "latency_p95_ms", "e2e_latency_p95_ms", "total_p95_ms")
P95_BY_STAT = ("e2e_ms", "e2e_latency_ms", "latency_ms", "total_ms")
N_STAR = ("n_star", "capacity_n", "n_star_streams")
GOODPUT = ("goodput_rps", "goodput_req_s", "goodput")
THROUGHPUT_PER_GPU = ("output_tok_s_per_gpu", "output_tok_s", "throughput_tok_s")
COST = ("usd_per_1m_output_tokens", "cost_usd_per_1m_output_tokens", "usd_per_1m_tokens", "cost_per_1m_tokens_usd")
CENTRAL = (None, "median", "p50", "mean")
SPARK = "▁▂▃▄▅▆▇█"
MAX_SUBJECTS = 6


def short(eid: str | None) -> str:
    return (eid or "?")[:15]


def one_line(text) -> str:
    """Trailer values are one line: newlines (and other control characters) become spaces."""
    return re.sub(r"[\x00-\x1f\x7f]+", " ", str(text)).strip()


def num(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    if float(x).is_integer() and abs(x) < 1e15:
        return str(int(x))
    if abs(x) >= 1000:
        return f"{x:.0f}"
    if abs(x) >= 100:
        return f"{x:.1f}"
    return f"{x:.{digits}g}"


# --------------------------------------------------------------------------- reading records

def metrics_named(rec: dict, names, stats=CENTRAL) -> list[dict]:
    out = []
    for name in names:
        for m in rec.get("metrics") or []:
            if not isinstance(m, dict) or m.get("name") != name or m.get("value") is None:
                continue
            if stats is not None and m.get("stat") not in stats:
                continue
            out.append(m)
        if out:
            return out
    return out


def pick(recs: list[dict], names, stats=CENTRAL, kinds=None) -> tuple[list[dict], dict] | None:
    """The first record (in the order given) holding metric ``names``: (metrics, record).

    A metric without a subject/bucket is the overall value and wins; otherwise every
    per-subject value is returned (e.g. one per input bucket).
    """
    for rec in recs:
        if kinds and rec.get("kind") not in kinds:
            continue
        found = metrics_named(rec, names, stats)
        if not found:
            continue
        overall = [m for m in found if m.get("subject") in (None, "", "all") and not m.get("bucket")]
        return (overall[:1] if overall else found[:MAX_SUBJECTS]), rec
    return None


def kind_of(rec: dict) -> str | None:
    """The record's kind; v2 kinds written as a v1 kind plus data.v2_kind (before the v2 schema merged) count."""
    data = rec.get("data") or {}
    return data.get("v2_kind") or ("gates" if data.get("gates_format") else rec.get("kind"))


def gpu_count(rec: dict) -> int:
    g = (rec.get("identity") or {}).get("gpu") or {}
    n = g.get("count") or (rec.get("data") or {}).get("gpu_count") or 1
    return int(n) if isinstance(n, (int, float)) and n >= 1 else 1


def is_load(rec: dict) -> bool:
    return ((rec.get("identity") or {}).get("workload") or {}).get("mode") == "load"


def label_of(rec: dict) -> str | None:
    label = (rec.get("data") or {}).get("label") or ((rec.get("identity") or {}).get("engine") or {}).get("name")
    return str(label) if label else None


def with_ci(m: dict) -> str:
    text = num(m["value"])
    ci = m.get("ci95")
    if isinstance(ci, list) and len(ci) == 2 and None not in ci:
        text += f" (95% CI {num(ci[0])}-{num(ci[1])})"
    return text


def values_text(ms: list[dict]) -> str:
    if len(ms) == 1:
        return with_ci(ms[0])
    # Several subjects (e.g. buckets) and no overall value: the first number is the first subject's.
    return "; ".join(f"{with_ci(m)} {one_line(m.get('subject') or m.get('bucket'))}" for m in ms)


def slo_text(slo: dict | None) -> str:
    if not isinstance(slo, dict):
        return ""
    parts = []
    if slo.get("ttft_p95_ms") is not None:
        parts.append(f"p95 TTFT <= {num(slo['ttft_p95_ms'])} ms")
    if slo.get("tpot_p95_ms") is not None:
        parts.append(f"p95 TPOT <= {num(slo['tpot_p95_ms'])} ms")
    if slo.get("max_fail") is not None:
        parts.append(f"failures <= {slo['max_fail']}")
    return ", ".join(parts)


def workload_text(rec: dict) -> str:
    ident = rec.get("identity") or {}
    m, g, e, w = (ident.get(k) or {} for k in ("model", "gpu", "engine", "workload"))
    parts = []
    model = " ".join(str(x) for x in (m.get("id"), (m.get("quant") or m.get("dtype") or "").upper() or None) if x)
    if model:
        parts.append(model)
    if g.get("name"):
        parts.append(f"{g['count']}x {g['name']}" if (g.get("count") or 1) > 1 else str(g["name"]))
    if e.get("name"):
        parts.append(" ".join(str(x) for x in (e.get("name"), e.get("version")) if x))
    if w.get("mode"):
        parts.append({"single": "single-user", "load": "open-loop load"}.get(w["mode"], str(w["mode"])))
    for key, label in (("batch", "batch"), ("concurrency", "concurrency"), ("rate_rps", "rate"),
                       ("prompt_len", "prompt"), ("gen_len", "gen"), ("bucket", "bucket")):
        v = w.get(key)
        if v is not None:
            shown = ",".join(num(x) if isinstance(x, (int, float)) else str(x) for x in v) if isinstance(v, list) \
                else (num(v) if isinstance(v, float) else v)
            parts.append(f"{label} {shown}" + (" req/s" if key == "rate_rps" else ""))
    ps = w.get("prompt_set")
    if isinstance(ps, dict) and ps.get("name"):
        parts.append(f"prompts {ps['name']}" + (f" ({str(ps['sha256'])[:12]})" if ps.get("sha256") else ""))
    return one_line(", ".join(parts))


# --------------------------------------------------------------------------- building trailers

class Builder:
    def __init__(self):
        self.trailers: list[dict] = []
        self.omitted: list[dict] = []
        self.limitations: list[str] = []

    def add(self, key: str, value: str, claim_class: str, sources: list[str]):
        self.trailers.append({"key": key, "value": one_line(value), "claim_class": claim_class,
                              "sources": [s for s in sources if s]})

    def omit(self, key: str, reason: str):
        self.omitted.append({"key": key, "reason": reason})

    def text(self) -> str:
        rank = {k: i for i, k in enumerate(ORDER)}
        rows = sorted(self.trailers, key=lambda t: rank.get(t["key"], len(ORDER)))
        return "\n".join(f"{t['key']}: {t['value']}" for t in rows)


def delta(metric: str, rec: dict, value: float, comparisons: list[dict], base: dict | None, label: str | None,
          higher_is_better: bool, b: Builder) -> str:
    """'[+2.1% (95% CI ...) improved vs 1a2b3c4]' from a comparison record, else a plain ratio of medians."""
    for c in comparisons:
        d = c.get("data") or {}
        if d.get("metric") != metric or (c is not rec and d.get("candidate") not in (None, rec.get("evidence_id"))):
            continue
        ratio = next((m for m in c.get("metrics") or [] if m.get("name") == f"{metric}_ratio"), None)
        if not ratio or ratio.get("value") is None:
            continue
        text = f"{(ratio['value'] - 1) * 100:+.1f}%"
        ci = ratio.get("ci95")
        if isinstance(ci, list) and len(ci) == 2 and None not in ci:
            text += f" (95% CI {(ci[0] - 1) * 100:+.1f}% to {(ci[1] - 1) * 100:+.1f}%)"
        if d.get("verdict"):
            text += f" {d['verdict']}"
        return f" [{text} vs {label or short(d.get('baseline'))}]"
    if base is None:
        return ""
    got = pick([base], DECODE if metric == "decode_tok_s" else TTFT if metric == "ttft_ms" else (metric,))
    if not got or len(got[0]) != 1 or not got[0][0].get("value"):
        b.limitations.append(f"baseline {short(base.get('evidence_id'))} has no overall {metric}: no delta")
        return ""
    ratio = value / got[0][0]["value"]
    b.limitations.append(f"{metric} delta is a ratio of medians without a CI or verdict; run `inferencekit compare` "
                         "on interleaved rounds for that")
    return f" [{(ratio - 1) * 100:+.1f}% vs {label or short(base.get('evidence_id'))}, ratio of medians, no CI]"


def build_trailers(recs: list[dict], base: dict | None, baseline_commit: str | None,
                   gpu_price_usd_h: float | None) -> Builder:
    b = Builder()
    comparisons = [r for r in recs if kind_of(r) == "comparison"]
    label = baseline_commit[:7] if baseline_commit else (short(base["evidence_id"]) if base else None)
    # Single-user headlines come from single-user records; load levels feed capacity, not TTFT.
    speed_all = [r for r in recs if kind_of(r) in SPEED_KINDS]
    speed_recs = [r for r in speed_all if not is_load(r)]
    engine = next((label_of(r) for r in speed_all if label_of(r)), None)
    capacity = [r for r in recs if kind_of(r) == "capacity"]
    capacity.sort(key=lambda r: label_of(r) != engine)  # the measured engine's capacity record first

    first = (speed_recs or speed_all or capacity or recs or [None])[0]
    wl = workload_text(first) if first else ""
    if wl:
        b.add("Perf-Workload", wl, "observed", [first.get("evidence_id")])
    else:
        b.omit("Perf-Workload", "no identity in the evidence")

    decode_value = None
    for key, names, metric, higher in (("Perf-Decode-Tok-S", DECODE, "decode_tok_s", True),
                                       ("Perf-TTFT-Ms", TTFT, "ttft_ms", False)):
        got = pick(speed_recs, names)
        if not got:  # a comparison record alone still holds the candidate's median
            got = next(((ms, c) for c in comparisons for ms in [[m for m in c.get("metrics") or []
                        if m.get("name") == metric and m.get("subject") == "candidate" and m.get("value") is not None]]
                        if ms), None)
        if not got:
            b.omit(key, f"no {metric} median in single-user benchmark evidence"
                   + (" (load-mode records feed Perf-Capacity-N, not this)" if speed_all else ""))
            continue
        ms, rec = got
        text, change = values_text(ms), ""
        if len(ms) == 1:
            change = delta(metric, rec, ms[0]["value"], comparisons, base, label, higher, b)
            if metric == "decode_tok_s":
                decode_value = (ms[0]["value"], rec)
        cls = "derived" if change or rec.get("kind") == "comparison" else rec.get("claim_class", "observed")
        b.add(key, text + change, cls, [rec.get("evidence_id")])

    got = pick(speed_recs, P95_DIRECT, stats=None) or pick(speed_recs, P95_BY_STAT, stats=("p95",))
    if got:
        b.add("Perf-P95-Latency-Ms", values_text(got[0]), got[1].get("claim_class", "observed"),
              [got[1].get("evidence_id")])
    else:
        b.omit("Perf-P95-Latency-Ms", "no single-user p95 end-to-end latency in the evidence (bench decode does "
               "not measure it; bench serve does)")

    quality, qsrc = quality_parts(recs, engine)
    if quality:
        b.add("Perf-Quality", "; ".join(quality), "derived", qsrc)
    else:
        b.omit("Perf-Quality", "no correctness / quality evidence")

    cost_trailer(b, capacity, decode_value, gpu_price_usd_h)

    n_star = None
    got = pick(capacity, N_STAR, stats=None) or pick(capacity, ("rate_star_rps",), stats=None)
    if got:
        m, rec = got[0][0], got[1]
        data = rec.get("data") or {}
        slo = data.get("slo") or ((rec.get("identity") or {}).get("workload") or {}).get("slo")
        s = slo_text(slo)
        # N* is server-wide (concurrent streams or arrival rate at the server), not per GPU.
        # The SLO held at the top of the grid: N* is a lower bound.
        lower = "at least " if str(data.get("reason") or "").startswith("SLO holds at every level") else ""
        if m["name"] == "rate_star_rps" or data.get("level_kind") == "rate":
            value = f"max rate {lower}{num(m['value'])} req/s"
        else:
            n_star = m["value"]
            value = f"{lower}{num(n_star)} streams"
        if gpu_count(rec) > 1:
            value += f" across {gpu_count(rec)} GPUs"
        b.add("Perf-Capacity-N", value + (f" (SLO {s})" if s else " (SLO not recorded)"),
              rec.get("claim_class", "observed"), [rec.get("evidence_id")])
        if not s:
            b.limitations.append("capacity evidence carries no SLO: N* cannot be interpreted without it")
    elif capacity:
        why = (capacity[0].get("data") or {}).get("reason") or "N* undefined"
        b.omit("Perf-Capacity-N", f"capacity evidence has no N*: {why}")
    else:
        b.omit("Perf-Capacity-N", "no capacity evidence with N*")
    got = pick(capacity, GOODPUT, stats=None)
    if got:
        m, rec = got[0][0], got[1]
        unit = m.get("unit") or "req/s"
        b.add("Perf-Goodput", f"{num(m['value'])} {unit}" + (f" at N*={num(n_star)}" if n_star is not None else ""),
              rec.get("claim_class", "observed"), [rec.get("evidence_id")])
    else:
        b.omit("Perf-Goodput", "no goodput in capacity evidence")

    ids = [r.get("evidence_id") for r in recs] + ([base.get("evidence_id")] if base else [])
    b.add("Perf-Evidence", ", ".join(short(i) for i in dict.fromkeys(ids) if i), "observed", [])
    baseline = baseline_commit or (short(base["evidence_id"]) if base else None)
    if baseline is None:
        baseline = next((short((c.get("data") or {}).get("baseline")) for c in comparisons
                         if (c.get("data") or {}).get("baseline")), None)
    if baseline:
        b.add("Perf-Baseline", baseline, "observed", [])
    else:
        b.omit("Perf-Baseline", "no --baseline-commit, --baseline-ev or comparison record")
    return b


def _subject_metric(rec: dict, names, subject: str | None, others=()) -> dict | None:
    """The metric named in ``names`` for ``subject`` (the engine label), else the first engine-level one.
    Metrics whose subject is in ``others`` (another server's, e.g. the sampler check's baseline) never stand in."""
    for name in names:
        ms = [m for m in rec.get("metrics") or [] if isinstance(m, dict) and m.get("name") == name
              and m.get("value") is not None and m.get("subject") not in others]
        if not ms:
            continue
        exact = [m for m in ms if subject is not None and m.get("subject") == subject]
        floor = (rec.get("data") or {}).get("noise_floor")
        plain = [m for m in ms if "@" not in str(m.get("subject") or "") and m.get("subject") != floor]
        found = (exact or plain or ms)[0]
        return found
    return None


def quality_parts(recs: list[dict], engine: str | None) -> tuple[list[str], list[str]]:
    """One phrase per quality record. The engine label goes last, so the leading number stays the value."""
    parts, sources = [], []
    for r in recs:
        kind, data = kind_of(r), r.get("data") or {}
        if kind == "correctness":
            pa = metrics_named(r, ("prefix_agreement_pct",), stats=None)
            em = metrics_named(r, ("exact_match_pct",), stats=None)
            if pa:
                text = f"greedy prefix agreement {pa[0]['value']:.2f}%"
                if em and em[0].get("n"):
                    text += f" ({round(em[0]['value'] * em[0]['n'] / 100)}/{em[0]['n']} identical)"
                if isinstance(data.get("verdict"), str):
                    text += f" {data['verdict']}"
                parts.append(text)
                sources.append(r.get("evidence_id"))
        elif kind == "quality_trajectory":
            m = _subject_metric(r, ("top1_agreement_pct", "tf_agreement_pct", "token_agreement_pct"), engine)
            if not m:
                continue
            subject = m.get("subject")
            text = f"{m['name'].replace('_pct', '').replace('_', '-')} {m['value']:.2f}%"
            floor_label = data.get("noise_floor")
            floor = next((x for x in r.get("metrics") or [] if x.get("name") == m["name"] and floor_label
                          and x.get("subject") == floor_label and x.get("value") is not None), None)
            if floor:
                text += f" (noise floor {floor['value']:.2f}%)"
            verdict = (data.get("verdicts") or {}).get(subject) or data.get("verdict")
            if isinstance(verdict, str):
                text += f" {verdict}"
            parts.append(text + (f" [{one_line(subject)}]" if subject else ""))
            sources.append(r.get("evidence_id"))
        elif kind in ("quality_task", "quality"):
            task = str(data.get("task") or "task")
            if task == "sampling":
                # only this server's own number: with --baseline-url the record also carries the baseline's
                m = _subject_metric(r, ("first_token_tv_excess",), engine, others=("baseline",))
                if not m:
                    who = data.get("label") or engine
                    parts.append("first token tv excess n/a" + (f" {data['verdict']}" if isinstance(
                        data.get("verdict"), str) else "") + (f" [{one_line(who)}]" if who else ""))
                    sources.append(r.get("evidence_id"))
                    continue
            else:
                m = _subject_metric(r, (f"{task}_accuracy_pct", "accuracy_pct", "score"), engine)
            if not m:
                m = next((x for x in r.get("metrics") or [] if x.get("value") is not None
                          and any(w in x.get("name", "") for w in ("accuracy", "wer", "score", "agreement"))), None)
            if not m:
                continue
            name = m["name"].replace("_pct", "").replace("_", " ")
            unit = "%" if m.get("unit") == "%" or m["name"].endswith("_pct") else ""
            text = f"{name} {num(m['value'])}{unit}"
            ci = m.get("ci95")
            if isinstance(ci, list) and len(ci) == 2 and None not in ci:
                sep = " to " if min(ci) < 0 else "-"  # a signed number (TV excess): "-0.0085-0.0116" misreads
                text += f" (95% CI {num(ci[0])}{sep}{num(ci[1])}{unit})"
            if m.get("n"):
                text += f" (n={m['n']})"
            if isinstance(data.get("verdict"), str):
                text += f" {data['verdict']}"
            parts.append(text + (f" [{one_line(m['subject'])}]" if m.get("subject") else ""))
            sources.append(r.get("evidence_id"))
    return parts, sources


def cost_trailer(b: Builder, capacity: list[dict], decode_value, price: float | None) -> None:
    """Perf-Cost. Prices are $ per GPU-hour (as bench serve takes them).

    A capacity record's own cost is printed at the price it was computed with. A different
    ``--gpu-price-usd-h`` reprices it from the record's per-GPU throughput at N* (derived);
    the record's number is never shown next to a price it was not computed at.
    """
    got = pick(capacity, COST, stats=None)
    if got:
        m, rec = got[0][0], got[1]
        own = (rec.get("data") or {}).get("gpu_price_usd_h")
        own = float(own) if isinstance(own, (int, float)) and not isinstance(own, bool) and own > 0 else None
        if price is None or (own is not None and abs(price - own) < 1e-9):
            b.add("Perf-Cost", f"${m['value']:.2f} per 1M output tokens" + (f" at ${own:.2f}/GPU-h" if own else "")
                  + " (at N*)", rec.get("claim_class", "observed"), [rec.get("evidence_id")])
            if own is None:
                b.limitations.append("Perf-Cost: the capacity record does not say which GPU price it used")
            return
        tput = pick([rec], ("output_tok_s_per_gpu",), stats=None)
        if tput and len(tput[0]) == 1 and tput[0][0]["value"]:
            usd = price / (tput[0][0]["value"] * 3600) * 1e6
            b.add("Perf-Cost", f"${usd:.2f} per 1M output tokens at ${price:.2f}/GPU-h (at N*, repriced)", "derived",
                  [rec.get("evidence_id")])
            b.limitations.append(f"Perf-Cost repriced at ${price:.2f}/GPU-h from output_tok_s_per_gpu at N*; capacity "
                                 f"record {short(rec.get('evidence_id'))} priced it at "
                                 + (f"${own:.2f}/GPU-h (${m['value']:.2f})" if own else "an unrecorded price"))
            return
        b.add("Perf-Cost", f"${m['value']:.2f} per 1M output tokens" + (f" at ${own:.2f}/GPU-h" if own else "")
              + " (at N*)", rec.get("claim_class", "observed"), [rec.get("evidence_id")])
        b.limitations.append(f"--gpu-price-usd-h {price:g} ignored: capacity record {short(rec.get('evidence_id'))} "
                             "has no per-GPU throughput at N* to reprice from, so its own cost and price are shown")
        return
    if price is None:
        b.omit("Perf-Cost", "no --gpu-price-usd-h and no cost in capacity evidence")
        return
    got = pick(capacity, THROUGHPUT_PER_GPU, stats=None)
    if got and got[0][0]["value"]:
        tps, rec, note = got[0][0]["value"], got[1], "at N*"
        # Server-wide throughput (not per GPU) is paid for by every GPU of the server.
        gpus = 1 if got[0][0]["name"] == "output_tok_s_per_gpu" else gpu_count(rec)
    elif decode_value and decode_value[0]:
        (tps, rec), note = decode_value, "1 stream"
        gpus = gpu_count(rec)  # one decode stream occupies every GPU of a tensor-parallel engine
        b.limitations.append("Perf-Cost uses single-stream decode speed: serving many users per GPU costs less; "
                             "add capacity evidence (bench serve load mode) for the cost at N*")
    else:
        b.omit("Perf-Cost", "no throughput to price")
        return
    usd = price * gpus / (tps * 3600) * 1e6
    at = f"${price:.2f}/GPU-h" + (f" x {gpus} GPUs" if gpus > 1 else "")
    b.add("Perf-Cost", f"${usd:.2f} per 1M output tokens at {at} ({note})", "derived", [rec.get("evidence_id")])


def select_engine(recs: list[dict], label: str | None) -> list[dict]:
    """The records of the one engine a commit measures; refuses anything ambiguous.

    Two engines' speed records (or a baseline's and a candidate's passed together) would
    otherwise put the first one's number in the commit as its headline.
    """
    measured = SPEED_KINDS + ("capacity",)
    labels = sorted({label_of(r) for r in recs if kind_of(r) in measured and label_of(r)})
    if label is not None:
        if label not in labels:
            raise OpError(f"--label {label!r}: no benchmark or capacity record has that engine label; found: "
                          f"{', '.join(labels) or 'none'}")
        recs = [r for r in recs if kind_of(r) not in measured or label_of(r) == label]
    elif len(labels) > 1:
        raise OpError(f"benchmark/capacity records of {len(labels)} engine labels ({', '.join(labels)}): pass --label "
                      "for the engine this commit measures, and the baseline's record as --baseline-ev (or a "
                      "`compare` record for a delta with a CI)")
    for what, group in (("single-user benchmark", [r for r in recs if kind_of(r) in SPEED_KINDS and not is_load(r)]),
                        ("capacity", [r for r in recs if kind_of(r) == "capacity"])):
        if len(group) > 1:
            raise OpError(f"{len(group)} {what} records for one engine "
                          f"({', '.join(short(r.get('evidence_id')) for r in group)}): pass only the one this commit "
                          "measures; a baseline goes in --baseline-ev")
    return recs


def _load(store: ev.Store, ref: str, what: str) -> dict:
    rec = store.load(ref)
    problem = ev.integrity_problem(rec)
    if problem:  # a ledger entry is permanent: never write numbers from an edited record into it
        raise OpError(f"{what} {problem}; refusing to write trailers from it")
    return rec


@operation("ledger_trailers", ("ledger", "trailers"), "Git trailers (Perf-Decode-Tok-S, Perf-TTFT-Ms, Perf-Quality, "
           "Perf-Cost, Perf-Capacity-N...) for a commit message, from evidence ids; missing metrics are omitted, "
           "never zero.",
           [Arg("ids", "array", "Evidence ids (or unique prefixes, or paths): benchmark, comparison, correctness, "
                "quality and capacity records.", required=True, positional=True),
            Arg("baseline_commit", "string", "Commit the change is measured against (shown in the delta and "
                "Perf-Baseline)."),
            Arg("baseline_ev", "string", "Baseline benchmark evidence id, for the delta when no comparison record "
                "is given."),
            Arg("label", "string", "Engine label the trailers describe, when the evidence holds several engines."),
            Arg("gpu_price_usd_h", "number", "GPU price in $ per GPU-hour (as bench serve takes it), for Perf-Cost; "
                "multiplied by the GPU count where the throughput is the whole server's."),
            Arg("format", "string", "text: the trailer block on stdout, what was omitted or limited on stderr; "
                "json: trailers with claim classes, sources and what was omitted.", default="text",
                choices=("text", "json"))])
def ledger_trailers(call, ids, baseline_commit, label, baseline_ev, gpu_price_usd_h, format):  # noqa: A002
    store = ev.Store()
    recs, seen = [], set()
    for ref in ids:
        rec = _load(store, ref, "evidence")
        if rec.get("evidence_id") not in seen:
            seen.add(rec.get("evidence_id"))
            recs.append(rec)
    base = _load(store, baseline_ev, "baseline") if baseline_ev else None
    if baseline_commit is not None and not re.fullmatch(r"[0-9A-Za-z._/-]{4,64}", baseline_commit):
        raise OpError(f"--baseline-commit {baseline_commit!r} does not look like a commit or ref")
    if gpu_price_usd_h is not None and gpu_price_usd_h <= 0:
        raise OpError("--gpu-price-usd-h must be > 0")
    recs = select_engine(recs, label)
    b = build_trailers(recs, base, baseline_commit, gpu_price_usd_h)
    text = b.text()
    out = {"trailers": b.trailers, "omitted": b.omitted, "limitations": b.limitations, "text": text,
           "evidence": [r.get("evidence_id") for r in recs] + ([base.get("evidence_id")] if base else [])}
    if format == "json":
        return out
    out_dir = store.root / "ledger"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"trailers-{sha256_hex(text)[:12]}.txt"
    path.write_text(text + "\n")
    # Provenance for the derived numbers (deltas, repriced costs): claim class and source record per trailer.
    path.with_suffix(".json").write_text(json.dumps({"request": call.request(), "generated": call.started, **out},
                                                    indent=2, ensure_ascii=False) + "\n")
    # stdout is the commit message ($(...) captures it); the caveats go to stderr so they are seen, not committed.
    for o in b.omitted:
        print(f"inferencekit ledger: omitted {o['key']}: {o['reason']}", file=sys.stderr)
    for lim in b.limitations:
        print(f"inferencekit ledger: limitation: {lim}", file=sys.stderr)
    return {"format": "md", "markdown": text, "path": str(path), **out}


# --------------------------------------------------------------------------- history

HIGHER_IS_BETTER = {"Perf-Decode-Tok-S": True, "Perf-Quality": True, "Perf-Capacity-N": True,
                    "Perf-Goodput": True, "Perf-TTFT-Ms": False, "Perf-P95-Latency-Ms": False, "Perf-Cost": False}
NOT_METRICS = ("Perf-Workload", "Perf-Evidence", "Perf-Baseline")
_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?")


def leading_number(value: str) -> float | None:
    """The first free-standing number of a trailer value ('$0.91 per 1M' -> 0.91; 'gsm8k 85%' -> 85)."""
    m = _NUMBER.search(value or "")
    return float(m.group(0)) if m else None


def direction(key: str, sample: str = "") -> bool | None:
    """True if higher is better, False if lower is, None if unknown. ``sample``: a value of that trailer."""
    if key == "Perf-Quality" and re.search(r"\b(c|w)er\b|\bkl\b|error", sample, re.IGNORECASE):
        return False  # an error rate or a divergence, e.g. "asr-wer 2.1%"
    if key in HIGHER_IS_BETTER:
        return HIGHER_IS_BETTER[key]
    low = key.lower()
    if low.endswith(("-ms", "-s-latency")) or "cost" in low or "latency" in low:
        return False
    if low.endswith(("-tok-s", "-rps")) or "goodput" in low or "throughput" in low:
        return True
    return None


def metric_key(name: str) -> str:
    """'decode_tok_s', 'decode-tok-s' or 'Perf-Decode-Tok-S' -> 'Perf-Decode-Tok-S' (case-insensitive match later)."""
    n = name.strip().replace("_", "-")
    return n if n.lower().startswith("perf-") else "Perf-" + n


def sparkline(values: list[float | None]) -> str:
    """One character per commit; '·' where a commit has no value (never drawn as zero)."""
    present = [v for v in values if v is not None]
    if not present:
        return ""
    lo, hi = min(present), max(present)
    out = []
    for v in values:
        if v is None:
            out.append("·")
        elif hi == lo:
            out.append(SPARK[len(SPARK) // 2])
        else:
            out.append(SPARK[min(len(SPARK) - 1, int((v - lo) / (hi - lo) * (len(SPARK) - 1) + 0.5))])
    return "".join(out)


def git_log(repo: str, rev: str, limit: int) -> list[dict]:
    fmt = "%H%x1f%h%x1f%cI%x1f%s%x1f%(trailers:only,unfold)%x1e"
    res = run_cmd(["git", "-C", repo, "log", f"--max-count={limit}", f"--format={fmt}", rev, "--"], timeout=60)
    if res.returncode == 127:
        raise OpError("git not found on PATH")
    if not res.ok:
        raise OpError(f"git log failed in {repo}: {res.stderr.strip()[:300]}")
    commits = []
    for chunk in res.stdout.split("\x1e"):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        fields = chunk.split("\x1f")
        if len(fields) < 5:
            continue
        sha, short_sha, date, subject, trailer_text = fields[:5]
        trailers: dict[str, str] = {}
        for line in trailer_text.splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip().lower().startswith("perf-"):
                trailers.setdefault(key.strip(), value.strip())  # first occurrence wins
        commits.append({"commit": sha, "short": short_sha, "date": date, "subject": subject, "trailers": trailers})
    return commits


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|")


@operation("history", ("history",), "Commits with Perf-* trailers from git log: a table, a sparkline per metric, "
           "the best commit per metric, and optional local perf/best-<metric> tags.",
           [Arg("repo", "string", "Git repository.", default="."),
            Arg("metric", "string", "Only this metric (e.g. decode-tok-s or Perf-TTFT-Ms)."),
            Arg("rev", "string", "Revision or range to read.", default="HEAD"),
            Arg("limit", "integer", "Most recent commits to read.", default=200),
            Arg("tag_best", "boolean", "Point local tags perf/best-<metric> at the best commit per metric (never "
                "pushed).", default=False),
            Arg("format", "string", "md: table and sparklines; json: parsed commits.", default="md",
                choices=("md", "json"))])
def history(call, repo, metric, rev, limit, tag_best, format):  # noqa: A002
    if not Path(repo).is_dir():
        raise OpError(f"repository not found: {repo}")
    if rev.startswith("-"):
        raise OpError(f"rev {rev!r} must not start with '-'")
    commits = [c for c in git_log(repo, rev, limit) if c["trailers"]]
    commits.reverse()  # oldest first: sparklines read left to right in time
    keys: list[str] = []
    for c in commits:
        for k in c["trailers"]:
            if k not in NOT_METRICS and k not in keys:
                keys.append(k)
    keys.sort(key=lambda k: (ORDER.index(k) if k in ORDER else len(ORDER), k))
    if metric:
        want = metric_key(metric).lower()
        keys = [k for k in keys if k.lower() == want]
        if not keys:
            raise OpError(f"no commits carry trailer {metric_key(metric)} in {repo} ({rev})")
    for c in commits:
        c["values"] = {k: leading_number(c["trailers"][k]) for k in keys if k in c["trailers"]}

    # Only numbers measured on the same workload are comparable: the newest commit's workload decides.
    latest_wl = next((c["trailers"].get("Perf-Workload") for c in reversed(commits)
                      if c["trailers"].get("Perf-Workload")), None)
    comparable = [c for c in commits if latest_wl is None or c["trailers"].get("Perf-Workload") in (None, latest_wl)]
    notes = []
    if len(comparable) < len(commits):
        notes.append(f"{len(commits) - len(comparable)} commit(s) measured another Perf-Workload: listed, but left out "
                     f"of sparklines and best-of (workload: {latest_wl})")

    summary = {}
    for k in keys:
        series = [c["values"].get(k) for c in comparable]
        present = [(c, c["values"][k]) for c in comparable if c["values"].get(k) is not None]
        d = direction(k, next((c["trailers"][k] for c in reversed(comparable) if k in c["trailers"]), ""))
        best = None
        if present and d is not None:
            target = max(v for _, v in present) if d else min(v for _, v in present)
            best_c = [c for c, v in present if v == target][-1]  # newest commit on a tie
            best = {"commit": best_c["commit"], "short": best_c["short"], "value": target}
        summary[k] = {"sparkline": sparkline(series), "n": len(present), "higher_is_better": d, "best": best,
                      "first": present[0][1] if present else None, "last": present[-1][1] if present else None,
                      "median": statistics.median([v for _, v in present]) if present else None}
        if d is None and present:
            notes.append(f"{k}: direction unknown (not a standard trailer), no best commit chosen")

    tags = []
    if tag_best:
        for k, s in summary.items():
            if not s["best"]:
                continue
            tag = "perf/best-" + re.sub(r"[^a-z0-9-]+", "-", k[len("Perf-"):].lower()).strip("-")
            # A lightweight local bookmark, moved when a better commit appears; never signed or pushed.
            res = run_cmd(["git", "-C", repo, "-c", "tag.gpgSign=false", "tag", "-f", tag, s["best"]["commit"]],
                          timeout=30)
            if not res.ok:
                raise OpError(f"git tag {tag} failed: {res.stderr.strip()[:300]}")
            tags.append({"tag": tag, "commit": s["best"]["short"], "value": s["best"]["value"]})

    out = {"repo": str(repo), "rev": rev, "commits": commits, "metrics": summary, "tags": tags, "notes": notes,
           "claim_class": "as recorded in each commit's trailers (see its Perf-Evidence)"}
    if format == "json":
        return out
    lines = [f"# Perf history: {Path(repo).resolve().name} ({rev}, {len(commits)} commits with Perf-* trailers)", ""]
    if not commits:
        lines.append("No commits with Perf-* trailers. Add them with `inferencekit ledger trailers <ev...>`.")
    else:
        short_keys = [k[len("Perf-"):] for k in keys]
        lines += ["| commit | date | subject | " + " | ".join(short_keys) + " |",
                  "|---|---|---|" + "---|" * len(keys)]
        for c in reversed(commits):  # newest first in the table, like git log
            vals = [num(c["values"][k]) if c["values"].get(k) is not None else "·" for k in keys]
            lines.append(f"| {c['short']} | {c['date'][:10]} | {_cell(c['subject'][:60])} | " + " | ".join(vals) + " |")
        lines += ["", "Sparklines, oldest to newest (· = no value on that commit):", ""]
        width = max((len(k) for k in short_keys), default=0)
        for k, sk in zip(keys, short_keys):
            s = summary[k]
            best = (f", best {num(s['best']['value'])} at {s['best']['short']}"
                    f" ({'higher' if s['higher_is_better'] else 'lower'} is better)") if s["best"] else ""
            lines.append(f"    {sk.ljust(width)}  {s['sparkline']}  first {num(s['first'])}, last {num(s['last'])}"
                         f"{best}")
    if tags:
        lines += ["", "Local tags (not pushed): " + ", ".join(f"{t['tag']} -> {t['commit']}" for t in tags)]
    if notes:
        lines += [""] + [f"Note: {n}" for n in notes]
    text = "\n".join(lines) + "\n"
    out_dir = ev.Store().root / "ledger"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"history-{sha256_hex(json.dumps([repo, rev, metric, text]))[:12]}.md"
    path.write_text(text)
    return {"format": "md", "markdown": text, "path": str(path), **out}
