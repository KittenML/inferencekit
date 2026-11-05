"""Numbers from timed requests: per request, per bucket (single-user mode) and per load level.

Per request (all from client timestamps, see client.py):
  ttft_ms        t_first - t_send
  e2e_ms         t_done - t_send
  decode_tok_s   (n_tokens - k_first) / (t_last - t_first), k_first = tokens in the first token event
                 (1 when the server sent no ids). "Tokens after the first, over time after the first",
                 counted from ids so a server that packs several tokens into one event is not undercounted.
  tpot_ms        (t_last - t_first) / (n_tokens - k_first) = 1000 / decode_tok_s
  itl_ms         gaps between consecutive token events (per event, as most serving benchmarks report)

Percentiles are nearest-rank (always an observed value), the same convention the bootstrap uses, so a
point estimate and its CI agree.

Single-user summaries: value = median (and p95) over every request of a bucket; 95% CI by cluster
bootstrap with the *round* as the unit (stats.bootstrap_ci), because requests inside one round share
the server's state and are not independent.

Load-level summaries: p50/p95/p99 over the level's requests; the p95 CI is a plain bootstrap over
requests (under load, requests interact, so treat it as optimistic; the record says so).
"""

from __future__ import annotations

from inferencekit import evidence as ev
from inferencekit import stats
from inferencekit.serve.client import RequestResult

SPEED_METRICS = (("ttft_ms", "ms"), ("tpot_ms", "ms"), ("decode_tok_s", "tok/s"), ("e2e_ms", "ms"))


def pct(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile (an observed value), as inside stats.bootstrap_ci."""
    return stats.nearest_rank(sorted(values), p)


def _ms(a: int | None, b: int | None) -> float | None:
    return (a - b) / 1e6 if a is not None and b is not None else None


def request_row(res: RequestResult, **extra) -> dict:
    """One request's numbers (no text): a row of requests.jsonl."""
    ttft = _ms(res.t_first_ns, res.t_send_ns)
    e2e = _ms(res.t_done_ns if res.t_done_ns is not None else res.t_end_ns, res.t_send_ns)
    n = res.n_tokens
    k_first = res.events[0].n_ids if res.events and res.events[0].n_ids else 1
    span = _ms(res.t_last_ns, res.t_first_ns)
    after = n - k_first if n is not None else None
    decode = tpot = None
    if res.ok and after and after > 0 and span and span > 0:
        decode = after / (span / 1000.0)
        tpot = span / after
    per_event = [e.n_ids for e in res.events if e.n_ids]
    row = {"id": res.id, "bucket": res.bucket, "salt": res.salt, "ok": res.ok, "error": res.error, "status": res.status,
           "ttft_ms": ttft if res.ok else None, "tpot_ms": tpot, "decode_tok_s": decode,
           "e2e_ms": e2e if res.ok else None, "connect_ms": _ms(res.t_connected_ns, res.t_send_ns),
           "n_tokens": n, "token_count_method": res.token_count_method, "count_mismatch": res.count_mismatch,
           "n_ids": res.n_ids, "n_usage": res.n_usage,
           "finish_reason": res.finish_reason, "n_token_events": len(res.events),
           "tokens_per_event_max": max(per_event) if per_event else None,
           "send_lag_ms": _ms(res.t_send_ns, res.t_scheduled_ns), "t_send_ns": res.t_send_ns,
           "t_end_ns": res.t_end_ns}
    row.update(extra)
    return row


def itl_ms(res: RequestResult) -> list[float]:
    ev_ = res.events
    return [(b.t_ns - a.t_ns) / 1e6 for a, b in zip(ev_, ev_[1:])]


def timings(res: RequestResult) -> dict:
    """The per-event timeline stored with each output (ms after t_send, and ids per event)."""
    return {"t_send_ns": res.t_send_ns,
            "event_ms": [round((e.t_ns - res.t_send_ns) / 1e6, 4) for e in res.events],
            "event_token_ids": [e.n_ids for e in res.events]}


# --------------------------------------------------------------------------- single-user mode

def bucket_metrics(rows: list[dict], buckets: list[str], resamples: int, seed: int = 0) -> tuple[list[dict], dict]:
    """Evidence metrics per bucket (and 'all'): median and p95, with CIs over rounds. Also a plain dict."""
    out, table = [], {}
    for subject in list(buckets) + ["all"]:
        sel = [r for r in rows if r["ok"] and (subject == "all" or r["bucket"] == subject)]
        table[subject] = {"n_requests": len(sel)}
        for name, unit in SPEED_METRICS:
            vals = [(r[name], r["round"]) for r in sel if r.get(name) is not None]
            values, units = [v for v, _ in vals], [u for _, u in vals]
            if not values:
                out.append(ev.metric(name, None, unit, "median", n=0, subject=subject))
                table[subject][name] = None
                continue
            ci = stats.bootstrap_ci(values, units, ps=(50, 95), resamples=resamples, seed=seed)
            med, p95 = pct(values, 50), pct(values, 95)
            out.append(ev.metric(name, med, unit, "median", n=len(values), ci95=ci[50], subject=subject))
            out.append(ev.metric(name, p95, unit, "p95", n=len(values), ci95=ci[95], subject=subject))
            table[subject][name] = {"median": med, "p95": p95, "ci95_median": list(ci[50]),
                                    "ci95_p95": list(ci[95]), "n": len(values), "rounds": len(set(units))}
    return out, table


def round_rows(rows: list[dict], round_walltimes: dict[int, tuple[float, float]]) -> list[dict]:
    """One row per round (medians over the round's requests), in the row shape of bench decode records
    (``compare`` reads that shape; it accepts serve_benchmark records only once that lands at merge)."""
    out = []
    for r, (t0, t1) in sorted(round_walltimes.items()):
        sel = [x for x in rows if x["round"] == r and x["ok"]]
        row = {"index": r, "t_start": t0, "t_end": t1, "n_requests": len(sel),
               "n_failed": sum(1 for x in rows if x["round"] == r and not x["ok"])}
        for name, key in (("decode_tok_s", "decode_tok_s"), ("ttft_ms", "ttft_ms"), ("tpot_ms", "tpot_ms"),
                          ("total_ms", "e2e_ms")):
            vals = [x[key] for x in sel if x.get(key) is not None]
            row[name] = pct(vals, 50) if vals else None
        out.append(row)
    return out


# --------------------------------------------------------------------------- load mode

def _pcts(values: list[float]) -> dict:
    if not values:
        return {"p50": None, "p95": None, "p99": None, "n": 0}
    return {"p50": pct(values, 50), "p95": pct(values, 95), "p99": pct(values, 99), "n": len(values)}


def meets_slo(row: dict, slo: dict) -> bool:
    """Per-request SLO attainment (for goodput): succeeded, TTFT and TPOT inside their targets."""
    if not row["ok"]:
        return False
    if slo.get("ttft_p95_ms") is not None and (row["ttft_ms"] is None or row["ttft_ms"] > slo["ttft_p95_ms"]):
        return False
    if slo.get("tpot_p95_ms") is not None and row["tpot_ms"] is not None and row["tpot_ms"] > slo["tpot_p95_ms"]:
        return False
    return True


def level_summary(rows: list[dict], itls: list[float], slo: dict, gpu_count: int, resamples: int = 2000,
                  seed: int = 0) -> dict:
    """One load level: latency percentiles, failures, goodput, throughput and whether the SLO held."""
    ok = [r for r in rows if r["ok"]]
    t0 = min((r["t_send_ns"] for r in rows), default=0)
    t1 = max((r["t_end_ns"] for r in rows), default=0)
    window_s = (t1 - t0) / 1e9 if rows else 0.0
    out_tokens = sum(r["n_tokens"] or 0 for r in ok)
    s = {"n_requests": len(rows), "n_ok": len(ok), "n_failed": len(rows) - len(ok), "window_s": window_s,
         "ttft_ms": _pcts([r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]),
         "tpot_ms": _pcts([r["tpot_ms"] for r in ok if r["tpot_ms"] is not None]),
         "e2e_ms": _pcts([r["e2e_ms"] for r in ok if r["e2e_ms"] is not None]),
         "itl_ms": _pcts(itls),
         "output_tokens": out_tokens,
         "output_tok_s": out_tokens / window_s if window_s > 0 else None,
         "output_tok_s_per_gpu": out_tokens / window_s / max(1, gpu_count) if window_s > 0 else None,
         "goodput_rps": sum(1 for r in rows if meets_slo(r, slo)) / window_s if window_s > 0 else None,
         "achieved_rps": len(ok) / window_s if window_s > 0 else None,
         # Little's law: time-averaged requests in flight over the window
         "mean_inflight": sum((r["t_end_ns"] - r["t_send_ns"]) for r in rows) / 1e9 / window_s if window_s else None,
         "send_lag_ms": _pcts([r["send_lag_ms"] for r in rows if r.get("send_lag_ms") is not None])}
    for name in ("ttft_ms", "tpot_ms"):
        vals = [r[name] for r in ok if r[name] is not None]
        s[name]["ci95_p95"] = list(stats.bootstrap_ci(vals, ps=(95,), resamples=resamples, seed=seed)[95])
    s["slo_checks"] = slo_checks(s, slo)
    s["slo_ok"] = all(c["ok"] for c in s["slo_checks"].values()) if s["slo_checks"] else None
    return s


def slo_checks(s: dict, slo: dict) -> dict:
    checks = {}
    if slo.get("ttft_p95_ms") is not None:
        v = s["ttft_ms"]["p95"]
        checks["ttft_p95_ms"] = {"value": v, "target": slo["ttft_p95_ms"], "ok": v is not None and v <= slo["ttft_p95_ms"]}
    if slo.get("tpot_p95_ms") is not None:
        v = s["tpot_ms"]["p95"]
        checks["tpot_p95_ms"] = {"value": v, "target": slo["tpot_p95_ms"], "ok": v is not None and v <= slo["tpot_p95_ms"]}
    if checks:  # failures only count when an SLO was given
        mf = slo.get("max_fail") or 0
        checks["failed"] = {"value": s["n_failed"], "target": mf, "ok": s["n_failed"] <= mf and s["n_ok"] > 0}
    return checks


def n_star(levels: list[dict]) -> tuple[float | None, str]:
    """Highest level at which the SLO holds at that level and at every lower one.

    Returns (level or None, reason). A level that passes above a failing one does not count.
    """
    if not levels:
        return None, "no load levels ran"
    best = None
    for lv in sorted(levels, key=lambda x: x["level"]):
        if lv["summary"].get("slo_ok") is None:
            return None, "no SLO given"
        if not lv["summary"]["slo_ok"]:
            failed = [k for k, c in lv["summary"]["slo_checks"].items() if not c["ok"]]
            return best, f"SLO first fails at level {lv['level']:g} ({', '.join(failed)})"
        best = lv["level"]
    return best, "SLO holds at every level tried: the true N* may be above the grid"


def usd_per_1m_tokens(tok_s: float | None, price_usd_h: float | None, gpu_count: int = 1) -> float | None:
    """$ per 1M output tokens = hourly price of the GPUs / tokens produced per hour x 1e6."""
    if not tok_s or not price_usd_h or tok_s <= 0:
        return None
    return price_usd_h * max(1, gpu_count) / (tok_s * 3600.0) * 1e6
