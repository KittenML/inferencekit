"""report: one fixed-format write-up from a set of evidence ids.

Sections, always in this order: verdict, environment card, results table, findings,
limitations, reproduce. A fixed shape makes reports comparable and makes omissions
visible (an empty limitations section is a claim in itself).

Claim verdicts: PASS | FAIL | UNPROVEN | NOT RUN.
  comparison   expect=improved: improved -> PASS, regressed / within_noise -> FAIL,
               inconclusive -> UNPROVEN. expect=no_regression: improved / within_noise
               -> PASS, regressed -> FAIL, inconclusive -> UNPROVEN. A PASS is
               downgraded to UNPROVEN when rounds were not verified as interleaved or
               either arm violates a hard gate.
  correctness  its own PASS / FAIL / UNPROVEN.
  benchmark    "runs under production-like settings": hard gates clean -> PASS,
               violated or unchecked -> UNPROVEN (the number is real, the claim is not).
  any record whose id does not match its content -> UNPROVEN.
Overall: any FAIL -> FAIL, else any UNPROVEN -> UNPROVEN, else PASS; no claims -> NOT RUN.
"""

from __future__ import annotations

import json

from inferencekit import evidence as ev
from inferencekit import registry, stats
from inferencekit.canonical import sha256_hex
from inferencekit.registry import Arg, operation

COMPARISON_TO_CLAIM = {
    "improved": {"improved": "PASS", "regressed": "FAIL", "within_noise": "FAIL", "inconclusive": "UNPROVEN"},
    "no_regression": {"improved": "PASS", "regressed": "FAIL", "within_noise": "PASS", "inconclusive": "UNPROVEN"},
}


def short(eid: str | None) -> str:
    return (eid or "?")[:15]


def _fmt(x, digits: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{digits}g}" if abs(x) < 1e6 else f"{x:.4e}"
    return str(x)


def _cell(x) -> str:
    return str(x).replace("|", "\\|").replace("\n", " ")


def claims_of(rec: dict, expect: str, integrity: dict) -> list[dict]:
    kind, data, eid = rec.get("kind"), rec.get("data", {}), rec.get("evidence_id")
    out = []
    if kind == "comparison":
        verdict = COMPARISON_TO_CLAIM[expect].get(data.get("verdict"), "UNPROVEN")
        reason = f"{data.get('metric')} {data.get('verdict')}: {data.get('decision', {}).get('reason', '')}"
        if verdict == "PASS" and data.get("interleaved") is not True:
            verdict, reason = "UNPROVEN", reason + "; rounds not verified as interleaved"
        if verdict == "PASS" and any("hard gates" in lim for lim in rec.get("limitations", [])):
            verdict, reason = "UNPROVEN", reason + "; an arm violates hard gates"
        out.append({"source": eid, "claim": f"candidate vs baseline ({expect})", "verdict": verdict,
                    "reason": reason})
        if data.get("source_kind") in ("image_benchmark", "audio_benchmark"):
            for claim, proof in data.get("required_proofs", {}).items():
                out.append({"source": eid, "claim": claim, "verdict": proof,
                            "reason": "A timing comparison does not supply this proof; cite external modality evidence."})
            if data.get("source_kind") == "audio_benchmark":
                for arm, timing in data.get("arm_timing_verdicts", {}).items():
                    out.append({"source": eid, "claim": f"{arm} speech latency and streaming SLOs",
                                "verdict": timing if timing in ("PASS", "FAIL") else "UNPROVEN",
                                "reason": "Absolute first-audio, RTF and underrun limits from the source benchmark."})
    elif kind in ("image_benchmark", "audio_benchmark"):
        validity = data.get("claim_validity") or {}
        structure = "PASS" if validity.get("valid") is True else "FAIL" if validity.get("valid") is False else "UNPROVEN"
        out.append({"source": eid, "claim": "complete media output and matching request settings",
                    "verdict": structure, "reason": "; ".join(validity.get("violations", []))})
        if kind == "audio_benchmark":
            timing = data.get("timing_verdict")
            out.append({"source": eid, "claim": "speech latency and streaming SLOs",
                        "verdict": timing if timing in ("PASS", "FAIL") else "UNPROVEN",
                        "reason": "First-audio p95, complete-audio RTF and simulated playback underruns."})
        out.append({"source": eid, "claim": "media quality and usability",
                    "verdict": "UNPROVEN", "reason": "Timing and structural checks do not establish image quality, "
                    "speech intelligibility, speaker fidelity or repeated-run reliability; cite task-specific quality evidence."})
    elif kind == "correctness":
        out.append({"source": eid, "claim": "greedy outputs agree", "verdict": data.get("verdict", "UNPROVEN"),
                    "reason": data.get("reason", "")})
    elif kind == "benchmark":
        validity = data.get("claim_validity") or {}
        if validity.get("valid") is True:
            out.append({"source": eid, "claim": "benchmark passes hard gates", "verdict": "PASS",
                        "reason": "checked: " + ", ".join(validity.get("checked", []))})
        else:
            why = "; ".join(validity.get("violations", [])) or "engine flags not visible: gates not checked"
            out.append({"source": eid, "claim": "benchmark passes hard gates", "verdict": "UNPROVEN", "reason": why})
    if not integrity["id_ok"]:
        for c in out:
            c["verdict"], c["reason"] = "UNPROVEN", "evidence id does not match content (edited after capture?)"
        if not out:
            out.append({"source": eid, "claim": "evidence integrity", "verdict": "UNPROVEN",
                        "reason": "evidence id does not match content"})
    return out


def workload_text(w: dict) -> str:
    """One line for the environment card: bench decode's prompt/gen/batch, or bench serve's mode, bucket,
    generation length, load grid and prompt set (a serve record has no single prompt length)."""
    if w.get("prompt_len") is not None:
        return f"prompt {w.get('prompt_len')}, gen {w.get('gen_len')}, batch {w.get('batch')}"
    if not w.get("mode"):
        return "-"
    parts = [f"{w['mode']}"]
    parts.append(f"bucket {w['bucket']}" if w.get("bucket") else "all buckets")
    if w.get("gen_len") is not None:
        parts.append(f"gen {w['gen_len']}")
    if w.get("concurrency") is not None:
        parts.append(f"concurrency {w['concurrency']}")
    if w.get("rate_rps") is not None:
        parts.append(f"rate {w['rate_rps']} req/s")
    ps = w.get("prompt_set") or {}
    if ps.get("name") or ps.get("sha256"):
        parts.append(f"set {ps.get('name') or '?'} ({str(ps.get('sha256') or '')[:12]})")
    if w.get("prefix_cache"):
        parts.append(f"{w['prefix_cache']} prefix cache")
    return ", ".join(parts)


def environment_rows(recs: list[dict]) -> list[dict]:
    seen, rows = set(), []
    for r in recs:
        i = r.get("identity", {})
        g, e, m, w = i.get("gpu", {}), i.get("engine", {}), i.get("model", {}), i.get("workload", {})
        gpu = g.get("name") or "?"
        row = {"gpu": f"{g['count']}x {gpu}" if g.get("count") else gpu,
               "driver / CUDA": f"{g.get('driver') or '?'} / {g.get('cuda') or '?'}", "clocks": g.get("clocks") or "?",
               "engine": " ".join(str(x) for x in (e.get("name"), e.get("version")) if x) or "-",
               "model": " ".join(str(x) for x in (m.get("id"), m.get("dtype"), m.get("quant")) if x) or "-",
               "workload": workload_text(w)}
        key = json.dumps(row, sort_keys=True)
        if key not in seen:
            seen.add(key)
            rows.append(row)
    return rows


def context_note(roof: dict, bench: dict, tolerance: float = 0.01) -> str:
    """A caveat when the roofline's KV-cache context does not match the benchmark's.

    The ceiling counts the KV cache at one context length. During a benchmark the context
    grows from prompt_len to prompt_len + gen_len, so its mean is prompt_len + gen_len / 2.
    If that changes the bytes read per token by more than ``tolerance``, the "% of ceiling"
    compares different workloads.
    """
    counts = roof.get("data", {}).get("counts", {})
    w = bench.get("identity", {}).get("workload", {})
    kv_per_ctx, per_token, ctx = (counts.get("kv_bytes_per_context_token"), counts.get("bytes_per_token"),
                                  counts.get("context"))
    if not kv_per_ctx or not per_token or ctx is None or w.get("prompt_len") is None:
        return ""
    mean_ctx = w["prompt_len"] + (w.get("gen_len") or 0) / 2
    diff = kv_per_ctx * abs(mean_ctx - ctx) / per_token
    if diff <= tolerance:
        return ""
    return (f" (roofline context {ctx} vs benchmark mean context {mean_ctx:.0f}: bytes per token differ by "
            f"~{100 * diff:.1f}%; rerun roofline --context {mean_ctx:.0f})")


def findings(recs: list[dict]) -> list[str]:
    out = [f"`{short(r['evidence_id'])}` ({r['kind']}, {r['claim_class']}): {r['summary']}" for r in recs]
    # Cross-record: measured decode speed as a share of the bandwidth ceiling.
    ceilings = [(r, m) for r in recs if r["kind"] == "roofline" for m in r["metrics"] if m["name"] == "ceiling_tok_s"]
    for b in (r for r in recs if r["kind"] == "benchmark"):
        tok = next((m["value"] for m in b["metrics"] if m["name"] == "decode_tok_s"), None)
        for roof, m in ceilings:
            if tok and m["value"]:
                same = roof["identity"]["model"].get("id") == b["identity"]["model"].get("id")
                note = "" if same else " (model ids differ: check they are the same weights)"
                note += context_note(roof, b)
                out.append(f"`{short(b['evidence_id'])}` decodes at {100 * tok / m['value']:.1f}% of the "
                           f"bandwidth ceiling from `{short(roof['evidence_id'])}` ({tok:.1f} of {m['value']:.1f} "
                           f"tok/s){note}")
    for r in recs:
        if r["kind"] == "ncu_kernels":
            for k in r["data"].get("kernels", [])[:8]:
                out.append(f"`{short(r['evidence_id'])}` kernel #{k['index']} `{k['kernel'][:60]}`: {k['bound']}, "
                           f"DRAM {_fmt(k['dram_pct'])}% / SM {_fmt(k['sm_pct'])}% of peak")
        if r["kind"] == "nsys_timeline":
            hist = r["data"].get("idle_gap_histogram", {})
            small = hist.get("10-100us", {}).get("count", 0)
            if small:
                out.append(f"`{short(r['evidence_id'])}`: {small} idle gaps of 10-100 us (typical of per-kernel "
                           f"launch overhead; CUDA graphs remove most of them)")
    return out


def reproduce_command(rec: dict) -> str | None:
    req = rec.get("data", {}).get("request")
    if not req:
        return None
    try:
        op = registry.get(req["op"])
    except registry.OpError:
        return None
    args = req.get("args", {})
    parts = ["inferencekit", *op.cli]
    tail = []
    for a in op.args:
        v = args.get(a.name)
        if v is None or v == a.default:
            continue
        shown = f"<{v['digest'][:19]}...>" if isinstance(v, dict) and "digest" in v else v
        if a.name == "argv":
            tail = ["--", str(shown)]
        elif a.positional:
            parts += [str(x) for x in (shown if isinstance(shown, list) else [shown])]
        elif a.type == "boolean":
            parts.append(f"--{a.name.replace('_', '-')}" if v else f"--no-{a.name.replace('_', '-')}")
        elif isinstance(shown, list):
            for x in shown:
                parts += [f"--{a.name.replace('_', '-')}", json.dumps(x) if not isinstance(x, str) else x]
        else:
            text = str(shown)
            parts += [f"--{a.name.replace('_', '-')}", f"'{text}'" if (" " in text or "{" in text) else text]
    return " ".join(parts + tail)


def _subject(rec: dict, m: dict) -> str:
    """The metric's subject; for bench serve records also the mode (and the record's bucket when the
    metric does not name one), so a serve row never reads like a bench decode row."""
    subject = m.get("subject") or ""
    w = (rec.get("identity") or {}).get("workload") or {}
    if rec.get("kind") not in ("serve_benchmark", "capacity") or not w.get("mode"):
        return subject
    if not subject and w.get("bucket"):
        subject = f"bucket {w['bucket']}"
    return f"{w['mode']}: {subject}" if subject else str(w["mode"])


def build_report(recs: list[dict], expect: str, title: str) -> dict:
    integrity = {r["evidence_id"]: ev.verify(r) for r in recs}
    claims = [c for r in recs for c in claims_of(r, expect, integrity[r["evidence_id"]])]
    verdict = stats.combine_claims(c["verdict"] for c in claims)
    results = []
    for r in recs:
        for m in r.get("metrics", [])[:15]:
            results.append({"evidence": short(r["evidence_id"]), "kind": r["kind"], "metric": m["name"],
                            "subject": _subject(r, m), "value": m["value"], "unit": m["unit"],
                            "stat": m["stat"], "n": m.get("n"), "ci95": m.get("ci95")})
    limitations = []
    for r in recs:
        for lim in r.get("limitations", []):
            limitations.append({"evidence": short(r["evidence_id"]), "text": lim})
        if not r.get("coverage", {}).get("complete", True):
            limitations.append({"evidence": short(r["evidence_id"]),
                                "text": "incomplete coverage, missing: " + ", ".join(r["coverage"]["missing"])})
    return {"title": title, "verdict": verdict, "expect": expect, "claims": claims,
            "environment": environment_rows(recs), "results": results, "findings": findings(recs),
            "limitations": limitations,
            "reproduce": [{"evidence": short(r["evidence_id"]), "command": reproduce_command(r)} for r in recs],
            "evidence": [r["evidence_id"] for r in recs], "integrity": integrity}


def to_markdown(rep: dict) -> str:
    lines = [f"# {rep['title']}", "", "## Verdict", "", f"**{rep['verdict']}**", ""]
    if rep["claims"]:
        lines += ["| claim | verdict | evidence | why |", "|---|---|---|---|"]
        lines += [f"| {_cell(c['claim'])} | {c['verdict']} | `{short(c['source'])}` | {_cell(c['reason'])} |"
                  for c in rep["claims"]]
    else:
        lines.append("No comparison, correctness or benchmark evidence: nothing was claimed (NOT RUN).")
    lines += ["", "## Environment card", ""]
    cols = ["gpu", "driver / CUDA", "clocks", "engine", "model", "workload"]
    lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(_cell(row[c]) for c in cols) + " |" for row in rep["environment"]]
    lines += ["", "## Results", "", "| evidence | metric | subject | value | unit | stat | n | 95% CI |",
              "|---|---|---|---|---|---|---|---|"]
    for x in rep["results"]:
        ci = x["ci95"]
        ci_text = f"{_fmt(ci[0])} - {_fmt(ci[1])}" if ci and ci[0] is not None else "-"
        lines.append(f"| `{x['evidence']}` | {x['metric']} | {_cell(x['subject'])[:60]} | {_fmt(x['value'])} | "
                     f"{x['unit']} | {x['stat']} | {_fmt(x['n'])} | {ci_text} |")
    lines += ["", "## Findings", ""] + [f"- {_cell(f)}" for f in rep["findings"]]
    lines += ["", "## Limitations", ""]
    lines += [f"- `{x['evidence']}`: {x['text']}" for x in rep["limitations"]] or ["- none recorded"]
    lines += ["", "## Reproduce", "", "```sh"]
    lines += [f"# {x['evidence']}\n{x['command'] or '(no recorded request)'}" for x in rep["reproduce"]]
    lines += ["```", "", "Evidence ids (full): " + ", ".join(f"`{e}`" for e in rep["evidence"]), ""]
    return "\n".join(lines)


@operation("report", ("report",), "Fixed-format report from evidence ids: verdict, environment card, results "
           "table, findings, limitations, reproduce commands.",
           [Arg("ids", "array", "Evidence ids (or unique prefixes) to include.", required=True, positional=True),
            Arg("format", "string", "Output format.", default="md", choices=("md", "json")),
            Arg("expect", "string", "What the comparisons are meant to show.", default="improved",
                choices=("improved", "no_regression")),
            Arg("title", "string", "Report title.", default="Inference optimisation report")])
def report(call, ids, format, expect, title):  # noqa: A002
    store = ev.Store()
    recs, seen = [], set()
    for ref in ids:
        rec = store.load(ref)
        if rec["evidence_id"] not in seen:
            seen.add(rec["evidence_id"])
            recs.append(rec)
    rep = build_report(recs, expect, title)
    out_dir = store.root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "report-" + sha256_hex("|".join(rep["evidence"]) + expect)[:12]
    if format == "json":
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(rep, indent=2) + "\n")
        return {"format": "json", "path": str(path), "verdict": rep["verdict"], "report": rep}
    text = to_markdown(rep)
    path = out_dir / f"{name}.md"
    path.write_text(text)
    return {"format": "md", "path": str(path), "verdict": rep["verdict"], "markdown": text}
