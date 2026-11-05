"""A stand-in for the inferencekit CLI in the ikstep proof-job tests (stdlib only).

It answers the commands a job script runs, writes evidence records built from real records
(tests/fixtures/ikstep), and prints what the CLI's terse view prints (the summary, then `evidence: <path>`).
Every call is appended to $FAKE_IK_LOG as a JSON list. $FAKE_IK (a JSON object) sets the outcomes:
  fail {command: exit code}   sleep {command: seconds}   tok {label: decode tok/s}   rounds {"base": x, "cand": y}
  verdict {command: verdict}  gate_fail [gate names]     wrong_every {label: n} (GSM8K: every n-th problem wrong)
  cells "exited_early" (every bench rounds cell)          engine "fastdec-flashattn" (default) or "fastdec" (score)
  replay {command: fixture} (save that real record as is)  norecord [commands] (exit 0, print a summary, save nothing)
  prompts {"verdict": ..., "exit": n, "stderr": text} (prompts verify)
  failed {label: n} (bench serve: n of its 16 requests failed, from the long bucket down, as the CLI records them)
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "ikstep"
cfg = json.loads(os.environ.get("FAKE_IK") or "{}")
argv = [a for a in sys.argv[1:] if a != "--json"]
cmd = " ".join(argv[:2])
if os.environ.get("FAKE_IK_LOG"):
    with open(os.environ["FAKE_IK_LOG"], "a") as f:
        f.write(json.dumps(argv) + "\n")


def opt(name, default=None):
    vals = [argv[i + 1] for i, x in enumerate(argv[:-1]) if x == name]
    return vals[-1] if vals else default


if cmd == "prompts verify":  # the frozen ops/prompts.py result: a FAIL verdict still exits 0
    pv = {"verdict": "PASS", "exit": 0, **cfg.get("prompts", {})}
    print(pv.get("stderr", ""), end="", file=sys.stderr)
    print(json.dumps({"set": "dev", "path": os.environ["FAKE_IK_PROMPTS"], "verdict": pv["verdict"], "checks": [
        {"check": "sha256", "ok": pv["verdict"] == "PASS", "detail": f"sha256 {pv['verdict']}: file vs manifest"}]})
          * (pv["exit"] == 0))
    sys.exit(pv["exit"])
time.sleep(cfg.get("sleep", {}).get(cmd, 0))
if cmd in cfg.get("fail", {}):
    print(f"inferencekit: error: {cmd} failed on purpose", file=sys.stderr)
    sys.exit(cfg["fail"][cmd])
run, label = Path(opt("--run-dir", ".")), opt("--label")
verdict = cfg.get("verdict", {}).get(cmd, "PASS")


def record(fixture, summary, **data):
    r = json.loads((FIX / f"{(replay := cfg.get('replay', {}).get(cmd)) or fixture}.json").read_text())
    if cmd in cfg.get("norecord", []):
        return print(summary)
    r["evidence_id"] = "ev_" + hashlib.sha256(json.dumps([argv, time.time()]).encode()).hexdigest()
    if not replay:
        r["summary"] = summary
        r["data"].update(data)
    summary = r["summary"]
    root = Path(os.environ["INFERENCEKIT_RUNS_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{r['evidence_id']}.json").write_text(json.dumps(r))
    print(summary)
    print(f"evidence: {(root / r['evidence_id'] / 'evidence.json').resolve()} (fake)")
    print(f"run-dir copy: {(root / (r['evidence_id'] + '.json')).resolve()}")


if cmd == "bench rounds":
    B, C = opt("--engines").split(",")
    n, speed = int(opt("--rounds")), {"base": 155.0, "cand": 170.0, **cfg.get("rounds", {})}
    tb, tc = speed["base"], speed["cand"]
    best = C if tc >= tb else B
    rows = []
    for k, b in enumerate(("all", "very_short", "short", "medium", "long")):
        for e, t in ((B, tb - k), (C, tc - k)):
            r = (t / (tc - k if best == C else tb - k)) if e != best else 1.0
            rows.append({"engine": e, "bucket": b, "metric": "decode_tok_s", "median": t, "ci95": [t - 0.2, t + 0.2],
                         "n": n, "best": best, "ratio_to_best": r,
                         "ratio_ci95": [None, None] if e == best else [r - 0.002, r + 0.002], "failed_requests": 0.0})
    cells = [{"round": i + 1, "label": e, "status": cfg.get("cells", "ok")} for i in range(n) for e in (B, C)]
    plan = {"rounds": n, "design": {"design": "williams", "period": 2, "balanced": n % 2 == 0},
            "engines": [{"label": B, "role": "baseline-recipe"}, {"label": C, "role": "custom"}]}
    record("rounds", f"bench rounds (single): 2 engines x {n} rounds", rows=rows, cells=cells, plan=plan,
           resolved_config={B: {"violations": []}, C: {"violations": []}})
elif cmd == "bench serve":
    t = cfg.get("tok", {}).get(label, 150.0)
    m = json.loads((FIX / "metrics.json").read_text())
    m["label"], m["speed"]["per_bucket"]["all"]["decode_tok_s"]["median"] = label, t
    per, f = m["speed"]["per_bucket"], cfg.get("failed", {}).get(label, 0)
    left = f
    for b in ("long", "medium", "short", "very_short"):  # metrics count the answered requests only
        k = min(left, per[b]["n_requests"])
        per[b]["n_requests"] -= k
        per["all"]["n_requests"] -= k
        left -= k
    (run / "metrics").mkdir(parents=True, exist_ok=True)
    (run / "metrics" / f"{label}.json").write_text(json.dumps(m))
    rounds = [{"index": 0, "n_requests": 16 - f, "n_failed": f, "decode_tok_s": t}]
    record("serve", f"{label} (chat, single user, 1 round(s) x 16 prompts): decode {t:.1f} tok/s"
           + f"; {f} failed" * bool(f), label=label, rounds=rounds)
elif cmd == "gates run":
    asked = [argv[i + 1] for i, x in enumerate(argv[:-1]) if x == "--gates"] or [
        "caching", "token_count", "output_length", "hidden_probes", "rare_tokens"]
    gates = {g: {"gate": g, "verdict": "FAIL" if g in cfg.get("gate_fail", []) else "PASS", "applicable": True,
                 "reason": "repeat speedup 1.9x (limit 1.10x)" if g in cfg.get("gate_fail", []) else "ok"} for g in asked}
    v = "FAIL" if cfg.get("gate_fail") else "PASS"
    record("gates", f"gates {v}: " + ", ".join(f"{g} {x['verdict']}" for g, x in gates.items()), verdict=v, gates=gates)
elif cmd == "quality trajectory":
    n = int(opt("--limit") or 64)
    (run / "quality" / label).mkdir(parents=True, exist_ok=True)
    record("trajectory", f"{label}: greedy trajectories for {n}/{n} prompts, 2048 tokens, top-20 logprobs")
elif cmd == "quality score":
    out, mine = opt("--out"), Path(opt("--trajectories")).name
    s = json.loads((FIX / "score-file.json").read_text())
    s["engines"] = {mine: {**s["engines"][cfg.get("engine", "fastdec-flashattn")], "label": mine}}
    s["verdicts"] = {mine: verdict}
    s["engines"][mine]["verdict"]["verdict"] = verdict
    Path(out).write_text(json.dumps(s))
    record("score", f"{verdict}: {mine} {verdict} (agreement 97.28%); noise floor 97.48%", verdict=verdict, score_file=out)
elif cmd == "quality gsm8k":
    n, every = int(opt("--limit") or 1319), cfg.get("wrong_every", {}).get(label, 15)
    rows = [{"idx": i, "correct": i % every != 0} for i in range(n)]
    (run / "quality" / "gsm8k").mkdir(parents=True, exist_ok=True)
    (run / "quality" / "gsm8k" / f"{label}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    c = sum(r["correct"] for r in rows)
    record("gsm8k", f"{label}: GSM8K {c}/{n} = {c / n:.2%}", label=label, n=n, correct=c, errors=0)
elif cmd == "quality sampling":
    record("sampling", f"{label}: {verdict}: sampled first tokens vs the server's own log-probabilities: mean TV "
           "0.0422, expected 0.0233", label=label, verdict=verdict)
else:
    print(f"inferencekit: error: the fake does not know {cmd!r}", file=sys.stderr)
    sys.exit(2)
