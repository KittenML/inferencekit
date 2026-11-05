"""Stand-in for `inferencekit bench serve` in the round-runner tests.

Accepts the flags the runner passes (the flags bench serve accepts), sends one chat request per prompt
to --base-url, and saves a record like bench serve's: per-request rows in a `requests` artifact under
$INFERENCEKIT_RUNS_DIR/<id>/requests.jsonl, carrying over the rows of --extend (so the record is cumulative,
as with the real command), and prints the record as JSON. Latencies come from the server's reply
(x_fake), so a test can tell which server was measured. --fail-label LABEL exits 5 for that engine;
--fail-requests LABEL exits 0 but marks every request of that engine failed.

Load mode returns a capacity-shaped record (as bench serve does): N*, cost and p95 TTFT at N*,
citing a measured record (data.measurement) whose data.e2e_ref_ms is --ref-e2e-ms when given,
else this engine's own "warm-up" e2e (3 x its latency).
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
for flag in ("--base-url", "--model", "--prompts", "--mode", "--engine-kind", "--label", "--run-dir"):
    ap.add_argument(flag, required=True)
ap.add_argument("--rounds", type=int, default=5)
ap.add_argument("--extend")
ap.add_argument("--engine-version")
ap.add_argument("--fail-label")
ap.add_argument("--fail-requests")
ap.add_argument("--ref-e2e-ms", type=float)
ap.add_argument("--rates", action="append")
a = ap.parse_args()
if a.fail_label == a.label:
    raise SystemExit(5)

root = Path(os.environ.get("INFERENCEKIT_RUNS_DIR") or os.environ["INFERKIT_RUNS_DIR"])
prev_rows, offset = [], 0
if a.extend:
    prev_rows = [json.loads(x) for x in (root / a.extend / "requests.jsonl").read_text().splitlines() if x.strip()]
    offset = max(r["round"] for r in prev_rows) + 1
prompts = [json.loads(line) for line in open(a.prompts) if line.strip()]
rows = []
for k in range(offset, offset + (a.rounds if a.mode == "single" else 1)):
    for p in prompts:
        body = json.dumps({"model": a.model, "messages": p["messages"], "max_tokens": 2}).encode()
        req = urllib.request.Request(a.base_url + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        fake = json.loads(urllib.request.urlopen(req, timeout=10).read())["x_fake"]
        assert fake["label"] == a.label and fake["model"] == a.model, (fake, a.label)  # measured the right server
        scale = {"very_short": 1.0, "short": 1.5}.get(p["bucket"], 2.0) * (1.0 + 0.01 * (k % 3))
        if a.fail_requests == a.label:
            rows.append({"id": p["id"], "bucket": p["bucket"], "round": k, "ok": False, "ttft_ms": None})
            continue
        rows.append({"id": p["id"], "bucket": p["bucket"], "round": k, "ok": True, "ttft_ms": fake["ttft_ms"] * scale,
                     "tpot_ms": fake["ttft_ms"] / 10, "decode_tok_s": 10000.0 / fake["ttft_ms"] / scale,
                     "e2e_ms": fake["ttft_ms"] * scale * 3})
all_rows = prev_rows + rows
eid = f"ev_{a.label}_{offset}_{a.mode}"
(root / eid).mkdir(parents=True, exist_ok=True)
(root / eid / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in all_rows))
record = {"evidence_id": eid, "kind": "serve_benchmark", "data": {"extends": a.extend, "mode": a.mode},
          "artifacts": [{"path": "requests.jsonl", "sha256": "0" * 64, "bytes": 1, "role": "requests"}],
          # cumulative summary metrics, as bench serve reports them (must NOT be read as this round's value)
          "metrics": [{"name": "ttft_ms", "value": 999.0, "unit": "ms", "stat": "median", "subject": "all"}]}
if a.mode == "load":  # a capacity-style record: no requests artifact, metrics without a bucket
    meas = f"ev_{a.label}_{os.getpid()}_measured"
    e2e_ref = a.ref_e2e_ms if a.ref_e2e_ms is not None else fake["ttft_ms"] * 3
    (root / meas).mkdir(parents=True, exist_ok=True)
    (root / meas / "evidence.json").write_text(json.dumps({"evidence_id": meas, "kind": "serve_benchmark",
                                                           "data": {"e2e_ref_ms": e2e_ref}}))
    record = {"evidence_id": f"ev_{a.label}_{os.getpid()}_capacity", "kind": "capacity",
              "data": {"measurement": meas, "ref_e2e_ms_passed": a.ref_e2e_ms},
              "metrics": [{"name": "n_star", "value": 100.0 / fake["ttft_ms"], "unit": "streams", "stat": "threshold"},
                          {"name": "goodput_rps", "value": 50.0 / fake["ttft_ms"], "unit": "req/s",
                           "stat": "at_n_star"},
                          {"name": "ttft_ms", "value": fake["ttft_ms"] * 2, "unit": "ms", "stat": "p95_at_n_star"},
                          {"name": "usd_per_1m_output_tokens", "value": fake["ttft_ms"] / 10, "unit": "USD",
                           "stat": "at_n_star"}]}
print(json.dumps(record, indent=2))
