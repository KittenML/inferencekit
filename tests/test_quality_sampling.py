"""quality sampling: a server's seeded first-token draws against the distribution its own top-20 log-probabilities
imply under the request's sampling settings (self-consistency), one server at a time.

Fake OpenAI servers (``inferencekit.quality.fake``) return raw top-k logprobs and sample with temperature,
top-k, top-p and a per-request seed; variants play sampler bugs (wrong top-p, wrong temperature), an engine
whose logits change with the batch, one that ignores the seed, one that always draws the same token, and
engines that fail requests. The math (``inferencekit.quality.sampling``) is tested directly at the end.
"""

import json
import math
import random
from collections import Counter

import pytest

from inferencekit import evidence as ev
from inferencekit import registry
from inferencekit.quality import sampling as sm
from inferencekit.quality.fake import FakeLM, serve
from inferencekit.registry import OpError

SETTINGS = {"temperature": 1.0, "top_k": 0, "top_p": 0.9}


def write_prompts(path, n=24):
    rows = [{"id": f"p{i:02d}", "bucket": "short",
             "messages": [{"role": "user", "content": f"q{i} " + "xyz"[: i % 3 + 1] * (i + 1)}], "max_tokens": 8}
            for i in range(n)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def sampling(url, run, prompts=None, settings=SETTINGS, **kw):
    # 24 prompts x 256 seeds (the default seeds): an exact sampler's CI then clears the 0.02 margin
    args = {"base_url": url, "model": "fake-model", "run_dir": str(run), "seeds": 256,
            "extra_body": json.dumps(settings), "resamples": 2000, **kw}
    if prompts is not None:
        args["prompts"] = str(prompts)
    return registry.call("quality_sampling", args)


def chat_bodies(lm):
    return [r["body"] for r in lm.requests if r["path"] == "/v1/chat/completions"]


def results(run, label="engine"):
    return json.loads((run / "quality" / "sampling" / f"{label}.json").read_text())


# --------------------------------------------------------------------------- verdicts

def test_exact_sampler_passes_and_every_request_is_as_planned(tmp_path, store):
    prompts = write_prompts(tmp_path / "p.jsonl", n=16)
    run = tmp_path / "runs" / "s1"
    lm = FakeLM(vocab=10)
    with serve(lm) as url:
        rec = sampling(url, run, prompts, label="exact", seeds=128)
    assert ev.verify(rec)["id_ok"]
    data = rec["data"]
    assert data["task"] == "sampling" and data["method"] == "self_consistency"
    assert data["verdict"] == "PASS", data["reason"]
    st = data["statistic"]
    # an exact sampler: the observed TV is what sampling noise alone gives (its exact expectation)
    assert st["n_prompts"] == 16 and st["ci95"][1] < 0.02 and abs(st["excess"]) < 0.01
    assert st["mean_tv"] > 0.02 and st["mean_tv_expected"] > 0.02 and st["binned_by"] == "token_id"
    assert data["out_of_support"]["draws"] == 0 and data["out_of_support"]["of"] == 16 * 128
    assert data["prompts_used"] == {"asked": 16, "informative": 16, "support_checked": 16, "dropped": {},
                                    "dropped_why": {}}
    assert data["seed_checks"]["candidate"]["status"] == "honoured"
    assert data["baseline"] is None and data["cross_server"] is None
    # per prompt: two logprob requests (top 20, vLLM's id fields) and S seeded draws, all max_tokens 1
    bodies = chat_bodies(lm)
    lp = [b for b in bodies if b.get("logprobs")]
    draws = [b for b in bodies if not b.get("logprobs")]
    assert len(lp) == 2 * 16 and all(b["top_logprobs"] == 20 and b["return_tokens_as_token_ids"] for b in lp)
    assert all(b["max_tokens"] == 1 and b["top_p"] == 0.9 and b["temperature"] == 1.0 for b in bodies)
    assert Counter(b["seed"] for b in draws if b["seed"] >= 8) == {s: 16 for s in range(8, 128)}
    assert all(b["return_token_ids"] for b in draws)
    names = {m["name"]: m for m in rec["metrics"] if m.get("subject") == "exact"}
    assert names["first_token_tv_excess"]["unit"] == "probability"
    assert names["first_token_tv_excess"]["ci95"] == st["ci95"]
    assert names["first_token_out_of_support"]["value"] == 0.0
    doc = results(run, "exact")
    assert doc["format"] == "inferencekit.quality.sampling.v2" and doc["baseline"] is None
    row = doc["candidate"]["per_prompt"][0]
    assert sum(n for _, n in row["histogram"]) == 128 and abs(sum(p for _, p in row["implied"]) - 1) < 1e-9
    assert len(data["per_prompt"]) == 16 and {"tv", "tv_expected", "excess"} <= set(data["per_prompt"][0])
    assert any(a["role"] == "sampling_histograms" for a in rec["artifacts"])
    assert (run / "evidence" / f"{rec['evidence_id']}.json").is_file()


def test_record_is_reproducible(tmp_path, store):
    """Fixed draws (seeded fakes) and a seeded bootstrap: the same statistic every time."""
    prompts = write_prompts(tmp_path / "p.jsonl", n=10)
    with serve(FakeLM(vocab=10)) as url:
        a = sampling(url, tmp_path / "runs" / "r1", prompts, seeds=32)["data"]["statistic"]
        b = sampling(url, tmp_path / "runs" / "r2", prompts, seeds=32)["data"]["statistic"]
    assert a == b


@pytest.mark.parametrize("bug", [{"wrong_top_p": 1.0}, {"wrong_top_p": 0.5}, {"wrong_temperature": 0.4},
                                 {"wrong_temperature": 2.0}])
def test_a_sampler_that_ignores_top_p_or_temperature_fails(tmp_path, store, bug):
    prompts = write_prompts(tmp_path / "p.jsonl", n=12)
    with serve(FakeLM(vocab=10, **bug)) as url:
        rec = sampling(url, tmp_path / "runs" / "s2", prompts, seeds=128)
    data = rec["data"]
    assert data["verdict"] == "FAIL", data["reason"]
    # min_p and repetition_penalty were not sent: a server default for them is named as another explanation
    assert "min_p, repetition_penalty not sent and read as off" in data["reason"]
    if bug.get("wrong_top_p") == 1.0:  # no top-p at all: tokens the settings remove are drawn
        oos = data["out_of_support"]
        assert oos["fail"] and oos["ci95"][0] > 0.005 and "tokens these settings remove" in data["reason"]
    else:
        assert data["statistic"]["verdict"] == "FAIL" and data["statistic"]["ci95"][0] > 0.02


class TemperedLogprobs(FakeLM):
    """Samples correctly but lists log-probabilities after temperature (the contract requires raw ones)."""

    def chat(self, body):
        status, resp = super().chat(body)
        if status == 200 and body.get("logprobs"):
            t = float(body["temperature"])
            for entry in resp["choices"][0]["logprobs"]["content"]:
                scaled = [a["logprob"] / t for a in entry["top_logprobs"]]  # vocab 10: all of it is listed
                m = max(scaled)
                lse = m + math.log(sum(math.exp(x - m) for x in scaled))
                for alt, x in zip(entry["top_logprobs"], scaled):
                    alt["logprob"] = x - lse
        return status, resp


def test_logprobs_after_temperature_fail_and_say_so(tmp_path, store):
    prompts = write_prompts(tmp_path / "p.jsonl", n=12)
    with serve(TemperedLogprobs(vocab=10)) as url:
        rec = sampling(url, tmp_path / "runs" / "temp", prompts, seeds=128,
                       settings={"temperature": 0.6, "top_k": 0, "top_p": 0.95})
    data = rec["data"]
    assert data["verdict"] == "FAIL" and "returns log-probabilities after temperature" in data["reason"]
    assert data["statistic"]["excess_if_logprobs_were_tempered"]["ci95"][1] < 0.02


def test_logits_that_change_with_the_batch_still_pass_one_at_a_time(tmp_path, store):
    """The GPU validation case: an engine whose logits depend on what else is in flight. Its own draws follow
    its own logprobs when both run alone (concurrency 1); in a batch they do not."""
    prompts = write_prompts(tmp_path / "p.jsonl")
    lm = FakeLM(vocab=10, batch_shift=0.4)
    alone = lm.logprobs([1, 5])
    lm._tl.inflight = 8  # this thread's request as if 8 were in flight: the shift is real
    assert lm.logprobs([1, 5]) != alone
    lm._tl.inflight = 1
    with serve(lm) as url:
        rec = sampling(url, tmp_path / "runs" / "s3", prompts)
    assert rec["data"]["verdict"] == "PASS", rec["data"]["reason"]
    assert lm.max_inflight == 1 and rec["identity"]["workload"]["concurrency"] == 1
    assert not any("concurrency" in lim for lim in rec["limitations"])
    # the same engine with 8 requests in flight: logprobs and draws see other batches, and it does not PASS
    lm = FakeLM(vocab=10, batch_shift=0.4, delay_s=0.002)
    with serve(lm) as url:
        rec = sampling(url, tmp_path / "runs" / "s3b", prompts, concurrency=8)
    assert lm.max_inflight > 1 and rec["data"]["verdict"] != "PASS", rec["data"]["reason"]
    assert any(lim.startswith("concurrency 8") for lim in rec["limitations"])


def test_a_server_that_ignores_the_seed_is_not_blocked(tmp_path, store):
    # concurrency 1: the ignoring engine's draws follow request order, so this test is deterministic
    prompts = write_prompts(tmp_path / "p.jsonl")
    with serve(FakeLM(vocab=10, seeds="ignore")) as url:
        rec = sampling(url, tmp_path / "runs" / "s4", prompts)
    data = rec["data"]
    check = data["seed_checks"]["candidate"]
    assert check["status"] == "ignored" and check["same"] < 0.9 * check["repeats"]
    assert data["verdict"] == "PASS", data["reason"]
    assert any("does not honour the request seed" in lim and "information only" in lim for lim in rec["limitations"])


def test_one_token_for_every_seed_fails(tmp_path, store):
    prompts = write_prompts(tmp_path / "p.jsonl", n=12)
    with serve(FakeLM(vocab=10, seeds="fixed")) as url:
        rec = sampling(url, tmp_path / "runs" / "s5", prompts, seeds=64)
    assert rec["data"]["verdict"] == "FAIL" and rec["data"]["statistic"]["verdict"] == "FAIL"


def test_too_few_prompts_is_unproven(tmp_path, store):
    prompts = write_prompts(tmp_path / "p.jsonl", n=5)
    with serve(FakeLM(vocab=10)) as url:
        rec = sampling(url, tmp_path / "runs" / "s6", prompts, seeds=64)
    data = rec["data"]
    assert data["verdict"] == "UNPROVEN" and "5 informative prompt(s), need 8" in data["reason"]
    assert data["statistic"]["min_prompts"] == 8 and data["statistic"]["ci95"][0] is not None


def test_greedy_settings_leave_nothing_to_test(tmp_path, store):
    prompts = write_prompts(tmp_path / "p.jsonl", n=4)
    with serve(FakeLM(vocab=10)) as url:
        rec = sampling(url, tmp_path / "runs" / "s7", prompts, seeds=4, settings={"temperature": 0})
    data = rec["data"]
    assert data["verdict"] == "UNPROVEN" and data["prompts_used"]["dropped"] == {"near_deterministic": 4}
    assert data["out_of_support"]["draws"] == 0 and data["out_of_support"]["of"] == 16  # greedy is checked too


# --------------------------------------------------------------------------- shaped distributions (top_k > 20)

class Shaped(FakeLM):
    """vocab 100. ``peaked``: four head tokens hold almost all the mass (the nucleus is inside the top 20);
    ``flat``: every logit within +-0.5 (a top-p 0.95 nucleus needs most of the vocabulary)."""

    def __init__(self, shape, **kw):
        super().__init__(vocab=100, **kw)
        self.shape = shape

    def logprobs(self, context):
        a = context[-1] if context else 0
        b = context[-2] if len(context) > 1 else 0
        base = [math.sin(1.7 * a + 0.37 * b * j + 0.91 * j + 0.13 * j * j) for j in range(self.vocab)]
        if self.shape == "peaked":
            head = sorted(range(self.vocab), key=lambda j: -base[j])[:4]
            logits = [8.0 + base[j] if j in head else 0.5 * base[j] for j in range(self.vocab)]
        else:
            logits = [0.5 * x for x in base]
        m = max(logits)
        lse = m + math.log(sum(math.exp(x - m) for x in logits))
        return [x - lse for x in logits]


@pytest.mark.parametrize("shape", ["peaked", "flat"])
def test_top_k_above_20_is_tested_only_when_the_nucleus_is_provably_inside_the_top_20(tmp_path, store, shape):
    prompts = write_prompts(tmp_path / "p.jsonl")
    settings = {"temperature": 1.0, "top_k": 64, "top_p": 0.95}
    with serve(Shaped(shape)) as url:
        rec = sampling(url, tmp_path / "runs" / f"k64-{shape}", prompts, settings=settings,
                       seeds=256 if shape == "peaked" else 8)
    data = rec["data"]
    if shape == "peaked":
        assert data["verdict"] == "PASS", data["reason"]
        assert data["prompts_used"]["informative"] == 24 and data["out_of_support"]["draws"] == 0
    else:
        assert data["verdict"] == "UNPROVEN" and data["prompts_used"]["dropped"] == {"beyond_top": 24}
        assert "24 beyond_top" in data["reason"]
        assert data["prompts_used"]["dropped_why"] == {"beyond_top": "the kept tokens may reach past the listed top 20"}


class Headed(FakeLM):
    """vocab 100: eight head tokens falling off by ``decay`` nats per rank and 92 tail tokens about ``tail`` nats
    below, the tail's level moving from prompt to prompt: the top 20 hold all but 0.1-3% of the mass."""

    def __init__(self, tail, decay, **kw):
        super().__init__(vocab=100, **kw)
        self.tail, self.decay = tail, decay

    def logprobs(self, context):
        a = context[-1] if context else 0
        b = context[-2] if len(context) > 1 else 0
        base = [math.sin(1.7 * a + 0.37 * b * j + 0.91 * j + 0.13 * j * j) for j in range(self.vocab)]
        rank = {j: r for r, j in enumerate(sorted(range(self.vocab), key=lambda j: -base[j]))}
        tail = self.tail + 1.5 * math.sin(2.1 * a + 0.6 * b + 0.3 * len(context))
        logits = [-self.decay * rank[j] + 0.5 * base[j] if rank[j] < 8 else 0.3 * base[j] - tail
                  for j in range(self.vocab)]
        m = max(logits)
        lse = m + math.log(sum(math.exp(x - m) for x in logits))
        return [x - lse for x in logits]


@pytest.mark.parametrize("temperature", [0.6, 1.0])
def test_top_k_20_at_the_task_settings(tmp_path, store, temperature):
    """top_k 20 equal to the 20 listed tokens, vocabulary 100 (the Qwen tasks' default sampling)."""
    prompts = write_prompts(tmp_path / "p.jsonl")
    settings = {"temperature": temperature, "top_k": 20, "top_p": 0.95}
    with serve(Headed(tail=8.0, decay=0.9)) as url:
        rec = sampling(url, tmp_path / "runs" / f"k20-{temperature}", prompts, settings=settings)
    data = rec["data"]
    assert data["verdict"] == "PASS", data["reason"]
    assert data["prompts_used"]["informative"] >= 12 and data["out_of_support"]["draws"] == 0
    assert data["settings"]["not_sent"] == ["min_p", "repetition_penalty"]


def hf_and_joint_differ(lm, prompts_path, temperature, top_k, top_p):
    """Prompt ids where SGLang's kept set (top-p over everything, then the first top_k) is not Hugging Face's."""
    out = []
    for line in prompts_path.read_text().splitlines():
        row = json.loads(line)
        lp = lm.logprobs(lm.template(row["messages"]))
        w = {j: math.exp(v / temperature) for j, v in enumerate(lp)}
        order = sorted(w, key=lambda j: (-w[j], j))

        def nucleus(ids):
            z, kept, acc = sum(w[j] for j in ids), [], 0.0
            for j in ids:
                kept.append(j)
                acc += w[j] / z
                if acc >= top_p:
                    break
            return kept

        if set(nucleus(order[:top_k])) != set(nucleus(order)[:top_k]):
            out.append(row["id"])
    return out


@pytest.mark.parametrize("top_k,top_p,reason", [(20, 0.95, "beyond_top"), (10, 0.9, "top_p_order")])
def test_an_sglang_order_sampler_is_not_failed(tmp_path, store, top_k, top_p, reason):
    """SGLang takes top-p over the whole distribution and then the first top_k tokens; its kept set can be larger
    than top-p over the top-k set. Prompts where the two orders can differ are dropped, so a correct sampler in
    either order passes. With top_k 20 the tokens past the 20th that SGLang may keep are unlisted (beyond_top);
    with top_k 10 they are listed (top_p_order)."""
    prompts = write_prompts(tmp_path / "p.jsonl")
    settings = {"temperature": 1.0, "top_k": top_k, "top_p": top_p}
    lm = Headed(tail=7.5, decay=0.9, top_p_order="joint")
    differ = hf_and_joint_differ(lm, prompts, 1.0, top_k, top_p)
    assert len(differ) >= 6  # the case matters: on these the SGLang order draws tokens the other removes
    with serve(lm) as url:
        rec = sampling(url, tmp_path / "runs" / f"joint-{top_k}", prompts, settings=settings)
    data = rec["data"]
    assert data["verdict"] == "PASS", data["reason"]
    assert data["out_of_support"]["draws"] == 0 and data["prompts_used"]["dropped"].get(reason, 0) >= 6
    rows = {r["id"]: r for r in data["per_prompt"]}
    assert all(rows[i]["status"] != "informative" for i in differ)
    assert any("SGLang takes top-p over the whole distribution" in lim for lim in rec["limitations"])


class Fixed(FakeLM):
    """The same first-token distribution after every prompt: ``probs`` over the first tokens, ~0 for the rest."""

    def __init__(self, probs, **kw):
        super().__init__(vocab=10, **kw)
        rest = (self.vocab - len(probs)) * 1e-12
        self.lp = [math.log(p) for p in probs[:-1]] + [math.log(probs[-1] - rest)] + \
                  [math.log(1e-12)] * (self.vocab - len(probs))

    def logprobs(self, context):
        return list(self.lp)


def test_a_top_p_boundary_inside_the_tolerance_is_excluded(tmp_path, store):
    # cumulative 0.5, 0.9 at top_p 0.9: whether token 3 is kept depends on rounding
    prompts = write_prompts(tmp_path / "p.jsonl", n=8)
    with serve(Fixed([0.5, 0.4, 0.1])) as url:
        rec = sampling(url, tmp_path / "runs" / "b1", prompts, seeds=16)
    data = rec["data"]
    assert data["verdict"] == "UNPROVEN" and data["prompts_used"]["dropped"] == {"boundary": 8}
    with serve(Fixed([0.5, 0.35, 0.15])) as url:  # 0.85 < 0.9 < 1.0: clear
        rec = sampling(url, tmp_path / "runs" / "b2", prompts, seeds=64)
    assert rec["data"]["prompts_used"]["informative"] == 8 and rec["data"]["verdict"] != "FAIL"


class Unstable(FakeLM):
    """The first logprob request for each prompt sees moved logits, so the top-20 ids change between two
    identical requests."""

    def __init__(self, **kw):
        super().__init__(vocab=40, **kw)
        self.seen = Counter()

    def chat(self, body):
        if body.get("logprobs"):
            key = json.dumps(body["messages"])
            self.seen[key] += 1
            self.perturb = 1.5 if self.seen[key] == 1 else 0.0
            try:
                return super().chat(body)
            finally:
                self.perturb = 0.0
        return super().chat(body)


def test_logprobs_that_change_between_two_requests_are_excluded(tmp_path, store):
    prompts = write_prompts(tmp_path / "p.jsonl", n=8)
    with serve(Unstable()) as url:
        rec = sampling(url, tmp_path / "runs" / "u1", prompts, seeds=8, settings={"temperature": 1.0, "top_k": 5})
    data = rec["data"]
    assert data["prompts_used"]["dropped"].get("logprobs_unstable", 0) >= 6, data["prompts_used"]
    assert data["verdict"] == "UNPROVEN"
    assert "listed_again" in results(tmp_path / "runs" / "u1")["candidate"]["per_prompt"][0]


# --------------------------------------------------------------------------- two servers

def test_two_servers_each_get_a_verdict_and_their_distance_is_information(tmp_path, store):
    prompts = write_prompts(tmp_path / "p.jsonl")
    run = tmp_path / "runs" / "two"
    with serve(FakeLM(vocab=10, perturb=0.3, rng_salt="b")) as c_url, serve(FakeLM(vocab=10)) as b_url:
        rec = sampling(c_url, run, prompts, baseline_url=b_url, label="cand")
    data = rec["data"]
    assert data["verdict"] == "PASS" and data["baseline"]["verdict"] == "PASS", (data["reason"],
                                                                                   data["baseline"]["reason"])
    cross = data["cross_server"]
    # on the prompts both servers could build (the perturbed candidate has a near-tie cut on a third of them)
    assert data["prompts_used"]["dropped"] == {"boundary": 8} and data["baseline"]["prompts_used"]["dropped"] == {}
    assert cross["n_prompts"] == 16 and 0 < cross["mean"] <= cross["max"] < 1
    assert not any(lim.startswith("BASELINE") for lim in rec["limitations"])
    subjects = {(m["name"], m.get("subject")) for m in rec["metrics"]}
    assert {("first_token_tv_excess", "cand"), ("first_token_tv_excess", "baseline"),
            ("first_token_implied_tv_between_servers", "cand vs baseline")} <= subjects
    # the candidate's metric comes first (the ledger and the review read the first engine-level one)
    assert next(m for m in rec["metrics"] if m["name"] == "first_token_tv_excess")["subject"] == "cand"
    doc = results(run, "cand")
    assert len(doc["baseline"]["per_prompt"]) == 24 and len(doc["cross_server"]["per_prompt"]) == 16


def test_a_failing_baseline_is_a_loud_limitation_not_the_verdict(tmp_path, store):
    prompts = write_prompts(tmp_path / "p.jsonl")
    with serve(FakeLM(vocab=10)) as c_url, serve(FakeLM(vocab=10, wrong_top_p=0.5)) as b_url:
        rec = sampling(c_url, tmp_path / "runs" / "two-bad", prompts, baseline_url=b_url)
    data = rec["data"]
    assert data["verdict"] == "PASS" and data["baseline"]["verdict"] == "FAIL"
    assert rec["limitations"][0].startswith("BASELINE FAIL")
    assert "baseline FAIL on the same check" in rec["summary"]


def test_the_ledger_and_the_review_read_the_record(tmp_path, store):
    """The metric name, unit and data fields the ledger trailer and the run review read stay where they were."""
    from inferencekit.review.rundir import EngineInfo, Run, attach_evidence

    prompts = write_prompts(tmp_path / "p.jsonl")
    run_dir = tmp_path / "runs" / "led"
    with serve(FakeLM(vocab=10)) as c_url, serve(FakeLM(vocab=10, rng_salt="b")) as b_url:
        rec = sampling(c_url, run_dir, prompts, baseline_url=b_url, label="myengine")
    assert rec["data"]["verdict"] == "PASS"
    excess = rec["data"]["statistic"]["excess"]
    res = registry.call("ledger_trailers", {"ids": [str(run_dir / "evidence" / f"{rec['evidence_id']}.json")]})
    line = dict(x.split(": ", 1) for x in res["markdown"].splitlines())["Perf-Quality"]
    assert line.startswith(f"first token tv excess {excess:.4f}"[:26]) and line.endswith("PASS [myengine]")
    run = Run(root=tmp_path, modality="llm", engines=[EngineInfo("myengine")], reference=None, items=[], cells={})
    attach_evidence(run, [("evidence", rec)])
    got = run.engines[0].ev_quality
    assert got == [{"name": "sampling", "unit": "probability", "evidence": rec["evidence_id"],
                    "value": pytest.approx(excess), "verdict": "PASS"}]


# --------------------------------------------------------------------------- requests that fail

class FailsOn(FakeLM):
    """HTTP 500 on prompts containing ``marker``; HTTP 400 on every request when ``marker`` is None."""

    def __init__(self, marker, **kw):
        super().__init__(**kw)
        self.marker = marker

    def chat(self, body):
        if self.marker is None:
            return 400, {"error": {"message": "unknown field: top_k"}}
        if any(self.marker in str(m.get("content")) for m in body.get("messages") or []):
            return 500, {"error": {"message": "engine crashed"}}
        return super().chat(body)


def test_failed_requests_make_it_unproven(tmp_path, store):
    prompts = write_prompts(tmp_path / "p.jsonl")
    with serve(FailsOn("q5 ", vocab=10)) as url:
        rec = sampling(url, tmp_path / "runs" / "f1", prompts)
    data = rec["data"]
    assert data["verdict"] == "UNPROVEN"
    assert data["errors"] == {"candidate": 2 + 256}  # two logprob requests and 256 draws, each after a retry
    assert "requests failed after a retry" in data["reason"] and "statistic alone: PASS" in data["reason"]
    assert data["prompts_used"]["dropped"] == {"incomplete": 1} and data["statistic"]["n_prompts"] == 23
    assert "samples" in rec["coverage"]["missing"]


def test_refused_first_request_stops_that_server(tmp_path, store):
    prompts = write_prompts(tmp_path / "p.jsonl")
    bad = FailsOn(None, vocab=10)
    with serve(bad) as url:
        rec = sampling(url, tmp_path / "runs" / "f2", prompts, seeds=8)
    data = rec["data"]
    assert data["verdict"] == "UNPROVEN"
    assert "candidate: the first request failed" in data["reason"] and "top_k" in data["reason"]
    # the first request (and its retry), then without each vLLM id field in turn; nothing else
    asked = [(b.get("return_tokens_as_token_ids"), b.get("return_token_ids")) for b in chat_bodies(bad)]
    assert asked == [(True, True)] * 2 + [(None, True)] * 2 + [(None, None)] * 2
    assert data["statistic"]["n_prompts"] == 0 and data["statistic"]["ci95"] == [None, None]


def test_a_server_without_logprobs_is_unproven(tmp_path, store):
    prompts = write_prompts(tmp_path / "p.jsonl")
    lm = FakeLM(vocab=10, logprob_mode="ignore")
    with serve(lm) as url:
        rec = sampling(url, tmp_path / "runs" / "f3", prompts, seeds=8)
    assert rec["data"]["verdict"] == "UNPROVEN" and "no logprobs" in rec["data"]["reason"]
    assert len(chat_bodies(lm)) == 1  # an answer without logprobs: no point sampling what cannot be judged


class OtherDetokenizer(FakeLM):
    """Decodes the first token with a leading space (a one-token decode can keep it); with ``refuse_ids`` it
    refuses vLLM's id fields (HTTP 400), so it answers with text only."""

    def __init__(self, refuse_ids=False, **kw):
        super().__init__(**kw)
        self.refuse_ids = refuse_ids

    def chat(self, body):
        if self.refuse_ids and ("return_token_ids" in body or "return_tokens_as_token_ids" in body):
            return 400, {"error": {"message": "unknown field: return_token_ids"}}
        status, resp = super().chat(body)
        msg = resp["choices"][0]["message"]
        msg["content"] = " " + msg["content"]
        return status, resp


@pytest.mark.parametrize("refuse_ids", [False, True])
def test_a_detokenizer_difference_does_not_split_bins(tmp_path, store, refuse_ids):
    # by token id, or by text with the space removed on draws and listed tokens alike, the draws match
    prompts = write_prompts(tmp_path / "p.jsonl")
    with serve(OtherDetokenizer(refuse_ids, vocab=10)) as url:
        rec = sampling(url, tmp_path / "runs" / f"t-{refuse_ids}", prompts)
    data = rec["data"]
    assert data["statistic"]["binned_by"] == ("text" if refuse_ids else "token_id")
    assert data["verdict"] == "PASS" and data["out_of_support"]["draws"] == 0, data["reason"]
    assert any("refused a vLLM token-id field" in lim for lim in rec["limitations"]) == refuse_ids


# --------------------------------------------------------------------------- arguments and settings

def test_settings_fields_and_prompt_order(tmp_path, store):
    lm = FakeLM(vocab=10)
    extra = {**SETTINGS, "max_tokens": 50, "seed": 3, "logprobs": False, "chat_template_kwargs": {"enable_thinking": False}}
    with serve(lm) as url:
        rec = sampling(url, tmp_path / "runs" / "a1", n_prompts=6, seeds=4, settings=extra)
    data = rec["data"]
    assert data["prompts"]["name"] == "dev" and len(data["prompts"]["ids"]) == 6
    assert [r["bucket"] for r in data["per_prompt"]] == ["very_short", "short"] * 3
    assert data["settings"]["ignored_extra_body_keys"] == ["logprobs", "max_tokens", "seed"]
    assert data["settings"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert data["settings"]["sampling"] == {"temperature": 1.0, "top_k": None, "top_p": 0.9, "min_p": None}
    bodies = chat_bodies(lm)
    assert all(b["max_tokens"] == 1 and b["chat_template_kwargs"] == {"enable_thinking": False} for b in bodies)
    assert {b["seed"] for b in bodies} == set(range(4))
    assert rec["identity"]["workload"]["prompt_set"]["name"] == "dev"
    assert data["settings"]["not_sent"] == ["min_p", "repetition_penalty"]


def test_unsent_and_unmodelled_fields_are_said(tmp_path, store):
    prompts = write_prompts(tmp_path / "p.jsonl", n=8)
    with serve(FakeLM(vocab=10)) as url:
        rec = sampling(url, tmp_path / "runs" / "a2", prompts, seeds=16,
                       settings={"temperature": 1.0, "repetition_penalty": 1.2, "presence_penalty": 0.5})
    data = rec["data"]
    assert data["settings"]["not_sent"] == ["top_k", "top_p", "min_p"]
    assert any("top_k, top_p, min_p not sent" in lim for lim in rec["limitations"])
    assert data["verdict"] == "UNPROVEN" and "repetition_penalty" in data["reason"]
    assert "presence_penalty" not in data["reason"]  # acts on generated tokens only: none before the first


def test_unmodelled_settings_never_fail(tmp_path, store):
    """A setting the check does not compute (a repetition penalty acts on the prompt's tokens) can move a correct
    sampler's draws away from the raw logprobs: the verdict is UNPROVEN, and what the statistic alone says is kept."""
    prompts = write_prompts(tmp_path / "p.jsonl", n=12)
    with serve(FakeLM(vocab=10, wrong_top_p=0.5)) as url:
        rec = sampling(url, tmp_path / "runs" / "a2b", prompts, seeds=128,
                       settings={**SETTINGS, "repetition_penalty": 1.2})
    data = rec["data"]
    assert data["verdict"] == "UNPROVEN" and data["statistic"]["verdict"] == "FAIL"
    assert data["reason"].startswith("--extra-body sets repetition_penalty") and "would FAIL" in data["reason"]


def test_concurrency_above_one_is_a_limitation(tmp_path, store):
    prompts = write_prompts(tmp_path / "p.jsonl", n=8)
    with serve(FakeLM(vocab=10)) as url:
        rec = sampling(url, tmp_path / "runs" / "a3", prompts, seeds=16, concurrency=4)
    assert any(lim.startswith("concurrency 4") for lim in rec["limitations"])


def test_default_is_one_request_at_a_time():
    """A batched engine's logits change with the batch, so draws alongside other requests need not follow the
    logprobs of a request run alone."""
    args = {a.name: a for a in registry.get("quality_sampling").args}
    assert args["concurrency"].default == 1 and args["margin"].default == 0.02 and args["seeds"].default == 256
    assert not args["baseline_url"].required


def test_bad_arguments_are_refused(tmp_path, store):
    prompts = write_prompts(tmp_path / "p.jsonl", n=1)
    for kw in ({"seeds": 1}, {"n_prompts": 1}, {"extra_body": "[1]"}, {"concurrency": 0},
               {"extra_body": "{}"}, {"extra_body": '{"temperature": 1, "top_p": 0}'},
               {"extra_body": '{"temperature": "hot"}'}):
        with pytest.raises(OpError):
            registry.call("quality_sampling", {"base_url": "http://127.0.0.1:9", "model": "m",
                                               "run_dir": str(tmp_path / "runs" / "x"),
                                               "extra_body": json.dumps(SETTINGS), **kw})
    with pytest.raises(OpError, match="temperature is not set"):
        registry.call("quality_sampling", {"base_url": "http://127.0.0.1:9", "model": "m",
                                           "run_dir": str(tmp_path / "runs" / "x")})
    with pytest.raises(OpError, match="need at least 2"):
        sampling("http://127.0.0.1:9", tmp_path / "runs" / "x", prompts)


def test_cli(tmp_path, store):
    from inferencekit import cli

    prompts = write_prompts(tmp_path / "p.jsonl", n=3)
    run = tmp_path / "runs" / "c1"
    with serve(FakeLM(vocab=10)) as url, serve(FakeLM(vocab=10)) as b_url:
        code = cli.main(["--runs-dir", str(store.root), "quality", "sampling", "--base-url", url, "--model",
                         "fake-model", "--baseline-url", b_url, "--run-dir", str(run), "--prompts", str(prompts),
                         "--seeds", "8", "--extra-body", json.dumps(SETTINGS), "--margin", "0.05", "--label", "cli"])
    assert code == 0
    doc = results(run, "cli")
    assert doc["candidate"]["statistic"]["margin"] == 0.05 and doc["candidate"]["statistic"]["resamples"] == 10000
    assert doc["baseline"]["verdict"] == "UNPROVEN"  # 3 prompts: too few on both


# --------------------------------------------------------------------------- the math

def test_binomial_mean_absolute_deviation_is_exact():
    for n in (1, 2, 7, 64, 256):
        for p in (0.001, 0.1, 1 / 3, 0.5, 0.9):
            direct = sum(abs(x - n * p) * math.comb(n, x) * p ** x * (1 - p) ** (n - x) for x in range(n + 1))
            assert sm.binomial_mad(n, p) == pytest.approx(direct, rel=1e-9, abs=1e-12)
    assert sm.binomial_mad(10, 0.0) == sm.binomial_mad(10, 1.0) == 0.0
    assert sm.expected_tv({"a": 1.0}, 50) == 0.0


def test_expected_tv_matches_a_parametric_bootstrap():
    probs = {0: 0.5, 1: 0.3, 2: 0.15, 3: 0.05}
    rng = random.Random(0)
    cum = [0.5, 0.8, 0.95, 1.0]
    tvs = []
    for _ in range(4000):
        h = Counter(next(i for i, c in enumerate(cum) if u < c) for u in (rng.random() for _ in range(256)))
        tvs.append(sm.tv_to(h, probs))
    assert sm.expected_tv(probs, 256) == pytest.approx(sum(tvs) / len(tvs), abs=0.001)


def lp_of(probs):
    return [(i, math.log(p)) for i, p in enumerate(probs)]


def test_implied_top_k_within_the_listed_tokens_is_exact():
    top = lp_of([0.4, 0.3, 0.2, 0.05, 0.03, 0.02])
    got = sm.implied(top, temperature=0.5, top_k=3)
    w = [0.4 ** 2, 0.3 ** 2, 0.2 ** 2]  # temperature 0.5: p^(1/T), renormalised over the kept set
    assert got["status"] == "ok" and got["kept"] == 3
    assert [got["probs"][i] for i in range(3)] == pytest.approx([x / sum(w) for x in w])
    # a near-tie across the top-k cut is ambiguous
    assert sm.implied(lp_of([0.4, 0.3, 0.15, 0.1499, 0.0001]), 1.0, top_k=3)["status"] == "boundary"


def test_implied_top_p_cut_and_tolerance():
    top = lp_of([0.5, 0.3, 0.15, 0.05])
    assert sm.implied(top, 1.0, top_p=0.7)["kept"] == 2  # 0.5 < 0.7 <= 0.8
    assert sm.implied(top, 1.0, top_p=0.8)["status"] == "boundary"  # cumulative 0.8 at the threshold
    assert sm.implied(top, 1.0, top_p=0.8002)["status"] == "boundary"  # within the tolerance
    assert sm.implied(top, 1.0, top_p=0.8005)["kept"] == 3  # just outside it
    assert sm.implied(top, 1.0, top_p=0.81)["kept"] == 3


def test_implied_unlisted_mass_bound():
    # 20 listed tokens holding 0.99 of the mass, the 20th at 0.001: with top_k 64 up to 45 unlisted tokens
    # of <= 0.001 each may be in the top-k set (0.01 of mass at most)
    probs = [0.6, 0.25, 0.1] + [0.04 / 16] * 16 + [0.001]
    top = lp_of(probs)
    # top_p 0.9: the first two hold at most 0.858 and the first three at least 0.95, whatever the tail holds
    assert sm.implied(top, 1.0, top_k=64, top_p=0.9)["kept"] == 3
    # top_p 0.952: three tokens without the unlisted tail (0.959), four with all of it (0.95 then 0.9525)
    assert sm.implied(top, 1.0, top_k=64, top_p=0.952)["status"] == "beyond_top"
    assert sm.implied(top, 1.0, top_k=None, top_p=None)["status"] == "beyond_top"  # all of them kept
    # no bound on the unlisted weight above temperature 1 with no top-k: never provable unless min_p cuts
    assert sm.implied(top, 1.5, top_k=None, top_p=0.5)["status"] == "beyond_top"
    # ... unless min_p removes every unlisted token and top-p keeps at least what min_p keeps
    assert sm.implied(top, 1.5, top_k=None, top_p=0.99, min_p=0.1)["kept"] == 3
    assert sm.implied(top, 1.5, top_k=None, top_p=0.5, min_p=0.1)["status"] == "beyond_top"
    bound = sm.unseen_bound(0.01, 0.001, 0.6, 1.0, 45)
    assert bound == pytest.approx(0.01 / 0.6)


def test_implied_allows_any_number_of_unlisted_ties_with_the_20th():
    """Hugging Face and vLLM keep every token tied with the k-th logit. 20 listed tokens holding 0.9745, three
    unlisted ones tied with the 20th (0.0015 each), top_k 20, top_p 0.95: with the ties the top-k set holds 0.979
    and top-p keeps 6 tokens (0.928 < 0.95 x 0.979); without them it keeps 5. The ties cannot be seen in the
    listing, so the prompt is not testable."""
    probs = [0.40, 0.25, 0.15, 0.08, 0.048, 0.02, 0.007] + [0.0015] * 13
    top = lp_of(probs)
    got = sm.implied(top, 1.0, top_k=20, top_p=0.95)
    assert got["status"] == "beyond_top", got
    # top_p 0.85: 0.80 < 0.85 x 0.9745 and 0.88 >= 0.85, so 4 tokens whatever the unlisted ones hold
    assert sm.implied(top, 1.0, top_k=20, top_p=0.85)["kept"] == 4


def test_implied_sglang_order_must_agree():
    """SGLang: top-p over the whole distribution, then the first top_k. top_k 5 holding 0.85 of the mass, top_p 0.9:
    over the top 5 renormalised top-p keeps 4 (0.78 / 0.85 = 0.918 >= 0.9 before the 5th); over everything it keeps
    5 (0.78 < 0.9). At top_p 0.5 both keep 2."""
    probs = [0.4, 0.2, 0.1, 0.08, 0.07] + [0.15 / 15] * 15
    top = lp_of(probs)
    got = sm.implied(top, 1.0, top_k=5, top_p=0.9)
    assert got["status"] == "top_p_order" and "keeps 4 tokens" in got["detail"], got
    assert sm.implied(top, 1.0, top_k=5, top_p=0.5)["kept"] == 2
    assert sm.implied(top, 1.0, top_k=None, top_p=0.5)["status"] == "ok"  # no top-k: the same rule in every engine


def test_implied_min_p_order_and_greedy():
    top = lp_of([0.5, 0.3, 0.12, 0.08])
    # min_p 0.2 (threshold 0.1) removes 0.08; top_p 0.99 keeps the other three in either order
    assert sm.implied(top, 1.0, top_p=0.99, min_p=0.2)["kept"] == 3
    # min_p 0.01 removes nothing: top_p 0.9 keeps three (0.8 < 0.9 <= 0.92) in either order
    assert sm.implied(top, 1.0, top_p=0.9, min_p=0.01)["kept"] == 3
    # top_p 0.85 then min_p keeps three (0.8 < 0.85); min_p first renormalises 0.92 of mass, and top_p then
    # keeps two (0.5/0.92 = 0.54 < 0.85 <= 0.8/0.92 = 0.87): the engines' orders disagree
    assert sm.implied(top, 1.0, top_p=0.85, min_p=0.2)["status"] == "min_p_order"
    assert sm.implied(top, 0.0)["probs"] == {0: 1.0}
    assert sm.implied(lp_of([0.4, 0.3998, 0.2002]), 0.0)["status"] == "boundary"


def test_implied_refuses_bad_logprobs():
    assert sm.implied([], 1.0)["status"] == "bad_logprobs"
    assert sm.implied([(0, None)], 1.0)["status"] == "bad_logprobs"
    assert sm.implied([(0, 0.5)], 1.0)["status"] == "bad_logprobs"


def test_settings_parse():
    cfg, unsent = sm.sampling_settings({"temperature": 0.6, "top_k": 20, "top_p": 0.95})
    assert cfg == {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "min_p": None}
    assert unsent == ["min_p", "repetition_penalty"]  # vLLM fills these from generation_config.json when absent
    cfg, unsent = sm.sampling_settings({"temperature": 1, "top_k": -1, "top_p": 1.0, "min_p": 0,
                                        "repetition_penalty": 1.0})
    assert cfg == {"temperature": 1.0, "top_k": None, "top_p": None, "min_p": None} and unsent == []
    for bad in ({}, {"temperature": -1}, {"temperature": 1, "top_k": 2.5}, {"temperature": 1, "min_p": 2}):
        with pytest.raises(ValueError):
            sm.sampling_settings(bad)
