"""Anti-cheat gates: each planted cheat is caught by its gate, and an honest engine passes every gate."""

import json
import math
import re

import pytest

from inferencekit import evidence as ev
from inferencekit import prompts as prompt_sets
from inferencekit import registry
from inferencekit.ops import gates
from cheats.fake_engine import FakeEngine

LIVE = ["caching", "token_count", "output_length", "hidden_probes", "rare_tokens"]


@pytest.fixture(autouse=True)
def char_tokenizer(monkeypatch, store):
    # The fake model's tokens are characters; "--tokenizer chars" re-tokenizes the same way.
    monkeypatch.setattr(gates, "load_tokenizer", lambda name: (lambda text: len(text)) if name else None)


def run(url, **kw):
    args = {"base_url": url, "max_tokens": 24, "gates": LIVE, "tokenizer": "chars", **kw}
    rec = registry.call("run_gates", args)
    assert ev.verify(rec)["id_ok"]
    return rec, {name: g["verdict"] for name, g in rec["data"]["gates"].items()}


def test_honest_engine_passes_every_live_gate():
    with FakeEngine("honest") as eng:
        rec, v = run(eng.url)
    assert v == {g: "PASS" for g in LIVE}, json.dumps(rec["data"]["gates"], indent=1)[:4000]
    assert rec["data"]["verdict"] == "PASS" and rec["claim_class"] == "observed"
    assert any(a["role"] == "probe requests and replies" for a in rec["artifacts"])


def test_honest_engine_without_token_ids_or_logprobs_passes_what_it_can_and_is_unproven_on_logprobs():
    with FakeEngine("honest", token_ids=False, logprobs=False) as eng:
        rec, v = run(eng.url)
    assert v["caching"] == v["token_count"] == v["output_length"] == v["rare_tokens"] == "PASS", v
    # Sampling half PASS, but the logprob half could not run: missing evidence is never a pass.
    g = rec["data"]["gates"]["hidden_probes"]
    assert v["hidden_probes"] == "UNPROVEN" and "without logprobs" in g["reason"]
    assert rec["data"]["verdict"] == "UNPROVEN"


def test_refusing_engine_never_passes():
    # Hides a pruned vocabulary behind errors: 500 on any non-ASCII prompt, 400 on ignore_eos / top_logprobs.
    # Seed 0 includes an answered ASCII probe, so these cases reach refusal judging rather than the
    # no-usable-reply liveness path (covered separately below).
    with FakeEngine("refuser") as eng:
        rec, v = run(eng.url, seed=0)
    gs = rec["data"]["gates"]
    assert v["rare_tokens"] == "FAIL" and gs["rare_tokens"]["metrics"]["refused"] >= 8, gs["rare_tokens"]
    # A non-ASCII prompt gets the 500 before the unsupported-parameter 400.
    assert v["output_length"] == "UNPROVEN" and re.search(r"HTTP (400|500)", gs["output_length"]["reason"])
    assert v["hidden_probes"] == "UNPROVEN" and re.search(r"HTTP (400|500)", gs["hidden_probes"]["reason"])
    assert rec["data"]["verdict"] == "FAIL"
    with FakeEngine("refuser") as cand, FakeEngine("refuser") as ref:  # refusals the reference shares are excused
        rec, v = run(cand.url, gates=["rare_tokens"], reference_url=ref.url, seed=0)
    assert v["rare_tokens"] == "UNPROVEN" and rec["data"]["verdict"] == "UNPROVEN"
    with FakeEngine("refuser") as cand, FakeEngine("honest") as ref:
        rec, v = run(cand.url, gates=["rare_tokens"], reference_url=ref.url, seed=0)
    assert v["rare_tokens"] == "FAIL"


def test_refusing_every_probe_is_not_run_and_overall_unproven():
    # Seed 138 draws non-ASCII warmups and non-ASCII echo delimiters even for the ASCII code/number
    # targets. Both engines refuse every probe: no usable reply establishes liveness.
    with FakeEngine("refuser") as cand, FakeEngine("refuser") as ref:
        rec, verdicts = run(cand.url, gates=["rare_tokens"], reference_url=ref.url, seed=138)
    assert verdicts == {"rare_tokens": "NOT RUN"}
    assert "HTTP 500" in rec["data"]["gates"]["rare_tokens"]["reason"]
    assert rec["data"]["not_run_but_expected"] == ["rare_tokens"]
    assert rec["data"]["verdict"] == "UNPROVEN"


def test_engine_without_ignore_eos_is_unproven_not_failed():
    # max_tokens above the fake's natural EOS (24-71 tokens), so stopping at EOS is visible.
    with FakeEngine("no_eos") as eng:
        rec, v = run(eng.url, gates=["output_length"], max_tokens=120)
    g = rec["data"]["gates"]["output_length"]
    assert v["output_length"] == "UNPROVEN" and "reference" in g["reason"], g
    with FakeEngine("honest") as eng:
        rec, v = run(eng.url, gates=["output_length"], max_tokens=120)
    assert v["output_length"] == "PASS"
    with FakeEngine("truncate") as eng:  # finish_reason 'length' below max_tokens and lying usage stays FAIL
        rec, v = run(eng.url, gates=["output_length"], max_tokens=120)
    assert v["output_length"] == "FAIL"


def test_probes_are_not_predictable_from_the_code():
    hashes, seeds = [], []
    for _ in range(2):
        with FakeEngine("honest") as eng:
            rec, v = run(eng.url, gates=["caching", "rare_tokens"])
        assert v == {"caching": "PASS", "rare_tokens": "PASS"}, rec["data"]["gates"]
        seeds.append(rec["data"]["seed"])
        art = next(a for a in rec["artifacts"] if a["role"] == "probe requests and replies")
        probes = (ev.Store().run_dir(rec["evidence_id"]) / art["path"]).read_text().splitlines()
        hashes.append({json.loads(x)["prompt_sha256"] for x in probes})
    assert seeds[0] != seeds[1]  # drawn fresh per run and recorded
    assert not hashes[0] & hashes[1]  # no probe prompt repeats across runs
    import random

    rng = random.Random(5)
    styles = {gates.echo_prompt("PAYLOAD", gates.echo_style(rng)).replace("PAYLOAD", "") for _ in range(40)}
    assert len(styles) > 10
    targets = gates.rare_targets(random.Random(1)), gates.rare_targets(random.Random(2))
    assert {t for _, t in targets[0]}.isdisjoint({t for _, t in targets[1]})


def test_late_spec_divergence_is_not_a_pass():
    with FakeEngine("spec_late") as a, FakeEngine("honest") as b:
        rec = registry.call("gates_spec_exact", {"a": a.url, "b": b.url, "limit": 16, "max_tokens": 64})
    g = rec["data"]["gates"]["spec_exact"]
    assert g["metrics"]["ab_early_divergences"] == 0 and g["metrics"]["ab_identical"] < g["metrics"]["aa_identical"]
    assert g["verdict"] == "UNPROVEN" and rec["data"]["verdict"] == "UNPROVEN", g


def test_output_caching_engine_is_caught_by_the_caching_gate():
    with FakeEngine("cache") as eng:
        rec, v = run(eng.url)
    g = rec["data"]["gates"]["caching"]
    assert v["caching"] == "FAIL", g
    assert g["metrics"]["perturbed_cache_hits"] >= 1
    assert "faster" in g["reason"] and "one character" in g["reason"]
    assert rec["data"]["verdict"] == "FAIL"


def test_truncating_engine_is_caught_by_token_count_and_output_length():
    with FakeEngine("truncate") as eng:
        rec, v = run(eng.url)
    assert v["token_count"] == "FAIL" and v["output_length"] == "FAIL", v


def test_truncating_engine_without_ids_is_caught_by_retokenizing():
    with FakeEngine("truncate", token_ids=False, logprobs=False) as eng:
        rec, v = run(eng.url)
    assert v["token_count"] == "FAIL" and v["output_length"] == "FAIL", v
    assert "re-tokenized" in json.dumps(rec["data"]["gates"]["token_count"]["details"])


def test_truncating_engine_with_logprobs_only_never_passes():
    # Without token ids, a reply cut at max_tokens whose logprob stream is short of usage looks the
    # same as an honest engine that does not stream its special tokens: UNPROVEN, never PASS...
    with FakeEngine("truncate", token_ids=False) as eng:
        rec, v = run(eng.url, gates=["token_count", "output_length"], tokenizer=None)
    assert v == {"token_count": "UNPROVEN", "output_length": "UNPROVEN"}, rec["data"]["gates"]
    assert rec["data"]["verdict"] == "UNPROVEN"
    # ...but the replies without logprobs (the warm-ups) re-tokenize short of usage...
    with FakeEngine("truncate", token_ids=False) as eng:
        rec, v = run(eng.url, gates=["token_count", "output_length"])
    assert v["token_count"] == "FAIL" and "re-tokenized" in json.dumps(rec["data"]["gates"]["token_count"]["details"])
    # ...and a reply that ended at EOS has no such excuse, so the count probes catch it there.
    with FakeEngine("truncate", token_ids=False) as eng:
        rec, v = run(eng.url, gates=["token_count", "output_length"], max_tokens=120, tokenizer=None)
    assert v["token_count"] == "FAIL" and rec["data"]["verdict"] == "FAIL", rec["data"]["gates"]


def test_copied_strings_that_re_tokenize_differently_do_not_fail_token_count(monkeypatch):
    # 13 Oct, run with a tokenizer: rare:r5 'ArrF क्षत्रिय ज्ञान 6bzC' came back verbatim with usage 17, the
    # text re-tokenized to 14, and nothing streamed a count (rare probes ask for no logprobs, and the engine
    # gave no token ids). That FAILed token_count. Here the tokenizer miscounts non-ASCII text the same way.
    monkeypatch.setattr(gates, "load_tokenizer",
                        lambda name: (lambda text: len(text) + sum(ord(c) > 127 for c in text)) if name else None)
    with FakeEngine("honest", token_ids=False, logprobs=False) as eng:
        rec, v = run(eng.url, gates=["token_count", "rare_tokens"])
    g = rec["data"]["gates"]["token_count"]
    assert v == {"token_count": "PASS", "rare_tokens": "PASS"}, g
    assert g["metrics"]["retokenize_only_gaps"] >= 1 and g["metrics"]["mismatches"] == 0
    assert any("count probes check usage" in x for x in g["limitations"]), g["limitations"]
    assert all(d["key"].startswith("rare:") for d in g["details"])
    with FakeEngine("truncate", token_ids=False, logprobs=False) as eng:  # a lying usage is still caught
        rec, v = run(eng.url, gates=["token_count", "rare_tokens"])
    assert v["token_count"] == "FAIL", rec["data"]["gates"]["token_count"]


def test_engine_that_does_not_stream_special_tokens_is_unproven_not_failed():
    # SGLang-like: under ignore_eos the special tokens generated after EOS are counted in usage but
    # never streamed, and an EOS that ends a reply is counted but not streamed. max_tokens above the
    # fake's natural EOS (24-71 tokens), so there are tokens after EOS.
    with FakeEngine("sglang_like", token_ids=False) as eng:
        rec, v = run(eng.url, gates=["token_count", "output_length"], max_tokens=120)
    gs = rec["data"]["gates"]
    assert v == {"token_count": "UNPROVEN", "output_length": "UNPROVEN"}, gs
    for g in ("token_count", "output_length"):
        assert "not streamed: special tokens after EOS?" in gs[g]["reason"], gs[g]
    assert gs["output_length"]["metrics"]["unstreamed_at_limit"] == 3
    assert all(d["problem"].endswith("tokens not streamed: special tokens after EOS?")
               for d in gs["output_length"]["details"])
    # The same engine honouring skip_special_tokens: false streams them, and everything agrees. Its
    # "<eos>" text re-tokenizes to more tokens than were generated: a limitation, not a mismatch.
    with FakeEngine("sglang_like", token_ids=False, skip_special=True) as eng:
        rec, v = run(eng.url, gates=["token_count", "output_length"], max_tokens=120)
    gs = rec["data"]["gates"]
    assert v == {"token_count": "PASS", "output_length": "PASS"}, gs
    assert gs["token_count"]["metrics"]["retokenize_only_gaps"] == 3
    assert any("re-tokenize" in x for x in gs["token_count"]["limitations"])
    with FakeEngine("sglang_like", token_ids=False) as eng:  # with every other gate too: nothing FAILs
        rec, v = run(eng.url, max_tokens=120)
    assert "FAIL" not in v.values(), v


def _reply(usage, streamed, how="logprobs", finish="length", text="x"):
    r = gates.Reply(status=200, text=text, finish_reason=finish, usage={"completion_tokens": usage})
    if how == "token_ids":
        r.token_ids = list(range(streamed))
    else:
        r.lp = [{"token": "x", "logprob": -0.1}] * streamed
    return r


def test_count_rules_under_the_max_tokens_limit():
    def judge(key, r, mt=64, retok=None):
        rows = gates.count_rows([(key, r, mt)], (lambda t: retok) if retok is not None else None)
        return gates.judge_counts(rows, 0.02)

    # Logprob stream short of usage at the limit: unsettled, on count and ignore_eos replies alike.
    assert len(judge("ignore_eos:l0", _reply(64, 61))["unsettled"]) == 1
    assert len(judge("count:n0", _reply(64, 63))["unsettled"]) == 1
    # Token ids are one per token: short of usage is a lie, not a streaming quirk.
    assert len(judge("ignore_eos:l0", _reply(64, 61, how="token_ids"))["bad"]) == 1
    # Not at the limit, or the stream is longer than usage: FAIL as before.
    assert len(judge("count:n0", _reply(63, 61))["bad"]) == 1  # finish 'length' with usage < max_tokens
    assert len(judge("count:n0", _reply(64, 65))["bad"]) == 1  # streamed > usage
    assert len(judge("count:n0", _reply(64, 61, finish="stop"))["bad"]) == 1  # stopped at EOS: no excuse
    assert len(judge("count:n0", _reply(64, 61), mt=None)["bad"]) == 1  # limit unknown (run-dir rows)
    # Re-tokenized text is skipped where the stream already agrees with usage (ignore_eos, echo, rare)...
    for key in ("ignore_eos:l0", "rare:r5", "echo_orig:e1", "echo_perturbed:e1"):
        got = judge(key, _reply(22, 22, how="token_ids", finish="stop"), retok=17)
        assert got["bad"] == [] and len(got["retok_only"]) == 1, key
    # ...and on copy probes (rare, echo: no logprobs asked) where usage and the text are the only counts:
    # the 13 Oct rare:r5 row (usage 17, re-tokenized 14, nothing streamed), replayed. Nothing is checked there.
    for key in ("rare:r5", "echo_orig:e0", "echo_perturbed:e2"):
        r = gates.Reply(status=200, text="ArrF क्षत्रिय ज्ञान 6bzC", finish_reason="stop",
                        usage={"completion_tokens": 17})
        got = judge(key, r, retok=14)
        assert got["bad"] == [] and len(got["retok_only"]) == 1 and got["checked"] == 0, (key, got)
    # It still counts elsewhere, and on ignore_eos replies where it is the only count besides usage.
    assert len(judge("count:n0", _reply(22, 22, how="token_ids", finish="stop"), retok=17)["bad"]) == 1
    r = gates.Reply(status=200, text="x", finish_reason="length", usage={"completion_tokens": 64})
    assert len(judge("ignore_eos:l0", r, retok=40)["bad"]) == 1
    assert len(judge("warmup:w0", gates.Reply(status=200, text="x", finish_reason="stop",
                                              usage={"completion_tokens": 17}), retok=14)["bad"]) == 1
    # A short stream whose text covers usage is proven by the text (13 Oct count:n0: usage 64, streamed 63,
    # re-tokenized 64; observed). A text as short as the stream proves nothing: still unsettled.
    got = judge("count:n0", _reply(64, 63), retok=64)
    assert got["bad"] == got["unsettled"] == [] and len(got["by_text"]) == 1 and got["checked"] == 1, got
    got = judge("count:n0", _reply(64, 32), retok=32)
    assert got["bad"] == [] and len(got["unsettled"]) == 1, got
    # The note says what the gap may be: special tokens after EOS only where ignore_eos was asked for.
    assert got["unsettled"][0]["problem"] == ("32 tokens counted but not streamed (a held-back partial character "
                                              "at the cut?)")
    assert judge("ignore_eos:l0", _reply(64, 61))["unsettled"][0]["problem"] == \
        "3 tokens not streamed: special tokens after EOS?"
    rows = [{"key": "ignore_eos:l0", "usage": 64, "streamed": 61},
            {"key": "ignore_eos:l1", "usage": 64, "streamed": 63}, {"key": "count:n0", "usage": 64, "streamed": 63}]
    assert gates.unstreamed_note(rows) == ("1 to 3 tokens not streamed: special tokens after EOS?; 1 tokens counted "
                                           "but not streamed (a held-back partial character at the cut?)")


def test_output_length_rules_under_ignore_eos():
    def verdict(*replies, mt=64):
        jobs = {"ignore_eos": [gates.Job("ignore_eos", f"l{i}", {"max_tokens": mt}, {}, r)
                               for i, r in enumerate(replies)]}
        return gates.gate_output_length_live(jobs, None)

    assert verdict(_reply(64, 64), _reply(64, 64, how="token_ids"))["verdict"] == "PASS"
    g = verdict(_reply(64, 64), _reply(64, 52))
    assert g["verdict"] == "UNPROVEN" and "12 tokens not streamed" in g["reason"], g
    for bad in (_reply(65, 65),  # usage above max_tokens
                _reply(64, 66),  # streamed more than usage
                _reply(60, 60),  # finish 'length' below max_tokens
                _reply(64, 32, how="token_ids")):  # ids short of usage
        g = verdict(_reply(64, 64), bad)
        assert g["verdict"] == "FAIL", (bad.summary(), g)
    g = verdict(_reply(40, 40, finish="stop"))  # stopped at EOS, counts consistent: ignore_eos unsupported
    assert g["verdict"] == "UNPROVEN" and "does not honour ignore_eos" in g["reason"]
    # A short logprob stream whose re-tokenized text covers max_tokens: the text proves the length.
    jobs = {"ignore_eos": [gates.Job("ignore_eos", "l0", {"max_tokens": 64}, {}, _reply(64, 52))]}
    g = gates.gate_output_length_live(jobs, lambda text: 64)
    assert g["verdict"] == "PASS" and g["metrics"]["unstreamed_settled_by_text"] == 1, g
    assert any("proven by the text" in x for x in g["limitations"])
    g = gates.gate_output_length_live(jobs, lambda text: 52)
    assert g["verdict"] == "UNPROVEN" and "no token ids" in g["reason"], g


def test_pruned_vocabulary_engine_is_caught_by_the_rare_token_pack():
    with FakeEngine("pruned") as eng:
        rec, v = run(eng.url, gates=["rare_tokens"])
    g = rec["data"]["gates"]["rare_tokens"]
    assert v["rare_tokens"] == "FAIL"
    assert {"cjk", "emoji"} <= {m["class"] for m in g["details"]}
    assert all(m["class"] not in ("code", "number") for m in g["details"])


def test_a_slip_in_the_ascii_nonce_around_a_rare_string_is_not_a_miss():
    # 14 Oct: stock vLLM returned "wxS7 <emoji> 72R" for "wxS7 <emoji> 72Ry"; the rare part was intact.
    with FakeEngine("nonce_slip") as eng:
        rec, v = run(eng.url, gates=["rare_tokens"])
    g = rec["data"]["gates"]["rare_tokens"]
    assert v["rare_tokens"] == "PASS", g
    assert g["metrics"]["nonce_slips"] == 8 and g["metrics"]["counted_misses"] == 0  # the 8 non-ASCII classes
    assert any("slipped random ASCII nonce" in x for x in g["limitations"])
    assert all(d["reproduced"] and not d["nonces_intact"] for d in g["details"])
    with FakeEngine("nonce_slip") as cand, FakeEngine("honest") as ref:
        rec, v = run(cand.url, gates=["rare_tokens"], reference_url=ref.url)
    assert v["rare_tokens"] == "PASS"
    import random

    for seed in range(20):
        for cls, t in gates.rare_targets(random.Random(seed)):
            core = gates.rare_core(t)
            head, tail = t[:len(t) - len(core) - 6], t[-4:]
            assert t == f"{head} {core} {tail}" and len(head) == 4 and " " not in head + tail, (cls, t)
    assert gates.rare_core("ab12 한국어 맞춤법 검사기 cd34") == "한국어 맞춤법 검사기"
    assert gates.rare_core("no-wrapper") == "no-wrapper"


def test_the_observed_rare_token_failures_now_pass_and_a_damaged_string_still_fails():
    # The three rare_tokens FAILs in the 13-14 Oct records (target, reply text; each ended at EOS, 12-19
    # of 64 tokens): a cut or dropped trailing nonce, the rare string intact.
    seen = [("cjk", "cyhX 饕餮纹的青铜鼎 wVUR", "cyhX 饕餮纹的青铜鼎"),
            ("emoji", "wxS7 🧑‍🤝‍🧑 👨🏻‍💻 72Ry", "wxS7 🧑‍🤝‍🧑 👨🏻‍💻 72R"),
            ("hangul", "RLav 닭볶음탕과 떡볶이 G7HG", "RLav 닭볶음탕과 떡볶이 G7H")]

    def judge(cases):
        jobs = {"rare": [gates.Job("rare", f"r{i}", {}, {"class": c, "target": t, "core": gates.rare_core(t)},
                                   gates.Reply(status=200, text=got, finish_reason="stop"))
                         for i, (c, t, got) in enumerate(cases)]}
        return gates.gate_rare(jobs)

    g = judge(seen)
    assert g["verdict"] == "PASS" and g["metrics"]["nonce_slips"] == 3, g
    damaged = [("cjk", "cyhX 饕餮纹的青铜鼎 wVUR", "cyhX 饕餮纹的青 wVUR")]  # a rare character lost: still a miss
    assert judge(damaged)["verdict"] == "FAIL"


def test_rare_misses_shared_with_the_reference_do_not_count():
    with FakeEngine("pruned") as cand, FakeEngine("pruned") as ref:
        rec, v = run(cand.url, gates=["rare_tokens"], reference_url=ref.url)
    assert v["rare_tokens"] == "PASS"
    with FakeEngine("pruned") as cand, FakeEngine("honest") as ref:
        rec, v = run(cand.url, gates=["rare_tokens"], reference_url=ref.url)
    assert v["rare_tokens"] == "FAIL"


def test_temperature_ignoring_engine_is_caught_by_the_hidden_probes():
    with FakeEngine("notemp") as eng:
        rec, v = run(eng.url, gates=["hidden_probes"])
    g = rec["data"]["gates"]["hidden_probes"]
    assert v["hidden_probes"] == "FAIL" and "temperature" in g["reason"]
    assert g["metrics"]["sampling_prompts"] == g["metrics"]["samples_equal_greedy"] == 8
    with FakeEngine("notemp") as eng:  # three prompts can be chance: never a FAIL on its own
        rec, v = run(eng.url, gates=["hidden_probes"], sampling_prompts=3)
    g = rec["data"]["gates"]["hidden_probes"]
    assert v["hidden_probes"] == "UNPROVEN" and "3/3" in g["reason"]
    assert any("too few to FAIL" in x for x in g["limitations"])


def test_confident_honest_model_is_not_failed_by_the_hidden_probes():
    # A confident model returns its greedy answer at T=1 on some prompts (4 of 8 here). The old
    # rule (FAIL when half the prompts match) failed exactly this engine.
    with FakeEngine("confident") as eng:
        rec, v = run(eng.url, gates=["hidden_probes"])
    g = rec["data"]["gates"]["hidden_probes"]
    assert v["hidden_probes"] == "PASS", g
    assert g["metrics"]["samples_equal_greedy"] == 4 and g["metrics"]["sampling_prompts"] == 8
    with FakeEngine("confident", confident=0.75) as eng:  # 6 of 8: suspicious, not proof
        rec, v = run(eng.url, gates=["hidden_probes"])
    assert v["hidden_probes"] == "UNPROVEN" and "6/8" in rec["data"]["gates"]["hidden_probes"]["reason"]
    with FakeEngine("confident", confident=1.0) as eng:  # greedy everywhere looks like ignoring temperature
        rec, v = run(eng.url, gates=["hidden_probes"])
    assert v["hidden_probes"] == "FAIL"


def test_sampling_rule_and_its_chance_fail_rate():
    table = {(8, 8): "FAIL", (7, 8): "UNPROVEN", (6, 8): "UNPROVEN", (5, 8): "PASS", (0, 8): "PASS",
             (4, 4): "FAIL", (3, 4): "UNPROVEN", (3, 3): "UNPROVEN", (2, 3): "PASS", (1, 1): "UNPROVEN"}
    for (eq, n), want in table.items():
        assert gates.sampling_verdict(eq, n)[0] == want, (eq, n)
    # The numbers in the comment above gates.sampling_verdict, at p = 1/3 per prompt.
    p = 1 / 3

    def at_least(k, n):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))

    assert abs(at_least(2, 3) - 7 / 27) < 1e-12 and 0.25 < 7 / 27 < 0.27  # old rule: ~26% of honest runs
    assert abs(at_least(8, 8) - 1 / 6561) < 1e-15 and 1.5e-4 < 1 / 6561 < 1.6e-4
    assert 0.019 < at_least(6, 8) - at_least(8, 8) < 0.02  # UNPROVEN ~2%


def test_probe_budget_follows_the_gates_and_the_sampling_count():
    assert gates.probes_needed(set(LIVE), 6, 8) == 2 + 6 + 4 + 3 + 8 + 3 == 26  # fits the 32 dev short/medium
    assert gates.probes_needed({"hidden_probes"}, 6, 8) == 13
    assert gates.probes_needed({"rare_tokens"}, 6, 8) == 2
    with FakeEngine("honest") as eng, pytest.raises(registry.OpError, match="need 42"):
        run(eng.url, sampling_prompts=24)
    with pytest.raises(registry.OpError, match="at least 1"):
        registry.call("run_gates", {"base_url": "http://127.0.0.1:9", "model": "m", "sampling_prompts": 0})


def test_a_small_prompt_set_gets_fewer_sampling_prompts_unless_a_count_is_asked_for(tmp_path):
    pool = [it for it in prompt_sets.load("dev") if it["bucket"] in ("short", "medium")]
    assert gates.sampling_count(32, set(LIVE), 6, None) == (8, None)
    assert gates.sampling_count(24, set(LIVE), 6, None)[0] == 6  # 18 prompts go to the other probes
    assert gates.sampling_count(10, set(LIVE), 6, None)[0] == 4  # never below 4 (then the set is too small)
    assert gates.sampling_count(24, set(LIVE), 6, 8) == (8, None)  # asked for: used as given
    small = tmp_path / "small.jsonl"
    small.write_text("".join(json.dumps(it) + "\n" for it in pool[:24]))
    with FakeEngine("honest") as eng:
        rec, v = run(eng.url, prompts=str(small))
    g = rec["data"]["gates"]["hidden_probes"]
    assert v == {x: "PASS" for x in LIVE} and g["metrics"]["sampling_prompts"] == 6, rec["data"]["gates"]
    assert any("room for 6 sampling prompts" in x and "1 in 729, not 1 in 6561" in x for x in g["limitations"])
    with FakeEngine("notemp") as eng:  # 6 prompts are still enough to catch an engine that ignores temperature
        rec, v = run(eng.url, gates=["hidden_probes"], prompts=str(small))
    assert v["hidden_probes"] == "FAIL"
    with FakeEngine("honest") as eng, pytest.raises(registry.OpError, match="need 26"):
        run(eng.url, prompts=str(small), sampling_prompts=8)


def test_probes_ask_for_the_full_distribution_and_for_special_tokens():
    with FakeEngine("honest") as eng:
        rec, v = run(eng.url, gates=["token_count", "output_length", "hidden_probes"])
    assert set(v.values()) == {"PASS"}, rec["data"]["gates"]
    art = next(a for a in rec["artifacts"] if a["role"] == "probe requests and replies")
    probes = [json.loads(x) for x in (ev.Store().run_dir(rec["evidence_id"]) / art["path"]).read_text().splitlines()]
    by_kind: dict = {}
    for p in probes:
        by_kind.setdefault(p["kind"], []).append(p["request"])
    for kind in ("sample_a", "sample_a2", "sample_b"):
        assert all(r["top_p"] == 1.0 and r["top_k"] == -1 and r["temperature"] == 1.0 for r in by_kind[kind])
    assert all("top_k" not in r for r in by_kind["greedy"])
    for kind in ("count", "ignore_eos"):
        assert all(r["skip_special_tokens"] is False for r in by_kind[kind])
    assert all("skip_special_tokens" not in r for k in ("greedy", "logprobs", "warmup") for r in by_kind[k])


def test_bad_logprobs_are_caught():
    entry = {"token": "a", "logprob": -0.1, "top_logprobs": [{"token": "b", "logprob": -0.05},
                                                             {"token": "a", "logprob": -0.1}]}
    assert any("argmax" in p for p in gates.check_logprob_entry(entry, greedy=True))
    unnorm = {"token": "a", "logprob": -0.01, "top_logprobs": [{"token": "a", "logprob": -0.01},
                                                               {"token": "b", "logprob": -0.02}]}
    assert any("normalised" in p for p in gates.check_logprob_entry(unnorm, greedy=True))
    nan = {"token": "a", "logprob": float("nan"), "top_logprobs": []}
    assert gates.check_logprob_entry(nan, greedy=True)


def test_spec_decode_divergence_is_caught_and_exact_spec_passes():
    with FakeEngine("spec_diverge") as a, FakeEngine("honest") as b:
        rec = registry.call("gates_spec_exact", {"a": a.url, "b": b.url, "limit": 12, "max_tokens": 24})
    g = rec["data"]["gates"]["spec_exact"]
    assert g["verdict"] == "FAIL" and g["metrics"]["ab_early_divergences"] > g["metrics"]["aa_early_divergences"]
    assert g["metrics"]["methods"] == ["token_ids"]
    with FakeEngine("honest") as a, FakeEngine("honest") as b:
        rec = registry.call("gates_spec_exact", {"a": a.url, "b": b.url, "limit": 12, "max_tokens": 24})
    assert rec["data"]["gates"]["spec_exact"]["verdict"] == "PASS"


def test_spec_exact_runs_inside_gates_run():
    with FakeEngine("spec_diverge") as a, FakeEngine("honest") as b:
        rec, v = run(a.url, gates=["spec_exact"], spec_off_url=b.url, spec_limit=8)
    assert v == {"spec_exact": "FAIL"}


def test_unreachable_endpoint_reports_not_run_or_errors(tmp_path):
    with pytest.raises(registry.OpError):
        registry.call("run_gates", {"base_url": "http://127.0.0.1:9", "timeout": 2})
    with FakeEngine("honest") as eng:
        url = eng.url
    # The server is gone, but the model name is given: every live gate is NOT RUN, and the record,
    # which expected them, is UNPROVEN (never PASS, never a quiet NOT RUN).
    rec = registry.call("run_gates", {"base_url": url, "model": "m", "gates": LIVE, "timeout": 2, "max_tokens": 8})
    assert {g["verdict"] for g in rec["data"]["gates"].values()} == {"NOT RUN"}
    assert rec["data"]["verdict"] == "UNPROVEN" and set(rec["data"]["not_run_but_expected"]) == set(LIVE)


# --------------------------------------------------------------------------- static run-dir checks

def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _run_dir(tmp_path, repeat=False, truncated=False, ref_scale=1.0):
    rd = tmp_path / "run"
    for i in range(10):
        pid = f"short-{i:04d}"
        n = 64
        got = 32 if truncated and i < 3 else n
        _write(rd / "inputs" / "short" / f"{pid}.json", {"messages": [], "max_tokens": n, "ignore_eos": i < 5})
        _write(rd / "outputs" / "cand" / "short" / f"{pid}.json",
               {"text": "x" * got, "token_ids": list(range(got)), "finish_reason": "length",
                "usage": {"completion_tokens": n if truncated else got}})
        m = int(round(n * ref_scale)) if i >= 5 else n
        _write(rd / "outputs" / "ref" / "short" / f"{pid}.json",
               {"text": "x" * m, "token_ids": list(range(m)), "finish_reason": "stop", "usage": {"completion_tokens": m}})
    lines = []
    for level in (1, 4):
        for j in range(8):
            pid = f"short-load-{(j if not (repeat and level == 4 and j == 7) else 0):04d}"
            lines.append(json.dumps({"level": level, "prompt_id": pid, "t_arrival_s": j * 0.1}))
    (rd / "load" / "cand").mkdir(parents=True)
    (rd / "load" / "cand" / "schedule.jsonl").write_text("\n".join(lines) + "\n")
    return rd


def _static(rd, **kw):
    rec = registry.call("run_gates", {"run_dir": str(rd), "engine": "cand", "reference_engine": "ref",
                                      "gates": ["token_count", "output_length", "unique_prompts"], **kw})
    return rec, {name: g["verdict"] for name, g in rec["data"]["gates"].items()}


def test_static_checks_pass_on_an_honest_run_dir(tmp_path):
    rec, v = _static(_run_dir(tmp_path))
    assert v == {"token_count": "PASS", "output_length": "PASS", "unique_prompts": "PASS"}, rec["data"]["gates"]
    assert rec["claim_class"] == "derived"


def test_static_checks_catch_repeats_truncation_and_length_drift(tmp_path):
    rec, v = _static(_run_dir(tmp_path / "a", repeat=True))
    assert v["unique_prompts"] == "FAIL" and v["output_length"] == "PASS"
    rec, v = _static(_run_dir(tmp_path / "b", truncated=True))
    assert v["token_count"] == "FAIL" and v["output_length"] == "FAIL"
    rec, v = _static(_run_dir(tmp_path / "c", ref_scale=1.5))
    assert v["output_length"] == "FAIL"
    assert "outside" in rec["data"]["gates"]["output_length"]["reason"]


def test_static_unique_prompts_not_run_without_a_schedule(tmp_path):
    rd = tmp_path / "empty"
    rd.mkdir()
    rec = registry.call("run_gates", {"run_dir": str(rd), "gates": ["unique_prompts"]})
    assert rec["data"]["gates"]["unique_prompts"]["verdict"] == "NOT RUN"
    assert rec["data"]["verdict"] == "UNPROVEN"  # asked for, not run: not a pass


def test_static_length_counts_empty_and_missing_outputs(tmp_path):
    rd = _run_dir(tmp_path)
    for i in range(5, 10):  # five candidate outputs come back empty...
        _write(rd / "outputs" / "cand" / "short" / f"short-{i:04d}.json",
               {"text": "", "token_ids": [], "finish_reason": "stop", "usage": {"completion_tokens": 0}})
    rec, v = _static(rd, gates=["output_length"])
    g = rec["data"]["gates"]["output_length"]
    assert v["output_length"] == "FAIL" and g["metrics"]["outside_tolerance"] == 5, g
    rd2 = _run_dir(tmp_path / "m")
    (rd2 / "outputs" / "cand" / "short" / "short-0009.json").unlink()  # ...or one is missing
    rec, v = _static(rd2, gates=["output_length"])
    g = rec["data"]["gates"]["output_length"]
    assert v["output_length"] == "UNPROVEN" and g["metrics"]["missing_outputs"] == 1
    assert g["metrics"]["coverage"] == 0.9
    for i in range(6, 9):  # four missing out of ten: beyond the 10% outlier budget
        (rd2 / "outputs" / "cand" / "short" / f"short-{i:04d}.json").unlink()
    rec, v = _static(rd2, gates=["output_length"])
    assert v["output_length"] == "FAIL"


def _serve_load(rd, repeat=False, salted=False):
    """The bench serve load layout: inputs/<engine>/load-<level>/<bucket>/<id>[~k].json."""
    for level in ("c1", "c4"):
        for j in range(6):
            pid = f"short-load-{j:04d}"
            _write(rd / "inputs" / "cand" / f"load-{level}" / "short" / f"{pid}.json",
                   {"messages": [{"role": "user", "content": f"prompt {j}"}], "max_tokens": 32})
        if repeat or salted:
            content = ("[ik-c4-6] " if salted else "") + "prompt 0"
            _write(rd / "inputs" / "cand" / f"load-{level}" / "short" / "short-load-0000~2.json",
                   {"messages": [{"role": "user", "content": content}], "max_tokens": 32})


def test_unique_prompts_reads_the_bench_serve_load_layout(tmp_path):
    for name, kw, want in (("ok", {}, "PASS"), ("rep", {"repeat": True}, "FAIL"), ("salt", {"salted": True}, "PASS")):
        rd = tmp_path / name
        _serve_load(rd, **kw)
        rec = registry.call("run_gates", {"run_dir": str(rd)})  # default gates: unique_prompts is expected here
        g = rec["data"]["gates"]["unique_prompts"]
        assert g["verdict"] == want and g["applicable"], (name, g)
        assert g["metrics"]["requests"] == 12 + (2 if kw else 0)
        if name == "salt":
            assert g["metrics"]["salted_reuses"] == 2 and g["limitations"]


def test_hidden_probes_mixed_into_a_load_run_are_checked_from_the_run_dir(tmp_path):
    bodies = gates.probe_bodies(seed=3, model="fake-model", max_tokens=24)
    assert {b["kind"] for b in bodies} == {"greedy", "sample_a", "sample_a2", "sample_b", "logprobs"}
    assert len({b["group"] for b in bodies if b["kind"] == "greedy"}) == 8  # the same budget as the live probes
    for b in bodies:
        sampled = b["kind"].startswith("sample")
        assert (b["body"].get("top_p"), b["body"].get("top_k")) == ((1.0, -1) if sampled else (None, None))
        assert b["optional"] == (["top_p", "top_k"] if sampled else [])
    for cheat, want in (("honest", "PASS"), ("notemp", "FAIL")):
        rd = tmp_path / cheat
        with FakeEngine(cheat) as eng:
            c = gates.Client(eng.url, "fake-model")
            for b in bodies:  # what a load runner would do: send, then save the reply
                r = c.send(b["body"])
                _write(rd / "outputs" / "cand" / "probes" / f"{b['id']}.json",
                       {"id": b["id"], "kind": b["kind"], "group": b["group"], "text": r.text,
                        "token_ids": r.token_ids, "logprobs": r.lp, "finish_reason": r.finish_reason,
                        "usage": r.usage})
        rec = registry.call("run_gates", {"run_dir": str(rd), "engine": "cand", "gates": ["hidden_probes"]})
        g = rec["data"]["gates"]["hidden_probes"]
        assert g["verdict"] == want, (cheat, g)


def test_perturb_changes_exactly_one_character():
    import random

    rng = random.Random(3)
    for _ in range(50):
        s = gates.nonce(rng)
        p = gates.perturb(s, rng)
        assert len(p) == len(s) and sum(a != b for a, b in zip(s, p)) == 1


def test_interleave_keeps_pair_order_and_staggers_repeats():
    import random

    first, again = [f"f{i}" for i in range(4)], [f"r{i}" for i in range(4)]
    chain = gates.stagger(first, again)
    assert chain == ["f0", "f1", "r0", "f2", "r1", "f3", "r2", "r3"]
    merged = gates.interleave([chain, ["x"], ["y"], ["z"]], random.Random(1))
    assert sorted(merged) == sorted(chain + ["x", "y", "z"])
    assert [m for m in merged if m in chain] == chain


def test_reasoning_tokens_count_as_output():
    r = gates.Reply(status=200, text="answer", reasoning="think ", usage={"completion_tokens": 12})
    rows = gates.count_rows([("k", r)], lambda t: len(t))
    assert rows[0]["retokenized"] == 12
    assert gates.judge_counts(rows, 0.02) == {"bad": [], "unsettled": [], "by_text": [], "retok_only": [],
                                              "checked": 1}


def test_extra_body_is_sent_and_rejected_extensions_are_dropped():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen = []

    class Strict(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            if "return_token_ids" in body:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "unknown field"}')
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\ndata: [DONE]\n\n")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Strict)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = gates.Client(f"http://127.0.0.1:{srv.server_address[1]}/v1", "m", extra={"top_k": 1})
        r1 = c.send(c.body([{"role": "user", "content": "x"}], 4))
        r2 = c.send(c.body([{"role": "user", "content": "y"}], 4))
    finally:
        srv.shutdown()
        srv.server_close()
    assert r1.ok and r1.text == "hi" and r1.dropped == ["return_token_ids"]
    assert r2.ok and len(seen) == 3  # 400, retry, then straight to the trimmed body
    assert all(b["top_k"] == 1 for b in seen)
    with pytest.raises(registry.OpError):
        registry.call("run_gates", {"base_url": "http://127.0.0.1:9", "model": "m", "extra_body": "[1]"})


def test_a_refused_sampling_field_is_dropped_without_losing_token_ids():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen = []

    class NoTopK(BaseHTTPRequestHandler):  # takes return_token_ids, refuses top_k / top_p (some servers do)
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            if "top_k" in body or "top_p" in body:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "top_k must be > 0"}')
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = {"choices": [{"index": 0, "delta": {"content": "hi"}, "token_ids": [7], "finish_reason": "stop"}]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\ndata: [DONE]\n\n")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), NoTopK)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = gates.Client(f"http://127.0.0.1:{srv.server_address[1]}/v1", "m")
        msgs = [{"role": "user", "content": "x"}]
        r0 = c.send(c.body(msgs, 4))
        r1 = c.send(c.body(msgs, 4, temperature=1.0, seed=1, full_vocab=True))
        r2 = c.send(c.body(msgs, 4, temperature=1.0, seed=2, full_vocab=True))
    finally:
        srv.shutdown()
        srv.server_close()
    assert r0.ok and r0.dropped == [] and r0.token_ids == [7]
    assert r1.ok and r1.dropped == ["top_p", "top_k"] and r1.token_ids == [7]
    assert r2.ok and r2.dropped == ["top_p", "top_k"]  # left out up front: no second failed round trip
    # r1: refused, refused again without top_k alone, then accepted without both.
    assert len(seen) == 5 and all(b.get("return_token_ids") for b in seen)
    assert [("top_p" in b, "top_k" in b) for b in seen[1:4]] == [(True, True), (True, False), (False, False)]
    assert c.rejected == {"top_p", "top_k"} and c.ask_ids


def test_a_refused_top_k_keeps_top_p_and_the_users_own_top_k():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen = []

    class TopKPositive(BaseHTTPRequestHandler):  # takes top_p and top_k > 0, refuses top_k = -1
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            if body.get("top_k") == -1:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "top_k must be > 0"}')
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = {"choices": [{"index": 0, "delta": {"content": "hi"}, "token_ids": [7], "finish_reason": "stop"}]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\ndata: [DONE]\n\n")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), TopKPositive)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = gates.Client(f"http://127.0.0.1:{srv.server_address[1]}/v1", "m", extra={"top_k": 64})
        msgs = [{"role": "user", "content": "x"}]
        g0 = c.send(c.body(msgs, 4))
        s1 = c.send(c.body(msgs, 4, temperature=1.0, seed=1, full_vocab=True))
        g1 = c.send(c.body(msgs, 4))
        s2 = c.send(c.body(msgs, 4, temperature=1.0, seed=2, full_vocab=True))
    finally:
        srv.shutdown()
        srv.server_close()
    assert g0.ok and g1.ok and g0.dropped == g1.dropped == []
    greedy = [b for b in seen if b["temperature"] == 0.0]
    assert len(greedy) == 2 and all(b["top_k"] == 64 for b in greedy)  # never retried, the user's top_k kept
    assert s1.ok and s1.dropped == ["top_k"] and s2.ok and s2.dropped == ["top_k"]
    sampled = [b for b in seen if b["temperature"] == 1.0]
    assert [b["top_k"] for b in sampled] == [-1, 64, 64]  # refused once, then the user's value up front
    assert all(b["top_p"] == 1.0 for b in sampled)  # refusing top_k did not take top_p with it
    assert c.rejected == {"top_k"} and "top_p" in c.accepted


def test_hidden_probes_records_refused_sampling_fields():
    jobs = {"sampling": []}
    for i in range(4):
        for kind, text in (("greedy", "g"), ("sample_a", f"a{i}"), ("sample_a2", f"a{i}"), ("sample_b", f"b{i}")):
            r = gates.Reply(status=200, text=text, finish_reason="stop")
            r.dropped = ["top_k"] if kind != "greedy" else []
            jobs["sampling"].append(gates.Job(kind, f"s{i}", {}, {}, r))
    g = gates.gate_hidden_probes(jobs, where="run dir")
    assert any("refused top_k on the T=1 samples" in x for x in g["limitations"]), g["limitations"]
    for j in jobs["sampling"]:
        j.reply.dropped = []
    assert not any("refused" in x for x in gates.gate_hidden_probes(jobs, where="run dir")["limitations"])


def test_hidden_probes_records_an_ignored_seed_without_refused_fields():
    """A seed that does not reproduce its sample is a limitation whether or not the server refused top_p/top_k,
    and never a FAIL on its own (batch-dependent flips can break it on an honest engine)."""
    jobs = {"sampling": []}
    for i in range(4):
        rerun = f"a{i}" if i else "other"  # prompt 0: the same seed gave a different sample
        for kind, text in (("greedy", "g"), ("sample_a", f"a{i}"), ("sample_a2", rerun), ("sample_b", f"b{i}")):
            jobs["sampling"].append(gates.Job(kind, f"s{i}", {}, {},
                                              gates.Reply(status=200, text=text, finish_reason="stop")))
    g = gates.gate_hidden_probes(jobs, where="run dir")
    assert g["metrics"]["seed_reproduced"] == "3/4"
    assert any("the same seed reproduced the sample on 3/4 prompts" in x for x in g["limitations"]), g["limitations"]
    assert not any("refused" in x for x in g["limitations"])
    # the sampling part passes; the gate as a whole is UNPROVEN here only because no logprob probe was sent
    assert g["verdict"] != "FAIL" and g["reason"].startswith("PASS: T=1 samples differ from greedy on 4/4"), g
    for j in jobs["sampling"]:
        if j.kind == "sample_a2" and j.key == "s0":
            j.reply.text = "a0"
    assert not any("same seed" in x for x in gates.gate_hidden_probes(jobs, where="run dir")["limitations"])


def test_missing_prompt_set_is_a_clean_error(tmp_path):
    with pytest.raises(registry.OpError, match="cannot read prompt set"):
        registry.call("run_gates", {"run_dir": str(tmp_path), "prompts": str(tmp_path / "nope.jsonl")})
