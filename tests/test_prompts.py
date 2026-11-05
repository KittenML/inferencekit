"""Prompt sets: shipped dev sets verify, builds are deterministic, dev and sealed are disjoint."""

import json
import random
import re

import pytest

from inferencekit import prompts as ps
from inferencekit import registry
from inferencekit.prompts import build
from inferencekit.registry import OpError


@pytest.fixture(autouse=True)
def no_private_rules(monkeypatch):
    # Hermetic: never read a scrub-patterns file that happens to exist on the test machine.
    monkeypatch.setenv("INFERENCEKIT_PROMPT_BLOCKLIST", "none")


def _texts(path):  # .jsonl or .jsonl.gz
    return {ps.prompt_text(json.loads(line)) for line in ps.read_text(path).splitlines() if line.strip()}


def _items(*paths):
    return [json.loads(line) for p in paths for line in ps.read_text(p).splitlines() if line.strip()]


def test_shipped_dev_sets_verify_and_have_the_contract_shape():
    for name in ("dev", "dev-load"):
        res = registry.call("prompts_verify", {"set": name})
        assert res["verdict"] == "PASS", res["checks"]
    dev = ps.load("dev")
    assert len(dev) == 64
    assert {b: sum(it["bucket"] == b for it in dev) for b in ps.BUCKETS} == {b: 16 for b in ps.BUCKETS}
    load = ps.load("dev-load")
    assert len(load) >= 1200 and {it["bucket"] for it in load} == {"short", "medium"}
    for it in dev + load:
        assert set(it) >= {"id", "bucket", "messages", "max_tokens", "source", "kind"}
        assert it["messages"][0]["role"] == "user" and it["kind"] == "chat"
        assert "/" in it["source"]  # dataset/licence
    # single-user and load sets share no prompt and no id
    assert not _texts(ps.resolve("dev")) & _texts(ps.resolve("dev-load"))
    assert not {it["id"] for it in dev} & {it["id"] for it in load}
    man = ps.load_manifest("dev")
    assert man["sealed"] is False and man["sha256"] == ps.sha256_file(ps.resolve("dev"))


def test_show_reports_per_bucket_stats():
    res = registry.call("prompts_show", {"set": "dev"})
    assert res["claim_class"] == "derived" and res["n"] == 64
    long = res["buckets"]["long"]["proxy_tokens"]
    assert 2000 <= long["min"] and long["max"] <= 4000


def _fake_sources(root):
    rng = random.Random(7)
    vocab = [f"w{i}" for i in range(400)]
    dolly = []
    for i in range(240):
        dolly.append({"instruction": f"What is thing {i}?", "context": "", "response": "x", "category": "open_qa"})
        dolly.append({"instruction": " ".join(rng.choice(vocab) for _ in range(20)) + f" item{i}?", "context": "",
                      "response": "x", "category": "open_qa"})
        if i < 80:
            ctx = " ".join(rng.choice(vocab) for _ in range(220))
            dolly.append({"instruction": f"Summarize item {i}.", "context": ctx, "response": "x",
                          "category": "summarization"})
            if i < 30:  # several rows ask about one context paragraph (as in dolly)
                for q in ("List the key facts.", "What is the main point of this text?"):
                    dolly.append({"instruction": q, "context": ctx, "response": "x", "category": "closed_qa"})
        if i % 40 == 7:
            dolly.append({"instruction": f"Why do zebra herds in region {i} move at dawn, and how far do they "
                                         "usually travel before the heat of the day sets in?", "context": "",
                          "response": "x", "category": "open_qa"})
    (root / "dolly.jsonl").write_text("".join(json.dumps(d) + "\n" for d in dolly))
    gsm = [{"question": " ".join(rng.choice(vocab) for _ in range(25)) + f" q{i}?", "answer": "1"} for i in range(200)]
    (root / "gsm8k_train.jsonl").write_text("".join(json.dumps(g) + "\n" for g in gsm))
    (root / "gutenberg").mkdir()
    for n in build.BOOKS:
        paras = ["\n".join(" ".join(rng.choice(vocab) for _ in range(12)) for _ in range(5)) for _ in range(400)]
        body = "HEADER\n*** START OF THE EBOOK X ***\n\n" + "\n\n".join(paras) + "\n\n*** END OF THE EBOOK X ***\nlicence"
        (root / "gutenberg" / f"pg{n}.txt").write_text(body)


@pytest.fixture
def small_build(monkeypatch):
    monkeypatch.setattr(build, "BOOKS", {1: "Book One", 2: "Book Two", 3: "Book Three", 4: "Book Four",
                                         5: "Book Five", 6: "Book Six"})
    monkeypatch.setattr(build, "PER_BUCKET", 2)
    monkeypatch.setattr(build, "LOAD_TARGET", 40)
    monkeypatch.setattr(build, "LOAD_MEDIUM_MAX", 10)


def test_build_is_deterministic_and_dev_sealed_are_disjoint(tmp_path, small_build):
    src = tmp_path / "src"
    src.mkdir()
    _fake_sources(src)
    a, b, sealed = tmp_path / "a", tmp_path / "b", tmp_path / "sealed"
    registry.call("prompts_build", {"sources": str(src), "out": str(a)})
    registry.call("prompts_build", {"sources": str(src), "out": str(b)})
    for name in ("dev.jsonl", "dev-load.jsonl", "dev.manifest.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    res = registry.call("prompts_seal", {"sources": str(src), "out": str(sealed)})
    assert {s["name"] for s in res["sets"]} == {"sealed", "sealed-load"}
    assert (sealed / "SEED.secret").is_file() and res["secret_sha256"]
    assert all(s["sealed"] for s in res["sets"])
    dev_texts = _texts(a / "dev.jsonl") | _texts(a / "dev-load.jsonl")
    sealed_texts = _texts(sealed / "sealed.jsonl") | _texts(sealed / "sealed-load.jsonl")
    assert dev_texts and sealed_texts and not dev_texts & sealed_texts
    sums = (sealed / "SHA256SUMS").read_text()
    assert ps.sha256_file(sealed / "sealed.jsonl") in sums
    ok = registry.call("prompts_verify", {"set": str(sealed / "sealed.jsonl"), "sums": str(sealed / "SHA256SUMS")})
    assert ok["verdict"] == "PASS", ok["checks"]


def test_verify_catches_tampering_and_wrong_preregistered_hash(tmp_path, small_build):
    src = tmp_path / "src"
    src.mkdir()
    _fake_sources(src)
    out = tmp_path / "out"
    registry.call("prompts_build", {"sources": str(src), "out": str(out)})
    path = out / "dev.jsonl"
    bad = registry.call("prompts_verify", {"set": str(path), "expected_sha256": "0" * 64})
    assert bad["verdict"] == "FAIL"
    lines = path.read_text().splitlines()
    lines[0] = lines[0].replace("What", "Wht")
    path.write_text("\n".join(lines) + "\n")
    res = registry.call("prompts_verify", {"set": str(path)})
    assert res["verdict"] == "FAIL"
    assert not next(c for c in res["checks"] if c["check"] == "manifest_sha256")["ok"]


def test_seal_refuses_a_git_work_tree(tmp_path, small_build):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    with pytest.raises(OpError, match="git work tree"):
        registry.call("prompts_seal", {"sources": str(tmp_path), "out": str(repo / "sealed")})


def test_split_is_a_stable_hash_of_the_item():
    assert build.split_of("dolly:what is x?") == build.split_of("dolly:what is x?")
    splits = [build.split_of(f"k{i}") for i in range(400)]
    assert 150 < splits.count("dev") < 250


def test_bucket_rules_use_the_word_proxy():
    assert ps.in_bucket("What is EDM?", "very_short")
    assert not ps.in_bucket("one two three four five six", "very_short")
    assert ps.proxy_tokens("a " * 100) == 130
    assert ps.in_bucket("w " * 20, "short") and not ps.in_bucket("w " * 40, "short")


def test_sealed_selection_depends_on_the_secret_and_is_reproducible_with_it(tmp_path, small_build):
    src = tmp_path / "src"
    src.mkdir()
    _fake_sources(src)
    for name, secret in (("s1", "a" * 32), ("s1b", "a" * 32), ("s2", "b" * 32)):
        (tmp_path / f"{name}.secret").write_text(secret)
        registry.call("prompts_seal", {"sources": str(src), "out": str(tmp_path / name),
                                       "secret_file": str(tmp_path / f"{name}.secret")})
    one, again, other = ((tmp_path / n / "sealed-load.jsonl").read_bytes() for n in ("s1", "s1b", "s2"))
    assert one == again and one != other
    pub = registry.call("prompts_seal", {"sources": str(src), "out": str(tmp_path / "pub"), "public": True})
    assert pub["secret_sha256"] is None
    registry.call("prompts_build", {"sources": str(src), "out": str(tmp_path / "dev")})
    dev = _texts(tmp_path / "dev" / "dev.jsonl") | _texts(tmp_path / "dev" / "dev-load.jsonl")
    for n in ("s1", "s2", "pub"):
        sealed = _texts(tmp_path / n / "sealed.jsonl") | _texts(tmp_path / n / "sealed-load.jsonl")
        assert not dev & sealed


def _shingles(text, n=12):
    w = text.lower().split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def _context(item):
    text = ps.prompt_text(item)
    return text.rsplit("\n\n", 1)[0] if item["bucket"] == "medium" else None


def test_rows_sharing_a_context_never_straddle_dev_and_sealed(tmp_path, small_build):
    src = tmp_path / "src"
    src.mkdir()
    _fake_sources(src)
    registry.call("prompts_build", {"sources": str(src), "out": str(tmp_path / "dev")})
    registry.call("prompts_seal", {"sources": str(src), "out": str(tmp_path / "sealed")})
    registry.call("prompts_seal", {"sources": str(src), "out": str(tmp_path / "pub"), "public": True})
    dev = _items(tmp_path / "dev" / "dev.jsonl", tmp_path / "dev" / "dev-load.jsonl")
    for name in ("sealed", "pub"):
        sealed = _items(tmp_path / name / "sealed.jsonl", tmp_path / name / "sealed-load.jsonl")
        dev_ctx = {_context(it) for it in dev} - {None}
        sealed_ctx = [_context(it) for it in sealed if _context(it)]
        assert sealed_ctx and not dev_ctx & set(sealed_ctx)
        dev_sh = {}
        for it in dev:
            for g in _shingles(ps.prompt_text(it)):
                dev_sh.setdefault(g, set()).add(it["id"])
        for it in sealed:  # no sealed item shares > 50% of its 12-word shingles with one dev item
            sh = _shingles(ps.prompt_text(it))
            counts = {}
            for g in sh:
                for d in dev_sh.get(g, ()):
                    counts[d] = counts.get(d, 0) + 1
            assert not sh or max(counts.values(), default=0) <= 0.5 * len(sh), it["id"]
    for items in (dev, sealed):  # and inside a split, one prompt per context paragraph (no shared prefix)
        ctx = [_context(it) for it in items if _context(it)]
        assert len(ctx) == len(set(ctx))


def test_private_rules_exclude_rows_and_the_exclusions_file_reproduces_the_build(tmp_path, small_build, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    _fake_sources(src)
    rules = tmp_path / "private" / "scrub-patterns.txt"
    rules.parent.mkdir()
    rules.write_text("#@section test\n#@severity critical\nregex:(?i)zebra==>animal\n")
    monkeypatch.setenv("INFERENCEKIT_PROMPT_BLOCKLIST", str(rules))
    res = registry.call("prompts_build", {"sources": str(src), "out": str(tmp_path / "a")})
    assert res["excluded_rows"] >= 1 and res["private_rules"] == 1
    excl = json.loads((tmp_path / "a" / "exclusions.json").read_text())
    assert "zebra" not in json.dumps(excl).lower()
    assert all(r.startswith("dolly:row") for refs in excl["excluded"].values() for r in refs)
    built = (tmp_path / "a" / "dev.jsonl").read_text() + (tmp_path / "a" / "dev-load.jsonl").read_text()
    assert "zebra" not in built.lower()
    # Without the private rules, the exclusions file alone gives the same bytes.
    monkeypatch.setenv("INFERENCEKIT_PROMPT_BLOCKLIST", "none")
    registry.call("prompts_build", {"sources": str(src), "out": str(tmp_path / "b"),
                                    "exclusions": str(tmp_path / "a" / "exclusions.json")})
    for name in ("dev.jsonl", "dev-load.jsonl", "dev.manifest.json", "exclusions.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name
    # Sealing with private rules the exclusions file does not cover is refused (not rebuildable).
    monkeypatch.setenv("INFERENCEKIT_PROMPT_BLOCKLIST", str(rules))
    with pytest.raises(OpError, match="exclusions"):
        registry.call("prompts_seal", {"sources": str(src), "out": str(tmp_path / "s")})
    registry.call("prompts_seal", {"sources": str(src), "out": str(tmp_path / "s"),
                                   "exclusions": str(tmp_path / "a" / "exclusions.json")})
    assert "zebra" not in (tmp_path / "s" / "sealed-load.jsonl").read_text().lower()


def test_private_rules_are_read_only_from_an_explicit_file(tmp_path, small_build, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    _fake_sources(src)
    rules = tmp_path / "private" / "scrub-patterns.txt"
    rules.parent.mkdir()
    rules.write_text("regex:(?i)zebra==>animal\n")
    # A rules file in a well-known directory is never picked up on its own.
    monkeypatch.delenv("INFERENCEKIT_PROMPT_BLOCKLIST", raising=False)
    monkeypatch.setenv("INFERENCEKIT_PRIVATE_DIR", str(rules.parent))
    res = registry.call("prompts_build", {"sources": str(src), "out": str(tmp_path / "a")})
    assert res["private_rules"] == 0
    res = registry.call("prompts_build", {"sources": str(src), "out": str(tmp_path / "b"), "blocklist": str(rules)})
    assert res["private_rules"] == 1 and res["excluded_rows"] >= 1
    with pytest.raises(OpError, match="blocklist file not found"):
        registry.call("prompts_build", {"sources": str(src), "out": str(tmp_path / "c"),
                                        "blocklist": str(tmp_path / "missing.txt")})


def test_verify_checks_a_revealed_secret_and_rebuilds_the_sealed_set(tmp_path, small_build):
    import hashlib

    src = tmp_path / "src"
    src.mkdir()
    _fake_sources(src)
    out = tmp_path / "sealed"
    res = registry.call("prompts_seal", {"sources": str(src), "out": str(out)})
    secret = out / "SEED.secret"
    assert res["secret_sha256"] == hashlib.sha256(secret.read_bytes()).hexdigest()  # = `sha256sum SEED.secret`
    assert "sha256sum" in json.loads((out / "sealed.manifest.json").read_text())["secret_rule"]
    for name in ("sealed.jsonl", "sealed-load.jsonl"):
        ok = registry.call("prompts_verify", {"set": str(out / name), "secret_file": str(secret),
                                              "sources": str(src), "sums": str(out / "SHA256SUMS")})
        assert ok["verdict"] == "PASS", ok["checks"]
        assert {c["check"] for c in ok["checks"]} >= {"secret_sha256", "rebuild"}
    wrong = tmp_path / "wrong.secret"
    wrong.write_text("f" * 64 + "\n")
    bad = registry.call("prompts_verify", {"set": str(out / "sealed.jsonl"), "secret_file": str(wrong),
                                           "sources": str(src)})
    assert bad["verdict"] == "FAIL"


def test_builder_source_holds_no_hashed_or_spelled_out_blocklist():
    from pathlib import Path

    src = Path(build.__file__).read_text(encoding="utf-8")
    assert not re.findall(r"\b[0-9a-f]{64}\b", src)
