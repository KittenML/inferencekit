"""prompts build / seal / verify / show: the versioned prompt sets every engine is run on.

``build`` writes the shipped dev sets (``dev``: 16 prompts per bucket; ``dev-load``:
1,200+ short/medium prompts for load tests) and ``exclusions.json``. ``seal`` writes the
sealed sets, built by the same code from disjoint source groups, to a directory *outside*
any git work tree, plus a ``SHA256SUMS`` file whose hashes go into the pre-registration.
``verify`` re-hashes a set against its manifest (and optionally an expected sha256) and
checks every item against the format and bucket rules; with ``--secret-file`` it also
checks the revealed secret against the manifest's ``secret_sha256`` (``sha256sum`` of the
file) and rebuilds the sealed set from the sources to compare hashes. ``show`` prints
per-bucket length statistics.

Numbers printed here are computed from the files (claim class ``derived``).
"""

from __future__ import annotations

import json
import os
import statistics
from collections import Counter
from pathlib import Path

from inferencekit import prompts as ps
from inferencekit.registry import Arg, OpError, operation

_DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "inferencekit", "prompt-sources")


def _git_root(path: Path) -> Path | None:
    p = path.resolve()
    for d in (p, *p.parents):
        if (d / ".git").exists():
            return d
    return None


def _cache(sources: str | None, fetch: bool) -> Path:
    from inferencekit.prompts import build

    cache = Path(sources or _DEFAULT_CACHE).expanduser()
    if fetch:
        build.fetch(cache, log=lambda *_: None)
    return cache


def _exclusions(path: str | None) -> dict:
    from inferencekit.prompts import build

    return build.read_exclusions(Path(path).expanduser() if path else ps.data_dir() / build.EXCLUSIONS_FILE)


_EXCL_ARG = Arg("exclusions", "string", "Exclusions file to apply (default: the package's exclusions.json).")
_BLOCK_ARG = Arg("blocklist", "string", "Private scrub-patterns file whose matches are excluded and recorded in "
                 "exclusions.json (default: $INFERENCEKIT_PROMPT_BLOCKLIST if set, else none).")


def _rules(blocklist: str | None) -> list:
    from inferencekit.prompts import build

    try:
        return build.load_rules(build.blocklist_path(blocklist))
    except FileNotFoundError as e:
        raise OpError(str(e)) from None


@operation("prompts_build", ("prompts", "build"), "Build the shipped dev prompt sets (dev, dev-load) "
           "deterministically from the downloaded open-licence sources.",
           [Arg("sources", "string", f"Source cache directory (default {_DEFAULT_CACHE.replace(os.path.expanduser('~'), '~')})."),
            Arg("out", "string", "Output directory (default: the package's prompts directory). The package's "
                "prompts directory gets .jsonl.gz like the shipped sets; any other directory plain .jsonl."),
            Arg("fetch", "boolean", "Download missing source files first.", default=False),
            Arg("seed", "string", "Split/rank seed (changing it changes every set).", default="inferencekit-prompts-v1"),
            _EXCL_ARG, _BLOCK_ARG])
def prompts_build(call, sources, out, fetch, seed, exclusions, blocklist):
    from inferencekit.prompts import build

    cache = _cache(sources, fetch)
    out_dir = Path(out).expanduser() if out else ps.data_dir()
    # The shipped sets stay gzipped however the package directory is named (scripts/build_prompts.py
    # passes it explicitly); a rebuild elsewhere writes plain JSONL. The manifests hash the JSONL either way.
    gz = out_dir.resolve() == ps.data_dir().resolve()
    rules = _rules(blocklist)
    try:
        mans, applied = build.build_split(cache, out_dir, "dev", seed, exclusions=_exclusions(exclusions),
                                          rules=rules, gz=gz)
        build.write_exclusions(out_dir / build.EXCLUSIONS_FILE, applied, build.source_names(cache))
    except (FileNotFoundError, ValueError) as e:
        raise OpError(str(e)) from None
    return {"claim_class": "derived", "sets": [_brief(m) for m in mans],
            "excluded_rows": sum(len(v) for v in applied.values()), "private_rules": len(rules)}


@operation("prompts_seal", ("prompts", "seal"), "Build the sealed prompt sets (disjoint from dev) into a "
           "directory outside any git work tree and write SHA256SUMS for the pre-registration.",
           [Arg("out", "string", "Output directory, outside every git work tree.", required=True),
            Arg("sources", "string", "Source cache directory."),
            Arg("fetch", "boolean", "Download missing source files first.", default=False),
            Arg("seed", "string", "Split/rank seed (must match the dev build).", default="inferencekit-prompts-v1"),
            Arg("secret_file", "string", "File holding the selection secret (default <out>/SEED.secret, created "
                "with 32 random bytes if missing). Keep it private until the sealed run is over."),
            Arg("public", "boolean", "Rank by the public seed only (anyone can rebuild the set; tests and demos).",
                default=False),
            _EXCL_ARG, _BLOCK_ARG])
def prompts_seal(call, out, sources, fetch, seed, secret_file, public, exclusions, blocklist):
    import secrets as _secrets

    from inferencekit.prompts import build

    out_dir = Path(out).expanduser()
    root = _git_root(out_dir)
    if root is not None:
        raise OpError(f"refusing to write sealed prompts inside the git work tree {root}: "
                      "a sealed set must never be committable")
    secret = secret_sha = None
    if not public:
        sf = Path(secret_file).expanduser() if secret_file else out_dir / "SEED.secret"
        if _git_root(sf) is not None:
            raise OpError(f"refusing to keep the sealing secret inside a git work tree ({sf})")
        if not sf.is_file():
            sf.parent.mkdir(parents=True, exist_ok=True)
            sf.write_text(_secrets.token_hex(32) + "\n", encoding="utf-8")
            sf.chmod(0o600)
        secret, secret_sha = _read_secret(sf)
    cache = _cache(sources, fetch)
    listed = _exclusions(exclusions)
    rules = _rules(blocklist)
    try:
        if rules:  # a public rebuild must exclude the same rows: they have to be in the exclusions file
            _, by_rules = build.candidates(cache, seed, None, rules)
            extra = sum(len(refs - listed.get(sha, set())) for sha, refs in by_rules.items())
            if extra:
                raise OpError(f"the private rules exclude {extra} row(s) the exclusions file does not list, so the "
                              "sealed set could not be rebuilt without them: run `prompts build` first (it rewrites "
                              "exclusions.json) or pass --exclusions")
        mans, _ = build.build_split(cache, out_dir, "sealed", seed, secret, secret_sha, exclusions=listed,
                                    rules=rules)
    except (FileNotFoundError, ValueError) as e:
        raise OpError(str(e)) from None
    sums = build.write_sums(out_dir, mans)
    return {"claim_class": "derived", "sets": [_brief(m) for m in mans], "sha256sums": str(sums),
            "secret_sha256": secret_sha, "secret_rule": build.SECRET_RULE if secret else None}


def _read_secret(sf: Path) -> tuple[str, str]:
    """(secret text used as the ranking salt, sha256 of the file bytes: the published commitment)."""
    import hashlib

    raw = sf.read_bytes()
    secret = raw.decode("utf-8").strip()
    if len(secret) < 16:
        raise OpError(f"{sf}: the secret is too short (use at least 16 characters)")
    return secret, hashlib.sha256(raw).hexdigest()


def _brief(m: dict) -> dict:
    return {k: m[k] for k in ("name", "n", "buckets", "sha256", "sealed")}


def _sums_lookup(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[1].lstrip("*")] = parts[0]
    return out


def check_items(items: list[dict]) -> list[str]:
    """Format and bucket-rule problems (empty = fine)."""
    problems, ids, texts = [], Counter(), Counter()
    for n, it in enumerate(items, 1):
        for key in ("id", "bucket", "messages", "max_tokens", "source", "kind"):
            if key not in it:
                problems.append(f"line {n}: missing '{key}'")
        if it.get("bucket") not in ps.BUCKETS:
            problems.append(f"line {n}: unknown bucket {it.get('bucket')!r}")
            continue
        msgs = it.get("messages")
        if not isinstance(msgs, list) or not msgs or not all(isinstance(m, dict) and m.get("role") and
                                                             isinstance(m.get("content"), str) for m in msgs):
            problems.append(f"line {n}: messages must be a non-empty OpenAI chat list")
            continue
        if not isinstance(it.get("max_tokens"), int) or it["max_tokens"] < 1:
            problems.append(f"line {n}: max_tokens must be a positive integer")
        text = ps.prompt_text(it)
        if not ps.in_bucket(text, it["bucket"]):
            problems.append(f"line {n} ({it.get('id')}): {ps.words(text)} words / {ps.proxy_tokens(text)} proxy "
                            f"tokens is outside the {it['bucket']} rule {ps.BUCKET_RULES[it['bucket']]}")
        ids[it.get("id")] += 1
        texts[text] += 1
    problems += [f"duplicate id {i!r} ({c}x)" for i, c in ids.items() if c > 1]
    dup_texts = sum(1 for c in texts.values() if c > 1)
    if dup_texts:
        problems.append(f"{dup_texts} prompt text(s) appear more than once")
    return problems


@operation("prompts_verify", ("prompts", "verify"), "Check a prompt set: sha256 against its manifest (and an "
           "expected hash or SHA256SUMS), the item format, bucket rules, unique ids and prompts.",
           [Arg("set", "string", "'dev', 'dev-load', or a path to a .jsonl / .jsonl.gz prompt set.", default="dev"),
            Arg("expected_sha256", "string", "Pre-registered sha256 the JSONL bytes must have."),
            Arg("sums", "string", "A SHA256SUMS file listing the set."),
            Arg("secret_file", "string", "Sealed sets: the revealed SEED.secret. Checks `sha256sum` of the file "
                "against the manifest's secret_sha256 and, when the sources are present, rebuilds the set."),
            Arg("sources", "string", "Source cache directory for the rebuild (default "
                f"{_DEFAULT_CACHE.replace(os.path.expanduser('~'), '~')})."),
            Arg("fetch", "boolean", "Download missing source files for the rebuild.", default=False),
            _EXCL_ARG])
def prompts_verify(call, set, expected_sha256, sums, secret_file, sources, fetch, exclusions):  # noqa: A002
    path = ps.resolve(set)
    if not path.is_file():
        raise OpError(f"no prompt set at {path}")
    try:
        actual = ps.sha256_file(path)
    except (OSError, ValueError) as e:  # e.g. a truncated or corrupt .jsonl.gz
        raise OpError(f"{path}: cannot read ({e})") from None
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    man_path = ps.manifest_path(path)
    man = None
    if man_path.is_file():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        check("manifest_sha256", man.get("sha256") == actual, f"manifest {man.get('sha256')} vs file {actual}")
    else:
        check("manifest_present", False, f"{man_path.name} not found")
    if expected_sha256:
        check("expected_sha256", expected_sha256.strip().lower() == actual, f"expected {expected_sha256}")
    if sums:
        table = _sums_lookup(Path(sums))
        key = path.name if path.name in table else ps.set_stem(path) + ".jsonl"
        listed = table.get(key)
        check("sha256sums", listed == actual, f"SHA256SUMS lists {listed} for {key}")
    try:
        items = [json.loads(line) for line in ps.read_text(path).splitlines() if line.strip()]
    except (ValueError, OSError) as e:
        raise OpError(f"{path}: not JSONL ({e})") from None
    problems = check_items(items)
    check("items", not problems, problems[:20] or f"{len(items)} items fit the format and bucket rules")
    if man is not None:
        counts = {b: sum(1 for it in items if it.get("bucket") == b) for b in ps.BUCKETS}
        counts = {b: n for b, n in counts.items() if n}
        check("manifest_counts", man.get("n") == len(items) and man.get("buckets") == counts,
              f"manifest n={man.get('n')} buckets={man.get('buckets')}; file n={len(items)} buckets={counts}")
    if secret_file:
        _verify_secret(man, Path(secret_file).expanduser(), actual, sources, fetch, exclusions, check)
    ok = all(c["ok"] for c in checks)
    return {"set": man.get("name") if man else ps.set_stem(path), "path": str(path), "sha256": actual,
            "verdict": "PASS" if ok else "FAIL", "checks": checks}


def _verify_secret(man: dict | None, sf: Path, actual: str, sources, fetch, exclusions, check) -> None:
    from inferencekit.prompts import build

    if not man or not man.get("secret_sha256"):
        check("secret_sha256", False, "the manifest commits to no secret (not a secret-sealed set)")
        return
    if not sf.is_file():
        raise OpError(f"no secret file at {sf}")
    secret, got = _read_secret(sf)
    check("secret_sha256", got == man["secret_sha256"],
          f"sha256sum of {sf.name} = {got}; manifest secret_sha256 = {man['secret_sha256']}")
    if got != man["secret_sha256"]:
        return
    cache = _cache(sources, fetch)
    try:
        rebuilt = build.rebuild_sha256(cache, man.get("seed") or build.SEED, secret, _exclusions(exclusions))
    except FileNotFoundError as e:
        check("rebuild", False, f"cannot rebuild: {e} (pass --sources DIR or --fetch)")
        return
    except ValueError as e:
        check("rebuild", False, f"cannot rebuild: {e}")
        return
    want = rebuilt.get(man.get("role") or "single")
    check("rebuild", want == actual, f"rebuilt from the sources with this secret: {want}; file: {actual}")


@operation("prompts_show", ("prompts", "show"), "Per-bucket statistics of a prompt set (counts, word and "
           "proxy-token lengths, sources).",
           [Arg("set", "string", "'dev', 'dev-load', or a path to a .jsonl / .jsonl.gz prompt set.", default="dev"),
            Arg("examples", "integer", "Show the first N prompts of each bucket (truncated).", default=0)])
def prompts_show(call, set, examples):  # noqa: A002
    path = ps.resolve(set)
    if not path.is_file():
        raise OpError(f"no prompt set at {path}")
    try:
        items = ps.load(str(path))
    except (OSError, ValueError) as e:  # corrupt .jsonl.gz, not JSONL
        raise OpError(f"{path}: cannot read ({e})") from None
    try:
        man = ps.load_manifest(str(path))
    except (OSError, ValueError):
        man = {}
    buckets = {}
    for b in ps.BUCKETS:
        sel = [it for it in items if it.get("bucket") == b]
        if not sel:
            continue
        w = [ps.words(ps.prompt_text(it)) for it in sel]
        t = [ps.proxy_tokens(ps.prompt_text(it)) for it in sel]
        row = {"n": len(sel), "words": {"min": min(w), "median": statistics.median(w), "max": max(w)},
               "proxy_tokens": {"min": min(t), "median": statistics.median(t), "max": max(t)},
               "max_tokens": sorted({it.get("max_tokens") for it in sel}),
               "sources": dict(Counter(it.get("source", "?").split("/")[0].split("-")[0]
                                       if it.get("source", "").startswith("gutenberg") else it.get("source", "?")
                                       for it in sel))}
        if examples:
            row["examples"] = [{"id": it["id"], "text": ps.prompt_text(it)[:200]} for it in sel[:examples]]
        buckets[b] = row
    return {"set": man.get("name") or ps.set_stem(path), "sha256": ps.sha256_file(path), "sealed": man.get("sealed"),
            "n": len(items), "claim_class": "derived", "length_proxy": man.get("length_proxy",
                                                                                "proxy_tokens = words * 1.3"),
            "buckets": buckets}
