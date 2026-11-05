"""Build the prompt sets from openly licensed sources, deterministically.

Sources (downloaded once into a cache directory; their sha256 is recorded in every
manifest, so a rebuild from different bytes is visible):

* databricks-dolly-15k (CC BY-SA 3.0): instructions without context feed very_short
  and short; context + instruction pairs feed medium.
* GSM8K train split (MIT): grade-school maths questions feed short.
* Public-domain novels (Project Gutenberg plain-text editions, licence boilerplate
  stripped): passages of ~1,700-2,600 words plus a question feed long.

Groups and the dev / sealed split: every candidate belongs to a *group*, the unit that is
split and deduplicated. A dolly row with a context is grouped by the sha256 of its
normalised context paragraph (many rows ask different questions about one paragraph), a
row without context by its normalised instruction, a GSM8K question by its normalised
text, and a novel passage by its book. Each group yields at most one candidate (novels:
one per passage), and a group is assigned to ``dev`` or ``sealed`` by one bit of
``sha256(seed | split | group)``, so rows that share a context paragraph always land on
the same side, and no two prompts anywhere share a context (no shared prefix for a
prefix cache to reuse). As a second guard, a sealed candidate is dropped when more than
half of its 12-word shingles appear in the dev sets, and any candidate is dropped when
more than half of its shingles appear in an item already chosen for the same split.
Inside a split, groups are ranked by another seeded hash; the single-user set takes the
first 16 per bucket and the load set the next ones, so the four sets are disjoint.

The code and the seed are public, so a sealed set built from the public seed alone could
be rebuilt by anyone, including an agent being evaluated. ``prompts seal`` therefore
takes a secret: the sealed sets are drawn from every group the dev sets do not use and
ranked by ``sha256(secret | ...)``. The manifests commit to the secret by
``secret_sha256`` = the sha256 of the secret file's bytes as ``prompts seal`` writes them
(64 hex characters and a newline), i.e. ``sha256sum SEED.secret``. Revealing the file
after the evaluation lets anyone check that hash, rebuild the sealed set
(``prompts verify --secret-file``) and compare it with the pre-registered sha256.

Exclusions: text the release leak scanner would flag (names that must never appear in a
tracked file) is excluded. The scanner's rules are private and never in the repository;
they are read at build time only from a file given explicitly, with ``--blocklist FILE`` or
``$INFERENCEKIT_PROMPT_BLOCKLIST`` (scanner syntax). The source rows they
exclude are recorded, by position only, in ``exclusions.json`` next to the shipped sets
(keyed by the source file's sha256), so a public rebuild without the private rules
excludes the same rows and gives byte-identical sets.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path

from inferencekit.prompts import BUCKET_RULES, BUCKETS, PROXY_FACTOR, gzip_bytes, in_bucket, proxy_tokens, words

SEED = "inferencekit-prompts-v1"
VERSION = "1.0"
MAX_TOKENS = 256
PER_BUCKET = 16
LOAD_TARGET = 1280  # >= 1,200 unique short/medium prompts per load set
LOAD_MEDIUM_MAX = 280

DOLLY = {"name": "databricks-dolly-15k", "license": "CC-BY-SA-3.0", "file": "dolly.jsonl",
         "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/"
                "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/databricks-dolly-15k.jsonl"}
GSM8K = {"name": "gsm8k-train", "license": "MIT", "file": "gsm8k_train.jsonl",
         "url": "https://raw.githubusercontent.com/openai/grade-school-math/"
                "3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl"}
# Public-domain novels (US). Gutenberg ebook numbers -> title, for attribution.
BOOKS = {
    11: "Alice's Adventures in Wonderland (Lewis Carroll)", 35: "The Time Machine (H. G. Wells)",
    36: "The War of the Worlds (H. G. Wells)", 43: "The Strange Case of Dr Jekyll and Mr Hyde (R. L. Stevenson)",
    76: "Adventures of Huckleberry Finn (Mark Twain)", 84: "Frankenstein (Mary Shelley)",
    98: "A Tale of Two Cities (Charles Dickens)", 120: "Treasure Island (R. L. Stevenson)",
    158: "Emma (Jane Austen)", 161: "Sense and Sensibility (Jane Austen)",
    174: "The Picture of Dorian Gray (Oscar Wilde)", 215: "The Call of the Wild (Jack London)",
    219: "Heart of Darkness (Joseph Conrad)", 345: "Dracula (Bram Stoker)",
    768: "Wuthering Heights (Emily Bronte)", 1260: "Jane Eyre (Charlotte Bronte)",
    1342: "Pride and Prejudice (Jane Austen)", 1400: "Great Expectations (Charles Dickens)",
    1661: "The Adventures of Sherlock Holmes (Arthur Conan Doyle)",
    2554: "Crime and Punishment (Fyodor Dostoevsky, tr. Constance Garnett)",
    2701: "Moby Dick (Herman Melville)",
}
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{n}/pg{n}.txt"
GUTENBERG_LICENSE = "public-domain"
LONG_QUESTIONS = (
    "Summarize the passage above in about 150 words.",
    "List the characters who appear in the passage above and describe what each of them does.",
    "What is the central conflict or tension in the passage above? Support your answer with two details from the text.",
    "Describe the setting of the passage above and how it shapes the mood.",
    "Write five questions a teacher could ask about the passage above, each with a one-sentence answer.",
    "Explain what happens in the passage above, step by step, in the order the events occur.",
)
PASSAGE_FRACTIONS = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85)  # >= 3,000 words apart in every book used
EXCLUSIONS_FILE = "exclusions.json"
SHINGLE = 12  # words per shingle for the near-duplicate guard
MAX_SHARED = 0.5  # largest share of a candidate's shingles allowed to appear in the other split / same set


# --------------------------------------------------------------------------- sources

def source_files(cache: Path) -> dict[str, Path]:
    files = {"dolly": cache / DOLLY["file"], "gsm8k": cache / GSM8K["file"]}
    for n in BOOKS:
        files[f"gutenberg:{n}"] = cache / "gutenberg" / f"pg{n}.txt"
    return files


def fetch(cache: Path, log=print) -> dict[str, Path]:
    """Download any missing source file into ``cache``. Returns the source map."""
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "gutenberg").mkdir(exist_ok=True)
    urls = {"dolly": DOLLY["url"], "gsm8k": GSM8K["url"]}
    urls.update({f"gutenberg:{n}": GUTENBERG_URL.format(n=n) for n in BOOKS})
    files = source_files(cache)
    for key, path in files.items():
        if path.is_file() and path.stat().st_size > 0:
            continue
        log(f"downloading {urls[key]}")
        req = urllib.request.Request(urls[key], headers={"User-Agent": "inferencekit-prompt-builder"})
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 - fixed https URLs above
            data = r.read()
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)
    return files


def source_table(cache: Path) -> list[dict]:
    files = source_files(cache)
    rows = [{"name": DOLLY["name"], "license": DOLLY["license"], "url": DOLLY["url"],
             "sha256": sha256_file(files["dolly"])},
            {"name": GSM8K["name"], "license": GSM8K["license"], "url": GSM8K["url"],
             "sha256": sha256_file(files["gsm8k"])}]
    for n, title in BOOKS.items():
        rows.append({"name": f"gutenberg-{n}", "title": title, "license": GUTENBERG_LICENSE,
                     "url": GUTENBERG_URL.format(n=n), "sha256": sha256_file(files[f"gutenberg:{n}"])})
    return rows


# --------------------------------------------------------------------------- hashing

def _h(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def split_of(key: str, seed: str = SEED) -> str:
    """'dev' or 'sealed' from one bit of a seeded hash of the source item's key."""
    return "dev" if int(_h(seed, "split", key)[:2], 16) % 2 == 0 else "sealed"


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def blocklist_path(explicit: str | None = None) -> Path | None:
    """The private scrub-patterns file, only when one is named: ``explicit`` (the ``--blocklist``
    flag), else ``$INFERENCEKIT_PROMPT_BLOCKLIST``. Nothing is read by default; ``none`` or an
    empty value means no file. A named file that does not exist is an error."""
    value = explicit if explicit is not None else os.environ.get("INFERENCEKIT_PROMPT_BLOCKLIST")
    if value is None or not value.strip() or value.strip().lower() == "none":
        return None
    p = Path(value.strip()).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"blocklist file not found: {p}")
    return p


def load_rules(path: Path | None) -> list[re.Pattern]:
    """Byte regexes from a scrub-patterns file (the release scanner's syntax: ``literal:TEXT``,
    ``regex:PATTERN``, ``TEXT``, each optionally ``==>REPLACEMENT``; ``#scan RULE`` is a rule too;
    other ``#`` lines are comments or directives)."""
    if path is None:
        return []
    rules = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        raw = line.rstrip("\r\n")
        if not raw.strip() or raw.startswith("#@"):
            continue
        if raw.startswith("#scan "):
            raw = raw[len("#scan "):]
        elif raw.startswith("#"):
            continue
        match = raw.rsplit("==>", 1)[0] if "==>" in raw else raw
        if match.startswith("regex:"):
            rules.append(re.compile(match[len("regex:"):].encode("utf-8")))
        else:
            text = match[len("literal:"):] if match.startswith("literal:") else match
            if text:
                rules.append(re.compile(re.escape(text.encode("utf-8"))))
    return rules


def blocked(text: str, rules: list[re.Pattern]) -> bool:
    """True if the scanner would flag the text (as written, or as a JSONL string)."""
    if not rules:
        return False
    forms = (text.encode("utf-8"), json.dumps(text, ensure_ascii=False).encode("utf-8"))
    return any(r.search(f) for r in rules for f in forms)


def read_exclusions(path: Path) -> dict[str, set[str]]:
    """{source sha256: {row refs}} from an exclusions file (empty if there is none)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {sha: set(refs) for sha, refs in (data.get("excluded") or {}).items()}


def write_exclusions(path: Path, excluded: dict[str, set[str]], names: dict[str, str]) -> None:
    data = {"format": 1,
            "rule": "source rows the release leak scanner's private rules would flag; listed by position in the "
                    "source file (keyed by that file's sha256) so a rebuild without the private rules matches",
            "sources": {sha: names.get(sha, "?") for sha in sorted(excluded)},
            "excluded": {sha: sorted(refs, key=_ref_order) for sha, refs in sorted(excluded.items()) if refs}}
    Path(path).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def _ref_order(ref: str):
    return tuple(int(x) if x.isdigit() else x for x in re.split(r"[:]", ref.replace("row", "")))


def exclusions_digest(excluded: dict[str, set[str]]) -> str:
    """sha256 over the sorted (source sha256, ref) pairs actually applied: recorded in each manifest."""
    lines = sorted(f"{sha} {ref}" for sha, refs in excluded.items() for ref in refs)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def shingles(text: str, n: int = SHINGLE) -> set[str]:
    w = text.lower().split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def shared_fraction(sh: set[str], seen: set[str]) -> float:
    return len(sh & seen) / len(sh) if sh else 0.0


# --------------------------------------------------------------------------- candidates

def _dolly_candidates(path: Path) -> list[dict]:
    """One candidate per group: a context paragraph (rows asking about the same paragraph are one
    group) or, without context, a normalised instruction. The first row in file order that fits a
    bucket represents its group."""
    out, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        o = json.loads(line)
        ins, ctx = " ".join(o["instruction"].split()), o.get("context", "").strip()
        if not ins:
            continue
        if ctx:
            ctx = "\n\n".join(" ".join(p.split()) for p in re.split(r"\n\s*\n", ctx) if p.strip())
            text, bucket = f"{ctx}\n\n{ins}", "medium"
            key = "dolly-context:" + hashlib.sha256(_norm(ctx).encode("utf-8")).hexdigest()
            if not in_bucket(text, "medium"):
                continue
        else:
            text = ins
            key = "dolly:" + _norm(text)
            if in_bucket(text, "very_short") and text.endswith("?"):
                bucket = "very_short"
            elif in_bucket(text, "short"):
                bucket = "short"
            else:
                continue
        if key in seen:
            continue
        seen.add(key)
        out.append({"bucket": bucket, "text": text, "key": key, "ref": f"dolly:row{n + 1}", "src": "dolly",
                    "source": f"{DOLLY['name']}/{DOLLY['license']}", "source_ref": f"{DOLLY['name']}:line{n + 1}"})
    return out


def _gsm8k_candidates(path: Path) -> list[dict]:
    out, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        q = " ".join(json.loads(line)["question"].split())
        key = "gsm8k:" + _norm(q)
        if key in seen or not in_bucket(q, "short"):
            continue
        seen.add(key)
        out.append({"bucket": "short", "text": q, "key": key, "ref": f"gsm8k:row{n + 1}", "src": "gsm8k",
                    "source": f"{GSM8K['name']}/{GSM8K['license']}", "source_ref": f"{GSM8K['name']}:line{n + 1}"})
    return out


def gutenberg_paragraphs(raw: str) -> list[str]:
    """Body paragraphs with the distributor's licence header and footer removed."""
    text = raw.replace("\r\n", "\n").lstrip("﻿")
    start = re.search(r"^\*\*\* ?START OF.*$", text, re.M)
    end = re.search(r"^\*\*\* ?END OF.*$", text, re.M)
    body = text[start.end() if start else 0: end.start() if end else len(text)]
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", body) if p.strip()]
    return [p for p in paras if "gutenberg" not in p.lower()]


def _long_candidates(files: dict[str, Path], seed: str) -> list[dict]:
    out = []
    for n in BOOKS:
        paras = gutenberg_paragraphs(files[f"gutenberg:{n}"].read_text(encoding="utf-8", errors="replace"))
        for k, frac in enumerate(PASSAGE_FRACTIONS):
            target = 1700 + int(_h(seed, "len", n, k)[:4], 16) % 900  # 1,700-2,599 words
            i = int(len(paras) * frac)
            chunk, count = [], 0
            while i < len(paras) and count < target:
                chunk.append(paras[i])
                count += words(paras[i])
                i += 1
            question = LONG_QUESTIONS[int(_h(seed, "q", n, k)[:4], 16) % len(LONG_QUESTIONS)]
            passage = "\n\n".join(chunk)
            text = ("Read the following passage from a public-domain novel and answer the question after it.\n\n"
                    f"{passage}\n\nQuestion: {question}")
            if not in_bucket(text, "long"):
                continue
            out.append({"bucket": "long", "text": text, "key": f"gutenberg:{n}", "rank_key": f"gutenberg:{n}:{k}",
                        "ref": f"gutenberg:{n}:passage{k}", "src": f"gutenberg:{n}",
                        "order": k, "source": f"gutenberg-{n}/{GUTENBERG_LICENSE}",
                        "source_ref": f"{BOOKS[n]}, passage at {int(frac * 100)}% of the book"})
    return out


def candidates(cache: Path, seed: str = SEED, exclusions: dict[str, set[str]] | None = None,
               rules: list[re.Pattern] | None = None) -> tuple[list[dict], dict[str, set[str]]]:
    """(candidates, exclusions applied). A row is excluded when an exclusions file lists it for this
    exact source file (by sha256) or a private scanner rule matches its text."""
    files = source_files(cache)
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source files (run with fetch): {missing[:3]}")
    shas = {k: sha256_file(p) for k, p in files.items()}
    cands = _dolly_candidates(files["dolly"]) + _gsm8k_candidates(files["gsm8k"]) + _long_candidates(files, seed)
    listed = exclusions or {}
    applied: dict[str, set[str]] = {}
    kept = []
    for c in cands:
        sha = shas[c["src"]]
        if c["ref"] in listed.get(sha, ()) or blocked(c["text"], rules or []):
            applied.setdefault(sha, set()).add(c["ref"])
            continue
        c["split"] = split_of(c["key"], seed)
        kept.append(c)
    return kept, applied


def source_names(cache: Path) -> dict[str, str]:
    """{sha256: source name} for the exclusions file."""
    files = source_files(cache)
    names = {"dolly": DOLLY["name"], "gsm8k": GSM8K["name"]}
    return {sha256_file(p): names.get(k, k.replace(":", "-")) for k, p in files.items()}


# --------------------------------------------------------------------------- selection

def _item(c: dict, set_tag: str, i: int) -> dict:
    return {"id": f"{c['bucket']}-{set_tag}{i:04d}", "bucket": c["bucket"],
            "messages": [{"role": "user", "content": c["text"]}], "max_tokens": MAX_TOKENS,
            "source": c["source"], "kind": "chat", "source_ref": c["source_ref"],
            "proxy_tokens": proxy_tokens(c["text"])}


def _pick(cands: list[dict], split: str, seed: str, secret: str | None) -> tuple[dict, dict]:
    """({bucket: single candidates}, {bucket: load candidates}) for one split."""
    pools: dict[str, list[dict]] = {b: [] for b in BUCKETS}
    avoid: set[str] = set()  # shingles of every dev prompt (sealed only)
    if split == "sealed":
        # Never a group the dev sets use, and never text that mostly repeats a dev prompt.
        dev_single, dev_load = _pick(cands, "dev", seed, None)
        dev = [c for group in (dev_single, dev_load) for cs in group.values() for c in cs]
        used = {c["key"] for c in dev}
        for c in dev:
            avoid |= shingles(c["text"])
        for c in cands:
            # With a secret: every group the dev sets do not use (long passages: every book they do
            # not use), ranked by the secret, so nobody can predict the sealed set without it and
            # anyone can rebuild it once it is revealed. Without one: the public sealed half.
            if c["key"] not in used and (secret or c["split"] == "sealed"):
                pools[c["bucket"]].append(c)
        salt = secret or seed
    else:
        for c in cands:
            if c["split"] == split:
                pools[c["bucket"]].append(c)
        salt = seed
    seen: set[str] = set()
    for b in BUCKETS:
        # Long passages: one per book first (order k), then the next passage of each book.
        pool = sorted(pools[b], key=lambda c: (c.get("order", 0), _h(salt, "rank", split, b,
                                                                      c.get("rank_key", c["key"]))))
        kept = []
        for c in pool:  # near-duplicate guard, in rank order
            sh = shingles(c["text"])
            if shared_fraction(sh, avoid) > MAX_SHARED or shared_fraction(sh, seen) > MAX_SHARED:
                continue
            seen |= sh
            kept.append(c)
        pools[b] = kept
    single, rest = {}, {}
    for b in BUCKETS:
        if len(pools[b]) < PER_BUCKET:
            raise ValueError(f"{split}: only {len(pools[b])} {b} candidates (need {PER_BUCKET})")
        single[b] = pools[b][:PER_BUCKET]
        rest[b] = pools[b][PER_BUCKET:]
    n_med = min(LOAD_MEDIUM_MAX, len(rest["medium"]))
    n_short = LOAD_TARGET - n_med
    if len(rest["short"]) < n_short:
        raise ValueError(f"{split}: only {len(rest['short'])} short load candidates (need {n_short})")
    return single, {"short": rest["short"][:n_short], "medium": rest["medium"][:n_med]}


def select(cands: list[dict], split: str, seed: str = SEED, secret: str | None = None) -> dict[str, list[dict]]:
    """{'single': [...], 'load': [...]} for one split. Disjoint by construction.

    ``dev``: the dev half of the public hash split, ranked by the public seed.
    ``sealed`` without a secret: the other half, ranked by the public seed (reproducible by
    anyone, so only "sealed" against accidents). ``sealed`` with a secret: see ``_pick``.
    """
    tags = {"dev": ("", "load-"), "sealed": ("s", "sload-")}[split]
    single, load = _pick(cands, split, seed, secret)
    out_single = [_item(c, tags[0], i + 1) for b in BUCKETS for i, c in enumerate(single[b])]
    out_load = [_item(c, tags[1], i + 1) for b in ("short", "medium") for i, c in enumerate(load[b])]
    return {"single": out_single, "load": out_load}



def sha256_file(path: Path) -> str:
    """sha256 of a file's raw bytes (source files, manifests)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- writing

def jsonl_bytes(items: list[dict]) -> bytes:
    return "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items).encode("utf-8")


SECRET_RULE = ("secret_sha256 = sha256 of the secret file's bytes as prompts seal writes them (64 hex characters and "
               "a newline): `sha256sum SEED.secret`, or `echo <secret> | sha256sum`")


def manifest(name: str, items: list[dict], data: bytes, sealed: bool, role: str, sources: list[dict],
             seed: str = SEED, secret: str | None = None, secret_sha256: str | None = None,
             exclusions: dict | None = None) -> dict:
    counts = {b: sum(1 for it in items if it["bucket"] == b) for b in BUCKETS}
    return {"name": name, "version": VERSION, "n": len(items), "buckets": {b: n for b, n in counts.items() if n},
            "sha256": hashlib.sha256(data).hexdigest(), "sealed": sealed, "role": role, "seed": seed,
            "max_tokens": MAX_TOKENS,
            "length_proxy": f"proxy_tokens = whitespace words * {PROXY_FACTOR}", "bucket_rules": BUCKET_RULES,
            "split_rule": "sha256(seed|split|group): even first byte -> dev, odd -> sealed; group = sha256 of the "
                          "normalised dolly context (else the normalised instruction / question; long: the book); "
                          "one candidate per group; near-duplicates (>50% shared 12-word shingles with the dev sets "
                          "or the same split) dropped",
            "selection": ("secret-seeded: groups not used by dev/dev-load (long: books not used), ranked by "
                          "sha256(secret|rank|...); the secret is committed by secret_sha256 and revealed after use"
                          if secret else "ranked by sha256(seed|rank|split|bucket|key) within the split"),
            "secret_sha256": secret_sha256 if secret else None,
            "secret_rule": SECRET_RULE if secret else None,
            "exclusions": {"rows": sum(len(v) for v in (exclusions or {}).values()),
                           "sha256": exclusions_digest(exclusions or {}),
                           "rule": "rows listed in exclusions.json or flagged by the private leak-scanner rules"},
            "sources": sources, "builder": "scripts/build_prompts.py (inferencekit prompts build / seal)"}


def write_set(out_dir: Path, name: str, items: list[dict], sealed: bool, role: str, sources: list[dict],
              seed: str = SEED, secret: str | None = None, secret_sha256: str | None = None,
              exclusions: dict | None = None, gz: bool = False) -> dict:
    """Write ``<name>.jsonl`` (``gz``: ``<name>.jsonl.gz``, removing a stale twin) and its manifest.
    The manifest's sha256 is of the uncompressed JSONL bytes either way."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data = jsonl_bytes(items)
    plain, packed = out_dir / f"{name}.jsonl", out_dir / f"{name}.jsonl.gz"
    if gz:
        packed.write_bytes(gzip_bytes(data))
        plain.unlink(missing_ok=True)
    else:
        plain.write_bytes(data)
        packed.unlink(missing_ok=True)
    man = manifest(name, items, data, sealed, role, sources, seed, secret, secret_sha256, exclusions)
    (out_dir / f"{name}.manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n",
                                                   encoding="utf-8")
    return man


def build_split(cache: Path, out_dir: Path, split: str, seed: str = SEED, secret: str | None = None,
                secret_sha256: str | None = None, exclusions: dict[str, set[str]] | None = None,
                rules: list[re.Pattern] | None = None, gz: bool = False) -> tuple[list[dict], dict[str, set[str]]]:
    """Write ``<split>.jsonl`` and ``<split>-load.jsonl`` (``gz``: ``.jsonl.gz``) (+ manifests) into ``out_dir``.

    Returns (manifests, exclusions applied)."""
    cands, applied = candidates(cache, seed, exclusions, rules)
    sets = select(cands, split, seed, secret)
    sources = source_table(cache)
    sealed = split == "sealed"
    mans = [write_set(out_dir, split, sets["single"], sealed, "single", sources, seed, secret, secret_sha256, applied,
                      gz=gz),
            write_set(out_dir, f"{split}-load", sets["load"], sealed, "load", sources, seed, secret, secret_sha256,
                      applied, gz=gz)]
    return mans, applied


def rebuild_sha256(cache: Path, seed: str, secret: str | None, exclusions: dict[str, set[str]] | None,
                   rules: list[re.Pattern] | None = None) -> dict[str, str]:
    """{'single': sha256, 'load': sha256} of the sealed sets the given secret selects (nothing written)."""
    cands, _ = candidates(cache, seed, exclusions, rules)
    sets = select(cands, "sealed", seed, secret)
    return {role: hashlib.sha256(jsonl_bytes(items)).hexdigest() for role, items in sets.items()}


def write_sums(out_dir: Path, manifests: list[dict]) -> Path:
    path = out_dir / "SHA256SUMS"
    lines = [f"{m['sha256']}  {m['name']}.jsonl\n" for m in manifests]
    for m in manifests:
        mp = out_dir / f"{m['name']}.manifest.json"
        lines.append(f"{sha256_file(mp)}  {mp.name}\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path
