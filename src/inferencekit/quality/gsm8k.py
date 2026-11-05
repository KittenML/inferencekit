"""GSM8K: the pinned test split (downloaded once, cached), the zero-shot chain-of-thought prompt,
and answer extraction.

Data: the 1,319-problem test split of GSM8K (Cobbe et al. 2021,
github.com/openai/grade-school-math, MIT licence in ``data/gsm8k_LICENSE.txt``). It is not
shipped: the first ``quality gsm8k`` run downloads the Hugging Face dataset ``openai/gsm8k``
(config ``main``, split ``test``) at a pinned revision, checks the Parquet file's sha256,
converts it to JSONL (one ``{"question", "answer"}`` object per line, ``json.dumps``
defaults) and checks that file's sha256 too. Those JSONL bytes are identical to the
upstream ``test.jsonl`` and to the file early versions shipped, so ``dataset_sha256`` in evidence
records is unchanged. The cache is ``~/.cache/inferencekit/gsm8k/`` (``$XDG_CACHE_HOME``
and ``INFERENCEKIT_CACHE_DIR`` move it). The gold answer is the number after the last
``####`` in ``answer``.

Prompt (zero-shot, one user turn): the question, then an instruction to reason step by
step and finish with ``#### <number>``.

Extraction (first rule that finds a number wins; the rule used is recorded per item):
  1. ``hash``:  the first number after the last ``####``;
  2. ``boxed``: the number inside the last ``\\boxed{...}`` (many chat models use it);
  3. ``last``:  the last number anywhere in the response.
Numbers are normalised before comparison: thousands separators, ``$`` and a trailing
``.`` are dropped; ``-1,234.50`` becomes Decimal("-1234.5"). A response with no number
is wrong. Correct means numerically equal to the gold answer.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path

from inferencekit.canonical import sha256_hex
from inferencekit.registry import OpError

DATASET = "openai/gsm8k (main, test); github.com/openai/grade-school-math; MIT"
# Pinned source: the Hugging Face dataset repo at one commit, and the hashes of what it holds.
HF_REPO = "openai/gsm8k"
HF_REVISION = "740312add88f781978c0658806c59bc2815b9866"
HF_FILE = "main/test-00000-of-00001.parquet"
PARQUET_SHA256 = "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"  # = its LFS oid
JSONL_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
N_TEST = 1319
INSTRUCTION = ("Solve the problem step by step. At the end, write the final answer as a single number "
               "on its own line in the form \"#### <number>\".")

_NUM = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?|-?\.\d+")
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def cache_dir() -> Path:
    base = os.environ.get("INFERENCEKIT_CACHE_DIR") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache"), "inferencekit")
    return Path(base) / "gsm8k"


def source_url() -> str:
    endpoint = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    return f"{endpoint}/datasets/{HF_REPO}/resolve/{HF_REVISION}/{HF_FILE}"


def _parquet_path() -> Path:
    return cache_dir() / Path(HF_FILE).name


def manual_download_command() -> str:
    p = _parquet_path()
    return f"mkdir -p '{p.parent}' && curl -fL -o '{p}' '{source_url()}'"


def parquet_to_jsonl(parquet: str | Path | bytes) -> bytes:
    """The GSM8K Parquet table as JSONL bytes: {"question", "answer"} per line, json.dumps defaults."""
    from inferencekit.quality.parquet import read_table

    table = read_table(parquet)
    if "question" not in table or "answer" not in table:
        raise OpError(f"GSM8K Parquet file without question/answer columns (has {sorted(table)})")
    rows = []
    for q, a in zip(table["question"], table["answer"]):
        if q is None or a is None:
            raise OpError("GSM8K Parquet file with an empty question or answer")
        rows.append(json.dumps({"question": q.decode("utf-8"), "answer": a.decode("utf-8")}) + "\n")
    return "".join(rows).encode("utf-8")


def _download(dest: Path, timeout: float = 120.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".part-{os.getpid()}")
    req = urllib.request.Request(source_url(), headers={"User-Agent": "inferencekit (quality gsm8k)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as f:  # noqa: S310
            while chunk := resp.read(1 << 16):
                f.write(chunk)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def data_path(fetch: bool = True) -> Path:
    """The cached GSM8K test split as JSONL (sha256-checked), downloading it once if needed.

    Order: the cached JSONL; else the cached Parquet file (downloaded here, or placed by hand
    with ``manual_download_command()``); else a download from Hugging Face at HF_REVISION.
    ``fetch=False`` or ``HF_HUB_OFFLINE=1`` never touches the network."""
    jsonl = cache_dir() / f"gsm8k-test-{HF_REVISION[:12]}.jsonl"
    if jsonl.is_file() and _sha(jsonl.read_bytes()) == JSONL_SHA256:
        return jsonl
    pq = _parquet_path()
    if not (pq.is_file() and _sha(pq.read_bytes()) == PARQUET_SHA256):
        offline = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")
        bad_cache = pq.is_file()
        reason = ("HF_HUB_OFFLINE is set" if offline else "fetching is off" if not fetch else "")
        if bad_cache:
            reason = f"the cached {pq.name} has the wrong sha256" + (f" and {reason}" if reason else "")
        if fetch and not offline:  # (re)download; a cached file with the wrong hash is replaced
            reason = ""
            try:
                _download(pq)
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                reason = f"download failed: {getattr(e, 'reason', None) or e}"
        if not reason and _sha(pq.read_bytes()) != PARQUET_SHA256:
            reason = f"downloaded file sha256 {_sha(pq.read_bytes())} is not the pinned {PARQUET_SHA256}"
            pq.unlink(missing_ok=True)
        if reason:
            raise OpError(
                f"GSM8K test split is not available ({reason}). It is downloaded once from the Hugging Face "
                f"dataset {HF_REPO} (config main, split test) at revision {HF_REVISION} and cached in "
                f"{cache_dir()}. Fetch it by hand (on another machine if this one is offline, then copy it to "
                f"{_parquet_path()} here):\n  {manual_download_command()}\nexpected sha256 {PARQUET_SHA256}. "
                "Or pass --data with that Parquet file or any GSM8K-format JSONL.")
    data = parquet_to_jsonl(pq)
    if _sha(data) != JSONL_SHA256:
        raise OpError(f"GSM8K: the JSONL converted from {pq} has sha256 {_sha(data)}, not the pinned "
                      f"{JSONL_SHA256} (Parquet reader problem; please report it)")
    tmp = jsonl.with_name(jsonl.name + f".part-{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, jsonl)
    return jsonl


def load(path: str | Path | None = None, limit: int | None = None) -> tuple[list[dict], str]:
    """[{idx, question, gold}] and the sha256 of the JSONL bytes.

    ``path``: a GSM8K-format JSONL file, or a Parquet file with question/answer columns
    (converted as for the pinned split). None: the pinned test split (downloaded once)."""
    p = Path(path) if path else data_path()
    if not p.is_file():
        raise OpError(f"GSM8K data file not found: {p}")
    try:
        raw = parquet_to_jsonl(p) if p.suffix == ".parquet" else p.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, ValueError) as e:  # ParquetError, not UTF-8
        raise OpError(f"{p}: {e}") from None
    items = []
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            gold = obj.get("gold")
            if gold is None:
                gold = obj["answer"].split("####")[-1].strip()
            items.append({"idx": obj.get("idx", i), "question": obj["question"], "gold": str(gold)})
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise OpError(f"{p}:{i + 1}: not a GSM8K row ({type(e).__name__}: {e}); need "
                          "{\"question\", \"answer\" ending in '#### <n>'} or {\"question\", \"gold\"}") from None
    if limit is not None and limit >= 0:
        items = items[:limit]
    return items, sha256_hex(raw)


def messages(question: str) -> list[dict]:
    return [{"role": "user", "content": f"{question}\n\n{INSTRUCTION}"}]


def normalize(text: str | None) -> Decimal | None:
    if text is None:
        return None
    s = text.replace(",", "").replace("$", "").replace(" ", "").strip().rstrip(".")
    try:
        return Decimal(s).normalize() if s else None
    except InvalidOperation:
        return None


def _first_number(text: str) -> str | None:
    m = _NUM.search(text)
    return m.group(0) if m else None


def _last_number(text: str) -> str | None:
    found = _NUM.findall(text)
    return found[-1] if found else None


def extract(response: str | None) -> tuple[str | None, str]:
    """(number string, rule) from a model response; (None, "none") when no number is found."""
    if not response:
        return None, "none"
    if "####" in response:
        n = _first_number(response.rsplit("####", 1)[1])
        if n is not None:
            return n, "hash"
    boxed = _BOXED.findall(response)
    if boxed:
        n = _last_number(boxed[-1])
        if n is not None:
            return n, "boxed"
    n = _last_number(response)
    return (n, "last") if n is not None else (None, "none")


def is_correct(extracted: str | None, gold: str) -> bool:
    a, b = normalize(extracted), normalize(gold)
    return a is not None and b is not None and a == b
