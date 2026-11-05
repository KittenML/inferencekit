"""Read a prompt set: JSONL, one request per line (the format ``prompts build`` writes).

    {"id": "short-0007", "bucket": "short", "messages": [{"role": "user", "content": "..."}],
     "max_tokens": 256, "source": "dataset/license", "kind": "chat"}

This module only reads. Building and sealing sets is ``inferencekit prompts ...``. The file may
be gzipped (``.jsonl.gz``, as the shipped dev sets are). The set's identity is the sha256 of the
uncompressed JSONL bytes, the same hash its manifest carries, so a benchmark record can be
matched to the exact prompts it ran.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from inferencekit import prompts as _sets
from inferencekit.registry import OpError

BUCKETS = ("very_short", "short", "medium", "long")


@dataclass(frozen=True)
class Prompt:
    id: str
    bucket: str
    messages: tuple  # OpenAI chat messages (dicts); a tuple so the dataclass stays hashable
    max_tokens: int | None
    kind: str = "chat"
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    def as_completion_prompt(self) -> str:
        """Plain text for /v1/completions: the message contents joined (no chat template)."""
        return "\n\n".join(str(m.get("content", "")) for m in self.messages)


@dataclass(frozen=True)
class PromptSet:
    name: str
    path: Path
    sha256: str
    prompts: tuple[Prompt, ...]
    manifest: dict | None = None

    def buckets(self) -> list[str]:
        seen = []
        for p in self.prompts:
            if p.bucket not in seen:
                seen.append(p.bucket)
        return sorted(seen, key=lambda b: (BUCKETS.index(b) if b in BUCKETS else len(BUCKETS), b))

    def identity(self) -> dict:
        return {"name": self.name, "sha256": self.sha256}


def load(path: str | Path, buckets: list[str] | None = None, limit_per_bucket: int | None = None) -> PromptSet:
    """Load and check a prompt set. ``buckets`` / ``limit_per_bucket`` select a subset (file order)."""
    p = _sets.existing(path)
    if not p.is_file():
        raise OpError(f"prompt set {path} not found")
    try:
        raw = _sets.read_bytes(p)
        text = raw.decode("utf-8")
    except (OSError, ValueError) as e:  # a truncated or corrupt .jsonl.gz, or not UTF-8
        raise OpError(f"prompt set {p.name}: cannot read ({e})") from None
    sha = hashlib.sha256(raw).hexdigest()
    manifest = _manifest(p)
    if manifest and manifest.get("sha256") and manifest["sha256"] != sha:
        raise OpError(f"prompt set {p.name}: sha256 {sha[:12]}... does not match its manifest "
                      f"({str(manifest['sha256'])[:12]}...): the file changed after it was built")
    prompts, ids, per_bucket = [], set(), {}
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError as e:
            raise OpError(f"{p.name}:{n}: not JSON ({e})") from None
        prompt = _parse(obj, f"{p.name}:{n}")
        if prompt.id in ids:
            raise OpError(f"{p.name}:{n}: duplicate id {prompt.id!r}")
        ids.add(prompt.id)
        if buckets and prompt.bucket not in buckets:
            continue
        k = per_bucket.get(prompt.bucket, 0)
        if limit_per_bucket and k >= limit_per_bucket:
            continue
        per_bucket[prompt.bucket] = k + 1
        prompts.append(prompt)
    if not prompts:
        raise OpError(f"prompt set {p.name}: no prompts" + (f" in buckets {buckets}" if buckets else ""))
    name = (manifest or {}).get("name") or _sets.set_stem(p)
    return PromptSet(name=name, path=p, sha256=sha, prompts=tuple(prompts), manifest=manifest)


def _manifest(p: Path) -> dict | None:
    m = _sets.manifest_path(p)
    if not m.is_file():
        return None
    try:
        value = json.loads(m.read_text())
    except ValueError:
        raise OpError(f"{m.name}: not JSON") from None
    return value if isinstance(value, dict) else None


def _parse(obj, where: str) -> Prompt:
    if not isinstance(obj, dict):
        raise OpError(f"{where}: each line must be a JSON object")
    pid, bucket, messages = obj.get("id"), obj.get("bucket"), obj.get("messages")
    if not isinstance(pid, str) or not pid or "/" in pid or pid.startswith("."):
        raise OpError(f"{where}: 'id' must be a non-empty string without '/'")
    if not isinstance(bucket, str) or not bucket or "/" in bucket or bucket.startswith("."):
        raise OpError(f"{where}: 'bucket' must be a non-empty string without '/'")
    if not isinstance(messages, list) or not messages or not all(
            isinstance(m, dict) and isinstance(m.get("role"), str) for m in messages):
        raise OpError(f"{where}: 'messages' must be a non-empty list of {{role, content}} objects")
    max_tokens = obj.get("max_tokens")
    if max_tokens is not None and (not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1):
        raise OpError(f"{where}: 'max_tokens' must be a positive integer")
    extra = {k: v for k, v in obj.items() if k not in ("id", "bucket", "messages", "max_tokens", "kind")}
    return Prompt(id=pid, bucket=bucket, messages=tuple(messages), max_tokens=max_tokens,
                  kind=str(obj.get("kind") or "chat"), extra=extra)
