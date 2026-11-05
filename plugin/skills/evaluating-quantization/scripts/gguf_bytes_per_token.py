#!/usr/bin/env python3
"""Weight bytes read per decoded token at batch 1, from a GGUF file's header (claim class: derived).

    python3 gguf_bytes_per_token.py <file.gguf> [<shard-2.gguf> ...]
    python3 gguf_bytes_per_token.py --repo <org/name-GGUF> --file <name.gguf> [--file <shard 2>] [--revision <sha>]

`inferencekit roofline` reads safetensors headers only. This helper does the same for GGUF (llama.cpp's
format): it reads the header (metadata + tensor table), never the tensor data. From the Hub it uses
HTTP range requests, a few MB per file whatever the file size; HF_TOKEN (read from the environment by
name, never printed) is sent for gated repos and never forwarded to the storage host. Stdlib only.

What it counts (the weight term only; the same rules as `roofline --method headers`):
  * every tensor is read once per token, at its stored size (type block size x shape);
  * token_embd: one row per token, unless the file has no output.weight (tied lm_head): then it is the
    logits matrix and read in full, counted once;
  * MoE routed experts (`*_exps` tensors): x expert_used_count / expert_count (batch 1);
    shared experts (`*_shexp`) in full.
The KV cache does not depend on the weight format: take `kv_bytes` from `inferencekit roofline` on the
source safetensors repo at the same context and KV dtype, and add it.
Output: JSON on stdout. Exit 2 (reason on stderr) instead of guessing.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request

# ggml type id -> (name, elements per block, bytes per block); from ggml's type traits (gguf-py constants)
GGML = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 40), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17), 40: ("NVFP4", 64, 36),
}
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
CHUNK = 4 << 20
MAX_HEADER = 512 << 20  # stop before reading anything that could be weights


def fail(msg: str) -> None:
    print(f"gguf_bytes_per_token: {msg}", file=sys.stderr)
    sys.exit(2)


class _CommitRedirects(urllib.request.HTTPRedirectHandler):
    """Keeps the X-Repo-Commit header the Hub sends on the redirect to the storage host."""

    commit: str | None = None

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.commit = self.commit or headers.get("X-Repo-Commit")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Source:
    """Sequential reads from a local file or a Hub URL (range requests, CHUNK at a time)."""

    def __init__(self, path: str | None = None, url: str | None = None):
        self.path, self.url, self.buf, self.pos, self.commit = path, url, b"", 0, None
        self.fh = open(path, "rb") if path else None

    def _fetch(self, start: int, n: int) -> bytes:
        if self.fh:
            self.fh.seek(start)
            return self.fh.read(n)
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{start + n - 1}"})
        token = os.environ.get("HF_TOKEN")
        if token:
            req.add_unredirected_header("Authorization", f"Bearer {token}")
        redirects = _CommitRedirects()
        try:
            with urllib.request.build_opener(redirects).open(req, timeout=60) as r:
                self.commit = self.commit or redirects.commit or r.headers.get("X-Repo-Commit")
                if r.status == 200 and start > 0:  # server ignored the range
                    fail("the server ignored the Range header; download the file and pass its path")
                return r.read(n)
        except urllib.error.HTTPError as e:
            fail(f"HTTP {e.code} for {self.url.split('?')[0]} (gated repo: export HF_TOKEN)")
        return b""

    def read(self, n: int) -> bytes:
        while len(self.buf) - self.pos < n:
            have = len(self.buf)
            if have + CHUNK > MAX_HEADER:
                fail(f"header larger than {MAX_HEADER >> 20} MB: not a GGUF header?")
            more = self._fetch(have, max(CHUNK, n))
            if not more:
                fail("file ended inside the header")
            self.buf += more
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        return self.read(self.unpack("<Q")).decode("utf-8", "replace")

    def value(self, vtype: int):
        if vtype in SCALAR:
            return self.unpack(SCALAR[vtype])
        if vtype == 8:
            return self.string()
        if vtype == 9:
            etype, count = self.unpack("<I"), self.unpack("<Q")
            items = [self.value(etype) for _ in range(count)]
            return items if count <= 16 else f"<array of {count}>"  # tokenizer tables: not needed
        fail(f"unknown GGUF metadata value type {vtype}")


def parse(src: Source) -> tuple[dict, list[dict]]:
    if src.read(4) != b"GGUF":
        fail("not a GGUF file (bad magic)")
    version = src.unpack("<I")
    if version < 2:
        fail(f"GGUF version {version} is not supported (v2+ only)")
    n_tensors, n_kv = src.unpack("<Q"), src.unpack("<Q")
    meta = {"gguf_version": version}
    for _ in range(n_kv):
        key = src.string()
        meta[key] = src.value(src.unpack("<I"))
    tensors = []
    for _ in range(n_tensors):
        name = src.string()
        dims = [src.unpack("<Q") for _ in range(src.unpack("<I"))]
        ttype = src.unpack("<I")
        src.unpack("<Q")  # data offset
        if ttype not in GGML:
            fail(f"tensor {name}: unknown ggml type {ttype}; update the GGML table from gguf-py")
        tname, block, block_bytes = GGML[ttype]
        elems = 1
        for d in dims:
            elems *= d
        if elems % block:
            fail(f"tensor {name}: {elems} elements is not a multiple of the {tname} block ({block})")
        tensors.append({"name": name, "type": tname, "elements": elems, "rows": dims[1] if len(dims) > 1 else 1,
                        "bytes": elems // block * block_bytes})
    return meta, tensors


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="local .gguf file(s); every shard of a split model")
    ap.add_argument("--repo", help="Hugging Face repo id (instead of local paths)")
    ap.add_argument("--file", action="append", default=[], help="file in --repo (repeat for each shard)")
    ap.add_argument("--revision", default="main", help="branch, tag or commit sha (pin a sha)")
    a = ap.parse_args(argv)
    if bool(a.paths) == bool(a.repo):
        fail("pass local paths, or --repo with --file")
    endpoint = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    sources = [Source(path=p) for p in a.paths] or [
        Source(url=f"{endpoint}/{a.repo}/resolve/{urllib.parse.quote(a.revision, safe='')}/{urllib.parse.quote(f)}")
        for f in a.file]
    if not sources:
        fail("--repo needs at least one --file")

    meta, tensors = {}, []
    for s in sources:
        m, t = parse(s)
        meta = meta or m
        tensors += t
    arch = meta.get("general.architecture", "")
    n_exp, used = meta.get(f"{arch}.expert_count"), meta.get(f"{arch}.expert_used_count")
    names = {t["name"] for t in tensors}
    tied = "output.weight" not in names
    notes, by_type, groups = [], {}, {"linear": 0.0, "moe_experts": 0.0, "lm_head": 0.0, "embedding_row": 0.0,
                                      "norms_other": 0.0}
    lin_bytes = lin_elems = 0
    for t in tensors:
        n = t["name"]
        if n.startswith("token_embd."):
            if tied and n == "token_embd.weight":
                read, g = t["bytes"], "lm_head"
                notes.append("no output.weight: token_embd is the tied lm_head, read in full, counted once")
            else:
                read, g = t["bytes"] / t["rows"], "embedding_row"
        elif n.startswith("output.") and n.endswith("weight") and t["rows"] > 1:
            read, g = t["bytes"], "lm_head"
        elif "_exps" in n:
            if not n_exp or not used:
                fail(f"{n} is a routed-expert tensor but {arch}.expert_count / expert_used_count are missing")
            read, g = t["bytes"] * used / n_exp, "moe_experts"
        elif t["rows"] > 1:
            read, g = t["bytes"], "linear"
            lin_bytes, lin_elems = lin_bytes + t["bytes"], lin_elems + t["elements"]
        else:
            read, g = t["bytes"], "norms_other"
        groups[g] += read
        by_type[t["type"]] = by_type.get(t["type"], 0.0) + read
    lm_type = next((t["type"] for t in tensors if t["name"] == ("token_embd.weight" if tied else "output.weight")),
                   None)
    weights = sum(groups.values())
    out = {
        "method": "gguf-header",
        "source": a.paths or {"repo": a.repo, "files": a.file, "revision": a.revision,
                              "commit": sources[0].commit},
        "architecture": arch,
        "gguf_version": meta.get("gguf_version"),
        "file_type": meta.get("general.file_type"),
        "tensors": len(tensors),
        "stored_bytes": sum(t["bytes"] for t in tensors),
        "weight_bytes_per_token": weights,
        "groups_read_bytes": groups,
        "read_bytes_by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
        "lm_head": {"type": lm_type, "tied": tied, "bytes": groups["lm_head"]},
        "linear_bits_per_weight": round(8 * lin_bytes / lin_elems, 3) if lin_elems else None,
        "moe": {"expert_count": n_exp, "expert_used_count": used} if n_exp else None,
        "notes": list(dict.fromkeys(notes)) + [
            "weight term only: add kv_bytes from `inferencekit roofline` on the source safetensors repo (same "
            "context, the KV dtype the engine logged)",
            "MoE at batch 1 reads expert_used_count experts per layer per token; at batch > 1 tokens share experts"],
        "formula": "bytes_per_token = weight_bytes_per_token + kv_bytes(context, kv dtype)",
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
