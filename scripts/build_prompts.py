#!/usr/bin/env python3
"""Rebuild the prompt sets from their open-licence sources.

    uv run python scripts/build_prompts.py --cache ~/.cache/inferencekit/prompt-sources --fetch
    uv run python scripts/build_prompts.py --cache DIR --sealed-out /path/outside/the/repo/sealed

Deterministic: the same source bytes and seed give byte-identical JSONL files (the
manifests record each source file's sha256, so a changed upstream file is visible).
The dev sets are written into src/inferencekit/prompts/; the sealed sets go only to
--sealed-out, which must be outside every git work tree, together with SHA256SUMS.
See src/inferencekit/prompts/build.py for the sources, bucket rules and split rule.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from inferencekit import registry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "inferencekit" / "prompt-sources"),
                    help="directory for the downloaded source files")
    ap.add_argument("--fetch", action="store_true", help="download missing source files")
    ap.add_argument("--dev-out", default=str(REPO / "src" / "inferencekit" / "prompts"))
    ap.add_argument("--sealed-out", default=None, help="also build the sealed sets here (outside the repo)")
    ap.add_argument("--seed", default="inferencekit-prompts-v1")
    ap.add_argument("--blocklist", default=None,
                    help="private scrub-patterns file whose matches are excluded (default: none)")
    a = ap.parse_args()
    out = {"dev": registry.call("prompts_build", {"sources": a.cache, "out": a.dev_out, "fetch": a.fetch,
                                                  "seed": a.seed, "blocklist": a.blocklist})}
    if a.sealed_out:
        out["sealed"] = registry.call("prompts_seal", {"sources": a.cache, "out": a.sealed_out, "fetch": a.fetch,
                                                       "seed": a.seed, "blocklist": a.blocklist})
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
