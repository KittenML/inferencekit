# Prompt set attribution

The prompt sets in this directory (`dev.jsonl.gz`, `dev-load.jsonl.gz`: gzipped JSONL) are
built by `scripts/build_prompts.py` from the openly licensed sources below. Every item records
its source and licence in its `source` field (`<dataset>/<licence>`) and the exact origin in
`source_ref`. Each manifest (`*.manifest.json`) lists the source URLs and the sha256 of the
bytes the set was built from; its own `sha256` is that of the uncompressed JSONL
(`gunzip -c dev.jsonl.gz | sha256sum`).

| Source | Used for | Licence |
|---|---|---|
| [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) (Databricks, Inc.; revision `bdd27f4d94b9`) | very_short, short (instructions); medium (context paragraph + instruction) | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) |
| [GSM8K](https://github.com/openai/grade-school-math) train split (OpenAI; commit `3101c7d50724`) | short (questions only) | [MIT](https://github.com/openai/grade-school-math/blob/master/LICENSE) |
| Public-domain novels, plain-text editions from [Project Gutenberg](https://www.gutenberg.org/) (titles in the manifests) | long (passages of ~1,700-2,600 words plus a question) | Public domain in the USA |

## Changes made
- dolly: whitespace normalised; an item's context paragraph(s) and instruction joined as one
  user message; responses not used.
- GSM8K: whitespace normalised; answers not used.
- Novels: the distributor's licence header and footer removed; paragraphs re-flowed; a
  contiguous passage selected and followed by a question written for this set.

## Share-alike
Prompts derived from databricks-dolly-15k are adaptations of a CC BY-SA 3.0 work and are
distributed under the same licence (CC BY-SA 3.0). That covers the dolly-derived lines of
`dev.jsonl.gz` and `dev-load.jsonl.gz` (their `source` field starts with `databricks-dolly-15k`),
not the inferencekit code, which is Apache-2.0. Items from GSM8K keep their MIT licence
(copyright (c) 2021 OpenAI; full text in `../quality/data/gsm8k_LICENSE.txt`). The novels are in the public domain in the USA; check your
local law before redistributing them elsewhere.

## Lengths
Buckets are measured with a tokenizer-free proxy, `proxy_tokens = whitespace words * 1.3`:
very_short 1-5 words; short 20-50; medium 240-360; long 2,000-4,000 proxy tokens. For
English prose the real token count of a 30k-150k-vocabulary BPE tokenizer is usually within
about 15% of the proxy.

## Exclusions
A few source rows are left out because their text would trip the release leak scanner.
`exclusions.json` lists them by position only (dolly or GSM8K row, or novel and passage), keyed
by the sha256 of the source file, so a rebuild from the same source bytes gives the same sets.

## Sealed set
A sealed set is built by the same code (`inferencekit prompts seal`) from source groups the dev
sets do not use. A group is one dolly context paragraph (every row asking about that paragraph),
one instruction or question, or one novel, so the sealed set shares no prompt, no context
paragraph and no novel with the dev sets; a sealed prompt that shares more than half of its
12-word sequences with a dev prompt is dropped as well. Which groups are chosen depends on a
private random secret, so the sealed set cannot be rebuilt from this repository alone. It is not
distributed here. Its sha256 and the secret's `secret_sha256` are published ahead of the
evaluation, and the secret file is revealed afterwards so anyone can rebuild the set and check
the hashes.

`secret_sha256` is the sha256 of the secret file's bytes as `prompts seal` writes it (64 hex
characters and a newline): `sha256sum SEED.secret`, or `echo <secret> | sha256sum`.
`inferencekit prompts verify --set sealed.jsonl --secret-file SEED.secret --sources DIR` checks
that hash and rebuilds the set to compare it with the published sha256.
