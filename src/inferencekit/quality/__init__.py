"""Quality checks for LLM engines: does a faster engine still produce the model's output?

Two complementary packs (the operations live in ``inferencekit.ops.quality``):

* **Trajectory scoring** (``quality trajectory`` then ``quality score``): every engine
  generates its own greedy text with top-k logprobs; a higher-precision reference
  (e.g. vLLM serving BF16) then re-reads that exact text token by token ("teacher
  forcing") and reports, per position bucket, top-1 agreement, KL(reference || engine),
  the reference's negative log-likelihood of the engine's tokens, and where the engine
  first leaves the reference's own greedy path. An A/A run of the reference against
  itself gives the noise floor every engine is judged against.
* **Task accuracy** (``quality gsm8k``): GSM8K test split, zero-shot chain of thought,
  exact numeric answer match, with a Wilson CI and McNemar's exact test for pairs.

Module map:
  metrics.py   KL on truncated top-k lists, position buckets, bootstrap CIs, Wilson, McNemar
  client.py    a tiny stdlib OpenAI-compatible HTTP client
  logprobs.py  parse chat-completion logprobs and vLLM prompt_logprobs into token ids
  gsm8k.py     the GSM8K test split (pinned HF download, sha256-checked, cached), prompt and
               answer extraction
  parquet.py   a dependency-free reader for the flat Parquet file that split comes in
  fake.py      a deterministic fake OpenAI-compatible server (tests and dry runs)
"""
