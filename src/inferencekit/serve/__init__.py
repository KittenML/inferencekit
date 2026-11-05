"""bench serve: a client-side timer for OpenAI-compatible inference servers.

  prompts_io  read a prompt set (JSONL, as ``prompts build`` writes it) and its sha256
  profiles    per-engine request bodies: ignore_eos / min_tokens, token ids, usage
  client      async streaming SSE client on raw asyncio sockets, timed with perf_counter_ns
  metrics     per-request numbers, per-bucket summaries with bootstrap CIs over rounds
  loadgen     open-loop seeded Poisson load at a concurrency or rate grid; N* and cost
"""
