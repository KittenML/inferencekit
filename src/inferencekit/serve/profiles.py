"""Per-engine request profiles: how to ask each OpenAI-compatible server for an exact output length,
streamed token ids and a usage block, using only fields that server documents.

Why it matters: a fair speed comparison needs every engine to generate the same number of tokens
(otherwise an engine that stops early at EOS looks faster), and an honest decode speed needs a token
count that does not depend on how many tokens a server packs into one SSE event.

Sources (checked 13 Oct 2026; fields can move between releases, so the client parses every known
location and the record says which one was used):
  vLLM     vllm/entrypoints/openai/chat_completion/protocol.py (ChatCompletionRequest: ignore_eos,
           min_tokens, return_token_ids, return_tokens_as_token_ids, stream_options;
           ChatCompletionResponseStreamChoice.token_ids "for tracing the tokens") and
           docs.vllm.ai/en/v0.12.0/serving/openai_compatible_server/ ("Extra parameters": extra
           fields are merged into the JSON body; return_token_ids needs vLLM >= 0.10.2).
  SGLang   python/sglang/srt/entrypoints/openai/protocol.py (ignore_eos, min_tokens, stream_options,
           SglExt.output_ids) and serving_chat.py (return_token_ids is rejected when stream=true, so it
           is not requested; usage arrives in a final chunk when stream_options.include_usage is set).
  llama.cpp tools/server/README.md (ignore_eos and other /completion fields are accepted on the
           OpenAI-compatible endpoints) and tools/server/server-task.cpp (logprob entries carry "id";
           usage arrives in a trailing chunk when include_usage is set). llama-server has no
           min_tokens. Asking for logprobs makes it compute token probabilities every step, which
           costs decode time, so ids are requested only with --token-ids logprobs.
  TRT-LLM  tensorrt_llm/serve/openai_protocol.py (ChatCompletionRequest / CompletionRequest:
           ignore_eos, min_tokens, stream_options{include_usage}; CompletionResponseStreamChoice.token_ids;
           no request field to ask for chat token ids).
  generic  plain OpenAI fields only (max_tokens, stream, stream_options.include_usage).
"""

from __future__ import annotations

from dataclasses import dataclass, field

ENGINE_KINDS = ("vllm", "sglang", "llamacpp", "trtllm", "generic")
APIS = ("chat", "completions")
TOKEN_ID_MODES = ("auto", "logprobs", "off")


@dataclass(frozen=True)
class Profile:
    name: str
    # Body fields that force exactly max_tokens; "{n}" is replaced by max_tokens.
    exact_length: dict = field(default_factory=dict)
    # Body fields that make the server stream generated token ids at (near) zero cost, per API.
    ids_free: dict = field(default_factory=dict)
    # Body fields that stream token ids through logprobs (may slow decode), per API.
    ids_logprobs: dict = field(default_factory=dict)
    note: str = ""


PROFILES: dict[str, Profile] = {
    "vllm": Profile(
        "vllm",
        exact_length={"ignore_eos": True, "min_tokens": "{n}"},
        ids_free={"chat": {"return_token_ids": True}, "completions": {"return_token_ids": True}},
        ids_logprobs={"chat": {"logprobs": True, "top_logprobs": 0, "return_tokens_as_token_ids": True},
                      "completions": {"logprobs": 0, "return_tokens_as_token_ids": True}},
        note="token ids from choices[].token_ids (return_token_ids, vLLM >= 0.10.2)"),
    "sglang": Profile(
        "sglang",
        exact_length={"ignore_eos": True, "min_tokens": "{n}"},
        note="return_token_ids is not supported with streaming; tokens counted from usage unless the "
             "server adds sglext.output_ids"),
    "llamacpp": Profile(
        "llamacpp",
        exact_length={"ignore_eos": True},
        ids_logprobs={"chat": {"logprobs": True, "top_logprobs": 1}, "completions": {"logprobs": 1}},
        note="no min_tokens; ids only through logprobs (costs decode time), else usage"),
    "trtllm": Profile(
        "trtllm",
        exact_length={"ignore_eos": True, "min_tokens": "{n}"},
        note="completions stream choices may carry token_ids; chat has none, so usage is used"),
    "generic": Profile("generic", note="standard OpenAI fields only: no exact-length control"),
}


def build_body(api: str, engine_kind: str, model: str, messages: list[dict], max_tokens: int, *,
               token_ids: str = "auto", exact_length: bool = True, extra: dict | None = None) -> tuple[dict, dict]:
    """The JSON body for one streamed request, and a description of what was asked for.

    Greedy decoding (temperature 0) by default so outputs are comparable across engines; ``extra``
    (the user's --extra-body) is merged last and can override anything.
    """
    prof = PROFILES[engine_kind]
    body: dict = {"model": model}
    if api == "chat":
        body["messages"] = list(messages)
    else:
        body["prompt"] = "\n\n".join(str(m.get("content", "")) for m in messages)
    body.update({"max_tokens": int(max_tokens), "temperature": 0.0, "stream": True,
                 "stream_options": {"include_usage": True}})
    asked = {"exact_length": None, "token_ids": None}
    if exact_length and prof.exact_length:
        body.update({k: (int(max_tokens) if v == "{n}" else v) for k, v in prof.exact_length.items()})
        asked["exact_length"] = sorted(prof.exact_length)
    if token_ids != "off":
        fields = prof.ids_free.get(api)
        how = "free"
        if not fields and token_ids == "logprobs":
            fields, how = prof.ids_logprobs.get(api), "logprobs"
        if fields:
            body.update(fields)
            asked["token_ids"] = {"via": how, "fields": sorted(fields)}
    if extra:
        body.update(extra)
    return body, asked
