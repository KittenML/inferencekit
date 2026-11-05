"""Turn server logprob payloads into token ids.

Chat completions (OpenAI format) name tokens by *string*, which is ambiguous. Token ids
are recovered, in this order, and the method is recorded:
  ``token_id_strings``  vLLM with ``return_tokens_as_token_ids``: tokens read "token_id:1234";
  ``id_field``          servers that add an integer ``id`` to each logprob entry (llama-server);
  ``return_token_ids``  vLLM with ``return_token_ids``: ``choices[0].token_ids`` (chosen tokens only);
  ``strings``           none of the above: keys stay token strings (scoring then re-tokenizes).

Teacher forcing uses vLLM's ``prompt_logprobs`` on ``/v1/completions`` (a vLLM extension):
``choices[0].prompt_logprobs`` is a list with one entry per prompt token; entry 0 is
null (the first token has no context) and entry i maps token id -> {logprob, rank,
decoded_token} for the top-k tokens at position i plus the token actually there.
"""

from __future__ import annotations

import re

_TOKEN_ID = re.compile(r"^token_id:(\d+)$")


def _entry_id(entry: dict):
    tok = entry.get("token")
    if isinstance(tok, str):
        m = _TOKEN_ID.match(tok)
        if m:
            return int(m.group(1)), "token_id_strings"
    for key in ("id", "token_id"):
        if isinstance(entry.get(key), int):
            return entry[key], "id_field"
    return None, None


def parse_chat_logprobs(response: dict) -> dict:
    """{token_ids, tokens, logprobs, top: [{key: logprob}], method, top_k_returned, text, finish_reason}.

    ``top[i]`` always contains the chosen token as well as the returned alternatives.
    Keys are ints when ids could be recovered, else ``"s:" + token string``.
    """
    choice = (response.get("choices") or [{}])[0]
    content = ((choice.get("logprobs") or {}).get("content")) or []
    side_ids = choice.get("token_ids")
    ids, tokens, lps, tops, methods = [], [], [], [], set()
    for i, entry in enumerate(content):
        tid, how = _entry_id(entry)
        if tid is None and isinstance(side_ids, list) and i < len(side_ids):
            tid, how = side_ids[i], "return_token_ids"
        methods.add(how or "strings")
        key = tid if tid is not None else "s:" + str(entry.get("token"))
        top = {}
        for alt in entry.get("top_logprobs") or []:
            aid, _ = _entry_id(alt)
            akey = aid if aid is not None else "s:" + str(alt.get("token"))
            if akey not in top:
                top[akey] = alt.get("logprob")
        top_k = len(top)
        top.setdefault(key, entry.get("logprob"))
        ids.append(tid)
        tokens.append(entry.get("token"))
        lps.append(entry.get("logprob"))
        tops.append({"k": top_k, "top": top})
    if isinstance(side_ids, list) and side_ids and len(side_ids) == len(ids):
        mismatch = [i for i, (a, b) in enumerate(zip(ids, side_ids)) if a is not None and a != b]
    else:
        mismatch = []
    method = "strings" if "strings" in methods or not methods else sorted(methods)[0]
    if None in ids:
        method = "strings"
    msg = choice.get("message") or {}
    return {"token_ids": ids if method != "strings" else None, "tokens": tokens, "logprobs": lps,
            "top": [t["top"] for t in tops], "method": method if content else "none",
            "top_k_returned": min((t["k"] for t in tops), default=0),
            "id_mismatch_positions": mismatch, "text": msg.get("content") if isinstance(msg, dict) else None,
            "finish_reason": choice.get("finish_reason"), "side_token_ids": side_ids}


def parse_prompt_logprobs(response: dict) -> list[dict | None]:
    """[{token_id: (logprob, rank)} | None] per prompt position, from a vLLM /v1/completions response."""
    choice = (response.get("choices") or [{}])[0]
    plp = choice.get("prompt_logprobs")
    if plp is None:
        plp = response.get("prompt_logprobs")
    if not isinstance(plp, list):
        return []
    out: list[dict | None] = []
    for entry in plp:
        if not entry:
            out.append(None)
            continue
        pos = {}
        for k, v in entry.items():
            if isinstance(v, dict):
                pos[int(k)] = (v.get("logprob"), v.get("rank"))
            else:  # a bare float
                pos[int(k)] = (float(v), None)
        out.append(pos)
    return out


def argmax_token(position: dict) -> int:
    """The reference's most likely token at a position: rank 1 if ranks are given, else max logprob."""
    ranked = [k for k, (_, rank) in position.items() if rank == 1]
    if ranked:
        return ranked[0]
    return max(position, key=lambda k: position[k][0])
