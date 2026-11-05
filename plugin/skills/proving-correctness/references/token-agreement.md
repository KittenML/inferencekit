# Token agreement against a Hugging Face reference

## Building the reference

- Load the model with `transformers` at the candidate's precision when a faithful implementation exists; otherwise BF16. Use the same local model files as the candidate (pinned revision).
- Use the same chat template and tokenisation. Check that both sides produce identical prompt token ids before comparing anything else.
- Greedy decoding (`do_sample=False`), fixed `max_new_tokens`, EOS ignored or handled identically.
- Save, per prompt: prompt token ids, generated token ids, and the logits (or at least the top-k logits) at each generated position.

Minimal reference generation (BF16, for learning; adapt batching and storage):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "./models/qwen3-8b"                                  # pinned local copy
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()

ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                              add_generation_prompt=True, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=128, do_sample=False,
                         output_logits=True, return_dict_in_generate=True)
gen_ids = out.sequences[0, ids.shape[1]:].tolist()          # generated token ids
step_logits = torch.stack(out.logits)[:, 0].float().cpu()   # [steps, vocab], one row per generated token
```

`output_logits=True` returns the raw logits at each step, before any logits processor (repetition penalties and the like from the model's `generation_config` would otherwise alter `output_scores`). Check the generation config anyway: greedy reference outputs must not depend on processors the candidate does not apply.

### Feeding the reference to inferencekit

Use the engine's exact prompt token ids from the benchmark harness instead of re-templating, so both sides see identical input. For vLLM/SGLang `bench decode`, `data.prompt` contains the generation rule and vocabulary bound, not the ids; reproduce that rule or save the ids in the harness. Then write one JSON line per prompt:

```python
import json
ids = torch.tensor([prompt_ids], device="cuda")             # exact prompt ids from the benchmark harness
# ... model.generate(...) as above ...
logprobs = torch.log_softmax(step_logits, dim=-1)           # [steps, vocab]
chosen = logprobs[torch.arange(len(gen_ids)), gen_ids].tolist()
with open("ref_hf.jsonl", "a") as f:
    f.write(json.dumps({"tokens": gen_ids, "logprobs": chosen}) + "\n")
```

Then `inferencekit correctness --reference ref_hf.jsonl --candidate <ev_bench>`.

## First divergence and the margin test

For each prompt, find `j`, the first index where candidate and reference tokens differ. At position `j`, look at the reference's logits:

```
margin_j = logit_ref[top1] - logit_ref[top2]
```

- **Small margin** (a near-tie, within the control's noise): either token is a valid outcome of slightly different arithmetic. Expected.
- **Large margin**: the candidate chose a token the reference was confident against. Something computes wrong. Investigate before any speed work.

Calibrate "small": run the A/A control (for example the reference with a different attention implementation, or the old build twice under different kernels) and record the distribution of margins at its divergences. A candidate divergence with a margin above the control's maximum is a FAIL.

## Teacher forcing

Greedy comparisons cascade: after one flip, everything differs. Teacher forcing removes that: feed the *reference* token sequence to the candidate and compare the candidate's argmax and logits at every position against the reference's. This measures per-step error directly. Report:
- top-1 agreement (% of positions where argmaxes match);
- relative L2 of the logit vectors, per position: mean and max;
- KL(p_ref || p_cand) per position: mean and max.

## Reading the numbers

The patterns table is in [quality.md](quality.md), section 8. Here too, late divergences at small margins with teacher-forced agreement inside the control band mean correct, different arithmetic; a logit relative L2 rising with position means accumulating error.

## Why not just compare final text

Text comparison hides where things went wrong and treats a near-tie flip at token 120 the same as garbage at token 1. Positions, margins and teacher-forced metrics separate the cases.
