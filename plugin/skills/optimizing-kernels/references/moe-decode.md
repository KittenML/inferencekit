# MoE decode at batch 1: gather-GEMV over the chosen experts

Terms: a **mixture-of-experts (MoE)** layer has E expert MLPs. A **router** (a small E × H GEMV) scores them for each token; the top k run, and their outputs are summed with the router's weights. A **shared expert** or dense MLP runs for every token. A **grouped GEMM** is a library kernel that runs one GEMM per expert over the tokens routed to it.

## Bytes and why the library path is slow

At batch 1 a token reads the router, any shared expert or dense MLP, and only k of the E experts per layer, so bytes per token count only the active experts (`inferencekit roofline --method headers` does this). Batched MoE paths sort tokens by expert, align them to tiles, quantise activations, run grouped GEMMs, apply the activation and combine. At one token that is many kernels per layer, each one a hand-off, and the grouped-GEMM tiles waste most of their rows (see [fp8-gemv.md](fp8-gemv.md)).

## Two kernels per layer

1. **Routing + gate/up.** In every block, one warp reads the router logits (from the kernel that produced the MoE input, or a tiny GEMV) and does softmax, top-k and renormalisation in registers, plus any per-expert scale the model defines. Use exactly the reference's order of operations and tie rule. If the checkpoint quantises activations (for example 4-bit with a static input scale), quantise the MoE input with the checkpoint's scale and formula before the gate/up GEMV. Then the warps stream rows of the chosen experts' gate and up matrices: a gather-GEMV whose grid is k experts × row tiles, so the whole GPU is busy although each expert matrix is small. The epilogue computes activation(gate) × up and, if the down projection's input is quantised too, re-quantises the intermediate with that scale. Write the k intermediate vectors and the routing (expert ids, weights) to small device buffers.
2. **Down + combine.** Each output row reads the k intermediate vectors and the k experts' down rows, and sums the experts in top-k order inside one thread or warp. There are no atomics, so the result is deterministic run to run. Round where the reference rounds (for example, each expert's output to 16 bits before the weighted sum).

Rules that hold for both:
- **Loads in flight.** Issue all of a thread's 16-byte weight loads for an unrolled chunk before the first FMA, so each SM keeps enough bytes outstanding (Little's law, [fp8-gemv.md](fp8-gemv.md)).
- **Numerics as shipped.** Keep the checkpoint's activation scheme (input and intermediate). 16-bit activations change the numerics: use them only when the task allows it, declare them in the engine's `numerics`, and judge them by the matching row of [proving-correctness](../../proving-correctness/SKILL.md).
- **Stay inside the CUDA graph.** Expert ids live on the device. Grids are sized for k, which is fixed, never for the number of distinct experts picked. The host never reads the routing.
- **Fallback.** Shapes or settings the kernels do not support (another expert count, prefill with more than one token) go to the engine's own path, and the reason is logged once.
- **Re-quantising the intermediate** makes results sensitive to summation order: a different fp32 order flips a few 4-bit codes. Expect small per-call differences from the library kernel, and judge them by the equivalence row (noise floor), not by bitwise equality.

## What measured worse in our public engine

Our public batch-1 MoE engine uses the two-kernel design above. Its agent also tried, and dropped:
- **PDL between the two MoE kernels**, with the second prefetching its weights before the wait: slower in isolated timings of the pair (agent-measured; no whole-step A/B), with the trigger early or late. Cause not established.
- **Recomputing top-k in the second kernel** instead of reading the first kernel's routing buffer: slower. Inferred cause: every block repeats the routing work.

## Speculation reads more experts

Verifying n draft tokens in one step reads the union of their experts, up to n × k experts per layer. In a dense model, by contrast, the n tokens share one weight read. Speculative decoding therefore gains less on MoE at batch 1: measure acceptance and step cost together, on the user's prompts.

## Proof

- Routing: on recorded hidden states, the expert ids and weights equal the reference routing; constructed ties resolve as the reference resolves them.
- Per layer: the fused MoE output against a torch reference with the same numerics, within the spread an A/A run of the reference shows (bitwise only when the arithmetic order is the same).
- End to end: the equivalence row of proving-correctness, and the gates.
