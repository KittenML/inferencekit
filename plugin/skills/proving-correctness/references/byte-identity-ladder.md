# The byte-identity ladder for pure refactors

A pure refactor claims "no numeric change": moving code, renaming, restructuring launches, changing a build system, adding a switch that is off. For that claim, statistical agreement is too weak: the outputs must be identical bytes. Climb the ladder in order; each rung is cheaper to debug than the next.

## Rung 1: kernel outputs, byte-equal, across shapes

- Run old and new kernels on identical inputs over a shape sweep: M = 1 and small M, every layer's N and K, sizes that are not multiples of tile sizes, minimum and maximum context lengths.
- **Pre-fill outputs with NaN** (or a sentinel bit pattern) before each call. Why: a kernel that never writes leaves the buffer as it was, and two stale buffers compare equal.
- Compare raw bytes, not values: `torch.equal(a.view(torch.int16), b.view(torch.int16))` for BF16, or `memcmp` in C++. Why: value comparison treats `-0.0 == 0.0` and `NaN != NaN`, which hides or invents differences.

## Rung 2: identical SASS for untouched kernels

- Dump the machine code of both builds: `cuobjdump -sass old_binary > old.sass` and the same for the new build, then diff per kernel.
- Kernels the refactor should not touch must produce identical SASS. Why: a changed compiler flag, include order or template instantiation can alter codegen (for example FMA contraction) without anyone touching the kernel's source.
- For JIT-compiled kernels (Triton, NVRTC), compare the cached cubins or their SASS the same way.

## Rung 3: end-to-end logits, byte-equal

- Run the fixed prompt set through both builds, same batch shape, same graph buckets; compare logits byte for byte at every position (teacher-forced) and generated tokens (greedy).
- If rung 1 and 2 pass but this fails, look at launch order, stream assignment, graph capture differences and uninitialised buffers.

## Rung 4: under load, with an off/off control first

- Real serving runs with other requests, other streams and different timing. Some kernels choose paths by batch size or split heuristics that depend on load.
- **First run the control:** old build vs old build (the feature "off" on both sides) under the same load. It must be identical. Why: if the harness itself is not deterministic under load, any difference in the real comparison is uninterpretable.
- Then old vs new under the same load. Identical outputs complete the ladder.

## When the ladder does not apply

Any intended numeric change (new kernel algorithm, different split count, precision change) fails rung 1 by design. Use token agreement and logit tolerance instead, and say so: "not a pure refactor; statistical correctness only".
