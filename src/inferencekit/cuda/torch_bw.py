"""Bandwidth fallback with PyTorch, used when nvcc is not available.

Prints the same JSON lines as bw_peak.cu ("device", "read", "copy"). It runs in its
own process (``python torch_bw.py --sizes-mib ...``) so the caller never imports
torch or holds a CUDA context. Expect a few percent below bw_peak.cu: a reduction
kernel also does arithmetic and a final cross-block step, a pure load loop does not.
Standard library + torch only; does not import inferencekit.
"""

import argparse
import json
import statistics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes-mib", default="8,32,128,512,1024")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    import torch

    if not torch.cuda.is_available():
        print(json.dumps({"test": "error", "message": "torch.cuda is not available"}))
        return 2
    props = torch.cuda.get_device_properties(0)
    l2 = getattr(props, "L2_cache_size", 0) or (64 << 20)
    print(json.dumps({"test": "device", "name": props.name, "cc": f"{props.major}.{props.minor}",
                      "sms": props.multi_processor_count, "l2_bytes": l2, "theoretical_gbs": None}))
    flush = torch.empty(max(4 * l2, 64 << 20) // 4, dtype=torch.float32, device="cuda")
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def timed(fn) -> float:
        times = []
        for it in range(args.iters + 3):
            flush.add_(1.0)  # evict L2 so the timed op reads DRAM
            start.record()
            fn()
            end.record()
            end.synchronize()
            if it >= 3:
                times.append(start.elapsed_time(end))
        return statistics.median(times)

    for mib in [int(x) for x in args.sizes_mib.split(",") if x]:
        nbytes = mib << 20
        x = torch.ones(nbytes // 4, dtype=torch.float32, device="cuda")
        y = torch.empty(nbytes // 8, dtype=torch.float32, device="cuda")
        ms = timed(lambda: torch.sum(x))
        print(json.dumps({"test": "read", "bytes": nbytes, "iters": args.iters, "ms_median": ms,
                          "gbs": nbytes / ms / 1e6, "method": "torch.sum"}))
        half = x[: nbytes // 8]
        ms = timed(lambda: y.copy_(half))
        print(json.dumps({"test": "copy", "bytes": nbytes, "iters": args.iters, "ms_median": ms,
                          "gbs": nbytes / ms / 1e6, "method": "Tensor.copy_"}))
        del x, y
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
