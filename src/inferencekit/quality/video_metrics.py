"""Decode video clips and score them frame by frame, inside the video pipeline's own Python.

Usage: python video_metrics.py CONFIG.json
Prints exactly one line ``INFERKIT_RESULT {...}`` on stdout; the full results go to the file named by
``out`` in the config; logs go to stderr. Standard library at import time; the work needs PyAV and
numpy; LPIPS also needs torch and the ``lpips`` package (PSNR, SSIM and MAD are numpy only). A video
pipeline's venv has all of these; the inferencekit harness has none of them, which is why this runs as a worker (like
engines/vllm_worker.py) and never imports inferencekit.

Config: {"task": "probe" | "compare", "out": path,
         "files": [{"key", "path"}]                        probe: what decoding the file really gives
         "pairs": [{"key", "a", "b"}]                      compare: a = the reference side, b = the other
         "metrics": ["psnr", "ssim", "lpips", "mad"], "device": "auto" | "cpu" | "cuda",
         "frame_stride": 1, "psnr_cap_db": 100.0, "lpips_net": "alex", "ssim_backend": "auto" | "numpy" | "torch"}

probe, per file (decoded, not read from the container header):
  width/height of the decoded frames, the decoded frame count, fps from the decoded timestamps
  ((n - 1) / (t_last - t_first)), audio samples decoded / sample rate; the header's claims beside them.
compare, per pair, per frame (frames decoded in step, never all held in memory):
  psnr  10 log10(255^2 / MSE) over the RGB24 pixels, capped at psnr_cap_db (identical frames)
  ssim  SSIM of the BT.601 luma, 11x11 Gaussian window sigma 1.5, K1 0.01, K2 0.03, valid region
        (Wang et al. 2004; skimage structural_similarity with gaussian_weights=True,
        use_sample_covariance=False and data_range=255 computes the same)
  lpips LPIPS v0.1 (Zhang et al. 2018) with the given backbone, RGB scaled to [-1, 1]
  mad   mean absolute difference of the RGB24 pixels (0-255 levels)
Frames are compared in order up to the shorter clip; both counts are reported. Frames of different
sizes are not compared (the pair gets an error).
"""

import json
import math
import os
import sys
import time

# Run by path, this file's directory comes first on sys.path; its neighbours (metrics.py, client.py...)
# must not shadow a top-level module that torch or PyAV imports.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _HERE]

MARK = "INFERKIT_RESULT "
METRICS = ("psnr", "ssim", "lpips", "mad")
LUMA = (0.299, 0.587, 0.114)  # BT.601
SSIM_SIGMA, SSIM_WIN, SSIM_K1, SSIM_K2 = 1.5, 11, 0.01, 0.03


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- probe

def probe(path):
    """What decoding ``path`` gives, beside what its header claims."""
    import av

    out = {"decoder": "pyav", "video": None, "audio": None, "header": {}}
    with av.open(path) as c:
        out["header"]["format"] = c.format.name
        out["header"]["duration_s"] = c.duration / 1e6 if c.duration else None
        if c.streams.video:
            vs = c.streams.video[0]
            out["header"]["video"] = {"codec": vs.codec_context.name, "width": vs.codec_context.width,
                                      "height": vs.codec_context.height, "frames": vs.frames or None,
                                      "fps": float(vs.average_rate) if vs.average_rate else None}
            sizes, times, n = set(), [], 0
            for f in c.decode(vs):
                n += 1
                sizes.add((f.width, f.height))
                if f.time is not None:
                    times.append(float(f.time))
            times.sort()
            fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) >= 2 and times[-1] > times[0] else None
            w, h = (sorted(sizes)[0] if sizes else (None, None))
            out["video"] = {"frames": n, "width": w, "height": h, "sizes": sorted(sizes), "fps": fps,
                            "first_s": times[0] if times else None, "last_s": times[-1] if times else None,
                            "duration_s": n / fps if fps else None}
    with av.open(path) as c:
        if c.streams.audio:
            st = c.streams.audio[0]
            out["header"]["audio"] = {"codec": st.codec_context.name, "sample_rate": st.codec_context.sample_rate,
                                      "channels": st.codec_context.channels if hasattr(st.codec_context, "channels")
                                      else None,
                                      "duration_s": float(st.duration * st.time_base) if st.duration else None}
            samples, rate = 0, None
            for f in c.decode(st):
                samples += f.samples
                rate = rate or f.sample_rate
            out["audio"] = {"samples": samples, "sample_rate": rate,
                            "duration_s": samples / rate if rate else None}
    return out


# --------------------------------------------------------------------------- compare

class Scorer:
    """Per-frame metrics. PSNR and MAD run in numpy. SSIM runs in numpy, or in torch (same float64 maths) when
    torch is loaded anyway for LPIPS and the device is a GPU (``ssim_backend`` "auto"), or when asked ("torch"):
    at 1920x1088 numpy SSIM takes ~0.7 s a frame on one CPU core, which adds up over a clip set."""

    def __init__(self, metrics, device, psnr_cap_db, lpips_net, ssim_backend="auto"):
        import numpy as np

        self.np = np
        self.metrics = [m for m in metrics if m in METRICS]
        self.cap = float(psnr_cap_db)
        g = np.arange(SSIM_WIN, dtype=np.float64) - (SSIM_WIN - 1) / 2
        g = np.exp(-(g * g) / (2 * SSIM_SIGMA ** 2))
        self.window = g / g.sum()
        self.torch = self.dev = self.lpips = self.lpips_error = None
        self.ssim_backend = "numpy"
        if "lpips" in self.metrics or ("ssim" in self.metrics and ssim_backend == "torch"):
            try:
                import torch

                self.torch = torch
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                self.dev = torch.device(device)
            except ImportError as e:
                if ssim_backend == "torch":
                    raise
                self.lpips_error = f"torch is not installed ({e})"
        if "ssim" in self.metrics and self.torch is not None and (
                ssim_backend == "torch" or (ssim_backend == "auto" and self.dev.type == "cuda")):
            self.ssim_backend = "torch"
            self.twin = self.torch.tensor(self.window, dtype=self.torch.float64, device=self.dev)
        if "lpips" in self.metrics and self.torch is not None:
            try:
                import lpips

                self.lpips = lpips.LPIPS(net=lpips_net, verbose=False).to(self.dev).eval()
            except Exception as e:  # no package, or the backbone weights cannot be fetched
                self.lpips_error = f"{type(e).__name__}: {e}"[:500]
        if self.lpips_error:
            log(f"LPIPS unavailable: {self.lpips_error}")
        self.device = str(self.dev) if self.lpips is not None or self.ssim_backend == "torch" else "cpu"

    def _blur(self, x):
        """Separable Gaussian filter over the valid region (no padding): a sum of shifted slices."""
        k, n = self.window, SSIM_WIN
        h, w = x.shape
        rows = sum(k[i] * x[i:h - n + 1 + i, :] for i in range(n))
        return sum(k[j] * rows[:, j:w - n + 1 + j] for j in range(n))

    def _blur_torch(self, x):
        f = self.torch.nn.functional
        return f.conv2d(f.conv2d(x, self.twin.view(1, 1, -1, 1)), self.twin.view(1, 1, 1, -1))

    def ssim(self, a, b):
        if min(a.shape[0], a.shape[1]) < SSIM_WIN:
            return None
        c1, c2 = (SSIM_K1 * 255) ** 2, (SSIM_K2 * 255) ** 2
        if self.ssim_backend == "torch":
            torch = self.torch
            luma = torch.tensor(LUMA, dtype=torch.float64, device=self.dev)
            ya = (torch.from_numpy(a).to(self.dev, torch.float64) @ luma)[None, None]
            yb = (torch.from_numpy(b).to(self.dev, torch.float64) @ luma)[None, None]
            blur = self._blur_torch
        else:
            np = self.np
            luma = np.asarray(LUMA, dtype=np.float64)
            ya, yb = a.astype(np.float64) @ luma, b.astype(np.float64) @ luma
            blur = self._blur
        mu_a, mu_b = blur(ya), blur(yb)
        va = blur(ya * ya) - mu_a * mu_a
        vb = blur(yb * yb) - mu_b * mu_b
        cov = blur(ya * yb) - mu_a * mu_b
        s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a * mu_a + mu_b * mu_b + c1) * (va + vb + c2))
        return float(s.mean())

    def lpips_of(self, a, b):
        torch = self.torch

        def t(x):
            return torch.from_numpy(x).to(self.dev).permute(2, 0, 1)[None].float() / 127.5 - 1.0

        with torch.no_grad():
            return float(self.lpips(t(a), t(b)).item())

    def frame(self, a, b):
        np = self.np
        d = a.astype(np.float64) - b.astype(np.float64)
        mse = float(np.mean(d * d))
        row = {"identical": mse == 0.0}
        if "psnr" in self.metrics:
            row["psnr"] = self.cap if mse == 0.0 else min(self.cap, 10.0 * math.log10(255.0 ** 2 / mse))
        if "mad" in self.metrics:
            row["mad"] = float(np.mean(np.abs(d)))
        if "ssim" in self.metrics:
            row["ssim"] = 1.0 if mse == 0.0 else self.ssim(a, b)
        if "lpips" in self.metrics and self.lpips is not None:
            row["lpips"] = 0.0 if mse == 0.0 else self.lpips_of(a, b)
        return row


def _decoder(path):
    import av

    c = av.open(path)
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    return c, c.decode(s)


def compare(scorer, pa, pb, stride):
    out = {"frames_a": 0, "frames_b": 0, "frames_compared": 0, "size_a": None, "size_b": None,
           "identical_frames": 0, "per_frame": {m: [] for m in scorer.metrics}}
    ca, fa = _decoder(pa)
    cb, fb = _decoder(pb)
    try:
        i = 0
        while True:
            a = next(fa, None)
            b = next(fb, None)
            if a is not None:
                out["frames_a"] += 1
                out["size_a"] = out["size_a"] or [a.width, a.height]
            if b is not None:
                out["frames_b"] += 1
                out["size_b"] = out["size_b"] or [b.width, b.height]
            if a is None or b is None:
                if a is None and b is None:
                    break
                continue  # count the rest of the longer clip
            if (a.width, a.height) != (b.width, b.height):
                out["error"] = f"frame size differs: {a.width}x{a.height} vs {b.width}x{b.height} (frame {i})"
                break
            if i % stride == 0:
                row = scorer.frame(a.to_ndarray(format="rgb24"), b.to_ndarray(format="rgb24"))
                out["frames_compared"] += 1
                out["identical_frames"] += 1 if row.pop("identical") else 0
                for m, v in row.items():
                    out["per_frame"][m].append(v)
            i += 1
    finally:
        ca.close()
        cb.close()
    out["mean"] = {m: (sum(v for v in vs if v is not None) / len([v for v in vs if v is not None])
                       if any(v is not None for v in vs) else None) for m, vs in out["per_frame"].items()}
    out["frame_count_differs"] = out["frames_a"] != out["frames_b"]
    return out


# --------------------------------------------------------------------------- main

def versions():
    v = {"python": sys.version.split()[0]}
    for name in ("av", "numpy", "torch", "lpips"):
        mod = sys.modules.get(name)
        if mod is not None:
            v[name] = getattr(mod, "__version__", None)
    return v


def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    with open(argv[1]) as f:
        cfg = json.load(f)
    t0 = time.time()
    items, extra = [], {}
    try:
        import av  # noqa: F401  (both tasks need PyAV and numpy: a missing one fails the run, not each clip)
        import numpy  # noqa: F401

        if cfg["task"] == "probe":
            for it in cfg.get("files", []):
                try:
                    items.append({"key": it["key"], "ok": True, **probe(it["path"])})
                except Exception as e:
                    items.append({"key": it["key"], "ok": False, "error": f"{type(e).__name__}: {e}"[:500]})
        elif cfg["task"] == "compare":
            scorer = Scorer(cfg.get("metrics") or ["psnr", "ssim", "lpips"], cfg.get("device") or "auto",
                            cfg.get("psnr_cap_db", 100.0), cfg.get("lpips_net") or "alex",
                            cfg.get("ssim_backend") or "auto")
            stride = max(1, int(cfg.get("frame_stride") or 1))
            extra = {"device": scorer.device, "metrics": scorer.metrics, "lpips_error": scorer.lpips_error,
                     "ssim_backend": scorer.ssim_backend, "frame_stride": stride, "psnr_cap_db": scorer.cap}
            for it in cfg.get("pairs", []):
                try:
                    r = compare(scorer, it["a"], it["b"], stride)
                    items.append({"key": it["key"], "ok": "error" not in r, **r})
                except Exception as e:
                    items.append({"key": it["key"], "ok": False, "error": f"{type(e).__name__}: {e}"[:500]})
        else:
            raise ValueError(f"unknown task {cfg['task']!r}")
    except ImportError as e:  # the venv lacks PyAV / numpy / torch: say which, so the caller can point elsewhere
        print(MARK + json.dumps({"ok": False, "error": f"missing module: {e}", "versions": versions()}), flush=True)
        return 1
    except Exception as e:
        print(MARK + json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"[:500], "versions": versions()}),
              flush=True)
        return 1
    with open(cfg["out"], "w") as f:
        json.dump({"items": items, "versions": versions(), **extra}, f)
    print(MARK + json.dumps({"ok": True, "n": len(items), "seconds": round(time.time() - t0, 3),
                             "versions": versions(), **extra}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
