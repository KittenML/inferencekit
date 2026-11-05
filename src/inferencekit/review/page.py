"""Render a loaded run (``rundir.Run``) as one self-contained, offline HTML page.

Page shape:
  * a summary table: per engine, speed per bucket, % of ceiling, quality and gate
    verdicts, cost, agreement with the reference and how many cells were NOT RUN;
  * filters: bucket, engine, "only rows that differ";
  * one table per bucket: rows are inputs, columns are the reference then the engines.
    LLM cells show the text with the first divergence from the reference marked and the
    word diff highlighted (token-level position when both sides carry token ids). STT
    cells show word errors against the ground truth; TTS cells show an audio player and
    the ASR transcript's word errors. Video cells show the clip in a player beside the
    reference clip (one table per round, so every clip in a row had the same seed), with
    the client-timed clip time, the engine's self-reported stage times and shape, and
    PSNR/SSIM/LPIPS when quality clip measured them. A missing output is a NOT RUN cell,
    never a blank.

Safety: every piece of model, input or file-derived text goes through ``esc``; the
script reads only data-* attributes the page wrote itself; no external URL is loaded
(audio and video links are relative to the page). Claim classes: numbers read from metrics
files are marked observed (o); numbers this page computes (medians across inputs, divergence
positions, word error rates) are marked derived (d); what an engine says about itself (a
clip's stage times, steps and shape) is marked self-reported (s).
"""

from __future__ import annotations

import html
import os
import statistics
from urllib.parse import quote

from inferencekit.review.diff import Alignment, align_words, first_token_divergence, word_diff
from inferencekit.review.rundir import ROUND_DIR, Cell, Item, Run

# Speed columns per modality: (metric key aliases, label, unit, higher is better).
SPEED = {
    "llm": [(("decode_tok_s", "tok_s", "output_tok_s"), "decode", "tok/s", True),
            (("ttft_ms", "ttft_p50_ms"), "TTFT", "ms", False)],
    "tts": [(("ttfa_ms", "ttfa_p50_ms"), "TTFA", "ms", False), (("rtf",), "RTF", "", False),
            (("first_pcm_ms",), "first PCM", "ms", False),
            (("chars_s", "chars_per_s"), "chars", "/s", True)],
    "stt": [(("rtf",), "RTF", "", False), (("first_partial_ms",), "first partial", "ms", False),
            (("final_ms", "finalization_ms"), "final", "ms", False)],
    "video": [(("clip_s",), "clip", "s", False), (("clips_per_hour",), "clips", "/h", True)],
    "image": [(("image_ms", "e2e_ms"), "image", "ms", False),
              (("images_per_hour",), "images", "/h", True)],
}
# Per-cell chips, in this order, then any other numeric field.
CHIPS = [(("ttft_ms",), "TTFT", "ms"), (("decode_tok_s", "tok_s"), "decode", "tok/s"), (("tpot_ms",), "TPOT", "ms"),
         (("e2e_ms", "total_ms", "latency_ms"), "e2e", "ms"), (("output_tokens", "completion_tokens"), "out", "tok"),
         (("ttfa_ms",), "TTFA", "ms"), (("rtf",), "RTF", ""), (("duration_s",), "dur", "s"),
         (("first_pcm_ms",), "first PCM", "ms"), (("first_signal_ms",), "first nonzero chunk", "ms"),
         (("first_audio_ms",), "simulated nonzero onset", "ms"),
         (("complete_audio_ms",), "complete audio", "ms"), (("audio_duration_s",), "PCM duration", "s"),
         (("max_stream_gap_ms",), "max stream gap", "ms"), (("underrun_ms",), "simulated stall", "ms"),
         (("max_gap_ms", "gap_ms"), "max gap", "ms"), (("underruns",), "underruns", ""), (("wer",), "WER", ""),
         (("cer",), "CER", ""), (("utmos", "mos"), "MOS", ""), (("sim", "speaker_sim"), "spk sim", ""),
         (("lufs",), "LUFS", ""), (("first_partial_ms",), "first partial", "ms"), (("final_ms",), "final", "ms")]
MAX_EXTRA_CHIPS = 4
PREVIEW_CHARS = 160


def esc(x) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def fmt(x: float | None) -> str:
    if x is None:
        return "n/a"
    if abs(x) >= 10_000:
        return f"{x:,.0f}"
    return f"{x:.4g}"


def value_of(v):
    """(value, ci95 or None) from a number or {"value"|"median"|"p50"|"mean": x, "ci95": [lo, hi]}."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v), None
    if isinstance(v, dict):
        for k in ("value", "median", "p50", "mean"):
            x = v.get(k)
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                ci = v.get("ci95") if k == "value" else v.get(f"ci95_{k}", v.get("ci95"))
                ok = isinstance(ci, list) and len(ci) == 2 and all(isinstance(c, (int, float)) for c in ci)
                return float(x), (ci if ok else None)
    return None


def lookup(d: dict, aliases) -> tuple[float, list | None] | None:
    for k in aliases:
        if k in (d or {}):
            got = value_of(d[k])
            if got is not None:
                return got
    return None


def cc(kind: str) -> str:
    title = {"o": "observed: read from the run's metrics files (measured by the benchmark client)",
             "d": "derived: computed by this page from the run's files",
             "s": "self-reported by the engine: not measured by the client and not verified"}[kind]
    return f'<abbr class="cc cc-{kind}" title="{title}">{kind}</abbr>'


VERDICT_CLASS = {"PASS": "pass", "FAIL": "fail", "UNPROVEN": "warn", "NOT RUN": "notrun"}


def badge(text: str, kind: str = "", title: str = "", verdict: str | None = None) -> str:
    """A small label; coloured by ``verdict`` (or the text itself) when it is PASS / FAIL / UNPROVEN / NOT RUN."""
    cls = VERDICT_CLASS.get(str(verdict or text).upper(), kind or "info")
    t = f' title="{esc(title)}"' if title else ""
    return f'<span class="badge b-{cls}"{t}>{esc(text)}</span>'


# --------------------------------------------------------------------------- derived comparisons

def compare_cell(run: Run, item: Item, cell: Cell) -> dict:
    """Derived facts about one cell: divergence from the reference, or word errors vs ground truth."""
    out: dict = {"input": item.id, "bucket": item.bucket, "engine": cell.engine, "status": cell.status}
    if cell.status != "ok":
        return out
    if run.modality == "image":
        return out  # Viewing a PNG does not establish semantic quality or comparative image similarity.
    if run.modality == "llm":
        ref = run.cells.get((run.reference, run.key(item))) if run.reference else None
        if cell.engine == run.reference or ref is None or ref.status != "ok" or cell.text is None or ref.text is None:
            return out
        td = word_diff(ref.text, cell.text)
        out["diff"] = td
        # Identical means the same string: the word diff ignores whitespace, so newlines, spacing and
        # trailing whitespace are checked here, and the marker goes on the first differing character.
        out["identical_text"] = ref.text == cell.text
        out["first_divergence_word"] = td.first_word
        out["reference_chars"] = len(ref.text)
        if td.identical and not out["identical_text"]:
            out["whitespace_only"] = True
            out["first_divergence_char"] = len(os.path.commonprefix([ref.text, cell.text]))
            out["marker_char"] = out["first_divergence_char"]
        if cell.token_ids is not None and ref.token_ids is not None:
            d = first_token_divergence(ref.token_ids, cell.token_ids)
            out["first_divergence_token"] = d
            out["identical_tokens"] = d is None
            out["reference_tokens"] = len(ref.token_ids)
            if d is not None and cell.pieces and "".join(cell.pieces) == cell.text:
                out["marker_char"] = len("".join(cell.pieces[:d]))
        out["identical"] = out.get("identical_tokens", out["identical_text"])
        if out["identical"]:
            out.pop("marker_char", None)  # same tokens: nothing to mark, whatever the detokenized spacing
        return out
    if run.modality == "video":
        row = cell.clip_row or {}
        mismatch = row.get("effective_mismatch")
        if isinstance(mismatch, list) and mismatch:
            out["effective_mismatch"] = [str(x) for x in mismatch]  # the engine says it ran something else
        if row.get("cache"):
            out["cached"] = True
        if row.get("stage_sum_ok") is False:
            out["stages_over_e2e"] = True
        ref = run.cells.get((run.reference, run.key(item))) if run.reference else None
        if cell.engine != run.reference and ref is not None and ref.status == "ok" and (ref.clip_row or {}).get(
                "out_sha256") and ref.clip_row.get("out_sha256") == row.get("out_sha256"):
            out["same_bytes_as_reference"] = True
        return out
    hyp = cell.text if run.modality == "stt" else cell.transcript
    if item.ground_truth and hyp is not None:
        al = align_words(item.ground_truth, hyp)
        out["alignment"] = al
        out.update({"wer": al.wer, "sub": al.count("sub"), "del": al.count("del"), "ins": al.count("ins"),
                    "ref_words": al.ref_len, "exact_alignment": al.exact})
    return out


# --------------------------------------------------------------------------- cell bodies

MARK = '<span class="fd" title="first divergence from the reference" aria-label="first divergence"></span>'


def render_diff(info: dict, text: str) -> str:
    td = info["diff"]
    m = info.get("marker_char")
    pos = len(td.lead)
    if m is None and not td.identical:
        for tag, _r, c in td.segments:  # word-level: the start of the first differing run
            if tag != "equal":
                m = pos
                break
            pos += len(c)
    lead = td.lead
    if m is not None and m <= len(lead) and (m < len(lead) or not td.segments):  # leading whitespace differs
        out, marked = [esc(lead[:m]), MARK, esc(lead[m:])], True
    else:
        out, marked = [esc(lead)], m is None
    pos = len(lead)
    for tag, r, c in td.segments:
        end = pos + len(c)
        if tag == "equal":
            if not marked and pos <= m < end:
                k = m - pos
                out += [esc(c[:k]), MARK, esc(c[k:])]
                marked = True
            else:
                out.append(esc(c))
        else:
            if not marked and m == pos:
                out.append(MARK)
                marked = True
            if r:
                out.append(f'<del title="only in the reference">{esc(r)}</del>')
            if c:
                if not marked and pos < m < end:
                    k = m - pos
                    out.append(f"<ins>{esc(c[:k])}</ins>{MARK}<ins>{esc(c[k:])}</ins>")
                    marked = True
                else:
                    out.append(f"<ins>{esc(c)}</ins>")
        pos = end
    if not marked:
        out.append(MARK)
    return "".join(out)


def render_alignment(al: Alignment) -> str:
    parts = []
    for op, r, h in al.ops:
        if op == "ok":
            parts.append(esc(h))
        elif op == "sub":
            parts.append(f'<span class="w-sub" title="substitution; ground truth: {esc(r)}">{esc(h)}</span>')
        elif op == "ins":
            parts.append(f'<span class="w-ins" title="insertion">{esc(h)}</span>')
        else:
            parts.append(f'<span class="w-del" title="deletion: missing from the output">{esc(r)}</span>')
    return " ".join(parts)


def audio_tag(rel: str) -> str:
    return f'<audio controls preload="none" src="../{esc(quote(rel))}"></audio>'


def video_tag(rel: str) -> str:
    return f'<video controls muted playsinline preload="metadata" src="../{esc(quote(rel))}"></video>'


def image_tag(rel: str) -> str:
    return (f'<a href="../{esc(quote(rel))}"><img class="generated" loading="lazy" src="../{esc(quote(rel))}" '
            'alt="Generated image; open for full resolution"></a>')


def _secs(ms) -> str:
    return f"{fmt(ms / 1000)} s"


def clip_chips(cell: Cell) -> str:
    """Video cell chips: client-timed clip time (observed), the engine's stage times and effective shape
    (self-reported), then the frame-quality numbers quality clip measured against the reference (observed)."""
    m, row = cell.metrics, cell.clip_row or {}
    observed, reported = [], []
    got = value_of(m.get("clip_s")) if "clip_s" in m else None
    if got is not None:
        observed.append(f'<span class="chip">clip {esc(fmt(got[0]))} s</span>')
    for keys, label, unit in ((("psnr", "psnr_db"), "PSNR", " dB"), (("ssim",), "SSIM", ""),
                              (("lpips",), "LPIPS", "")):
        key = next((k for k in keys if k in m), None)
        got = value_of(m.get(key)) if key else None
        if got is not None:
            observed.append(f'<span class="chip">{label} {esc(fmt(got[0]))}{unit}</span>')
    stages = row.get("stage_ms") if isinstance(row.get("stage_ms"), dict) else {}
    for name, v in list(stages.items())[:6]:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            reported.append(f'<span class="chip">{esc(name)} {esc(_secs(v))}</span>')
    shape = [f"{row['w']}x{row['h']}" if row.get("w") and row.get("h") else "",
             f"{row['frames']} frames" if row.get("frames") else "", f"{row['fps']} fps" if row.get("fps") else "",
             f"{row['steps_run']} steps" if row.get("steps_run") else "", str(row.get("attention") or ""),
             str(row.get("dtype") or "")]
    if any(shape):
        reported.append(f'<span class="chip">{esc(", ".join(x for x in shape if x))}</span>')
    out = f'<div class="chips">{"".join(observed)}{cc("o")}</div>' if observed else ""
    return out + (f'<div class="chips">{"".join(reported)}{cc("s")}</div>' if reported else "")


def clip_spec_text(clip: dict) -> str:
    """What a clip request asked for, in one line: task, size, frames, fps, steps, CFG, seed, image."""
    parts = [str(clip.get("task") or "")]
    if clip.get("width") and clip.get("height"):
        parts.append(f"{clip['width']}x{clip['height']}")
    if clip.get("num_frames"):
        parts.append(f"{clip['num_frames']} frames" + (f" @ {clip['fps']} fps" if clip.get("fps") else ""))
    if clip.get("steps"):
        parts.append(f"{clip['steps']} steps")
    if clip.get("guidance_scale") is not None:
        parts.append(f"CFG {clip['guidance_scale']}")
    if clip.get("seed") is not None:
        parts.append(f"seed {clip['seed']}")
    if clip.get("audio") is not None:
        parts.append("audio" if clip["audio"] else "no audio")
    if clip.get("image_sha256"):
        parts.append(f"image sha256 {str(clip['image_sha256'])[:12]}")
    return " \u00b7 ".join(p for p in parts if p)


def chips(metrics: dict, derived=()) -> str:
    shown, computed, used = [], [], set()
    for aliases, label, unit in CHIPS:
        for k in aliases:
            got = value_of(metrics.get(k)) if k in metrics else None
            if got is not None:
                target = computed if k in derived else shown
                target.append(f'<span class="chip">{esc(label)} {esc(fmt(got[0]))}{esc(unit and " " + unit)}</span>')
                used.update(aliases)
                break
    extra = 0
    for k, v in metrics.items():
        if k in used or extra >= MAX_EXTRA_CHIPS:
            continue
        got = value_of(v)
        if got is not None:
            target = computed if k in derived else shown
            target.append(f'<span class="chip">{esc(k)} {esc(fmt(got[0]))}</span>')
            extra += 1
    return ((f'<div class="chips">{"".join(shown)}{cc("o")}</div>' if shown else "") +
            (f'<div class="chips">{"".join(computed)}{cc("d")}</div>' if computed else ""))


def quality_badges(metrics: dict) -> list[str]:
    out = []
    v = metrics.get("verdict")
    if isinstance(v, str):
        out.append(badge(v, title="per-input verdict from the metrics file"))
    if isinstance(metrics.get("pass"), bool):
        out.append(badge("PASS" if metrics["pass"] else "FAIL", title="per-input check from the metrics file"))
    q = metrics.get("quality")
    if isinstance(q, dict):
        for name, val in list(q.items())[:4]:
            verdict = val.get("verdict") if isinstance(val, dict) else val
            if isinstance(verdict, str):
                out.append(badge(f"{name}: {verdict}", title=f"quality check '{name}' from the metrics file",
                                 verdict=verdict))
    return out


def severity(cell: Cell, info: dict) -> float:
    """How bad a cell looks, for 'worst first': 2 = failed / not run, 1+ = diverged early / word errors."""
    if cell.status != "ok":
        return 2.0
    if info.get("effective_mismatch") or info.get("cached"):
        return 1.9
    if info.get("identical") is False:
        if info.get("first_divergence_token") is not None:
            d, n = info["first_divergence_token"], info.get("reference_tokens") or 1
        elif info.get("first_divergence_word") is not None:
            d, n = info["first_divergence_word"], info["diff"].ref_words or 1
        else:  # whitespace only: by character
            d, n = info.get("first_divergence_char") or 0, info.get("reference_chars") or 1
        return 1.0 + max(0.0, 1.0 - d / n) * 0.99  # earlier divergence = worse
    if info.get("wer"):
        return 1.0 + min(info["wer"], 1.0) * 0.99
    return 0.0


def cell_html(run: Run, item: Item, cell: Cell, info: dict, idx: int) -> str:
    head = f'<div class="cell-head">{esc(cell.engine)}</div>'
    if cell.status == "not_run":
        return (f'<td class="cell st-notrun" data-e="{idx}">{head}{badge("NOT RUN")}'
                f'<div class="why">{esc(cell.reason)}</div></td>')
    badges = []
    if cell.status == "failed":
        badges.append(badge("FAILED", "fail", cell.reason))
    body = f'<div class="why">{esc(cell.reason)}</div>' if cell.status == "failed" and cell.reason else ""
    if run.modality == "llm":
        if cell.engine == run.reference:
            badges.append(badge("reference", "info"))
        elif "diff" in info:
            if info.get("identical"):
                badges.append(badge("identical", "pass", "same tokens as the reference" if "identical_tokens" in info
                                    else "same text as the reference (no token ids to compare)") + cc("d"))
            elif info.get("whitespace_only") and info.get("first_divergence_token") is None:
                badges.append(badge(f"whitespace differs @ char {info['first_divergence_char']}", "warn",
                                    "same words as the reference, but spacing, newlines or trailing whitespace "
                                    "differ: not identical") + cc("d"))
            elif info.get("first_divergence_token") is not None:
                d, n = info["first_divergence_token"], info.get("reference_tokens")
                note = "same text, " if info.get("identical_text") else ""
                badges.append(badge(f"{note}diverges @ token {d}" + (f"/{n}" if n else ""), "warn",
                                    "first token id that differs from the reference") + cc("d"))
            else:
                badges.append(badge(f"diverges @ word {info['first_divergence_word']}", "warn",
                                    "no token ids on both sides: first differing word") + cc("d"))
        elif run.reference and cell.text is not None:
            badges.append(badge("no reference output", "notrun"))
        if cell.reasoning:
            body += (f'<details class="reasoning"><summary>reasoning ({len(cell.reasoning):,} chars, not compared)'
                     f'</summary><div class="text">{esc(cell.reasoning)}</div></details>')
        if cell.text is not None:
            inner = render_diff(info, cell.text) if "diff" in info else esc(cell.text)
            body += f'<div class="text">{inner}</div>' if cell.text else '<div class="why">(empty output)</div>'
        elif cell.token_ids is not None:
            body += f'<div class="why">{len(cell.token_ids)} token ids, no text</div>'
    elif run.modality == "image":
        if cell.engine == run.reference:
            badges.append(badge("reference", "info"))
        if cell.image:
            body += image_tag(cell.image)
    elif run.modality == "video":
        if cell.engine == run.reference:
            badges.append(badge("reference", "info"))
        if info.get("effective_mismatch"):
            badges.append(badge("ran something else", "fail", "the engine reports: " + "; ".join(
                info["effective_mismatch"])) + cc("s"))
        if info.get("cached"):
            badges.append(badge("cached stages", "fail", "the engine reports stages served from a cache") + cc("s"))
        if info.get("stages_over_e2e"):
            badges.append(badge("stages > clip time", "warn", "self-reported stage times sum to more than the "
                                "client-timed clip: overlapping stages or wrong self-reports") + cc("d"))
        if info.get("same_bytes_as_reference"):
            badges.append(badge("same bytes as the reference", "info", "the clip file is byte-identical to the "
                                "reference clip") + cc("d"))
        if cell.video:
            body += video_tag(cell.video)
        elif cell.text is not None:
            body += f'<div class="text">{esc(cell.text)}</div>'
    else:
        if cell.audio:
            body += audio_tag(cell.audio)
        if "alignment" in info:
            wer = info["wer"]
            label = f"WER {100 * wer:.1f}%" if wer is not None else "WER n/a"
            badges.append(badge(label, "pass" if wer == 0 else "warn",
                                f"S {info['sub']} / D {info['del']} / I {info['ins']} over {info['ref_words']} words"
                                + ("" if info["exact_alignment"] else " (anchored alignment)")) + cc("d"))
            body += f'<div class="text">{render_alignment(info["alignment"])}</div>'
        elif run.modality == "stt" and cell.text is not None:
            body += f'<div class="text">{esc(cell.text)}</div>'
        elif run.modality == "tts" and cell.audio:
            body += '<div class="why">no ASR transcript for this clip</div>'
        elif cell.text is not None:
            body += f'<div class="text">{esc(cell.text)}</div>'
    if cell.finish_reason and cell.finish_reason not in ("stop", "length"):
        badges.append(badge(f"finish: {cell.finish_reason}", "warn"))
    badges += quality_badges(cell.metrics)
    status = "st-failed" if cell.status == "failed" else ("st-diff" if info.get("identical") is False
                                                            or (info.get("wer") or 0) > 0
                                                            or info.get("effective_mismatch") else "st-ok")
    from inferencekit.review.media import DERIVED_AUDIO

    numbers = clip_chips(cell) if run.modality == "video" else chips(
        cell.metrics, DERIVED_AUDIO if run.engine(cell.engine).media_kind == "audio" else ())
    return (f'<td class="cell {status}" data-e="{idx}">{head}<div class="badges">{"".join(badges)}</div>'
            f'{numbers}{body}</td>')


def input_html(run: Run, item: Item) -> str:
    parts = [f'<div class="iid">{esc(item.id)}</div>']
    if item.audio:
        parts.append(audio_tag(item.audio))
    if item.clip:
        parts.append(f'<div class="spec">{esc(clip_spec_text(item.clip))}</div>')
    if run.modality == "video":
        parts.append('<button type="button" class="play" data-play="1" title="play every clip in this row from the '
                     'start, together">play row</button>')
    text = item.text
    if run.modality == "stt" and item.audio and text and text.strip() == (item.ground_truth or "").strip():
        text = None  # the .txt beside the audio is its ground truth, shown once below
    if text:
        if len(text) > PREVIEW_CHARS:
            parts.append(f'<details><summary>{esc(text[:PREVIEW_CHARS])}&hellip; '
                         f'<span class="more">({len(text):,} chars)</span></summary>'
                         f'<div class="text">{esc(text)}</div></details>')
        else:
            parts.append(f'<div class="text">{esc(text)}</div>')
    elif not item.audio and not item.clip:
        parts.append('<div class="why">input file not found under inputs/</div>')
    if item.ground_truth and run.modality == "stt":
        parts.append(f'<div class="gt"><b>ground truth:</b> {esc(item.ground_truth)}</div>')
    return f'<th scope="row" class="input">{"".join(parts)}</th>'


# --------------------------------------------------------------------------- summary

def speed_buckets(run: Run) -> list[str]:
    """Summary speed columns: one per bucket, or for video one over all rounds (a round is a seed, not a size)."""
    return ["all"] if run.modality == "video" else run.buckets()


def bucket_title(run: Run, b: str) -> str:
    if b == "all" and run.modality == "video":
        return "all rounds"
    if run.modality in ("video", "image", "tts") and ROUND_DIR.match(b):
        return f"round {b[1:]}"
    return b.replace("_", " ")


def engine_summary(run: Run, label: str, infos: dict) -> dict:
    eng = run.engine(label)
    cells = [(item, run.cells[(label, run.key(item))]) for item in run.items]
    s = {"engine": label, "role": eng.role, "inputs": len(cells),
         "ok": sum(1 for _, c in cells if c.status == "ok"),
         "failed": sum(1 for _, c in cells if c.status == "failed"),
         "not_run": sum(1 for _, c in cells if c.status == "not_run"), "speed": {}, "evidence": []}
    for b in speed_buckets(run):
        row = {}
        for aliases, name, unit, _higher in SPEED.get(run.modality, SPEED["llm"]):
            got = lookup(eng.per_bucket.get(b, {}), aliases)
            if got is not None:
                claim = next((eng.metric_classes[k] for k in aliases if k in eng.metric_classes), "observed")
                row[name] = {"value": got[0], "ci95": got[1], "unit": unit, "claim_class": claim}
                continue
            vals = [lookup(c.metrics, aliases) for item, c in cells
                    if item.bucket == b or (b == "all" and run.modality == "video")]
            vals = [v[0] for v in vals if v is not None]
            if vals:
                row[name] = {"value": statistics.median(vals), "ci95": None, "unit": unit, "n": len(vals),
                             "claim_class": "derived"}
        s["speed"][b] = row
    m = eng.metrics
    pct = value_of(m.get("pct_of_ceiling"))
    ceiling = value_of(m.get("ceiling_tok_s"))
    if pct is not None:
        s["pct_of_ceiling"] = {"value": pct[0], "claim_class": "observed"}
    elif ceiling is not None and ceiling[0] > 0:
        overall = lookup(m, ("decode_tok_s",)) or lookup(eng.per_bucket.get("all", {}), ("decode_tok_s",))
        if overall is None:
            vals = [lookup(c.metrics, ("decode_tok_s",)) for _, c in cells]
            vals = [v[0] for v in vals if v is not None]
            overall = (statistics.median(vals), None) if vals else None
        if overall is not None:
            s["pct_of_ceiling"] = {"value": 100 * overall[0] / ceiling[0], "claim_class": "derived"}
    s["quality"] = _verdicts(m.get("quality"))
    gates = m.get("gates")
    if isinstance(gates, str):
        s["gates"] = {"verdict": gates, "violations": []}
    elif isinstance(gates, dict):
        s["gates"] = {"verdict": str(gates.get("verdict") or "UNPROVEN"),
                      "violations": [str(v) for v in gates.get("violations") or []]}
    cost = m.get("cost")
    got = value_of(cost) if not isinstance(cost, dict) else lookup(
        cost, ("usd_per_1m_output_tokens", "usd_per_1m", "usd_per_1m_tokens", "usd_per_1m_chars",
               "usd_per_audio_hour"))
    if got is not None:
        unit = "$ / 1M output tokens" if run.modality == "llm" else "$ per unit"
        if isinstance(cost, dict) and isinstance(cost.get("unit"), str):
            unit = cost["unit"]
        s["cost"] = {"value": got[0], "unit": unit, "claim_class": "observed",
                     "gpu_price_usd_h": cost.get("gpu_price_usd_h") if isinstance(cost, dict) else None}
    ev_ids = m.get("evidence")
    if isinstance(ev_ids, list):
        s["evidence"] = [str(e) for e in ev_ids][:20]
    # Verdicts from evidence records (quality / gates / capacity records) complement the metrics file.
    for q in eng.ev_quality:
        s["quality"].append({**q, "claim_class": "derived" if q.get("verdict") else "observed"})
    if "gates" not in s and eng.ev_gates:
        s["gates"] = {"verdict": eng.ev_gates["verdict"], "violations": eng.ev_gates["violations"]}
    cap = eng.capacity
    if cap:
        s["capacity"] = {"n_star": cap.get("n_star"), "reason": cap.get("reason"), "level_kind": cap.get("level_kind"),
                         "slo": cap.get("slo"), "claim_class": "derived", "edited": cap.get("edited")}
        if "cost" not in s and cap.get("usd_per_1m") is not None and not cap.get("edited"):
            s["cost"] = {"value": cap["usd_per_1m"], "unit": "$ / 1M output tokens at N*", "claim_class": "derived",
                         "gpu_price_usd_h": cap.get("price")}
    s["evidence"] = list(dict.fromkeys(s["evidence"] + eng.evidence))[:20]
    if eng.stages:
        s["stages_ms"] = {"value": dict(eng.stages), "claim_class": "self-reported"}
    mine = [infos[(label, run.key(item))] for item in run.items]
    if run.modality == "llm" and label != run.reference and run.reference:
        compared = [i for i in mine if "identical" in i]
        s["vs_reference"] = {"compared": len(compared), "identical": sum(1 for i in compared if i["identical"]),
                             "claim_class": "derived"}
        tok = [i["first_divergence_token"] for i in compared if i.get("first_divergence_token") is not None]
        if tok:
            s["vs_reference"]["median_first_divergence_token"] = statistics.median(tok)
    errs = [i for i in mine if i.get("wer") is not None]
    if errs:
        words = sum(i["ref_words"] for i in errs)
        wrong = sum(i["sub"] + i["del"] + i["ins"] for i in errs)
        s["wer"] = {"value": wrong / words if words else None, "clips": len(errs), "claim_class": "derived"}
    return s


def _verdicts(q) -> list[dict]:
    out = []
    items = q.items() if isinstance(q, dict) else ((x.get("name"), x) for x in q if isinstance(x, dict)) \
        if isinstance(q, list) else []
    for name, val in items:
        if isinstance(val, str):
            out.append({"name": str(name), "verdict": val})
        elif isinstance(val, dict):
            got = value_of(val)
            out.append({"name": str(name or val.get("name") or "quality"),
                        "verdict": str(val.get("verdict") or "UNPROVEN"),
                        "value": got[0] if got else None, "unit": val.get("unit")})
    return out


def _quality_badge(x: dict) -> str:
    value = f"{fmt(x['value'])}{x['unit'] if x.get('unit') == '%' else (' ' + x['unit'] if x.get('unit') else '')}" \
        if x.get("value") is not None else ""
    if x.get("verdict"):
        return badge(f"{x['name']}: {x['verdict']}", verdict=x["verdict"],
                     title=f"value {value}" if value else "") + (cc("d") if x.get("evidence") else "")
    return badge(f"{x['name']} {value or 'n/a'}", "info", "measured; no verdict") + cc("o")


def _speed_cell(row: dict) -> str:
    if not row:
        return '<span class="na">not measured</span>'
    parts = []
    for name, v in row.items():
        ci = v.get("ci95")
        ci_text = f' <span class="ci">({fmt(ci[0])}&ndash;{fmt(ci[1])})</span>' if ci else ""
        unit = f" {v['unit']}" if v["unit"] else ""
        n = f' <span class="ci">(median of {v["n"]})</span>' if v.get("n") else ""
        parts.append(f'<div>{esc(name)} <b>{esc(fmt(v["value"]))}</b>{esc(unit)}{ci_text}{n}'
                     f'{cc("o" if v["claim_class"] == "observed" else "d")}</div>')
    return "".join(parts)


NO_EVIDENCE = ("no evidence attached", ("no evidence records were attached to this page: pass --evidence <id> "
                                        "(gates, quality, capacity records) or copy them into <run-dir>/evidence/"))


def capacity_cell(cap: dict | None, absent: str) -> str:
    if not cap:
        return f"<td>{absent}</td>"
    if cap.get("edited"):
        return f"<td>{badge('UNPROVEN', title='capacity evidence was edited after capture')}</td>"
    unit = "req/s" if cap.get("level_kind") == "rate" else "streams"
    reason = cap.get("reason") or ""
    if cap.get("n_star") is None:
        if reason.startswith("SLO first fails"):  # missed at the lowest level tried
            return (f"<td>{badge('FAIL', title=reason)} <span class='sub'>SLO missed at every level "
                    f"tried</span></td>")
        verdict = "NOT RUN" if reason in ("no load levels ran", "") else "UNPROVEN"
        return f"<td>{badge(verdict, title=reason or 'N* not computed')} <span class='sub'>{esc(reason)}</span></td>"
    slo = cap.get("slo") or {}
    slo_txt = ", ".join(f"{esc(k)} &le; {esc(fmt(v))}" for k, v in slo.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool))
    lower = "at least " if reason.startswith("SLO holds at every level") else ""
    return (f"<td>{lower}<b>{esc(fmt(cap['n_star']))}</b> {unit}{cc('d')}"
            f"<div class='sub'>{slo_txt or 'no SLO recorded'}</div></td>")


def summary_html(run: Run, summaries: list[dict]) -> str:
    buckets = speed_buckets(run)
    absent = (badge(NO_EVIDENCE[0], "notrun", NO_EVIDENCE[1]) if not run.evidence_attached else badge("NOT RUN"))
    head = ["engine"] + [bucket_title(run, b) for b in buckets] + ["% of ceiling", "quality", "gates", "cost"]
    has_cap = any("capacity" in s for s in summaries)
    if has_cap:
        head.insert(head.index("cost"), "N*")
    if run.modality == "llm" and run.reference:
        head.append("vs reference")
    if any("wer" in s for s in summaries):
        head.append("WER")
    head.append("outputs")
    rows = []
    for idx, s in enumerate(summaries):
        cells = [f'<th scope="row">{esc(s["engine"])}' + (f' <span class="role">{esc(s["role"])}</span>'
                                                         if s.get("role") else "") + "</th>"]
        stages = ""
        if s.get("stages_ms"):  # video: the engine's own stage medians, under the client-timed clip time
            stages = ("<div class='sub'>stages: " + ", ".join(f"{esc(k)} {esc(_secs(v))}" for k, v in
                                                               list(s["stages_ms"]["value"].items())[:6])
                      + f"{cc('s')}</div>")
        cells += [f"<td>{_speed_cell(s['speed'].get(b, {}))}{stages}</td>" for b in buckets]
        pct = s.get("pct_of_ceiling")
        cells.append(f"<td>{esc(fmt(pct['value']))}%{cc('o' if pct['claim_class'] == 'observed' else 'd')}</td>"
                     if pct else '<td><span class="na">no ceiling</span></td>')
        q = s["quality"]
        cells.append("<td>" + ("".join(_quality_badge(x) for x in q) if q else absent) + "</td>")
        g = s.get("gates")
        cells.append("<td>" + (badge(g["verdict"], title="; ".join(g["violations"])) if g else absent) + "</td>")
        if has_cap:
            cells.append(capacity_cell(s.get("capacity"), absent))
        c = s.get("cost")
        cells.append(f"<td>${esc(fmt(c['value']))} <span class='unit'>{esc(c['unit'].lstrip('$ '))}</span>"
                     f"{cc('o' if c.get('claim_class', 'observed') == 'observed' else 'd')}</td>" if c
                     else '<td><span class="na">unknown</span></td>')
        if run.modality == "llm" and run.reference:
            vr = s.get("vs_reference")
            if s["engine"] == run.reference:
                cells.append("<td>(reference)</td>")
            elif vr and vr["compared"]:
                med = vr.get("median_first_divergence_token")
                cells.append(f"<td>{vr['identical']}/{vr['compared']} identical"
                             + (f"<div class='sub'>median first divergence: token {esc(fmt(med))}</div>"
                                if med is not None else "") + f"{cc('d')}</td>")
            else:
                cells.append('<td><span class="na">nothing to compare</span></td>')
        if any("wer" in x for x in summaries):
            w = s.get("wer")
            cells.append(f"<td>{100 * w['value']:.2f}% <span class='sub'>({w['clips']} clips)</span>{cc('d')}</td>"
                         if w and w["value"] is not None else '<td><span class="na">n/a</span></td>')
        nr = f"<div class='bad'>{s['not_run']} NOT RUN</div>" if s["not_run"] else ""
        fl = f"<div class='bad'>{s['failed']} failed</div>" if s["failed"] else ""
        cells.append(f"<td>{s['ok']}/{s['inputs']}{nr}{fl}</td>")
        rows.append(f'<tr data-e="{idx}">{"".join(cells)}</tr>')
    ths = "".join(f'<th scope="col">{esc(h)}</th>' for h in head)
    return (f'<div class="scroll"><table class="summary"><thead><tr>{ths}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


# --------------------------------------------------------------------------- page

CSS = """
:root{--bg:#fbfbfa;--panel:#fff;--fg:#1d1f23;--muted:#5d636e;--line:#dfe1e5;--accent:#2457c5;
--ok:#1b7a3d;--okbg:#e3f4e8;--bad:#b42318;--badbg:#fde8e6;--warn:#8a5a00;--warnbg:#fdf1d7;
--na:#6b7280;--nabg:#eceef1;--ins:#d7f2dd;--del:#fadcd9;--sub:#fdeab8;--mark:#d03b2f}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--panel:#1a1d21;--fg:#e6e8eb;--muted:#9aa1ab;
--line:#2d3238;--accent:#7aa2ff;--ok:#5fd38a;--okbg:#173322;--bad:#ff8a80;--badbg:#3a1a18;--warn:#f2c46b;
--warnbg:#3a2e12;--na:#a3a9b2;--nabg:#262a30;--ins:#1d4429;--del:#4a2220;--sub:#4a3a12;--mark:#ff7a6e}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",
Roboto,sans-serif}
header,main,footer,.bar{padding:0 16px;max-width:100%}
header{padding-top:16px}
h1{font-size:20px;margin:0 0 4px}
h2{font-size:16px;margin:24px 0 8px}
.meta{color:var(--muted);font-size:13px}
.meta span{margin-right:12px;display:inline-block}
.warnings{font-size:13px;color:var(--warn);margin:8px 0;padding-left:18px}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;border:1px solid var(--line);border-radius:8px;
background:var(--panel)}
table{border-collapse:collapse;width:100%}
th,td{border-bottom:1px solid var(--line);padding:8px;vertical-align:top;text-align:left}
thead th{font-size:12px;color:var(--muted);font-weight:600;background:var(--panel);white-space:nowrap}
table.summary td,table.summary th{font-size:13px;white-space:nowrap}
.role{font-size:11px;color:var(--muted);font-weight:400;display:block}
.ci,.sub,.unit,.more{color:var(--muted);font-size:12px}
.na{color:var(--na);font-style:italic}
.bad{color:var(--bad);font-size:12px}
.bar{position:sticky;top:0;z-index:2;background:var(--bg);padding-top:8px;padding-bottom:8px;
border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center}
.bar .grp{display:flex;flex-wrap:wrap;gap:4px;align-items:center}
.bar .lbl{font-size:12px;color:var(--muted);margin-right:2px}
.bar label{font-size:12px;border:1px solid var(--line);border-radius:999px;padding:2px 9px;background:var(--panel);
cursor:pointer;user-select:none;white-space:nowrap}
.bar input{margin:0 4px 0 0;vertical-align:-1px}
table.grid{table-layout:fixed;min-width:calc(220px + var(--cols) * 260px)}
table.grid col.c-in{width:220px}
th.input{font-weight:400;background:var(--panel)}
.iid{font:600 12px ui-monospace,SFMono-Regular,Menlo,monospace;margin-bottom:4px;word-break:break-all}
.text{white-space:pre-wrap;word-wrap:break-word;overflow-wrap:anywhere;max-height:22em;overflow-y:auto;
font-size:13px}
details summary{cursor:pointer;white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}
details.reasoning{margin-bottom:6px}details.reasoning summary{font-size:12px;color:var(--muted)}
.gt{font-size:12px;color:var(--muted);margin-top:6px;overflow-wrap:anywhere}
.cell-head{display:none;font-weight:600;font-size:12px;color:var(--muted);margin-bottom:4px}
.badges{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:4px;align-items:center}
.badge{display:inline-block;font-size:11px;font-weight:600;border-radius:4px;padding:1px 6px;white-space:nowrap}
.b-pass{background:var(--okbg);color:var(--ok)}.b-fail{background:var(--badbg);color:var(--bad)}
.b-warn{background:var(--warnbg);color:var(--warn)}.b-notrun{background:var(--nabg);color:var(--na)}
.b-info{background:var(--nabg);color:var(--fg)}
.chips{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:6px;align-items:center}
.chip{font-size:11px;border:1px solid var(--line);border-radius:4px;padding:0 5px;color:var(--muted);
white-space:nowrap}
.cc{font-size:9px;text-decoration:none;border:1px solid var(--line);border-radius:3px;padding:0 3px;
margin-left:3px;color:var(--muted);vertical-align:1px;cursor:help}
.cc-d{border-style:dashed}
.why{color:var(--na);font-size:12px;font-style:italic}
td.st-notrun{background:repeating-linear-gradient(135deg,transparent 0 8px,var(--nabg) 8px 9px)}
td.st-failed{box-shadow:inset 3px 0 0 var(--bad)}
td.st-diff{box-shadow:inset 3px 0 0 var(--warn)}
ins{background:var(--ins);text-decoration:none;border-radius:2px}
del{background:var(--del);color:var(--muted);border-radius:2px}
.fd{display:inline-block;width:0;height:1.1em;border-left:2px solid var(--mark);vertical-align:-0.2em;
margin:0 1px;position:relative}
.fd::before{content:"";position:absolute;top:-4px;left:-5px;border:4px solid transparent;
border-top-color:var(--mark)}
.w-sub{background:var(--sub);border-radius:2px}
.w-ins{background:var(--ins);border-radius:2px}
.w-del{background:var(--del);text-decoration:line-through;border-radius:2px}
audio{width:100%;max-width:320px;height:32px;display:block;margin:2px 0 6px}
video{width:100%;max-width:480px;display:block;margin:2px 0 6px;background:#000;border-radius:4px}
img.generated{width:100%;max-width:480px;height:auto;display:block;margin:2px 0 6px;border-radius:4px}
.spec{font-size:12px;color:var(--muted);margin:4px 0;overflow-wrap:anywhere}
button.play{font:12px system-ui,sans-serif;border:1px solid var(--line);border-radius:999px;padding:2px 10px;
background:var(--panel);color:var(--fg);cursor:pointer;margin:4px 0}
.cc-s{border-style:dotted}
section.bucket{content-visibility:auto;contain-intrinsic-size:auto 1200px}
.legend{font-size:12px;color:var(--muted);display:flex;flex-wrap:wrap;gap:4px 14px;margin:8px 0}
footer{color:var(--muted);font-size:12px;padding-top:16px;padding-bottom:32px;overflow-wrap:anywhere}
footer code{font-size:11px}
[hidden]{display:none!important}
@media (max-width:760px){
 table.grid{min-width:0;table-layout:auto}
 table.grid colgroup,table.grid thead{display:none}
 table.grid,table.grid tbody,table.grid tr,table.grid th,table.grid td{display:block;width:100%}
 table.grid tr{border-bottom:2px solid var(--line)}
 table.grid th.input{border-bottom:1px dashed var(--line)}
 .cell-head{display:block}
 .text{max-height:16em}
 .bar{position:static}
}
"""

JS = """
(function(){
  var hide={b:{},e:{}},onlyDiff=false;
  function apply(){
    document.querySelectorAll('section.bucket').forEach(function(s){s.hidden=!!hide.b[s.dataset.b];});
    document.querySelectorAll('[data-e]').forEach(function(el){el.hidden=!!hide.e[el.dataset.e];});
    document.querySelectorAll('tr.item').forEach(function(r){r.hidden=onlyDiff&&r.dataset.flag!=='1';});
  }
  document.querySelectorAll('input[data-fb]').forEach(function(cb){cb.addEventListener('change',function(){
    hide.b[cb.dataset.fb]=!cb.checked;apply();});});
  document.querySelectorAll('input[data-fe]').forEach(function(cb){cb.addEventListener('change',function(){
    hide.e[cb.dataset.fe]=!cb.checked;apply();});});
  var d=document.getElementById('only-diff');
  if(d){d.addEventListener('change',function(){onlyDiff=d.checked;apply();});}
  document.querySelectorAll('button[data-play]').forEach(function(b){b.addEventListener('click',function(){
    var row=b.closest('tr');if(!row){return;}
    row.querySelectorAll('video').forEach(function(v){try{v.currentTime=0;}catch(e){}var p=v.play();
      if(p&&p.catch){p.catch(function(){});}});});});
  var w=document.getElementById('worst-first');
  if(w){w.addEventListener('change',function(){
    document.querySelectorAll('table.grid tbody').forEach(function(tb){
      var rows=Array.prototype.slice.call(tb.querySelectorAll('tr.item'));
      rows.sort(function(a,b){return w.checked?(parseFloat(b.dataset.sev)-parseFloat(a.dataset.sev))||
        (parseInt(a.dataset.i,10)-parseInt(b.dataset.i,10)):(parseInt(a.dataset.i,10)-parseInt(b.dataset.i,10));});
      rows.forEach(function(r){tb.appendChild(r);});
    });});}
})();
"""


def render(run: Run, generated: str, tool_version: str) -> tuple[str, dict]:
    """(html, data): the page and a machine-readable copy of what it shows."""
    infos = {(info.label, run.key(item)): compare_cell(run, item, run.cells[(info.label, run.key(item))])
             for item in run.items for info in run.engines}
    summaries = [engine_summary(run, e.label, infos) for e in run.engines]
    buckets = run.buckets()
    labels = [e.label for e in run.engines]

    filters = ['<div class="grp"><span class="lbl">Buckets</span>']
    for bi, b in enumerate(buckets):
        n = sum(1 for i in run.items if i.bucket == b)
        filters.append(f'<label><input type="checkbox" checked data-fb="{bi}">{esc(bucket_title(run, b))} '
                       f'({n})</label>')
    filters.append('</div><div class="grp"><span class="lbl">Engines</span>')
    for ei, label in enumerate(labels):
        filters.append(f'<label><input type="checkbox" checked data-fe="{ei}">{esc(label)}</label>')
    filters.append('</div><div class="grp"><label><input type="checkbox" id="only-diff">only rows that differ, '
                   'fail or were not run</label><label><input type="checkbox" id="worst-first">worst first</label>'
                   '</div>')

    sections = []
    for bi, b in enumerate(buckets):
        rows = []
        for item in (i for i in run.items if i.bucket == b):
            tds, flag, sev = [], False, 0.0
            for ei, label in enumerate(labels):
                cell = run.cells[(label, run.key(item))]
                info = infos[(label, run.key(item))]
                if (cell.status != "ok" or info.get("identical") is False or (info.get("wer") or 0) > 0
                        or info.get("effective_mismatch") or info.get("cached")):
                    flag = True  # a missing reference output also leaves the row unchecked
                sev = max(sev, severity(cell, info))
                tds.append(cell_html(run, item, cell, info, ei))
            rows.append(f'<tr class="item" data-i="{len(rows)}" data-flag="{1 if flag else 0}" data-sev="{sev:.4f}">'
                        f'{input_html(run, item)}{"".join(tds)}</tr>')
        cols = '<col class="c-in">' + "<col>" * len(labels)
        ths = '<th scope="col">input</th>' + "".join(
            f'<th scope="col" data-e="{ei}">{esc(label)}' + (" (reference)" if label == run.reference
                                                              and label != "reference" else "")
            + "</th>" for ei, label in enumerate(labels))
        n = len(rows)
        seeds = sorted({str(i.clip["seed"]) for i in run.items if i.bucket == b and i.clip
                        and i.clip.get("seed") is not None}) if run.modality == "video" else []
        seed = f", seed {', '.join(seeds[:3])}" if seeds else ""
        sections.append(f'<section class="bucket" data-b="{bi}"><h2>{esc(bucket_title(run, b))} '
                        f'<span class="sub">({n} input{"s" if n != 1 else ""}{esc(seed)})</span></h2>'
                        f'<div class="scroll"><table class="grid" style="--cols:{len(labels)}"><colgroup>{cols}'
                        f'</colgroup><thead><tr>{ths}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
                        f'</section>')
    if not run.items:
        sections.append('<p class="na">No inputs or outputs found in this run directory.</p>')

    name = run.root.resolve().name
    warnings = "".join(f"<li>{esc(w)}</li>" for w in run.warnings[:30])
    if len(run.warnings) > 30:
        warnings += f"<li>... and {len(run.warnings) - 30} more (see review/data.json)</li>"
    if run.modality == "video":
        legend = ('<div class="legend"><span>one table per round: every clip in a row was asked for with the same '
                  'seed, prompt and shape</span><span>"play row" starts every clip of the row together</span>'
                  f'<span>{cc("o")} observed (client timer, quality clip) {cc("d")} derived (computed by this page) '
                  f'{cc("s")} self-reported by the engine</span></div>')
    elif run.modality == "image":
        legend = ('<div class="legend"><span>open an image for full resolution; viewing it does not establish '
                  'semantic quality</span>'
                  f'<span>{cc("o")} observed (client timer) {cc("d")} derived</span></div>')
    else:
        legend = ('<div class="legend"><span><ins>inserted</ins> / <del>deleted</del> vs the reference</span>'
                  f'<span>{MARK} first divergence</span>'
                  '<span><span class="w-sub">substitution</span> <span class="w-ins">insertion</span> '
                  '<span class="w-del">deletion</span> vs the ground truth</span>'
                  f'<span>{cc("o")} observed (metrics files) {cc("d")} derived (computed by this page)</span></div>')
    ev_lines = "".join(f"<div>{esc(s['engine'])}: <code>{esc(', '.join(s['evidence']))}</code></div>"
                       for s in summaries if s["evidence"])
    page = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<meta name=\"color-scheme\" content=\"light dark\">"
        f"<title>Review {esc(name)}</title><style>{CSS}</style></head><body>"
        f"<header><h1>Review: {esc(name)}</h1><div class=\"meta\"><span>modality <b>{esc(run.modality)}</b></span>"
        f"<span>reference <b>{esc(run.reference or 'none')}</b></span><span>{len(run.items)} inputs</span>"
        f"<span>{len(labels)} column{'s' if len(labels) != 1 else ''}</span>"
        f"<span>built {esc(generated)} by inferencekit {esc(tool_version)}</span></div>"
        + (f'<ul class="warnings">{warnings}</ul>' if warnings else "")
        + f"<h2>Summary</h2>{summary_html(run, summaries)}{legend}</header>"
        f'<nav class="bar" aria-label="filters">{"".join(filters)}</nav>'
        f"<main>{''.join(sections)}</main>"
        f"<footer>{ev_lines}<div>Built from {len(run.sources)} files under the run directory; their sha256 "
        "digests are in review/data.json. Verdicts: PASS / FAIL / UNPROVEN / NOT RUN; a missing output is NOT RUN, "
        f"never a pass.</div></footer><script>{JS}</script></body></html>\n")

    data = {"run": name, "modality": run.modality, "reference": run.reference, "engines": summaries,
            "buckets": buckets, "warnings": run.warnings,
            "cells": [{k: v for k, v in info.items() if k not in ("diff", "alignment")} for info in infos.values()]}
    return page, data
