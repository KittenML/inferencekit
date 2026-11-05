"""Text comparison for the review page: first divergence, word diffs and word-error alignment.

Three questions, three functions:
  * ``first_token_divergence``: where do two greedy generations stop agreeing, in tokens?
  * ``word_diff``: which words of a candidate text differ from the reference text, for
    highlighting (inserted, deleted and replaced runs of words, whitespace preserved)?
  * ``align_words``: which words of a hypothesis transcript are substitutions,
    insertions or deletions against a ground truth, and the word error rate.

Everything is plain Python (``difflib`` plus a small Levenshtein table), so the page can
be built anywhere without extra dependencies.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field

_WORD_SPACE = re.compile(r"(\S+)(\s*)")
_LEAD_SPACE = re.compile(r"\s*")
# Exact Levenshtein alignment up to this many table cells; above it, difflib anchors the
# alignment on long matching runs and the table only fills the gaps between them.
EXACT_CELLS = 250_000


def first_token_divergence(a: list | None, b: list | None) -> int | None:
    """Index of the first differing token id; None if the sequences are identical (length included)."""
    a, b = list(a or []), list(b or [])
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def words(text: str) -> tuple[str, list[tuple[str, str]]]:
    """(leading whitespace, [(word, whitespace after it)]): the text, losslessly, as words."""
    text = text or ""
    lead = _LEAD_SPACE.match(text).group(0)
    return lead, _WORD_SPACE.findall(text[len(lead):])


@dataclass
class TextDiff:
    """A candidate text as segments against a reference text.

    ``segments`` holds ``(tag, reference_text, candidate_text)`` with tag one of
    ``equal``, ``replace``, ``delete`` (only in the reference) or ``insert`` (only in the
    candidate). Joining every candidate_text gives back the candidate exactly.
    """

    lead: str
    segments: list[tuple[str, str, str]] = field(default_factory=list)
    first_word: int | None = None  # index of the first differing word in the candidate; None = identical
    ref_words: int = 0
    cand_words: int = 0

    @property
    def identical(self) -> bool:
        return self.first_word is None


def word_diff(reference: str, candidate: str) -> TextDiff:
    """Word-level diff of ``candidate`` against ``reference`` (whitespace kept for display)."""
    _, ref = words(reference)
    lead, cand = words(candidate)
    sm = difflib.SequenceMatcher(a=[w for w, _ in ref], b=[w for w, _ in cand], autojunk=False)
    out = TextDiff(lead=lead, ref_words=len(ref), cand_words=len(cand))
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        r = "".join(w + s for w, s in ref[i1:i2])
        c = "".join(w + s for w, s in cand[j1:j2])
        if tag != "equal" and out.first_word is None:
            out.first_word = j1
        out.segments.append((tag, r, c))
    return out


# --------------------------------------------------------------------------- word errors

def normalize_word(word: str) -> str:
    """Lowercase, Unicode NFKC, punctuation removed: the usual WER normalisation for one word."""
    w = unicodedata.normalize("NFKC", word).lower()
    return "".join(ch for ch in w if not unicodedata.category(ch).startswith("P"))


@dataclass
class Alignment:
    """Word-level alignment of a hypothesis against a ground truth.

    ``ops`` holds ``(op, reference_word, hypothesis_word)`` with op one of ``ok``, ``sub``,
    ``del`` (a ground-truth word the hypothesis missed) or ``ins`` (an extra hypothesis
    word). Words are shown in their original spelling; matching uses ``normalize_word``.
    """

    ops: list[tuple[str, str | None, str | None]]
    ref_len: int
    exact: bool  # True: minimal edit distance; False: anchored on long matches (very long texts)

    def count(self, op: str) -> int:
        return sum(1 for o, _, _ in self.ops if o == op)

    @property
    def wer(self) -> float | None:
        if self.ref_len == 0:
            return None
        return (self.count("sub") + self.count("del") + self.count("ins")) / self.ref_len


def _tokens(text: str) -> list[tuple[str, str]]:
    """(normalised, original) for each word that is not pure punctuation."""
    out = []
    for w in (text or "").split():
        n = normalize_word(w)
        if n:
            out.append((n, w))
    return out


def _levenshtein(ref: list[tuple[str, str]], hyp: list[tuple[str, str]]) -> list[tuple[str, str | None, str | None]]:
    """Minimal-edit alignment (substitution, insertion, deletion all cost 1), by dynamic programming."""
    n, m = len(ref), len(hyp)
    # dist[i][j]: edits to turn ref[:i] into hyp[:j].
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dist[i][0] = i
    for j in range(m + 1):
        dist[0][j] = j
    for i in range(1, n + 1):
        ri, row, prev = ref[i - 1][0], dist[i], dist[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1][0] else 1
            row[j] = min(prev[j - 1] + cost, prev[j] + 1, row[j - 1] + 1)
    ops = []
    i, j = n, m
    while i > 0 or j > 0:  # walk back from the corner; prefer match/substitution, then deletion
        if i > 0 and j > 0 and dist[i][j] == dist[i - 1][j - 1] + (0 if ref[i - 1][0] == hyp[j - 1][0] else 1):
            ops.append(("ok" if ref[i - 1][0] == hyp[j - 1][0] else "sub", ref[i - 1][1], hyp[j - 1][1]))
            i, j = i - 1, j - 1
        elif i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            ops.append(("del", ref[i - 1][1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1][1]))
            j -= 1
    ops.reverse()
    return ops


def align_words(reference: str, hypothesis: str, exact_cells: int = EXACT_CELLS) -> Alignment:
    """Align a hypothesis transcript to the ground truth: substitutions, insertions, deletions."""
    ref, hyp = _tokens(reference), _tokens(hypothesis)
    if (len(ref) + 1) * (len(hyp) + 1) <= exact_cells:
        return Alignment(_levenshtein(ref, hyp), len(ref), exact=True)
    # Long transcripts (minutes of audio): anchor on matching runs, align each gap exactly.
    sm = difflib.SequenceMatcher(a=[r for r, _ in ref], b=[h for h, _ in hyp], autojunk=False)
    ops: list[tuple[str, str | None, str | None]] = []
    i = j = 0
    for block in sm.get_matching_blocks():
        ops += _levenshtein(ref[i:block.a], hyp[j:block.b])
        ops += [("ok", ref[block.a + k][1], hyp[block.b + k][1]) for k in range(block.size)]
        i, j = block.a + block.size, block.b + block.size
    return Alignment(ops, len(ref), exact=False)
