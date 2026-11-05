"""User-facing text is plain English: no internal jargon ("hero", "arm", "lane") in the docs people read.

Scope: every Markdown file under plugin/ (skills, references, agents, commands) and the repo-level docs
people read (README.md, CONTRIBUTING.md, SECURITY.md, docs/install.md and docs/examples.md, plus METHOD.md where it exists), each when present. A fenced block is skipped only when its info string names a programming or data
language (```cuda, ```bash, ```json, ...): an untagged, ```text or ```markdown fence holds checklists and
templates people read, so it is checked like any other line. Link targets and URLs are skipped. Inside an
inline code span only paths and file names are skipped: `hero/usage.py` could be a real directory name, so it
cannot be reworded, but a placeholder such as `<that arm's id>` is prose in
backticks and counts. "ARM", the processor architecture, is not the jargon word.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JARGON = re.compile(r"\b(?:hero(?:es)?|arms?|lanes?)\b", re.IGNORECASE)
ALLOWED = {"ARM"}  # the processor architecture
FENCE = re.compile(r"^\s*(`{3,}|~{3,})\s*\{?\.?([^\s`{}]*)")  # the fence and the first word of its info string
# Info strings of code (skipped). Anything else, no info string included, is text people read.
CODE_LANGS = {
    "bash", "sh", "shell", "zsh", "fish", "console", "shell-session", "powershell", "ps1", "bat", "cmd",
    "python", "py", "python3", "pycon", "ipython", "c", "h", "cpp", "c++", "cc", "cxx", "hpp", "cuda", "cu",
    "ptx", "asm", "nasm", "llvm", "mlir", "triton", "rust", "rs", "go", "java", "kotlin", "scala", "swift",
    "js", "javascript", "ts", "typescript", "jsx", "tsx", "lua", "ruby", "rb", "perl", "r", "julia", "sql",
    "json", "jsonc", "json5", "jsonl", "ndjson", "yaml", "yml", "toml", "ini", "cfg", "conf", "xml", "html",
    "css", "diff", "patch", "make", "makefile", "cmake", "dockerfile", "docker", "nix", "hcl", "proto",
    "graphql",
}
CODE_SPAN = re.compile(r"(`+)(?:(?!\1).)+?\1")
PATH = re.compile(r"[\w.-]*/[\w./-]*|[\w-]+(?:\.[\w-]+)*\.[A-Za-z]\w*")  # a/b, a/, a.py, x.NOTES.md
LINK_TARGET = re.compile(r"\]\([^)\s]*\)")
URL = re.compile(r"\b(?:https?://|www\.)\S+")


# Repo-level docs people read (the plugin's own Markdown is added below).
REPO_DOCS = ("README.md", "CONTRIBUTING.md", "SECURITY.md", "docs/install.md", "docs/examples.md", "METHOD.md")


def user_docs() -> list[Path]:
    files = sorted((ROOT / "plugin").rglob("*.md"))
    files += [ROOT / rel for rel in REPO_DOCS]
    return [p for p in files if p.is_file()]


def prose(text: str):
    """Yield (line number, line) with code blocks dropped, link targets, URLs and paths in code spans blanked.

    A code block is a fence whose info string is in CODE_LANGS; the lines of any other fence are prose.
    """
    fence, code = None, False
    for n, line in enumerate(text.splitlines(), 1):
        m = FENCE.match(line)
        if fence is None and m:
            fence = m.group(1)[0] * 3  # the closing fence uses the same character
            code = m.group(2).lower() in CODE_LANGS
            continue
        if fence is not None and line.strip().startswith(fence):
            fence = None
            continue
        if fence is not None and code:
            continue
        for rx in (LINK_TARGET, URL):
            line = rx.sub(" ", line)
        line = CODE_SPAN.sub(lambda m: " " + PATH.sub(" ", m.group(0).strip("`")) + " ", line)
        yield n, line


def jargon(text: str) -> list[tuple[int, str]]:
    return [(n, m.group(0)) for n, line in prose(text) for m in JARGON.finditer(line) if m.group(0) not in ALLOWED]


def test_prose_filter_skips_code_links_and_the_cpu_architecture():
    text = ("Runs on ARM CPUs; see [`hero/usage.py`](https://example.com/hero/usage.py) and `lane_id`.\n"
            "```cuda\nconst int lane = threadIdx.x % 32;\n```\n"
            "Each arm runs fresh. The hero run used two lanes.\n"
            "~~~python\narm = 1\n~~~\n"
            "[the record](records/hero-run/) www.example.com/arm\n"
            "Read `hero/verify/` and `hero.md`, then pass `--extend <that arm's id>`.\n")
    assert jargon(text) == [(5, "arm"), (5, "hero"), (5, "lanes"), (10, "arm")]


def test_prose_filter_reads_untagged_and_text_fences():
    # An untagged fence is a checklist or a template people read: its words count, paths in code spans don't.
    text = ("```\n- [ ] 9. % of ceiling per arm (derived); see `hero/usage.py`\n```\n"
            "```cuda\nconst int lane = threadIdx.x & 31;  // the lane in its warp\n```\n"
            "``` text\nMetric: median per arm\n```\n"
            "~~~markdown\nThe hero run.\n~~~\n"
            "```{.bash}\necho arm\n```\n"
            "After the fences: plain words.\n")
    assert jargon(text) == [(2, "arm"), (8, "arm"), (11, "hero")]


def test_user_docs_use_plain_words():
    hits = []
    for path in user_docs():
        for n, word in jargon(path.read_text(encoding="utf-8")):
            hits.append(f"{path.relative_to(ROOT)}:{n}: {word}")
    assert not hits, ("internal jargon in user-facing text; say what is meant instead (an arm of a comparison "
                      "is an engine or a side, a warp lane is a thread, a hero run is an agent run):\n"
                      + "\n".join(hits))
