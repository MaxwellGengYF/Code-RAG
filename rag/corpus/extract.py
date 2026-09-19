"""Page extraction: HTML doc page -> markdown input for the corpus LLM.

Wraps the shared cleaners (``doc_clean``) and the markdown renderer
(``dumpdoc.render``) so the corpus pipeline feeds on exactly the same cleaned
text that the reader tools produce. Input is capped at ``max_chars`` with an
explicit truncation marker.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

from doc_clean import BOILER, content_root, page_title
from rag import rel_source

import dumpdoc

TRUNCATION_MARKER = "\n\n[... page truncated: {n} more chars not shown ...]\n"


@dataclass
class PageInput:
    """One documentation page ready for LLM corpus generation."""

    source: str  # ROOT-relative posix path, e.g. "ScriptReference/Rigidbody.html"
    title: str
    markdown: str  # cleaned markdown, already capped
    char_len: int  # length BEFORE capping (for stats/cost estimation)


def render_markdown(html_text: str) -> tuple[str, str]:
    """(markdown, title) for one HTML doc page, boiler stripped, dumpdoc rendering."""
    soup = BeautifulSoup(html_text, "lxml")
    title = page_title(soup)
    root = content_root(soup)
    out: list[str] = []
    dumpdoc.render(root, out)
    # Same dedup/boiler filter as dumpdoc.dump (minus the file heading).
    res: list[str] = []
    prev: str | None = None
    for line in out:
        line = line.strip()
        if not line or line == prev or BOILER.search(line):
            continue
        res.append(line)
        prev = line
    return "\n\n".join(res) + "\n", title


def extract_page(
    html_path: str | Path,
    *,
    root: Path | None = None,
    max_chars: int = 24_000,
) -> PageInput | None:
    """Extract a PageInput from one HTML file; None when the page has no content.

    ``root`` defaults to the repo root (used for stable rel paths).
    """
    p = Path(html_path)
    try:
        html_text = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    markdown, title = render_markdown(html_text)
    if root is not None:
        try:
            source = p.resolve().relative_to(root).as_posix()
        except (ValueError, OSError):
            source = rel_source(p)
    else:
        source = rel_source(p)
    char_len = len(markdown)
    if char_len > max_chars:
        keep = max_chars - len(TRUNCATION_MARKER.format(n=char_len - max_chars))
        markdown = markdown[:keep].rstrip() + TRUNCATION_MARKER.format(
            n=char_len - max_chars)
    if len(markdown.strip()) < 30:
        return None
    return PageInput(source=source, title=title, markdown=markdown, char_len=char_len)
