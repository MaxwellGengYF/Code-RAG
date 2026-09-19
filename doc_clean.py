"""Shared HTML cleaning utilities for the Unity Manual / ScriptReference mirror.

Used by hybrid_retrieve.py (chunk extraction for the BM25 index) and dumpdoc.py
(page -> markdown). Extracting these here keeps the content-root selector and the
boilerplate rules in one place so the index and the page dumps never drift apart.

Unity doc pages are static HTML with this shape (docs.unity3d.com, Unity 6.x era)::

    <div id="master-wrapper">
      <div id="header">...</div>
      <div id="sidebar">...</div>
      <div id="content-wrap" class="content-wrap">
        <div class="content-block">
          <div class="content">
            <div class="section"> ... real documentation ... </div>
            <div class="section"><div id="_content"></div></div>   <!-- JS-populated -->
          </div>
          <div class="footer-wrapper"><div class="footer">... site chrome ...</div></div>
        </div>
      </div>
    </div>

Notes:
* ``#_content`` is filled by JavaScript at view time and is EMPTY in the static
  mirror, so it cannot be the content root — ``#content-wrap`` is the right root.
* The footer wrapper is a plain ``<div>`` (not a ``<footer>`` tag) sitting inside
  ``#content-wrap``, so decomposing semantic tags alone leaves site chrome in the
  text. ``strip_boiler`` removes those lines by pattern.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

# Sentence/line starters of the feedback widget, suggestion form and site footer.
# They appear on every page, so they carry zero ranking signal; stripping them keeps
# snippets and page dumps clean. Patterns are anchored to the whole line because
# renderers (dumpdoc inline_text, table cells) sometimes concatenate siblings, e.g.
# "Suggest a changeSuccess!Thank you for helping us improve ...".
_BOILER_PREFIXES = (
    r"Success!.*",
    r"Submission failed.*",
    r"Thank you for helping us improve the quality of Unity Documentation\b.*",
    r"For some reason your suggested change could not be submitted\b.*",
    r"Suggest a change.*",
    r"Leave feedback.*",
    r"Switch to Manual.*",
    r"Switch to Scripting API.*",
    r"Is something described here not working as you expect it to\s*\?.*",
    r"It might be a\s*Known Issue\s*\..*",
    r"issuetracker\.unity3d\.com.*",
    r"Copyright ©\s*\d{4}.*Unity Technologies\b.*",
    r"Built from job ID \d+.*",
    r"Your name.*",
    r"Your email.*",
    r"Suggestion\*?.*",
    r"Submit suggestion.*",
)

# Short standalone chrome links/labels. Anchored exactly: a line must BE the chrome
# text (checked: no Manual/ScriptReference h1 collides with any of these).
_BOILER_EXACT = (
    r"Terms of use",
    r"Documentation Terms of Use",
    r"Privacy Policy",
    r"Do Not Sell or Share My Personal Information",
    r"Cookies",
    r"Legal",
    r"Asset Store",
    r"Cancel",
    r"Close",
    r"Tutorials",
    r"Community Answers",
    r"Knowledge Base",
    r"Forums",
    r"Your Privacy Choices.*",
)

BOILER = re.compile(
    r"^(?:" + "|".join(_BOILER_PREFIXES + _BOILER_EXACT) + r")$"
)


def is_boiler_line(line: str) -> bool:
    """True when *line* (already stripped) is site chrome, not documentation."""
    return bool(BOILER.search(line.strip()))


def strip_boiler(text: str) -> str:
    """Drop boilerplate lines from newline-joined extracted text."""
    return "\n".join(ln for ln in text.splitlines() if ln.strip() and not is_boiler_line(ln))


def content_root(soup: BeautifulSoup) -> Tag:
    """The element that holds the actual documentation for *soup*.

    Prefers ``#content-wrap``; falls back to ``<body>`` (or the document itself) so
    callers never get None even on unexpected/legacy page layouts.
    """
    for sel in ("#content-wrap", "#bodyContent", "main", "article"):
        el = soup.select_one(sel)
        if el is not None:
            return el
    return soup.body or soup


_TITLE_PREFIX_RE = re.compile(r"^Unity\s*-\s*(?:Scripting API|Manual)\s*:\s*", re.I)


def page_title(soup: BeautifulSoup) -> str:
    """Human title of a doc page: "Rigidbody", "Unity 6.5 User Manual".

    Uses the ``<title>`` tag with the "Unity - Scripting API: " / "Unity - Manual: "
    prefix removed; falls back to the first ``<h1>`` and finally to "".
    """
    if soup.title and soup.title.string:
        title = _TITLE_PREFIX_RE.sub("", soup.title.string.strip()).strip()
        if title:
            return title
    h1 = soup.find("h1")
    if h1 is not None:
        txt = h1.get_text(strip=True)
        if txt:
            return txt
    return ""
