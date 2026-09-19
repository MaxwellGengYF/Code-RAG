"""Dump Unity doc HTML pages as clean markdown, in document order.

Usage: uv run python dumpdoc.py -o out.md path1.html path2.html ...
"""
import sys, re, io, argparse
from pathlib import Path
from bs4 import BeautifulSoup

from doc_clean import BOILER, content_root, is_boiler_line

CONTAINERS = {"div", "section", "article", "main", "header", "footer", "nav", "ul", "ol",
              "table", "thead", "tbody", "tr", "pre", "h1", "h2", "h3", "h4", "h5", "h6",
              "dl", "dt", "dd", "form", "fieldset"}
LEAF = {"p", "li", "td", "th", "blockquote", "caption", "figcaption", "dt", "dd", "span"}
DROP = {"script", "style", "noscript", "svg", "button", "select", "option", "iframe"}

NL = chr(10)

def inline_text(el) -> str:
    t = el.get_text(" ", strip=True)
    return re.sub(r"\s*\n\s*", " ", t)

def has_block_child(el) -> bool:
    return any(getattr(c, "name", None) in CONTAINERS for c in el.children)

def render(el, out, depth=0):
    name = el.name
    if name is None:
        return
    if name in DROP:
        return
    cls = " ".join(el.get("class", []))
    idv = el.get("id", "") or ""
    if name in ("nav", "footer") or "breadcrumb" in cls or idv in ("sidebar", "header", "master-wrapper"):
        return
    if "sig-block" in cls:
        txt = el.get_text("").strip()
        txt = re.sub(r"^Declaration", "", txt)
        txt = re.sub(r"\s*\n\s*", " ", txt)
        txt = re.sub(r"(?<=[A-Za-z0-9_>\)\]]) (?=[A-Za-z(])", " ", txt)
        txt = re.sub(r"\s{2,}", " ", txt).strip()
        if txt:
            out.append("```csharp" + NL + txt + NL + "```")
        return
    if name == "pre":
        for br in el.find_all("br"):
            br.replace_with(NL)
        code = el.get_text("", strip=False)
        code = "\n".join(ln.rstrip() for ln in code.splitlines())
        out.append("```csharp\n" + code.strip("\n") + "\n```")
        return
    if name.startswith("h") and len(name) == 2 and name[1].isdigit():
        txt = inline_text(el)
        # skip boilerplate headings (e.g. the feedback widget's "Success!")
        if txt and not is_boiler_line(txt):
            out.append("#" * min(int(name[1]) + 1, 6) + " " + txt)
        return
    if name in ("ul", "ol"):
        for li in el.find_all("li", recursive=False):
            if has_block_child(li):
                render(li, out, depth)
                continue
            t = inline_text(li)
            if t:
                out.append("- " + t)
        return
    if name == "table":
        rows = []
        for tr in el.find_all("tr"):
            cells = [inline_text(c) for c in tr.find_all(["th", "td"], recursive=False)]
            cells = [re.sub(r"\s+", " ", c) for c in cells]
            if any(cells):
                rows.append("| " + " | ".join(cells) + " |")
        if rows:
            sep = "| " + " | ".join(["---"] * (rows[0].count("|") - 1)) + " |"
            out.append("\n".join([rows[0], sep] + rows[1:]))
        return
    if name == "tr":
        cells = [inline_text(c) for c in el.find_all(["th", "td"], recursive=False)]
        out.append("| " + " | ".join(cells) + " |")
        return
    if name == "p":
        if el.find("pre"):
            for c in el.children:
                render(c, out, depth)
            return
        if el.find("table"):
            for c in el.children:
                render(c, out, depth)
            return
        txt = inline_text(el)
        if txt:
            out.append(txt)
        return
    if name in ("div", "section", "article", "main"):
        if has_block_child(el):
            for c in el.children:
                render(c, out, depth)
            return
        txt = inline_text(el)
        if txt and len(txt) < 6000:
            out.append(txt)
        return
    # anything else with block children
    if has_block_child(el):
        for c in el.children:
            render(c, out, depth)
    else:
        txt = inline_text(el)
        if txt:
            out.append(txt)

def dump(p: str, heading: bool = True) -> str:
    f = Path(p)
    if not f.exists():
        return f"### MISSING: {p}\n"
    soup = BeautifulSoup(f.read_text(encoding="utf-8", errors="ignore"), "lxml")
    root = content_root(soup)
    out = []
    if heading:
        out.append(f"{'=' * 100}\n## {p}\n{'=' * 100}")
    render(root, out)
    res, prev = [], None
    for l in out:
        l = l.strip()
        if not l or l == prev or BOILER.search(l):
            continue
        res.append(l)
        prev = l
    return "\n\n".join(res) + "\n"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out")
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    buf = io.StringIO()
    for p in a.paths:
        buf.write(dump(p))
        buf.write("\n")
    text = buf.getvalue()
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"wrote {a.out}: {len(text)} chars, {text.count(chr(10))} lines")
    else:
        sys.stdout.write(text)
