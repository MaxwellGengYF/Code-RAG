"""Markdown translators for engine results (default search output).

The CLI's consumer is a coding agent (LLM): markdown is the compact,
token-saving rendering of a search/mentions payload. JSON stays available
via --json (and --out always writes raw JSON).
"""
from __future__ import annotations


def result_to_markdown(r: dict, *, dense=None, mode: str = "") -> str:
    """One engine result dict {query, hits, ...} -> markdown."""
    heading = f"### {r.get('query', '')}"
    if dense is False and mode in ("hybrid", "dense"):
        heading += " (bm25-only)"
    lines = [heading]
    hits = r.get("hits") or []
    if not hits:
        lines.append(r.get("hint") or "no results")
    for h in hits:
        lines.append(f"{h['rank']}. **{h['title']}** `{h['source']}` "
                     f"score={float(h['fused_score']):.4f}")
        text = " ".join(str(h.get("text", "")).split())
        lines.append(f"  > {text}")
        lines.append(f"  read: {h.get('read_more', '')}")
    explain = r.get("explain")
    if explain:
        known = [t["term"] for t in explain if t.get("in_index")]
        lines.append(f"terms in index: {len(known)}/{len(explain)} {known}")
    return "\n".join(lines)


def results_to_markdown(payload: dict) -> str:
    """Full search payload {_meta, results} -> markdown (blank-line joined)."""
    meta = payload.get("_meta") or {}
    dense = meta.get("dense")
    mode = meta.get("mode", "")
    return "\n\n".join(result_to_markdown(r, dense=dense, mode=mode)
                       for r in payload.get("results", []))


def mentions_to_markdown(term: str, rows: list[dict]) -> str:
    """Mentions payload rows -> markdown bullet list."""
    lines = [f"### mentions: {term} ({len(rows)} files)"]
    for r in rows:
        line = f"- `{r['source']}` ×{r['count']}"
        if r.get("context"):
            line += f" — {r['context']}"
        lines.append(line)
    return "\n".join(lines)
