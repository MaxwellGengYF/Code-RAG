"""Search engine: lazy-loaded indexes, BM25 + dense candidate retrieval, RRF fusion.

Deterministic (fuzziness=0 everywhere; ties break by doc id). One chunk table
(``index/chunks.msgpack``) feeds both index layers; doc ids are positional row
ids and ``chunk_uid`` keys both.
"""
from __future__ import annotations

import heapq
import json
from pathlib import Path

import msgspec
import numpy as np

from rag import resolve_path
from rag.corpus.schema import snippet as make_snippet
from rag.index.build import ChunkRow
from rag.index.fuse import rrf_fuse
from rag.index.vector_index import embed_query, load_meta, load_vectors


class SearchEngine:
    """Serves ranked queries over the built corpus indexes."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.index_dir = resolve_path(cfg.get("index_dir", "index"))
        self.rows: list[ChunkRow] | None = None
        self.index = None
        self.searcher = None
        self.vectors: np.memmap | None = None
        self.vector_meta: dict | None = None
        self.embed_model = cfg.get("embed_model", "BAAI/bge-m3")
        self._loaded = False

    # ------------------------------------------------------------------ loading

    def load(self) -> None:
        if self._loaded:
            return
        from rag.index.bm25_index import new_searcher
        from retrieval import InvertedIndex

        chunks_path = self.index_dir / "chunks.msgpack"
        bm25_path = self.index_dir / "bm25_word.pkl"
        meta_path = self.index_dir / "vector_meta.json"
        vecs_path = self.index_dir / "vectors.f32"
        missing = [p for p in (chunks_path, bm25_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "index artefacts not found: " + ", ".join(str(m) for m in missing)
                + "\n  build them with: uv run python rag.py compile --steps index")
        # Decode with the type so nested QA structs come back as objects, not
        # dicts — ChunkRow(**r) would leave qa as dicts and to_index_text would
        # then raise AttributeError on q.q.
        from rag.index.build import load_rows
        self.rows = load_rows(chunks_path)
        self.index = InvertedIndex()
        self.index.load(str(bm25_path))
        self.searcher = new_searcher(
            self.index,
            min_should_match=float(self.cfg.get("min_should_match", 0.6)))
        if meta_path.exists() and vecs_path.exists():
            self.vector_meta = load_meta(meta_path)
            self.vectors = load_vectors(
                vecs_path, dim=int(self.vector_meta["dim"]),
                count=int(self.vector_meta["count"]))
        self._loaded = True

    @property
    def has_dense(self) -> bool:
        return self.vectors is not None

    @property
    def n_chunks(self) -> int:
        return len(self.rows or [])

    # ------------------------------------------------------------------ search

    def _bm25_candidates(self, query: str, k: int) -> list[tuple[int, float]]:
        return self.searcher.search(query, top_k=k)

    def _dense_candidates(self, query: str, k: int) -> list[tuple[int, float]]:
        if self.vectors is None:
            return []
        q = embed_query(query, model=self.embed_model)
        scores = self.vectors @ q
        top = heapq.nlargest(k, range(len(scores)), key=lambda i: (float(scores[i]), -i))
        return [(i, float(scores[i])) for i in top]

    def search(
        self,
        query: str,
        *,
        k: int = 8,
        mode: str = "hybrid",
        per_file: int = 1,
        explain: bool = False,
        no_rerank: bool = False,
        snippet_width: int = 500,
    ) -> dict:
        """One ranked query → {query, hits, explain?}."""
        self.load()
        bm25_k = int(self.cfg.get("bm25_k", 200))
        dense_k = int(self.cfg.get("dense_k", 200))
        rrf_k = int(self.cfg.get("rrf_k", 60))

        bm25 = self._bm25_candidates(query, bm25_k)
        dense = self._dense_candidates(query, dense_k) if mode in ("hybrid", "dense") else []

        if mode == "bm25":
            fused = bm25
        elif mode == "dense":
            fused = dense
        else:
            fused = rrf_fuse(bm25, dense, k=rrf_k) if dense else bm25

        deduped = self._dedupe_by_source(fused, per_file)

        rerank_scores: dict[int, float] = {}
        reranked = False
        if (not no_rerank) and self.cfg.get("rerank", False):
            # Rerank a WIDER pool than k, then truncate: a cross-encoder is the
            # most accurate signal available, so it must be able to promote a
            # page that fusion ranked below k. Truncating to k first would make
            # the reranker a no-op reordering of results already shown.
            pool = max(k * 5, self.cfg.get("rerank_pool", 50))
            deduped, rerank_scores = self._rerank(
                query, deduped,
                model=self.cfg.get("rerank_model", "BAAI/bge-reranker-v2-m3"),
                top_n=pool)
            reranked = True
        fused = deduped[:k]

        bm25_scores = dict(bm25)
        dense_scores = dict(dense)
        bm25_rank = {d: r for r, (d, _s) in enumerate(bm25)}
        dense_rank = {d: r for r, (d, _s) in enumerate(dense)}
        terms = self.searcher.tokenizer.query_terms(query) if explain else []

        hits = []
        for rank, (doc_id, score) in enumerate(fused):
            row = self.rows[doc_id]
            snip, truncated = make_snippet(
                row.to_corpus_chunk(), terms or None, width=snippet_width)
            hit = {
                "source": row.source,
                "title": row.title or Path(row.source).stem,
                "chunk_uid": row.chunk_uid,
                "rank": rank + 1,
                "fused_score": round(score, 6),
                "text": snip,
                "snippet_truncated": truncated,
                "read_more": f"uv run python dumpdoc.py {row.source}",
            }
            if reranked and doc_id in rerank_scores:
                # reported alongside fused_score (never replacing it): the two are
                # on different scales, and the rerank score is what actually
                # determines the displayed order for reranked hits.
                hit["rerank_score"] = round(rerank_scores[doc_id], 6)
            if explain:
                hit["explain_scores"] = {
                    "bm25": round(bm25_scores.get(doc_id, 0.0), 4),
                    "dense": round(dense_scores.get(doc_id, 0.0), 4),
                    "bm25_rank": bm25_rank.get(doc_id),
                    "dense_rank": dense_rank.get(doc_id),
                    "rrf": (round(1.0 / (rrf_k + bm25_rank[doc_id]), 6)
                            + (round(1.0 / (rrf_k + dense_rank[doc_id]), 6)
                               if doc_id in dense_rank else 0.0)) if mode == "hybrid" else None,
                }
            hits.append(hit)

        out: dict = {"query": query, "hits": hits}
        if reranked:
            out["reranked"] = True
        if explain:
            out["explain"] = self._explain_terms(terms)
        if not hits:
            out["hint"] = ("0 results. Try --explain to see which query terms are "
                           "indexed, or 'rag.py search --mentions TERM' to enumerate "
                           "literal occurrences.")
        return out

    def _explain_terms(self, terms: list[str], limit: int = 12) -> list[dict]:
        info = []
        for t in terms[:limit]:
            try:
                df = self.searcher.index.doc_freq(t)
            except Exception:
                df = -1
            info.append({"term": t, "df": df, "in_index": df > 0})
        return info

    def _dedupe_by_source(self, fused: list[tuple[int, float]],
                          per_file: int) -> list[tuple[int, float]]:
        """Keep at most *per_file* chunks per source document, preserving order."""
        if not per_file:
            return fused
        seen: dict[str, int] = {}
        out: list[tuple[int, float]] = []
        for d, s in fused:
            src = self.rows[d].source
            n = seen.get(src, 0)
            if n >= per_file:
                continue
            seen[src] = n + 1
            out.append((d, s))
        return out

    def _rerank(self, query: str, fused: list[tuple[int, float]], *,
                model: str, top_n: int = 20) -> tuple[list[tuple[int, float]], dict[int, float]]:
        """Reorder the head of *fused* by cross-encoder relevance.

        Returns (reordered list, {doc_id: rerank_score}). The rerank score is
        returned separately rather than substituted into ``fused_score``: the two
        are on different scales and silently overwriting one with the other would
        make the reported score unable to explain the reported order.
        """
        from rag.search.rerank import rerank_pairs
        head = fused[:top_n]
        pairs = [(query, self.rows[d].text) for d, _s in head]
        scores = rerank_pairs(pairs, model=model)
        order = sorted(range(len(head)), key=lambda i: (-scores[i], i))
        reranked = [head[i] for i in order]
        return reranked + fused[top_n:], {head[i][0]: scores[i] for i in order}

    # ------------------------------------------------------------------ mentions

    def mentions(self, term: str, *, limit: int = 60, context: int = 0,
                 dirs: tuple[str, ...] | None = None) -> list[dict]:
        """Literal enumeration: every chunk's page containing *term*, with counts."""
        import re
        self.load()
        needle = term.lower()
        per_file: dict[str, int] = {}
        samples: dict[str, str] = {}
        for row in self.rows:
            src = row.source
            if dirs and not src.startswith(dirs):
                continue
            hay = row.text.lower()
            n = hay.count(needle)
            if not n:
                continue
            per_file[src] = per_file.get(src, 0) + n
            if context and src not in samples:
                i = hay.find(needle)
                lo = max(0, i - context)
                samples[src] = re.sub(r"\s+", " ", row.text[lo:i + len(term) + context])
        ranked = heapq.nlargest(limit, per_file, key=lambda s: (per_file[s], s))
        return [{"source": s, "count": per_file[s],
                 **({"context": samples[s]} if context and s in samples else {})}
                for s in ranked]
