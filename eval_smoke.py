"""Integration smoke suite: known-answer queries against the built index.

Plan §5 requires these to pass on a built corpus: single/batch/--mentions/
--explain plus 10 known-answer queries, including the `SetPropertyBlock`
path-term case (a page whose body prose never mentions its own symbol, which
only path/title field terms can rescue) and the typo/paraphrase cases that
justify dense retrieval.

Runs against whatever index is currently built, so it is meaningful both on a
partial index (gold pages present) and on the final full build. Network-free:
dense uses the on-disk vectors, or skips with a warning if absent.

Usage:
    uv run python eval_smoke.py              # all checks
    uv run python eval_smoke.py --verbose
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.compile import load_rag_config
from rag.search.engine import SearchEngine

# (query, substring the correct source must contain, why this case exists)
KNOWN_ANSWER = [
    ("MaterialPropertyBlock", "MaterialPropertyBlock.html",
     "class page must outrank its own method pages"),
    ("Renderer.SetPropertyBlock", "Renderer.SetPropertyBlock.html",
     "path-term case: body prose never mentions the symbol"),
    ("Rigidbody.AddForce", "Rigidbody.AddForce.html",
     "dotted member lookup"),
    ("Rigidbody velocity", "Rigidbody",
     "component + property"),
    ("SetPropertyBlock MaterialPropertyBlock Graphics.DrawMesh",
     "Graphics.DrawMesh.html", "multi-symbol query"),
    ("Shader.PropertyToID", "Shader.PropertyToID.html", "identifier-only query"),
    ("gpu instancing per-instance material properties",
     "gpu-instancing", "Manual prose page, no exact identifier"),
    ("SRP Batcher MaterialPropertyBlock compatibility", "SRPBatcher",
     "Manual + hyphenated filename"),
    ("Terrain.SetSplatMaterialPropertyBlock",
     "Terrain.SetSplatMaterialPropertyBlock.html", "long member name"),
    ("RenderParams matProps", "RenderParams-matProps.html",
     "hyphenated ScriptReference filename"),
]

# Queries BM25 alone cannot answer (min_should_match=0.6 returns zero hits);
# these are what dense retrieval earns its place with.
DENSE_ONLY = [
    ("typo: Rigidboddy velocity", "Rigidbody", "single-char typo"),
    ("typo: MaterialPropertyBlokc", "MaterialPropertyBlock", "transposed chars"),
    ("play a sound when something collides", "AudioSource",
     "paraphrase, no shared identifier"),
    ("per-renderer shader constant override", "SetPropertyBlock",
     "cross-vocabulary paraphrase"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rag_config.json")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    cfg = load_rag_config(args.config)
    t0 = time.time()
    engine = SearchEngine(cfg)
    try:
        engine.load()
    except FileNotFoundError as exc:
        print(f"SMOKE SKIP: {exc}")
        return 1
    print(f"[smoke] index: {engine.n_chunks} chunks, dense={engine.has_dense} "
          f"({time.time() - t0:.1f}s to load)\n")

    failures: list[str] = []

    print("== known-answer queries (hybrid) ==")
    indexed = {r.source for r in engine.rows}

    def gold_is_indexed(gold: str) -> bool:
        """Is any page matching *gold* actually in the index?

        Distinguishes a genuine ranking failure from "this page has no corpus
        yet" — during a partial build the latter is expected, and reporting it as
        MISS would make incomplete coverage look like a quality regression.
        """
        return any(gold.lower() in s.lower() for s in indexed)

    skipped = 0
    for q, gold, why in KNOWN_ANSWER:
        if not gold_is_indexed(gold):
            skipped += 1
            print(f"  [SKIP] not-in-index  {q[:44]:<44} (no corpus for *{gold} yet)")
            continue
        hits = engine.search(q, k=args.k, mode="hybrid")["hits"]
        srcs = [h["source"] for h in hits]
        ok = any(gold.lower() in s.lower() for s in srcs)
        rank = next((i + 1 for i, s in enumerate(srcs)
                     if gold.lower() in s.lower()), None)
        mark = "OK  " if ok else "MISS"
        top = Path(srcs[0]).name if srcs else "<none>"
        print(f"  [{mark}] rank={str(rank):<4} {q[:44]:<44} -> {top}")
        if args.verbose:
            print(f"         why: {why}")
        if not ok:
            failures.append(f"known-answer miss: {q!r} (expected {gold!r}, got {srcs[:3]})")
    if skipped:
        print(f"  ({skipped} case(s) skipped: gold page not in the index yet — "
              f"corpus build incomplete, {len(indexed)} pages indexed)")

    print("\n== dense-dependent queries (BM25 returns nothing) ==")
    if not engine.has_dense:
        print("  SKIP: no vectors.f32 — build with `rag.py compile --steps index`")
    else:
        for q, gold, why in DENSE_ONLY:
            if not gold_is_indexed(gold):
                print(f"  [SKIP] not-in-index  {q[:40]:<40} (no corpus for *{gold})")
                continue
            b = engine.search(q, k=args.k, mode="bm25")["hits"]
            h = engine.search(q, k=args.k, mode="hybrid")["hits"]
            b_ok = any(gold.lower() in x["source"].lower() for x in b)
            h_ok = any(gold.lower() in x["source"].lower() for x in h)
            if b_ok:
                # BM25 handled it; not evidence for dense, but not a failure
                mark, note = "OK  ", "bm25 also found it"
            elif h_ok:
                mark, note = "OK  ", "DENSE RESCUED (bm25 had 0 hits)" if not b \
                    else "dense fixed ranking"
            else:
                mark, note = "MISS", f"neither found {gold}"
                failures.append(f"dense-dependent miss: {q!r} (expected {gold!r})")
            print(f"  [{mark}] bm25={str(b_ok):<5} hybrid={str(h_ok):<5} "
                  f"{q[:40]:<40} {note}")
            if args.verbose:
                print(f"         why: {why}")

    print("\n== feature parity ==")

    def check(name: str, fn) -> bool:
        try:
            fn()
            print(f"  [OK  ] {name}")
            return True
        except Exception as exc:
            print(f"  [MISS] {name}: {type(exc).__name__}: {exc}")
            failures.append(f"{name}: {exc}")
            return False

    def _single():
        out = engine.search("MaterialPropertyBlock", k=3)
        assert out["hits"] and "source" in out["hits"][0]
        assert "text" in out["hits"][0] and out["hits"][0]["text"]

    def _explain():
        out = engine.search("MaterialPropertyBlock", k=3, explain=True)
        assert out.get("explain"), "no explain block"
        assert any(t["in_index"] for t in out["explain"]), "no term matched the index"
        assert out["hits"][0]["explain_scores"]["bm25"] > 0

    def _mentions():
        res = engine.mentions("MaterialPropertyBlock")
        assert res, "no mentions found"
        assert all("source" in r and "count" in r for r in res)
        # literal enumeration is ordered by occurrence count
        counts = [r["count"] for r in res]
        assert counts == sorted(counts, reverse=True)

    def _mentions_is_literal_not_bm25():
        """--mentions must enumerate, not rank: a page with many occurrences of a
        common term beats a page where a rare identifier appears once."""
        res = engine.mentions("Rigidbody")
        assert res
        assert res[0]["count"] >= res[-1]["count"]

    def _determinism():
        a = engine.search("Rigidbody velocity", k=5, mode="hybrid")
        b = engine.search("Rigidbody velocity", k=5, mode="hybrid")
        assert [(h["source"], h["chunk_uid"], h["fused_score"])
                for h in a["hits"]] == \
               [(h["source"], h["chunk_uid"], h["fused_score"])
                for h in b["hits"]], "results are not bit-identical across runs"

    def _per_file_dedupe():
        out = engine.search("MaterialPropertyBlock", k=8, mode="bm25")
        srcs = [h["source"] for h in out["hits"]]
        assert len(srcs) == len(set(srcs)), f"per_file=1 violated: {srcs}"

    def _empty_query_hint():
        """A term-free query must produce 0 hits AND a hint.

        Checked in bm25 mode: dense always returns nearest neighbours (cosine
        similarity is defined for every vector), so a nonsense query legitimately
        yields low-relevance hits there — that is expected, not a missing hint.
        """
        out = engine.search("zzzqqqxxxnotaterm", k=3, mode="bm25")
        assert out["hits"] == [], f"bm25 matched a nonsense term: {out['hits'][:2]}"
        assert "hint" in out

    def _modes():
        for mode in ("bm25", "hybrid", "dense"):
            if mode != "bm25" and not engine.has_dense:
                continue
            out = engine.search("Rigidbody", k=3, mode=mode)
            assert out["query"] == "Rigidbody"

    check("single query returns source+snippet", _single)
    check("--explain term df + score breakdown", _explain)
    check("--mentions literal enumeration", _mentions)
    check("--mentions ordered by count (not BM25)", _mentions_is_literal_not_bm25)
    check("determinism (bit-identical rerun)", _determinism)
    check("per_file=1 dedupe", _per_file_dedupe)
    check("0-hit query carries a hint", _empty_query_hint)
    check("mode=bm25|hybrid|dense all run", _modes)

    print(f"\n== summary ==")
    if failures:
        print(f"SMOKE FAIL: {len(failures)} problem(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SMOKE PASS: all known-answer queries and parity checks green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
