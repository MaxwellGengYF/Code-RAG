"""Does dense rescue queries where BM25 returns zero hits?

That is the only remaining justification for shipping mode=hybrid as the default,
given that on the independent base-24 gold set BM25 alone scores best
(MRR 0.922) and dense fusion does not improve it. BM25 here uses
min_should_match=0.6, so a query whose terms are mostly absent from a page can
score nothing at all; a semantic embedder can still rank the right page.

Probes paraphrase / typo / cross-vocabulary queries that are deliberately NOT
lexical matches for their gold page, and reports BM25 hits vs hybrid hits.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.compile import load_rag_config
from rag.search.engine import SearchEngine

# (query, substring the correct source must contain)
CASES = [
    ("how to make an object fall with gravity", "Rigidbody"),
    ("change color of one object without affecting others", "MaterialPropertyBlock"),
    ("stop a component from moving", "Rigidbody-constraints"),
    ("read pixels from a render texture", "RenderTexture"),
    ("make two objects stick together", "Joint"),
    ("play a sound when something collides", "AudioSource"),
    ("reduce draw calls by combining meshes", "DrawCallBatching"),
    ("load a scene from another thread", "SceneManager"),
    ("typo: Rigidboddy velocity", "Rigidbody"),
    ("typo: MaterialPropertyBlokc", "MaterialPropertyBlock"),
    ("cross vocabulary: per-renderer shader constant override", "SetPropertyBlock"),
    ("cross vocabulary: instanced rendering per-object data", "gpu-instancing"),
]


def main():
    cfg = load_rag_config("rag_probe_config.json")
    engine = SearchEngine(cfg)
    engine.load()
    print(f"chunks={engine.n_chunks} dense={engine.has_dense}\n")
    print(f"{'query':<48} {'bm25':>5} {'hyb':>4}  {'bm25 top1':<38} rescue")
    b_hits = h_hits = 0
    rescued = 0
    for q, gold in CASES:
        b = engine.search(q, k=5, mode="bm25")["hits"]
        h = engine.search(q, k=5, mode="hybrid")["hits"]
        b_ok = any(gold.lower() in x["source"].lower() for x in b)
        h_ok = any(gold.lower() in x["source"].lower() for x in h)
        b_hits += b_ok
        h_hits += h_ok
        is_rescue = (not b) and bool(h)
        rescued += is_rescue
        top1 = b[0]["source"].split("/")[-1][:36] if b else "<NO BM25 HITS>"
        flag = ""
        if is_rescue:
            flag = "<-- dense rescued zero-hit"
        elif h_ok and not b_ok:
            flag = "<-- dense fixed ranking"
        print(f"{q[:46]:<48} {str(b_ok):>5} {str(h_ok):>4}  {top1:<38} {flag}")
    print(f"\ngold found in top-5: bm25={b_hits}/{len(CASES)}  "
          f"hybrid={h_hits}/{len(CASES)}  dense-rescued-zero-hit={rescued}")


if __name__ == "__main__":
    main()
