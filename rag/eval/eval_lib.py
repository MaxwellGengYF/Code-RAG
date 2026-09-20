"""Retrieval quality eval harness for the Unity manual hybrid retriever.

Loads the BM25 index ONCE, then sweeps tokenizer/scoring configurations against a
query set with known-good answer files (gold), and reports recall@k / MRR / precision@k.

Gold sets are derived from an actual documented research task (per-object material
properties / MaterialPropertyBlock) where the correct source pages were verified by
reading the HTML directly.

Usage:
    uv run python eval_retrieval.py                 # sweep configs
    uv run python eval_retrieval.py --configs bm25,word
    uv run python eval_retrieval.py --show-gold
"""
from __future__ import annotations

import json
from pathlib import Path

from rag.legacy.hybrid_retrieve import (
    Chunk,
    HashEmbedder,
    cosine_scores,
    dedupe_by_source,
    minmax_normalize,
)
from rag.legacy.retrieval import Searcher

CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "retriever_config.json")

# (query, [acceptable gold source paths as substrings])
GOLD: list[tuple[str, list[str]]] = [
    (
        "MaterialPropertyBlock",
        ["MaterialPropertyBlock.html"],
    ),
    (
        "SetPropertyBlock on Renderer to override material properties per object",
        ["Renderer.SetPropertyBlock.html"],
    ),
    (
        "renderer.SetPropertyBlock property block float vector color",
        ["Renderer.SetPropertyBlock.html"],
    ),
    (
        "sharedMaterial vs material instance same material multiple objects",
        ["Renderer-sharedMaterial.html", "Renderer-material.html", "DrawCallBatching-Properties"],
    ),
    (
        "Unity - Scripting API: MaterialPropertyBlock",
        ["MaterialPropertyBlock.html"],
    ),
    (
        "per-object material property without instantiating material",
        ["MaterialPropertyBlock.html", "DrawCallBatching-Properties"],
    ),
    (
        "MaterialPropertyBlock SetFloat SetColor SetVector SetTexture SetMatrix SetBuffer",
        ["MaterialPropertyBlock.html"],
    ),
    (
        "SetPropertyBlock MaterialPropertyBlock Graphics.DrawMesh",
        ["Graphics.DrawMesh.html", "MaterialPropertyBlock.html", "Graphics.RenderMesh.html"],
    ),
    (
        "GPU instancing per-instance material properties",
        ["gpu-instancing-per-instance-properties", "MaterialPropertyBlock.html"],
    ),
    (
        "SRP Batcher MaterialPropertyBlock compatibility",
        ["SRPBatcher-Incompatible", "SRPBatcher-Materials", "MaterialPropertyBlock.html",
         "DrawCallBatching-Properties"],
    ),
    (
        "ShaderLab property block _Color per renderer override",
        ["SL-Properties", "writing-shader-material-properties"],
    ),
    (
        "Renderer.GetPropertyBlock returns copies of the values",
        ["Renderer.GetPropertyBlock.html"],
    ),
    (
        "optimization avoid material instance explosion",
        ["optimizing-draw-calls-choose-method", "DrawCallBatching-Properties"],
    ),
    (
        "MaterialPropertyBlock arrays Vector4[] Matrix4x4[] ComputeBuffer",
        ["MaterialPropertyBlock.SetVectorArray.html", "MaterialPropertyBlock.SetMatrixArray.html",
         "Graphics.DrawMeshInstanced.html"],
    ),
    (
        "URP HDRP custom vertex color per instance MaterialPropertyBlock",
        ["renderer-shader-user-value", "MaterialPropertyBlock.html", "gpu-resident-drawer"],
    ),
    (
        "maximum of 1023 elements MaterialPropertyBlock arrays",
        ["MaterialPropertyBlock.CopySHCoefficientArraysFrom.html",
         "MaterialPropertyBlock.CopyProbeOcclusionArrayFrom.html",
         "Graphics.DrawMeshInstanced.html", "Graphics.RenderMeshInstanced.html"],
    ),
    (
        "Shader.PropertyToID faster than string name property lookup",
        ["Shader.PropertyToID.html", "MaterialPropertyBlock.SetColor.html"],
    ),
    (
        "light probe data arrays unity_SHAr unity_ProbesOcclusion instanced rendering",
        ["MaterialPropertyBlock.CopySHCoefficientArraysFrom.html",
         "LightProbes.CalculateInterpolatedLightAndOcclusionProbes.html",
         "Rendering.LightProbeUsage.CustomProvided.html"],
    ),
    (
        "pass null to disable per-renderer or per-material overrides",
        ["Renderer.SetPropertyBlock.html"],
    ),
    (
        "SetShaderUserValue unity_RendererUserValue per renderer color",
        ["MeshRenderer.SetShaderUserValue.html", "renderer-shader-user-value"],
    ),
    (
        "Terrain splat material property block",
        ["Terrain.SetSplatMaterialPropertyBlock.html", "Terrain.GetSplatMaterialPropertyBlock.html"],
    ),
    (
        "PerRendererData texture property read-only material inspector",
        ["SL-Properties", "Rendering.ShaderPropertyFlags.PerRendererData.html"],
    ),
    (
        "Material variants change materials at runtime limitation",
        ["materialvariant-concept"],
    ),
    (
        "RenderParams matProps material properties used for rendering",
        ["RenderParams-matProps.html", "RenderParams.html", "Graphics.RenderMesh.html"],
    ),
]



def search(searcher: Searcher, query: str, k: int) -> list[tuple[int, float]]:
    return searcher.search(query, top_k=k)


def hybrid_rank(
    searcher: Searcher,
    chunks: list[Chunk],
    embedder: HashEmbedder | None,
    query: str,
    bm25_k: int,
    final_k: int,
    alpha: float,
    per_file: int | None = 1,
) -> list[int]:
    """Return final_k doc ids using BM25 (+ optional dense fusion)."""
    bm25 = searcher.search(query, top_k=bm25_k)
    if not bm25:
        return []
    ids = [d for d, _ in bm25]
    if embedder is None or alpha >= 1.0:
        return dedupe_by_source(ids, chunks, per_file)[:final_k]
    ids = dedupe_by_source(ids, chunks, per_file)
    scores = {d: s for d, s in bm25}
    bnorm = minmax_normalize(scores)
    cand = [chunks[i].text for i in ids]
    qe = embedder.embed([query])[0]
    de = embedder.embed(cand)
    dscores = {i: s for i, s in zip(ids, cosine_scores(qe, de))}
    dnorm = minmax_normalize(dscores)
    fused = {d: alpha * bnorm[d] + (1 - alpha) * dnorm[d] for d in ids}
    return sorted(ids, key=lambda d: fused[d], reverse=True)[:final_k]


def evaluate(
    name: str,
    chunks: list[Chunk],
    run_rank,
    ks=(1, 3, 5, 10),
    verbose: bool = False,
    gold_set: list[tuple[str, list[str]]] | None = None,
) -> dict:
    """Score *run_rank* against a gold set.

    ``gold_set`` defaults to this module's GOLD (the 24-query baseline set) so
    existing callers are unaffected; eval_rag.py passes its own set so the
    extended/all gold sets are actually scored instead of silently falling back
    to the base 24.
    """
    queries = GOLD if gold_set is None else gold_set
    stats = {f"recall@{k}": 0.0 for k in ks}
    stats.update({f"hit@{k}": 0 for k in ks})
    rr = 0.0
    n = 0
    for query, gold in queries:
        ranked = run_rank(query)
        paths = [Path(chunks[i].source).as_posix() for i in ranked]
        n += 1
        first_rank = None
        for idx, p in enumerate(paths):
            if any(g in p for g in gold):
                first_rank = idx + 1
                break
        if first_rank:
            rr += 1.0 / first_rank
        for k in ks:
            top = paths[:k]
            if any(any(g in p for g in gold) for p in top):
                stats[f"hit@{k}"] += 1
                # recall@k = fraction of the gold list covered by top-k
                covered = sum(1 for g in gold if any(g in p for p in top))
                stats[f"recall@{k}"] += covered / len(gold)
        if verbose:
            ok = "OK " if first_rank else "MISS"
            print(f"  [{ok}] {query[:58]:<58} gold={gold[0][:40]!r} "
                  f"rank={first_rank} top1={Path(paths[0]).name if paths else '-'}")
    out = {"config": name, "queries": n, "MRR": round(rr / n, 4)}
    for k in ks:
        out[f"hit@{k}"] = round(stats[f"hit@{k}"] / n, 3)
        out[f"recall@{k}"] = round(stats[f"recall@{k}"] / n, 3)
    return out
