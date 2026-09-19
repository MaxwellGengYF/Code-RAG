# Retrieval evaluation results

Same gold set as eval_retrieval.py; new engine = rag.search (BM25 + BGE-M3, RRF).

| config | MRR | hit@1 | hit@10 |
| --- | --- | --- | --- |
| **old baseline** | **0.875** | **0.833** | **0.917** |
| table bm25 path_boost=3 (ext-16) | 0.9062 | 0.875 | 0.938 |
| table bm25 path_boost=5 (ext-16) | 0.9062 | 0.875 | 0.938 |
