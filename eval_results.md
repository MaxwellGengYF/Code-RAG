# Retrieval evaluation results

Same gold set as eval_retrieval.py; new engine = rag.search (BM25 + BGE-M3, RRF).

| config | MRR | hit@1 | hit@10 |
| --- | --- | --- | --- |
| **old baseline** | **0.875** | **0.833** | **0.917** |
| engine mode=hybrid rerank=False (ext-8) | 0.0 | 0.0 | 0.0 |
