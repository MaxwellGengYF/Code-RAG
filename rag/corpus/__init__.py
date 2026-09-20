"""Page extraction + LLM corpus generation."""
from rag.corpus.extract import PageInput, extract_page, render_markdown
from rag.corpus.schema import (
    SCHEMA_VERSION,
    EXTRACTOR_VERSION,
    CorpusChunk,
    PageCorpus,
    QA,
    make_chunk_uid,
    to_embed_text,
    to_index_text,
    snippet,
)
from rag.corpus.generate import (
    CorpusGenerationError,
    generate_page_corpus,
    PageGenStats,
)

__all__ = [
    "SCHEMA_VERSION", "EXTRACTOR_VERSION", "CorpusChunk", "CorpusGenerationError",
    "PageCorpus", "PageInput", "PageGenStats", "QA", "extract_page",
    "generate_page_corpus", "make_chunk_uid", "render_markdown", "snippet",
    "to_embed_text", "to_index_text",
]
