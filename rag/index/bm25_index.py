"""BM25 index over corpus chunks — reuses the proven retrieval.py + unity_tokenizer.

Index text = ``to_index_text`` (clean text + synthetic aux fields, optionally
disabled for ablation) + path/title field terms repeated ``path_boost`` times.
Defaults are the measured-best legacy config: word tokenizer, fuzziness=0,
min_should_match=0.6.
"""
from __future__ import annotations

from retrieval import InvertedIndex, Searcher
from unity_tokenizer import WordTokenizer, path_terms
from rag.corpus.schema import CorpusChunk, to_index_text


def doc_tokens(
    chunk: CorpusChunk,
    source: str,
    title: str,
    tok: WordTokenizer,
    path_boost: int = 3,
    *,
    aux: bool = True,
) -> list[str]:
    """Terms for one chunk: index text (aux optional) + path terms x path_boost."""
    toks = tok.tokenize(to_index_text(chunk, title, aux=aux))
    if path_boost:
        toks = toks + path_terms(source) * path_boost
    return toks


def build_bm25(
    chunks: list[CorpusChunk],
    sources: list[str],
    titles: list[str],
    *,
    path_boost: int = 3,
    aux: bool = True,
    verbose: bool = True,
) -> tuple[InvertedIndex, Searcher]:
    """Build the word-tokenizer BM25 index; doc ids are positional chunk ids."""
    tok = WordTokenizer()
    index = InvertedIndex()
    from tqdm import tqdm
    iterable = tqdm(range(len(chunks)), desc="BM25 index") if verbose else range(len(chunks))
    for i in iterable:
        index.add_document(i, doc_tokens(chunks[i], sources[i], titles[i], tok,
                                         path_boost, aux=aux))
    index.finalize()
    searcher = Searcher(index, tokenizer=tok, fuzziness=0, min_should_match=0.6)
    return index, searcher


def new_searcher(index: InvertedIndex, *, min_should_match: float = 0.6) -> Searcher:
    return Searcher(index, tokenizer=WordTokenizer(), fuzziness=0,
                    min_should_match=min_should_match)
