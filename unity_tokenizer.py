"""Identifier-aware tokenizer for Unity documentation (C# API + ShaderLab).

Unity docs are dense with code identifiers. Character n-grams destroy them:
"MaterialPropertyBlock" -> mat/ate/ter/eri/ria/ial/alp/lpr/pro/rop/ope/per/ert/
rty/tyb/ybl/blo/loc/ock -- 63% of which are generic English trigrams that occur in
every page, so BM25 loses all discriminative power and min_should_match becomes
meaningless.

This tokenizer emits, for each raw identifier/word:
  * the lowercased whole token              (materialpropertyblock)
  * camel/Pascal-case sub-words             (material, property, block)
  * useful prefixes for dotted paths        (renderer.setpropertyblock, setpropertyblock)
  * trailing-'s' folding and simple plural stripping
plus shader property names verbatim (_color, unity_shar).

Stopwords are dropped so that query length no longer inflates the term set.
"""
from __future__ import annotations

import functools
import re
from pathlib import Path

# Words that carry no topical signal in this corpus.
STOPWORDS = frozenset("""
a an the and or but if then else when while for to of in on at by with from into as is are was
were be been being do does did doing have has had having this that these those it its they them
their there here you your we our us he she his her him not no nor so such than too very can could
will would shall should may might must let via using use used uses using-named it's e.g. i.e. etc
also more most other all any each both few another same what which who whom whose how why
""".split())

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]+|[a-z0-9]+")


def split_identifier(tok: str) -> list[str]:
    """Split a code identifier into its camel/Pascal/snake_case parts."""
    parts: list[str] = []
    for chunk in tok.split("_"):
        if not chunk:
            continue
        parts.extend(_CAMEL_RE.findall(chunk))
    return [p.lower() for p in parts if p]


class WordTokenizer:
    """Identifier-aware word tokenizer implementing the retrieval.Tokenizer protocol."""
    __slots__ = ("keep_stopwords", "n")

    def __init__(self, keep_stopwords: bool = False, n: int = 2) -> None:
        # `n` is accepted for drop-in compatibility with NgramTokenizer call sites.
        self.n = n
        self.keep_stopwords = keep_stopwords

    @staticmethod
    def normalize(text: str) -> str:
        return text.lower()

    @staticmethod
    @functools.lru_cache(maxsize=262144)
    def _expand_word(word: str) -> tuple[str, ...]:
        w = word.lower()
        out: list[str] = [w]
        # dotted API paths: Renderer.SetPropertyBlock / Manual/foo-bar.html
        if "." in word:
            head, _, tail = word.rpartition(".")
            if head and tail:
                out.extend([head.lower(), tail.lower()])
        # split on the ORIGINAL casing -- lowercasing first destroys camelCase
        subs = split_identifier(word)
        if len(subs) > 1:
            out.extend(subs)
            # contiguous two-word compounds help "property block" <-> "propertyblock"
            for a, b in zip(subs, subs[1:]):
                out.append(a + b)
        # crude singular/plural fold for API-ish words (SetFloats -> setfloat)
        if w.endswith("ies") and len(w) > 4:
            out.append(w[:-3] + "y")
        elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            out.append(w[:-1])
        return tuple(dict.fromkeys(out))

    def tokenize(self, text: str, n: int | None = None) -> list[str]:
        tokens: list[str] = []
        low = text.lower()
        # keep leading '_' shader props like _Color and dotted names intact
        for m in re.finditer(r"[_A-Za-z][A-Za-z0-9_.]*", low):
            raw = text[m.start():m.end()]
            for term in self._expand_word(raw):
                if not self.keep_stopwords and term in STOPWORDS:
                    continue
                if len(term) < 2:
                    continue
                tokens.append(term)
        return tokens

    def query_terms(self, query: str) -> list[str]:
        """De-duplicated primary query terms (whole words only), for diagnostics."""
        seen: list[str] = []
        for m in re.finditer(r"[_A-Za-z][A-Za-z0-9_.]*", query):
            raw = query[m.start():m.end()]
            for t in self._expand_word(raw):
                if t in STOPWORDS or len(t) < 2:
                    continue
                if t not in seen:
                    seen.append(t)
        return seen


# --- path / title field terms -------------------------------------------------------------
# Many Unity API pages never mention their own symbol in the body prose: the chunk for
# Renderer.SetPropertyBlock.html says only "Lets you set or clear per-renderer ... overrides".
# Without the class name in the text, BM25 can never rank that page for a "Renderer..."
# query. Emitting terms from the source filename + page title fixes it (BM25F-lite).

_PATH_STRIP_RE = re.compile(r"\.(html?|md)$", re.I)


def path_terms(source: str) -> list[str]:
    """Terms derived from a doc file path, e.g. ScriptReference/Renderer.SetPropertyBlock.html."""
    p = Path(source.replace("\\", "/"))
    stem = _PATH_STRIP_RE.sub("", p.name)
    out: list[str] = []
    for raw in (stem, *stem.split("-"), *stem.split(".")):
        if not raw:
            continue
        out.extend(WordTokenizer._expand_word(raw))
        for sub in split_identifier(raw):
            if sub not in out:
                out.append(sub)
    # directory hints (Manual / ScriptReference) are not topical; skip them.
    return [t for t in dict.fromkeys(out) if len(t) > 2 and t not in STOPWORDS]
