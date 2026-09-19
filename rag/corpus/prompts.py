"""Versioned prompts for LLM corpus generation.

``PROMPT_VERSION`` is part of the ``gen_key`` — bumping it (or the model, the
extractor, or the schema) forces regeneration of every corpus file.
"""
from __future__ import annotations

PROMPT_VERSION = "v2"

SYSTEM_PROMPT = """You are a corpus builder for a retrieval system over Unity documentation.

You receive ONE documentation page as markdown. Split it into retrieval chunks and
return STRICT JSON — no markdown fences, no commentary, nothing outside the JSON object.

Rules:
1. Chunk at heading boundaries. Keep related paragraphs together, in source order.
2. NEVER split a signature block or code sample: a fenced ```csharp block must be
   entirely inside one chunk.
3. Each chunk's "text" MUST be a VERBATIM excerpt of the source markdown: copy the
   exact characters in source order, do not paraphrase, translate, summarize,
   re-format, or reorder. The renderer sometimes inserts spaces around punctuation
   where the HTML used inline spans (e.g. a heading rendered as
   "WheelFrictionCurve .extremumSlip" or "Rigidbody .velocity") — copy such
   spacing EXACTLY as it appears, odd as it looks.
4. "heading_path" is the breadcrumb of section titles the chunk lives under,
   including the page title as the first element.
5. Keep each chunk at or under {max_chars} characters when possible; a chunk may
   exceed it only to avoid splitting a code block.
6. Auxiliary fields are retrieval-oriented and synthetic:
   - "summary": at most one sentence describing what the chunk documents.
   - "keywords": 3-8 topical terms found in or directly relevant to the chunk.
   - "synonyms": alternate phrasings / script aliases a user might search with
     (e.g. "linearVelocity (script alias)" for Rigidbody.velocity).
   - "qa": up to 2 question/answer pairs the chunk answers, each answer at most
     one sentence.
7. Skip chunks that carry no documentation value (pure navigation, empty sections).

Return exactly this shape:
{{"chunks": [{{"heading_path": ["...", "..."], "text": "...", "summary": "...",
"keywords": ["..."], "synonyms": ["..."], "qa": [{{"q": "...", "a": "..."}}]}}]}}
"""

USER_PROMPT = """Page source: {source}
Page title: {title}
Page length: {char_len} chars (markdown{truncated_note})

Return the chunk JSON for this page now.

{source_markdown}"""

REPAIR_SUFFIX = """

Your previous response failed validation:
{errors}

Return the corrected STRICT JSON for the SAME page now. Remember: chunk "text"
values must be VERBATIM excerpts of the source markdown, no markdown fences, no
commentary."""


def build_user_prompt(source: str, title: str, markdown: str, char_len: int,
                      truncated: bool) -> str:
    return USER_PROMPT.format(
        source=source,
        title=title,
        char_len=char_len,
        truncated_note=", truncated" if truncated else "",
        source_markdown=markdown,
    )


def build_repair_prompt(user_prompt: str, errors: list[str]) -> str:
    return user_prompt + REPAIR_SUFFIX.format(errors="\n".join(f"- {e}" for e in errors))


def system_prompt(max_chars: int) -> str:
    return SYSTEM_PROMPT.format(max_chars=max_chars)
