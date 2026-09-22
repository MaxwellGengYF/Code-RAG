"""Versioned prompts for LLM corpus generation.
``PROMPT_VERSION`` is recorded in the (diagnostic) ``gen_key``. It does NOT
force regeneration: only an md5 diff or a missing corpus file requeues pages,
so old corpus keeps the prompt generation it was built with. Bump it when the
task semantics change; new/changed pages pick the new prompt up automatically.
"""
from __future__ import annotations

# NOTE: PROMPT_VERSION stays "v2" — the wording was made generic and concise
# (deployment-specific examples removed) without changing the task semantics,
# so existing v2 corpus files remain valid.
PROMPT_VERSION = "v2"

SYSTEM_PROMPT = """You build a retrieval corpus. You receive ONE document as markdown.
Split it into retrieval chunks and return STRICT JSON — no markdown fences, no
commentary, nothing outside the JSON object.
Rules:
1. Chunk at heading boundaries; keep related paragraphs together, in source order.
2. NEVER split a code block: a fenced ``` block must lie entirely inside one chunk.
3. Each chunk's "text" MUST be a VERBATIM excerpt of the source markdown — exact
   characters, source order; no paraphrasing, summarizing, or re-formatting. The
   markdown renderer may insert odd spaces around punctuation (e.g. "Foo .bar");
   copy such spacing EXACTLY as it appears.
4. "heading_path": the breadcrumb of section titles the chunk lives under,
   starting with the document title.
5. Keep chunks at or under {max_chars} characters when possible; exceed it only to
   avoid splitting a code block.
6. Auxiliary fields are synthetic retrieval hints:
   - "summary": at most one sentence on what the chunk documents.
   - "keywords": 3-8 topical terms from or directly relevant to the chunk.
   - "synonyms": alternate phrasings, aliases, or symbols a user might search with.
   - "qa": up to 2 question/answer pairs the chunk answers, answers <= 1 sentence.
7. Skip chunks with no informational value (pure navigation, empty sections).
Return exactly this shape:
{{"chunks": [{{"heading_path": ["...", "..."], "text": "...", "summary": "...",
  "keywords": ["..."], "synonyms": ["..."], "qa": [{{"q": "...", "a": "..."}}]}}]}}
"""

USER_PROMPT = """Document: {source}
Title: {title}
Length: {char_len} chars (markdown{truncated_note})
Return the chunk JSON for this document now.
{source_markdown}"""

REPAIR_SUFFIX = """
Your previous response failed validation:
{errors}
Return the corrected STRICT JSON for the SAME document now. Remember: chunk
"text" values must be VERBATIM excerpts of the source markdown, no markdown
fences, no commentary."""


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
