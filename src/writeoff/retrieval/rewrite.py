"""Query rewriting: plain-English questions -> tax-law search terms (spec section 3).

"Can I write off my truck?" shares few words with § 280F or Pub 463. The rewriter adds the
terminology the sources use. Its phrases feed keyword search as alternatives, and they are
appended to the text embedded for semantic search. The reranker still scores against the
user's original question, so a poor rewrite can widen recall but cannot change what
"relevant" means.
"""

import re
from abc import ABC, abstractmethod

import anthropic

SYSTEM_PROMPT = """\
You turn a U.S. small-business owner's tax question into search phrases for a library of \
the Internal Revenue Code, Treasury Regulations, IRS publications and form instructions.

Return 3 to 8 short search phrases, one per line: the formal tax concepts, Internal Revenue \
Code sections (written like "§ 280F"), forms and publication topics that govern the \
question. Prefer the exact terms those sources use.

Example: for "can I write off my truck" return:
vehicle expense deduction
listed property
§ 280F
standard mileage rate
Pub 463 car expenses

Do not answer the question. Output only the phrases, with no numbering or commentary."""

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_MAX_TERM_CHARS = 80


class QueryRewriteError(RuntimeError):
    """The rewriter could not produce search terms; callers fall back to the raw query."""


class QueryRewriter(ABC):
    @abstractmethod
    async def rewrite(self, query: str) -> list[str]:
        """Search phrases in the sources' terminology (possibly empty)."""


class NoRewrite(QueryRewriter):
    """Search with the user's words only (evaluation baseline, or when rewriting is off)."""

    async def rewrite(self, query: str) -> list[str]:
        return []


class ClaudeQueryRewriter(QueryRewriter):
    def __init__(self, client: anthropic.AsyncAnthropic, model: str, *, max_terms: int = 8) -> None:
        self._client = client
        self._model = model
        self._max_terms = max_terms

    async def rewrite(self, query: str) -> list[str]:
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=200,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": query}],
            )
        except anthropic.APIError as exc:
            raise QueryRewriteError(f"rewrite request failed: {exc}") from exc
        if response.stop_reason == "refusal":
            raise QueryRewriteError("model declined to rewrite the query")
        text = "\n".join(b.text for b in response.content if b.type == "text")
        terms = parse_terms(text, self._max_terms)
        if not terms:
            raise QueryRewriteError("rewrite returned no terms")
        return terms


def parse_terms(text: str, max_terms: int) -> list[str]:
    """One phrase per line; strips bullets, numbering and quotes; drops duplicates."""
    terms: list[str] = []
    for line in text.splitlines():
        term = _BULLET.sub("", line).strip().strip("\"'").strip()
        if term and len(term) <= _MAX_TERM_CHARS and term.lower() not in {t.lower() for t in terms}:
            terms.append(term)
    return terms[:max_terms]
