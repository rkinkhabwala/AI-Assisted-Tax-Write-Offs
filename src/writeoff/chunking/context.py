"""Contextual retrieval: a short, LLM-written situating summary per child chunk.

The summary is prepended to the chunk only for embedding (`Chunk.embedding_text`); the
stored `text` stays the verbatim source passage used for display and citation.

The model sees the chunk's parent section rather than the whole document. That is enough
to say which provision a passage belongs to and what it covers, and it keeps each request
to about 3k input tokens. Prompt caching doesn't help here: a parent (at most 2k tokens)
is below Haiku 4.5's 4,096-token cache minimum.
"""

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

import anthropic

SYSTEM_PROMPT = """\
You write retrieval context for passages taken from U.S. federal tax law and IRS guidance.

You receive a document title, the section a passage belongs to, and the passage. Write one \
or two sentences that situate the passage: the document and provision it comes from, and \
the tax topic it addresses (for example, which expense, deduction, limit, or taxpayer type). \
Use the terms a small-business owner might search for.

Describe only what the section and passage say. Don't restate the rule in full, add facts, \
or give advice. Reply with the sentences only."""


class ContextSummaryError(RuntimeError):
    """The model returned no usable summary."""


class ContextSummarizer(ABC):
    @abstractmethod
    async def summarize(self, *, document_title: str, section_text: str, chunk_text: str) -> str:
        """One or two sentences situating `chunk_text` within its section and document."""


class ClaudeContextSummarizer(ContextSummarizer):
    def __init__(
        self, client: anthropic.AsyncAnthropic, model: str, *, max_tokens: int = 300
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    async def summarize(self, *, document_title: str, section_text: str, chunk_text: str) -> str:
        user = (
            f"<document_title>{document_title}</document_title>\n"
            f"<section>\n{section_text}\n</section>\n"
            f"<passage>\n{chunk_text}\n</passage>"
        )
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        if response.stop_reason == "refusal":
            raise ContextSummaryError(f"model declined to summarize a chunk of {document_title!r}")
        text = " ".join(b.text for b in response.content if b.type == "text").strip()
        if not text:
            raise ContextSummaryError(f"empty summary for a chunk of {document_title!r}")
        return text


class CachingSummarizer(ContextSummarizer):
    """Persists each summary to disk as soon as it is generated.

    Summaries are produced before embedding, so without this a failed embedding call (or a
    stopped run) would discard paid model calls. Keyed by model and full input, so a
    changed section, passage or model produces a fresh summary.
    """

    def __init__(self, inner: ContextSummarizer, cache_dir: Path, *, model: str) -> None:
        self._inner = inner
        self._dir = cache_dir
        self._model = model

    async def summarize(self, *, document_title: str, section_text: str, chunk_text: str) -> str:
        raw = "\0".join([self._model, document_title, section_text, chunk_text])
        path = self._dir / f"{hashlib.sha256(raw.encode()).hexdigest()}.txt"
        if path.is_file():
            return path.read_text(encoding="utf-8")
        summary = await self._inner.summarize(
            document_title=document_title, section_text=section_text, chunk_text=chunk_text
        )
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(summary, encoding="utf-8")
        tmp.replace(path)
        return summary
