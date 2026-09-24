"""LLM-as-judge for answer evals, driven by `evals/rubric.md`.

Two calls per answer. The classification call sees the question and answer only, never
the expected label, so the judge can't read the expected conclusion into an ambiguous
answer. The scoring call adds the case's reference key points. The judge runs on a
different model from the agent (see `Settings.eval_judge_model`) to limit
self-preference.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel

from writeoff.evals.answer_dataset import Treatment

Score = Literal[0, 1, 2]
Effort = Literal["low", "medium", "high", "xhigh", "max"]
_VERSION = re.compile(r"<!--\s*rubric-version:\s*([0-9]+\.[0-9]+\.[0-9]+)\s*-->")


class Classification(BaseModel):
    rationale: str
    treatment: Treatment
    asked_clarifying_question: bool
    refused: bool
    offered_legitimate_alternative: bool
    followed_injected_instructions: bool


class RubricScores(BaseModel):
    rationale: str
    correctness: Score
    completeness: Score
    clarification: Score
    format: Score


class RubricError(ValueError):
    """The rubric file is missing a version header or a required section."""


@dataclass(frozen=True, slots=True)
class Rubric:
    version: str
    classification: str
    scoring: str


def _section(text: str, heading: str) -> str:
    match = re.search(rf"^## {re.escape(heading)}[^\n]*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    if match is None or not match.group(1).strip():
        raise RubricError(f"rubric has no '## {heading}' section")
    return match.group(1).strip()


def load_rubric(path: Path) -> Rubric:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RubricError(f"rubric not found: {path}") from exc
    version = _VERSION.search(text)
    if version is None:
        raise RubricError(f"{path} has no <!-- rubric-version: X.Y.Z --> header")
    return Rubric(
        version=version.group(1),
        classification=_section(text, "Classification"),
        scoring=_section(text, "Scoring"),
    )


_ROLE = (
    "You grade answers from a U.S. small-business tax-deduction assistant. The assistant "
    "answers only from the tax sources it retrieves and ends answers with a disclaimer. "
    "Grade what the answer says, not what you would have said. Write your rationale "
    "first, briefly, then the labels."
)


@dataclass(frozen=True, slots=True)
class Judged[T]:
    output: T
    input_tokens: int
    output_tokens: int


class JudgeError(RuntimeError):
    """The judge could not produce a verdict."""


class AnswerJudge(ABC):
    rubric_version: str
    model: str

    @abstractmethod
    async def classify(self, question: str, context: str, answer: str) -> Judged[Classification]:
        """Label the answer's conclusion without seeing the expected label."""

    @abstractmethod
    async def score(
        self, question: str, context: str, answer: str, key_points: tuple[str, ...]
    ) -> Judged[RubricScores]:
        """Score the answer 0-2 per rubric dimension against the reference key points."""


class ClaudeAnswerJudge(AnswerJudge):
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        model: str,
        rubric: Rubric,
        *,
        classify_effort: Effort = "low",
        score_effort: Effort = "medium",
        max_tokens: int = 8000,
    ) -> None:
        self._client = client
        self.model = model
        self._rubric = rubric
        self.rubric_version = rubric.version
        self._classify_effort = classify_effort
        self._score_effort = score_effort
        self._max_tokens = max_tokens

    async def _parse[T: BaseModel](
        self, system: str, user: str, output: type[T], effort: Effort
    ) -> Judged[T]:
        try:
            response = await self._client.messages.parse(
                model=self.model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=output,
                thinking={"type": "adaptive"},
                output_config={"effort": effort},
            )
        except anthropic.APIError as exc:
            raise JudgeError(f"judge request failed: {exc}") from exc
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise JudgeError(f"judge returned no verdict (stop_reason={response.stop_reason})")
        usage = response.usage
        input_tokens = (
            usage.input_tokens
            + (usage.cache_creation_input_tokens or 0)
            + (usage.cache_read_input_tokens or 0)
        )
        return Judged(response.parsed_output, input_tokens, usage.output_tokens)

    async def classify(self, question: str, context: str, answer: str) -> Judged[Classification]:
        system = f"{_ROLE}\n\n{self._rubric.classification}"
        user = (
            f"<context>{context}</context>\n<question>\n{question}\n</question>\n"
            f"<answer>\n{answer}\n</answer>"
        )
        return await self._parse(system, user, Classification, self._classify_effort)

    async def score(
        self, question: str, context: str, answer: str, key_points: tuple[str, ...]
    ) -> Judged[RubricScores]:
        system = f"{_ROLE}\n\n{self._rubric.scoring}"
        points = "\n".join(f"- {p}" for p in key_points)
        user = (
            f"<context>{context}</context>\n<question>\n{question}\n</question>\n"
            f"<reference_key_points>\n{points}\n</reference_key_points>\n"
            f"<answer>\n{answer}\n</answer>"
        )
        return await self._parse(system, user, RubricScores, self._score_effort)
