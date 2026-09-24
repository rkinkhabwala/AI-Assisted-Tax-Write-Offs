"""Retrieval metrics over labeled targets (spec section 7a).

A target (a citation) is *found* at rank r when the r-th retrieved chunk is one of the
target's resolved chunks. Metrics are target-based, so retrieving three chunks of the
same provision counts once:

- Recall@k: share of a case's targets found in the top k.
- MRR: 1 / rank of the first result that finds any target (0 if none in the list).
- nDCG@10: graded (grade 2 = answers it, 1 = supports it). Each target is credited once,
  at the first rank that finds it. The ideal ranking lists targets by grade.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from writeoff.evals.dataset import Target


@dataclass(frozen=True, slots=True)
class CaseScores:
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    found: dict[str, int]  # target citation -> rank it was first found at


def score_case(
    retrieved: Sequence[UUID], targets: Sequence[Target], resolved: Mapping[str, frozenset[UUID]]
) -> CaseScores:
    """`resolved` maps each target citation to its chunk ids in the index under test."""
    found: dict[str, int] = {}
    first_relevant: int | None = None
    dcg = 0.0
    for rank, chunk_id in enumerate(retrieved, start=1):
        hits = [t for t in targets if chunk_id in resolved.get(t.citation, frozenset())]
        if hits and first_relevant is None:
            first_relevant = rank
        new = [t for t in hits if t.citation not in found]
        for t in new:
            found[t.citation] = rank
        if new and rank <= 10:
            dcg += (2 ** max(t.grade for t in new) - 1) / math.log2(rank + 1)
    ideal = sorted((t.grade for t in targets), reverse=True)[:10]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    n = len(targets)
    return CaseScores(
        recall_at_5=sum(r <= 5 for r in found.values()) / n,
        recall_at_10=sum(r <= 10 for r in found.values()) / n,
        mrr=1.0 / first_relevant if first_relevant else 0.0,
        ndcg_at_10=dcg / idcg if idcg else 0.0,
        found=found,
    )


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def best_threshold(in_scope: Sequence[float], out_of_scope: Sequence[float]) -> tuple[float, float]:
    """Top-score cutoff that best separates answerable from unanswerable queries.

    Returns (threshold, balanced accuracy); a query is flagged weak when its best
    rerank score is below the threshold.
    """
    if not in_scope or not out_of_scope:
        return 0.0, 0.0
    candidates = sorted({*in_scope, *out_of_scope, 0.0, 1.0})
    best = (0.0, 0.0)
    for threshold in candidates:
        tpr = sum(s >= threshold for s in in_scope) / len(in_scope)
        tnr = sum(s < threshold for s in out_of_scope) / len(out_of_scope)
        accuracy = (tpr + tnr) / 2
        if accuracy > best[1]:
            best = (threshold, accuracy)
    return best
