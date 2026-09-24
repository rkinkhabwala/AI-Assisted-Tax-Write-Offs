"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

    score(d) = sum over rankings r of 1 / (k + rank_r(d))

RRF combines rankings using only positions. Dense (cosine) and lexical (ts_rank_cd)
scores are on incomparable scales, so fusing the raw scores would need per-query
normalization that is fragile. k = 60 is the paper's value. It damps the advantage of the
very top ranks, so a passage ranked 3rd by both retrievers beats one ranked 1st by only
one, which is what we want: agreement between exact-token and semantic matching is a
strong relevance signal for legal text.
"""

from collections.abc import Hashable, Mapping, Sequence

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion[T: Hashable](
    rankings: Mapping[str, Sequence[T]],
    k: int = DEFAULT_RRF_K,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[T, float]]:
    """Fused (item, score) pairs, best first. Ties keep first-seen order.

    `weights` scales each ranking's contribution (weighted RRF); missing names weigh 1.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    scores: dict[T, float] = {}
    for name, ranking in rankings.items():
        weight = (weights or {}).get(name, 1.0)
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
