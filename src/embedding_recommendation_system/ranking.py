"""Compute deterministic exact cosine-neighbor rankings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from embedding_recommendation_system.artifacts import (
    ArtifactError,
    normalize_embeddings,
)


class RankingError(ValueError):
    """Raised when neighbor rankings cannot be computed safely."""


@dataclass(frozen=True, slots=True)
class Neighbor:
    """One ranked related-video candidate."""

    video_id: str
    rank: int
    similarity: float


@dataclass(frozen=True, slots=True)
class VideoRanking:
    """Ordered related videos for one query video."""

    video_id: str
    neighbors: tuple[Neighbor, ...]


def rank_neighbors(
    video_ids: Sequence[str],
    embeddings: ArrayLike,
    *,
    top_k: int = 10,
) -> tuple[VideoRanking, ...]:
    """Rank exact cosine neighbors for every video."""

    if len(video_ids) < 2:
        raise RankingError("at least two videos are required")

    if any(not video_id for video_id in video_ids):
        raise RankingError("video IDs must be non-empty")

    if len(set(video_ids)) != len(video_ids):
        raise RankingError("video IDs must be unique")

    if top_k <= 0:
        raise RankingError("top_k must be positive")

    try:
        matrix = normalize_embeddings(embeddings)
    except ArtifactError as error:
        raise RankingError("embedding matrix is invalid") from error

    if matrix.shape[0] != len(video_ids):
        raise RankingError("embedding row count must match the video ID count")

    similarities = matrix @ matrix.T
    candidate_indexes = np.arange(len(video_ids))
    result_count = min(top_k, len(video_ids) - 1)
    rankings: list[VideoRanking] = []

    for query_index, video_id in enumerate(video_ids):
        candidates = candidate_indexes[candidate_indexes != query_index]
        scores = similarities[query_index, candidates]

        order = np.lexsort(
            (
                candidates,
                -scores,
            )
        )
        selected = candidates[order[:result_count]]

        neighbors = tuple(
            Neighbor(
                video_id=video_ids[int(candidate)],
                rank=rank,
                similarity=float(
                    np.clip(
                        similarities[query_index, candidate],
                        -1.0,
                        1.0,
                    )
                ),
            )
            for rank, candidate in enumerate(selected, start=1)
        )

        rankings.append(
            VideoRanking(
                video_id=video_id,
                neighbors=neighbors,
            )
        )

    return tuple(rankings)
