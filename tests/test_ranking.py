import numpy as np
import pytest

from embedding_recommendation_system.ranking import (
    RankingError,
    rank_neighbors,
)


def test_rank_neighbors_orders_by_cosine_and_excludes_self() -> None:
    rankings = rank_neighbors(
        ["video-a", "video-b", "video-c"],
        np.array(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        top_k=2,
    )

    first = rankings[0]

    assert first.video_id == "video-a"
    assert [neighbor.video_id for neighbor in first.neighbors] == [
        "video-b",
        "video-c",
    ]
    assert [neighbor.rank for neighbor in first.neighbors] == [1, 2]
    assert [neighbor.similarity for neighbor in first.neighbors] == pytest.approx(
        [0.8, 0.0]
    )
    assert all(
        ranking.video_id not in {neighbor.video_id for neighbor in ranking.neighbors}
        for ranking in rankings
    )


def test_rank_neighbors_uses_input_order_to_break_ties() -> None:
    rankings = rank_neighbors(
        ["query", "first", "second"],
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ],
            dtype=np.float32,
        ),
        top_k=2,
    )

    assert [neighbor.video_id for neighbor in rankings[0].neighbors] == [
        "first",
        "second",
    ]


def test_rank_neighbors_caps_top_k_at_available_candidates() -> None:
    rankings = rank_neighbors(
        ["video-a", "video-b"],
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        top_k=10,
    )

    assert len(rankings[0].neighbors) == 1
    assert len(rankings[1].neighbors) == 1


def test_rank_neighbors_rejects_duplicate_video_ids() -> None:
    with pytest.raises(RankingError, match="must be unique"):
        rank_neighbors(
            ["duplicate", "duplicate"],
            np.eye(2, dtype=np.float32),
        )
