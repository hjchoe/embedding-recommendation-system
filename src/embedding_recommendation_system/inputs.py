"""Build deterministic text inputs for embedding experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from embedding_recommendation_system.catalog import Video


@dataclass(frozen=True, slots=True)
class EmbeddingInput:
    """Text to embed and the video UUID it represents."""

    video_id: str
    text: str


def build_title_inputs(
    videos: Sequence[Video],
) -> tuple[EmbeddingInput, ...]:
    """Build one raw-title embedding input per video."""

    return tuple(
        EmbeddingInput(
            video_id=video.id,
            text=video.title,
        )
        for video in videos
    )
