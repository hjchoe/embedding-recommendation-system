from embedding_recommendation_system.catalog import Video
from embedding_recommendation_system.inputs import (
    EmbeddingInput,
    build_title_inputs,
)


def test_build_title_inputs_preserves_order_and_video_ids() -> None:
    videos = (
        Video(
            id="51f7d6d8-a5b0-4695-a997-61b047babdb7",
            title="Neuro-Symbolic AI",
            authors=("Author One",),
            category="Talk",
            tag_ids=(),
            tag_labels=(),
        ),
        Video(
            id="049e0179-4399-47c8-882a-6c38dfe195f1",
            title="Knowledge Representation",
            authors=("Author Two",),
            category=None,
            tag_ids=("knowledge_representation",),
            tag_labels=("Knowledge Representation",),
        ),
    )

    inputs = build_title_inputs(videos)

    assert inputs == (
        EmbeddingInput(
            video_id="51f7d6d8-a5b0-4695-a997-61b047babdb7",
            text="Neuro-Symbolic AI",
        ),
        EmbeddingInput(
            video_id="049e0179-4399-47c8-882a-6c38dfe195f1",
            text="Knowledge Representation",
        ),
    )
