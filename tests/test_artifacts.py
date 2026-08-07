import json
from pathlib import Path

import numpy as np
import pytest

from embedding_recommendation_system.artifacts import (
    ArtifactError,
    EmbeddingRun,
    load_embedding_artifact,
    normalize_embeddings,
    save_embedding_artifact,
)
from embedding_recommendation_system.inputs import EmbeddingInput


def test_normalize_embeddings_returns_unit_float32_rows() -> None:
    embeddings = np.array(
        [
            [3.0, 4.0],
            [0.0, 2.0],
        ]
    )

    normalized = normalize_embeddings(embeddings)

    assert normalized.dtype == np.float32
    np.testing.assert_allclose(
        normalized,
        np.array(
            [
                [0.6, 0.8],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )


def test_normalize_embeddings_rejects_zero_length_rows() -> None:
    with pytest.raises(ArtifactError, match="zero-length rows"):
        normalize_embeddings([[0.0, 0.0]])


def test_save_embedding_artifact_writes_reproducible_files(
    tmp_path: Path,
) -> None:
    inputs = (
        EmbeddingInput(
            video_id="51f7d6d8-a5b0-4695-a997-61b047babdb7",
            text="Neuro-Symbolic AI",
        ),
        EmbeddingInput(
            video_id="049e0179-4399-47c8-882a-6c38dfe195f1",
            text="Knowledge Representation",
        ),
    )
    run = EmbeddingRun(
        model="Qwen/Qwen3-Embedding-0.6B",
        revision="example-revision",
        field="title",
        catalog_canonical_sha256="example-checksum",
    )
    output_dir = tmp_path / "title-run"

    save_embedding_artifact(
        output_dir,
        inputs,
        [[3.0, 4.0], [0.0, 2.0]],
        run,
    )

    embeddings = np.load(
        output_dir / "embeddings.npy",
        allow_pickle=False,
    )
    items = [
        json.loads(line)
        for line in (output_dir / "items.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    manifest = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))

    np.testing.assert_allclose(
        embeddings,
        np.array(
            [
                [0.6, 0.8],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    assert items == [
        {
            "index": 0,
            "text": "Neuro-Symbolic AI",
            "video_id": "51f7d6d8-a5b0-4695-a997-61b047babdb7",
        },
        {
            "index": 1,
            "text": "Knowledge Representation",
            "video_id": "049e0179-4399-47c8-882a-6c38dfe195f1",
        },
    ]
    assert manifest == {
        "catalog_canonical_sha256": "example-checksum",
        "count": 2,
        "dimensions": 2,
        "dtype": "float32",
        "field": "title",
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "normalized": True,
        "revision": "example-revision",
        "schema_version": 1,
    }


def test_save_embedding_artifact_rejects_row_count_mismatch(
    tmp_path: Path,
) -> None:
    inputs = (
        EmbeddingInput(
            video_id="51f7d6d8-a5b0-4695-a997-61b047babdb7",
            text="Neuro-Symbolic AI",
        ),
        EmbeddingInput(
            video_id="049e0179-4399-47c8-882a-6c38dfe195f1",
            text="Knowledge Representation",
        ),
    )
    run = EmbeddingRun(
        model="Qwen/Qwen3-Embedding-0.6B",
        revision=None,
        field="title",
        catalog_canonical_sha256="example-checksum",
    )

    with pytest.raises(ArtifactError, match="row count"):
        save_embedding_artifact(
            tmp_path / "invalid-run",
            inputs,
            [[1.0, 0.0]],
            run,
        )


def test_load_embedding_artifact_reconstructs_valid_artifact(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "valid-run"
    inputs = (
        EmbeddingInput(
            video_id="51f7d6d8-a5b0-4695-a997-61b047babdb7",
            text="Neuro-Symbolic AI",
        ),
        EmbeddingInput(
            video_id="049e0179-4399-47c8-882a-6c38dfe195f1",
            text="Knowledge Representation",
        ),
    )
    run = EmbeddingRun(
        model="Qwen/Qwen3-Embedding-0.6B",
        revision="example-revision",
        field="title",
        catalog_canonical_sha256="a" * 64,
    )

    save_embedding_artifact(
        output_dir,
        inputs,
        [[3.0, 4.0], [0.0, 2.0]],
        run,
    )

    loaded = load_embedding_artifact(output_dir)

    assert loaded.run == run
    assert loaded.video_ids == tuple(item.video_id for item in inputs)
    assert loaded.texts == tuple(item.text for item in inputs)
    assert loaded.embeddings.shape == (2, 2)
    assert loaded.embeddings.dtype == np.float32


def test_load_embedding_artifact_rejects_discontinuous_indexes(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "invalid-index"
    inputs = (
        EmbeddingInput(
            video_id="51f7d6d8-a5b0-4695-a997-61b047babdb7",
            text="Neuro-Symbolic AI",
        ),
    )
    run = EmbeddingRun(
        model="example-model",
        revision=None,
        field="title",
        catalog_canonical_sha256="a" * 64,
    )

    save_embedding_artifact(output_dir, inputs, [[1.0, 0.0]], run)

    items_path = output_dir / "items.jsonl"
    items_path.write_text(
        items_path.read_text(encoding="utf-8").replace(
            '"index": 0',
            '"index": 1',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="indexes must be continuous"):
        load_embedding_artifact(output_dir)


def test_load_embedding_artifact_rejects_non_normalized_matrix(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "invalid-matrix"
    inputs = (
        EmbeddingInput(
            video_id="51f7d6d8-a5b0-4695-a997-61b047babdb7",
            text="Neuro-Symbolic AI",
        ),
    )
    run = EmbeddingRun(
        model="example-model",
        revision=None,
        field="title",
        catalog_canonical_sha256="a" * 64,
    )

    save_embedding_artifact(output_dir, inputs, [[1.0, 0.0]], run)

    np.save(
        output_dir / "embeddings.npy",
        np.array([[2.0, 0.0]], dtype=np.float32),
        allow_pickle=False,
    )

    with pytest.raises(ArtifactError, match="not normalized"):
        load_embedding_artifact(output_dir)
