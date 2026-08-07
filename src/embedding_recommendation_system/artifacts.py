"""Persist normalized embedding artifacts for local analysis."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from embedding_recommendation_system.inputs import EmbeddingInput


class ArtifactError(ValueError):
    """Raised when an embedding artifact would be invalid."""


@dataclass(frozen=True, slots=True)
class EmbeddingRun:
    """Metadata required to reproduce an embedding run."""

    model: str
    revision: str | None
    field: str
    catalog_canonical_sha256: str


def normalize_embeddings(embeddings: ArrayLike) -> NDArray[np.float32]:
    """Convert an embedding matrix to finite unit-length float32 rows."""

    matrix = np.asarray(embeddings, dtype=np.float32)

    if matrix.ndim != 2:
        raise ArtifactError("embeddings must be a two-dimensional matrix")

    if not np.isfinite(matrix).all():
        raise ArtifactError("embeddings must contain only finite values")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)

    if np.any(norms == 0):
        raise ArtifactError("embeddings must not contain zero-length rows")

    normalized = matrix / norms
    return normalized.astype(np.float32, copy=False)


def save_embedding_artifact(
    output_dir: Path,
    inputs: Sequence[EmbeddingInput],
    embeddings: ArrayLike,
    run: EmbeddingRun,
) -> None:
    """Save normalized embeddings and their row-to-video mapping."""

    if not inputs:
        raise ArtifactError("cannot save an empty embedding artifact")

    video_ids = [item.video_id for item in inputs]

    if len(set(video_ids)) != len(video_ids):
        raise ArtifactError("embedding inputs contain duplicate video IDs")

    matrix = normalize_embeddings(embeddings)

    if matrix.shape[0] != len(inputs):
        raise ArtifactError("embedding row count must match the number of inputs")

    output_dir.mkdir(parents=True, exist_ok=False)

    np.save(
        output_dir / "embeddings.npy",
        matrix,
        allow_pickle=False,
    )

    with (output_dir / "items.jsonl").open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for index, item in enumerate(inputs):
            record = {
                "index": index,
                "video_id": item.video_id,
                "text": item.text,
            }
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            file.write("\n")

    manifest = {
        "schema_version": 1,
        **asdict(run),
        "count": matrix.shape[0],
        "dimensions": matrix.shape[1],
        "dtype": str(matrix.dtype),
        "normalized": True,
    }

    with (output_dir / "run.json").open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")
