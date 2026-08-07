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


@dataclass(frozen=True, slots=True)
class LoadedEmbeddingArtifact:
    """A validated embedding matrix and its row metadata."""

    run: EmbeddingRun
    video_ids: tuple[str, ...]
    texts: tuple[str, ...]
    embeddings: NDArray[np.float32]


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


def _required_manifest_text(manifest: dict[str, object], field: str) -> str:
    value = manifest.get(field)

    if not isinstance(value, str) or not value:
        raise ArtifactError(f"artifact manifest has an invalid {field}")

    return value


def _positive_manifest_integer(
    manifest: dict[str, object],
    field: str,
) -> int:
    value = manifest.get(field)

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArtifactError(f"artifact manifest has an invalid {field}")

    return value


def load_embedding_artifact(
    artifact_dir: Path,
) -> LoadedEmbeddingArtifact:
    """Load an embedding artifact after validating every stored component."""

    manifest_path = artifact_dir / "run.json"
    items_path = artifact_dir / "items.jsonl"
    embeddings_path = artifact_dir / "embeddings.npy"

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError("could not read artifact manifest") from error

    if not isinstance(manifest, dict):
        raise ArtifactError("artifact manifest must be an object")

    if manifest.get("schema_version") != 1:
        raise ArtifactError("artifact manifest has an unsupported schema version")

    if manifest.get("normalized") is not True:
        raise ArtifactError("artifact manifest must declare normalized embeddings")

    if manifest.get("dtype") != "float32":
        raise ArtifactError("artifact manifest must declare float32 embeddings")

    count = _positive_manifest_integer(manifest, "count")
    dimensions = _positive_manifest_integer(manifest, "dimensions")

    revision = manifest.get("revision")

    if revision is not None and (not isinstance(revision, str) or not revision):
        raise ArtifactError("artifact manifest has an invalid revision")

    run = EmbeddingRun(
        model=_required_manifest_text(manifest, "model"),
        revision=revision,
        field=_required_manifest_text(manifest, "field"),
        catalog_canonical_sha256=_required_manifest_text(
            manifest,
            "catalog_canonical_sha256",
        ),
    )

    try:
        lines = items_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ArtifactError("could not read artifact items") from error

    if len(lines) != count:
        raise ArtifactError("artifact item count does not match its manifest")

    video_ids: list[str] = []
    texts: list[str] = []

    for expected_index, line in enumerate(lines):
        if not line:
            raise ArtifactError("artifact items must not contain blank lines")

        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ArtifactError(
                f"artifact item {expected_index} is invalid JSON"
            ) from error

        if not isinstance(item, dict):
            raise ArtifactError(f"artifact item {expected_index} must be an object")

        index = item.get("index")
        video_id = item.get("video_id")
        text = item.get("text")

        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index != expected_index
        ):
            raise ArtifactError("artifact item indexes must be continuous")

        if not isinstance(video_id, str) or not video_id:
            raise ArtifactError(
                f"artifact item {expected_index} has an invalid video ID"
            )

        if not isinstance(text, str) or not text:
            raise ArtifactError(f"artifact item {expected_index} has invalid text")

        video_ids.append(video_id)
        texts.append(text)

    if len(set(video_ids)) != len(video_ids):
        raise ArtifactError("artifact items contain duplicate video IDs")

    try:
        embeddings = np.load(embeddings_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ArtifactError("could not read artifact embeddings") from error

    if embeddings.ndim != 2:
        raise ArtifactError("artifact embeddings must be two-dimensional")

    if embeddings.dtype != np.dtype("float32"):
        raise ArtifactError("artifact embeddings must use float32")

    if embeddings.shape != (count, dimensions):
        raise ArtifactError("artifact embedding shape does not match its manifest")

    if not np.isfinite(embeddings).all():
        raise ArtifactError("artifact embeddings contain non-finite values")

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-5, rtol=0.0):
        raise ArtifactError("artifact embeddings are not normalized")

    return LoadedEmbeddingArtifact(
        run=run,
        video_ids=tuple(video_ids),
        texts=tuple(texts),
        embeddings=embeddings,
    )
