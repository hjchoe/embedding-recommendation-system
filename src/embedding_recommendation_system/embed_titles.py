"""Generate a title-only embedding artifact."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from embedding_recommendation_system.artifacts import (
    EmbeddingRun,
    save_embedding_artifact,
)
from embedding_recommendation_system.catalog import load_catalog
from embedding_recommendation_system.inputs import build_title_inputs
from embedding_recommendation_system.vllm_backend import embed_with_vllm

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_REVISION = "66e95e324bebb9453d3b5be447c898dca1ba0eb0"
DEFAULT_MAX_MODEL_LEN = 512

Embedder = Callable[..., NDArray[np.float32]]


def file_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""

    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def generate_title_embeddings(
    *,
    video_catalog: Path,
    tag_labels: Path,
    output_dir: Path,
    model: str = DEFAULT_MODEL,
    revision: str | None = DEFAULT_REVISION,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    embedder: Embedder = embed_with_vllm,
) -> int:
    """Generate and save one title embedding per catalog video."""

    videos = load_catalog(video_catalog, tag_labels)
    inputs = build_title_inputs(videos)
    texts = [item.text for item in inputs]

    embeddings = embedder(
        texts,
        model=model,
        revision=revision,
        max_model_len=max_model_len,
    )

    run = EmbeddingRun(
        model=model,
        revision=revision,
        field="title",
        catalog_sha256=file_sha256(video_catalog),
    )

    save_embedding_artifact(
        output_dir,
        inputs,
        embeddings,
        run,
    )

    return len(inputs)


def build_parser() -> argparse.ArgumentParser:
    """Build the title-embedding command-line parser."""

    parser = argparse.ArgumentParser(
        description="Generate title embeddings for the video catalog."
    )
    parser.add_argument(
        "--video-catalog",
        type=Path,
        default=Path("data/catalog/Video.json"),
    )
    parser.add_argument(
        "--tag-labels",
        type=Path,
        default=Path("data/catalog/tag_labels.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
    )

    return parser


def main() -> None:
    """Run the title-embedding command."""

    arguments = build_parser().parse_args()

    count = generate_title_embeddings(
        video_catalog=arguments.video_catalog,
        tag_labels=arguments.tag_labels,
        output_dir=arguments.output,
        model=arguments.model,
        revision=arguments.revision,
        max_model_len=arguments.max_model_len,
    )

    print(f"Saved {count} title embeddings to {arguments.output}")
