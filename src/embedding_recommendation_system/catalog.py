"""Load and validate the video catalog used by embedding experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID


class CatalogError(ValueError):
    """Raised when catalog data does not match the expected schema."""


@dataclass(frozen=True, slots=True)
class Video:
    """Metadata required by the related-video experiments."""

    id: str
    title: str
    authors: tuple[str, ...]
    category: str | None
    tag_ids: tuple[str, ...]
    tag_labels: tuple[str, ...]


def _required_text(value: object, field: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"video {index} has an invalid {field}")

    return value.strip()


def load_tag_labels(path: Path) -> dict[str, str]:
    """Load canonical tag IDs and their human-readable labels."""

    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != ["tag_id", "display_name"]:
            raise CatalogError(
                "tag label CSV must contain tag_id and display_name columns"
            )

        labels: dict[str, str] = {}

        for line_number, row in enumerate(reader, start=2):
            tag_id = (row["tag_id"] or "").strip()
            display_name = (row["display_name"] or "").strip()

            if not tag_id or not display_name:
                raise CatalogError(
                    f"tag label CSV has an empty value on line {line_number}"
                )

            if tag_id in labels:
                raise CatalogError(f"duplicate tag ID: {tag_id}")

            labels[tag_id] = display_name

    if not labels:
        raise CatalogError("tag label CSV is empty")

    return labels


def load_catalog(video_path: Path, tag_label_path: Path) -> tuple[Video, ...]:
    """Load videos and attach human-readable labels to their tag IDs."""

    tag_labels = load_tag_labels(tag_label_path)

    with video_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    raw_videos = payload.get("seed") if isinstance(payload, dict) else None

    if not isinstance(raw_videos, list):
        raise CatalogError("video catalog must contain a seed list")

    videos: list[Video] = []
    seen_ids: set[str] = set()

    for index, raw_video in enumerate(raw_videos):
        if not isinstance(raw_video, dict):
            raise CatalogError(f"video {index} must be an object")

        video_id = _required_text(raw_video.get("id"), "id", index)

        try:
            UUID(video_id)
        except ValueError as error:
            raise CatalogError(f"video {index} has an invalid UUID") from error

        if video_id in seen_ids:
            raise CatalogError(f"duplicate video ID: {video_id}")

        seen_ids.add(video_id)

        title = _required_text(raw_video.get("title"), "title", index)

        raw_authors = raw_video.get("authors")

        if not isinstance(raw_authors, list) or not raw_authors:
            raise CatalogError(f"video {index} has invalid authors")

        authors = tuple(
            _required_text(author, "author", index) for author in raw_authors
        )

        raw_category = raw_video.get("category")
        category = (
            None
            if raw_category is None
            else _required_text(raw_category, "category", index)
        )

        raw_tags = raw_video.get("tags")

        if not isinstance(raw_tags, list):
            raise CatalogError(f"video {index} has invalid tags")

        tag_ids = tuple(_required_text(tag, "tag", index) for tag in raw_tags)
        unknown_tags = sorted(set(tag_ids) - tag_labels.keys())

        if unknown_tags:
            unknown = ", ".join(unknown_tags)
            raise CatalogError(f"video {index} contains unknown tag IDs: {unknown}")

        videos.append(
            Video(
                id=video_id,
                title=title,
                authors=authors,
                category=category,
                tag_ids=tag_ids,
                tag_labels=tuple(tag_labels[tag_id] for tag_id in tag_ids),
            )
        )

    return tuple(videos)
