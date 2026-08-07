import json
from pathlib import Path

import pytest

from embedding_recommendation_system.catalog import (
    CatalogError,
    load_catalog,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _write_fixture(
    tmp_path: Path,
    tags: list[str],
) -> tuple[Path, Path]:
    video_path = tmp_path / "Video.json"
    tag_path = tmp_path / "tag_labels.csv"

    payload = {
        "seed": [
            {
                "id": "51f7d6d8-a5b0-4695-a997-61b047babdb7",
                "title": "Neuro-Symbolic AI",
                "authors": ["Luís C. Lamb"],
                "category": "Talk",
                "tags": tags,
            }
        ]
    }

    video_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    tag_path.write_text(
        "tag_id,display_name\nlogical_reasoning,Logical Reasoning\n",
        encoding="utf-8",
    )

    return video_path, tag_path


def test_load_catalog_preserves_metadata_and_maps_tags(
    tmp_path: Path,
) -> None:
    video_path, tag_path = _write_fixture(
        tmp_path,
        ["logical_reasoning"],
    )

    videos = load_catalog(video_path, tag_path)

    assert len(videos) == 1
    assert videos[0].title == "Neuro-Symbolic AI"
    assert videos[0].authors == ("Luís C. Lamb",)
    assert videos[0].category == "Talk"
    assert videos[0].tag_ids == ("logical_reasoning",)
    assert videos[0].tag_labels == ("Logical Reasoning",)


def test_load_catalog_rejects_unknown_tags(tmp_path: Path) -> None:
    video_path, tag_path = _write_fixture(tmp_path, ["unknown_tag"])

    with pytest.raises(CatalogError, match="unknown tag IDs: unknown_tag"):
        load_catalog(video_path, tag_path)


def test_checked_in_catalog_is_valid() -> None:
    videos = load_catalog(
        REPOSITORY_ROOT / "data/catalog/Video.json",
        REPOSITORY_ROOT / "data/catalog/tag_labels.csv",
    )

    assert len(videos) == 204
    assert len({video.id for video in videos}) == 204
