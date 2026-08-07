import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from embedding_recommendation_system.embed_titles import (
    generate_title_embeddings,
)


def test_generate_title_embeddings_wires_complete_pipeline(
    tmp_path: Path,
) -> None:
    video_catalog = tmp_path / "Video.json"
    tag_labels = tmp_path / "tag_labels.csv"
    output_dir = tmp_path / "output"

    video_catalog.write_text(
        json.dumps(
            {
                "seed": [
                    {
                        "id": "51f7d6d8-a5b0-4695-a997-61b047babdb7",
                        "title": "Neuro-Symbolic AI",
                        "authors": ["Luís C. Lamb"],
                        "category": "Talk",
                        "tags": ["logical_reasoning"],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    tag_labels.write_text(
        "tag_id,display_name\nlogical_reasoning,Logical Reasoning\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_embedder(
        texts: Sequence[str],
        *,
        model: str,
        revision: str | None,
        max_model_len: int,
    ) -> NDArray[np.float32]:
        captured.update(
            {
                "texts": list(texts),
                "model": model,
                "revision": revision,
                "max_model_len": max_model_len,
            }
        )
        return np.array([[3.0, 4.0]], dtype=np.float32)

    count = generate_title_embeddings(
        video_catalog=video_catalog,
        tag_labels=tag_labels,
        output_dir=output_dir,
        model="example/model",
        revision="example-revision",
        max_model_len=256,
        embedder=fake_embedder,
    )

    manifest = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    items = [
        json.loads(line)
        for line in (output_dir / "items.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    embeddings = np.load(
        output_dir / "embeddings.npy",
        allow_pickle=False,
    )

    assert count == 1
    assert captured == {
        "texts": ["Neuro-Symbolic AI"],
        "model": "example/model",
        "revision": "example-revision",
        "max_model_len": 256,
    }
    assert items[0]["video_id"] == ("51f7d6d8-a5b0-4695-a997-61b047babdb7")
    assert items[0]["text"] == "Neuro-Symbolic AI"
    assert manifest["field"] == "title"
    assert manifest["model"] == "example/model"
    assert manifest["revision"] == "example-revision"
    assert (
        manifest["catalog_sha256"]
        == hashlib.sha256(video_catalog.read_bytes()).hexdigest()
    )
    np.testing.assert_allclose(
        embeddings,
        np.array([[0.6, 0.8]], dtype=np.float32),
    )
