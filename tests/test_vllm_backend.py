import sys
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from embedding_recommendation_system import vllm_backend
from embedding_recommendation_system.vllm_backend import (
    VllmBackendError,
    embed_with_vllm,
)


def test_embed_with_vllm_rejects_non_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vllm_backend.sys, "platform", "win32")

    with pytest.raises(VllmBackendError, match="requires Linux"):
        embed_with_vllm(
            ["Neuro-Symbolic AI"],
            model="example/model",
            revision=None,
        )


def test_embed_with_vllm_uses_pooling_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLLM:
        init_arguments: ClassVar[dict[str, object]] = {}
        prompts: ClassVar[list[str]] = []

        def __init__(self, **arguments: object) -> None:
            type(self).init_arguments = arguments

        def embed(self, prompts: list[str]) -> list[SimpleNamespace]:
            type(self).prompts = prompts
            return [
                SimpleNamespace(outputs=SimpleNamespace(embedding=[1.0, 2.0])),
                SimpleNamespace(outputs=SimpleNamespace(embedding=[3.0, 4.0])),
            ]

    fake_vllm = ModuleType("vllm")
    fake_vllm.LLM = FakeLLM

    monkeypatch.setattr(vllm_backend.sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    embeddings = embed_with_vllm(
        ["First title", "Second title"],
        model="Qwen/Qwen3-Embedding-0.6B",
        revision="example-revision",
        max_model_len=512,
    )

    assert FakeLLM.init_arguments == {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "example-revision",
        "runner": "pooling",
        "max_model_len": 512,
    }
    assert FakeLLM.prompts == ["First title", "Second title"]
    assert embeddings.dtype == np.float32
    np.testing.assert_array_equal(
        embeddings,
        np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=np.float32,
        ),
    )


def test_embed_with_vllm_rejects_empty_inputs() -> None:
    with pytest.raises(VllmBackendError, match="at least one"):
        embed_with_vllm(
            [],
            model="example/model",
            revision=None,
        )
