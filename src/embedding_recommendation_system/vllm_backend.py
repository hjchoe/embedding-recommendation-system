"""Generate embeddings with the Linux-only vLLM offline API."""

from __future__ import annotations

import sys
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


class VllmBackendError(RuntimeError):
    """Raised when vLLM cannot generate the requested embeddings."""


def embed_with_vllm(
    texts: Sequence[str],
    *,
    model: str,
    revision: str | None,
    max_model_len: int = 512,
) -> NDArray[np.float32]:
    """Generate one raw embedding row per input text."""

    if not texts:
        raise VllmBackendError("at least one input text is required")

    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise VllmBackendError("input texts must be non-empty strings")

    if max_model_len <= 0:
        raise VllmBackendError("max_model_len must be positive")

    if sys.platform != "linux":
        raise VllmBackendError(
            "the vLLM backend requires Linux and the gpu dependency extra"
        )

    try:
        from vllm import LLM
    except ModuleNotFoundError as error:
        if error.name == "vllm":
            raise VllmBackendError(
                "vLLM is not installed; run uv sync --extra gpu --frozen"
            ) from error

        raise

    llm = LLM(
        model=model,
        revision=revision,
        runner="pooling",
        max_model_len=max_model_len,
    )
    outputs = llm.embed(list(texts))

    if len(outputs) != len(texts):
        raise VllmBackendError("vLLM output count does not match the input count")

    return np.asarray(
        [output.outputs.embedding for output in outputs],
        dtype=np.float32,
    )
