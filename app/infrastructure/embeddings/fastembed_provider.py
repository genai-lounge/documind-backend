"""EmbeddingProvider adapter for FastEmbed: open-source models run locally (ONNX).

No API key or network is needed after the model's one-time download. FastEmbed
is an optional dependency (`uv sync --extra fastembed`), so it is imported only
when this adapter is actually built.
"""

import logging
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol

from app.application.ports import EmbeddingProvider
from app.domain.errors import EmbeddingUnavailable
from app.domain.models import EmbeddingModel, Vector

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
NOT_INSTALLED = (
    "FastEmbed is not installed. Run `uv sync --extra fastembed` in backend/, "
    "or choose another EMBEDDING_PROVIDER."
)
FAILED = "The local embedding model failed."


class TextEmbeddingModel(Protocol):
    """The part of fastembed.TextEmbedding this adapter uses."""

    def passage_embed(self, texts: Iterable[str], **kwargs: Any) -> Iterable[Any]: ...

    def query_embed(
        self, query: str | Iterable[str], **kwargs: Any
    ) -> Iterable[Any]: ...


class FastEmbedProvider(EmbeddingProvider):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        cache_dir: str | None = None,
        batch_size: int = 128,
        model: TextEmbeddingModel | None = None,
        dimensions: int | None = None,
    ) -> None:
        """`model` and `dimensions` let tests supply a stand-in instead of a real model."""
        if model is None:
            model, dimensions = _load(model_name, cache_dir)
        if dimensions is None:
            raise ValueError("dimensions are required when a model is supplied.")
        self.model = EmbeddingModel(name=model_name, dimensions=dimensions)
        self._model = model
        self._batch_size = batch_size

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        return self._run(
            lambda: self._model.passage_embed(list(texts), batch_size=self._batch_size),
            len(texts),
            "document",
        )

    def embed_query(self, text: str) -> Vector:
        return self._run(lambda: self._model.query_embed(text), 1, "query")[0]

    def _run(
        self, embed: Callable[[], Iterable[Any]], count: int, input_type: str
    ) -> list[Vector]:
        started = time.perf_counter()
        try:
            vectors = [tuple(float(x) for x in vector) for vector in embed()]
        except Exception as exc:
            logger.exception("embedding.failed", extra={"provider": "fastembed"})
            raise EmbeddingUnavailable(FAILED) from exc
        logger.info(
            "embedding.completed",
            extra={
                "provider": "fastembed",
                "model": self.model.name,
                "input_type": input_type,
                "texts": count,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return vectors


def _load(model_name: str, cache_dir: str | None) -> tuple[TextEmbeddingModel, int]:
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise ImportError(NOT_INSTALLED) from exc

    known = {m["model"]: m["dim"] for m in TextEmbedding.list_supported_models()}
    if model_name not in known:
        raise ValueError(f"FastEmbed does not support the model '{model_name}'.")
    logger.info(
        "embedding.model_loading", extra={"provider": "fastembed", "model": model_name}
    )
    return TextEmbedding(model_name=model_name, cache_dir=cache_dir), int(
        known[model_name]
    )
