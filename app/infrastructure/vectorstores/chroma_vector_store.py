"""Vector store adapter for Chroma: embedded on local disk, or Chroma Cloud.

Implements VectorIndexWriter, VectorSearcher, and VectorIndexReader. Each
embedding model gets its own collection (Chroma locks a collection to one
vector length, and vectors of different models must never be compared).

Vectors are stored at unit length, and scores are computed here as cosine
similarity from the returned vectors. Results are therefore correct whatever
distance the server's index uses: Chroma Cloud's SPANN index ignores custom
settings, so its distance metric isn't under our control.
"""

import hashlib
import logging
import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.collection_configuration import CreateCollectionConfiguration
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

from app.application.ports import VectorIndexReader, VectorIndexWriter, VectorSearcher
from app.domain.errors import StorageUnavailable
from app.domain.models import (
    Chunk,
    EmbeddedChunk,
    EmbeddingModel,
    SearchHit,
    Vector,
    chunk_id,
)
from app.domain.similarity import cosine_similarity

logger = logging.getLogger(__name__)

FAILED = "The vector store is unavailable."
DEFAULT_CLOUD_HOST = "api.trychroma.com"
T = TypeVar("T")


def collection_name(model: EmbeddingModel) -> str:
    """E.g. "chunks-text-embedding-3-small-1536-1b2c3d4e".

    Chroma allows 3-512 characters from [a-zA-Z0-9._-]; model names like
    "BAAI/bge-small-en-v1.5" are cleaned, and a short hash of the exact name
    keeps two similar names from colliding.
    """
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", model.name).strip("-._")[:100]
    digest = hashlib.sha256(model.name.encode()).hexdigest()[:8]
    return f"chunks-{slug}-{model.dimensions}-{digest}"


class ChromaVectorStore(VectorIndexWriter, VectorSearcher, VectorIndexReader):
    def __init__(
        self, client: ClientAPI, index_configuration: CreateCollectionConfiguration
    ) -> None:
        """Prefer `local()` or `cloud()`; tests may pass their own client."""
        self._client = client
        self._index_configuration = index_configuration
        self._collections: dict[EmbeddingModel, Collection] = {}

    @classmethod
    def local(cls, path: Path) -> "ChromaVectorStore":
        """Embedded Chroma persisted in `path`. Single process; no network or key."""
        path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        return cls(client, {"hnsw": {"space": "cosine"}})

    @classmethod
    def cloud(
        cls,
        api_key: str,
        tenant: str | None = None,
        database: str | None = None,
        host: str = DEFAULT_CLOUD_HOST,
    ) -> "ChromaVectorStore":
        """Chroma Cloud. Tenant and database may be left out when the API key
        belongs to a single database. Checks the connection immediately, so bad
        credentials fail at startup rather than on the first upload."""
        client = chromadb.CloudClient(
            tenant=tenant,
            database=database,
            api_key=api_key,
            cloud_host=host,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        client.heartbeat()
        client.list_collections(limit=1)
        return cls(client, {"spann": {"space": "cosine"}})

    def upsert(
        self,
        model: EmbeddingModel,
        document_id: str,
        filename: str,
        chunks: Sequence[EmbeddedChunk],
    ) -> None:
        if not chunks:
            return
        embeddings: list[Sequence[float]] = [_unit(c.vector) for c in chunks]
        self._guard(
            lambda: self._collection(model).upsert(
                ids=[chunk_id(document_id, c.chunk.index) for c in chunks],
                embeddings=embeddings,
                documents=[c.chunk.text for c in chunks],
                metadatas=[_metadata(document_id, filename, c.chunk) for c in chunks],
            ),
            "upsert",
        )

    def delete_document(self, model: EmbeddingModel, document_id: str) -> None:
        self._guard(
            lambda: self._collection(model).delete(where={"document_id": document_id}),
            "delete",
        )

    def search(
        self, model: EmbeddingModel, vector: Vector, top_k: int
    ) -> list[SearchHit]:
        unit_query = _unit(vector)
        query: list[Sequence[float]] = [unit_query]
        result = self._guard(
            lambda: self._collection(model).query(
                query_embeddings=query,
                n_results=top_k,
                include=["documents", "metadatas", "embeddings"],
            ),
            "search",
        )
        documents = (result["documents"] or [[]])[0]
        metadatas = (result["metadatas"] or [[]])[0]
        embeddings = result["embeddings"]
        stored_vectors = embeddings[0] if embeddings is not None else []
        scored = sorted(
            (
                (cosine_similarity(unit_query, [float(x) for x in stored]), text, meta)
                for text, meta, stored in zip(
                    documents, metadatas, stored_vectors, strict=True
                )
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return [
            SearchHit(
                rank=rank,
                score=score,
                document_id=str(meta["document_id"]),
                filename=str(meta["filename"]),
                chunk=_chunk(text or "", meta),
            )
            for rank, (score, text, meta) in enumerate(scored, start=1)
        ]

    def document_chunks(self, model: EmbeddingModel, document_id: str) -> list[Chunk]:
        result = self._guard(
            lambda: self._collection(model).get(
                where={"document_id": document_id},
                include=["documents", "metadatas"],
            ),
            "get",
        )
        chunks = [
            _chunk(text or "", meta)
            for text, meta in zip(
                result["documents"] or [], result["metadatas"] or [], strict=True
            )
        ]
        return sorted(chunks, key=lambda c: c.index)

    def _collection(self, model: EmbeddingModel) -> Collection:
        if model not in self._collections:
            self._collections[model] = self._client.get_or_create_collection(
                name=collection_name(model),
                embedding_function=None,
                configuration=self._index_configuration,
                metadata={
                    "embedding_model": model.name,
                    "dimensions": model.dimensions,
                },
            )
        return self._collections[model]

    def _guard(self, operation: Callable[[], T], name: str) -> T:
        try:
            return operation()
        except Exception as exc:
            logger.exception("vector_store.failed", extra={"operation": name})
            raise StorageUnavailable(FAILED) from exc


def _metadata(document_id: str, filename: str, chunk: Chunk) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "document_id": document_id,
        "filename": filename,
        "chunk_index": chunk.index,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
    }
    if chunk.page_number is not None:
        metadata["page_number"] = chunk.page_number
    return metadata


def _chunk(text: str, meta: Any) -> Chunk:
    page = meta.get("page_number")
    return Chunk(
        index=int(meta["chunk_index"]),
        page_number=int(page) if page is not None else None,
        text=text,
        start_char=int(meta["start_char"]),
        end_char=int(meta["end_char"]),
    )


def _unit(vector: Sequence[float]) -> list[float]:
    """Scale to length 1: then cosine and Euclidean distance rank results identically."""
    norm = math.sqrt(math.sumprod(vector, vector))
    return [x / norm for x in vector] if norm else list(vector)
