"""Qdrant vector-store integration for read-only inference."""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.vectorstores import VectorStore
from qdrant_client import QdrantClient

from config import AppConfig
from langchain_core.documents import Document
from src.core.factory import get_embedding_dimension, get_embeddings, get_collection_name

try:
    from langchain_qdrant import QdrantVectorStore
except ImportError:  # pragma: no cover - dependency is optional at import time.
    QdrantVectorStore = None

try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.http.models import Distance, VectorParams
except ImportError:  # pragma: no cover - dependency is optional at import time.
    AsyncQdrantClient = None
    Distance = None
    VectorParams = None


class LangChainEmbeddingFactory:
    """Factory for LangChain embedding models used in retrieval and ingestion.

    Input:
        Uses application configuration to select an embedding model.
    Output:
        Returns a LangChain embeddings implementation.
    Security context:
        Centralizes embedding model construction so outbound embedding requests
        remain explicit and configurable.
    """

    def __init__(self, config: AppConfig) -> None:
        """Initialize the embedding factory.

        Input:
            Application configuration with embedding model settings.
        Output:
            Stores configuration for later embedding construction.
        Security context:
            Keeps provider and model selection explicit and outside business
            logic call sites.
        """

        self._config = config

    def build(self) -> Any:
        """Build the configured LangChain embeddings instance.

        Input:
            Uses the configured embedding provider and model.
        Output:
            Returns a LangChain embeddings implementation.
        Security context:
            Uses environment-managed credentials through LangChain integrations
            rather than embedding secrets in source code.
        """

        return get_embeddings()


class QdrantVectorStoreClient:
    """LangChain-compatible Qdrant vector-store adapter.

    Input:
        Uses Qdrant connection settings and LangChain embeddings.
    Output:
        Exposes a LangChain vector store and collection-management helpers.
    Security context:
        Keeps all vector-store access bound to the configured Qdrant collection
        so vulnerability data is not written to arbitrary backends.
    """

    def __init__(
        self,
        config: AppConfig,
        embeddings: Optional[Any] = None,
        client: Optional[Any] = None,
    ) -> None:
        """Initialize the LangChain Qdrant adapter.

        Input:
            Application configuration, an optional embeddings implementation,
            and an optional injected Qdrant client.
        Output:
            Stores the collection, embeddings, and async client.
        Security context:
            Allows dependency injection for tests while keeping production
            retrieval limited to the configured Qdrant endpoint.
        """

        if client is not None:
            self._client = client
        elif AsyncQdrantClient is None:
            raise ImportError("qdrant-client is required to use QdrantVectorStoreClient")
        else:
            self._client = QdrantClient(url=config.qdrant_url)
        self._embeddings = embeddings or LangChainEmbeddingFactory(config=config).build()
        self._collection_name = get_collection_name(prefix=config.qdrant_collection)
        self._vector_name = config.qdrant_vector_name
        self._vector_size = get_embedding_dimension()
        self._distance = config.qdrant_distance.lower()

    def build_vector_store(self) -> VectorStore:
        """Build a LangChain Qdrant vector store bound to the configured collection.

        Input:
            Uses configured embeddings and Qdrant client state.
        Output:
            Returns a LangChain `QdrantVectorStore`.
        Security context:
            Ensures all retrieval and ingestion operations target the configured
            collection and use the intended embedding model.
        """

        if QdrantVectorStore is None:
            raise ImportError("langchain-qdrant is required to build the vector store")
        return QdrantVectorStore(
            client=self._client,
            collection_name=self._collection_name,
            embedding=self._embeddings,
            vector_name=self._vector_name,
        )

    async def ensure_collection(self) -> None:
        """Ensure the configured Qdrant collection exists with the expected schema.

        Input:
            Uses collection settings from application configuration.
        Output:
            Creates the collection when missing and leaves it unchanged
            otherwise.
        Security context:
            Constrains vector schema management to the configured collection so
            retrieval setup does not mutate unrelated vector data.
        """

        if Distance is None or VectorParams is None:
            raise ImportError("qdrant-client is required to manage Qdrant collections")

        collections = await self._client.get_collections()
        existing_names = {collection.name for collection in collections.collections}
        if self._collection_name in existing_names:
            return

        await self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config={
                self._vector_name: VectorParams(
                    size=self._vector_size,
                    distance=self._distance_enum(),
                )
            },
        )

    async def aadd_documents(self, documents: list[Document]) -> None:
        """Add LangChain documents to the configured vector store.

        Input:
            A list of LangChain documents.
        Output:
            Persists the documents to Qdrant using LangChain's vector-store API.
        Security context:
            Stores only attributable chunk content and metadata supplied by the
            caller, preserving source identifiers in vector-store payloads.
        """

        if not documents:
            return
        vector_store = self.build_vector_store()
        await vector_store.aadd_documents(documents=documents)

    def _distance_enum(self) -> Any:
        """Translate configured distance text into a Qdrant distance enum.

        Input:
            Uses the configured distance metric string.
        Output:
            Returns the corresponding Qdrant distance enum value.
        Security context:
            Restricts collection creation to known distance metrics to avoid
            malformed schema configuration at runtime.
        """

        if Distance is None:
            raise ImportError("qdrant-client is required to resolve Qdrant distance values")

        mapping = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclid": Distance.EUCLID,
            "manhattan": Distance.MANHATTAN,
        }
        if self._distance not in mapping:
            raise ValueError("Unsupported Qdrant distance metric: {value}".format(value=self._distance))
        return mapping[self._distance]
