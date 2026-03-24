"""Async NVD harvester for LangChain document ingestion."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional
from uuid import NAMESPACE_URL, uuid5

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from config import AppConfig
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from src.core.factory import get_collection_name, get_embedding_dimension, get_embeddings


logger = logging.getLogger(__name__)


async def run_ingestion(config: AppConfig) -> None:
    """Run the standalone ingestion flow for NVD documents.

    Input:
        Application configuration with NVD, embeddings, and Qdrant settings.
    Output:
        Fetches NVD documents and incrementally upserts them into Qdrant.
    Security context:
        Keeps ingestion as a standalone write-time operation separate from the
        read-only LangGraph inference workflow.
    """

    harvester = NvdHarvester(config=config)
    documents = await harvester.fetch_documents()

    batch_size = 10
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        await harvester.sync_documents(documents=batch)
    return documents


class NvdHarvester:
    """Async harvester that fetches CVEs from the NVD API and indexes them.

    Input:
        Uses HTTP access to the NVD API plus LangChain embeddings and Qdrant
        vector store integrations.
    Output:
        Returns LangChain documents and incrementally inserts or updates them in
        the configured Qdrant collection.
    Security context:
        Restricts ingestion to normalized NVD CVE data and preserves source,
        CVSS, and version metadata for grounded downstream retrieval.
    """

    def __init__(
        self,
        config: AppConfig,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initialize the NVD harvester.

        Input:
            Application configuration and an optional injected async HTTP client.
        Output:
            Stores the configuration and HTTP client used for NVD fetches.
        Security context:
            Keeps network access explicit and configurable, avoiding hidden
            external dependencies during ingestion.
        """

        self._config = config
        self._http_client = http_client
        self._embeddings = get_embeddings()
        self._client = QdrantClient(url=self._config.qdrant_url)
        self._collection_name = get_collection_name(prefix=self._config.qdrant_collection)
        self._vector_store: Optional[QdrantVectorStore] = None

    async def fetch_documents(self, keyword_search: Optional[str] = None) -> list[Document]:
        """Fetch CVEs from the NVD API and map them into LangChain documents.

        Input:
            An optional NVD keyword filter.
        Output:
            Returns LangChain documents derived from NVD CVE items.
        Security context:
            Normalizes untrusted external JSON into bounded `Document` objects
            with explicit metadata fields required for retrieval filtering.
        """

        params: dict[str, Any] = {"resultsPerPage": 100}
        if keyword_search:
            params["keywordSearch"] = keyword_search

        client = self._http_client or httpx.AsyncClient(
            base_url="https://services.nvd.nist.gov",
            timeout=30.0,
        )
        owns_client = self._http_client is None
        try:
            response = await client.get("/rest/json/cves/2.0", params=params)
            response.raise_for_status()
            payload = response.json()
        finally:
            if owns_client:
                await client.aclose()

        vulnerabilities = payload.get("vulnerabilities", [])
        return [self._cve_to_document(item) for item in vulnerabilities]
    

    @retry(
        retry=retry_if_exception_type(Exception), # Specifically catches the Google 429
        wait=wait_exponential(multiplier=2, min=15, max=60), # Wait at least 15s (as per your error)
        stop=stop_after_attempt(5),
        before_sleep=lambda retry_state: logger.warning(f"Rate limited! Retrying in {retry_state.next_action.sleep}s...")
    )
    async def sync_documents(
        self,
        documents: list[Document],
    ) -> Optional[QdrantVectorStore]:
        """Synchronize NVD documents into Qdrant incrementally.

        Input:
            A list of LangChain documents derived from NVD CVE items.
        Output:
            Returns a `QdrantVectorStore` or `None` when there are no documents.
        Security context:
            Uses only normalized document content and metadata while preserving
            stable CVE-based identifiers so repeated ingestion updates existing
            entries instead of recreating the full collection.
        """

        if not documents:
            logger.info("No NVD documents to synchronize into Qdrant.")
            return None

        try:
            await asyncio.to_thread(self._ensure_collection_exists, self._client)
            vector_store = self._get_vector_store()
            document_ids = [self._document_id(document=document) for document in documents]
            await asyncio.to_thread(
                vector_store.add_documents,
                documents,
                ids=document_ids,
            )
        except Exception:
            logger.exception(
                "Failed to synchronize %s NVD documents into Qdrant collection '%s'.",
                len(documents),
                self._collection_name,
            )
            raise

        logger.info(
            "Synchronized %s NVD documents into Qdrant collection '%s'.",
            len(documents),
            self._collection_name,
        )
        return vector_store

    def _get_vector_store(self) -> QdrantVectorStore:
        """Return a lazily initialized Qdrant vector store for the collection.

        Input:
            Uses the configured client, collection name, and embedding model.
        Output:
            Returns a `QdrantVectorStore` bound to the ensured collection.
        Security context:
            Delays vector-store construction until after collection creation so
            ingestion does not probe or depend on a missing Qdrant collection.
        """

        if self._vector_store is None:
            self._vector_store = QdrantVectorStore(
                client=self._client,
                collection_name=self._collection_name,
                embedding=self._embeddings,
                vector_name=self._config.qdrant_vector_name,
            )
        return self._vector_store

    def _ensure_collection_exists(self, client: QdrantClient) -> None:
        """Create the Qdrant collection only when it does not already exist.

        Input:
            A synchronous Qdrant client connected to the configured server.
        Output:
            Ensures the configured collection is ready for incremental upserts.
        Security context:
            Restricts collection management to the configured collection name so
            ingestion cannot mutate unrelated vector data.
        """

        collections = client.get_collections()
        existing_names = {collection.name for collection in collections.collections}
        if self._collection_name in existing_names:
            return

        client.create_collection(
            collection_name=self._collection_name,
            vectors_config={
                self._config.qdrant_vector_name: VectorParams(
                    size=self._config.qdrant_vector_size or get_embedding_dimension(),
                    distance=self._distance_enum(),
                )
            },
        )

    def _document_id(self, document: Document) -> str:
        """Build a stable Qdrant point identifier for an ingested document.

        Input:
            A normalized LangChain document with CVE metadata.
        Output:
            Returns a stable document identifier used for upsert semantics.
        Security context:
            Uses the authoritative `cve_id` field so repeated ingestion updates
            the same vulnerability record instead of producing duplicates.
        """

        cve_id = document.metadata.get("cve_id")
        if cve_id:
            return str(uuid5(NAMESPACE_URL, f"cve:{cve_id}"))
        return str(uuid5(NAMESPACE_URL, document.page_content))

    def _distance_enum(self) -> Distance:
        """Translate configured distance text into a Qdrant distance enum.

        Input:
            Uses the configured Qdrant distance metric string.
        Output:
            Returns the matching Qdrant distance enum.
        Security context:
            Restricts collection schema creation to approved distance metrics.
        """

        mapping = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclid": Distance.EUCLID,
            "manhattan": Distance.MANHATTAN,
        }
        value = self._config.qdrant_distance.lower()
        if value not in mapping:
            raise ValueError("Unsupported Qdrant distance metric: {value}".format(value=value))
        return mapping[value]

    def _cve_to_document(self, item: dict[str, Any]) -> Document:
        """Convert a single NVD CVE item into a LangChain document.

        Input:
            A vulnerability item from the NVD API response.
        Output:
            Returns a LangChain `Document` with CVE description and metadata.
        Security context:
            Preserves CVE identifiers, CVSS data, and affected software version
            fields so retrieval remains verifiable and filterable.
        """

        cve = item.get("cve", {})
        cve_id = cve.get("id", "unknown-cve")
        descriptions = cve.get("descriptions", [])
        description = self._first_english_description(descriptions=descriptions)
        metrics = cve.get("metrics", {})
        cvss = self._extract_cvss(metrics=metrics)
        versions = self._extract_software_versions(configurations=cve.get("configurations", []))

        return Document(
            page_content=description,
            metadata={
                "source": "nvd",
                "cve_id": cve_id,
                "cvss": cvss,
                "software_versions": versions,
            },
        )

    def _first_english_description(self, descriptions: list[dict[str, Any]]) -> str:
        """Select the first English CVE description from the NVD payload.

        Input:
            A list of NVD description entries.
        Output:
            Returns the English description text or an empty string.
        Security context:
            Limits stored content to the descriptive text needed for retrieval
            and avoids carrying unrelated API fields into vector indexing.
        """

        for description in descriptions:
            if description.get("lang") == "en":
                return str(description.get("value", ""))
        return ""

    def _extract_cvss(self, metrics: dict[str, Any]) -> Optional[float]:
        """Extract the primary CVSS base score from NVD metrics.

        Input:
            The NVD metrics dictionary for a CVE item.
        Output:
            Returns the most relevant CVSS base score when present.
        Security context:
            Preserves severity context in metadata so downstream ranking and
            generation can reason over authoritative NVD scores.
        """

        for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            metric_list = metrics.get(key, [])
            if metric_list:
                cvss_data = metric_list[0].get("cvssData", {})
                score = cvss_data.get("baseScore")
                if score is not None:
                    return float(score)
        return None

    def _extract_software_versions(self, configurations: list[dict[str, Any]]) -> list[str]:
        """Extract affected software version strings from NVD configurations.

        Input:
            The NVD configurations array for a CVE item.
        Output:
            Returns a de-duplicated list of software version or CPE strings.
        Security context:
            Preserves affected version context in metadata so retrieval filters
            can be grounded in authoritative NVD data.
        """

        versions: list[str] = []
        for configuration in configurations:
            for node in configuration.get("nodes", []):
                for cpe_match in node.get("cpeMatch", []):
                    criteria = cpe_match.get("criteria")
                    version_start = cpe_match.get("versionStartIncluding")
                    version_end = cpe_match.get("versionEndExcluding")
                    values = [value for value in (criteria, version_start, version_end) if value]
                    for value in values:
                        text = str(value)
                        if text not in versions:
                            versions.append(text)
        return versions
