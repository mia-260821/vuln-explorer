"""Async NVD harvester for LangChain document ingestion."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional
from uuid import NAMESPACE_URL, uuid5

import httpx
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_qdrant import QdrantVectorStore
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ingestion.chunker import chunk_record
from models import VulnerabilityRecord


logger = logging.getLogger(__name__)
DEFAULT_CHUNK_SIZE = 1000


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
        vectorstore: VectorStore,
        http_client: Optional[httpx.AsyncClient] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
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
        self._http_client = http_client
        self._vector_store = vectorstore
        self._chunk_size = chunk_size

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
        documents: list[Document] = []
        for item in vulnerabilities:
            documents.extend(self._cve_to_documents(item))
        return documents
    

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
            vector_store = self._vector_store
            document_ids = [self._document_id(document=document) for document in documents]
            await asyncio.to_thread(
                vector_store.add_documents,
                documents,
                ids=document_ids,
            )
        except Exception:
            logger.exception(
                "Failed to synchronize %s NVD documents into Qdrant collection.",
                len(documents),
            )
            raise

        logger.info(
            "Synchronized %s NVD documents into Qdrant collection.",
            len(documents),
        )
        return vector_store


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

        chunk_id = document.metadata.get("chunk_id")
        if chunk_id:
            return str(chunk_id)
        cve_id = document.metadata.get("cve_id")
        if cve_id:
            return str(uuid5(NAMESPACE_URL, f"cve:{cve_id}"))
        return str(uuid5(NAMESPACE_URL, document.page_content))

    def _cve_to_documents(self, item: dict[str, Any]) -> list[Document]:
        """Convert a single NVD CVE item into chunked LangChain documents.

        Input:
            A vulnerability item from the NVD API response.
        Output:
            Returns chunked LangChain `Document` objects with CVE metadata.
        Security context:
            Preserves CVE identifiers, CVSS data, affected software version
            fields, and reference URLs so retrieval remains verifiable and
            filterable after chunking.
        """

        record = self._cve_to_record(item=item)
        chunks = chunk_record(record=record, chunk_size=self._chunk_size)
        return [
            Document(page_content=chunk.text, metadata={**chunk.metadata, "chunk_id": chunk.chunk_id})
            for chunk in chunks
        ]

    def _cve_to_record(self, item: dict[str, Any]) -> VulnerabilityRecord:
        """Convert a single NVD CVE item into a normalized vulnerability record.

        Input:
            A vulnerability item from the NVD API response.
        Output:
            Returns a `VulnerabilityRecord` suitable for shared chunking logic.
        Security context:
            Normalizes untrusted API fields into the canonical ingestion model
            so all indexed content flows through one metadata-preserving path.
        """

        cve = item.get("cve", {})
        cve_id = str(cve.get("id", "unknown-cve"))
        descriptions = cve.get("descriptions", [])
        description = self._first_english_description(descriptions=descriptions)
        metrics = cve.get("metrics", {})
        cvss = self._extract_cvss(metrics=metrics)
        versions = self._extract_software_versions(configurations=cve.get("configurations", []))
        references = self._extract_references(cve.get("references", []))

        return VulnerabilityRecord(
            source_id=cve_id,
            summary=description,
            details=description,
            severity=self._cvss_to_severity(cvss),
            cvss_score=cvss,
            platform=versions,
            references=references,
            metadata={
                "source": "nvd",
                "cve_id": cve_id,
                "cvss": cvss,
                "software_versions": versions,
            },
        )

    def _extract_references(self, references: list[dict[str, Any]]) -> list[str]:
        """Extract NVD reference URLs from a CVE item.

        Input:
            The NVD references array for a CVE item.
        Output:
            Returns a de-duplicated list of advisory or vendor reference URLs.
        Security context:
            Keeps retrievable source links attached to the ingested record so
            downstream answers can cite authoritative remediation context.
        """

        urls: list[str] = []
        for reference in references:
            url = reference.get("url")
            if not url:
                continue
            text = str(url)
            if text not in urls:
                urls.append(text)
        return urls

    def _cvss_to_severity(self, cvss: Optional[float]) -> str:
        """Map a CVSS score to an NVD-style severity label.

        Input:
            An optional CVSS base score.
        Output:
            Returns a normalized severity label for chunk metadata and headers.
        Security context:
            Derives severity from authoritative NVD scoring to avoid storing
            unverified qualitative labels from external transformations.
        """

        if cvss is None:
            return "UNKNOWN"
        if cvss >= 9.0:
            return "CRITICAL"
        if cvss >= 7.0:
            return "HIGH"
        if cvss >= 4.0:
            return "MEDIUM"
        if cvss > 0.0:
            return "LOW"
        return "NONE"

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
