"""Chunking logic for vulnerability descriptions."""

from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import DocumentChunk, VulnerabilityRecord


def record_to_document(record: VulnerabilityRecord) -> Document:
    """Convert a normalized vulnerability record into a LangChain document.

    Input:
        A normalized vulnerability record.
    Output:
        Returns a LangChain `Document` with advisory content and metadata.
    Security context:
        Preserves source IDs and structured metadata so later retrieval and
        generation stages remain attributable and filterable.
    """

    metadata: dict[str, Any] = {
        "source_id": record.source_id,
        "severity": record.severity,
        "cvss_score": record.cvss_score,
        "platform": record.platform,
        "is_exploited": record.is_exploited,
        "references": record.references,
    }
    metadata.update(record.metadata)
    platform_text = ", ".join(record.platform) if record.platform else "unknown"
    context_header = " | ".join(
        [
            f"Source ID: {record.source_id}",
            f"Severity: {record.severity}",
            f"Platform: {platform_text}",
        ]
    )
    return Document(
        page_content=f"{context_header}\nSummary: {record.summary}\n\nDetails: {record.details}".strip(),
        metadata=metadata,
    )


def chunk_record(record: VulnerabilityRecord, chunk_size: int) -> list[DocumentChunk]:
    """Split a vulnerability record into semantically ordered text chunks.

    Input:
        A normalized vulnerability record and target chunk size.
    Output:
        Returns deterministic chunks suitable for indexing in Qdrant or a
        secondary keyword engine.
    Security context:
        Keeps each chunk attached to the original source ID and metadata so
        retrieval results can be filtered and fully attributed.
    """

    document = record_to_document(record=record)
    if not document.page_content:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=max(int(chunk_size * 0.1), 0),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    split_documents = splitter.split_documents([document])
    chunks: list[DocumentChunk] = []
    for index, split_document in enumerate(split_documents):
        chunks.append(
            DocumentChunk(
                chunk_id=str(uuid5(NAMESPACE_URL, f"{record.source_id}:{index}")),
                source_id=record.source_id,
                text=split_document.page_content.strip(),
                metadata=dict(split_document.metadata),
            )
        )
    return chunks
