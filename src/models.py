"""Shared domain models used across vulne-explorer modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class VulnerabilityRecord:
    """Normalized vulnerability record collected from external sources.

    Input:
        Receives parsed advisory fields from ingestion connectors.
    Output:
        Produces a canonical representation suitable for chunking, storage,
        retrieval, and report generation.
    Security context:
        Retains source identifiers for every record so downstream reasoning can
        attribute claims to CVE or advisory IDs without fabricating provenance.
    """

    source_id: str
    summary: str
    details: str
    severity: str
    cvss_score: Optional[float]
    platform: list[str] = field(default_factory=list)
    is_exploited: bool = False
    references: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """Chunked representation of vulnerability content for indexing.

    Input:
        Accepts a normalized vulnerability record and chunk text.
    Output:
        Encapsulates the text and metadata required for vector or keyword search.
    Security context:
        Carries forward the original source ID and key metadata so retrieval
        responses remain traceable and filterable by environment constraints.
    """

    chunk_id: str
    source_id: str
    text: str
    metadata: dict[str, Any]


@dataclass
class SearchQuery:
    """User-facing retrieval request describing environment and intent.

    Input:
        Captures the natural language question and optional environment filters.
    Output:
        Provides a structured retrieval contract for the agent pipeline.
    Security context:
        Environment filters constrain results to the relevant operating systems
        and software stack, reducing unsafe or irrelevant remediation guidance.
    """

    question: str
    operating_system: Optional[str] = None
    software_stack: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """Single retrieved chunk plus scoring metadata.

    Input:
        Contains indexed chunk data and retrieval scores from hybrid search.
    Output:
        Serves as the grounding unit for grading and generation steps.
    Security context:
        Includes explicit source IDs so every downstream claim can be verified
        against a retrieved advisory or CVE record.
    """

    chunk: DocumentChunk
    score: float
    matched_terms: list[str] = field(default_factory=list)


@dataclass
class GeneratedReport:
    """Structured analyst output returned to the caller.

    Input:
        Accepts synthesized severity, remediation, and reference content.
    Output:
        Produces a JSON-serializable report aligned with the system design.
    Security context:
        The `evidence` field must contain source IDs for all claims, and the
        summary must fall back to a database-miss message when grounding fails.
    """

    summary: str
    severity_summary: str
    remediation_steps: list[str]
    advisory_links: list[str]
    evidence: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert the report into a JSON-serializable dictionary.

        Input:
            Uses the current report fields.
        Output:
            Returns a plain dictionary for CLI output or API responses.
        Security context:
            Preserves evidence identifiers during serialization so attribution
            cannot be silently dropped before the response reaches users.
        """

        return {
            "summary": self.summary,
            "severity_summary": self.severity_summary,
            "remediation_steps": self.remediation_steps,
            "advisory_links": self.advisory_links,
            "evidence": self.evidence,
        }
