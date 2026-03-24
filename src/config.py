"""Application configuration for vulne-explorer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import Optional


@dataclass
class AppConfig:
    """Runtime configuration for the vulnerability intelligence system.

    Input:
        Reads environment variables and optional defaults to determine runtime paths
        and external service settings.
    Output:
        A strongly typed configuration object shared across ingestion, retrieval,
        and generation modules.
    Security context:
        Stores only connection metadata and model names. Secrets must be supplied
        through the environment and are never hard-coded in source control.
    """

    data_dir: Path = field(default_factory=lambda: Path(os.getenv("VULN_EXPLORER_DATA_DIR", "data")))
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "vulnerabilities"))
    qdrant_vector_size: Optional[int] = field(
        default_factory=lambda: int(os.getenv("QDRANT_VECTOR_SIZE"))
        if os.getenv("QDRANT_VECTOR_SIZE")
        else None
    )
    qdrant_vector_name: str = field(default_factory=lambda: os.getenv("QDRANT_VECTOR_NAME", "default"))
    qdrant_distance: str = field(default_factory=lambda: os.getenv("QDRANT_DISTANCE", "cosine"))
    synthesis_llm_provider: str = field(default_factory=lambda: os.getenv("SYNTHESIS_LLM_PROVIDER", "openai"))
    synthesis_llm_model: str = field(default_factory=lambda: os.getenv("SYNTHESIS_LLM_MODEL", "gpt-4o"))

    def ensure_directories(self) -> None:
        """Create required local directories for application state.

        Input:
            Uses the configured `data_dir`.
        Output:
            Ensures the local cache directory exists.
        Security context:
            Restricts filesystem writes to project-local cache storage to avoid
            leaking vulnerability artifacts into uncontrolled locations.
        """

        self.data_dir.mkdir(parents=True, exist_ok=True)
