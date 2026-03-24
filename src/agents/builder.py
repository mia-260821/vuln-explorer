"""Compatibility builder for the inference LangGraph."""

from __future__ import annotations

from agents.graph import build_agent_graph
from config import AppConfig


def build_graph(config: AppConfig):
    """Build the read-only inference LangGraph.

    Input:
        Application configuration with LLM and Qdrant settings.
    Output:
        Returns a compiled LangGraph application.
    Security context:
        Delegates to the inference graph builder so LangGraph remains read-only
        and separate from ingestion concerns.
    """

    return build_agent_graph(config=config)
