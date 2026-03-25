"""Read-only LangGraph workflow for vulnerability inference."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from agents.nodes import (
    build_fetch_live_nvd_node, 
    build_generate_remediation_node, 
    build_grade_documents_node, 
    build_retrieve_vulnerabilities_node,
    build_rewrite_query_node, 
    grade_edge,
)
from langgraph.graph import END, START, StateGraph
from agents.schema import AgentState


def build_agent_graph(
    vectorstore: VectorStore,
    fast_llm: BaseChatModel,
    synthesis_llm: BaseChatModel,
) -> Any:
    
    """Build the inference-only LangGraph workflow from the node map.

    Input:
        Application configuration with LLM and Qdrant settings.
    Output:
        Returns a compiled LangGraph application.
    Security context:
        Ensures orchestration resides entirely within a `StateGraph` while
        keeping the graph read-only with respect to Qdrant and external sources.
    """

    graph = StateGraph(AgentState)
    graph.add_node("rewrite_query", build_rewrite_query_node(fast_llm))
    graph.add_node(
        "retrieve_vulnerabilities",
        build_retrieve_vulnerabilities_node(vectorstore),
    )
    graph.add_node("grade_documents", build_grade_documents_node(fast_llm))
    graph.add_node("fetch_live_nvd", build_fetch_live_nvd_node())
    graph.add_node(
        "generate_remediation",
        build_generate_remediation_node(synthesis_llm),
    )

    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve_vulnerabilities")
    graph.add_edge("retrieve_vulnerabilities", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        grade_edge,
        {
            "relevant": "generate_remediation",
            "irrelevant": "fetch_live_nvd",
            'reach_loop_limit': 'generate_remediation'
        },
    )
    graph.add_edge("fetch_live_nvd", "retrieve_vulnerabilities")
    graph.add_edge("generate_remediation", END)
    return graph.compile()

