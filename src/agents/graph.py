"""Read-only LangGraph workflow for vulnerability inference."""

from __future__ import annotations
import os

from typing import Any, Optional
from agents.handlers import build_langfuse_handler
from agents.nodes import (
    build_fetch_live_nvd_node, 
    build_generate_remediation_node, 
    build_grade_documents_node, 
    build_retrieve_vulnerabilities_node,
    build_rewrite_query_node, 
    grade_edge,
)
from config import AppConfig
from langgraph.graph import END, START, StateGraph
from langfuse.langchain import CallbackHandler
from retrieval.vectorstore import QdrantVectorStoreClient
from src.core.factory import get_embeddings, get_fastllm, get_synthesisllm
from agents.schema import AgentState


def build_agent_graph(config: AppConfig) -> Any:
    """Build the inference-only LangGraph workflow from the node map.

    Input:
        Application configuration with LLM and Qdrant settings.
    Output:
        Returns a compiled LangGraph application.
    Security context:
        Ensures orchestration resides entirely within a `StateGraph` while
        keeping the graph read-only with respect to Qdrant and external sources.
    """

    fast_llm = get_fastllm(temperature=0)
    synthesis_llm = get_synthesisllm(temperature=0)
    vector_store_client = QdrantVectorStoreClient(
        config=config,
        embeddings=get_embeddings(),
    )

    graph = StateGraph(AgentState)
    graph.add_node("rewrite_query", build_rewrite_query_node(fast_llm=fast_llm))
    graph.add_node(
        "retrieve_vulnerabilities",
        build_retrieve_vulnerabilities_node(vector_store_client=vector_store_client),
    )
    graph.add_node("grade_documents", build_grade_documents_node(fast_llm=fast_llm))
    graph.add_node("fetch_live_nvd", build_fetch_live_nvd_node())
    graph.add_node(
        "generate_remediation",
        build_generate_remediation_node(synthesis_llm=synthesis_llm),
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


async def run_agent_graph(
    state: AgentState,
    config: AppConfig,
    invoke_config: Optional[dict[str, Any]] = None,
) -> AgentState:
    """Run the compiled inference graph with optional Langfuse tracing.

    Input:
        Receives the initial agent state, runtime application config, and an
        optional LangGraph invocation config.
    Output:
        Returns the final graph state after asynchronous execution completes.
    Security context:
        Applies Langfuse callbacks only when tracing credentials are configured
        through environment variables, keeping observability opt-in and secret
        values out of source code.
    """

    app = build_agent_graph(config=config)
    execution_config = dict(invoke_config or {})
    langfuse_handler = build_langfuse_handler()
    if langfuse_handler is not None:
        callbacks = list(execution_config.get("callbacks", []))
        callbacks.append(langfuse_handler)
        execution_config["callbacks"] = callbacks
    return await app.ainvoke(state, config=execution_config or None)

