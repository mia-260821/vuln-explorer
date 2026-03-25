

from typing import Any, Optional

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from agents.graph import build_agent_graph
from agents.handlers import build_langfuse_handler
from config import AppConfig
from src.core.factory import get_collection_name, get_embeddings, get_fastllm, get_synthesisllm
from agents.schema import AgentState



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

    fast_llm = get_fastllm(temperature=0)
    synthesis_llm = get_synthesisllm(temperature=0)
    vectorstore_client = QdrantClient(url=config.qdrant_url)
    vectorstore = QdrantVectorStore(
        client=vectorstore_client,
        collection_name=get_collection_name(prefix=config.qdrant_collection),
        embedding=get_embeddings(),
        vector_name=config.qdrant_vector_name,
    )
    
    app = build_agent_graph(vectorstore, fast_llm, synthesis_llm)

    execution_config = dict(invoke_config or {})
    langfuse_handler = build_langfuse_handler()
    if langfuse_handler is not None:
        callbacks = list(execution_config.get("callbacks", []))
        callbacks.append(langfuse_handler)
        execution_config["callbacks"] = callbacks

    return await app.ainvoke(state, config=execution_config or None)
