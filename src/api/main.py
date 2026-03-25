"""FastAPI server for the vulne-explorer LangGraph inference workflow."""

from __future__ import annotations

from dataclasses import asdict
import os
from typing import Any
import uuid

from config import AppConfig
from fastapi import FastAPI
from langchain_core.documents import Document
from pydantic import BaseModel, Field
import uvicorn

from langfuse import Langfuse
from agents.usecase import run_agent_graph

langfuse = Langfuse(
    public_key=os.environ['LANGFUSE_PUBLIC_KEY'],
    secret_key=os.environ['LANGFUSE_SECRET_KEY'],
    host=os.environ['LANGFUSE_HOST'],  # Optional, default shown
)


class QueryRequest(BaseModel):
    """Request schema for chat-style vulnerability analysis.

    Input:
        Receives the user's raw message string.
    Output:
        Validates request payloads sent to the `/chat` endpoint.
    Security context:
        Restricts API input to an explicit message field so the LangGraph
        receives only the intended user query.
    """

    message: str = Field(..., min_length=1, description="User query for vulnerability analysis")


class RetrievedDocument(BaseModel):
    """Serializable representation of a retrieved LangChain document.

    Input:
        Accepts a LangChain document's content and metadata.
    Output:
        Returns a JSON-safe response model for API clients.
    Security context:
        Preserves retrieved source metadata for transparent downstream citation
        without exposing hidden server-side state.
    """

    page_content: str
    metadata: dict[str, Any]


class ChatResponse(BaseModel):
    """Response schema for LangGraph chat inference.

    Input:
        Accepts the final generated answer and the retrieved evidence documents.
    Output:
        Returns a validated response body for the `/chat` endpoint.
    Security context:
        Exposes the final answer together with retrieved evidence so responses
        remain inspectable and grounded.
    """

    final_generation: str
    retrieved_documents: list[RetrievedDocument]


app = FastAPI(title="vulne-explorer", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    """Return a health indicator for deployment monitoring.

    Input:
        No request body.
    Output:
        Returns a simple health status payload.
    Security context:
        Exposes only a minimal readiness signal without leaking application
        configuration or internal graph state.
    """

    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: QueryRequest) -> ChatResponse:
    """Run the LangGraph inference workflow for a user message.

    Input:
        A validated `QueryRequest` containing the user's message.
    Output:
        Returns the final generation and the retrieved documents used during
        graph execution.
    Security context:
        Invokes the read-only LangGraph inference workflow and returns grounded
        evidence alongside the final answer for inspection.
    """
    trace_id = Langfuse.create_trace_id()
    config = AppConfig()
    result = await run_agent_graph(
        state={"query": request.message, "software_stack": []},
        config=config,
        invoke_config={},
        trace_id=trace_id,
    )
    documents = [_serialize_document(document) for document in result.get("retrieved_documents", [])]
    return ChatResponse(
        final_generation=result.get("final_answer", "Information not found in database."),
        retrieved_documents=documents,
    )


def _serialize_document(document: Document) -> RetrievedDocument:
    """Convert a LangChain document into the API response model.

    Input:
        A LangChain `Document` returned by the retrieval node.
    Output:
        Returns a `RetrievedDocument` suitable for JSON serialization.
    Security context:
        Preserves document content and metadata only, preventing accidental
        exposure of non-serializable or hidden graph internals.
    """

    return RetrievedDocument(
        page_content=document.page_content,
        metadata=dict(document.metadata),
    )


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
