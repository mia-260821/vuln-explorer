

import operator
from typing import Annotated, Literal, Optional, TypedDict
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class AgentState(TypedDict, total=False):
    """Typed state shared across the LangGraph inference workflow.

    Input:
        Carries the user query, intermediate retrieval artifacts, and final
        response fields across graph nodes.
    Output:
        Provides a strict state contract matching the node map in `design.md`.
    Security context:
        Keeps all evidence and model outputs in explicit typed state so the
        inference workflow remains read-only and grounded to retrieved data.
    """

    query: str
    rewritten_query: str
    messages: list[BaseMessage]
    retrieved_documents: list[Document]
    graded_documents: list[Document]
    live_nvd_documents: list[Document]
    relevance: Literal["relevant", "irrelevant"]
    final_answer: str
    operating_system: Optional[str]
    software_stack: list[str]

    fallback_attempted: bool  # Track if we've already checked the live NVD
    loop_count: Annotated[int, operator.add]  # Optional: for more complex multi-try logic


class GradeDecision(BaseModel):
    """Structured output schema for document relevance grading.

    Input:
        Populated by the grading LLM from the query and retrieved excerpts.
    Output:
        Restricts the grade decision to a literal yes/no value.
    Security context:
        Prevents ambiguous free-form LLM output from driving LangGraph routing.
    """

    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Documents are relevant to the query, 'yes' or 'no'",
    )
