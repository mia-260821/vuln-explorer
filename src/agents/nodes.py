
from __future__ import annotations

import httpx
from typing import Any, Optional

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from agents.schema import AgentState, GradeDecision


def build_rewrite_query_node(llm: BaseChatModel):
    """Create the query-rewrite node using a LangChain OpenAI chat model.

    Input:
        A configured fast chat model for short control-flow reasoning.
    Output:
        Returns a node callable compatible with LangGraph.
    Security context:
        Keeps LLM-backed query expansion explicit and bounded to the original
        user question.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", (
                "You rewrite cybersecurity questions into a concise retrieval query for this system's "
                "vulnerability corpus and NVD fallback search.\n"
                "Preserve the user's intent and keep only the most useful search terms.\n"
                "Include exact product names, vendors, versions, CVE IDs, operating systems, and "
                "software stack terms when they are present.\n"
                "Do not invent exploit types, CVE IDs, products, versions, or remediation steps that "
                "the user did not mention.\n"
                "If the user asks a broad question, keep the query broad but security-focused.\n"
                "Output only a single rewritten search query string."
            )),
            ("human", (
                "User question: {query}\n"
                "Operating system: {operating_system}\n"
                "Software stack: {software_stack}"
            )),
        ]
    )

    from langchain_core.output_parsers import StrOutputParser
    chain = prompt | llm | StrOutputParser()
    
    async def rewrite_query(state: AgentState) -> AgentState:
        """Expand the user question into technical retrieval terms.

        Input:
            Agent state containing the raw user query.
        Output:
            Returns updated state with the rewritten query.
        Security context:
            Records the model interaction in `messages` so query transformation
            remains inspectable during workflow execution.
        """

        query = state.get("query", "")
        operating_system = state.get("operating_system") or "unspecified"
        software_stack = ", ".join(state.get("software_stack", [])) or "unspecified"
        rewritten_query = await chain.ainvoke(
            {
                "query": query,
                "operating_system": operating_system,
                "software_stack": software_stack,
            }
        )

        messages = list(state.get("messages", []))
        messages.append(HumanMessage(content=query))
        messages.append(AIMessage(content=rewritten_query))
        return {
            **state,
            "rewritten_query": rewritten_query,
            "messages": messages,
        }

    return rewrite_query


def build_retrieve_vulnerabilities_node(vectorstore: VectorStore):
    """Create the Qdrant retrieval node for inference.

    Input:
        A configured Qdrant vector-store adapter.
    Output:
        Returns a node callable compatible with LangGraph.
    Security context:
        Restricts graph retrieval to read-only Qdrant access and never performs
        indexing or mutation of stored vulnerability documents.
    """

    async def retrieve_vulnerabilities(state: AgentState) -> AgentState:
        """Retrieve top-k vulnerability documents from Qdrant.

        Input:
            Agent state containing the rewritten query.
        Output:
            Returns updated state with retrieved documents.
        Security context:
            Uses Qdrant only as a read-only retriever so inference cannot modify
            indexed vulnerability intelligence.
        """

        from retrieval.base import SearchRequest, run_base_search

        search_req = SearchRequest(query=state.get("rewritten_query"))
        if state.get("operating_system"):
            search_req.filters["platform"] = state.get("operating_system")

        documents = await run_base_search(vectorstore, search_req)

        if state.get("fallback_attempted"):
            live_nvd_documents = state.get("live_nvd_documents", [])
            documents = _merge_documents(
                primary_documents=documents,
                secondary_documents=live_nvd_documents,
            )

        return {
            **state,
            "retrieved_documents": documents,
        }

    return retrieve_vulnerabilities


def build_grade_documents_node(llm: BaseChatModel):
    """Create the document-grading node using a fast LangChain OpenAI model.

    Input:
        A configured fast chat model for relevance checks.
    Output:
        Returns a node callable compatible with LangGraph.
    Security context:
        Centralizes relevance decisions in a dedicated checker step before the
        workflow can synthesize remediation guidance.
    """
    
    systemPrompt = """You are a grader assessing relevance of a retrieved document to a user query. 
    If the document contains keywords, software names, or versions related to the user query, grade it as relevant. 
    Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the query."""
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", systemPrompt),
            ("human", "Retrieved document: \n\n {document} \n\n User query: {query}"),        
        ]
    )
    chain = prompt | llm.with_structured_output(GradeDecision)

    async def grade_documents(state: AgentState) -> AgentState:
        """Mark the retrieved document set as relevant or irrelevant.

        Input:
            Agent state containing the current retrieved documents.
        Output:
            Returns updated state with grading results.
        Security context:
            Prevents unsupported generation by forcing retrieved evidence
            through an explicit relevance gate.
        """
        user_intent = state.get('query')
        docs = state.get("retrieved_documents", [])
        if not docs:
            return {
                **state,
                "graded_documents": [],
                "relevance": "irrelevant",
            }
        
        # We check the top document for relevance to decide the path
        doc_txt = docs[0].page_content
        result: GradeDecision = await chain.ainvoke({"document": doc_txt, "query": user_intent})
    
        relevance = "relevant" if result.binary_score == "yes" else "irrelevant"
        graded_documents = docs if relevance == 'relevant' else []
        return {
            **state,
            "graded_documents": graded_documents,
            "relevance": relevance,
        }
    
    return grade_documents


def build_fetch_live_nvd_node():
    """Create the live NVD fallback node.

    Input:
        No direct caller input. Returns a LangGraph-compatible node callable.
    Output:
        Returns a node that fetches live NVD evidence and marks fallback usage.
    Security context:
        Restricts fallback behavior to read-only NVD API access and does not
        index or mutate Qdrant during inference.
    """

    async def fetch_live_nvd(state: AgentState) -> AgentState:
        """Fetch live NVD CVE records for low-relevance queries.

        Input:
            Agent state marked as having low-relevance Qdrant results.
        Output:
            Returns updated state with live NVD documents and
            `fallback_attempted=True`.
        Security context:
            Uses only read-only NVD API calls and preserves source metadata so
            fallback evidence remains attributable.
        """

        if state.get("fallback_attempted"):
            return {
                **state,
                "fallback_attempted": True,
            }

        query = state.get("rewritten_query") or state.get("query", "")
        params: dict[str, Any] = {"resultsPerPage": 5}
        if query:
            params["keywordSearch"] = query

        async with httpx.AsyncClient(
            base_url="https://services.nvd.nist.gov",
            timeout=30.0,
        ) as client:
            response = await client.get("/rest/json/cves/2.0", params=params)
            response.raise_for_status()
            payload = response.json()

        live_nvd_documents = [
            _nvd_item_to_document(item)
            for item in payload.get("vulnerabilities", [])
        ]
        return {
            **state,
            "fallback_attempted": True,
            "live_nvd_documents": live_nvd_documents,
            "retrieved_documents": live_nvd_documents,
            "loop_count": 1
        }

    return fetch_live_nvd


def build_generate_remediation_node(llm: BaseChatModel):
    """Create the final remediation-generation node.

    Input:
        A configured synthesis model for final response generation.
    Output:
        Returns a node callable compatible with LangGraph.
    Security context:
        Constrains final answer generation to documents admitted by the grading
        step and preserves the no-hallucination fallback.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Senior Security Engineer. Use the context below to answer.\n\nCONTEXT:\n{context}"),
        ("human", "{input}")
    ])
    combine_chain = create_stuff_documents_chain(llm, prompt)

    async def generate_remediation(state: AgentState) -> AgentState:
        """Generate the final remediation response from grounded evidence.

        Input:
            Agent state containing graded vulnerability documents.
        Output:
            Returns updated state with the final answer.
        Security context:
            Preserves the no-hallucination rule by restricting synthesis to the
            graded document set.
        """

        graded_documents = state.get("graded_documents", [])
        if not graded_documents:
            final_answer = "Information not found in database."
        else:
            final_answer = await combine_chain.ainvoke(
                {
                    "input": state.get("query"),
                    "context": graded_documents,
                }
            )

        messages = list(state.get("messages", []))
        messages.append(AIMessage(content=str(final_answer)))
        return {
            **state,
            "messages": messages,
            "final_answer": str(final_answer),
        }
    return generate_remediation


def grade_edge(state: AgentState) -> str:
    """Choose the next edge based on document relevance.

    Input:
        Agent state containing the relevance decision.
    Output:
        Returns the next edge label expected by LangGraph.
    Security context:
        Keeps the fallback loop explicit while preventing generation from
        bypassing the grading step.
    """
    relevance = state.get("relevance")
    fallback_tried = state.get("fallback_attempted", False)

    # Path A: Data is good -> Generate
    if relevance == "relevant":
        return "relevant"

    # Path B: Data is bad BUT we already checked the API -> Force Exit
    if fallback_tried:
        return "relevant" # Force to generator to show "No results" message

    # Path C: Data is bad and we haven't checked API yet -> Try NVD
    return "irrelevant"


def _nvd_item_to_document(item: dict[str, Any]) -> Document:
    """Convert an NVD API vulnerability item into a LangChain document.

    Input:
        A single vulnerability object from the NVD `cves/2.0` API response.
    Output:
        Returns a LangChain `Document` for fallback evidence generation.
    Security context:
        Preserves source identifiers and CVSS data so fallback answers remain
        attributable to authoritative NVD records.
    """

    cve = item.get("cve", {})
    cve_id = str(cve.get("id", "unknown-cve"))

    description = ""
    for entry in cve.get("descriptions", []):
        if entry.get("lang") == "en":
            description = str(entry.get("value", ""))
            break

    cvss_score: Optional[float] = None
    metrics = cve.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        metric_list = metrics.get(key, [])
        if metric_list:
            base_score = metric_list[0].get("cvssData", {}).get("baseScore")
            if base_score is not None:
                cvss_score = float(base_score)
                break

    return Document(
        page_content=description,
        metadata={
            "source": "nvd_live",
            "cve_id": cve_id,
            "cvss": cvss_score,
        },
    )


def _merge_documents(
    primary_documents: list[Document],
    secondary_documents: list[Document],
) -> list[Document]:
    """Merge document lists while preserving order and removing duplicates.

    Input:
        Primary retrieval results and secondary fallback documents.
    Output:
        Returns a single ordered document list with duplicates removed.
    Security context:
        Keeps fallback evidence attributable by de-duplicating only on explicit
        source identifiers or content, without inventing synthetic results.
    """

    merged_documents: list[Document] = []
    seen_keys: set[str] = set()
    for document in [*primary_documents, *secondary_documents]:
        key = str(document.metadata.get("cve_id") or document.page_content)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged_documents.append(document)
    return merged_documents
