
from __future__ import annotations

import httpx
from typing import Any, Optional

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from retrieval.vectorstore import QdrantVectorStoreClient
from agents.schema import AgentState, GradeDecision


def build_rewrite_query_node(fast_llm: BaseChatModel):
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
                "You are a Security Research Assistant. Your task is to rewrite user "
                "input into a concise, technical search query for a vulnerability database.\n"
                "Focus on: Software Name, Version Numbers, and Attack Vectors (RCE, SQLi, etc.).\n"
                "Output ONLY the optimized search string. No conversational filler."
            )),
            ("human", "{query}"),
        ]
    )

    from langchain_core.output_parsers import StrOutputParser
    chain = prompt | fast_llm | StrOutputParser()
    
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
        rewritten_query = await chain.ainvoke({"query": query})

        messages = list(state.get("messages", []))
        messages.append(HumanMessage(content=query))
        messages.append(AIMessage(content=rewritten_query))
        return {
            **state,
            "rewritten_query": rewritten_query,
            "messages": messages,
        }

    return rewrite_query


def build_retrieve_vulnerabilities_node(vector_store_client: QdrantVectorStoreClient):
    """Create the Qdrant retrieval node for inference.

    Input:
        A configured Qdrant vector-store adapter.
    Output:
        Returns a node callable compatible with LangGraph.
    Security context:
        Restricts graph retrieval to read-only Qdrant access and never performs
        indexing or mutation of stored vulnerability documents.
    """

    vectorstore = vector_store_client.build_vector_store()

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
        if state.get("fallback_attempted"):
            live_nvd_documents = state.get("live_nvd_documents", [])
            retrieved_documents = _merge_documents(
                primary_documents=retrieved_documents,
                secondary_documents=live_nvd_documents,
            )
    
        search_query = state.get("rewritten_query")

        search_kwargs = {"k": 5}
        if state.get("operating_system"):
            search_kwargs["filter"] = {"platform": state.get("operating_system")}

        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
        documents = await retriever.ainvoke(search_query)
        return {
            **state,
            "retrieved_documents": documents,
        }

    return retrieve_vulnerabilities


def build_grade_documents_node(fast_llm: BaseChatModel):
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
    chain = prompt | fast_llm.with_structured_output(GradeDecision)

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
        graded_documents = docs if relevance == 'relevance' else []
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


def build_generate_remediation_node(synthesis_llm: BaseChatModel):
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
        ("system", (
            "You are a Senior Security Engineer. Use the following CVE context to answer the user's query.\n\n"
            "CONTEXT:\n{context}\n\n"
            "USER QUERY: {input}\n\n"
            "Format your response as follows:\n"
            "### ⚠️ Risk Assessment\n(Briefly explain the danger)\n"
            "### 🛠 Remediation Steps\n(List terminal commands or config changes)\n"
            "### 🔗 References\n(List the CVE IDs used)"
        ))
    ])
    combine_chain = create_stuff_documents_chain(llm=synthesis_llm, prompt=prompt)

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


