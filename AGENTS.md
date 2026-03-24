# Agent Instructions: vulne-explorer

You are an expert AI Engineer specialized in CyberSecurity and Advanced RAG. Your goal is to build "vulne-explorer," a real-time vulnerability intelligence system.

## 🛠 Tech Stack
- **Orchestration**: LangGraph (for state-aware, agentic loops).
- **Vector DB**: Qdrant (using HNSW indexing and metadata filtering).
- **Embeddings**: nomic-embed-text (primary) or bge-large-en.
- **Reranker**: FlashRank or Cohere Rerank 3.5.
- **LLM**: GPT-4o-mini (for fast logic) and GPT-4o (for complex synthesis).

## 📐 Coding Standards
- **Modular Design**: Separate ingestion, retrieval, and generation into distinct modules.
- **Type Hinting**: Use strict Python type hints for all functions.
- **Documentation**: Every function must include a docstring explaining its input, output, and security context.
- **Async First**: Use `asyncio` for all API calls and database operations to ensure low latency.

## 🚫 Boundaries & Guardrails
- **Never** commit API keys or `.env` files to the repository.
- **No Hallucinations**: If a retrieved chunk does not contain the answer, the agent must explicitly state "Information not found in database."
- **Verification**: Every claim about a CVE must be accompanied by its source ID (e.g., CVE-2024-XXXX).

## 🚀 Key Commands
- `pip install -r requirements.txt`
- `pytest -v` (Run all security and logic tests)
- `python main.py --ingest` (Trigger manual data sync)

## 🛑 Mandatory Framework Constraints
- **Core Engine**: ALL logic must reside within a `langgraph.graph.StateGraph`. No standalone "while True" loops.
- **Data Model**: Use `typing.TypedDict` for the Graph State.
- **Components**: 
  - Use `langchain_community.vectorstores.Qdrant` for all DB interactions.
  - Use `langchain_core.messages` for all communication.
  - Use `langchain.tools` for the NVD API harvester.
- **Embeddings**: Use `langchain_core.embeddings.Embeddings` interface.
- **Retrieval**: Use `langchain.retrievers.ContextualCompressionRetriever` with a Reranker for high-precision security data.
- **No Custom RAG**: Do not write custom similarity math. Use `langchain.chains.combine_documents`.

## 🏗 Architectural Separation
- **Ingestion**: MUST be a standalone module (`src/ingestion/`) executed via CLI. It is strictly for Write operations to Qdrant.
- **Inference (LangGraph)**: MUST be a Read-only pipeline. It retrieves existing data from Qdrant and generates answers. 
- **Constraint**: Do not add nodes to the LangGraph that perform data scraping or indexing.

## 🚀 Deployment Standards
- **Containerization**: Use Multi-stage Docker builds to keep images small.
- **Orchestration**: Use Docker Compose for local development (App + Qdrant).
- **Environment**: All secrets (API keys) MUST be pulled from environment variables, never hardcoded.
- **Health Checks**: Include a `/health` endpoint in the API to monitor the LangGraph service.


## 🤖 Model Abstraction Standards
-   **Requirement**: The system must support both `OpenAI` and `Google Gemini`.
-   **Implementation**: Use a `model_factory.py` to initialize models based on the `MODEL_PROVIDER` env var.
-   **Interfaces**:
    -   Use `langchain_core.language_models.chat_models.BaseChatModel` for LLMs.
    -   Use `langchain_core.embeddings.Embeddings` for vectorization.


## 📊 Observability Standards (Langfuse)
- **Tracing**: All LangChain/LangGraph calls MUST include the `LangfuseCallbackHandler`.
- **Environment**: Use `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST`.
- **Naming**: Every trace should be tagged with the `user_id` or `session_id` if available.

