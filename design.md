# System Design: vulne-explorer

## 1. Project Goal
To provide a "zero-touch" vulnerability intelligence platform that translates raw CVE data into actionable remediation plans for DevOps and Security teams.

## 2. Architecture Overview
The system is built as an **Agentic RAG pipeline** using a "Plan -> Retrieve -> Grade -> Generate" loop.

### A. Ingestion Layer (The "Harvester")
- **Sources**: Real-time sync with NVD API and GitHub Advisory Database.
- **Processing**: Semantic chunking of vulnerability descriptions to preserve technical context.
- **Enrichment**: Attach metadata tags: `severity`, `cvss_score`, `platform`, `is_exploited`.

### B. Retrieval Layer (The "Researcher")
- **Hybrid Search**: Combine vector similarity (dense) with keyword matching (BM25) for specific CVE IDs.
- **Filtering**: Apply Qdrant metadata filters based on the user's OS and software stack.
- **Reranking**: Use a Cross-Encoder to prioritize vulnerabilities with active "In-the-wild" exploit status.

### C. Generation Layer (The "Analyst")
- **Reasoning**: The LLM synthesizes retrieved "PoC" code and "Patch Notes."
- **Output**: Generates a structured JSON report containing:
  - Severity Summary
  - Step-by-step terminal commands for patching.
  - Links to official advisories.

## 3. Directory Structure
```text
vulne-explorer/
├── data/               # Local cache for raw JSON feeds
├── src/
│   ├── ingestion/      # NVD API connectors & chunking logic
│   ├── retrieval/      # Qdrant client & Reranking logic
│   ├── agents/         # LangGraph state machines
│   └── ui/             # Streamlit frontend code
├── tests/              # RAGAS evaluation scripts
├── AGENTS.md           # Agent instructions
└── design.md           # System architecture
```

## 🗺 LangGraph Node Map
1. **rewrite_query**: Uses LLM to expand user input into technical CVE search terms.
2. **retrieve_vulnerabilities**: Uses `Qdrant.as_retriever()` to fetch top-k docs.
3. **grade_documents**: A "Checker" node that filters out irrelevant CVEs.
4. **fetch_live_nvd**: A fallback node that calls the NVD API if Qdrant results are "Low Relevance".
5. **generate_remediation**: The final node that synthesizes the "Action Plan".

## 🔄 Edge Logic
- `START` -> `rewrite_query` -> `retrieve_vulnerabilities` -> `grade_documents`
- `grade_documents` -> IF "relevant" -> `generate_remediation` -> `END`
- `grade_documents` -> IF "irrelevant" -> `fetch_live_nvd` -> `retrieve_vulnerabilities` (Loop)

## 📥 Ingestion Strategy: Semantic Indexing
1. **Source**: Async fetch from NVD API (CVE-2024+ focus).
2. **Document Model**: Convert JSON to `langchain_core.documents.Document`.
3. **Metadata**: Force-inject `cve_id`, `cvss`, and `software_list` into metadata fields.
4. **Indexing**: Use `QdrantVectorStore.add_documents` with an incremental check (don't duplicate CVEs).


## 🌐 Presentation Layer (API & UI)
1. **Backend**: FastAPI server to wrap the LangGraph agent.
   - **Endpoint**: `POST /chat` - Accepts user queries and returns the agent's full state or final response.
   - **Schema**: Use Pydantic models for request/response validation.
2. **Frontend**: Streamlit or React dashboard.
   - **Features**: Chat interface, "Thinking" step visibility (streaming), and CVE source citation links.

