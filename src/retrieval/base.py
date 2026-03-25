
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel
from typing import Any, Dict

class SearchRequest(BaseModel):
    query: str
    filters: Dict[str, Any] = {}
    limit: int = 5
    

async def run_base_search(
    vectorstore: VectorStore, 
    request: SearchRequest
):
    """
    Handles the LLM generation of sub-queries and executes the search.
    """
    
    base_retriever = vectorstore.as_retriever(
        search_kwargs={"k": request.limit, "filter": request.filters}
    )

    return await base_retriever.ainvoke(request.query)
