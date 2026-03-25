
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from retrieval.base import SearchRequest


async def run_multi_query_search(
    vectorstore: VectorStore, 
    llm: BaseChatModel, 
    request: SearchRequest
):
    """
    Handles the LLM generation of sub-queries and executes the search.
    """
    
    # We build it here to ensure it uses the current LLM/Vectorstore state
    base_retriever = vectorstore.as_retriever(
        search_kwargs={"k": request.limit, "filter": request.filters} 
    )
    
    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, 
        llm=llm
    )
    
    return await mq_retriever.ainvoke(request.query)

