from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.vectorstores import VectorStore

from retrieval.base import SearchRequest


async def run_parent_document_search(vectorstore: VectorStore, request: SearchRequest, docstore = None ):
    """
    Retrieves large parent documents based on small chunk matches.
    """
    # In production, this store would be Redis or MongoDB
    if docstore:
        store = docstore
    else:
        from langchain_classic.storage.in_memory import InMemoryStore
        store = InMemoryStore()
    
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key="doc_id",
        search_kwargs={"k": request.limit, "filter": request.filters}
    )
    
    return await retriever.ainvoke(request.query)
