from typing import Dict, List, Any, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pydantic import Field

class StandardRetriever(BaseRetriever):
    vector_store: PineconeVectorStore = Field(description = "Vector store to retrieve from")
    search_kwargs: Dict[str, Any] = Field(default_factory=lambda: {"k": 4}, description="Keyword arguments for similarity search")
    
    def __init__(
        self,
        vector_store: PineconeVectorStore,
        search_kwargs: Optional[Dict[str, Any]] = None
    ):
        search_kwargs = search_kwargs or {"k": 4}
        super().__init__(vector_store=vector_store, search_kwargs=search_kwargs)

    def get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs
    ) -> List[Document]:
        search_kwargs = {**self.search_kwargs, **kwargs}
        docs_and_scores = self.vector_store.similarity_search_with_score(query, **search_kwargs)
        docs = []
        for doc, score in docs_and_scores:
            doc.metadata["score"] = score
            docs.append(doc)
        return docs

class ContextualHeaderRetriever(BaseRetriever):
    """
    Contextual header RAG retriever using similarity search.
    """
    
    vector_store: PineconeVectorStore = Field(description="Vector store to retrieve from")
    search_kwargs: Dict[str, Any] = Field(default_factory=lambda: {"k": 4}, description="Keyword arguments for similarity search")
    
    def __init__(
        self,
        vector_store: PineconeVectorStore,
        search_kwargs: Optional[Dict[str, Any]] = None
    ):
        search_kwargs = search_kwargs or {"k": 4}
        super().__init__(vector_store=vector_store, search_kwargs=search_kwargs)
        
    def get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs
    ) -> List[Document]:
        search_kwargs = {**self.search_kwargs, **kwargs}
        docs_and_scores = self.vector_store.similarity_search_with_score(query, **search_kwargs)
        processed_docs = []
        for doc, score in docs_and_scores:
            doc.metadata["score"] = score
            content = doc.page_content
            
            if "CONTEXT:" in content:
                parts = content.split("\n\n", 1)
                if len(parts) > 1:
                    original_content = parts[1]
                else:
                    original_content = content
            else:
                original_content = content
                
            processed_doc = Document(
                page_content=original_content,
                metadata={
                    **doc.metadata,
                    "score": doc.metadata.get("score", None)
                }
            )
            
            processed_docs.append(processed_doc)
            
        return processed_docs
