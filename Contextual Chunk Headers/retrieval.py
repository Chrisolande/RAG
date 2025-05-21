"""
Retrieval components for the RAG experiment.
Implements standard and contextual header retrieval approaches.
"""
from typing import List, Dict, Any, Optional

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pydantic import Field


class StandardRetriever(BaseRetriever):
    """
    Standard RAG retriever using similarity search.
    """
    
    vector_store: PineconeVectorStore = Field(description="Vector store to retrieve from")
    search_kwargs: Dict[str, Any] = Field(default_factory=lambda: {"k": 4}, description="Keyword arguments for similarity search")
    
    def __init__(
        self,
        vector_store: PineconeVectorStore,
        search_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the StandardRetriever.
        
        Args:
            vector_store: Vector store to retrieve from
            search_kwargs: Optional keyword arguments for similarity search
        """
        search_kwargs = search_kwargs or {"k": 4}
        super().__init__(vector_store=vector_store, search_kwargs=search_kwargs)
        
    def get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs
    ) -> List[Document]:
        """
        Get documents relevant to the query.
        
        Args:
            query: Query string
            run_manager: Callback manager for the retriever run
            **kwargs: Additional arguments
            
        Returns:
            List of relevant documents
        """
        # Combine search_kwargs with any additional kwargs
        search_kwargs = {**self.search_kwargs, **kwargs}
        
        # Retrieve documents using similarity search with scores
        docs_and_scores = self.vector_store.similarity_search_with_score(query, **search_kwargs)
        
        # Convert to Document objects with scores in metadata
        docs = []
        for doc, score in docs_and_scores:
            # Create a new document with the score in metadata
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
        """
        Initialize the ContextualHeaderRetriever.
        
        Args:
            vector_store: Vector store to retrieve from
            search_kwargs: Optional keyword arguments for similarity search
        """
        search_kwargs = search_kwargs or {"k": 4}
        super().__init__(vector_store=vector_store, search_kwargs=search_kwargs)
        
    def get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs
    ) -> List[Document]:
        """
        Get documents relevant to the query.
        
        Args:
            query: Query string
            run_manager: Callback manager for the retriever run
            **kwargs: Additional arguments
            
        Returns:
            List of relevant documents
        """
        # Combine search_kwargs with any additional kwargs
        search_kwargs = {**self.search_kwargs, **kwargs}
        
        # Retrieve documents using similarity search with scores
        docs_and_scores = self.vector_store.similarity_search_with_score(query, **search_kwargs)
        
        # Process documents to include contextual headers
        processed_docs = []
        for doc, score in docs_and_scores:
            # Add score to metadata
            doc.metadata["score"] = score
            content = doc.page_content
            
            # Extract original content (remove the header)
            if "CONTEXT:" in content:
                parts = content.split("\n\n", 1)
                if len(parts) > 1:
                    original_content = parts[1]
                else:
                    original_content = content
            else:
                original_content = content
                
            # Create new document with original content and metadata
            processed_doc = Document(
                page_content=original_content,
                metadata={
                    **doc.metadata,
                    "score": doc.metadata.get("score", None)
                }
            )
            
            processed_docs.append(processed_doc)
            
        return processed_docs
