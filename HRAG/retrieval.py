import logging
from typing import List, Dict, Any, Optional, Tuple

from embedding import EmbeddingGenerator
from vector_store import HierarchicalVectorStore

from config import TOP_K_ROOT, TOP_K_LEAF

# Configure logging
logger = logging.getLogger(__name__)

class HierarchicalRetriever:
    def __init__(self,
                top_k_root: int = TOP_K_ROOT,
                top_k_leaf: int = TOP_K_LEAF):
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = HierarchicalVectorStore()
        self.top_k_root = top_k_root
        self.top_k_leaf = top_k_leaf
        logger.info(f"Initialized HierarchicalRetriever with top_k_root={top_k_root}, top_k_leaf={top_k_leaf}")

    def retrieve(
        self, 
        query: str, 
        root_filter: Optional[Dict[str, Any]] = None,
        leaf_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Retrieve relevant chunks using hierarchical search."""

        if not query or not query.strip():
            logger.warning("Empty query")
            return {"root_chunks": [], "leaf_chunks": [], "context": ""}
        
        logger.info(f"Retrieving chunks for query: {query}")

        # Generate embeddings for the query
        query_embedding = self.embedding_generator.generate_query_embedding(query)

        # Query the root chunks
        root_chunks = self.vector_store.query_root(
            query_embedding = query_embedding,
            top_k = self.top_k_root,
            filter = root_filter
        )

        # Extract parent IDs for filtering leaf chunks
        parent_ids = [chunk["id"] for chunk in root_chunks]
        # Query the leaf level with parent filtering
        leaf_chunks = []
        if parent_ids:
            leaf_chunks = self.vector_store.query_leaf(
                query_embedding=query_embedding,
                parent_ids=parent_ids,
                top_k=self.top_k_leaf * len(parent_ids),  # Get top_k_leaf per parent
                filter=leaf_filter
            )

        # Format the retrieved chunks into a context string
        context = self.format_retrieved_chunks(root_chunks, leaf_chunks)

        logger.info(f"Retrieved {len(root_chunks)} root chunks and {len(leaf_chunks)} leaf chunks")
        return {
            "query": query,
            "root_chunks": root_chunks,
            "leaf_chunks": leaf_chunks,
            "context": context
        }