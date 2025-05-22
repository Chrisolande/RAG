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

    def format_retrieved_chunks(
        self, 
        root_chunks: List[Dict[str, Any]], 
        leaf_chunks: List[Dict[str, Any]]
    ) -> str:

        """Format the retrieved chunks into a string"""
        if not root_chunks and not leaf_chunks:
            return ""

        # Format root chunks by score
        sorted_root_chunks = sorted(root_chunks, key=lambda x: x.get("score", 0), reverse = True)
        # Group leaf chunks by parent ID
        leaf_chunks_by_parent = {}
        for chunk in leaf_chunks:
            parent_id = chunk.get("metadata", {}).get("parent_id", "unknown")
            if parent_id not in leaf_chunks_by_parent:
                leaf_chunks_by_parent[parent_id] = []
            leaf_chunks_by_parent[parent_id].append(chunk)

        # Format each root chunk with its leaf chunks
        formatted_sections = []
        
        for i, root_chunk in enumerate(sorted_root_chunks):
            root_id = root_chunk.get("id", "")
            
            # Format the root chunk
            source = root_chunk.get("metadata", {}).get("source", "unknown")
            
            section_header = f"[Section {i+1}] (Source: {source})"
            section_content = []
            
            # Add summary if available
            if root_chunk.get("summary"):
                section_content.append(f"Summary: {root_chunk['summary']}")
            
            # Add relevant leaf chunks for this root
            relevant_leaf_chunks = leaf_chunks_by_parent.get(root_id, [])
            
            # Sort leaf chunks by score
            sorted_leaf_chunks = sorted(relevant_leaf_chunks, key=lambda x: x.get("score", 0), reverse=True)
            
            # Add top leaf chunks
            for j, leaf_chunk in enumerate(sorted_leaf_chunks):
                leaf_text = leaf_chunk.get("text", "")
                section_content.append(f"Passage {j+1}: {leaf_text}")
            
            # If no leaf chunks, use the root text
            if not section_content:
                section_content.append(root_chunk.get("text", ""))
            
            # Format the section
            formatted_section = f"{section_header}\n\n{chr(10).join(section_content)}"
            formatted_sections.append(formatted_section)
        
        # Join all formatted sections
        context = "\n\n" + "\n\n".join(formatted_sections)
        return context