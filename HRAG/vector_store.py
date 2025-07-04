import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from pinecone import Pinecone, ServerlessSpec
from config import (
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE_ROOT,
    PINECONE_NAMESPACE_LEAF,
    EMBEDDING_DIMENSIONS
)

# Configure logging
logger = logging.getLogger(__name__)

class HierarchicalVectorStore:
    def __init__(self):
        if not PINECONE_API_KEY:
            raise ValueError("No Pinecone api key found in the environment")

        # Initialize Pinecone
        self.pc = Pinecone(api_key = PINECONE_API_KEY)
        # Check if index exists, if not create it
        self.index_name = PINECONE_INDEX_NAME
        self.root_namespace = PINECONE_NAMESPACE_ROOT
        self.leaf_namespace = PINECONE_NAMESPACE_LEAF
        self.dimension = EMBEDDING_DIMENSIONS

        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)

    def _ensure_index_exists(self):
        """Ensure that the Pinecone index exists, creating it if necessary."""
        try:
            # List existing indexes
            existing_indexes = self.pc.list_indexes()

            # Check if our index exists
            if self.index_name not in existing_indexes.names():
                logger.info(f"Creating Pinecone index: {self.index_name}")

                self.pc.create_index(
                    name = self.index_name,
                    dimension = self.dimension,
                    metric="cosine", # Use cosine similarity
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
            
                # Wait for the index to be ready
                logger.info("Waiting for index to be ready...")
                time.sleep(10) # Give some time for the index to initialize
            
            else:
                logger.info(f"Pinecone index already exists: {self.index_name}")
                # Verify the dimension of the existing index
                index_info = self.pc.describe_index(self.index_name)
                existing_dimension = index_info.dimension
                
                if existing_dimension != self.dimension:
                    logger.warning(
                        f"Dimension mismatch: Expected {self.dimension}, "
                        f"but index has {existing_dimension}"
                    )

        except Exception as e:
            logger.error(f"Error ensuring index exists: {str(e)}")
            raise

    def upsert_root_chunks(self, root_chunks_with_embeddings: List[Dict[str, Any]]) -> int:
        """Upsert root/parent document chunks with embeddings to Pinecone."""

        if not root_chunks_with_embeddings:
            logger.warning("No root chunks to upsert.")
            return 0

        logger.info(f"Upserting {len(root_chunks_with_embeddings)} root chunks to Pinecone")
        
        # Prepare vectors for upserting
        vectors = []
        batch_size = 100

        for chunk in root_chunks_with_embeddings:
            # Use the chunk ID as the vector ID
            vector_id = chunk.get("id", f"root_{int(time.time())}")
            
            # Extract the embedding - prefer summary embedding if available
            if "summary_embedding" in chunk:
                embedding = chunk.get("summary_embedding", [])
            else:
                embedding = chunk.get("text_embedding", chunk.get("embedding", []))
            
            # Extract metadata (excluding the embeddings)
            # Start with base metadata
            metadata = {"text": chunk["text"]}
            
            # Add metadata fields, filtering out null values
            if "metadata" in chunk and isinstance(chunk["metadata"], dict):
                for key, value in chunk["metadata"].items():
                    # Skip null values as Pinecone doesn't accept them
                    if value is not None:
                        metadata[key] = value
            
            # Add summary to metadata if available and not None
            if "summary" in chunk and chunk["summary"] is not None:
                metadata["summary"] = chunk["summary"]
            
            # Add to vectors list
            vectors.append((vector_id, embedding, metadata))
        
        # Upsert vectors in batches
        successful_upserts = 0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            
            try:
                # Format batch for Pinecone
                pinecone_batch = [
                    (id, embedding, metadata) 
                    for id, embedding, metadata in batch
                ]
                
                # Upsert to Pinecone
                self.index.upsert(
                    vectors=pinecone_batch,
                    namespace=self.root_namespace
                )
                
                successful_upserts += len(batch)
                logger.info(f"Upserted root batch {i // batch_size + 1}/{(len(vectors) - 1) // batch_size + 1}")
            
            except Exception as e:
                logger.error(f"Error upserting root batch {i // batch_size + 1}: {str(e)}")
        
        logger.info(f"Successfully upserted {successful_upserts}/{len(vectors)} root vectors")
        return successful_upserts

    def upsert_leaf_chunks(self, leaf_chunks_with_embeddings: List[Dict[str, Any]]) -> int:
        if not leaf_chunks_with_embeddings:
            logger.warning("No leaf chunks to upsert")
            return 0
        
        logger.info(f"Upserting {len(leaf_chunks_with_embeddings)} leaf chunks to Pinecone")
        
        # Prepare vectors for upserting
        vectors = []
        batch_size = 100  # Pinecone recommends batches of 100 vectors
        
        for chunk in leaf_chunks_with_embeddings:
            # Use the chunk ID as the vector ID
            vector_id = chunk.get("id", f"leaf_{int(time.time())}")
            
            # Extract the embedding
            embedding = chunk.get("embedding", [])
            
            # Extract metadata (excluding the embedding)
            # Start with base metadata
            metadata = {"text": chunk["text"]}
            
            # Add metadata fields, filtering out null values
            if "metadata" in chunk and isinstance(chunk["metadata"], dict):
                for key, value in chunk["metadata"].items():
                    # Skip null values as Pinecone doesn't accept them
                    if value is not None:
                        metadata[key] = value
            
            # Add to vectors list
            vectors.append((vector_id, embedding, metadata))
        
        # Upsert vectors in batches
        successful_upserts = 0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            
            try:
                # Format batch for Pinecone
                pinecone_batch = [
                    (id, embedding, metadata) 
                    for id, embedding, metadata in batch
                ]
                
                # Upsert to Pinecone
                self.index.upsert(
                    vectors=pinecone_batch,
                    namespace=self.leaf_namespace
                )
                
                successful_upserts += len(batch)
                logger.info(f"Upserted leaf batch {i // batch_size + 1}/{(len(vectors) - 1) // batch_size + 1}")
            
            except Exception as e:
                logger.error(f"Error upserting leaf batch {i // batch_size + 1}: {str(e)}")
        
        logger.info(f"Successfully upserted {successful_upserts}/{len(vectors)} leaf vectors")
        return successful_upserts

    def query_root(self, query_embedding, top_k: int = 3, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query the root/parent level of the vector store for similar chunks."""
        if not query_embedding:
            logger.warning("No query embedding provided")
            return []

        try:
            query_results = self.index.query(
                vector = query_embedding,
                top_k = top_k,
                namespace = self.root_namespace,
                include_metadata = True,
                filter = filter
            )

            # Format results
            results = []
            for match in query_results.matches:
                result = {
                    "id": match.id,
                    "score": match.score,
                    "text": match.metadata.get("text", ""),
                    "summary": match.metadata.get("summary", ""),
                    "metadata": {
                        k: v for k, v in match.metadata.items() if k not in ["text", "summary"]
                    }
                }
                results.append(result)

            logger.info(f"Root query returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error querying root: {str(e)}")
            return []

    def query_leaf(
        self, 
        query_embedding: List[float], 
        parent_ids: List[str] = None,
        top_k: int = 5, 
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:

        """Query the leaf level of the vector store for similar chunks."""
        if not query_embedding:
            logger.warning("No query embedding provided")
            return []

        try: 
            # Add parent_ids to filter if provided
            if parent_ids:
                if filter is None:
                    filter = {}
                filter["parent_id"] = {"$in": parent_ids}

            # Query Pinecone
            query_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=self.leaf_namespace,
                include_metadata=True,
                filter=filter
            )

            # Format the results, log info then exit
            results = []
            for match in query_results.matches:
                result = {
                    "id": match.id,
                    "score": match.score,
                    "text": match.metadata.get("text", ""),
                    "metadata": {
                        k: v for k, v in match.metadata.items() if k != "text"
                    }
                }
                results.append(result)
            
            logger.info(f"Leaf query returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error querying leaf: {str(e)}")
            return []

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Pinecone index.

        """
        try:
            # Describe the index
            index_stats = self.pc.describe_index(self.index_name)
            
            # Get namespace statistics
            stats = self.index.describe_index_stats()
            
            # Format the statistics
            formatted_stats = {
                "index_name": self.index_name,
                "dimension": index_stats.dimension,
                "metric": index_stats.metric,
                "total_vector_count": stats.total_vector_count,
                "namespaces": stats.namespaces
            }
            
            return formatted_stats
        
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {"error": str(e)}
    
    def delete_all(self, namespace: Optional[str] = None) -> bool:
        """
        Delete all vectors from the specified namespace or all namespaces.

        """
        try:
            if namespace:
                logger.warning(f"Deleting all vectors from namespace: {namespace}")
                self.index.delete(delete_all=True, namespace=namespace)
                logger.info(f"Successfully deleted all vectors from namespace: {namespace}")
            else:
                # Delete from both namespaces
                logger.warning(f"Deleting all vectors from all namespaces")
                self.index.delete(delete_all=True, namespace=self.root_namespace)
                self.index.delete(delete_all=True, namespace=self.leaf_namespace)
                logger.info("Successfully deleted all vectors from all namespaces")
            
            return True
        
        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")
            return False


if __name__ == "__main__":
    vector_store = HierarchicalVectorStore()
    
    # Print index statistics
    stats = vector_store.get_index_stats()
    print(f"Index Statistics:")
    print(f"Index Name: {stats.get('index_name')}")
    print(f"Dimension: {stats.get('dimension')}")
    print(f"Total Vector Count: {stats.get('total_vector_count')}")
    

