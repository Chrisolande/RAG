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

    