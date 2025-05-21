import os
from typing import List, Optional, Dict, Any

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv
from tqdm.auto import tqdm

# Load environment variables
load_dotenv()

class VectorstoreManager:
    def __init__(self,
                embeddings: Embeddings,
                index_name: str = "rag-experiment",
                namespace_standard: str = "standard",
                namespace_contextual: str = "contextual"):

        self.embeddings = embeddings
        self.index_name = index_name
        self.namespace_standard = namespace_standard
        self.namespace_contextual = namespace_contextual
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Create index if it doesn't exist
        self._create_index_if_not_exists()
        
    def _create_index_if_not_exists(self):
        # List existing indexes
        existing_indexes = [index.name for index in self.pc.list_indexes()]

        # Create index if it doesn't exist
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=768,  # Dimension for Gemini text-embedding-004
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

    def get_vector_store(self, namespace: Optional[str] = None)  -> PineconeVectorStore:
        return PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace=namespace
        )

    def get_standard_vector_store(self) -> PineconeVectorStore:
        return self.get_vector_store(self.namespace_standard)

    def get_contextual_vector_store(self) -> PineconeVectorStore:
        return self.get_vector_store(self.namespace_contextual)

    def add_documents(self, documents: List[Document], namespace: Optional[str] = None, batch_size: int = 100, **kwargs):
        vector_store = self.get_vector_store(namespace)

        # Add documents in batches to avoid potential issues with large no of docs
        num_batches = (len(documents) + batch_size - 1) // batch_size
        print(f"Adding {len(documents)} documents to vector store in {num_batches} batches...")

        for i in tqdm(range(0, len(documents), batch_size), total = num_batches, desc = f"Adding to {namespace or "default"} namespace"):
            batch = documents[i:i + batch_size]
            PineconeVectorStore.from_documents(
                documents=batch,
                embedding=self.embeddings,
                index_name=self.index_name,
                namespace=namespace,
                **kwargs
            )
    
    def add_standard_documents(self, documents: List[Document], **kwargs):
        self.add_documents(documents, namespace=self.namespace_standard, **kwargs)

    def add_contextual_documents(self, documents: List[Document], **kwargs):
        self.add_documents(documents, namespace=self.namespace_contextual, **kwargs)

    def clear_namespace(self, namespace: Optional[str] = None):
        try: 
            index = self.pc.Index(self.index_name)
            index.delete(delete_all = True, namespace=namespace)
            print(f"Cleared namespace: {namespace or 'default'}")

        except Exception as e:
            # Handle the case when the namespace doesn't existyet
            if "Namespace not found" in str(e):
                print(f"Namespace {namespace or 'default'} doesn't exist yet. Nothing to clear.")

            else:
                # Re-raise other exceptions
                raise e

    def clear_all(self):
        self.clear_namespace(self.namespace_standard)
        self.clear_namespace(self.namespace_contextual)