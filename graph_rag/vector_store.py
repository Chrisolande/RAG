from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from typing import List, Optional
from config import GEMINI_API_KEY, EMBEDDING_MODEL
from knowledge_graph import KnowledgeGraph

class VectorStore:
    def __init__(
        self, 
        knowledge_graph: KnowledgeGraph,
        embedding_model: Optional[str] = None
    ):
        self.knowledge_graph = knowledge_graph
        self.embedding_model = embedding_model or EMBEDDING_MODEL
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model = f"models/{self.embedding_model}",
            google_api_key = GEMINI_API_KEY
        )

        # Initialize the vector index
        self.vector_index = None

    def create_vector_index(
        self,
        documents: List[Document],
        node_label: str = "Document",
        text_node_property: str = "text",
        embedding_node_property: str = "embedding"
    ):
        """Create vector index from documents"""

        self.vector_index = Neo4jVector.from_documents(
            documents,
            self.embeddings,
            url = self.knowledge_graph.url,
            username = self.knowledge_graph.username,
            password = self.knowledge_graph.password,
            index_name = "vector",
            node_label = node_label,
            text_node_property = text_node_property,
            embedding_node_property = embedding_node_property
        )

    def create_hybrid_index(
        self,
        node_label: str = "Document",
        text_node_properties: List[str] = ["text"],
        embedding_node_property:str = "embedding"
    ):
        """Create hybrid index"""
        self.vector_index = Neo4j.from_existing_graph(
            self.embeddings,
            url = self.knowledge_graph.uri,
            username = self.knowledge_graph.username,
            password = self.knowledge_graph.password,
            search_type = "hybrid",
            node_label = node_label,
            text_node_properties = text_node_properties,
            embedding_node_property = embedding_node_property
        )

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search on the vector index"""
        if self.vector_index is None:
            raise ValueError("Vector index not initialized. Call create_vector_index or create_hybrid_index first.")
        return self.vector_index.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """Perform similarity search with scores on the vector index"""
        if self.vector_index is None:
            raise ValueError("Vector index not initialized. Call create_vector_index or create_hybrid_index first")
        return self.vector_index.similarity_search_with_score(query, k=k)