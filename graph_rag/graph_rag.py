from document_processor import DocumentProcessor
from knowledge_graph import KnowledgeGraph
from vector_store import VectorStore
from entity_extraction import EntityExtractor
from retrieval import Retriever
from rag_chain import RagChain
from typing import Optional, List
from config import MODEL_NAME, EMBEDDING_MODEL
class GraphRagAPP:
    def __init__(
        self,
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None
    ):

        self.embedding_model = embedding_model or EMBEDDING_MODEL
        self.llm_model = llm_model or MODEL_NAME

        self.document_processor = DocumentProcessor()
        self.knowledge_graph = KnowledgeGraph()
        self.vector_store = VectorStore(self.knowledge_graph, embedding_model = self.embedding_model)
        self.entity_extractor = EntityExtractor(model = self.llm_model)
        self.retriever = Retriever(self.knowledge_graph, self.vector_store, self.entity_extractor)
        self.rag_chain = RagChain(self.retriever, model = self.llm_model)

        # Initialize the chat history
        self.chat_history = []

    def initialize_from_kb(self, document: List[str]):
        """ Initialize the systemwith the documents from the knowledge base"""
        print(f"Loading documents from the knowledge base...")
        documents = self.document_processor.process_documents()
        self.knowledge_graph.add_documents(documents)

        print(f"Creating knowledge graph from {len(documents)}")
        self.knowledge_graph.clear_database()
        self.knowledge_graph.create_graph_from_documents(documents)

        print(f"Creating the vector index")
        self.vector_store.create_hybric_index()

        print("Initialization complete")

    def ask(self, question: str) -> str:
        """ Ask a question to the system"""
        answer= self.rag_chain.invoke({
            "question": question,
            "chat_history": self.chat_history
        })

        # Update the chat history
        self.chat_history.append((question, answer))
        return answer

    def reset_chat_history(self):
        """Reset the chat history."""
        self.chat_history = []
