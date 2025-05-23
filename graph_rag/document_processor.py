from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import TokenTextSplitter
from typing import List
from langchain_core.documents import Document
from config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, KNOWLEDGE_BASE_PATH

class DocumentProcessor:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int= 24):
        """Initialize the document processor"""

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = TokenTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap
        )

    
    def load_documents(self, knowledge_base_path: str = KNOWLEDGE_BASE_PATH):
        # Add Error handling in case the knowledge base doesn't exist
        
        loader = DirectoryLoader(
            knowledge_base_path,
            glob = "**/*.txt",
            loader_cls = TextLoader
        )

        documents = loader.load()
        return documents
