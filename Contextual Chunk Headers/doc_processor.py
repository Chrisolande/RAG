import os
from typing import Dict, List, Any, Optional

from langchain_core.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from tqdm.auto import tqdm

class DocumentProcessor:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200, docs_dir: str = "books"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.docs_dir = docs_dir

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap
        )

    def load_documents(self) -> List[Document]:
        if not os.path.exists(self.docs_dir):
            raise FileNotFoundError(f"The directory {self.docs_dir} does not exist.")

        loader = DirectoryLoader(self.docs_dir, loader_cls = TextLoader,
        glob = "**/*.txt")
        return loader.load()

    def split_documents(self, documents: List[Document]) -> List[Document]:
        print(f"Splitting {len(documents)}")
        return self.text_splitter.split_documents(documents)

    def process_documents(self) -> List[Document]:
        """Load and split the documents"""
        documents = self.load_documents()
        return self.split_documents(documents)



