import os
import logging
from typing import List, Optional, Dict, Type

from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.schema import Document
from langchain.text_splitter import TokenTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    SUPPORTED_EXTENSIONS: Dict[str, Type] = {
        "txt": TextLoader,
        "pdf": PyPDFLoader,
    }

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 128, extension: str = "txt"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extension = extension
        self.text_splitter = TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def _get_loader(self, extension: str, directory_path: str) -> DirectoryLoader:
        loader_cls = self.SUPPORTED_EXTENSIONS[extension]
        return DirectoryLoader(
            directory_path,
            glob=f"**/*.{extension}",
            loader_cls=loader_cls,
            show_progress=True
        )

    def load_documents(self, directory_path: str, extension: Optional[str] = None) -> List[Document]:
        extension = extension or self.extension

        if not os.path.isdir(directory_path):
            raise FileNotFoundError(f"Directory '{directory_path}' does not exist or is not accessible.")

        extensions_to_load = (
            list(self.SUPPORTED_EXTENSIONS.keys()) if extension == "all" else [extension]
        )

        all_documents = []
        for ext in extensions_to_load:
            if ext not in self.SUPPORTED_EXTENSIONS:
                logger.warning(f"Skipping unsupported extension: {ext}")
                continue

            try:
                loader = self._get_loader(ext, directory_path)
                documents = loader.load()
                logger.info(f"Loaded {len(documents)} {ext.upper()} documents.")
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error loading {ext.upper()} documents: {str(e)}")

        if not all_documents:
            raise ValueError(f"No supported documents found in '{directory_path}'.")

        return all_documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        logger.info("Splitting documents...")
        return self.text_splitter.split_documents(documents)

    def process_documents(self, directory_path: str, extension: Optional[str] = None) -> List[Document]:
        documents = self.load_documents(directory_path, extension)
        return self.split_documents(documents)

if __name__ == "__main__":
    try:
        processor = DocumentProcessor()
        docs = processor.process_documents(
            "/home/olande/Desktop/Rag_Techniques/RAPTOR/knowledge_base",
            "all"  # "txt", "pdf", or "all"
        )
        print(f"Loaded and split {len(docs)} chunks. First:\n{docs[3]}")
    except Exception as e:
        logger.error(f"Error: {e}")
