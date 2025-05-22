import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, 
    DirectoryLoader, 
    PyPDFLoader
)
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from config import (
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    SUMMARY_CHUNK_SIZE, 
    KB_PATH,
    OPENROUTER_API_KEY
)

# Configure Logging
logger = logging.getLogger(__name__)

class HierarchicalDocumentProcessor:
    def __init__(self, kb_path = KB_PATH):
        self.kb_path = kb_path

        # Text Splitter for the leaf nodes
        self.leaf_splitter = RecursiveCharacterTextSplitter(
            chunk_size = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP,
            length_function = len,
        )

        # Text Splitter for the root nodes  
        self.root_splitter = RecursiveCharacterTextSplitter(
            chunk_size = SUMMARY_CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP * 2,
            length_function = len
        )

        # Initialize LLM for summarization
        self.llm = ChatOpenAI(
            model_name="meta-llama/llama-3.1-8b-instruct",
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            max_tokens=1024,
            temperature=0.3
        )

        logger.info(f"Initialized HierarchicalDocumentProcessor with knowledge base path: {kb_path}")
        logger.info(f"Leaf chunk size: {CHUNK_SIZE}, Root chunk size: {SUMMARY_CHUNK_SIZE}")

    def load_documents(self) -> List[Dict[str, Any]]:
        """ Load all the documents from the knowledge base directory """
        if not os.path.exists(self.kb_path):
            logger.error(f"Knowledge base path does not exist: {self.kb_path}")
            return []
        
        logger.info(f"Loading documents from: {self.kb_path}")
        
        # Create loaders for different file types
        loaders = {
            ".txt": DirectoryLoader(
                self.kb_path, 
                glob="**/*.txt", 
                loader_cls=TextLoader,
                show_progress=True
            ),
            ".pdf": DirectoryLoader(
                self.kb_path, 
                glob="**/*.pdf", 
                loader_cls=PyPDFLoader,
                show_progress=True
            ),
        }

        all_documents = []
        for ext, loader in loaders.items():
            try:
                logger.info(f"Loading {ext} documents ...")
                docs = loader.load()
                logger.info(f"Loaded {len(docs)} {ext} documents")

                # Convert LangChain documents to our standard format
                for doc in docs:
                    all_documents.append({
                        "text": doc.page_content,
                        "metadata": {
                            "source": doc.metadata.get("source", "unknown"),
                            "page": doc.metadata.get("page", None),
                            "file_type": ext,
                        }
                    })
            except Exception as e:
                logger.error(f"Error loading {ext} documents: {str(e)}")
        
        logger.info(f"Loaded {len(all_documents)} total documents")
        return all_documents