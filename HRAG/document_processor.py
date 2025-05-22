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

