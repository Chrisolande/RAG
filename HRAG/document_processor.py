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

    def create_hierarchical_chunks(self, documents: List[Dict[str, Any]]):
        """ Split documents into hierarchical chunks with root and leaf nodes. """
        if not documents:
            logger.warning("No documents to chunk")
            return [], []

        logger.info(f"Creating hierarchical chunks for {len(documents)} documents")
        root_chunks = []
        leaf_chunks = []

        for doc_idx, doc in enumerate(documents):
            text = doc["text"]
            metadata = doc["metadata"]
            
            # Create root/parent chunks (larger sections)
            root_texts = self.root_splitter.split_text(text)
            
            for root_idx, root_text in enumerate(root_texts):
                # Create a unique ID for the root chunk
                root_id = f"root_{doc_idx}_{root_idx}"
                
                # Create the root chunk
                root_chunk = {
                    "id": root_id,
                    "text": root_text,
                    "metadata": {
                        **metadata,
                        "chunk_type": "root",
                        "chunk_id": root_idx,
                        "doc_id": doc_idx,
                    }
                }
                
                # Add to root chunks
                root_chunks.append(root_chunk)
                
                # Create leaf chunks from this root chunk
                leaf_texts = self.leaf_splitter.split_text(root_text)
                
                for leaf_idx, leaf_text in enumerate(leaf_texts):
                    # Create a unique ID for the leaf chunk that references its parent
                    leaf_id = f"leaf_{doc_idx}_{root_idx}_{leaf_idx}"
                    
                    # Create the leaf chunk with reference to its parent
                    leaf_chunk = {
                        "id": leaf_id,
                        "text": leaf_text,
                        "metadata": {
                            **metadata,
                            "chunk_type": "leaf",
                            "chunk_id": leaf_idx,
                            "parent_id": root_id,
                            "doc_id": doc_idx,
                        }
                    }
                    
                    # Add to leaf chunks
                    leaf_chunks.append(leaf_chunk)
        
        logger.info(f"Created {len(root_chunks)} root chunks and {len(leaf_chunks)} leaf chunks")
        return root_chunks, leaf_chunks

    def generate_summaries_for_root_chunks(self, root_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate summaries for root chunks to improve retrieval.
        This will ensure less relevant info are not passed to the LLM
        """
        if not root_chunks:
            return []
        
        logger.info(f"Generating summaries for {len(root_chunks)} root chunks")
        
        # Load the summarization chain
        summarize_chain = load_summarize_chain(
            llm=self.llm,
            chain_type="stuff",
            verbose=False
        )
        
        root_chunks_with_summaries = []
        
        for i, chunk in enumerate(root_chunks):
            try:
                # Convert to LangChain document format
                doc = Document(
                    page_content=chunk["text"],
                    metadata=chunk["metadata"]
                )
                
                # Generate summary
                summary = summarize_chain.run([doc])
                
                # Add summary to the chunk
                chunk_with_summary = {
                    **chunk,
                    "summary": summary
                }
                
                root_chunks_with_summaries.append(chunk_with_summary)
                
                # Log progress
                if (i + 1) % 5 == 0 or i == len(root_chunks) - 1:
                    logger.info(f"Generated summaries for {i + 1}/{len(root_chunks)} root chunks")
                
            except Exception as e:
                logger.error(f"Error generating summary for chunk {i}: {str(e)}")
                # Add the original chunk without summary
                root_chunks_with_summaries.append(chunk)
        
        return root_chunks_with_summaries
    
    def process_documents(self):
        """
        Load and process documents from the knowledge base with hierarchical chunking.
      
        """
        # Load documents
        documents = self.load_documents()
        
        # Create hierarchical chunks
        root_chunks, leaf_chunks = self.create_hierarchical_chunks(documents)
        
        # Generate summaries for root chunks
        root_chunks_with_summaries = self.generate_summaries_for_root_chunks(root_chunks)
        
        return root_chunks_with_summaries, leaf_chunks


def get_document_stats(kb_path: str = KB_PATH) -> Dict[str, Any]:
    """
    Get statistics about the documents in the knowledge base.

    """
    if not os.path.exists(kb_path):
        logger.error(f"Knowledge base path does not exist: {kb_path}")
        return {"error": "Knowledge base path does not exist"}
    
    stats = {
        "total_files": 0,
        "file_types": {},
        "total_size_bytes": 0,
    }
    
    for root, _, files in os.walk(kb_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            # Update statistics
            stats["total_files"] += 1
            stats["file_types"][file_ext] = stats["file_types"].get(file_ext, 0) + 1
            stats["total_size_bytes"] += os.path.getsize(file_path)
    
    # Convert bytes to MB for readability
    stats["total_size_mb"] = round(stats["total_size_bytes"] / (1024 * 1024), 2)
    
    return stats


if __name__ == "__main__":
    # Test if everythin is okay!?
    processor = HierarchicalDocumentProcessor()
    root_chunks, leaf_chunks = processor.process_documents()
    
    print(f"Processed {len(root_chunks)} root chunks and {len(leaf_chunks)} leaf chunks")
    
    # Print document statistics
    stats = get_document_stats()
    print(f"Knowledge Base Statistics:")
    print(f"Total Files: {stats['total_files']}")
    print(f"Total Size: {stats['total_size_mb']} MB")
    print(f"File Types: {stats['file_types']}")

    