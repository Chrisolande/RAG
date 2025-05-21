
import os
import argparse
import time
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from dotenv import load_dotenv

from embedding import GeminiEmbeddings
from document_processor import DocumentProcessor, ContextualHeaderProcessor
from vector_store import VectorstoreManager
from retrieval import StandardRetriever, ContextualHeaderRetriever
from llm_interface import get_openrouter_llm, StandardRAGChain, ContextualHeaderRAGChain
from evaluation import RAGEvaluator

# Load environment variables
load_dotenv()


def setup_rag_systems(
    docs_dir: str = "../books",
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    rebuild_index: bool = False
):

    print("Setting up RAG systems...")
    
    # Initialize embedding model
    print("Initializing embedding model...")
    embeddings = GeminiEmbeddings()
    
    # Initialize LLM
    print("Initializing LLM...")
    llm = get_openrouter_llm()
    
    # Initialize vector store manager
    print("Initializing vector store...")
    vector_store_manager = VectorstoreManager(embeddings)
    
    # Process documents and build vector stores if needed
    if rebuild_index:
        print("Processing documents for standard RAG...")
        # Standard document processing
        doc_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            docs_dir=docs_dir
        )
        standard_docs = doc_processor.process_documents()
        
        print(f"Processed {len(standard_docs)} document chunks for standard RAG")
        
        print("Processing documents for contextual header RAG...")
        # Contextual header document processing
        contextual_processor = ContextualHeaderProcessor(
            llm=llm,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            docs_dir=docs_dir
        )
        contextual_docs = contextual_processor.process_documents_with_headers()
        
        print(f"Processed {len(contextual_docs)} document chunks for contextual header RAG")
        
        # Clear existing vector stores
        print("Clearing existing vector stores...")
        vector_store_manager.clear_all()
        
        # Add documents to vector stores
        print("Adding documents to standard RAG vector store...")
        vector_store_manager.add_standard_documents(standard_docs)
        
        print("Adding documents to contextual header RAG vector store...")
        vector_store_manager.add_contextual_documents(contextual_docs)
    
    # Get vector stores
    standard_vector_store = vector_store_manager.get_standard_vector_store()
    contextual_vector_store = vector_store_manager.get_contextual_vector_store()
    
    # Initialize retrievers
    standard_retriever = StandardRetriever(standard_vector_store)
    contextual_retriever = ContextualHeaderRetriever(contextual_vector_store)
    
    # Initialize RAG chains
    standard_rag = StandardRAGChain(llm, standard_retriever)
    contextual_rag = ContextualHeaderRAGChain(llm, contextual_retriever)
    
    # Initialize evaluator
    evaluator = RAGEvaluator(embeddings)
    
    return standard_rag, contextual_rag, evaluator


def run_interactive_mode(standard_rag, contextual_rag, evaluator):

    print("\n=== Interactive Mode ===")
    print("Enter your questions (or 'q' to quit):")
    
    while True:
        query = input("\nQuestion: ")
        
        if query.lower() in ['q', 'quit', 'exit']:
            break
        
        print("\nProcessing...")
        
        # Measure standard RAG performance
        standard_start = time.time()
        standard_response = standard_rag.invoke(query)
        standard_time = time.time() - standard_start
        
        # Measure contextual header RAG performance
        contextual_start = time.time()
        contextual_response = contextual_rag.invoke(query)
        contextual_time = time.time() - contextual_start
        
        # Print results
        print("\n=== Standard RAG ===")
        print(f"Time: {standard_time:.2f}s")
        print(f"Response: {standard_response}")
        
        print("\n=== Contextual Header RAG ===")
        print(f"Time: {contextual_time:.2f}s")
        print(f"Response: {contextual_response}")


def run_benchmark_mode(standard_rag, contextual_rag, evaluator, output_path: Optional[str] = None):
 
    print("\n=== Benchmark Mode ===")
    

    queries = [
        "What are the key principles in the Declaration of Independence?",
        "Who wrote the Declaration of Independence?",
        "What grievances were listed in the Declaration of Independence?",
        "What rights are mentioned in the Declaration of Independence?",
        "When was the Declaration of Independence signed?",
    ]
    
    print(f"Running evaluation on {len(queries)} queries...")
    
    # Run evaluation
    evaluator.run_evaluation(queries, standard_rag, contextual_rag)
    
    # Get summary metrics
    summary = evaluator.get_summary_metrics()
    
    # Print summary
    print("\n=== Evaluation Results ===")
    print("\nStandard RAG:")
    for metric, value in summary["standard"].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nContextual Header RAG:")
    for metric, value in summary["contextual"].items():
        print(f"  {metric}: {value:.4f}")
    
    # Visualize results
    print("\nGenerating visualization...")
    evaluator.visualize_results(output_path)
    
    if output_path:
        print(f"Visualization saved to {output_path}")
    else:
        print("Visualization displayed")


def main():
    """
    Main function for the RAG experiment.
    """
    parser = argparse.ArgumentParser(description="RAG Experiment CLI")
    parser.add_argument(
        "--mode", 
        choices=["interactive", "benchmark"], 
        default="interactive",
        help="Mode to run the RAG experiment in"
    )
    parser.add_argument(
        "--rebuild", 
        action="store_true",
        help="Rebuild the vector store index"
    )
    parser.add_argument(
        "--docs-dir", 
        default="../books",
        help="Directory containing documents to process"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=1000,
        help="Size of text chunks for splitting"
    )
    parser.add_argument(
        "--chunk-overlap", 
        type=int, 
        default=200,
        help="Overlap between chunks"
    )
    parser.add_argument(
        "--output", 
        help="Path to save the benchmark visualization"
    )
    
    args = parser.parse_args()
    
    # Setup RAG systems
    standard_rag, contextual_rag, evaluator = setup_rag_systems(
        docs_dir=args.docs_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        rebuild_index=args.rebuild
    )
    
    # Run in the specified mode
    if args.mode == "interactive":
        run_interactive_mode(standard_rag, contextual_rag, evaluator)
    else:  # benchmark mode
        run_benchmark_mode(standard_rag, contextual_rag, evaluator, args.output)


if __name__ == "__main__":
    main()
