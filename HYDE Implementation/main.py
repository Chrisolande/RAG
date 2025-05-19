import os
from dotenv import load_dotenv
from hyde import (connect_to_pinecone, create_pinecone_index, insert_documents_to_pinecone,
                search_pinecone, initialize_llm, hyde_search)

# Load environment variables
load_dotenv()

# Test corpus of documents
corpus = [
    "It usually takes between 30 minutes and two hours to remove a wisdom tooth.",
    "The COVID-19 pandemic has significantly impacted mental health, increasing depression and anxiety.",
    "Humans have used fire for approximately 800,000 years.",
    "Milvus is a cloud based database for vector storage."
]

def main():
    # Connect to Pinecone
    connect_to_pinecone()
    llm = initialize_llm()
    # Create or connect to index with 768 dimensions
    index = create_pinecone_index(dimension=768)
    
    # Insert documents
    insert_documents_to_pinecone(index, corpus)
    

    query = "How long does wisdom tooth extraction take?"
    
    print("\n========== TRADITIONAL SEARCH ==========\n")
    # Perform traditional search
    traditional_results = search_pinecone(index, query)
    
    print("\nTraditional Search Results:")
    for doc, score in traditional_results:
        print(f"Score: {score:.4f} - {doc}")
    
    print("\n\n========== HYDE SEARCH ==========\n")
    # Perform HyDE search
    hyde_results = hyde_search(index, query, llm, use_hyde=True)
    
    print("\nHyDE Search Results:")
    for doc, score in hyde_results:
        print(f"Score: {score:.4f} - {doc}")
        
    # Additional example query for comparison
    print("\n\n========== ANOTHER EXAMPLE ==========")
    query2 = "What health effects did COVID cause?"
    
    print("\n--- Traditional Search ---")
    trad_results2 = search_pinecone(index, query2)
    for doc, score in trad_results2:
        print(f"Score: {score:.4f} - {doc}")
        
    print("\n--- HyDE Search ---")
    hyde_results2 = hyde_search(index, query2, llm, use_hyde=True)
    for doc, score in hyde_results2:
        print(f"Score: {score:.4f} - {doc}")

if __name__ == "__main__":
    main()
    print("\nComparison complete! HyDE generates hypothetical documents that better match the relevant content.")
    print("This often leads to improved retrieval performance, especially for complex queries.")

