import os
from dotenv import load_dotenv
from hyde import connect_to_pinecone, create_pinecone_index, insert_documents_to_pinecone, search_pinecone, initialize_llm

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
    
    # Perform search
    query = "How long does wisdom tooth extraction take?"
    results = search_pinecone(index, query)
    
    print("\nSearch Results:")
    for doc, score in results:
        print(f"Score: {score:.4f} - {doc}")

if __name__ == "__main__":
    main()
