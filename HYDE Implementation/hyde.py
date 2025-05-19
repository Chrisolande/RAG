import os
import google.generativeai as genai
from pinecone import Pinecone
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI

# Initialize the LLM
def initialize_llm(model_name="meta-llama/llama-3.1-8b-instruct",
                  temperature=0.4,
                  use_streaming=True):

    api_key = os.getenv("OPENROUTER_API_KEY")
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        streaming=use_streaming,
        callbacks=callbacks,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1"
    )
    return llm


# Connect to Pinecone
def connect_to_pinecone(api_key = os.getenv("PINECONE_API_KEY")):
    """
    Connect to Pinecone with your API key
    """
    # Initialize the Pinecone client with the new API
    pc = Pinecone(api_key=api_key)
    print("Connected to Pinecone")
    return pc

# Create Pinecone index with 768 dimensions to match existing index
def create_pinecone_index(dimension=768, index_name="hyde-index", metric="cosine"):
    """
    Create or connect to a Pinecone index
    """
    pc = connect_to_pinecone()
    
    # Check if index already exists
    if index_name not in pc.list_indexes().names():
        # Create index with the new API - using gcp-starter which is free tier compatible
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric
            # Free tier automatically uses gcp-starter, no need to specify
        )
        print(f"Created new index: {index_name}")
    else:
        print(f"Using existing index: {index_name}")
    
    # Return the index
    return pc.Index(index_name)

# Get embeddings that match the 768 dimension requirement
def get_embedding(text, model_name="models/embedding-001"):
    """
    Get 768-dimensional embeddings using Google's text-embedding-gecko model
    
    This model produces 768-dimensional embeddings, which match your Pinecone index
    """
    result = genai.embed_content(
        model=model_name,
        content=text,
        task_type="RETRIEVAL_DOCUMENT"
    )
    
    return result["embedding"]

# Insert documents to Pinecone
def insert_documents_to_pinecone(index, documents, namespace=""):
    """
    Insert documents with their embeddings to Pinecone
    """
    # Generate embeddings that match the dimension of the index (768)
    embeddings = []
    for doc in documents:
        embedding = get_embedding(doc)
        embeddings.append(embedding)
    
    # Prepare data for upsert
    vectors = []
    for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
        vectors.append({
            "id": f"doc_{i}",
            "values": embedding,
            "metadata": {"text": doc}
        })
    
    # Upsert vectors in batches
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch, namespace=namespace)
    
    print(f"Inserted {len(vectors)} documents into Pinecone")
    return len(vectors)

# Search in Pinecone
def search_pinecone(index, query, top_k=2):
    """
    Search for similar documents in Pinecone
    """
    # Get embedding for the query
    query_embedding = get_embedding(query)
    
    # Query the index
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Format results
    search_results = []
    for match in results["matches"]:
        search_results.append((match["metadata"]["text"], match["score"]))
    
    return search_results
