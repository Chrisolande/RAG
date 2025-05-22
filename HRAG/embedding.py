# Substituting the prior approach to using gemini embedding models with a Langchain one!
import logging
from typing import List, Dict, Any, Union
import numpy as np
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import GEMINI_API_KEY, EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSIONS

#Configure Logging
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("Gemini API Key is not set in the environment variables")

        # Configure the Gemini API Key
        genai.configure(api_key = GEMINI_API_KEY)
        self.embedding_model = EMBEDDING_MODEL_NAME
        self.embedding_dimensions = EMBEDDING_DIMENSIONS

        # Initialize the embedding model
        self.langchain_embeddings = GoogleGenerativeAIEmbeddings(
            model = self.embedding_model.replace("models/", ""),
            task_type = "retrieval_document",
            google_api_key = GEMINI_API_KEY
        )

        logger.info(f"Initialized EmbeddingGenerator with model: {self.embedding_model}")
        logger.info(f"Embedding dimensions: {self.embedding_dimensions}")

    def generate_embedding(self, text:str, is_query: bool = False):
        """ Generate an embedding for a single text"""
        if not text or not text.strip():
            logger.warning("Attempted to generate embedding for empty text")
            # Return a vector of zeros
            return [0.0] * self.embedding_dimensions

        try:
            # Truncate long texts due to gemini's token limits
            if len(text) > 10000:
                logger.warning(f"Text is too long ({len(text)} characters), truncating to 10000 characters")
                text = text[:10000]

            # Generate embedding using Gemini
            task_type = "retrieval_query" if is_query else "retrieval_document"
            embedding = genai.embed_content(
                model = self.embedding_model,
                content = text,
                task_type = task_type
            )

            # Extract the embedding values
            embedding_values = embedding["embedding"]
            # Verify dimensions
            if len(embedding_values) != self.embedding_dimensions:
                logger.warning(
                    f"Expected embedding dimension {self.embedding_dimensions}, "
                    f"but got {len(embedding_values)}"
                )
            
            return embedding_values
        
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return a zero vector on error
            return [0.0] * self.embedding_dimensions

    def generate_embeddings_for_chunks(self, chunks, chunk_type):
        """ Generate embeddings for a list of chunks"""
        if not chunks:
            logger.warning(f"No {chunk_type} chunks to generate embeddings for")
            return []

        logger.info(f"Generating embeddings for {len(chunks)} {chunk_type} chunks")
        chunks_with_embeddings = []

        for i, chunk in enumerate(chunks):
            try: 
                # For root chunnks, we use the summary if provided
                if chunk_type == "root" and "summary" in chunk:
                    summary_embedding = self.generate_embedding(chunk["summary"])

                    # Generate embedding for the full text
                    text_embedding = self.generate_embedding(chunk["text"])

                    # Add both embeddings to the chunk
                    chunk_with_embedding  = {
                        **chunk,
                        "summary_embedding": summary_embedding,
                        "text_embedding": text_embedding
                    }
                else:
                    # For leaf chunks or root chunks without summaries
                    text = chunk["text"]
                    embedding = self.generate_embedding(text)
                    
                    # Add the embedding to the chunk
                    chunk_with_embedding = {
                        **chunk,
                        "embedding": embedding
                    }
                
                chunks_with_embeddings.append(chunk_with_embedding)
                
                # Log progress periodically
                if (i + 1) % 10 == 0 or i == len(chunks) - 1:
                    logger.info(f"Generated embeddings for {i + 1}/{len(chunks)} {chunk_type} chunks")
            
            except Exception as e:
                logger.error(f"Error generating embeddings for {chunk_type} {i}: {str(e)}")
        logger.info(f"Generated embeddings for {len(chunks_with_embeddings)}/{len(chunks)} {chunk_type} chunks")
        return chunks_with_embeddings

    def generate_query_embedding(self, query):
        """Generate Embedding for the query"""
        return self.generate_embedding(query, is_query = True)  

    def get_langchain_embeddings(self):
        """Get the langchain compatible embeddings"""
        return self.langchain_embeddings