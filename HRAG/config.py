import os
import logging
from dotenv import load_dotenv

# Configure logger
logging.basicConfig(level = logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# Load environment variables from .env file
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path)
logger.info(f"Loading environment variables from: {dotenv_path}")

# API KEYS
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
EMBEDDING_DIMENSIONS = 768

# Vector Store Configuration
PINECONE_INDEX_NAME = "hrag-gemini-768"
PINECONE_NAMESPACE_ROOT = "root"
PINECONE_NAMESPACE_LEAF = "leaf"

# LLM Configuration
LLM_MODEL_NAME = "meta-llama/llama-3.1-8b-instruct"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_REFERRER = "http://localhost:3000"

# Knowledge Base Configuration
KB_PATH = "/home/olande/Desktop/Rag_Techniques/HRAG/books"

# Document Processing Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SUMMARY_CHUNK_SIZE = 4000

# Retrieval Configuration
TOP_K_RESULTS = 5
TOP_K_ROOT = 3
TOP_K_LEAF = 5

# Validate configuration
def validate_config():
    """Validate that all required configuration values are present."""
    required_vars = {
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "PINECONE_API_KEY": PINECONE_API_KEY,
        "PINECONE_ENVIRONMENT": PINECONE_ENVIRONMENT,
        "OPENROUTER_API_KEY": OPENROUTER_API_KEY
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error(f"Please check your .env file at: {dotenv_path}")
        return False
    
    logger.info("All required configuration variables are present.")
    return True

# Check if KB_PATH exists
if not os.path.exists(KB_PATH):
    logger.warning(f"Knowledge base path does not exist: {KB_PATH}")
    logger.warning("Please check the path and ensure it contains the required documents.")