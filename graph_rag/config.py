import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# OpenAI Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Google Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Default parameters
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 24

KNOWLEDGE_BASE_PATH = "graph_rag/knowledge_base"

MODEL_NAME = "mistralai/mistral-7b-instruct"
EMBEDDING_MODEL = "models/text-embedding-004"