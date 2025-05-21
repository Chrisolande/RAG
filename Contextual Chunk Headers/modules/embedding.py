import os
from typing import List, override

import google.generativeai as genai
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

load_dotenv()

# Configure google genai api key
genai.configure(api_key = os.getenv("GEMINI_API_KEY"))

class GeminiEmbeddings(Embeddings):
    def __init__(self, model_name: str = "text-embedding-004", task_type:str = "RETRIEVAL_DOCUMENT"):
        self.model_name = model_name
        self.task_type = task_type

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            embedding = genai.embed_content(model =self.model_name, content = text, task_type = self.task_type)
            embeddings.append(embedding["embedding"])

        return embeddings

    def embed_query(self, text:str) -> List[float]:
        embedding = genai.embed_content(model = self.model_name, content = text, task_type = "RETRIEVAL_QUERY")
        return embedding["embedding"]