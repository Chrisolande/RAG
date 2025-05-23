from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional
from config import OPENROUTER_API_KEY, MODEL_NAME

class Entities(BaseModel):
    """Pydantic model for entity extraction results"""
    names: List[str] = Field(description = "List of entity names extracted from the text")

class EntityExtractor:
    """Handles entity extraction from user queries"""
    def __init__(self, model: Optional[str] = None,
                 api_key: Optional[str] = None):
        
        self.api_key = api_key or OPENROUTER_API_KEY
        self.llm = ChatOpenAI(model = model or MODEL_NAME,
                              temperature = 0,
                              api_key = self.api_key,
                              openai_api_base = "https://openrouter.ai/api/v1")


        # Create the extraction prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at identifying entities in text."),
            ("user", "Extract all entities from this question: {question}")
        ])

        # Create extraction chain
        self.entity_chain = (
            self.prompt 
            | self.llm.with_structured_output(Entities)
        )

    def extract_entities(self, question: str) -> List[str]:
        """
        Extract entities from a question.

        """
        result = self.entity_chain.invoke({"question": question})
        return result.names

    def generate_full_text_query(self, entity: str) -> str:
        """Generate a full-text search query for a given entity"""
        return f"{entity}~"
        