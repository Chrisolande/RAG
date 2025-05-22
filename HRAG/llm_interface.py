import logging
import json
import requests
from typing import List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_API_BASE,
    OPENROUTER_REFERRER,
    LLM_MODEL_NAME
)

# Configure Logging
logger = logging.getLogger(__name__)

class LLMInterface:
    def __init__(self) -> None:
        """Initialize the LLM interface."""
        self._api_key = OPENROUTER_API_KEY
        self._api_base = OPENROUTER_API_BASE
        self._referrer = OPENROUTER_REFERRER
        self._model_name = LLM_MODEL_NAME

        if not self._api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")

        self.langchain_llm = ChatOpenAI(
            model_name=self._model_name,
            openai_api_key=self._api_key,
            openai_api_base=self._api_base,
            temperature=0.5
        )

        logger.info(f"Successfully initialized the model with {self._model_name}")
    
    def generate_response(self,
                        prompt: str,
                        system_prompt: Optional[str] = None,
                        temperature = 0.5) -> Dict[str, Any]:


        """ Generate response from the llm"""
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt")
            return {"error": "Empty prompt", "text": ""}

        logger.info(f"Generating response for prompt: {prompt[:50]} ...")

        # Prepare the messages
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add user message
        messages.append({"role": "user", "content": prompt})

        # Prepare the request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.referrer, 
            "X-Title": "Hierarchical RAG System", 
            "Content-Type": "application/json"
        }

        data = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature,
        }

        try:
            response = response.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data
            )

            response.raise_for_status()

            # Parse the response
            result = response.json()

            # Extract the generated text
            generated_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            # Format the response
            formatted_response = {
                "text": generated_text,
                "model": result.get("model", self.model),
                "usage": result.get("usage", {}),
                "id": result.get("id", "")
            }
            
            logger.info(f"Generated response: {generated_text[:50]}...")
            return formatted_response
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {"error": str(e), "text": ""}

