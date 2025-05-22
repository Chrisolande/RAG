import logging
import json
from typing import Dict, Any, List, Optional
import requests
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_API_BASE,
    OPENROUTER_REFERRER,
    LLM_MODEL_NAME
)

# Configure logging
logger = logging.getLogger(__name__)

class LLMInterface:
    """Class for interacting with the LLM via OpenRouter."""
    
    def __init__(self):
        """Initialize the LLM interface."""
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is not set in the environment variables")
        
        self.api_key = OPENROUTER_API_KEY
        self.api_base = OPENROUTER_API_BASE
        self.model = LLM_MODEL_NAME
        self.referrer = OPENROUTER_REFERRER
        
        # Initialize LangChain LLM
        self.langchain_llm = ChatOpenAI(
            model_name=self.model,
            openai_api_key=self.api_key,
            openai_api_base=self.api_base,
            temperature=0.7
        )
        
        logger.info(f"Initialized LLMInterface with model: {self.model}")
    
    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        use_stream: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM.

        """
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt")
            return {"error": "Empty prompt", "text": ""}
        
        logger.info(f"Generating response for prompt: {prompt[:50]}...")
        
        # Prepare the messages
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        # Prepare the request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.referrer,  
            "X-Title": "Hierarchical RAG System",  
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        
        try:
            # Make the request to OpenRouter
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data
            )
            
            # Check if the request was successful
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
    
    def generate_hierarchical_rag_response(
        self,
        query: str,
        context: str,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate a response for a hierarchical RAG query.

        """
        # Create a system prompt for hierarchical RAG
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided hierarchical context.
            Follow these guidelines:
            1. Answer the question based ONLY on the provided context.
            2. The context is organized in sections, with each section containing a summary and relevant passages.
            3. If the context doesn't contain the information needed to answer the question, say "I don't have enough information to answer this question."
            4. Don't make up information that's not in the context.
            5. Provide detailed and accurate answers, citing the relevant sections when appropriate.
            6. Format your answer in a clear and readable way using markdown.
            7. If appropriate, use bullet points or numbered lists for better readability."""
        
        # Create the prompt with the query and hierarchical context
        prompt = f"""
            Question: {query}

            Hierarchical Context:
            {context}

            Please provide a comprehensive answer to the question based solely on the information in the hierarchical context above."""
        
        # Generate the response
        return self.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature)
    
    def get_langchain_llm(self):
        """
        Get the LangChain LLM object for use with LangChain.

        """
        return self.langchain_llm


if __name__ == "__main__":
    # Test LLMInterface
    llm = LLMInterface()
    
    test_prompt = "Explain the concept of hierarchical retrieval in simple terms."
    response = llm.generate_response(test_prompt)
    print(f"Generated response: {response['text']}")
        
   
    test_query = "What is the declaration of independence?"
    test_context = """
[Section 1] (Source: HRAG/books/declaration_of_independence_of_the_united_states.txt)

Summary: This section contains the introduction and preamble to the Declaration of Independence, explaining the necessity of declaring independence from Great Britain and asserting the right of the people to establish a new government.

Passage 1: When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation.

Passage 2: We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness.--That to secure these rights, Governments are instituted among Men, deriving their just powers from the consent of the governed.
"""
    rag_response = llm.generate_hierarchical_rag_response(test_query, test_context)
    print(f"\nHierarchical RAG response: {rag_response['text']}")
