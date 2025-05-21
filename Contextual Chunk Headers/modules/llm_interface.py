"""
LLM integration module for the RAG experiment.
Uses OpenRouter for LLM access via LangChain's ChatOpenAI integration.
"""
import os
from typing import Dict, Any, Optional, List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_openrouter_llm(
    model_name: str = "meta-llama/llama-3.2-1b-instruct",
    temperature: float = 0.1,
    **kwargs
) -> ChatOpenAI:
    """
    Get a ChatOpenAI instance configured to use OpenRouter.
    
    Args:
        model_name: Name of the model to use
        temperature: Temperature parameter for the model
        **kwargs: Additional arguments for ChatOpenAI
        
    Returns:
        ChatOpenAI instance
    """
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        **kwargs
    )


class RAGChain:
    """
    Base class for RAG chains.
    """
    
    def __init__(
        self,
        llm,
        retriever,
        prompt_template: Optional[str] = None
    ):
        """
        Initialize the RAGChain.
        
        Args:
            llm: Language model to use
            retriever: Retriever to use
            prompt_template: Optional custom prompt template
        """
        self.llm = llm
        self.retriever = retriever
        
        # Use default prompt template if none provided
        if prompt_template is None:
            prompt_template = """You are a helpful assistant that provides accurate information based on the context provided.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer the question based on the context provided. If the answer cannot be found in the context, say "I don't have enough information to answer this question." and suggest what information might help.
            """
        
        self.prompt = PromptTemplate.from_template(prompt_template)
        
        # Build the RAG chain
        self.chain = (
            {
                "context": self.retriever,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def format_docs(self, docs: List[Any]) -> str:
        """
        Format a list of documents into a string.
        
        Args:
            docs: List of documents
            
        Returns:
            Formatted string
        """
        return "\n\n".join(doc.page_content for doc in docs)
    
    def invoke(
        self,
        query: str,
        config: Optional[RunnableConfig] = None
    ) -> str:
        """
        Invoke the RAG chain with a query.
        
        Args:
            query: Query string
            config: Optional configuration for the runnable
            
        Returns:
            Response from the LLM
        """
        return self.chain.invoke(query, config=config)
    
    def stream(
        self,
        query: str,
        config: Optional[RunnableConfig] = None
    ):
        """
        Stream the response from the RAG chain.
        
        Args:
            query: Query string
            config: Optional configuration for the runnable
            
        Returns:
            Generator yielding response chunks
        """
        return self.chain.stream(query, config=config)


class StandardRAGChain(RAGChain):
    """
    Standard RAG chain implementation.
    """
    
    def __init__(
        self,
        llm,
        retriever,
        prompt_template: Optional[str] = None
    ):
        """
        Initialize the StandardRAGChain.
        
        Args:
            llm: Language model to use
            retriever: Retriever to use
            prompt_template: Optional custom prompt template
        """
        super().__init__(llm, retriever, prompt_template)
        
        # Modify the chain to format documents
        self.chain = (
            {
                "context": retriever | (lambda docs: self.format_docs(docs)),
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )


class ContextualHeaderRAGChain(RAGChain):
    """
    Contextual header RAG chain implementation.
    """
    
    def __init__(
        self,
        llm,
        retriever,
        prompt_template: Optional[str] = None
    ):
        """
        Initialize the ContextualHeaderRAGChain.
        
        Args:
            llm: Language model to use
            retriever: Retriever to use
            prompt_template: Optional custom prompt template
        """
        super().__init__(llm, retriever, prompt_template)
        
        # Modify the chain to format documents
        self.chain = (
            {
                "context": retriever | (lambda docs: self.format_docs(docs)),
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
