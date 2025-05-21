import os
from typing import Dict, Any, List, Optional

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
    """_summary_

    Args:
        model_name (str, optional): Name of the model to use, defaults to "meta-llama/llama-3.2-1b-instruct".
        temperature (float, optional): Temperature parameter for the model, defaults to 0.1.

    Returns:
        ChatOpenAI: OpenRouter instance configured to use the specified model.
    """
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        **kwargs
    )

class RAGChain:
    # Base Class for the RAG Chain
    def __init__(
        self, 
        llm,
        retriever,
        prompt
    ):
        
        self.llm = llm
        self.retriever = retriever
        
        if prompt_template is None:
            prompt_template = """You are a helpful assistant that provides accurate information based on the context provided.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer the question based on the context provided. If the answer cannot be found in the context, say "I don't have enough information to answer this question." and suggest what information might help.
            """

        self.prompt = PromptTemplate.from_template(prompt_template)
        # Build the RAG Chain
        self.chain = (
            {
                "context": self.retriever,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def format_docs(self, docs: List[Dict[str, Any]]) -> str:
        """Format a list of documents to strings"""
        return "\n\n".join(doc["page_content"] for doc in docs)

    def invoke(
        self,
        query: str, 
        config: Optional[RunnableConfig] = None,
        **kwargs
    ):
        """Invoke the RAG Chain"""
        return self.chain.invoke(query, config=config, **kwargs)

    def stream(
        self, 
        query: str,
        config: Optional[RunnableConfig] = None,
        **kwargs
    ):
        """ 
        Stream the output of the chain

        Args:
            query (str): The users question
            config (Optional[RunnableConfig]): Optional configuration for the runnable. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the chain.
        """
        return self.chain.stream(query, config=config, **kwargs)

class StandardRAGChain(RAGChain):
    def __init__(
        self,
        llm,
        retriever,
        prompt_template: Optional[str] = None
    ):
        super().__init__(llm, retriever, prompt_template)
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
    def __init__(
        self,
        llm,
        retriever,
        prompt_template: Optional[str] = None
    ):
        super().__init__(llm, retriever, prompt_template)
        self.chain = (
            {
                "context": retriever | (lambda docs: self.format_docs(docs)),
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )