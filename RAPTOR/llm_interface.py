from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import logging
import os
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class LLMInterface:
    def __init__(self):
        self.llm = ChatOpenAI(
                model_name = "mistralai/mistral-7b-instruct",
                openai_api_key = os.getenv("OPENROUTER_API_KEY"),
                openai_api_base = "https://openrouter.ai/api/v1",
                temperature = 0,
                streaming = True
            )

        logger.info("Initialized LLM interface with model: mistralai-7b instruct")

        self.summarization_prompt = ChatPromptTemplate.from_template(prompt_template = """You are a professional summarization assistant. Create a comprehensive yet concise summary of the provided text that captures the key points, main arguments, and essential details.

            Instructions:
            - Identify and highlight the central themes and main ideas
            - Preserve important facts, figures, and specific details
            - Maintain the original tone and context
            - Structure the summary logically with clear organization
            - Aim for approximately 20-30% of the original length while retaining all critical information

            Text to summarize:
            {text}

            Summary:""")

        self.summarization_chain = ({"text": RunnablePassthrough()}
            | self.summarization_prompt
            | self.llm
            | StrOutputParser())

        




if __name__ == "__main__":
    llm_interface = LLMInterface()
    print(llm_interface.llm.invoke("Hello, how are you?"))