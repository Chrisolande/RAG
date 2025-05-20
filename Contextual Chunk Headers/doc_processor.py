import os
from typing import Dict, List, Any, Optional

from langchain_core.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from tqdm.auto import tqdm

class DocumentProcessor:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200, docs_dir: str = "books"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.docs_dir = docs_dir

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap
        )

    def load_documents(self) -> List[Document]:
        if not os.path.exists(self.docs_dir):
            raise FileNotFoundError(f"The directory {self.docs_dir} does not exist.")

        loader = DirectoryLoader(self.docs_dir, loader_cls = TextLoader,
        glob = "**/*.txt")
        return loader.load()

    def split_documents(self, documents: List[Document]) -> List[Document]:
        print(f"Splitting {len(documents)}")
        return self.text_splitter.split_documents(documents)

    def process_documents(self) -> List[Document]:
        """Load and split the documents"""
        documents = self.load_documents()
        return self.split_documents(documents)


class ContextualHeaderProcessor(DocumentProcessor):
    """
    Extends DocumentProcessor to add contextual headers to document chunks.
    """
    def __init__(self, llm, chunk_size: int = 2000, chunk_overlap: int = 200, docs_dir: str = "books"):
        super().__init__(chunk_size, chunk_overlap, docs_dir)
        self.llm = llm

        # Define the header prompt
        self.header_prompt = PromptTemplate.from_template(
            f"""
            You are an expert at summarizing text.
            Given the following text, create a concise contextual header (1-2 sentences) 
            that captures the main topics and context of this text segment.
            The header should help in retrieving this text when relevant questions are asked.
            
            TEXT:
            {text}
            
            CONTEXTUAL HEADER:
            """
        )
        # Create chain for generating headers
        self.header_chain = (
            {"text": RunnablePassthrough()} 
            | self.header_prompt 
            | self.llm 
            | StrOutputParser()
        )

    def add_contextual_headers(self, 
                                documents: List[Document],
                                config: Optional[RunnableConfig] = None) -> List[Document]:
        """
        Add contextual headers to document chunks.
        
        Args:
            documents: List of document chunks
            config: Optional configuration for the runnable
            
        Returns:
            List of documents with contextual headers
        """

        enhanced_docs = []

        print(f"Generating contextual headers for {len(documents)} document chunks...")
        for doc in tqdm(documents, desc = "Generating headers"):
            header = self.header_chain.invoke(doc.page_content, config = config)

            # Create new document with header prepended to content
            enhanced_content = f"CONTEXT: {header}\n\n{doc.page_content}"

            # Create new document with enhanced content and original metadata
            enhanced_doc = Document(
                page_content=enhanced_content,
                metadata={
                    **doc.metadata,
                    "header": header
                }
            )

            enhanced_docs.append(enhanced_doc)

        return enhanced_docs
        