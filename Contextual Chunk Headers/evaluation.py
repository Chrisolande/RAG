import time
from typing import List, Dict, Any, Callable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from tqdm.auto import tqdm

class RAGEvaluator:
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings

        self.results = {
            "standard": {},
            "contextual": {}
        }

    def evaluate_retrieval_accuracy(self, query: str, retrieved_docs: List[Document], relevant_docs: List[Document]) -> float:
        # Get the document IDs for comparison
        retrieved_ids = [doc.metadata.get("id", doc.metadata.get("source", "")) for doc in retrieved_docs]
        relevant_ids = [doc.metadata.get("id", doc.metadata.get("source", "")) for doc in relevant_docs]

        # Calculate precision (proportion of retrieved documents that are relevant)
        if not retrieved_ids:
            return 0.0

        relevant_retrieved = [doc_id for doc_id in retrieved_ids if doc_id in relevant_ids]
        precision = len(relevant_retrieved) / len(retrieved_ids)

        return precision

    def evaluate_semantic_relevance(
        self,
        query: str,
        retrieved_docs: List[Document]
    ) -> float:
        """
        Evaluate semantic relevance of retrieved documents to the query.
        
        Args:
            query: Query string
            retrieved_docs: List of Document objects retrieved by the system
        Returns:
            float: Average cosine similarity score (0-1)
        """
        # Calculate the average cosine similarity
        if not retrieved_docs:
            return 0.0

        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)

        # Get document embeddings
        doc_contents = [doc.page_content for doc in retrieved_docs]
        doc_embeddings = self.embeddings.embed_documents(doc_contents)

        # Calculate cosine similarities
        similarities = []
        query_norm = np.linalg.norm(query_embedding)
        for doc_embedding in doc_embeddings:
            # Compute cosine similarity
            dot_product = np.dot(query_embedding, doc_embedding)
            doc_norm = np.linalg.norm(doc_embedding)

            if query_norm == 0 or doc_norm == 0:
                similarity = 0
            else:
                similarity = dot_product / (query_norm * doc_norm)

            similarities.append(similarity)

        # Return average similarity
        return np.mean(similarities)

    def evaluate_response_quality(self, query: str, response: str, reference_answer: Optional[str] = None, llm_evaluator = None) -> float:
        if reference_answer and self.embeddings:
            # Compare response to query using semantic similarity
            response_embedding = self.embeddings.embed_query(response)
            reference_embedding = self.embeddings.embed_query(reference_answer)

            # Compute cosine similarity
            dot_product = np.dot(response_embedding, reference_embedding)
            response_norm = np.linalg.norm(response_embedding)
            reference_norm = np.linalg.norm(reference_embedding)
            
            if response_norm == 0 or reference_norm == 0:
                return 0
                
            similarity = dot_product / (response_norm * reference_norm)
            return similarity

        elif llm_evaluator:
            # Use LLM to evaluate response quality
            evaluation = llm_evaluator.evaluate_response(query, response)
            return evaluation

        else:
            # If no reference answer or LLM evaluator, return None
            return None


    def measure_query_time(self, query_func: Callable[[str], Any], query: str) -> float:
        start_time = time.time()
        query_func(query)
        end_time = time.time()
        return end_time - start_time
        
    def run_evaluation(self, queries: List[str], standard_rag, contextual_rag, relevant_docs: Optional[Dict[str, List[Document]]] = None, reference_answers: Optional[Dict[str, str]] = None, llm_evaluator = None):
        print(f"Running evaluation on {len(queries)} queries ...")
        for query in tqdm(queries, desc = "Evaluating queries"):
            # Evaluate the standard RAG
            standard_time = self.measure_query_time(standard_rag.invoke, query)
            standard_response = standard_rag.invoke(query)
            standard_docs = standard_rag.retriever.get_relevant_documents(query)

            # Evaluate the contextual RAG
            contextual_time = self.measure_query_time(contextual_rag.invoke, query)
            contextual_response = contextual_rag.invoke(query)
            contextual_docs = contextual_rag.retriever.get_relevant_documents(query)

            # Store results for this query
            self.results["standard"][query] = {
                "response": standard_response,
                "retrieved_docs": standard_docs,
                "query_time": standard_time,
                "semantic_relevance": self.evaluate_semantic_relevance(query, standard_docs)
            }
            
            self.results["contextual"][query] = {
                "response": contextual_response,
                "retrieved_docs": contextual_docs,
                "query_time": contextual_time,
                "semantic_relevance": self.evaluate_semantic_relevance(query, contextual_docs)
            }
            
            # Add retrieval accuracy if relevant docs are provided
            if relevant_docs and query in relevant_docs:
                self.results["standard"][query]["retrieval_accuracy"] = self.evaluate_retrieval_accuracy(
                    query, standard_docs, relevant_docs[query]
                )
                self.results["contextual"][query]["retrieval_accuracy"] = self.evaluate_retrieval_accuracy(
                    query, contextual_docs, relevant_docs[query]
                )
            
            # Add response quality if reference answers or LLM evaluator are provided
            if reference_answers and query in reference_answers:
                self.results["standard"][query]["response_quality"] = self.evaluate_response_quality(
                    query, standard_response, reference_answers[query]
                )
                self.results["contextual"][query]["response_quality"] = self.evaluate_response_quality(
                    query, contextual_response, reference_answers[query]
                )
            elif llm_evaluator:
                self.results["standard"][query]["response_quality"] = self.evaluate_response_quality(
                    query, standard_response, llm_evaluator=llm_evaluator
                )
                self.results["contextual"][query]["response_quality"] = self.evaluate_response_quality(
                    query, contextual_response, llm_evaluator=llm_evaluator
                )


    def get_summary_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary metrics for both RAG systems.
        
        Returns:
            Dictionary of summary metrics
        """
        summary = {
            "standard": {},
            "contextual": {}
        }
        
        for system in ["standard", "contextual"]:
            # Calculate average query time
            query_times = [result["query_time"] for result in self.results[system].values()]
            summary[system]["avg_query_time"] = np.mean(query_times)
            
            # Calculate average semantic relevance
            semantic_relevances = [result["semantic_relevance"] for result in self.results[system].values()]
            summary[system]["avg_semantic_relevance"] = np.mean(semantic_relevances)
            
            # Calculate average retrieval accuracy if available
            retrieval_accuracies = [
                result.get("retrieval_accuracy") 
                for result in self.results[system].values() 
                if "retrieval_accuracy" in result
            ]
            if retrieval_accuracies:
                summary[system]["avg_retrieval_accuracy"] = np.mean(retrieval_accuracies)
            
            # Calculate average response quality if available
            response_qualities = [
                result.get("response_quality") 
                for result in self.results[system].values() 
                if "response_quality" in result
            ]
            if response_qualities:
                summary[system]["avg_response_quality"] = np.mean(response_qualities)
        
        return summary
    
    def visualize_results(self, output_path: Optional[str] = None):
        """
        Visualize the evaluation results.
        
        Args:
            output_path: Optional path to save the visualization
        """
        summary = self.get_summary_metrics()
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('RAG System Comparison: Standard vs. Contextual Header', fontsize=16)
        
        # Plot query time
        axs[0, 0].bar(
            ['Standard RAG', 'Contextual Header RAG'],
            [summary['standard']['avg_query_time'], summary['contextual']['avg_query_time']]
        )
        axs[0, 0].set_title('Average Query Processing Time (s)')
        axs[0, 0].set_ylabel('Time (seconds)')
        
        # Plot semantic relevance
        axs[0, 1].bar(
            ['Standard RAG', 'Contextual Header RAG'],
            [summary['standard']['avg_semantic_relevance'], summary['contextual']['avg_semantic_relevance']]
        )
        axs[0, 1].set_title('Average Semantic Relevance')
        axs[0, 1].set_ylabel('Cosine Similarity')
        
        # Plot retrieval accuracy if available
        if 'avg_retrieval_accuracy' in summary['standard'] and 'avg_retrieval_accuracy' in summary['contextual']:
            axs[1, 0].bar(
                ['Standard RAG', 'Contextual Header RAG'],
                [summary['standard']['avg_retrieval_accuracy'], summary['contextual']['avg_retrieval_accuracy']]
            )
            axs[1, 0].set_title('Average Retrieval Accuracy')
            axs[1, 0].set_ylabel('Precision')
        else:
            axs[1, 0].text(0.5, 0.5, 'Retrieval Accuracy Not Available', 
                         horizontalalignment='center', verticalalignment='center')
            axs[1, 0].set_title('Average Retrieval Accuracy')
        
        # Plot response quality if available
        if 'avg_response_quality' in summary['standard'] and 'avg_response_quality' in summary['contextual']:
            axs[1, 1].bar(
                ['Standard RAG', 'Contextual Header RAG'],
                [summary['standard']['avg_response_quality'], summary['contextual']['avg_response_quality']]
            )
            axs[1, 1].set_title('Average Response Quality')
            axs[1, 1].set_ylabel('Quality Score')
        else:
            axs[1, 1].text(0.5, 0.5, 'Response Quality Not Available', 
                         horizontalalignment='center', verticalalignment='center')
            axs[1, 1].set_title('Average Response Quality')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save or show the figure
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()

        