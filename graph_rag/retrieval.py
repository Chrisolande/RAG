from knowledge_graph import KnowledgeGraph
from vector_store import VectorStore
from entity_extraction import EntityExtractor
from typing import List
from langchain_core.documents import Document

class Retriever: 
    """ Handles retrieval operations for the graph rag system"""
    def __init__(
        self, 
        knowledge_graph: KnowledgeGraph,
        vector_store: VectorStore,
        entity_extractor: EntityExtractor
    ):

        """Initialize the retriever"""
        self.knowledge_graph = knowledge_graph
        self.vector_store = vector_store
        self.entity_extractor = entity_extractor

    def structured_retrieval(self, question: str) -> str:
        """ Retrieve information from the knowledge graph based on entities in the question"""
        
        # Extract entities from the question
        entities = self.entity_Extractor.extract_entities(question)

        # Initialize result
        result = ""
        for entity in entities:
            # Generate full-text query
            query = self.entity_extractor.generate_full_text_query(entity)
            # Execute Cypher query to find entity and its neighborhood
            response = self.knowledge_graph.query(
                """
                CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node, score
                CALL {
                  WITH node
                  MATCH (node)-[r]->(neighbor)
                  RETURN node.id + " - " + type(r) + " -> " + neighbor.id AS output
                  UNION
                  MATCH (neighbor)-[r]->(node)
                  RETURN neighbor.id + " - " + type(r) + " -> " + node.id AS output
                }
                RETURN output
                """,
                {"query": query}
            )

            # Append results
            result += "\n".join([el['output'] for el in response])
        
        return result