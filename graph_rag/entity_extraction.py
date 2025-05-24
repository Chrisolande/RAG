from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional
from config import OPENROUTER_API_KEY, MODEL_NAME
from langchain_neo4j import Neo4jGraph
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars


class Entities(BaseModel):
    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that appear in the text",
    )

class GraphEntityRetriever:
    def __init__(
        self,
        model_name: Optional[str] = None,
        entity_index_name: str = "entity",
        max_entities_per_query: int = 10,
        max_results_per_entity: int = 50,
        similarity_threshold: int = 2
    ):
        self.graph_driver = Neo4jGraph()
        self.llm = ChatOpenAI(
            openai_api_key=OPENROUTER_API_KEY,
            model_name=model_name or MODEL_NAME,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0,
            streaming = True
        )
        self.entity_index_name = entity_index_name
        self.max_entities_per_query = max_entities_per_query
        self.max_results_per_entity = max_results_per_entity
        self.similarity_threshold = similarity_threshold
        self._setup_entity_extraction_chain()

    def _setup_entity_extraction_chain(self):
        """Set up the entity extraction prompt and chain."""
        self.entity_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are extracting organization and person entities from the text. "
                "Focus on identifying proper nouns that represent people, organizations, "
                "companies, institutions, or other named entities that could be found "
                "in a knowledge graph."
            ),
            (
                "human",
                "Use the given format to extract information from the following input: {question}"
            ),
        ])
        self.entity_chain = self.entity_prompt | self.llm.with_structured_output(Entities)

    def extract_entities(self, question: str) -> List[str]:
        try:
            # Extract the entities from the question
            print(f"Extracting entities for question: {question}")
            entities_result = self.entity_chain.invoke({"question": question})
            entities = entities_result.names[:self.max_entities_per_query]
            
            print(f"Extracted {len(entities)} entities: {entities}")
            return entities
            
        except Exception as e:
            print(f"Failed to extract entities: {str(e)}")
            return []

    def generate_full_text_query(self, input_text: str) -> str:
        if not input_text or not input_text.strip():
            return ""

        clean_text = remove_lucene_chars(input_text)

        # Split into words and filter out empty strings
        words = [word for word in clean_text.split() if word.strip()]
        if not words:
            return ""

        # Build full-text query with similarity threshold
        full_text_query = ""
        
        # Add all words except the last with AND
        for word in words[:-1]:
            full_text_query += f" {word}~{self.similarity_threshold} AND"
        
        # Add the last word without AND
        full_text_query += f" {words[-1]}~{self.similarity_threshold}"
        
        return full_text_query.strip()  
        
    def retrieve_entity_neighborhood(self, entity: str) -> List[str]:
        """Retrieve the entity of a single entity"""
        try:
            """ Generate full text query """
            search_query = self.generate_full_text_query(entity)
            if not search_query:
                print("No search query generated")
                return []

            # Fixed Cypher query with proper variable scope clause
            cypher_query = """
            CALL db.index.fulltext.queryNodes($index_name, $query, {limit: 2})
            YIELD node, score
            CALL (node) {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            }
            RETURN output LIMIT $limit
            """

            response = self.graph_driver.query(
                cypher_query,
                {
                    "index_name": self.entity_index_name,
                    "query": search_query,
                    "limit": self.max_results_per_entity
                }
            )
            # Extract the output strings
            results = [record['output'] for record in response if 'output' in record]
            print(f"Retrieved {len(results)} relationships for entity '{entity}'")
            return results
        except Exception as e:
            print(f"Failed to retrieve entity neighborhood: {str(e)}")
            return []

    def structured_retriever(self, question: str) -> str:
        """
        Collect the neighborhood of entities mentioned in the question.
        
        This is the main retrieval method that:
        1. Extracts entities from the question
        2. Retrieves graph neighborhoods for each entity
        3. Combines all results into a single string
        """
        print(f"Processing structured retrieval for question: {question}")

        entities = self.extract_entities(question)
        if not entities:
            print("No entities found")
            return ""
        
        all_neighborhoods = []
        
        for entity in entities:
            try:
                neighborhood = self.retrieve_entity_neighborhood(entity)
                if neighborhood:
                    print(f"Found {len(neighborhood)} relationships for entity '{entity}'")
                    for rel in neighborhood[:5]:
                        print(f"  - {rel}")
                    if len(neighborhood) > 5:
                        print(f"  ... and {len(neighborhood) - 5} more")
                    all_neighborhoods.extend(neighborhood)
                else:
                    print("No relationships found")
            except Exception as e:
                print(f"Error: {str(e)}")
        
        # Return combined neighborhoods as a single string
        return "\n".join(all_neighborhoods) if all_neighborhoods else ""

    def check_database_status(self):
        """Helper method to check if the database has data and indexes"""
        try:
            # Check if entity index exists
            index_check = self.graph_driver.query(
                "SHOW INDEXES YIELD name WHERE name = $index_name",
                {"index_name": self.entity_index_name}
            )
            
            if not index_check:
                print(f"Warning: Index '{self.entity_index_name}' not found")
                print("Available indexes:")
                all_indexes = self.graph_driver.query("SHOW INDEXES YIELD name")
                for idx in all_indexes:
                    print(f"  - {idx['name']}")
            else:
                print(f"Index '{self.entity_index_name}' exists")
            
            # Check node count
            node_count = self.graph_driver.query("MATCH (n) RETURN count(n) as count")[0]['count']
            print(f"Total nodes in database: {node_count}")
            
            # Check relationship count
            rel_count = self.graph_driver.query("MATCH ()-[r]->() RETURN count(r) as count")[0]['count']
            print(f"Total relationships in database: {rel_count}")
            
            # Sample some nodes to see what's available
            sample_nodes = self.graph_driver.query("MATCH (n) RETURN n.id as id LIMIT 5")
            print("Sample node IDs:")
            for node in sample_nodes:
                print(f"  - {node['id']}")
                
        except Exception as e:
            print(f"Error checking database status: {str(e)}")

    def retrive_with_metadata(self, question: str) -> str:
        print(f"Processing enhanced retrieval for question: {question}")
        # Extract entities
        entities = self.extract_entities(question)

        # Collect detailed results for each query
        entity_results = {}
        total_relationships = 0

        for entity in entities:
            entity_neighborhood = self.retrieve_entity_neighborhood(entity)
            entity_results[entity] = entity_neighborhood
            total_relationships += len(entity_neighborhood)

        # Combine all results
        combined_result = "\n".join([
            relationship
            for neighborhoods in entity_results.values()
            for relationship in neighborhoods
        ])
        
        return {
            'question': question,
            'entities_found': entities,
            'entity_count': len(entities),
            'results_per_entity': {
                entity: len(results) for entity, results in entity_results.items()
            },
            'total_relationships': total_relationships,
            'combined_result': combined_result,
            'entity_details': entity_results
        }


if __name__ == "__main__":
    # Set up Neo4j connection 
    graph_driver = Neo4jGraph()

    # Instantiate the retriever
    retriever = GraphEntityRetriever(
        graph_driver=graph_driver,
        model_name=MODEL_NAME
    )

    print(retriever)
    # Check database status first
    print("=== Database Status Check ===")
    retriever.check_database_status()
    print()

    # Input text
    text_input = "Chris Olande is a student in Kenyatta University"

    # Extract entities
    entities = retriever.extract_entities(text_input)
    print("Extracted Entities:", entities)

    entity_query = retriever.generate_full_text_query(text_input)
    print("Entity Query:", entity_query)

    neighborhoods = retriever.structured_retriever(text_input)
    print("Neighborhoods:", neighborhoods if neighborhoods else "No neighborhoods found")

    metadata = retriever.retrive_with_metadata(text_input)
    print("Metadata:", metadata)