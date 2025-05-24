from langchain_neo4j import Neo4jGraph
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, MODEL_NAME
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any, Optional, Set, Tuple
import os
from langchain_core.documents import Document
from tqdm.notebook import tqdm
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from collections import defaultdict

class KnowledgeGraph:
    def __init__(self, batch_size: int = 5, entity_batch_size: int = 100, rel_batch_size: int = 50, 
                 url: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None, 
                 model_name: Optional[str] = None):
        self.batch_size = batch_size
        self.entity_batch_size = entity_batch_size  # Batch size for entity creation
        self.rel_batch_size = rel_batch_size        # Batch size for relationship creation
        self.url = url or NEO4J_URI
        self.username = username or NEO4J_USERNAME
        self.password = password or NEO4J_PASSWORD
        self.model_name = model_name or MODEL_NAME
        self.llm = ChatOpenAI(
            model = self.model_name,
            temperature = 0, 
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            streaming = True
        )

        self.graph = Neo4jGraph()

    def clear_database(self):
        self.graph.query("MATCH (n) DETACH DELETE n")

    def _clean_relationship_name(self, relation: str) -> str:
        """Clean relationship name to be valid for Neo4j Cypher"""
        import re
        # Convert to uppercase
        clean = relation.upper()
        # Replace spaces and common separators with underscores
        clean = re.sub(r'[\s\-\.]+', '_', clean)
        # Remove all non-alphanumeric characters except underscores
        clean = re.sub(r'[^A-Z0-9_]', '', clean)
        # Remove multiple consecutive underscores
        clean = re.sub(r'_+', '_', clean)
        # Remove leading/trailing underscores
        clean = clean.strip('_')
        # Ensure it's not empty and doesn't start with a number
        if not clean or clean[0].isdigit():
            clean = 'RELATED_TO'
        return clean

    def _create_entities_batch(self, entities: Set[Tuple[str, str]]):
        """Create entities in batches for better performance."""
        entity_list = list(entities)
        
        for i in tqdm(range(0, len(entity_list), self.entity_batch_size), 
                     desc="Creating entities in batches", leave=False):
            batch = entity_list[i:i + self.entity_batch_size]
            
            # Prepare batch data
            batch_data = []
            for entity_name, entity_type in batch:
                batch_data.append({
                    "name": entity_name,
                    "type": entity_type,
                    "label": entity_type.replace(' ', '_').replace('/', '_')
                })
            
            # Batch create entities with UNWIND
            batch_query = """
            UNWIND $batch_data AS entity_data
            MERGE (e:__Entity__ {id: entity_data.name})
            SET e.type = entity_data.type
            """
            
            try:
                self.graph.query(batch_query, {"batch_data": batch_data})
            except Exception as e:
                print(f"Batch entity creation failed, falling back to individual creation: {e}")
                # Fallback to individual creation
                for entity_name, entity_type in batch:
                    try:
                        self.graph.query(
                            "MERGE (e:__Entity__ {id: $name}) SET e.type = $type",
                            {"name": entity_name, "type": entity_type}
                        )
                    except Exception as individual_e:
                        print(f"Failed to create entity {entity_name}: {individual_e}")

    def _create_relationships_batch(self, relationships: List[Tuple[str, str, str]]):
        """Create relationships in batches, grouped by relationship type."""
        # Group relationships by type
        relationships_by_type = defaultdict(list)
        for entity1, relation, entity2 in relationships:
            clean_relation = self._clean_relationship_name(relation)
            relationships_by_type[clean_relation].append((entity1, entity2))
        
        # Process each relationship type in batches
        for rel_type, entity_pairs in relationships_by_type.items():
            self._create_relationship_type_batch(rel_type, entity_pairs)

    def _create_relationship_type_batch(self, rel_type: str, entity_pairs: List[Tuple[str, str]]):
        """Create relationships of a specific type in batches."""
        for i in tqdm(range(0, len(entity_pairs), self.rel_batch_size), 
                     desc=f"Creating {rel_type} relationships", leave=False):
            batch = entity_pairs[i:i + self.rel_batch_size]
            
            # Prepare batch data
            batch_data = [{"entity1": e1, "entity2": e2} for e1, e2 in batch]
            
            # First ensure all entities exist in this batch
            all_entities = set()
            for e1, e2 in batch:
                all_entities.add(e1)
                all_entities.add(e2)
            
            # Batch create missing entities
            entity_batch_query = """
            UNWIND $entities AS entity_name
            MERGE (e:__Entity__ {id: entity_name})
            """
            try:
                self.graph.query(entity_batch_query, {"entities": list(all_entities)})
            except Exception as e:
                print(f"Failed to ensure entities exist: {e}")
            
            # Batch create relationships using dynamic Cypher
            rel_batch_query = f"""
            UNWIND $batch_data AS rel_data
            MATCH (e1:__Entity__ {{id: rel_data.entity1}})
            MATCH (e2:__Entity__ {{id: rel_data.entity2}})
            MERGE (e1)-[r:{rel_type}]->(e2)
            """
            
            try:
                self.graph.query(rel_batch_query, {"batch_data": batch_data})
            except Exception as e:
                print(f"Batch relationship creation failed for {rel_type}, falling back to individual creation: {e}")
                # Fallback to individual creation
                for entity1, entity2 in batch:
                    try:
                        individual_query = f"""
                        MATCH (e1:__Entity__ {{id: $entity1}})
                        MATCH (e2:__Entity__ {{id: $entity2}})
                        MERGE (e1)-[r:{rel_type}]->(e2)
                        """
                        self.graph.query(individual_query, {
                            "entity1": entity1,
                            "entity2": entity2
                        })
                    except Exception as individual_e:
                        print(f"Failed to create relationship {entity1} -[{rel_type}]-> {entity2}: {individual_e}")

    def create_graph_from_documents(self, documents: List[Document]):
        extraction_prompt = ChatPromptTemplate.from_template(
            """You are an expert information extraction system. Extract entities and relationships from the provided text with high precision and completeness.

            ## Entity Types to Extract:
            - **People**: Full names, titles, roles (e.g., "Dr. Sarah Johnson", "CEO Martinez")
            - **Organizations**: Companies, institutions, agencies, groups (e.g., "Microsoft", "Harvard University", "UN Security Council")
            - **Locations**: Countries, cities, regions, addresses, landmarks (e.g., "Tokyo", "Silicon Valley", "Building 42")
            - **Products/Services**: Brand names, software, publications, projects (e.g., "iPhone", "ChatGPT", "Nature journal")
            - **Events**: Meetings, conferences, incidents, programs (e.g., "COP28", "quarterly earnings call")
            - **Concepts**: Key topics, technologies, methodologies (e.g., "artificial intelligence", "renewable energy")

            ## Relationship Types:
            - **Organizational**: works_for, leads, owns, partners_with, competes_with, subsidiary_of
            - **Personal**: married_to, sibling_of, mentor_of, colleague_of, reports_to
            - **Locational**: based_in, headquarters_in, operates_in, born_in, located_at
            - **Temporal**: founded_in, occurred_on, started_in, ended_in, during
            - **Functional**: develops, uses, produces, sells, studies, regulates
            - **Ownership/Control**: owns, controls, manages, oversees, governs
            - **Participation**: attends, speaks_at, participates_in, member_of, part_of

            ## Instructions:
            1. Extract entities exactly as they appear in the text (preserve capitalization and full names)
            2. Focus on meaningful, factual relationships (avoid vague connections)
            3. Use the most specific relationship type available
            4. Include temporal context when mentioned (e.g., "former_CEO_of" vs "CEO_of")
            5. Prioritize relationships that add substantive information about the entities
            6. **IMPORTANT**: Relationship names must only contain letters, numbers, and underscores. NO parentheses, spaces, or special characters.

            ## Text to Analyze:
            {text}

            ## Output Format:
            Return your results as a structured list of triplets in this exact format:

            **Entities:**
            - [Entity Name] (Type)

            **Relationships:**
            - [Entity1] → [relationship_type] → [Entity2]
            - [Entity1] → [relationship_type] → [Entity2]

            ## Example:
            **Entities:**
            - Apple Inc. (Organization)
            - Tim Cook (Person)
            - iPhone (Product)
            - Cupertino (Location)

            **Relationships:**
            - Tim Cook → CEO_of → Apple Inc.
            - Apple Inc. → develops → iPhone
            - Apple Inc. → headquarters_in → Cupertino
            """
        )

        extraction_chain = extraction_prompt | self.llm

        # Collect all entities and relationships across all documents
        all_entities = set()
        all_relationships = []

        print("Extracting entities and relationships from documents...")
        for doc in tqdm(documents, desc="Processing documents"):
            result = extraction_chain.invoke({"text": doc.page_content})
            
            # Parse the result
            lines = result.content.strip().split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                    
                # Identify sections
                if line.startswith("**Entities:**"):
                    current_section = "entities"
                    continue
                elif line.startswith("**Relationships:**"):
                    current_section = "relationships"
                    continue
                
                # Parse entities
                if current_section == "entities" and line.startswith("- "):
                    entity_line = line[2:].strip()  # Remove "- "
                    if "(" in entity_line and entity_line.endswith(")"):
                        entity_name = entity_line.split("(")[0].strip()
                        entity_type = entity_line.split("(")[1].replace(")", "").strip()
                        all_entities.add((entity_name, entity_type))
                    else:
                        # Just entity name without type
                        all_entities.add((entity_line, "Entity"))
                
                # Parse relationships
                elif current_section == "relationships" and line.startswith("- ") and "→" in line:
                    rel_line = line[2:].strip()  # Remove "- "
                    parts = [part.strip() for part in rel_line.split("→")]
                    
                    if len(parts) == 3:
                        entity1, relation, entity2 = parts
                        all_relationships.append((entity1, relation, entity2))

        print(f"Extracted {len(all_entities)} unique entities and {len(all_relationships)} relationships")

        # Create entities in batches
        if all_entities:
            print("Creating entities...")
            self._create_entities_batch(all_entities)

        # Create relationships in batches
        if all_relationships:
            print("Creating relationships...")
            self._create_relationships_batch(all_relationships)

        print("Graph creation completed!")

    def create_graph(self, documents: List[Document]):
        llm_transformer = LLMGraphTransformer(llm = self.llm)

        # Convert the documents to graph format in batches
        for i in tqdm(range(0, len(documents), self.batch_size), desc = "Processing documents in batches"):
            batch = documents[i: i + self.batch_size]
            graph_documents = llm_transformer.convert_to_graph_documents(batch)
            self.graph.add_graph_documents(graph_documents,
                                    baseEntityLabel=True,
                                    include_source=True)

    def visualize_graph(self, cypher_query = "MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t LIMIT 50"):
        try:
            from neo4j import GraphDatabase
            from yfiles_jupyter_graphs import GraphWidget
            driver = GraphDatabase.driver(self.url, 
                                        auth=(self.username, self.password))
            session = driver.session()
            widget = GraphWidget(graph = session.run(cypher_query).graph())
            widget.node_label_mapping = 'id'
            #display(widget)
            return widget
        except ImportError as e:
            print(f"Error importing required packages: {e}")
            print("Make sure yfiles_jupyter_graphs is installed. Run: pip install yfiles_jupyter_graphs")

        except Exception as e:
            print(f"Error visualizing graph: {str(e)}")
            import traceback
            traceback.print_exc()

    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query on the graph."""
        return self.graph.query(query, params)

    def get_graph_stats(self) -> Dict[str, int]:
        """Get statistics about the current graph."""
        stats = {}
        
        # Count nodes
        node_count = self.graph.query("MATCH (n) RETURN count(n) as count")[0]['count']
        stats['nodes'] = node_count
        
        # Count relationships
        rel_count = self.graph.query("MATCH ()-[r]->() RETURN count(r) as count")[0]['count']
        stats['relationships'] = rel_count
        
        # Count relationship types
        rel_types = self.graph.query("""
            MATCH ()-[r]->() 
            RETURN type(r) as rel_type, count(r) as count 
            ORDER BY count DESC
        """)
        stats['relationship_types'] = {rt['rel_type']: rt['count'] for rt in rel_types}
        
        # Count entity types
        entity_types = self.graph.query("""
            MATCH (n:__Entity__) 
            WHERE n.type IS NOT NULL
            RETURN n.type as entity_type, count(n) as count 
            ORDER BY count DESC
        """)
        stats['entity_types'] = {et['entity_type']: et['count'] for et in entity_types}
        
        return stats