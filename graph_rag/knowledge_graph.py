from langchain_neo4j import Neo4jGraph
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, MODEL_NAME
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any, Optional
import os
from langchain_core.documents import Document
from tqdm.notebook import tqdm
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer

class KnowledgeGraph:
    def __init__(self, batch_size: int = 5,url: Optional[str] = None,username: Optional[str] = None,password: Optional[str] = None,model_name: Optional[str] = None):
        self.batch_size = batch_size
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

        for doc in tqdm(documents, desc="Processing entities and relationships"):
            result = extraction_chain.invoke({"text": doc.page_content})
            
            # Parse the result correctly
            lines = result.content.strip().split('\n')
            
            # Track entities and relationships separately
            entities = set()
            relationships = []
            
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
                    # Format: - Apple Inc. (Organization)
                    entity_line = line[2:].strip()  # Remove "- "
                    if "(" in entity_line and entity_line.endswith(")"):
                        entity_name = entity_line.split("(")[0].strip()
                        entity_type = entity_line.split("(")[1].replace(")", "").strip()
                        entities.add((entity_name, entity_type))
                    else:
                        # Just entity name without type
                        entities.add((entity_line, "Entity"))
                
                # Parse relationships
                elif current_section == "relationships" and line.startswith("- ") and "→" in line:
                    # Format: - Tim Cook → CEO_of → Apple Inc.
                    rel_line = line[2:].strip()  # Remove "- "
                    parts = [part.strip() for part in rel_line.split("→")]
                    
                    if len(parts) == 3:
                        entity1, relation, entity2 = parts
                        relationships.append((entity1, relation, entity2))
            
            # Create entities in Neo4j
            for entity_name, entity_type in entities:
                # Create entity with both generic and specific labels
                create_entity_query = f"""
                MERGE (e:__Entity__ {{id: $entity_name}})
                SET e.type = $entity_type
                SET e:{entity_type.replace(' ', '_').replace('/', '_')}
                """
                try:
                    self.graph.query(create_entity_query, {
                        "entity_name": entity_name,
                        "entity_type": entity_type
                    })
                except Exception as e:
                    # Fallback to basic entity creation
                    basic_query = "MERGE (e:__Entity__ {id: $entity_name}) SET e.type = $entity_type"
                    self.graph.query(basic_query, {
                        "entity_name": entity_name,
                        "entity_type": entity_type
                    })
            
            # Create relationships in Neo4j
            for entity1, relation, entity2 in relationships:
                # Clean relationship name for Cypher
                clean_relation = relation.upper().replace(' ', '_').replace('-', '_')
                
                # Ensure both entities exist
                self.graph.query("MERGE (e:__Entity__ {id: $entity})", {"entity": entity1})
                self.graph.query("MERGE (e:__Entity__ {id: $entity})", {"entity": entity2})
                
                # Create relationship
                create_rel_query = f"""
                MATCH (e1:__Entity__ {{id: $entity1}})
                MATCH (e2:__Entity__ {{id: $entity2}})
                MERGE (e1)-[r:{clean_relation}]->(e2)
                """
                
                try:
                    self.graph.query(create_rel_query, {
                        "entity1": entity1,
                        "entity2": entity2
                    })
                except Exception as e:
                    print(f"Failed to create relationship {entity1} -[{clean_relation}]-> {entity2}: {e}")

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
    
