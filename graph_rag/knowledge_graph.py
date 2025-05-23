from langchain_community.graphs import Neo4jGraph
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, MODEL_NAME
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any, Optional
import os
from langchain_core.documents import Document

class KnowledgeGraph:
    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """Initialize the knowledge graph"""
        self.uri = uri or NEO4J_URI
        self.username = username or NEO4J_USERNAME
        self.password = password or NEO4J_PASSWORD

        self.graph = Neo4jGraph(
            uri = self.uri,
            username = self.username,
            password = self.password
        )

    def clear_database(self):
        """Clear all the data in the Neo4j database. """
        self.graph.query("MATCH (n) DETACH DELETE n")"

    def create_graph_from_documents(
        self,
        documents: List[Document],
        llm_model: str = MODEL_NAME
    ):
        # Create the LLM for extraction
        llm = ChatOpenAI(
            model = llm_model,
            temperature = 0, 
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1"
        )

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

        extraction_chain = extraction_prompt | llm

        for doc in documents:
            result = extraction_chain.invoke({"text": doc.page_content})

            # Parse the result and create cypher queries
            lines = result.content.strip().split('\n')
            for line in lines:
                if '[' in line and ']' in line:
                    triplet = line.strip('[]').split(',')
                    if len(triplet) == 3:
                        entity1 = triplet[0].strip()
                        relation = triplet[1].strip().upper().replace(' ', '_')
                        entity2 = triplet[2].strip()

                        query = f"""
                        MERGE (e1:__Entity__ {{id: $entity1}})
                        MERGE (e2:__Entity__ {{id: $entity2}})
                        MERGE (e1)-[r:{relation}]->(e2)
                        """

                        self.graph.query(query, {"entity1": entity1, "entity2": entity2})

    def visualize_graph(self, cypher_query: str = "MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t LIMIT 50"):
        """Visualize the graph using cypher query"""
        try:
            from yfiles_jupyter_graphs import GraphWidget
            session = self.graph._driver.session()

            # Create widget
            widget = GraphWidget(graph = session.run(cypher_query).graph())
            widget.node_label_mapping = 'id'

            return widget
        except ImportError:
            print("yfiles_jupyter_graphs not available. Install it to visualize graphs")
            return None

    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query on the graph."""
        return self.graph.query(query, params)