"""Semantic graph utilities for Part 2 of the workshop."""

from __future__ import annotations

import asyncio
import html
import json
from pathlib import Path
from typing import Any

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from neo4j import Driver
from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.generation.prompts import RagTemplate
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import Text2CypherRetriever

SEMANTIC_GRAPH_SCHEMA = """
Node properties:
- Venue {name: STRING}
- Service {name: STRING}
- Activity {name: STRING}
- Requirement {name: STRING}
- Itinerary {name: STRING}
- Chunk {index: INTEGER, text: STRING}

Relationships:
- (:Venue)-[:OFFERS]->(:Service)
- (:Venue)-[:SUPPORTS]->(:Activity)
- (:Venue)-[:SATISFIES]->(:Requirement)
- (:Service)-[:SATISFIES]->(:Requirement)
- (:Itinerary)-[:INCLUDES]->(:Activity)
- (:Itinerary)-[:INCLUDES]->(:Service)
- (:Venue)-[:FROM_CHUNK]->(:Chunk)
- (:Service)-[:FROM_CHUNK]->(:Chunk)
- (:Activity)-[:FROM_CHUNK]->(:Chunk)
- (:Requirement)-[:FROM_CHUNK]->(:Chunk)
- (:Itinerary)-[:FROM_CHUNK]->(:Chunk)
- (:Chunk)-[:NEXT_CHUNK]->(:Chunk)
"""

SEMANTIC_CYPHER_EXAMPLES = [
    "Question: What services does each venue offer?\n"
    "Cypher: MATCH (v:Venue)-[:OFFERS]->(s:Service) "
    "RETURN v.name AS venue, collect(DISTINCT s.name) AS services",
    "Question: Which venues support activities?\n"
    "Cypher: MATCH (v:Venue)-[:SUPPORTS]->(a:Activity) "
    "RETURN v.name AS venue, collect(DISTINCT a.name) AS activities",
    "Question: What source text supports the information about each venue?\n"
    "Cypher: MATCH (v:Venue)-[:FROM_CHUNK]->(c:Chunk) "
    "RETURN v.name AS venue, c.index AS chunk_index, c.text AS source_text",
    "Question: What does the source say about the services offered by each venue?\n"
    "Cypher: MATCH (v:Venue)-[:OFFERS]->(s:Service), "
    "(v)-[:FROM_CHUNK]->(c:Chunk) "
    "RETURN v.name AS venue, collect(DISTINCT s.name) AS services, "
    "collect(DISTINCT c.text) AS source_texts",
]

HYBRID_TEXT2CYPHER_PROMPT = """
Generate one read-only Cypher query for the question using only the supplied
schema. When the question concerns a semantic entity, traverse FROM_CHUNK and
return the supporting Chunk.text together with the requested semantic facts.
Use DISTINCT to avoid duplicate records. Do not modify the database and do not
include explanations.

Schema:
{schema}

Examples:
{examples}

Question:
{query_text}

Cypher:
"""

SEMANTIC_NODE_TYPES = ["Venue", "Service", "Activity", "Requirement", "Itinerary"]
SEMANTIC_RELATIONSHIP_TYPES = ["OFFERS", "SUPPORTS", "SATISFIES", "INCLUDES"]
KNOWLEDGE_GRAPH_NODE_TYPES = [*SEMANTIC_NODE_TYPES, "Chunk"]
MANAGED_GRAPH_NODE_TYPES = [*KNOWLEDGE_GRAPH_NODE_TYPES, "Document"]
KNOWLEDGE_GRAPH_RELATIONSHIP_TYPES = [
    *SEMANTIC_RELATIONSHIP_TYPES,
    "FROM_CHUNK",
    "NEXT_CHUNK",
]


class ChromaDefaultEmbedder(Embedder):
    """Adapt Chroma's local default embedding model to Neo4j GraphRAG."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding_function = DefaultEmbeddingFunction()

    def embed_query(self, text: str) -> list[float]:
        """Create one local embedding for a text value."""
        embedding = self.embedding_function([text])[0]
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)


def create_simple_kg_pipeline(
    driver: Driver,
    model_name: str,
    base_url: str,
    api_key: str,
    schema: dict[str, Any],
    neo4j_database: str = "neo4j",
) -> SimpleKGPipeline:
    """Configure Neo4j's official semantic knowledge-graph pipeline."""
    llm = OpenAILLM(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        model_params={
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        },
    )
    return SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=ChromaDefaultEmbedder(),
        schema=schema,
        from_file=False,
        on_error="IGNORE",
        perform_entity_resolution=True,
        neo4j_database=neo4j_database,
    )


def build_semantic_kg(pipeline: SimpleKGPipeline, text: str):
    """Run the asynchronous SimpleKGPipeline from a normal Python script."""
    return asyncio.run(pipeline.run_async(text=text))


def clear_generated_knowledge_graph(
    driver: Driver,
    neo4j_database: str = "neo4j",
) -> None:
    """Remove the graph produced by an earlier SimpleKGPipeline run."""
    with driver.session(database=neo4j_database) as session:
        session.run("MATCH (node:__KGBuilder__) DETACH DELETE node")


def remove_document_nodes(
    driver: Driver,
    neo4j_database: str = "neo4j",
) -> None:
    """Remove pipeline metadata documents while preserving chunks and entities."""
    with driver.session(database=neo4j_database) as session:
        session.run(
            """
            MATCH (document:Document)
            WHERE document:__KGBuilder__
               OR document.graph_rag_workshop_graph = 'semantic-fallback'
            DETACH DELETE document
            """
        )


def ensure_semantic_graph(
    driver: Driver,
    fallback_path: Path,
    neo4j_database: str = "neo4j",
) -> bool:
    """Restore the saved knowledge graph when Neo4j has no complete KG.

    Returns ``True`` when the fallback was loaded and ``False`` when an existing
    generated graph was already available.
    """
    with driver.session(database=neo4j_database) as session:
        evidence_path_count = session.run(
            """
            MATCH (entity)-[:FROM_CHUNK]->(chunk:Chunk)
            WHERE any(label IN labels(entity) WHERE label IN $entity_types)
              AND chunk.text IS NOT NULL
            RETURN count(*) AS count
            """,
            entity_types=SEMANTIC_NODE_TYPES,
        ).single()["count"]

        if evidence_path_count:
            return False

        if not fallback_path.is_file():
            raise FileNotFoundError(f"Semantic graph fallback not found: {fallback_path}")

        graph = json.loads(fallback_path.read_text(encoding="utf-8"))
        session.run(
            """
            MATCH (node)
            WHERE (node:__KGBuilder__ OR
                   node.graph_rag_workshop_graph = 'semantic-fallback')
              AND any(label IN labels(node) WHERE label IN $node_types)
            DETACH DELETE node
            """,
            node_types=MANAGED_GRAPH_NODE_TYPES,
        )

        for node in graph["nodes"]:
            node_labels = node["labels"]
            if not node_labels or any(
                label not in KNOWLEDGE_GRAPH_NODE_TYPES for label in node_labels
            ):
                raise ValueError(f"Unsupported fallback node labels: {node_labels}")
            labels = ":".join(node_labels)
            session.run(
                f"MERGE (node:{labels} {{fallback_id: $id}}) "
                "SET node += $properties, "
                "node.graph_rag_workshop_graph = 'semantic-fallback'",
                id=node["id"],
                properties=node["properties"],
            )

        for relationship in graph["relationships"]:
            relationship_type = relationship["type"]
            if relationship_type not in KNOWLEDGE_GRAPH_RELATIONSHIP_TYPES:
                raise ValueError(
                    f"Unsupported fallback relationship type: {relationship_type}"
                )
            session.run(
                "MATCH (source {fallback_id: $source_id}) "
                "MATCH (target {fallback_id: $target_id}) "
                f"MERGE (source)-[relationship:{relationship_type}]->(target) "
                "SET relationship += $properties",
                source_id=relationship["source"],
                target_id=relationship["target"],
                properties=relationship.get("properties", {}),
            )

    return True


def search_semantic_graph_rag(
    driver: Driver,
    question: str,
    model_name: str,
    base_url: str,
    api_key: str,
    neo4j_database: str = "neo4j",
) -> str:
    """Answer a question from SimpleKG entities using text-to-Cypher Graph-RAG."""
    llm = OpenAILLM(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        model_params={"temperature": 0.0},
    )
    retriever = Text2CypherRetriever(
        driver=driver,
        llm=llm,
        neo4j_schema=SEMANTIC_GRAPH_SCHEMA,
        examples=SEMANTIC_CYPHER_EXAMPLES,
        custom_prompt=HYBRID_TEXT2CYPHER_PROMPT,
        neo4j_database=neo4j_database,
    )
    graph_rag = GraphRAG(
        retriever=retriever,
        llm=llm,
        prompt_template=RagTemplate(
            system_instructions=(
                "Answer only from the Neo4j records supplied as context. Use both "
                "semantic relationships and the original Chunk.text evidence. "
                "Clearly distinguish venues, services, activities, requirements, "
                "and itineraries. Prefer answers supported by source chunk text. "
                "If the records do not contain the answer, say so."
            )
        ),
    )
    result = graph_rag.search(
        query_text=question,
        response_fallback="No matching facts were found in the knowledge graph.",
    )
    return result.answer


def save_semantic_graph_html(
    driver: Driver,
    schema: dict[str, Any],
    output_path: Path,
    neo4j_database: str = "neo4j",
) -> Path:
    """Export entities, chunks, documents, and their links as interactive HTML."""
    from pyvis.network import Network

    node_types = list(dict.fromkeys([*schema.get("node_types", []), "Chunk"]))
    relationship_types = list(
        dict.fromkeys(
            [
                *schema.get("relationship_types", []),
                "FROM_CHUNK",
                "NEXT_CHUNK",
            ]
        )
    )

    with driver.session(database=neo4j_database) as session:
        nodes = session.run(
            """
            MATCH (n)
            WHERE any(label IN labels(n) WHERE label IN $node_types)
            RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties
            """,
            node_types=node_types,
        ).data()
        relationships = session.run(
            """
            MATCH (source)-[relationship]->(target)
            WHERE type(relationship) IN $relationship_types
              AND any(label IN labels(source) WHERE label IN $node_types)
              AND any(label IN labels(target) WHERE label IN $node_types)
            RETURN elementId(source) AS source_id,
                   elementId(target) AS target_id,
                   type(relationship) AS relationship_type
            """,
            node_types=node_types,
            relationship_types=relationship_types,
        ).data()

    network = Network(
        height="760px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#111827",
        cdn_resources="remote",
    )
    palette = [
        "#2563eb",
        "#16a34a",
        "#f97316",
        "#9333ea",
        "#dc2626",
        "#0891b2",
        "#64748b",
    ]
    colors = {
        node_type: palette[index % len(palette)]
        for index, node_type in enumerate(node_types)
    }

    for node in nodes:
        semantic_labels = [label for label in node["labels"] if label in node_types]
        primary_label = semantic_labels[0] if semantic_labels else node["labels"][0]
        properties = node["properties"]
        if primary_label == "Chunk":
            label = f"Chunk {properties.get('index', '?')}"
        else:
            label = properties.get("name") or properties.get("id") or primary_label
        visible_properties = {
            key: value
            for key, value in properties.items()
            if "embedding" not in key.lower()
        }
        title_lines = [f"type: {primary_label}"] + [
            f"{key}: {json.dumps(value, ensure_ascii=False, default=str)}"
            for key, value in visible_properties.items()
        ]
        network.add_node(
            node["id"],
            label=str(label),
            title="<br>".join(html.escape(line) for line in title_lines),
            color=colors.get(primary_label, "#64748b"),
            shape="dot",
        )

    for relationship in relationships:
        network.add_edge(
            relationship["source_id"],
            relationship["target_id"],
            label=relationship["relationship_type"],
            title=relationship["relationship_type"],
            arrows="to",
        )

    network.set_options(
        """
        {
          "interaction": {"hover": true, "navigationButtons": true},
          "physics": {"stabilization": {"iterations": 200}},
          "edges": {"font": {"align": "middle"}, "smooth": {"type": "dynamic"}}
        }
        """
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    network.write_html(str(output_path), open_browser=False)
    return output_path
