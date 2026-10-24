"""Build a semantic knowledge graph with Neo4j's official SimpleKGPipeline.

Unlike 01_graph_construction.py, this script creates entity nodes and semantic
relationships inferred by an LLM. It reuses the Markdown extracted in Part 01.
"""

import asyncio
import os
from pathlib import Path

from neo4j import GraphDatabase
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

from argo.utils.console_utils import (
    INFO_STYLE,
    console,
    print_result,
    print_step,
    run_with_progress,
)
from argo.utils.part_02_semantic_graph_utils import (
    ChromaDefaultEmbedder,
    clear_generated_knowledge_graph,
    remove_document_nodes,
    save_semantic_graph_html,
)
from argo.utils.pydantic_utils import get_llm_settings

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "data"
WEBPAGE_MARKDOWN_PATH = DATA_DIR / "webpage_rag" / "ARGO_Usecase.md"
SEMANTIC_GRAPH_HTML_PATH = DATA_DIR / "graph_solutions" / "simplekg_graph.html"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
LLM_MODEL, LLM_BASE_URL, LLM_API_KEY = get_llm_settings()

SEMANTIC_SCHEMA = {
    "node_types": [
        "Venue",
        "Service",
        "Activity",
        "Requirement",
        "Itinerary",
    ],
    "relationship_types": [
        "OFFERS",
        "SUPPORTS",
        "SATISFIES",
        "INCLUDES",
    ],
    "patterns": [
        ("Venue", "OFFERS", "Service"),
        ("Venue", "SUPPORTS", "Activity"),
        ("Venue", "SATISFIES", "Requirement"),
        ("Service", "SATISFIES", "Requirement"),
        ("Itinerary", "INCLUDES", "Activity"),
        ("Itinerary", "INCLUDES", "Service"),
    ],
}


#################################################################
# STEP 1 - Load the Markdown extracted in Part 01
#################################################################

print_step("STEP 1 - Load the webpage Markdown")

if not WEBPAGE_MARKDOWN_PATH.is_file():
    raise FileNotFoundError(
        f"Markdown not found: {WEBPAGE_MARKDOWN_PATH}. Run Part 01 webpage RAG first."
    )

webpage_markdown = WEBPAGE_MARKDOWN_PATH.read_text(encoding="utf-8")
console.print("Loaded the existing webpage Markdown.", style=INFO_STYLE)
print_result(
    f"Characters: {len(webpage_markdown)}\nSource file: {WEBPAGE_MARKDOWN_PATH}"
)


#################################################################
# STEP 2 - Connect to Neo4j
#################################################################

print_step("STEP 2 - Connect to Neo4j")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
)
driver.verify_connectivity()
console.print("Connected to Neo4j.", style=INFO_STYLE)
print_result(f"URI: {NEO4J_URI}\nDatabase: {NEO4J_DATABASE}")


#################################################################
# STEP 3 - Configure Neo4j SimpleKGPipeline
#################################################################

print_step("STEP 3 - Configure SimpleKGPipeline")

llm = OpenAILLM(
    model_name=LLM_MODEL,
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    model_params={
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    },
)

# SOLUTION - Configure the semantic knowledge-graph pipeline:
# Instantiate SimpleKGPipeline with the LLM, Neo4j driver, local embedder, and
# semantic schema. Entity resolution merges references to the same entity.
pipeline = SimpleKGPipeline(
    llm=llm,
    driver=driver,
    embedder=ChromaDefaultEmbedder(),
    schema=SEMANTIC_SCHEMA,
    from_file=False,
    on_error="IGNORE",
    perform_entity_resolution=True,
    neo4j_database=NEO4J_DATABASE,
)

console.print(
    "Configured semantic entity and relationship extraction.",
    style=INFO_STYLE,
)
print_result(
    f"Node types: {', '.join(SEMANTIC_SCHEMA['node_types'])}\n"
    f"Relationship types: {', '.join(SEMANTIC_SCHEMA['relationship_types'])}"
)


#################################################################
# STEP 4 - Build and write the semantic knowledge graph
#################################################################

print_step("STEP 4 - Build the semantic knowledge graph")

# Remove the output of an earlier run so one input Markdown always
# produces one consistent set of chunks and entities.
clear_generated_knowledge_graph(driver, NEO4J_DATABASE)

# SOLUTION - Build the semantic knowledge graph:
# Run the asynchronous pipeline on the webpage Markdown from this synchronous
# script. The driver is always closed, even if graph construction fails.
try:
    pipeline_result, processing_time = run_with_progress(
        asyncio.run,
        pipeline.run_async(text=webpage_markdown),
        description="Building semantic knowledge graph",
    )
    remove_document_nodes(driver, NEO4J_DATABASE)
    html_path = save_semantic_graph_html(
        driver=driver,
        schema=SEMANTIC_SCHEMA,
        output_path=SEMANTIC_GRAPH_HTML_PATH,
        neo4j_database=NEO4J_DATABASE,
    )
finally:
    driver.close()

console.print("Semantic knowledge graph written to Neo4j.", style=INFO_STYLE)
print_result(
    f"{pipeline_result}\n"
    f"HTML visualization: {html_path}\n"
    f"Total processing time: {processing_time:.2f} seconds"
)


#################################################################
# STEP 5 - Visualize the semantic graph in Neo4j
#################################################################

print_step("STEP 5 - Visualize in Neo4j")

console.print(
    "Open Neo4j Browser at http://localhost:7474 and run:",
    style=INFO_STYLE,
)
print_result("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 1000")
