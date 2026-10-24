"""Graph-RAG over the complete knowledge graph built by SimpleKGPipeline.

Run ``03_simpleKG_construction.py`` first. This example uses an LLM to turn a
question into read-only Cypher, retrieves matching facts from Neo4j, and asks
the LLM to answer using only those facts.
"""

import os
import sys
from pathlib import Path

from neo4j import GraphDatabase
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.generation.prompts import RagTemplate
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import Text2CypherRetriever

from argo.utils.console_utils import (
    INFO_STYLE,
    console,
    print_question_answer,
    print_result,
    print_step,
    run_with_progress,
)
from argo.utils.part_02_semantic_graph_utils import (
    HYBRID_TEXT2CYPHER_PROMPT,
    SEMANTIC_CYPHER_EXAMPLES,
    SEMANTIC_GRAPH_SCHEMA,
    ensure_semantic_graph,
)
from argo.utils.pydantic_utils import get_llm_settings

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
LLM_MODEL, LLM_BASE_URL, LLM_API_KEY = get_llm_settings()
FALLBACK_GRAPH_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "graph_solutions"
    / "semantic_graph_fallback.json"
)

DEFAULT_QUERY = (
    "What venues are available, and which services or activities do they offer?"
)
USER_QUERY = " ".join(sys.argv[1:]).strip() or DEFAULT_QUERY

#################################################################
# STEP 1 - Connect to the knowledge graph
#################################################################

print_step("STEP 1 - Connect to the complete knowledge graph")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
)
driver.verify_connectivity()
restored_fallback = ensure_semantic_graph(
    driver=driver,
    fallback_path=FALLBACK_GRAPH_PATH,
    neo4j_database=NEO4J_DATABASE,
)

console.print("Connected to Neo4j.", style=INFO_STYLE)
print_result(
    f"Database: {NEO4J_DATABASE}\n"
    f"Graph source: {'saved workshop fallback' if restored_fallback else 'generated graph'}"
)


#################################################################
# STEP 2 - Configure hybrid Text-to-Cypher GraphRAG
#################################################################

print_step("STEP 2 - Configure entity-and-chunk GraphRAG")

llm = OpenAILLM(
    model_name=LLM_MODEL,
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    model_params={"temperature": 0.0},
)

# Here we use Neo4j GraphRAG's imported Text2CypherRetriever, which translates a
# natural-language question into Cypher using the schema and examples, executes
# the query in Neo4j, and returns the resulting records as context.
retriever = Text2CypherRetriever(
    driver=driver,
    llm=llm,
    neo4j_schema=SEMANTIC_GRAPH_SCHEMA,
    examples=SEMANTIC_CYPHER_EXAMPLES,
    custom_prompt=HYBRID_TEXT2CYPHER_PROMPT,
    neo4j_database=NEO4J_DATABASE,
)

# EXERCISE - Configure GraphRAG:
# Here we use Neo4j GraphRAG's imported GraphRAG class, which asks the retriever
# for relevant Neo4j records and passes that context to the LLM to generate a
# grounded answer using the instructions below.
graph_rag = ...(
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

console.print(f"Question: {USER_QUERY}", style=INFO_STYLE)


#################################################################
# STEP 3 - Retrieve graph facts and generate the answer
#################################################################

print_step("STEP 3 - Query the graph and generate an answer")

try:
    result, processing_time = run_with_progress(
        graph_rag.search,
        query_text=USER_QUERY,
        return_context=True,
        response_fallback="No matching facts were found in the knowledge graph.",
    )
finally:
    driver.close()
retriever_result = result.retriever_result
if retriever_result:
    console.print("Generated Cypher:", style=INFO_STYLE)
    print_result(retriever_result.metadata.get("cypher", "Unavailable"))

print_question_answer(USER_QUERY, result.answer, processing_time)
