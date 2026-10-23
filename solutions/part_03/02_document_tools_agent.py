"""Agentic workflow with document, structural, and semantic graph tools.

The agent can choose between:
- listing available source documents,
- searching with Graph-RAG,
- querying the SimpleKG semantic graph in Neo4j,
- searching the web.
"""

import os
from pathlib import Path

import chromadb
import uvicorn
from neo4j import GraphDatabase
from pydantic_ai import Agent, ModelSettings

from graph_rag_workshop.settings import DEFAULT_NB_RETRIEVED_CHUNKS
from graph_rag_workshop.utils.console_utils import INFO_STYLE, console, print_step
from graph_rag_workshop.utils.part_01_document_tools import (
    extract_text_from_image_file,
    extract_text_from_pdf_file,
    list_my_available_documents,
)
from graph_rag_workshop.utils.part_02_graph_utils import (
    search_structural_graph_rag_context,
)
from graph_rag_workshop.utils.part_02_semantic_graph_utils import (
    ensure_semantic_graph,
    search_semantic_graph_rag,
)
from graph_rag_workshop.utils.part_03_resilience_utils import (
    RetryAndRecoverCapability,
)
from graph_rag_workshop.utils.part_03_web_tools import safe_duckduckgo_search
from graph_rag_workshop.utils.pydantic_utils import get_llm_model, get_llm_settings

# Define file paths and constants
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "data"
VECTORSTORES_DIR = DATA_DIR / "vectorstores"
VECTORSTORE_NAME = "part_01_argo_usecase_webpage"
GRAPH_JSON_PATH = (
    DATA_DIR / "graph_solutions" / "webpage_chunk_graph_after_llm_review.json"
)
SEMANTIC_GRAPH_FALLBACK_PATH = (
    DATA_DIR / "graph_solutions" / "semantic_graph_fallback.json"
)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
LLM_MODEL, LLM_BASE_URL, LLM_API_KEY = get_llm_settings()
VECTORSTORE_DATABASE = chromadb.PersistentClient(
    path=str(VECTORSTORES_DIR)
).get_collection(name=VECTORSTORE_NAME)

agent = Agent(
    model=get_llm_model(),
    instructions=(
        "You are a helpful assistant. Use the available tools to answer the user "
        "question. For every question about the Wayfarer event webpage, always call "
        "search_wayfarer_webpage_graph_rag first. Answer from that structural "
        "Graph-RAG result when it contains sufficient evidence. Call "
        "search_wayfarer_semantic_graph_rag only when the structural result is "
        "missing required information, incomplete, or ambiguous; use its entities, "
        "relationships, and source chunks to complete the answer. Do not call the "
        "semantic tool by default. Answer only from the graph tool results. Treat "
        "Name and Section path as authoritative metadata. If a tool returns TOOL_ERROR or "
        "WEB_SEARCH_ERROR, report it briefly and continue with another available "
        "tool or with the context already collected. Never abandon the complete "
        "answer only because one tool failed. Before every tool call, write one "
        "short user-visible sentence explaining which tool you will use and why. "
        "After the tool returns, briefly summarize what useful information it "
        "provided before choosing the next action. Do not reveal private "
        "chain-of-thought."
    ),
    capabilities=[RetryAndRecoverCapability()],
    # SOLUTION - Document tools:
    # Register the Part 1 helpers as tools the agent can choose from.
    tools=[
        extract_text_from_image_file,
        extract_text_from_pdf_file,
        list_my_available_documents,
        safe_duckduckgo_search,
    ],
    model_settings=ModelSettings(
        thinking="minimal",
    ),
    tool_retries=2,
    output_retries=2,
)


# SOLUTION - Graph-RAG tool:
# This is the same structural webpage Graph-RAG tool used in the previous solution.
@agent.tool_plain
def search_wayfarer_webpage_graph_rag(question: str) -> str:
    """Retrieve webpage chunks and all their one-hop structural neighbors."""
    return search_structural_graph_rag_context(
        vector_database=VECTORSTORE_DATABASE,
        graph_json_path=GRAPH_JSON_PATH,
        question=question,
        top_k=DEFAULT_NB_RETRIEVED_CHUNKS,
    )


# SOLUTION - Neo4j entity-and-chunk Graph-RAG tool:
# Open a short-lived connection and query the complete SimpleKG graph.
@agent.tool_plain
def search_wayfarer_semantic_graph_rag(question: str) -> str:
    """Answer from Neo4j entities, relationships, and supporting chunks."""
    with GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    ) as driver:
        driver.verify_connectivity()
        ensure_semantic_graph(
            driver=driver,
            fallback_path=SEMANTIC_GRAPH_FALLBACK_PATH,
            neo4j_database=NEO4J_DATABASE,
        )
        return search_semantic_graph_rag(
            driver=driver,
            question=question,
            model_name=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            neo4j_database=NEO4J_DATABASE,
        )


if __name__ == "__main__":
    print_step("Document Tools Agent")
    app = agent.to_web()
    console.print(
        "Starting Document Tools Agent on http://127.0.0.1:8000",
        style=INFO_STYLE,
    )
    uvicorn.run(app, host="127.0.0.1", port=8000)
