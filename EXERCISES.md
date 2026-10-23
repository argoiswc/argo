# Workshop Exercise Guide

Complete the exercises in order. Run all commands from the project root. 

Each exercise file contains placeholders marked `EXERCISE` and `...`; replace those
placeholders, then run the script to check your work.

## Part 01 — Agents, document tools, and webpage RAG

### Exercise 1.1 — Entry point

File: [`exercises/part_01/01_entry_point.py`](exercises/part_01/01_entry_point.py)

This completed example introduces a Pydantic AI agent. It sends a smoke-test
question to the selected model and exposes the agent through a local web app.

```bash
uv run python exercises/part_01/01_entry_point.py
```

Open <http://127.0.0.1:8000> and press `Ctrl+C` when finished.

### Exercise 1.2 — Extract content from documents

File: [`exercises/part_01/02_document_tools.py`](exercises/part_01/02_document_tools.py)

Complete the three extraction tasks:

1. Extract text from the supplied PDF with `pdfminer`.
2. Create a `RapidOCR` engine and extract text from the supplied image.
3. Extract the Wayfarer webpage as Markdown and save it in
   `data/webpage_rag/ARGO_Usecase.md`.

```bash
uv run python exercises/part_01/02_document_tools.py
```

### Exercise 1.3 — Build webpage RAG

File: [`exercises/part_01/03_webpage_rag.py`](exercises/part_01/03_webpage_rag.py)

Complete the RAG pipeline:

1. Split the Markdown by headings.
2. Recursively split large sections with overlap.
3. Choose how many chunks to retrieve with `TOP_K`.
4. Pass the question and retrieved context to the answer function.

The script saves the chunks to `data/webpage_rag/ARGO_Usecase_chunks.md` and
stores their embeddings in the ChromaDB collection under `data/vectorstores/`.

```bash
uv run python exercises/part_01/03_webpage_rag.py
```

Check that chunks are created, relevant context is retrieved, and the model
answers using that context.

## Part 02 — Graph construction and Graph-RAG

Part 02 reuses the chunks and ChromaDB collection produced in Part 01. Run
Exercise 1.3 before continuing.

### Exercise 2.1 — Build a structural graph

File: [`exercises/part_02/01_graph_construction.py`](exercises/part_02/01_graph_construction.py)

Complete the graph construction and review:

1. Iterate over the prepared section links and add `SECTION` edges.
2. Iterate over the prepared subsection links and add `SUBSECTION` edges.
3. Configure the model review for more consistent output.

```bash
uv run python exercises/part_02/01_graph_construction.py
```

The script writes the graph JSON and interactive HTML visualizations to
`data/graph_solutions/`. Compare the visualizations from before and after the
model review.

### Exercise 2.2 — Add graph context to vector retrieval

File: [`exercises/part_02/02_graph_rag.py`](exercises/part_02/02_graph_rag.py)

Complete the structural Graph-RAG flow:

1. Query the ChromaDB collection for the most relevant chunks.
2. Find the incoming and outgoing one-hop neighbors of those chunks.
3. Run the final prompt with both the vector results and graph neighbors.

```bash
uv run python exercises/part_02/02_graph_rag.py
```

Check that the output reports retrieved chunks, followed graph links, and a
grounded final answer.

### Exercise 2.3 — Build a semantic knowledge graph

File: [`exercises/part_02/03_simpleKG_construction.py`](exercises/part_02/03_simpleKG_construction.py)

This exercise requires Neo4j. If it is not already running, start it first:

```bash
docker compose up -d
```

Complete the semantic graph pipeline:

1. Instantiate `SimpleKGPipeline` with the supplied model, Neo4j driver,
   embedder, and schema.
2. Run the asynchronous pipeline on the webpage Markdown.

```bash
uv run python exercises/part_02/03_simpleKG_construction.py
```

Open <http://localhost:7474>, sign in with `neo4j` / `password123`, and run:

```cypher
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 1000
```

The script also saves an interactive visualization as
`data/graph_solutions/simplekg_graph.html`.

### Exercise 2.4 — Query the semantic graph with Graph-RAG

File: [`exercises/part_02/04_semantic_graph_rag.py`](exercises/part_02/04_semantic_graph_rag.py)

Keep Neo4j running. Complete the `GraphRAG` configuration using the provided
retriever, model, and prompt template.

Run the default question:

```bash
uv run python exercises/part_02/04_semantic_graph_rag.py
```

Or supply your own question:

```bash
uv run python exercises/part_02/04_semantic_graph_rag.py "Which venue supports outdoor activities?"
```

The script displays the generated Cypher and the grounded answer. If the live
semantic graph is unavailable, it restores the workshop fallback graph.

## Part 03 — Tool calling and agentic workflows

### Exercise 3.1 — Give an agent a simple tool

File: [`exercises/part_03/01_simple_tool_call.py`](exercises/part_03/01_simple_tool_call.py)

Define a function that returns the current date and time, and register it with
the agent using the decorator indicated in the exercise comments.

```bash
uv run python exercises/part_03/01_simple_tool_call.py
```

Open <http://127.0.0.1:8000>, ask for the current date or time, and verify that
the agent calls the tool. Press `Ctrl+C` before starting the next web app.

### Exercise 3.2 — Build the complete document-tools agent

File: [`exercises/part_03/02_document_tools_agent.py`](exercises/part_03/02_document_tools_agent.py)

This exercise reuses the vector store and structural graph from Parts 01 and 02.
Keep Neo4j running to enable the semantic graph tool.

Complete the agent configuration:

1. Register the PDF, image, document-listing, and safe web-search helpers.
2. Register the structural Graph-RAG function as an agent tool.
3. Register the semantic Neo4j Graph-RAG function as an agent tool.

```bash
uv run python exercises/part_03/02_document_tools_agent.py
```

Open <http://127.0.0.1:8000> and try questions that require different tools,
for example:

- What documents are available?
- What venues are available and what are their characteristics?
- Which services and activities are connected to each venue?
- Based on the email exchange, which venue is compliant? 

The agent should explain which tool it is using, recover when an optional tool
is unavailable, and base webpage answers on the graph results.

## Solutions

Completed versions of every exercise are under `solutions/`, following the same
part and file numbering. To run a solution, use the corresponding path, for
example:

```bash
uv run python solutions/part_02/02_graph_rag.py
```
