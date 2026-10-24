# ARGO: Agentic Retrieval and Graph Orchestration for Document Knowledge Systems

A hands-on tutorial for the International Semantic Web Conference (ISWC 2026).

## Installation

### 1. Install `uv`

macOS and Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart the terminal, open the project directory, and install the Python
dependencies and Playwright's Chromium browser:

```bash
uv sync
uv run playwright install --only-shell chromium
```

### 2. Configure the language model

The workshop supports two OpenAI-compatible model endpoints.

#### Option 1— Run Ollama directly on your computer

1. Install Ollama from <https://ollama.com/download>.
2. Start Ollama on your computer.
3. Download the local model:

```bash
ollama pull qwen3:1.7b
```

Select Ollama for one command:

```bash
LLM_PROVIDER=ollama uv run python exercises/part_01/01_entry_point.py
```

Or change the central setting in `src/argo/settings.py` to:

```python
LLM_PROVIDER = "ollama"
```

The local defaults are:

```text
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:1.7b
OLLAMA_BASE_URL=http://localhost:11434/v1
```

#### Option 2 — Use the HEIA vLLM endpoint

HEIA provides a larger vLLM model running in its Kubernetes infrastructure:

```text
LLM_PROVIDER=vllm
VLLM_MODEL=Qwen/Qwen3.8-27B-FP8
VLLM_BASE_URL=https://litellm.kube-ext.isc.heia-fr.ch/v1
```

The workshop copy includes a temporary `.vllm_api_key` file in the project
root. The file contains only the shared key supplied by the instructors; do not
add quotes or a variable name around it. The instructors will remove and rotate
this temporary key after the workshop.

The shared endpoint can occasionally slow down or time out when many
participants send requests simultaneously. Wait briefly and retry, or use
Ollama locally.

### 3. Install Docker Desktop

The semantic KG exercises use Neo4j, which runs in a Docker container.

Install and start [Docker Desktop](https://www.docker.com/products/docker-desktop/), then
verify that Docker is available:

```bash
docker --version
docker compose version
```

#### Start Neo4j with Docker Compose

Install and start [Docker Desktop](https://www.docker.com/products/docker-desktop/),
then run from the project root:

```bash
docker compose up -d
docker compose logs -f neo4j
```

When Neo4j is ready, press `Ctrl+C`; the container will continue running in the
background. Neo4j Browser is available at <http://localhost:7474> with username
`neo4j` and password `password123`.

To stop the workshop container later, run:

```bash
docker compose down
```

## Verify the installation

Run the entry point:

```bash
uv run python exercises/part_01/01_entry_point.py
```

It should answer a simple question and start a web app at
<http://127.0.0.1:8000>. Press `Ctrl+C` to stop it.
