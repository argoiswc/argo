# Use the shared HEIA vLLM service by default. Set this to "ollama" to run the
# workshop with Ollama installed directly on the participant's computer.

#LLM_PROVIDER = "ollama"

LLM_PROVIDER = "vllm"

# Option 1: vLLM running on the HEIA Kubernetes infrastructure.
DEFAULT_VLLM_MODEL = "Qwen/Qwen3.8-27B-FP8"
DEFAULT_VLLM_BASE_URL = "https://litellm.kube-ext.isc.heia-fr.ch/v1"

# Option 2: Ollama running on the participant's computer.
DEFAULT_OLLAMA_MODEL = "qwen3:1.7b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"

DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 400
DEFAULT_VECTORSTORE_NAME = "part_01_argo_usecase_webpage"

DEFAULT_NB_RETRIEVED_CHUNKS = 5
