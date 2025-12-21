# Hybrid RAG Chatbot Prototype

Minimal yet structured playground for a hybrid Retrieval-Augmented Generation (RAG) chatbot that blends FAISS vector search with a Neo4j knowledge graph. The starter corpus focuses on technical documentation.

## Key Capabilities
- Markdown/Text/PDF/HTML ingestion with chunking metadata suitable for Neo4j.
- FAISS-backed embedding store built with LangChain + Ollama embeddings.
- Neo4j synchronization that mirrors documents/chunks as nodes and relationships.
- Typer-powered CLI with `ingest` and `chat` commands, plus an optional Streamlit UI.
- Hybrid pipeline that merges vector context with graph reasoning through LangChain's GraphCypher chain.

## Project Structure

```
.
├── data/docs/              # Place your technical docs here (starter sample included)
├── src/hybrid_rag/         # Core package (config, ingest, graph, vector, CLI)
├── storage/vector_store/   # FAISS artifacts get persisted here
├── .env.example            # Copy to .env and fill credentials
└── pyproject.toml          # Dependency manifest
```

## Prerequisites

- Python 3.10 – 3.12
- [Ollama](https://ollama.com) installed locally with the `mistral` and `nomic-embed-text` models pulled
- Running Neo4j instance reachable from your machine

## Setup

1. **Create a virtual environment** (recommended):
	```bash
	python -m venv .venv && source .venv/bin/activate
	```
2. **Install dependencies**:
	```bash
	python -m pip install -e .
	```
3. **Configure environment variables**:
	```bash
	cp .env.example .env
	# edit .env with Neo4j credentials + Ollama overrides if needed
	```

## Usage

### Ingest documentation

```bash
python -m hybrid_rag.cli ingest \
  --data-dir data/docs        # optional override; defaults to config value
```

The command will:
1. Load markdown, text, HTML, and PDF files.
2. Chunk them with overlap-aware splitter.
3. Build/update the FAISS vector index under `storage/vector_store/`.
4. Bootstrap Neo4j constraints and sync chunk/document nodes.

### Chat with the hybrid pipeline

Interactive mode:

```bash
python -m hybrid_rag.cli chat
```

Single-shot question:

```bash
python -m hybrid_rag.cli chat -q "How does the ingestion flow work?" --debug
```

Use `--debug` to print the vector chunks and graph reasoning returned for each answer.

### Streamlit UI

If you prefer a lightweight web UI, launch Streamlit from the project root:

```bash
streamlit run streamlit_app.py
# ou avec uv : uv run streamlit run streamlit_app.py
```

The sidebar lets you point to a custom `.env`, toggle retrieval context, and reload the pipeline without restarting the server.

## Environment Variables

| Variable | Description |
| --- | --- |
| `OLLAMA_BASE_URL` | Base URL for the local Ollama daemon |
| `NEO4J_URI` | Bolt URI, e.g. `bolt://localhost:7687` |
| `NEO4J_USERNAME` / `NEO4J_PASSWORD` | Neo4j credentials |
| `NEO4J_DATABASE` | Target Neo4j database (defaults to `neo4j`) |
| `EMBEDDING_MODEL` / `CHAT_MODEL` | Override default Ollama models (e.g., `mistral:instruct`) |
| `CHAT_TEMPERATURE` | Sampling temperature for chat responses |
| `TOP_K_VECTORS` / `TOP_K_GRAPH` | Retrieval fan-out controls |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Text splitter tuning |
| `DATA_DIR` / `VECTOR_STORE_PATH` | Paths for data + FAISS artifacts |

## Next Steps

- Integrate evaluation harnesses (LLM-as-a-judge or golden answers).
- Enrich the Neo4j schema (sections, topics, authors) for deeper graph reasoning.
- Experiment with alternative embedding backends (e.g., local models, Voyage). 
