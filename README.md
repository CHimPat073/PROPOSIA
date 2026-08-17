# RAG Support

RAG pipeline for an AI Sales Proposal Generator. Loads company knowledge from
`knowledge_base/`, chunks it, embeds it with `BAAI/bge-small-en-v1.5`, and stores
it in a local ChromaDB collection. Provides a single CLI for ingesting and
querying.

## Layout

```
backend/
  app/
    cli.py              # single CLI entry point (ingest / query / verify / info)
    ingestion/          # loader, chunker, embedder, ingest pipeline
    database/           # chroma vector store wrapper
knowledge_base/         # source documents (.md, .json)
verify_chunks.py        # legacy wrapper -> `rag-support verify`
```

## Setup

```bash
uv sync
# or: pip install -r requirements.txt
cp .env.example .env   # optional, only needed for HF_TOKEN
```

## Usage

All commands are run from the project root.

```bash
# 1. Ingest the knowledge base into ChromaDB (idempotent)
python -m backend.app.cli ingest

# 2. Ask a question
python -m backend.app.cli query -q "Does the company have experience with data engineering and cloud solutions?" -k 5

# 3. End-to-end verify: ingest + counts + one sanity query
python -m backend.app.cli verify

# 4. Show collection stats
python -m backend.app.cli info
```

Equivalent console scripts (after `uv sync`):

```bash
rag-support ingest
rag-support query -q "..."
rag-verify
```

## Expected output (verify)

```
Documents          : 12
Chunks             : 27
Embeddings         : 27 (dim=384)
Chroma total       : 27
```
