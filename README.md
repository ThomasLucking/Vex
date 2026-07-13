# Vex

A local-first RAG (Retrieval-Augmented Generation) application. React frontend, Python/FastAPI backend, Postgres + pgvector for vector storage, and a local LLM served via Ollama — running entirely on a Mac Mini M1 (16GB unified memory).

## Stack

- **Frontend:** React (streams the LLM response via SSE / fetch reader)
- **Backend:** Python, FastAPI
- **Vector store:** PostgreSQL + pgvector (HNSW index)
- **LLM + embeddings:** Ollama, running locally on Mac Mini M1 16GB
- **Streaming:** Server-Sent Events (SSE) between FastAPI and React

## Architecture

1. Documents are chunked (recursive/structure-aware splitting, ~400–512 tokens, 10–20% overlap).
2. Chunks are embedded locally via Ollama's `/api/embed` and stored in a `vector` column in Postgres (pgvector).
3. A user query is embedded with the **same** embedding model, then matched against stored chunks using cosine similarity (HNSW index).
4. Retrieved chunks are inserted into a prompt template instructing the LLM to answer only from the provided context (and say when it can't).
5. FastAPI streams the LLM's grounded answer back to React token-by-token over SSE.

## Hardware constraints (Mac Mini M1, 16GB)

- Pick embedding/LLM model sizes that fit comfortably in 16GB unified memory — leave headroom for Postgres and the OS.
- Use the same embedding model for indexing and querying; the pgvector column dimension must match that model's output size exactly and is fixed at table-creation time.
- Prefer HNSW over IVFFlat as the default index unless memory becomes a constraint (HNSW uses more memory but updates incrementally and gives better recall).

## Build order

1. Get embeddings working locally (Ollama `/api/embed`).
2. Implement chunking for source documents.
3. Store and query vectors in Postgres/pgvector with one index type (HNSW).
4. Build prompt augmentation/grounding with a synchronous `/query` FastAPI endpoint — get this fully correct before adding streaming.
5. Add SSE streaming to the FastAPI endpoint.
6. Wire the React frontend to consume the SSE stream.

## Docs

See [`docs/sources.md`](docs/sources.md) for curated reference material and gotchas for each part of the stack (embeddings, chunking, pgvector, grounding, FastAPI+React streaming, and end-to-end examples).
