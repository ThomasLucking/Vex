A:x# RAG Learning Sources — Python (FastAPI) + React + pgvector + Ollama

Curated, verified-live sources for building and *understanding* a RAG system with this stack. Each link includes what it teaches, why it's the right source, and a specific detail worth remembering.

---

## 1. Embeddings

**Key concept:** An embedding model turns text into a fixed-length vector of numbers such that semantically similar text ends up geometrically close in that vector space (usually measured with cosine similarity). RAG uses this to turn "search by meaning" into "search by nearest neighbor." With Ollama, you run an embedding model locally and call a simple HTTP endpoint to get vectors back — no external API key needed. The two things that trip people up are: using a different model for indexing vs. querying (the vector spaces aren't compatible across models), and forgetting to normalize/compare with the right distance metric.

- **[Ollama — Embeddings (docs.ollama.com)](https://docs.ollama.com/capabilities/embeddings)** — Jan 2026
  Official docs for generating embeddings locally via `/api/embed`, with Python, JS, and curl examples for single and batch requests. This is the primary source since you're using Ollama specifically — it's the authoritative API reference, not a third-party wrapper.

  Gotcha: **use the same embedding model for both indexing and querying** — mixing models silently produces garbage similarity scores. Also recommends cosine similarity for most semantic search cases and lists current model options (embeddinggemma, qwen3-embedding, all-minilm).

- **[Ollama embedding models library](https://ollama.com/search?c=embedding)**
  Browse available local embedding models (nomic-embed-text, mxbai-embed-large, all-minilm, etc.) with size/dimension info. Useful for picking a model that fits your machine's RAM before committing to a schema (vector dimension is fixed once you create your pgvector column).

  Gotcha: smaller models (all-minilm, 384-dim) are much faster to embed/store but lower quality retrieval than larger ones (mxbai-embed-large, 1024-dim) — dimension choice is a real tradeoff, not just a config detail.

- **[Sentence Transformers — Quickstart (sbert.net)](https://www.sbert.net/docs/quickstart.html)**
  The canonical reference for the embedding *concept* itself, independent of Ollama — loading a model, calling `.encode()`, computing similarity. Worth reading even if you use Ollama in production, because it explains embeddings more pedagogically than API docs do.

  Gotcha: embeddings from different sentence-transformers checkpoints are not interchangeable, and `model.encode()` batches internally — passing one string at a time in a loop is much slower than passing a list.

- **[A Guide to Embeddings and pgvector (Google AI / DEV Community)](https://dev.to/googleai/a-guide-to-embeddings-and-pgvector-df0)**
  Bridges the embeddings concept directly to storing/querying them in Postgres — a good next step after sbert's quickstart, before jumping into the pgvector section below.

  Gotcha: covers why the vector *dimension* in your table schema must exactly match your embedding model's output size, or inserts will fail.

**Common mistake:** Switching embedding models mid-project (e.g. trying a bigger model later) without re-embedding *all* existing documents. Old and new vectors live in the same geometric space only if they came from the same model — mixing them makes retrieval silently worse, not obviously broken.

---

## 2. Chunking strategies

**Key concept:** Documents are too long to embed and retrieve as a single unit — you split them into smaller "chunks" so retrieval can pull just the relevant passage instead of an entire document. Chunk size is a tradeoff: too small and you lose context (a sentence without its paragraph is ambiguous); too large and irrelevant text dilutes the embedding, hurting retrieval precision. Overlap between consecutive chunks exists to prevent a key sentence from being cut in half at a chunk boundary. There's no universal "right" number — it depends on your documents and query type — but there are sane defaults to start from.

- **[Pinecone — Chunking Strategies for LLM Applications](https://www.pinecone.io/learn/chunking-strategies/)**
  The most commonly cited practical guide, from a vector-DB vendor with obvious incentive to get retrieval quality right. Covers fixed-size, content-aware (sentence/paragraph), document-structure-aware, semantic, and LLM-contextual chunking, with trade-offs for each.

  Gotcha: recommends testing multiple sizes empirically (128–256 tokens for narrow factual queries, 512–1024 for broader context) rather than picking one number — and warns different embedding models tokenize differently, so a "500 token" chunk isn't the same size across models.

- **[Best Chunking Strategies for RAG in 2026 (Firecrawl)](https://www.firecrawl.dev/blog/best-chunking-strategies-rag)**
  A newer, stack-agnostic guide with concrete numeric starting points (400–512 tokens, 10–20% overlap) rather than just theory — good as your default baseline before you tune anything.

  Gotcha: notes overlap's benefit is inconsistent — a Jan 2026 study using SPLADE retrieval found overlap gave no measurable improvement on some benchmarks, so don't assume more overlap always helps; measure it on your own data.

- **[Document Chunking for RAG: 9 Strategies (LangCopilot)](https://langcopilot.com/posts/2025-10-11-document-chunking-for-rag-practical-guide)**
  Practical, code-oriented walkthrough of 9 chunking approaches with example parameters, useful once you're past picking a default and want to compare specific techniques (recursive, sentence-window, semantic, etc.) side by side.

  Gotcha: stresses attaching metadata (title, section, chunk index, page number) to every chunk — without it you can retrieve a correct chunk but have no way to show the user *where* it came from.

**Common mistake:** Splitting mid-sentence or mid-code-block with a naive fixed-character splitter (e.g. `text[:500]`) instead of a recursive/structure-aware splitter — this breaks the coherence of the embedded text and directly hurts retrieval quality, and is one of the most common first-attempt bugs.

---

## 3. Vector similarity search with pgvector

**Key concept:** pgvector is a Postgres extension that adds a `vector` column type and distance operators, so you can store embeddings alongside your normal relational data and query them with SQL. A plain sequential scan compares your query vector against every row — correct but slow at scale — so pgvector offers approximate-nearest-neighbor indexes (IVFFlat, HNSW) that trade a small amount of recall for large speedups. Getting index *and* query distance operator to match (e.g. both cosine, or both L2) is essential, or Postgres silently ignores the index.

- **[pgvector — official README (GitHub)](https://github.com/pgvector/pgvector)**
  The single source of truth for syntax: extension setup, column types, distance operators (`<->`, `<=>`, `<#>`), and both index types' creation syntax and tuning parameters (`lists` for IVFFlat, `m`/`ef_construction` for HNSW). Always check this first since flags/defaults change between pgvector versions.

  Gotcha: the index's distance operator must match the one used in your query's `ORDER BY` clause exactly, or Postgres falls back to a full scan without warning — an easy silent-performance-bug to hit.

- **[IVFFlat vs HNSW in pgvector: Which Index Should You Use? (DEV Community)](https://dev.to/philip_mcclarence_2ef9475/ivfflat-vs-hnsw-in-pgvector-which-index-should-you-use-305p)**
  A clear side-by-side comparison for choosing between the two index types before you commit — HNSW gives better recall/query speed at the cost of build time and 2–5x more memory; IVFFlat is faster/cheaper to build but its quality depends on the data distribution present at build time.

  Gotcha: IVFFlat's index quality is fixed at creation time via k-means clustering — if you bulk-insert a lot of new, differently-distributed data afterward, you should rebuild the index, whereas HNSW updates incrementally without a rebuild.

- **[AWS — Optimize generative AI applications with pgvector indexing](https://aws.amazon.com/blogs/database/optimize-generative-ai-applications-with-pgvector-indexing-a-deep-dive-into-ivfflat-and-hnsw-techniques/)**
  A deeper technical dive into how each index algorithm actually works internally (graph layers for HNSW, cluster/probe counts for IVFFlat), useful once you want to actually tune parameters instead of using defaults.

  Gotcha: covers the `ef_search` / `probes` runtime query parameters that trade recall for speed *per query* — these are separate from the index build-time parameters and are easy to forget to set.

- **[Supabase — pgvector: Embeddings and vector similarity](https://supabase.com/docs/guides/database/extensions/pgvector)**
  A vendor doc, but a genuinely good practical walkthrough of enabling the extension, creating a vector column, and querying it — useful as a second, more beginner-oriented reference alongside the official README.

  Gotcha: explicitly calls out that for most RAG/semantic-search apps, HNSW should be your default unless you have a specific reason (huge static dataset, tight memory budget) to use IVFFlat.

**Common mistake:** Creating an index with one distance operator class (e.g. `vector_l2_ops`) but querying with a different operator (`<=>` for cosine) — Postgres won't error, it will just silently not use the index and fall back to a full sequential scan, so query performance looks fine on small tables and falls apart at scale.

---

## 4. Prompt augmentation / grounding

**Key concept:** "Augmentation" is the step where retrieved chunks get inserted into the prompt sent to the LLM, alongside the user's question and an instruction to answer *using only* the provided context. This is what "grounds" the model's answer in real, retrievable text instead of relying purely on what it memorized during training, which is why RAG reduces (but doesn't eliminate) hallucination. Streaming means sending the LLM's response back to the client token-by-token as it's generated, rather than waiting for the full answer — this matters for RAG because retrieval + generation can take a few seconds, and streaming keeps the UI feeling responsive.

- **[AWS Prescriptive Guidance — Grounding and Retrieval Augmented Generation](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-serverless/grounding-and-rag.html)**
  A vendor-neutral, architecture-level explanation of what grounding actually means and how RAG achieves it, good for building the mental model before writing prompt templates.

  Gotcha: emphasizes that grounding reduces hallucination but doesn't guarantee correctness — if retrieval returns irrelevant chunks, the model will still confidently answer from bad context, so retrieval quality is the actual bottleneck, not the prompt template.

- **[Prompting Guide — Retrieval Augmented Generation (RAG) for LLMs](https://www.promptingguide.ai/research/rag)**
  A well-maintained reference on the RAG technique itself, including prompt-construction patterns (how to format retrieved context, instruct the model to cite sources or say "I don't know").

  Gotcha: covers why instructing the model explicitly to answer only from provided context (and to say when it can't) matters — without that instruction, models default to blending retrieved context with parametric memory, muddying provenance.

- **[FastAPI — Server-Sent Events (official tutorial)](https://fastapi.tiangolo.com/tutorial/server-sent-events/)**
  Official docs for streaming responses from FastAPI using `yield` and `EventSourceResponse` — this is the mechanism you'll use to stream the LLM's grounded answer back to the frontend as it's generated.

  Gotcha: FastAPI's SSE tutorial notes you should check `await request.is_disconnected()` inside your generator loop — without it, if a client closes the tab mid-stream, the server keeps generating (and paying for) tokens indefinitely.

**Common mistake:** Stuffing retrieved chunks into the prompt with no instruction about what to do when the retrieved context doesn't answer the question — the model then hallucinates a confident answer anyway instead of saying "I don't know," which defeats the point of grounding.

---

## 5. FastAPI + React integration (with streaming)

**Key concept:** FastAPI serves your RAG pipeline (retrieval + LLM call) as an HTTP API; React consumes it. For a chat-style RAG app, a plain request/response cycle feels sluggish because the user waits for the entire answer to generate before seeing anything. Server-Sent Events (SSE) let the backend push a stream of small text events over a single HTTP connection, and the browser's native `EventSource` API (or a fetch-based reader) consumes them incrementally on the frontend, so the UI can render tokens as they arrive — the same UX pattern used by ChatGPT.
Note: some tutorials use WebSockets instead of SSE for this — SSE is simpler (plain HTTP, auto-reconnect, one-directional) and is the better default for streaming a one-way LLM response; reach for WebSockets only if you need bidirectional communication.

- **[FastAPI — Server-Sent Events (official docs)](https://fastapi.tiangolo.com/tutorial/server-sent-events/)**
  Same source as above — this is the backend half of streaming and the correct starting point since it's official and current.

  Gotcha: FastAPI implements keep-alive pings and cache-prevention headers automatically for you when you use `EventSourceResponse` — reimplementing these yourself with a raw `StreamingResponse` is unnecessary extra work.

- **[Server-Sent Events using FastAPI and ReactJS (GitHub, harshitsinghai77)](https://github.com/harshitsinghai77/server-sent-events-using-fastapi-and-reactjs)**
  A small, focused, working example repo pairing a FastAPI SSE backend with a React frontend using `EventSource` — good to read end-to-end since it's short enough to fully understand in one sitting, unlike a full production app.

  Gotcha: demonstrates the frontend side detail that's easy to miss — `EventSource` only supports GET requests natively, so if your RAG query needs a POST body (e.g. a long question + retrieved context), you need a fetch-based streaming reader instead of the native `EventSource` object.

- **[Full-stack RAG: Streaming with FastAPI & React, Part 2 (Joey O'Neill, Medium)](https://medium.com/@o39joey/full-stack-rag-streaming-with-fastapi-react-part-2-33ec2f76ce8a)**
  Directly extends a RAG pipeline (built in Part 1) with a React TypeScript frontend and streaming — closer to your actual end goal than the generic SSE demo above, showing how the RAG chain output gets wired to the stream.

  Gotcha: worth reading Part 1 first (linked from Part 2) since the streaming layer assumes a working non-streaming RAG endpoint already exists — build that first, then add streaming, rather than both at once.

**Common mistake:** Building the streaming UI before the non-streaming RAG pipeline works end-to-end — debugging retrieval quality and prompt construction is much harder when you're also fighting stream parsing bugs at the same time. Get a synchronous `/query` endpoint fully correct first, then layer streaming on top.

---

## 6. End-to-end tutorials (this exact stack)

**Key concept:** Full walkthroughs matter because RAG has more integration surface area than any single piece — chunking choices affect retrieval quality, retrieval quality affects prompt construction, and streaming touches both backend and frontend. Reading one complete tutorial before diving into the pieces individually gives you a skeleton to hang the deep-dive knowledge from. None of the sources below use Ollama specifically end-to-end (most default to OpenAI's embedding API), but the FastAPI/pgvector/React architecture transfers directly — swap the embedding call for Ollama's `/api/embed`.

- **[Building a Full-Stack Portfolio Website with a RAG-Powered Chatbot (Neon Guides)](https://neon.com/guides/react-fastapi-rag-portfolio)** — 2026
  The most complete match for your stack: React frontend, FastAPI backend, pgvector (via Neon's managed Postgres) for storage, walked through as one cohesive project rather than isolated snippets. Swap Neon for a local Postgres+pgvector and OpenAI embeddings for Ollama's, and the architecture is directly transferable.

  Gotcha: explicitly notes you need the FastAPI backend actually running (not just the React dev server) to test the chatbot — an obvious-in-hindsight but common first-run confusion point.

- **[Full-stack RAG: FastAPI Backend, Part 1 (Joey O'Neill, Medium)](https://medium.com/@o39joey/full-stack-rag-fastapi-backend-part-1-eab28eb21392)**
  Builds the backend half of the pipeline — read before Part 2 (linked in section 5) to get the non-streaming RAG endpoint working first, per the ordering advice above.

  Gotcha: structures the RAG logic as a reusable class/chain separate from the API route handlers — a useful pattern for keeping retrieval logic testable independent of FastAPI.

- **[Building a FastAPI-Powered RAG Backend with PostgreSQL & pgvector (Fredy, Medium)](https://medium.com/@fredyriveraacevedo13/building-a-fastapi-powered-rag-backend-with-postgresql-pgvector-c239f032508a)**
  Backend-only but goes deeper into the actual pgvector-facing Python code (a `RAGLocal`-style class handling storing, indexing, and querying) than the Neon guide does — good for understanding the database interaction layer specifically.

  Gotcha: shows connection-pooling considerations for the Postgres client under concurrent FastAPI requests — a detail that's easy to skip in a first pass and only bites you under load.

**Common mistake:** Trying to copy one of these tutorials verbatim including its choice of embedding provider (usually OpenAI) without adjusting vector dimensions in the Postgres schema — pgvector column dimension is fixed at table-creation time, and OpenAI's `text-embedding-3-small` (1536-dim) won't match Ollama's `nomic-embed-text` (768-dim) or `all-minilm` (384-dim), so the schema needs to match whichever model you actually use.

---

## Suggested reading order

Estimated time is for the core path only — the "why is this the source" notes above tell you which deep-dive links to save for later.

1. **Embeddings concept** — sbert.net Quickstart, then Ollama's embeddings docs. Understand what a vector is and how to generate one locally. *(~30 min)*
2. **Chunking** — Pinecone's chunking strategies guide. Understand why documents get split and what size/overlap defaults to start with. *(~20 min)*
3. **pgvector basics** — pgvector README (extension setup, column type, one distance operator, one index type — pick HNSW). Get a table storing and querying vectors with a toy dataset. *(~40 min)*
4. **Grounding / prompt augmentation** — AWS grounding guide + Prompting Guide's RAG page. Understand how retrieved chunks become part of the LLM prompt, and why instructing the model to stick to context matters. *(~20 min)*
5. **Build the non-streaming pipeline** — wire steps 1–4 together behind a single synchronous FastAPI endpoint (`/query`). This is where understanding solidifies — nothing above is fully real until it's running. *(implementation, not a link)*
6. **FastAPI SSE** — official server-sent-events tutorial. Add streaming to the working endpoint from step 5. *(~20 min)*
7. **React streaming consumption** — the harshitsinghai77 SSE example repo, read end-to-end. Wire your React frontend to consume the stream. *(~20 min)*
8. **pgvector indexing deep dive** — IVFFlat vs HNSW comparison + AWS deep-dive. Come back to this once retrieval works correctly and you care about query latency at scale, not before.
9. **End-to-end tutorials** (Neon guide, Joey O'Neill's two-parter) — read *after* building your own version, to compare architectural decisions against what you built and spot things you missed.

**Core concepts (steps 1–7): roughly 2.5–3 hours of reading plus the implementation time to get a working non-streaming and then streaming endpoint.** Steps 8–9 are reference material to return to as needed, not part of the initial learning path.
