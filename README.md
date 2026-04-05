# Real_State_Agent
Real Estate Intelligence Platform for Prime Lands


## RAGService with LCEL

**What RAGService does (conceptually)**
- **Input**: user question.
- **Retrieve**: find the most relevant chunks from your vector store (Qdrant) using embeddings.
- **Generate**: pass those chunks to the LLM so it answers using *evidence* instead of guessing.
- **Output**: a final answer plus citations (URLs from the chunks).

**LCEL  in practice**
LCEL lets you build this as a clean pipeline of `Runnable` steps. Think of it like:

1) **Query → Retriever**  
   Turns the question into a list of relevant chunks.

2) **Chunks → Context formatter**  
   Joins those chunks into a compact context string + keeps their source URLs.

3) **Prompt → LLM**  
   Builds a prompt like: “Answer using only the context. Add citations after each claim.”

4) **LLM → Post‑process**  
   Ensures the answer includes inline citations and returns it to the user.

**“Proper retriever integration”**
the chain is actually calling your retriever *inside* the LCEL pipeline, not doing it manually outside. That makes it composable and easy to swap retrieval methods.

That builds trust and makes your answer verifiable.

## CAG (Cache-Augmented Generation)
CAG is a thin layer that sits in front of RAG and adds a semantic cache so repeated or paraphrased questions can return instantly.

**Two-tier cache (FAQ + history)**
There are two caches on disk: a static FAQ cache (never expires) and a dynamic history cache (expires after a TTL). Both are persisted in the cache directory so they survive restarts.

**Semantic lookup with cosine similarity**
Each cached question has an embedding saved. When a new user question arrives, it is embedded once and compared to cached embeddings with cosine similarity. If the best match is above the threshold (default 0.90), the cached answer is returned.

**FAQ warm-up**
FAQs are registered first (question + embedding) and can be pre-warmed by generating their answers via RAG. This makes high-frequency questions return with near-zero latency.

**History behavior**
History entries store the user question, its embedding, the answer, and a timestamp. Old entries are removed based on TTL and the oldest entries are evicted when the history reaches its size cap.

## CRAG (Corrective RAG)
CRAG adds a confidence-aware retrieval loop before generation. It first retrieves a small set of documents, calculates a confidence score, and only if confidence is low does it expand retrieval and try again. The best evidence set is then used to generate the final answer. This reduces hallucinations and improves grounding on harder queries.




