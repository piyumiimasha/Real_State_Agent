# Real_State_Agent
Real Estate Intelligence Platform for Prime Lands


## RAGService with LCEL

**What RAGService does (conceptually)**
- **Input**: user question.
- **Retrieve**: find the most relevant chunks from your vector store (Qdrant) using embeddings.
- **Generate**: pass those chunks to the LLM so it answers using *evidence* instead of guessing.
- **Output**: a final answer plus citations (URLs from the chunks).

**What LCEL means in practice**
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
Means the chain is actually calling your retriever *inside* the LCEL pipeline, not doing it manually outside. That makes it composable and easy to swap retrieval methods.

**“Inline citations with evidence URLs”**
Each claim in the answer should be followed by its source URL, for example:
> “The project is in Colombo 05. (https://primelands.lk/...)”

That builds trust and makes your answer verifiable.



