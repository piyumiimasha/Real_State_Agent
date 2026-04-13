# Executive Summary
This report reviews the Primelands real estate RAG system across crawling, chunking, retrieval, and caching, using evaluation artifacts in src/eval_outputs. The current run is a token-saving, retrieval-only pass (3 queries x 5 strategies). Because generation is disabled, answer relevance and end-to-end latency are not populated, and the consensus relevance heuristic yields precision/recall of 0.0 for all strategies. Under those constraints, retrieval latency is the only discriminating metric. The clear winner is late_base at 586.52 ms average retrieval latency, followed by sliding (637.55 ms) and fixed (646.50 ms); semantic is slowest at 1475.82 ms. 

CAG caching shows strong operational impact: 82% hit rate, 618 ms average hit latency versus 3572 ms misses, an 82.7% latency improvement in simulation. CRAG applied corrections on 6 of 20 queries, with average confidence gain of 0.073 but a small average answer-quality change of -0.0535, suggesting confidence improves but quality needs validation at full generation. 

Recommendation: adopt late_base as the provisional chunking winner for speed, keep CAG enabled for latency and cost savings, and re-run full RAG generation with a larger query set to confirm quality-based winners before production.

# Methodology
- Crawling: run_primelands_crawl.py uses PrimelandsWebCrawler against https://www.primelands.lk/ with depth 3, 2.0s delay, and exclusion patterns for media, auth, and non-English paths. This balances coverage with politeness, avoids duplicate or irrelevant assets, and yields both JSONL and markdown for downstream chunking and inspection.

- Chunking: five strategies in chunkers.py (semantic heading-aware, fixed, sliding, parent-child, late chunking) are configured via config.yaml. Sizes are tuned to balance context and recall: fixed 800/100 for predictable embeddings, semantic 200-1000 to preserve section structure, sliding 512/256 to reduce boundary loss, parent 1200 + child 250/50 to retrieve precise snippets with larger context, and late base 1000 + split 300 + context window 150 to defer fine splits until query time.

- RAG architecture: LCEL pipeline in rag_service.py (Retriever -> format_docs -> prompt -> LLM -> parser) with Qdrant as the vector store. Embeddings use OpenRouter text-embedding-3-large; chat generation is configured via the Groq provider in the notebook. The prompt enforces citation-backed answers.

- Caching: CAG uses a two-tier semantic cache (FAQ + history) with similarity threshold 0.90 and 24h TTL to capture paraphrases and reduce repeated calls. CRAG uses confidence-based corrective retrieval (threshold 0.6, expanded_k 8) to expand evidence when initial retrieval is weak.

# Analysis - The Chunking Showdown
Quantitative comparison (mean retrieval latency; n=15 rows, 3 queries x 5 strategies):

| Strategy | Avg Retrieval Latency (ms) | Precision@K | Recall@K | Notes |
| --- | ---: | ---: | ---: | --- |
| late_base | 586.52 | 0.00 | 0.00 | Fastest in current run; late splitting reduces index bloat |
| sliding | 637.55 | 0.00 | 0.00 | Good coverage, higher index size |
| fixed | 646.50 | 0.00 | 0.00 | Simple, predictable size |
| parent_child_children | 676.40 | 0.00 | 0.00 | Rich context but heavier retrieval |
| semantic | 1475.82 | 0.00 | 0.00 | Slowest; heading-aware parsing overhead |

Qualitative notes: semantic chunking preserves topical boundaries but can create uneven chunk sizes and slower retrieval; fixed and sliding trade semantic coherence for simplicity and recall; parent-child boosts context for child hits but increases storage and retrieval work; late chunking defers fine splits until retrieval, reducing index size and query-time overhead in this run. Example queries such as "Which apartments are available in Colombo?" and "List any land projects in Kaduwatha" show identical precision/recall across strategies because the run uses only 3 queries and a consensus relevance heuristic. Recommendation: late_base is the clear latency winner today, but a full run with generation enabled (answer relevance + end-to-end latency) and a larger query set is required to validate quality trade-offs before final selection.

# Conclusion & Recommendations
- Provisional architecture: late_base chunking + LCEL RAG + CAG semantic cache, with CRAG enabled for low-confidence queries.
- Scalability: late chunking and caching reduce index size and repeated generation costs; Qdrant supports larger corpora with minimal code changes.
- Cost projection: CAG simulation shows 0.82 hit rate and 82.7% latency improvement; cost savings in cag_stats.json are based on hard-coded assumptions and should be updated with real provider pricing.
- Next steps: re-run evaluation with TOKEN_SAVING=False, increase query count, enable answer relevance and end-to-end latency, and re-score chunking based on both quality and speed.
