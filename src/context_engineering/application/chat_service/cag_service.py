"""
CAG (Cache-Augmented Generation) service combining caching with RAG.

Provides:
- CAGService: RAG with intelligent semantic caching
- Static FAQs: Predefined questions always cached
- Dynamic History: User queries with 24-hour TTL
- Instant responses for cached queries

Workflow:
    Query → Embed → Check FAQs (semantic) → Check History (semantic) → Hit? Return / Miss? RAG + Cache

Benefits:
- ⚡ Near-0s latency for cache hits
- 💰 Zero API costs for cached queries  
- 🎯 Semantic matching catches paraphrased questions
- 🪶 Lightweight: only new queries need embedding
- 📋 Two-tier: Static FAQs + Dynamic History (24h TTL)
"""

from typing import Any, Dict, List
import time

from context_engineering.application.chat_service.cag_cache import CAGCache
from context_engineering.application.chat_service.rag_service import RAGService


class CAGService:
    """
    Cache-Augmented Generation service with semantic matching.
    
    Two-tier caching:
    1. Static FAQs: Predefined, never expire
    2. Dynamic History: User queries, 24-hour TTL
    
    All lookups use lightweight semantic similarity (cosine of embeddings).
    
    Usage:
        from context_engineering.infrastructure.llm_providers import get_default_embeddings
        
        embedder = get_default_embeddings()
        cache = CAGCache(
            cache_dir=Path("data/cache"),
            embedder=embedder,
            similarity_threshold=0.90,
            history_ttl_hours=24
        )
        
        cag_service = CAGService(rag_service, cache)
        
        # Load and warm FAQs
        cag_service.load_faqs([
            "What are the visiting hours?",
            "How do I book an appointment?",
        ])
        cag_service.warm_faqs()
        
        # Now queries hit cache (semantic matching!)
        result = cag_service.generate("Tell me the visiting hours")
        # cache_hit: True, source: 'faq', similarity: 0.94
    """
    
    def __init__(
        self,
        rag_service: RAGService,
        cache: CAGCache
    ):
        """
        Initialize CAG service.
        
        Args:
            rag_service: RAGService instance for generation
            cache: CAGCache instance
        """
        self.rag_service = rag_service
        self.cache = cache
        
        # Hit rate tracking
        self._hits = 0
        self._misses = 0
        self._faq_hits = 0
        self._history_hits = 0
    
    @property
    def hit_rate(self) -> float:
        """Return cache hit rate (0.0 to 1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    # =========================================================================
    # FAQ Management
    # =========================================================================
    
    def load_faqs(self, faq_queries: List[str]) -> int:
        """
        Load FAQ questions into cache (without responses).
        
        Call warm_faqs() after to generate responses.
        
        Args:
            faq_queries: List of FAQ questions
        
        Returns:
            Number of new FAQs loaded
        """
        return self.cache.load_faqs(faq_queries)
    
    def warm_faqs(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Generate responses for all pending FAQs.
        
        Args:
            verbose: Show progress
        
        Returns:
            Dict with 'warmed' count and 'total_time'
        """
        pending = self.cache.get_pending_faqs()
        
        if not pending:
            if verbose:
                print("✅ All FAQs already have responses!")
            return {'warmed': 0, 'total_time': 0.0}
        
        warmed = 0
        start_time = time.perf_counter()
        
        for i, query in enumerate(pending):
            if verbose:
                print(f"🔥 Warming FAQ [{i+1}/{len(pending)}]: {query[:50]}...")
            
            # Generate response via RAG
            rag_result = self.rag_service.generate(query)
            
            # Update FAQ with response
            self.cache.update_faq_response(query, {
                'answer': rag_result['answer'],
                'evidence_urls': rag_result['evidence_urls']
            })
            warmed += 1
        
        total_time = time.perf_counter() - start_time
        
        if verbose:
            print(f"\n✅ FAQ warming complete!")
            print(f"   Warmed: {warmed}, Time: {total_time:.1f}s")
        
        return {'warmed': warmed, 'total_time': total_time}
    
    def list_faqs(self) -> List[Dict[str, Any]]:
        """List all FAQ entries with their status."""
        return self.cache.list_faqs()
    
    # =========================================================================
    # Generation
    # =========================================================================
    
    def generate(
        self,
        query: str,
        use_cache: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Generate answer with CAG (cache-augmented generation).
        
        Lookup order:
        1. Check FAQs (static, never expire)
        2. Check History (dynamic, 24h TTL)
        3. If miss: run RAG and add to history
        
        Args:
            query: User question
            use_cache: Whether to use cache (default True)
            verbose: Print cache hit/miss logs
        
        Returns:
            Dict with:
            - answer: Generated answer
            - evidence_urls: Source URLs
            - cache_hit: Whether answer came from cache
            - cache_source: 'faq', 'history', or None
            - generation_time: Time taken
            - similarity_score: (semantic mode) similarity of matched query
            - matched_query: (semantic mode) the cached query that matched
        """
        if verbose:
            print(f"🔍 Query: {query}")
            print(f"   Similarity threshold: {self.cache.similarity_threshold}")
        
        # Step 1: Check cache (FAQs first, then history)
        if use_cache:
            lookup_start = time.perf_counter()
            cached = self.cache.get(query)
            lookup_time = time.perf_counter() - lookup_start
            
            if cached:
                self._hits += 1
                source = cached.get('source', 'unknown')
                similarity = cached.get('similarity_score', 1.0)
                matched_query = cached.get('matched_query', query)
                
                # Track source-specific hits
                if source == 'faq':
                    self._faq_hits += 1
                else:
                    self._history_hits += 1
                
                if verbose:
                    source_label = "FAQ" if source == 'faq' else "HISTORY"
                    print(f"📦 From Cache ({source_label})")
                    print(f"   ✅ HIT! (similarity: {similarity:.3f}, lookup: {lookup_time*1000:.1f}ms)")
                    if matched_query.lower().strip() != query.lower().strip():
                        display_query = matched_query[:60] + "..." if len(matched_query) > 60 else matched_query
                        print(f"   Matched: \"{display_query}\"")
                
                return {
                    'answer': cached['answer'],
                    'evidence_urls': cached.get('evidence_urls', []),
                    'cache_hit': True,
                    'cache_source': source,
                    'generation_time': lookup_time,
                    'lookup_time': lookup_time,
                    'similarity_score': similarity,
                    'matched_query': matched_query
                }
            
            self._misses += 1
            if verbose:
                print(f"📚 From Index (cache miss, lookup: {lookup_time*1000:.1f}ms)")
                print(f"   Running RAG retrieval...")
        
        # Step 2: Run RAG
        rag_result = self.rag_service.generate(query)
        
        result = {
            'answer': rag_result['answer'],
            'evidence_urls': rag_result['evidence_urls'],
            'cache_hit': False,
            'cache_source': None,
            'generation_time': rag_result['generation_time'],
            'num_docs': rag_result['num_docs']
        }
        
        # Step 3: Cache to history for future lookups
        if use_cache:
            cache_start = time.perf_counter()
            self.cache.set(query, result)
            cache_time = time.perf_counter() - cache_start
            if verbose:
                print(f"💾 Cached to History ({cache_time*1000:.1f}ms) - next lookup will be instant!")
        
        return result
    
    # =========================================================================
    # History Management
    # =========================================================================
    
    def get_recent_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent queries from history (within TTL).
        
        Args:
            limit: Maximum number to return
        
        Returns:
            List of query info sorted by recency
        """
        return self.cache.get_history_queries(limit)
    
    def warm_cache(self, queries: List[str], verbose: bool = True) -> Dict[str, Any]:
        """
        Pre-populate history cache with queries.
        
        For FAQs, use load_faqs() + warm_faqs() instead.
        
        Args:
            queries: List of queries to pre-cache
            verbose: Show progress
        
        Returns:
            Dict with cached/skipped counts and total_time
        """
        cached_count = 0
        skipped_count = 0
        start_time = time.perf_counter()
        
        for i, query in enumerate(queries):
            if verbose:
                print(f"🔥 Warming cache [{i+1}/{len(queries)}]: {query[:50]}...")
            
            if query not in self.cache:
                self.generate(query, use_cache=True, verbose=False)
                cached_count += 1
            else:
                skipped_count += 1
                if verbose:
                    print("   (already cached)")
        
        total_time = time.perf_counter() - start_time
        
        if verbose:
            print(f"\n✅ Cache warming complete!")
            print(f"   New: {cached_count}, Skipped: {skipped_count}, Time: {total_time:.1f}s")
        
        return {
            'cached': cached_count,
            'skipped': skipped_count,
            'total_time': total_time
        }
    
    # =========================================================================
    # Statistics & Management
    # =========================================================================
    
    def cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics including hit rates by source.
        
        Returns:
            Dict with cache metrics and session stats
        """
        stats = self.cache.stats()
        stats['session_hits'] = self._hits
        stats['session_misses'] = self._misses
        stats['session_faq_hits'] = self._faq_hits
        stats['session_history_hits'] = self._history_hits
        stats['session_hit_rate'] = f"{self.hit_rate:.1%}"
        return stats
    
    def clear_cache(self, clear_faqs: bool = False) -> None:
        """
        Clear cache.
        
        Args:
            clear_faqs: If True, also clear FAQ cache (default False, preserves FAQs)
        """
        self.cache.clear(clear_faqs=clear_faqs)
    
    def reset_stats(self) -> None:
        """Reset session hit/miss counters."""
        self._hits = 0
        self._misses = 0
        self._faq_hits = 0
        self._history_hits = 0


__all__ = ['CAGService']
