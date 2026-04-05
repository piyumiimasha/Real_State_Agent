"""
Cache-Augmented Generation (CAG) cache implementation.

Provides:
- CAGCache: Semantic similarity-based cache
- Static FAQs: Predefined questions always in cache
- Dynamic History: User queries with 24-hour TTL

How it works:
- Stores embeddings alongside cached responses
- New queries: embed once, compare against all cached embeddings
- Cached queries: pre-computed embeddings (no re-embedding needed)
- Cosine similarity for matching (simple dot product)

Benefits:
- ⚡ Near-0s latency for cache hits
- 💰 Zero API costs for cached queries
- 🎯 Catches paraphrased questions
- 🪶 Lightweight: only new queries need embedding
- 📋 Two-tier: Static FAQs + Dynamic History (24h TTL)
"""

import hashlib
import pickle
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime


class CAGCache:
    """
    Semantic similarity-based cache with FAQ + History support.
    
    Two-tier caching:
    1. Static FAQs: Predefined questions, never expire
    2. Dynamic History: User queries, configurable TTL (default 24h)
    
    All lookups use cosine similarity between query embeddings.
    
    Usage:
        from context_engineering.infrastructure.llm_providers import get_default_embeddings
        
        embedder = get_default_embeddings()
        cache = CAGCache(
            cache_dir=Path("data/cache"),
            embedder=embedder,
            similarity_threshold=0.90,
            history_ttl_hours=24
        )
        
        # Load FAQs
        cache.load_faqs(["What are the visiting hours?", ...])
        
        # Lookup (semantic matching)
        cached = cache.get("Tell me visiting hours")  # Matches FAQ!
    """
    
    def __init__(
        self,
        cache_dir: Path,
        embedder: Any,
        similarity_threshold: float = 0.90,
        max_cache_size: int = 1000,
        history_ttl_hours: float = 24.0
    ):
        """
        Initialize semantic cache.
        
        Args:
            cache_dir: Directory to store cache files
            embedder: Embedding model (required)
            similarity_threshold: Min similarity for match (0.0-1.0)
                - 0.95+: Very strict (almost exact)
                - 0.90-0.95: Recommended (catches paraphrases)
                - 0.85-0.90: Loose (may match less relevant)
            max_cache_size: Maximum entries in dynamic history
            history_ttl_hours: Hours before history entries expire (default 24)
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.history_ttl_hours = history_ttl_hours
        
        # Cache files
        self.faq_cache_file = cache_dir / "cag_faqs.pkl"
        self.history_cache_file = cache_dir / "cag_history.pkl"
        
        # Two-tier cache structure
        self.faq_cache: Dict[str, Any] = self._load_cache(self.faq_cache_file)
        self.history_cache: Dict[str, Any] = self._load_cache(self.history_cache_file)
        
        # Clean expired history on load
        self._cleanup_expired_history()
        
        # Build embedding matrices for fast lookup
        self._update_faq_embedding_matrix()
        self._update_history_embedding_matrix()
    
    # =========================================================================
    # Core Operations
    # =========================================================================
    
    def _load_cache(self, cache_file: Path) -> Dict[str, Any]:
        """Load cache from disk if exists."""
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_faq_cache(self) -> None:
        """Persist FAQ cache to disk."""
        with open(self.faq_cache_file, 'wb') as f:
            pickle.dump(self.faq_cache, f)
    
    def _save_history_cache(self) -> None:
        """Persist history cache to disk."""
        with open(self.history_cache_file, 'wb') as f:
            pickle.dump(self.history_cache, f)
    
    def _cleanup_expired_history(self) -> None:
        """Remove entries older than TTL from history."""
        cutoff_time = time.time() - (self.history_ttl_hours * 3600)
        expired_keys = [
            key for key, entry in self.history_cache.items()
            if entry.get('timestamp', 0) < cutoff_time
        ]
        
        if expired_keys:
            for key in expired_keys:
                del self.history_cache[key]
            self._save_history_cache()
    
    def _generate_key(self, query: str) -> str:
        """Generate unique key for a query."""
        return hashlib.md5(f"{query}_{time.time()}".encode()).hexdigest()
    
    # =========================================================================
    # Embedding Operations
    # =========================================================================
    
    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query (only call needed for new queries)."""
        embedding = self.embedder.embed_query(query)
        return np.array(embedding)
    
    def _update_faq_embedding_matrix(self) -> None:
        """Build FAQ embedding matrix for fast batch similarity."""
        valid_faqs = {k: v for k, v in self.faq_cache.items() if v.get('has_response')}
        if not valid_faqs:
            self._faq_embedding_matrix = None
            self._faq_cache_ids = []
            return
        
        self._faq_cache_ids = list(valid_faqs.keys())
        embeddings = [valid_faqs[cid]['embedding'] for cid in self._faq_cache_ids]
        self._faq_embedding_matrix = np.vstack(embeddings)
    
    def _update_history_embedding_matrix(self) -> None:
        """Build history embedding matrix for fast batch similarity."""
        self._cleanup_expired_history()
        
        if not self.history_cache:
            self._history_embedding_matrix = None
            self._history_cache_ids = []
            return
        
        self._history_cache_ids = list(self.history_cache.keys())
        embeddings = [self.history_cache[cid]['embedding'] for cid in self._history_cache_ids]
        self._history_embedding_matrix = np.vstack(embeddings)
    
    def _find_similar(
        self, 
        query_embedding: np.ndarray,
        embedding_matrix: Optional[np.ndarray],
        cache_ids: List[str]
    ) -> Optional[Tuple[str, float]]:
        """
        Find most similar entry using cosine similarity.
        
        This is the core "lightweight semantic check" - just a dot product.
        """
        if embedding_matrix is None or len(cache_ids) == 0:
            return None
        
        # Vectorized cosine similarity (efficient NumPy operation)
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        cache_norms = embedding_matrix / (
            np.linalg.norm(embedding_matrix, axis=1, keepdims=True) + 1e-10
        )
        similarities = np.dot(cache_norms, query_norm)
        
        best_idx = np.argmax(similarities)
        best_similarity = float(similarities[best_idx])
        
        if best_similarity >= self.similarity_threshold:
            return (cache_ids[best_idx], best_similarity)
        return None
    
    # =========================================================================
    # FAQ Management
    # =========================================================================
    
    def load_faqs(self, faq_queries: List[str], responses: Optional[List[Dict[str, Any]]] = None) -> int:
        """
        Load static FAQs into cache.
        
        If responses not provided, call warm_faqs() via CAGService later.
        
        Args:
            faq_queries: List of FAQ questions
            responses: Optional list of pre-computed responses
        
        Returns:
            Number of new FAQs loaded
        """
        loaded = 0
        for i, query in enumerate(faq_queries):
            # Check if similar FAQ already exists
            query_embedding = self._embed_query(query)
            existing = self._find_similar(
                query_embedding, 
                self._faq_embedding_matrix, 
                self._faq_cache_ids
            )
            
            if existing and existing[1] > 0.95:  # Very similar FAQ exists
                continue
            
            key = self._generate_key(query)
            entry = {
                'query': query,
                'embedding': query_embedding,
                'is_faq': True,
                'timestamp': time.time()
            }
            
            # Add response if provided
            if responses and i < len(responses):
                entry['answer'] = responses[i].get('answer', '')
                entry['evidence_urls'] = responses[i].get('evidence_urls', [])
                entry['has_response'] = True
            else:
                entry['has_response'] = False
            
            self.faq_cache[key] = entry
            loaded += 1
        
        if loaded > 0:
            self._save_faq_cache()
            self._update_faq_embedding_matrix()
        
        return loaded
    
    def get_pending_faqs(self) -> List[str]:
        """Get FAQ queries that don't have responses yet."""
        return [
            entry['query'] for entry in self.faq_cache.values()
            if not entry.get('has_response', False)
        ]
    
    def update_faq_response(self, query: str, response: Dict[str, Any]) -> bool:
        """
        Update response for an FAQ entry.
        
        Args:
            query: The FAQ question
            response: Dict with 'answer' and 'evidence_urls'
        
        Returns:
            True if updated, False if FAQ not found
        """
        # Find the FAQ by semantic similarity
        query_embedding = self._embed_query(query)
        match = self._find_similar(
            query_embedding,
            self._faq_embedding_matrix if self._faq_embedding_matrix is not None else None,
            self._faq_cache_ids
        )
        
        # Also check pending FAQs (no embedding matrix yet)
        for key, entry in self.faq_cache.items():
            if entry['query'].lower().strip() == query.lower().strip():
                self.faq_cache[key]['answer'] = response['answer']
                self.faq_cache[key]['evidence_urls'] = response.get('evidence_urls', [])
                self.faq_cache[key]['has_response'] = True
                self.faq_cache[key]['timestamp'] = time.time()
                self._save_faq_cache()
                self._update_faq_embedding_matrix()
                return True
        
        if match:
            key = match[0]
            self.faq_cache[key]['answer'] = response['answer']
            self.faq_cache[key]['evidence_urls'] = response.get('evidence_urls', [])
            self.faq_cache[key]['has_response'] = True
            self.faq_cache[key]['timestamp'] = time.time()
            self._save_faq_cache()
            self._update_faq_embedding_matrix()
            return True
        
        return False
    
    def list_faqs(self) -> List[Dict[str, Any]]:
        """List all FAQ entries with their status."""
        return [
            {
                'query': entry['query'],
                'has_response': entry.get('has_response', False),
                'timestamp': datetime.fromtimestamp(entry['timestamp']).isoformat()
            }
            for entry in self.faq_cache.values()
        ]
    
    # =========================================================================
    # Public Interface
    # =========================================================================
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response using semantic similarity.
        
        Lookup order: FAQs first → History second
        
        Args:
            query: User query
        
        Returns:
            Cached response with 'source', 'similarity_score', 'matched_query'
            or None if no match
        """
        # Clean expired history
        self._cleanup_expired_history()
        
        # Embed the query (only embedding call needed)
        query_embedding = self._embed_query(query)
        
        # 1. Check FAQ cache first (higher priority)
        faq_match = self._find_similar(
            query_embedding,
            self._faq_embedding_matrix,
            self._faq_cache_ids
        )
        
        if faq_match:
            cache_id, similarity = faq_match
            cached = self.faq_cache[cache_id].copy()
            cached.pop('embedding', None)
            cached['similarity_score'] = similarity
            cached['matched_query'] = cached['query']
            cached['source'] = 'faq'
            return cached
        
        # 2. Check history cache
        # Rebuild matrix in case of expirations
        self._update_history_embedding_matrix()
        
        history_match = self._find_similar(
            query_embedding,
            self._history_embedding_matrix,
            self._history_cache_ids
        )
        
        if history_match:
            cache_id, similarity = history_match
            entry = self.history_cache[cache_id]
            
            # Double-check TTL
            if time.time() - entry['timestamp'] < self.history_ttl_hours * 3600:
                cached = entry.copy()
                cached.pop('embedding', None)
                cached['similarity_score'] = similarity
                cached['matched_query'] = cached['query']
                cached['source'] = 'history'
                return cached
        
        return None
    
    def set(self, query: str, response: Dict[str, Any]) -> None:
        """
        Cache a response to history.
        
        Args:
            query: User query
            response: Dict with 'answer' and optionally 'evidence_urls'
        """
        key = self._generate_key(query)
        embedding = self._embed_query(query)
        
        self.history_cache[key] = {
            'query': query,
            'embedding': embedding,
            'answer': response['answer'],
            'evidence_urls': response.get('evidence_urls', []),
            'timestamp': time.time(),
            'is_faq': False
        }
        
        # Evict oldest if over limit
        if len(self.history_cache) > self.max_cache_size:
            oldest_key = min(
                self.history_cache.keys(),
                key=lambda k: self.history_cache[k]['timestamp']
            )
            del self.history_cache[oldest_key]
        
        self._update_history_embedding_matrix()
        self._save_history_cache()
    
    def clear(self, clear_faqs: bool = False) -> None:
        """
        Clear cache.
        
        Args:
            clear_faqs: If True, also clear FAQ cache (default False)
        """
        self.history_cache = {}
        self._history_embedding_matrix = None
        self._history_cache_ids = []
        self._save_history_cache()
        
        if clear_faqs:
            self.faq_cache = {}
            self._faq_embedding_matrix = None
            self._faq_cache_ids = []
            self._save_faq_cache()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        faq_size = self.faq_cache_file.stat().st_size if self.faq_cache_file.exists() else 0
        history_size = self.history_cache_file.stat().st_size if self.history_cache_file.exists() else 0
        
        faqs_ready = sum(1 for e in self.faq_cache.values() if e.get('has_response'))
        faqs_pending = len(self.faq_cache) - faqs_ready
        
        self._cleanup_expired_history()
        
        return {
            'total_cached': len(self.faq_cache) + len(self.history_cache),
            'faq_count': len(self.faq_cache),
            'faq_ready': faqs_ready,
            'faq_pending': faqs_pending,
            'history_count': len(self.history_cache),
            'history_ttl_hours': self.history_ttl_hours,
            'similarity_threshold': self.similarity_threshold,
            'cache_size_kb': (faq_size + history_size) / 1024
        }
    
    def get_history_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent queries from history (within TTL).
        
        Args:
            limit: Maximum number to return
        
        Returns:
            List of query info sorted by recency
        """
        self._cleanup_expired_history()
        
        entries = [
            {
                'query': entry['query'],
                'timestamp': datetime.fromtimestamp(entry['timestamp']).isoformat(),
                'age_hours': (time.time() - entry['timestamp']) / 3600
            }
            for entry in self.history_cache.values()
        ]
        
        entries.sort(key=lambda x: x['age_hours'])
        return entries[:limit]
    
    def __len__(self) -> int:
        return len(self.faq_cache) + len(self.history_cache)
    
    def __contains__(self, query: str) -> bool:
        return self.get(query) is not None


__all__ = ['CAGCache']
