"""
Chat service for RAG-based question answering.

Provides:
- RAGService: Standard RAG with modern LCEL
- CAGService: Cache-Augmented Generation
- CRAGService: Corrective RAG with confidence scoring
- CAGCache: Semantic similarity cache with FAQs + History (24h TTL)
"""

from .rag_service import RAGService, build_rag_chain
from .cag_cache import CAGCache
from .cag_service import CAGService
from .crag_service import CRAGService

__all__ = [
    # RAG
    "RAGService",
    "build_rag_chain",
    # Caching
    "CAGCache",
    "CAGService",
    # Corrective RAG
    "CRAGService"
]
