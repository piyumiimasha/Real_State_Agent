"""
Text chunking strategies for document ingestion.

Provides 5 chunking strategies:
1. Semantic/Heading-Aware - Split by document structure
2. Fixed-Window - Uniform chunks with overlap
3. Sliding-Window - Overlapping windows for better recall
4. Parent-Child (Two-Tier) - Small children with large parent context
5. Query-Focused Late Chunking - Large base passages, split on retrieval

All strategies use configuration from context_engineering.config
"""

from typing import List, Dict, Any, Tuple, Iterable
from functools import lru_cache
import tiktoken

from context_engineering.config import (
    FIXED_CHUNK_SIZE,
    FIXED_CHUNK_OVERLAP,
    SEMANTIC_MAX_CHUNK_SIZE,
    SEMANTIC_MIN_CHUNK_SIZE,
    SLIDING_WINDOW_SIZE,
    SLIDING_STRIDE_SIZE,
    PARENT_CHUNK_SIZE,
    CHILD_CHUNK_SIZE,
    CHILD_OVERLAP,
    LATE_CHUNK_BASE_SIZE,
    LATE_CHUNK_SPLIT_SIZE,
    LATE_CHUNK_CONTEXT_WINDOW
)


# ============================================================================
# Utility Functions
# ============================================================================

@lru_cache(maxsize=4)
def _get_encoding(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text using tiktoken."""
    encoding = _get_encoding(model)
    return len(encoding.encode(text))


def _split_tokens(
    text: str,
    chunk_size: int,
    overlap: int,
    model: str = "gpt-4"
) -> Iterable[str]:
    if chunk_size <= 0:
        return []
    encoding = _get_encoding(model)
    tokens = encoding.encode(text)
    step = max(chunk_size - overlap, 1)
    chunks = []
    for start in range(0, len(tokens), step):
        end = min(start + chunk_size, len(tokens))
        if start >= end:
            break
        chunk_tokens = tokens[start:end]
        chunks.append(encoding.decode(chunk_tokens))
        if end == len(tokens):
            break
    return chunks


# ============================================================================
# 1. SEMANTIC / HEADING-AWARE CHUNKING
# ============================================================================

def semantic_chunk(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split documents by markdown heading structure.
    
    Use when: Documents have clear heading hierarchy
    Pros: Preserves topic coherence
    Cons: Variable chunk sizes
    
    Args:
        documents: List of dicts with 'url', 'title', 'content'
    
    Returns:
        List of chunk dicts with 'url', 'title', 'text', 'strategy', 'chunk_index'
    """
    chunks = []
    chunk_idx = 0
    
    # Define heading hierarchy
    headers_to_split = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    
    try:
        from langchain_text_splitters import MarkdownHeaderTextSplitter
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split,
            strip_headers=False
        )
    except Exception:
        splitter = None
    
    for doc in documents:
        content = doc['content']
        url = doc['url']
        title = doc['title']
        
        try:
            # Split by headings if available
            sections = splitter.split_text(content) if splitter else []
            
            if not sections:
                # No headings, use full content
                sections = [type('obj', (object,), {'page_content': content, 'metadata': {}})()]
            
            for section in sections:
                text = section.page_content.strip()
                
                if not text or len(text) < SEMANTIC_MIN_CHUNK_SIZE:
                    continue
                
                # If section too large, split further by tokens
                if count_tokens(text) > SEMANTIC_MAX_CHUNK_SIZE:
                    sub_chunks = _split_tokens(
                        text,
                        SEMANTIC_MAX_CHUNK_SIZE,
                        overlap=max(SEMANTIC_MIN_CHUNK_SIZE // 10, 20)
                    )
                    
                    for sub_text in sub_chunks:
                        if sub_text.strip():
                            chunks.append({
                                "url": url,
                                "title": title,
                                "text": sub_text.strip(),
                                "strategy": "semantic",
                                "chunk_index": chunk_idx,
                                "heading": section.metadata.get('h1', '') or section.metadata.get('h2', ''),
                                "token_count": count_tokens(sub_text)
                            })
                            chunk_idx += 1
                else:
                    chunks.append({
                        "url": url,
                        "title": title,
                        "text": text,
                        "strategy": "semantic",
                        "chunk_index": chunk_idx,
                        "heading": section.metadata.get('h1', '') or section.metadata.get('h2', ''),
                        "token_count": count_tokens(text)
                    })
                    chunk_idx += 1
                    
        except Exception as e:
            # Fallback: treat as single chunk
            if content.strip():
                chunks.append({
                    "url": url,
                    "title": title,
                    "text": content.strip(),
                    "strategy": "semantic",
                    "chunk_index": chunk_idx,
                    "heading": "",
                    "token_count": count_tokens(content)
                })
                chunk_idx += 1
    
    return chunks


# ============================================================================
# 2. FIXED-WINDOW CHUNKING
# ============================================================================

def fixed_chunk(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split documents into fixed-size chunks with overlap.
    
    Use when: Need predictable chunk sizes for embedding
    Pros: Uniform sizes, simple
    Cons: Breaks semantic boundaries
    
    Args:
        documents: List of dicts with 'url', 'title', 'content'
    
    Returns:
        List of chunk dicts with 'url', 'title', 'text', 'strategy', 'chunk_index'
    """
    chunks = []
    chunk_idx = 0
    
    for doc in documents:
        content = doc['content']
        url = doc['url']
        title = doc['title']

        doc_chunks = _split_tokens(
            content,
            FIXED_CHUNK_SIZE,
            FIXED_CHUNK_OVERLAP
        )

        for text in doc_chunks:
            if text.strip():
                token_count = count_tokens(text)
                chunks.append({
                    "url": url,
                    "title": title,
                    "text": text.strip(),
                    "strategy": "fixed",
                    "chunk_index": chunk_idx,
                    "token_count": token_count,
                    "overlap_tokens": FIXED_CHUNK_OVERLAP
                })
                chunk_idx += 1
    
    return chunks


# ============================================================================
# 3. SLIDING-WINDOW CHUNKING
# ============================================================================

def sliding_chunk(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create overlapping sliding windows for better recall.
    
    Use when: Need better coverage of content
    Pros: Better recall, no missed boundaries
    Cons: More chunks (index bloat)
    
    Args:
        documents: List of dicts with 'url', 'title', 'content'
    
    Returns:
        List of chunk dicts with 'url', 'title', 'text', 'strategy', 'chunk_index'
    """
    chunks = []
    chunk_idx = 0
    
    for doc in documents:
        content = doc['content']
        url = doc['url']
        title = doc['title']

        encoding = _get_encoding("gpt-4")
        tokens = encoding.encode(content)
        window_idx = 0
        start = 0
        while start < len(tokens):
            end = min(start + SLIDING_WINDOW_SIZE, len(tokens))
            window_tokens = tokens[start:end]
            window_text = encoding.decode(window_tokens)

            if window_text.strip():
                chunks.append({
                    "url": url,
                    "title": title,
                    "text": window_text.strip(),
                    "strategy": "sliding",
                    "chunk_index": chunk_idx,
                    "window_index": window_idx,
                    "token_count": len(window_tokens),
                    "overlap_tokens": max(SLIDING_WINDOW_SIZE - SLIDING_STRIDE_SIZE, 0) if window_idx > 0 else 0
                })
                chunk_idx += 1
                window_idx += 1

            start += max(SLIDING_STRIDE_SIZE, 1)
            if start >= len(tokens):
                break
    
    return chunks


# ============================================================================
# 4. PARENT-CHILD (TWO-TIER) CHUNKING
# ============================================================================

def parent_child_chunk(documents: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create parent-child chunk pairs for precise retrieval with rich context.
    
    How it works:
    1. Split document into large "parent" chunks (1200 tokens)
    2. Within each parent, create small "child" chunks (250 tokens)
    3. Store children in index with parent_id reference
    4. On retrieval: fetch children, return parent context to LLM
    
    Use when: Want precise retrieval but rich context for generation
    Pros: Best of both worlds - precision + context
    Cons: More complex retrieval logic needed
    
    Returns:
        Tuple of (children_chunks, parent_chunks)
        Children have 'parent_id' field linking to parent
    """
    parent_chunks = []
    child_chunks = []
    parent_idx = 0
    child_idx = 0
    
    parent_overlap = max(PARENT_CHUNK_SIZE // 10, 50)
    
    for doc in documents:
        content = doc['content']
        url = doc['url']
        title = doc['title']
        
        # Create parent chunks
        parent_texts = _split_tokens(
            content,
            PARENT_CHUNK_SIZE,
            parent_overlap
        )
        
        for parent_text in parent_texts:
            if not parent_text.strip():
                continue
            
            parent_id = f"{url}::parent::{parent_idx}"
            
            # Store parent
            parent_chunks.append({
                "parent_id": parent_id,
                "url": url,
                "title": title,
                "text": parent_text.strip(),
                "strategy": "parent",
                "chunk_index": parent_idx,
                "token_count": count_tokens(parent_text)
            })
            
            # Create children within this parent
            child_texts = _split_tokens(
                parent_text,
                CHILD_CHUNK_SIZE,
                CHILD_OVERLAP
            )
            
            for child_text in child_texts:
                if child_text.strip():
                    child_chunks.append({
                        "child_id": f"{parent_id}::child::{child_idx}",
                        "parent_id": parent_id,
                        "url": url,
                        "title": title,
                        "text": child_text.strip(),
                        "strategy": "child",
                        "chunk_index": child_idx,
                        "token_count": count_tokens(child_text)
                    })
                    child_idx += 1
            
            parent_idx += 1
    
    return child_chunks, parent_chunks


# ============================================================================
# 5. QUERY-FOCUSED LATE CHUNKING
# ============================================================================

def late_chunk_index(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create large base passages for indexing (split on retrieval).
    
    How it works:
    1. Index large passages (1000 tokens)
    2. On retrieval, split near query matches into smaller chunks
    3. Provides tighter quotes without exploding index size
    
    Use when: Need precision without pre-micro-chunking everything
    Pros: Smaller index, better match density
    Cons: Requires custom retrieval logic
    
    Returns:
        List of base passage chunks (to be split later on query)
    """
    chunks = []
    chunk_idx = 0
    
    base_overlap = max(LATE_CHUNK_BASE_SIZE // 10, 50)
    
    for doc in documents:
        content = doc['content']
        url = doc['url']
        title = doc['title']
        
        # Create base passages
        passages = _split_tokens(
            content,
            LATE_CHUNK_BASE_SIZE,
            base_overlap
        )
        
        for passage in passages:
            if passage.strip():
                chunks.append({
                    "url": url,
                    "title": title,
                    "text": passage.strip(),
                    "strategy": "late_chunk_base",
                    "chunk_index": chunk_idx,
                    "token_count": count_tokens(passage),
                    "splittable": True  # Mark for late splitting
                })
                chunk_idx += 1
    
    return chunks


def late_chunk_split(passage: str, query: str) -> List[Dict[str, Any]]:
    """
    Split a base passage near query matches for precise retrieval.
    
    This is called at RETRIEVAL TIME, not indexing time.
    
    Args:
        passage: The base passage text
        query: User query
    
    Returns:
        List of smaller chunks around query matches
    """
    # Find query term positions
    query_terms = query.lower().split()
    passage_lower = passage.lower()
    
    # Find all match positions
    match_positions = []
    for term in query_terms:
        pos = 0
        while True:
            pos = passage_lower.find(term, pos)
            if pos == -1:
                break
            match_positions.append(pos)
            pos += len(term)
    
    if not match_positions:
        # No matches, return full passage as one chunk
        return [{"text": passage, "score": 0.0}]
    
    # Create chunks around matches
    chunks = []
    context_chars = LATE_CHUNK_CONTEXT_WINDOW * 4
    split_size_chars = LATE_CHUNK_SPLIT_SIZE * 4
    
    for match_pos in match_positions:
        # Extract context around match
        start = max(0, match_pos - context_chars)
        end = min(len(passage), match_pos + split_size_chars)
        
        chunk_text = passage[start:end].strip()
        
        # Calculate relevance score (proximity to query)
        score = 1.0 if match_pos in match_positions else 0.5
        
        chunks.append({
            "text": chunk_text,
            "match_position": match_pos,
            "score": score
        })
    
    # Deduplicate overlapping chunks
    unique_chunks = []
    seen_texts = set()
    for chunk in sorted(chunks, key=lambda x: x['score'], reverse=True):
        if chunk['text'] not in seen_texts:
            unique_chunks.append(chunk)
            seen_texts.add(chunk['text'])
    
    return unique_chunks[:5]  # Return top 5 relevant splits


# ============================================================================
# Chunking Service Class
# ============================================================================

class ChunkingService:
    """
    Unified service for all chunking strategies.
    
    Usage:
        service = ChunkingService()
        chunks = service.chunk(documents, strategy="semantic")
    """
    
    def __init__(self):
        self.strategies = {
            "semantic": semantic_chunk,
            "fixed": fixed_chunk,
            "sliding": sliding_chunk,
            "parent_child": parent_child_chunk,
            "late_chunk": late_chunk_index
        }
    
    def chunk(
        self,
        documents: List[Dict[str, Any]],
        strategy: str = "semantic"
    ) -> Any:
        """
        Chunk documents using specified strategy.
        
        Args:
            documents: List of document dicts
            strategy: One of 'semantic', 'fixed', 'sliding', 'parent_child', 'late_chunk'
        
        Returns:
            List of chunks (or tuple for parent_child)
        """
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(self.strategies.keys())}")
        
        return self.strategies[strategy](documents)
    
    def available_strategies(self) -> List[str]:
        """Return list of available chunking strategies."""
        return list(self.strategies.keys())


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "semantic_chunk",
    "fixed_chunk",
    "sliding_chunk",
    "parent_child_chunk",
    "late_chunk_index",
    "late_chunk_split",
    "ChunkingService",
    "count_tokens"
]

