"""
CRAG (Corrective RAG) service with self-correcting retrieval.

Provides:
- CRAGService: Self-correcting RAG with confidence scoring
- Automatic query expansion on low confidence
- Better grounding and reduced hallucinations

Workflow:
    1. Initial retrieval (k=4)
    2. Calculate confidence score
    3. If low: Corrective retrieval (k=8, expanded)
    4. Generate with best evidence

Benefits:
- 🎯 Better accuracy for complex queries
- 🛡️ Reduces hallucinations
- 🔄 Automatic self-correction
"""

from typing import Any, Dict, List
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever

from context_engineering.config import (
    CRAG_CONFIDENCE_THRESHOLD,
    CRAG_EXPANDED_K,
    TOP_K_RESULTS
)
RAG_TEMPLATE = (
    "You are a helpful assistant. Use the provided context to answer the question. "
    "If the answer is not in the context, say you do not know.\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer:"
)


def format_docs(docs: List[Any]) -> str:
    """
    Format documents into a single context string.
    """
    if not docs:
        return ""
    chunks = []
    for doc in docs:
        content = getattr(doc, "page_content", str(doc))
        chunks.append(content.strip())
    return "\n\n".join(chunk for chunk in chunks if chunk)


def calculate_confidence(docs: List[Any], query: str) -> float:
    """
    Estimate confidence based on retrieval density and light query overlap.
    """
    if not docs:
        return 0.0

    total_chars = 0
    overlap_hits = 0
    query_terms = {t for t in query.lower().split() if t}

    for doc in docs:
        text = getattr(doc, "page_content", "") or ""
        total_chars += len(text)
        if query_terms:
            tokens = set(text.lower().split())
            overlap_hits += len(query_terms.intersection(tokens))

    avg_len = total_chars / max(len(docs), 1)
    length_score = min(avg_len / 1000.0, 1.0)
    overlap_score = min(overlap_hits / max(len(query_terms), 1), 1.0)
    return round((0.6 * length_score) + (0.4 * overlap_score), 4)


class CRAGService:
    """
    Corrective RAG service with automatic self-correction.
    
    Features:
    - Initial retrieval with confidence scoring
    - Automatic corrective retrieval if confidence low
    - Query expansion strategies
    - Detailed metrics for debugging
    
    Usage:
        service = CRAGService(retriever, llm)
        result = service.generate(query, confidence_threshold=0.6)
        
        print(result['answer'])
        print(f"Confidence: {result['confidence_final']}")
        print(f"Correction applied: {result['correction_applied']}")
    """
    
    def __init__(
        self,
        retriever: VectorStoreRetriever,
        llm: Any,
        initial_k: int = TOP_K_RESULTS,
        expanded_k: int = CRAG_EXPANDED_K
    ):
        """
        Initialize CRAG service.
        
        Args:
            retriever: Vector store retriever
            llm: LangChain LLM instance
            initial_k: Number of docs for initial retrieval
            expanded_k: Number of docs for corrective retrieval
        """
        self.retriever = retriever
        self.llm = llm
        self.initial_k = initial_k
        self.expanded_k = expanded_k
        
        # Create prompt
        self.prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    
    def generate(
        self,
        query: str,
        confidence_threshold: float = CRAG_CONFIDENCE_THRESHOLD,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Generate answer with CRAG (Corrective RAG).
        
        Workflow:
        1. Initial retrieval (k=initial_k)
        2. Calculate confidence score
        3. If confidence < threshold:
           - Apply corrective retrieval (k=expanded_k)
           - Recalculate confidence
        4. Generate answer with best evidence
        
        Args:
            query: User question
            confidence_threshold: Minimum confidence score (0-1)
            verbose: Print progress logs
        
        Returns:
            Dict with:
            - answer: Generated answer
            - confidence_initial: Initial confidence score
            - confidence_final: Final confidence score
            - correction_applied: Whether corrective retrieval was used
            - docs_used: Number of documents in final generation
            - generation_time: Total time taken
            - evidence_urls: List of source URLs
        """
        if verbose:
            print(f"🔍 Query: {query}")
            print(f"🎯 Confidence threshold: {confidence_threshold}\n")
        
        # Step 1: Initial retrieval
        if verbose:
            print(f"1️⃣  Initial retrieval (k={self.initial_k})...")
        
        self.retriever.search_kwargs["k"] = self.initial_k
        docs_initial = self.retriever.invoke(query)
        confidence_initial = calculate_confidence(docs_initial, query)
        
        if verbose:
            print(f"   📊 Confidence: {confidence_initial:.2f}")
        
        # Step 2: Check if correction needed
        if confidence_initial >= confidence_threshold:
            if verbose:
                print(f"   ✅ Confidence sufficient - proceeding with initial retrieval")
            final_docs = docs_initial
            confidence_final = confidence_initial
            correction_applied = False
        else:
            if verbose:
                print(f"   ⚠️  Low confidence - applying corrective retrieval...\n")
            
            # Step 3: Corrective retrieval
            if verbose:
                print(f"2️⃣  Corrective retrieval (k={self.expanded_k}, expanded)...")
            
            # Expand k for more diverse results
            self.retriever.search_kwargs["k"] = self.expanded_k
            docs_corrected = self.retriever.invoke(query)
            confidence_final = calculate_confidence(docs_corrected, query)
            
            if verbose:
                print(f"   📊 Corrected confidence: {confidence_final:.2f}")
                improvement = (confidence_final - confidence_initial) * 100
                print(f"   📈 Confidence improved by {improvement:.1f}%")
            
            final_docs = docs_corrected
            correction_applied = True
        
        # Step 4: Generate answer
        if verbose:
            print(f"\n3️⃣  Generating answer...")
        
        start = time.time()
        
        # Format docs and generate
        context = format_docs(final_docs)
        prompt_input = {"context": context, "question": query}
        answer = (self.prompt | self.llm | StrOutputParser()).invoke(prompt_input)
        
        elapsed = time.time() - start
        
        # Extract evidence URLs
        evidence_urls = []
        for doc in final_docs:
            url = None
            if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                url = doc.metadata.get("url")
            if url:
                evidence_urls.append(url)
        evidence_urls = list(set(evidence_urls))
        
        return {
            'answer': answer,
            'confidence_initial': confidence_initial,
            'confidence_final': confidence_final,
            'correction_applied': correction_applied,
            'docs_used': len(final_docs),
            'generation_time': elapsed,
            'evidence_urls': evidence_urls,
            'evidence': final_docs
        }
    
    def batch_generate(
        self,
        queries: List[str],
        confidence_threshold: float = CRAG_CONFIDENCE_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        Generate answers for multiple queries with CRAG.
        
        Args:
            queries: List of user questions
            confidence_threshold: Minimum confidence score
        
        Returns:
            List of result dicts (same format as generate())
        """
        results = []
        for query in queries:
            result = self.generate(query, confidence_threshold, verbose=False)
            results.append(result)
        return results
    
    def analyze_confidence(self, query: str) -> Dict[str, Any]:
        """
        Analyze confidence without generating answer (for debugging).
        
        Args:
            query: User question
        
        Returns:
            Dict with confidence metrics
        """
        # Initial retrieval
        self.retriever.search_kwargs["k"] = self.initial_k
        docs_initial = self.retriever.invoke(query)
        confidence_initial = calculate_confidence(docs_initial, query)
        
        # Expanded retrieval
        self.retriever.search_kwargs["k"] = self.expanded_k
        docs_expanded = self.retriever.invoke(query)
        confidence_expanded = calculate_confidence(docs_expanded, query)
        
        return {
            'query': query,
            'confidence_initial': confidence_initial,
            'confidence_expanded': confidence_expanded,
            'improvement': confidence_expanded - confidence_initial,
            'docs_initial': len(docs_initial),
            'docs_expanded': len(docs_expanded)
        }


__all__ = ['CRAGService']

