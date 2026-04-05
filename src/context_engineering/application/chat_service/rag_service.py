"""
RAG (Retrieval-Augmented Generation) service using modern LangChain LCEL.

Provides:
- build_rag_chain: Create modern LCEL RAG chain
- RAGService: High-level RAG orchestration class
- Uses LangChain Expression Language (Runnables + | operator)

Architecture:
    Query → Retriever → Format Docs → Prompt → LLM → Parse → Answer
    
Modern LCEL approach (NOT legacy chains):
    rag_chain = (
        RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
    )
"""

from typing import Any, Dict, Iterable, List, Tuple
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, Runnable
from langchain_core.vectorstores import VectorStoreRetriever

from context_engineering.config import TOP_K_RESULTS


RAG_TEMPLATE = """
You are a real estate assistant for Primelands. Answer the user's question using
only the provided sources. If the answer is not in the sources, say you do not
have enough information.

Use inline citations with source numbers in square brackets like [1] or [2].
Each statement that depends on sources must include a citation.

User question:
{question}

Sources:
{context}

Answer:
""".strip()


def _normalize_text(text: str, max_chars: int = 600) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) > max_chars:
        return cleaned[: max_chars - 3].rstrip() + "..."
    return cleaned


def _doc_metadata(doc: Any) -> Dict[str, Any]:
    metadata = getattr(doc, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _doc_title(metadata: Dict[str, Any]) -> str:
    for key in ("title", "heading", "property_name", "name"):
        value = metadata.get(key)
        if value:
            return str(value)
    return "Untitled"


def _doc_url(metadata: Dict[str, Any]) -> str:
    for key in ("url", "source", "link"):
        value = metadata.get(key)
        if value:
            return str(value)
    return ""


def format_docs(docs: Iterable[Any]) -> str:
    """
    Format retrieved docs into a numbered source list for inline citations.
    """
    lines: List[str] = []
    for idx, doc in enumerate(docs, start=1):
        metadata = _doc_metadata(doc)
        title = _doc_title(metadata)
        url = _doc_url(metadata)
        snippet = _normalize_text(getattr(doc, "page_content", ""))
        header = f"[{idx}] {title}"
        if url:
            header = f"{header} - {url}"
        lines.append(header)
        if snippet:
            lines.append(f"Snippet: {snippet}")
        lines.append("")
    return "\n".join(lines).strip()


def build_rag_chain(
    retriever: VectorStoreRetriever,
    llm: Any,
    k: int = TOP_K_RESULTS,
    template: str = RAG_TEMPLATE
) -> Runnable:
    """
    Build modern RAG chain using LangChain Expression Language (LCEL).
    
    This uses Runnables and the | operator - the MODERN LangChain way.
    NO legacy chains (RetrievalQA, create_stuff_documents_chain, etc.)
    
    Chain structure:
        1. RunnableParallel: Retrieves docs + passes question through
        2. format_docs: Converts docs to context string
        3. Prompt: Fills template with context + question
        4. LLM: Generates answer
        5. StrOutputParser: Extracts string from LLM response
    
    Args:
        retriever: VectorStore retriever (from vectorstore.as_retriever())
        llm: LangChain LLM instance (ChatOpenAI, etc.)
        k: Number of docs to retrieve (default from config)
        template: Prompt template string
    
    Returns:
        Runnable chain that can be invoked with query string
    
    Usage:
        chain = build_rag_chain(retriever, llm)
        answer = chain.invoke("What are the cardiology services?")
    """
    # Update retriever k if specified
    if k != TOP_K_RESULTS:
        retriever.search_kwargs["k"] = k
    
    # Create prompt template (Runnable)
    rag_prompt = ChatPromptTemplate.from_template(template)
    
    # BUILD THE CHAIN (Modern LCEL approach!)
    rag_chain = (
        RunnableParallel(
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
        )
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


class RAGService:
    """
    High-level RAG service for question answering.
    
    Encapsulates:
    - RAG chain management
    - Retrieval + generation
    - Evidence tracking
    - Timing metrics
    
    Usage:
        service = RAGService(retriever, llm)
        result = service.generate(query)
        print(result['answer'])
        print(result['evidence_urls'])
    """
    
    def __init__(
        self,
        retriever: VectorStoreRetriever,
        llm: Any,
        k: int = TOP_K_RESULTS
    ):
        """
        Initialize RAG service.
        
        Args:
            retriever: Vector store retriever
            llm: LangChain LLM instance
            k: Number of documents to retrieve
        """
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.chain = build_rag_chain(retriever, llm, k)
    
    def generate(self, query: str) -> Dict[str, Any]:
        """
        Generate answer for query using RAG.
        
        Args:
            query: User question
        
        Returns:
            Dict with:
            - answer: Generated answer string
            - evidence: List of retrieved documents
            - evidence_urls: List of unique source URLs
            - generation_time: Seconds taken
        """
        start = time.time()
        
        # Retrieve evidence
        evidence = self.retriever.invoke(query)
        
        # Generate answer
        answer = self.chain.invoke(query)
        
        elapsed = time.time() - start
        
        # Extract unique URLs
        evidence_urls = []
        for doc in evidence:
            metadata = _doc_metadata(doc)
            url = _doc_url(metadata)
            if url:
                evidence_urls.append(url)
        evidence_urls = list(dict.fromkeys(evidence_urls))
        
        return {
            "answer": answer,
            "evidence": evidence,
            "evidence_urls": evidence_urls,
            "generation_time": elapsed,
            "num_docs": len(evidence)
        }
    
    def stream(self, query: str):
        """
        Stream answer generation (for real-time UI).
        
        Args:
            query: User question
        
        Yields:
            String chunks as they're generated
        """
        for chunk in self.chain.stream(query):
            yield chunk
    
    def batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Generate answers for multiple queries in batch.
        
        Args:
            queries: List of user questions
        
        Returns:
            List of result dicts (same format as generate())
        """
        results = []
        for query in queries:
            results.append(self.generate(query))
        return results


__all__ = ["build_rag_chain", "RAGService", "format_docs", "RAG_TEMPLATE"]

