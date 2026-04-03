"""
Application configuration - loads from YAML config files.

CONFIGURATION POLICY:
====================
Configuration is loaded from config/config.yaml and config/models.yaml.
Secrets (API keys) live ONLY in .env and are loaded via os.getenv().

Supports multiple LLM providers via OpenRouter unified API.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import os
import yaml

# ========================================
# Project Paths
# ========================================

# Get project root (parent of src/context_engineering/)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"

# ========================================
# YAML Config Loading
# ========================================

def _load_yaml(filename: str) -> Dict[str, Any]:
    """Load a YAML config file."""
    filepath = _CONFIG_DIR / filename
    if not filepath.exists():
        return {}
    with open(filepath, "r") as f:
        return yaml.safe_load(f) or {}


def _get_nested(d: Dict, *keys, default=None):
    """Get nested dictionary value safely."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d if d is not None else default


# Load configs
_CONFIG = _load_yaml("config.yaml")
_MODELS = _load_yaml("models.yaml")

# ========================================
# Provider Configuration
# ========================================

PROVIDER = _get_nested(_CONFIG, "provider", "default", default="openrouter")
MODEL_TIER = _get_nested(_CONFIG, "provider", "tier", default="general")
OPENROUTER_BASE_URL = _get_nested(_CONFIG, "provider", "openrouter_base_url", 
                                   default="https://openrouter.ai/api/v1")

# ========================================
# Model Names (from models.yaml)
# ========================================

def get_chat_model(provider: Optional[str] = None, tier: Optional[str] = None) -> str:
    """Get chat model name for specified provider and tier."""
    provider = provider or PROVIDER
    tier = tier or MODEL_TIER
    return _get_nested(_MODELS, provider, "chat", tier, default="openai/gpt-4o-mini")


def get_embedding_model(provider: Optional[str] = None, tier: str = "default") -> str:
    """Get embedding model name for specified provider and tier."""
    provider = provider or PROVIDER
    return _get_nested(_MODELS, provider, "embedding", tier, default="openai/text-embedding-3-large")


# Default model names (for backward compatibility)
CHAT_MODEL = get_chat_model()
EMBEDDING_MODEL = get_embedding_model()

# Legacy aliases
OPENAI_CHAT_MODEL = CHAT_MODEL  # Backward compatibility

# ========================================
# LLM Defaults
# ========================================

LLM_TEMPERATURE = _get_nested(_CONFIG, "llm", "temperature", default=0.0)
LLM_MAX_TOKENS = _get_nested(_CONFIG, "llm", "max_tokens", default=2000)
LLM_STREAMING = _get_nested(_CONFIG, "llm", "streaming", default=False)

# ========================================
# Embedding Defaults
# ========================================

EMBEDDING_BATCH_SIZE = _get_nested(_CONFIG, "embedding", "batch_size", default=100)
EMBEDDING_SHOW_PROGRESS = _get_nested(_CONFIG, "embedding", "show_progress", default=False)

# ========================================
# Project Paths (from config.yaml)
# ========================================

DATA_DIR = _PROJECT_ROOT / _get_nested(_CONFIG, "paths", "data_dir", default="data")
VECTOR_DIR = _PROJECT_ROOT / _get_nested(_CONFIG, "paths", "vector_dir", default="data/vectorstore")
MARKDOWN_DIR = _PROJECT_ROOT / _get_nested(_CONFIG, "paths", "markdown_dir", default="data/primelands_markdown")
CACHE_DIR = _PROJECT_ROOT / _get_nested(_CONFIG, "paths", "cache_dir", default="data/cag_cache")
CRAWL_OUT_DIR = DATA_DIR  # Root for crawl outputs

# ========================================
# Chunking Configuration
# ========================================

# Fixed-size chunking
FIXED_CHUNK_SIZE = _get_nested(_CONFIG, "chunking", "fixed", "chunk_size", default=800)
FIXED_CHUNK_OVERLAP = _get_nested(_CONFIG, "chunking", "fixed", "chunk_overlap", default=100)

# Semantic chunking
SEMANTIC_MAX_CHUNK_SIZE = _get_nested(_CONFIG, "chunking", "semantic", "max_chunk_size", default=1000)
SEMANTIC_MIN_CHUNK_SIZE = _get_nested(_CONFIG, "chunking", "semantic", "min_chunk_size", default=200)

# Sliding-window chunking
SLIDING_WINDOW_SIZE = _get_nested(_CONFIG, "chunking", "sliding", "window_size", default=512)
SLIDING_STRIDE_SIZE = _get_nested(_CONFIG, "chunking", "sliding", "stride_size", default=256)

# Parent-child chunking
PARENT_CHUNK_SIZE = _get_nested(_CONFIG, "chunking", "parent_child", "parent_size", default=1200)
CHILD_CHUNK_SIZE = _get_nested(_CONFIG, "chunking", "parent_child", "child_size", default=250)
CHILD_OVERLAP = _get_nested(_CONFIG, "chunking", "parent_child", "child_overlap", default=50)

# Late chunking
LATE_CHUNK_BASE_SIZE = _get_nested(_CONFIG, "chunking", "late", "base_size", default=1000)
LATE_CHUNK_SPLIT_SIZE = _get_nested(_CONFIG, "chunking", "late", "split_size", default=300)
LATE_CHUNK_CONTEXT_WINDOW = _get_nested(_CONFIG, "chunking", "late", "context_window", default=150)

# ========================================
# Retrieval Configuration
# ========================================

TOP_K_RESULTS = _get_nested(_CONFIG, "retrieval", "top_k", default=4)
SIMILARITY_THRESHOLD = _get_nested(_CONFIG, "retrieval", "similarity_threshold", default=0.7)

# ========================================
# CAG Configuration
# ========================================

CAG_CACHE_TTL = _get_nested(_CONFIG, "cag", "cache_ttl", default=86400)
CAG_CACHE_MAX_SIZE = _get_nested(_CONFIG, "cag", "max_cache_size", default=1000)
CAG_SIMILARITY_THRESHOLD = _get_nested(_CONFIG, "cag", "similarity_threshold", default=0.90)
CAG_HISTORY_TTL_HOURS = _get_nested(_CONFIG, "cag", "history_ttl_hours", default=24)

# ========================================
# FAQ Loading
# ========================================

def load_faqs() -> list:
    """
    Load known FAQs from config/faqs.yaml.
    
    Returns:
        List of FAQ question strings (flattened from all categories)
    """
    faqs_config = _load_yaml("faqs.yaml")
    all_faqs = []
    
    # Flatten all categories into a single list
    for category, questions in faqs_config.items():
        if isinstance(questions, list):
            all_faqs.extend(questions)
    
    return all_faqs


# Pre-load FAQs for easy access
KNOWN_FAQS = load_faqs()

# ========================================
# CRAG Configuration
# ========================================

CRAG_CONFIDENCE_THRESHOLD = _get_nested(_CONFIG, "crag", "confidence_threshold", default=0.6)
CRAG_EXPANDED_K = _get_nested(_CONFIG, "crag", "expanded_k", default=8)

# ========================================
# Crawling Configuration
# ========================================

CRAWL_MAX_DEPTH = _get_nested(_CONFIG, "crawling", "max_depth", default=3)
CRAWL_DELAY_SECONDS = _get_nested(_CONFIG, "crawling", "delay_seconds", default=2.0)
CRAWL_MAX_PAGES = _get_nested(_CONFIG, "crawling", "max_pages", default=100)

# ========================================
# Helper Functions
# ========================================

def get_api_key(provider: Optional[str] = None) -> Optional[str]:
    """Get API key for the specified provider."""
    provider = provider or PROVIDER
    key_map = {
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    env_var = key_map.get(provider, f"{provider.upper()}_API_KEY")
    return os.getenv(env_var)


def validate() -> None:
    """
    Validate configuration and create required directories.
    
    Raises:
        ValueError: If required secrets are missing
        OSError: If directories cannot be created
    """
    # Check required secrets based on provider
    api_key = get_api_key()
    if not api_key:
        key_name = "OPENROUTER_API_KEY" if PROVIDER == "openrouter" else f"{PROVIDER.upper()}_API_KEY"
        raise ValueError(
            f"❌ Missing required secret: {key_name}\n"
            f"Please add it to your .env file."
        )
    
    # Create required directories
    required_dirs = [DATA_DIR, VECTOR_DIR, MARKDOWN_DIR, CACHE_DIR]
    
    for dir_path in required_dirs:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            raise OSError(f"❌ Cannot create directory {dir_path}: {e}")


def dump() -> None:
    """Print all active non-secret configuration values for debugging."""
    print("\n" + "=" * 60)
    print("CONFIGURATION (NON-SECRETS ONLY)")
    print("=" * 60)
    
    print("\n🌐 Provider:")
    print(f"   Provider: {PROVIDER}")
    print(f"   Model Tier: {MODEL_TIER}")
    print(f"   Chat Model: {CHAT_MODEL}")
    print(f"   Embedding Model: {EMBEDDING_MODEL}")
    
    print("\n📁 Directories:")
    print(f"   Data Root: {DATA_DIR}")
    print(f"   Vector Store: {VECTOR_DIR}")
    print(f"   Markdown: {MARKDOWN_DIR}")
    print(f"   Cache: {CACHE_DIR}")
    
    print("\n🔧 Chunking:")
    print(f"   Fixed Size: {FIXED_CHUNK_SIZE} tokens")
    print(f"   Fixed Overlap: {FIXED_CHUNK_OVERLAP} tokens")
    print(f"   Sliding Window: {SLIDING_WINDOW_SIZE} tokens")
    print(f"   Sliding Stride: {SLIDING_STRIDE_SIZE} tokens")
    print(f"   Parent-Child: {CHILD_CHUNK_SIZE} → {PARENT_CHUNK_SIZE} tokens")
    print(f"   Late Chunk: {LATE_CHUNK_BASE_SIZE} → {LATE_CHUNK_SPLIT_SIZE} tokens")
    
    print("\n🔍 Retrieval:")
    print(f"   Top-K Results: {TOP_K_RESULTS}")
    print(f"   Similarity Threshold: {SIMILARITY_THRESHOLD}")
    
    print("\n💾 CAG:")
    print(f"   Cache TTL: {CAG_CACHE_TTL}s")
    print(f"   Max Cache Size: {CAG_CACHE_MAX_SIZE}")
    
    print("\n🎯 CRAG:")
    print(f"   Confidence Threshold: {CRAG_CONFIDENCE_THRESHOLD}")
    print(f"   Expanded K: {CRAG_EXPANDED_K}")
    
    print("\n" + "=" * 60 + "\n")


def get_all_models() -> Dict[str, Any]:
    """Return all available models from models.yaml."""
    return _MODELS


def get_config() -> Dict[str, Any]:
    """Return full config dictionary."""
    return _CONFIG
