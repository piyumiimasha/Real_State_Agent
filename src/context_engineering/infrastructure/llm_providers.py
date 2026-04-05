"""
LLM provider factories for OpenAI-compatible APIs.

Supports OpenRouter, Groq, DeepSeek, and OpenAI by configuring base URLs
and API keys from context_engineering.config.
"""

from typing import Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from context_engineering.config import (
	PROVIDER,
	CHAT_MODEL,
	EMBEDDING_MODEL,
	LLM_TEMPERATURE,
	LLM_MAX_TOKENS,
	LLM_STREAMING,
	EMBEDDING_BATCH_SIZE,
	EMBEDDING_SHOW_PROGRESS,
	OPENROUTER_BASE_URL,
	get_api_key,
)


def _resolve_base_url(provider: str, override: Optional[str] = None) -> Optional[str]:
	if override:
		return override
	provider = (provider or "").lower()
	if provider == "openrouter":
		return OPENROUTER_BASE_URL
	if provider == "groq":
		return "https://api.groq.com/openai/v1"
	if provider == "deepseek":
		return "https://api.deepseek.com/v1"
	return None


def _require_api_key(provider: str) -> str:
	api_key = get_api_key(provider)
	if not api_key:
		raise ValueError(
			f"❌ Missing API key for provider '{provider}'. "
			f"Set {provider.upper()}_API_KEY in .env"
		)
	return api_key


def get_chat_llm(
	temperature: Optional[float] = None,
	max_tokens: Optional[int] = None,
	streaming: Optional[bool] = None,
	provider: Optional[str] = None,
	model: Optional[str] = None,
	base_url: Optional[str] = None,
) -> ChatOpenAI:
	"""Return a ChatOpenAI-compatible LLM instance based on config."""
	provider = (provider or PROVIDER).lower()
	model = model or CHAT_MODEL
	api_key = _require_api_key(provider)
	resolved_base_url = _resolve_base_url(provider, base_url)

	return ChatOpenAI(
		model=model,
		api_key=api_key,
		base_url=resolved_base_url,
		temperature=LLM_TEMPERATURE if temperature is None else temperature,
		max_tokens=LLM_MAX_TOKENS if max_tokens is None else max_tokens,
		streaming=LLM_STREAMING if streaming is None else streaming,
	)


def get_default_embeddings(
	provider: Optional[str] = None,
	model: Optional[str] = None,
	base_url: Optional[str] = None,
) -> OpenAIEmbeddings:
	"""Return OpenAI-compatible embeddings client based on config."""
	provider = (provider or PROVIDER).lower()
	model = model or EMBEDDING_MODEL
	api_key = _require_api_key(provider)
	resolved_base_url = _resolve_base_url(provider, base_url)

	return OpenAIEmbeddings(
		model=model,
		api_key=api_key,
		base_url=resolved_base_url,
		chunk_size=EMBEDDING_BATCH_SIZE,
		show_progress_bar=EMBEDDING_SHOW_PROGRESS,
	)


__all__ = ["get_chat_llm", "get_default_embeddings"]
