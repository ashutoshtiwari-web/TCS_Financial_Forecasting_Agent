"""LLM/embedding factory.

Centralizes provider selection so the rest of the codebase can ask for a chat model
or embeddings without knowing whether the backend is Ollama or OpenAI.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from app.config import settings


def get_chat_model(temperature: float = 0) -> BaseChatModel:
    """Return a configured chat model for synthesis/extraction prompts."""
    provider = settings.llm_provider.lower().strip()
    if provider == "openai":
        # Import lazily so local Ollama-only installs don't require OpenAI deps at import time.
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured")
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=temperature,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")


def get_embeddings() -> Embeddings:
    """Return embeddings model used by Chroma for qualitative RAG retrieval."""
    provider = settings.llm_provider.lower().strip()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured")
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
