"""Shared model factory for LLM and embedding providers."""

from __future__ import annotations

from functools import lru_cache
import os
from typing import List

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient


load_dotenv()


def get_qdrant_client() -> QdrantClient:
    _require_env('QDRANT_URL')
    return QdrantClient(url=os.environ['QDRANT_URL'])


def get_llm(temperature: float = 0) -> BaseChatModel:
    """Build the configured chat model for inference.

    Input:
        Receives the desired sampling temperature for the selected provider.
    Output:
        Returns a LangChain `BaseChatModel` implementation backed by OpenAI
        or Google Gemini.
    Security context:
        Reads provider credentials from environment variables loaded via
        `python-dotenv` and never embeds API keys in source code.
    """

    provider, model_type = _get_model_provider(), _get_model_type()
    return _get_llm(provider, model_type, temperature=temperature)


def get_fastllm(temperature: float = 0) -> BaseChatModel:
    provider = os.environ.get('FAST_LLM_PROVIDER') or _get_model_provider()
    model_type = os.environ.get('FAST_LLM_MODEL') or _get_model_type()
    return _get_llm(provider, model_type, temperature=temperature)


def get_synthesisllm(temperature: float = 0) -> BaseChatModel:
    provider = os.environ.get('SYNTHESIS_LLM_PROVIDER') or _get_model_provider()
    model_type = os.environ.get('SYNTHESIS_LLM_MODEL') or _get_model_type()
    return _get_llm(provider, model_type, temperature=temperature)


def get_embeddings() -> Embeddings:
    """Build the configured embedding model for indexing and retrieval.

    Input:
        Uses the `MODEL_PROVIDER` environment variable to select the provider.
    Output:
        Returns a LangChain `Embeddings` implementation for OpenAI or Google.
    Security context:
        Reads API credentials from environment variables so embedding access
        remains provider-scoped and no secrets are committed to the repository.
    """

    provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower() or _get_model_provider()
    embedding_model = os.getenv("EMBEDDING_MODEL", "").strip().lower()

    return _get_embeddings(provider, model=embedding_model)


@lru_cache(maxsize=1)
def get_embedding_dimension() -> int:
    """Probe the active embedding model and return its vector dimension.

    Input:
        Uses the currently configured embedding provider and model from the
        environment.
    Output:
        Returns the length of a single embedded probe query vector.
    Security context:
        Issues exactly one embedding request per process and caches the result,
        avoiding hard-coded vector dimensions that could drift from the active
        provider configuration.
    """

    probe_vector = get_embeddings().embed_query("Determining dimension.")
    return len(probe_vector)


def get_collection_name(prefix: str = "vuln_explorer") -> str:
    """Build a provider-scoped collection name for Qdrant resources.

    Input:
        Accepts an optional collection name prefix from application
        configuration.
    Output:
        Returns a collection name suffixed with the active model provider.
    Security context:
        Keeps provider-specific vector indexes isolated so embeddings from
        different backends are not mixed in the same collection.
    """

    provider = _get_model_provider()
    return f'{prefix}_{provider}'


def _get_llm(provider: str, model: str, temperature: float = 0):
    provider = provider.lower().strip()
    if provider == "openai":
        _require_env("OPENAI_API_KEY")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.environ["OPENAI_API_KEY"],
        )
    elif provider == "google":
        _require_env("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=os.environ["GOOGLE_API_KEY"],
        )
    else:
        raise ValueError(f"Unsupported MODEL_PROVIDER value: {provider}")


def _get_model_provider() -> str:
    """Return the normalized model provider from environment configuration.

    Input:
        Reads `MODEL_PROVIDER` from the environment after `.env` loading.
    Output:
        Returns the lower-cased provider identifier.
    Security context:
        Restricts provider selection to explicit environment configuration so
        runtime behavior is auditable and separate from application code.
    """

    return os.getenv("MODEL_PROVIDER", "openai").strip().lower()


def _get_model_type() -> str:
    return os.getenv("MODEL_TYPE", "").strip().lower()


def _require_env(variable_name: str) -> None:
    """Validate that a required environment variable is present.

    Input:
        Receives the name of a required environment variable.
    Output:
        Raises `ValueError` when the variable is missing or empty.
    Security context:
        Fails closed when credentials are absent, preventing accidental model
        calls against misconfigured providers.
    """

    if not os.getenv(variable_name):
        raise ValueError(f"Missing required environment variable: {variable_name}")


def _get_embeddings(provider: str, model: str=None) -> Embeddings:
    if provider == "openai":
        _require_env("OPENAI_API_KEY")
        return OpenAIEmbeddings(
            model=model or "text-embedding-3-small",
            api_key=os.environ["OPENAI_API_KEY"],
        )
    if provider == "google":
        _require_env("GOOGLE_API_KEY")
        return GoogleGenerativeAIEmbeddings(
            model=model or "text-embedding-004",
            google_api_key=os.environ["GOOGLE_API_KEY"],
        )
    raise ValueError(f"Unsupported Embedding Model value: {provider}")


