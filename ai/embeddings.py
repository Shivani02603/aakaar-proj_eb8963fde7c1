"""Embeddings — any OpenAI-compatible provider in production, local HuggingFace for free,
or a deterministic FakeEmbedder when TESTING=1.

Provider is pure configuration (no code changes needed to switch):
  EMBEDDING_MODEL = "local-huggingface"  → free, no API key needed, downloads ~500MB once
  EMBEDDING_MODEL = "text-embedding-3-small"  → OpenAI (requires OPENAI_API_KEY)
  EMBEDDING_MODEL = "text-embedding-004"  → Gemini (requires LLM_API_KEY)
  EMBEDDING_BASE_URL  — empty = OpenAI; Gemini/other APIs need their endpoint
  EMBEDDING_API_KEY   — falls back to OPENAI_API_KEY if not set

The fake (TESTING=1) produces stable pseudo-vectors so tests need no API key or network.
"""
import hashlib
import math
import os
from typing import List

TESTING = os.getenv("TESTING") == "1"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "local-huggingface")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
_FAKE_DIM = 64

_local_model = None  # Cache for local HuggingFace model


def _get_local_model():
    """Load the local HuggingFace embedding model once (lazy, cached)."""
    global _local_model
    if _local_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _local_model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
        except ImportError:
            raise RuntimeError(
                "Local embeddings require 'sentence-transformers' package. "
                "Install with: pip install sentence-transformers\n"
                "Or set EMBEDDING_MODEL to 'text-embedding-3-small' and provide OPENAI_API_KEY."
            )
    return _local_model


def _client():
    """OpenAI-compatible client for cloud providers."""
    from openai import OpenAI
    base_url = os.getenv("EMBEDDING_BASE_URL") or None
    api_key = os.getenv("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            f"EMBEDDING_MODEL='{EMBEDDING_MODEL}' requires an API key.\n"
            "Set EMBEDDING_API_KEY or OPENAI_API_KEY in .env\n"
            "Or switch to EMBEDDING_MODEL='local-huggingface' (free, no API key needed)."
        )
    return OpenAI(api_key=api_key, base_url=base_url)


def _fake_embed(text: str) -> List[float]:
    """Stable hash-based embeddings for testing (no API, no network)."""
    vec = [0.0] * _FAKE_DIM
    for word in text.lower().split():
        h = int(hashlib.md5(word.encode(), usedforsecurity=False).hexdigest(), 16)
        vec[h % _FAKE_DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a list of texts using configured provider."""
    if TESTING:
        return [_fake_embed(t) for t in texts]

    if EMBEDDING_MODEL == "local-huggingface":
        model = _get_local_model()
        return model.encode(texts, convert_to_tensor=False).tolist()

    # OpenAI-compatible cloud provider
    resp = _client().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def get_embedding(text: str) -> List[float]:
    """Get embedding for a single text."""
    return get_embeddings([text])[0]


def cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (da * db)
