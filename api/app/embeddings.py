from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.config import get_settings

MSMARCO_MODEL_NAME = "msmarco-MiniLM-L6-cos-v5"


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Lazily load the embedding model.

    The first call downloads/loads the model. Later calls reuse it.
    Anime embeddings are precomputed offline. Query embeddings are generated at runtime.
    """
    settings = get_settings()
    device = _best_device()
    model = SentenceTransformer(settings.embedding_model_name, device=device)
    print(f"Embedding model: {settings.embedding_model_name} | device: {device}")
    return model


@lru_cache(maxsize=1)
def get_msmarco_model() -> SentenceTransformer:
    """Lazily load the asymmetric msmarco model (text→item retrieval path)."""
    device = _best_device()
    model = SentenceTransformer(MSMARCO_MODEL_NAME, device=device)
    print(f"Embedding model: {MSMARCO_MODEL_NAME} | device: {device}")
    return model


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize each vector to length 1.

    This makes cosine similarity equal to dot product.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


def embed_texts(texts: list[str], batch_size: int = 64, show_progress_bar: bool = False) -> np.ndarray:
    """
    Encode text into normalized 384-dimensional embeddings.
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=show_progress_bar,
    ).astype(np.float32)
    return normalize_rows(embeddings)


def embed_texts_msmarco(texts: list[str], batch_size: int = 64, show_progress_bar: bool = False) -> np.ndarray:
    """Encode text using the asymmetric msmarco model (text→item retrieval path)."""
    model = get_msmarco_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=show_progress_bar,
    ).astype(np.float32)
    return normalize_rows(embeddings)
