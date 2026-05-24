from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

_model = None


def _load_model(model_name: str = "all-MiniLM-L6-v2"):
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {model_name}")
            _model = SentenceTransformer(model_name)
        except ImportError:
            raise RuntimeError("sentence-transformers not installed. Run: pip install sentence-transformers")
    return _model


def embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> list[list[float]]:
    model = _load_model(model_name)
    vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def embed_text(text: str, model_name: str = "all-MiniLM-L6-v2") -> list[float]:
    return embed_texts([text], model_name)[0]
