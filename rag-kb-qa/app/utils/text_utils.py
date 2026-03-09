from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.utils.model_utils import EmbeddingModel

_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model() -> EmbeddingModel:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model


def extract_keywords_simple(text: str, topK: int = 5) -> List[Any]:
    try:
        import jieba.analyse
        return jieba.analyse.extract_tags(text, topK=topK, withWeight=False)
    except ImportError:
        words = re.findall(r"[\u4e00-\u9fff]{2,}", text)
        return list(dict.fromkeys(words))[:topK]
    except Exception as exc:
        print(f"Keyword extraction failed: {exc}")
        return []


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?。！？])\s*", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def texts_to_embeddings(texts: List[str]) -> np.ndarray:
    embedding_model = get_embedding_model()
    try:
        response = embedding_model.create_embeddings(texts)
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype=np.float32)
    except Exception as exc:
        print(f"Batch embedding failed: {exc}")
        return np.array([])


def text_to_embedding(text: str) -> np.ndarray:
    embeddings = texts_to_embeddings([text])
    if len(embeddings) == 0:
        return np.array([])
    return embeddings[0]
