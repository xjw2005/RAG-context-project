from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None


def cosine_similarity_vectors(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity for two vectors."""
    vector_a = np.asarray(a, dtype=np.float32)
    vector_b = np.asarray(b, dtype=np.float32)

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


def find_semantic_breakpoints(similarities: Sequence[float], threshold: float) -> List[int]:
    """Return sentence indices where a new chunk should start."""
    return [index + 1 for index, similarity in enumerate(similarities) if similarity < threshold]


def group_sentences_by_breakpoints(
    sentences: Sequence[str],
    breakpoints: Sequence[int],
    max_chunk_size: int,
) -> List[str]:
    """Merge sentences into chunks, respecting semantic breaks and size limits."""
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0
    breakpoint_set = set(breakpoints)

    for index, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_length = len(sentence)
        should_split = (
            current_chunk
            and (
                index in breakpoint_set
                or current_length + sentence_length > max_chunk_size
            )
        )

        if should_split:
            chunks.append(" ".join(current_chunk).strip())
            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks or [" ".join(sentences).strip()]


def _fallback_character_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Simple fallback splitter for environments without compatible LangChain packages."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")

    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step
    return chunks


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?。！？])\s*", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


class TextChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, **kwargs) -> List[str]:
        raise NotImplementedError


class SmartChunker(TextChunker):
    def chunk(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        if RecursiveCharacterTextSplitter is not None:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            return splitter.split_text(text)
        return _fallback_character_split(text, chunk_size, chunk_overlap)


class SemanticAwareChunker(TextChunker):
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name

    def chunk(
        self,
        text: str,
        similarity_threshold: float = 0.5,
        max_chunk_size: int = 500,
    ) -> List[str]:
        """基于语义相似度的文本分块。

        Args:
            text: 待分块的文本
            similarity_threshold: 语义相似度阈值，低于此值则分块
            max_chunk_size: 单个块的最大字符数

        Returns:
            分块后的文本列表
        """
        # 将输入文本按句子分割，使用正则表达式识别句号、感叹号、问号等标点
        sentences = _split_sentences(text)

        # 如果文本只有一句话或为空，直接返回原文本（去除首尾空格）
        if len(sentences) <= 1:
            return [text.strip()] if text.strip() else []

        # 动态导入文本嵌入工具函数，避免循环导入问题
        from app.utils.text_utils import texts_to_embeddings

        # 将所有句子转换为向量嵌入表示，用于计算语义相似度
        sentence_embeddings = texts_to_embeddings(sentences)

        # 检查嵌入向量数量是否与句子数量匹配，如果不匹配则降级为简单分块
        if len(sentence_embeddings) != len(sentences):
            return group_sentences_by_breakpoints(sentences, [], max_chunk_size)

        # 计算相邻句子之间的余弦相似度，用于判断语义连贯性
        similarities = [
            cosine_similarity_vectors(sentence_embeddings[i], sentence_embeddings[i + 1])
            for i in range(len(sentence_embeddings) - 1)
        ]

        # 根据相似度阈值找到语义断点，相似度低于阈值的位置作为分块边界
        breakpoints = find_semantic_breakpoints(similarities, threshold=similarity_threshold)

        # 根据语义断点和最大块大小限制，将句子组合成最终的文本块
        return group_sentences_by_breakpoints(sentences, breakpoints, max_chunk_size)
