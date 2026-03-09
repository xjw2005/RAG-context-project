from __future__ import annotations

#app/engine/retrieval_strategy.py


from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np

from app.storage.stores import VectorStore
from app.utils.model_utils import EmbeddingModel
from app.utils.text_utils import extract_keywords_simple, texts_to_embeddings




class RetrievalStrategy(ABC):
    """检索抽象基类"""
    @abstractmethod
    def retrieve(self, query, top_k):
        """检索方法

        Args:
            query (str): 查询字符串
            top_k (int): 返回的结果数
        Returns:
            List[Tuple[float, str, str]]: 返回的结果列表，每个结果是一个元组，包含相似度得分、文档ID和文档内容
        """
        pass
    
class VectorRetrievalStrategy(RetrievalStrategy):
    """"向量检索策略"""    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        
    def retrieve(self, query: str, top_k: int) -> List[Tuple[float, str, str]]:
        """使用向量相似度搜索"""
        # 使用 texts_to_embeddings 返回 2D 数组 (1, dim)，符合 FAISS 要求
        query_embedding = texts_to_embeddings([query])
        similarities, indices = self.vector_store.search(query_embedding, top_k)
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            chunk_data = self.vector_store.get_chunk(idx)
            if chunk_data:
                chunk_id, chunk_text, _, _ = chunk_data
                results.append((float(sim), chunk_id, chunk_text))
        return results
    
class KeywordRetrievalStrategy(RetrievalStrategy):
    """关键词向量检索策略"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        
    def retrieve(self, query: str, top_k: int) -> List[Tuple[float, str, str]]:
        """使用关键词向量检索"""
        
        query_keywords = extract_keywords_simple(query, topK=5)
        if not query_keywords:
            return []
        
        query_kw_embeddings = texts_to_embeddings(query_keywords)
        if len(query_kw_embeddings) == 0:
            return []
        
        results = []
        
        # 遍历向量库
        for chunk_id, chunk_text, keywords, keyword_embeddings in self.vector_store:
            if keywords is None or len(keywords) == 0:
                continue
            # 计算余弦相似度
            max_similarity = 0
            for q_emb in query_kw_embeddings:
                for d_emb in keyword_embeddings:
                    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)
                    d_norm = d_emb / (np.linalg.norm(d_emb) + 1e-8)
                    similarity = np.dot(q_norm, d_norm)
                    if similarity > max_similarity:
                        max_similarity = similarity
            # 保存结果
            if max_similarity > 0.5:
                results.append((max_similarity, chunk_id, chunk_text))
        results = sorted(results, key=lambda x: x[0], reverse=True)
        return results[:top_k]
      
class MultiVectorRetrievalStrategy(RetrievalStrategy):
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    def retrieve(self, question: str, top_k: int = 5) -> List[Tuple[str, str]]:
        from app.utils.text_utils import text_to_embedding
        
        query_embedding = text_to_embedding(question)
        if len(query_embedding) == 0:
            return []
        
        # 使用带权重的搜索
        similarities, indices = self.vector_store.search_with_vector_type_weights(
            query_embedding, top_k
        )
        
        results = []
        seen_content = set()
        
        for sim, idx in zip(similarities[0], indices[0]):
            chunk_info = self.vector_store.get_chunk(idx)
            if chunk_info and len(chunk_info) >= 2:
                chunk_id, chunk_text = chunk_info[:2]
                if chunk_text not in seen_content:
                    seen_content.add(chunk_text)
                    results.append((chunk_id, chunk_text))
                    if len(results) >= top_k:
                        break
        
        return results