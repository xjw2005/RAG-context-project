from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

class DocumentStore:
    """文档存储类"""
    
    def __init__(self):
        self._documents: Dict[str, str] = {}
        
    def save(self, doc_id: str, text: str) -> None:
        """保存文档"""
        self._documents[doc_id] = text
        
    def get(self, doc_id: str) -> str:
        """获取文档"""
        return self._documents.get(doc_id, "")
    
    def exists(self, doc_id: str) -> bool:
        """检查文档是否存在"""
        return doc_id in self._documents
    
    def delete(self, doc_id: str) -> None:
        """删除文档"""
        if doc_id in self._documents:
            del self._documents[doc_id]
            
class VectorStore:
    """向量存储"""
    
    def __init__(self, dimension: int = 4096):
        import faiss  # 确保安装了 faiss 库
        self._dimension = dimension
        self._index = faiss.IndexFlatIP(self._dimension)
        self._doc_map = [] 
        
    def add(self, doc_id: str, chunks: list[str], embeddings: np.ndarray, keywords_data: List = []) -> None:
        """添加文档"""
        import faiss  # 确保安装了 faiss 库

        # 确保是 numpy 数组且类型为 float32
        embeddings = np.array(embeddings, dtype=np.float32)

        # 确保是二维数组 (n_vectors, dimension)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        faiss.normalize_L2(embeddings)  # 归一化向量

        self._index.add(embeddings)  # type: ignore
        
        #更新映射
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            if keywords_data and i < len(keywords_data):
                keywords, keyword_embeddings = keywords_data[i]
                self._doc_map.append((chunk_id, chunk_text, keywords, keyword_embeddings))
            else:
                self._doc_map.append((chunk_id, chunk_text, None, None))

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """搜索文档"""
        import faiss

        if self._index.ntotal == 0:
            return np.array([]), np.array([])

        # 确保是 numpy 数组且类型为 float32
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # 确保是二维数组 (1, dimension)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        faiss.normalize_L2(query_embedding)
        similarities, indices = self._index.search(query_embedding, min(top_k, self._index.ntotal))  # type: ignore
        return similarities, indices
    
    def get_chunk(self, idx: int) -> str:
        """根据索引获取切块
            Args:
                idx (int): 索引
            Returns:
                包含(chunk_id, chunk_text, keywords, keyword_embeddings)的元组
        """
        if 0 <= idx < len(self._doc_map):
            return self._doc_map[idx]
        else:
            return ""
        
    def add_multi_vectors(self, doc_id: str, chunks: list[str], multi_embeddings: dict) -> None:
        """添加多向量数据（扩展现有功能）"""
        # 如果只有原始向量，使用现有的 add 方法
        if 'original' in multi_embeddings and len(multi_embeddings) == 1:
            self.add(doc_id, chunks, multi_embeddings['original'])
            return

        # 多向量处理逻辑
        import faiss

        # 为不同类型创建标记
        for vector_type, embeddings in multi_embeddings.items():
            if vector_type in ['question_mapping', 'summary_texts', 'question_texts']:
                continue

            embeddings = np.array(embeddings, dtype=np.float32)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            faiss.normalize_L2(embeddings)
            self._index.add(embeddings)  # type: ignore

            # 更新映射，添加类型标记
            if vector_type == 'original':
                for i, chunk_text in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    self._doc_map.append((chunk_id, chunk_text, vector_type, None))
            elif vector_type == 'summary':
                summary_texts = multi_embeddings.get('summary_texts', [])
                for i, summary_text in enumerate(summary_texts):
                    chunk_id = f"{doc_id}_summary_{i}"
                    # 存储摘要文本，但关联到原始chunk
                    original_chunk = chunks[i] if i < len(chunks) else ""
                    self._doc_map.append((chunk_id, original_chunk, vector_type, summary_text))
            elif vector_type == 'questions':
                question_texts = multi_embeddings.get('question_texts', [])
                question_mapping = multi_embeddings.get('question_mapping', [])
                for i, (question_text, chunk_idx) in enumerate(zip(question_texts, question_mapping)):
                    if chunk_idx < len(chunks):
                        chunk_id = f"{doc_id}_question_{i}"
                        # 存储问题文本，但关联到原始chunk
                        original_chunk = chunks[chunk_idx]
                        self._doc_map.append((chunk_id, original_chunk, vector_type, question_text))

    def search_with_vector_type_weights(self, query_embedding: np.ndarray, top_k: int = 5) -> tuple:
        """带向量类型权重的搜索"""
        similarities, indices = self.search(query_embedding, top_k * 3)  # 获取更多结果
        
        # 应用权重
        weights = {'original': 1.0, 'summary': 0.9, 'questions': 0.7}
        weighted_results = []
        
        for sim, idx in zip(similarities[0], indices[0]):
            if 0 <= idx < len(self._doc_map):
                chunk_info = self._doc_map[idx]
                if len(chunk_info) >= 3:
                    chunk_id, chunk_text, vector_type = chunk_info[:3]
                    weight = weights.get(vector_type, 1.0)
                    weighted_score = float(sim) * weight
                    weighted_results.append((weighted_score, idx))
        
        # 重新排序
        weighted_results.sort(key=lambda x: x[0], reverse=True)
        
        # 返回前 top_k 个
        final_similarities = np.array([[r[0] for r in weighted_results[:top_k]]])
        final_indices = np.array([[r[1] for r in weighted_results[:top_k]]])
        
        return final_similarities, final_indices
    
    def __iter__(self):
        """返回一个迭代器"""
        return iter(self._doc_map)

    def __len__(self):
        """返回向量数量"""
        return len(self._doc_map)
    
        

    
    
