from __future__ import annotations

from typing import Text, List, Tuple
import logging

from app.engine.retrieval_strategy import KeywordRetrievalStrategy, VectorRetrievalStrategy
from app.processors.chunk_processor import TextChunker
from app.storage.stores import DocumentStore, VectorStore
from app.utils.search_utils import merge_and_rank, normalize_search_results
from app.utils.text_utils import text_to_embedding, texts_to_embeddings

logger = logging.getLogger(__name__)


class RAGEngine:
    """"RAG核心引擎类"""
    
    def __init__ (self, chunker: TextChunker, vector_store: VectorStore, document_store: DocumentStore):
        self.chunker = chunker
        self.vector_store = vector_store
        self.document_store = document_store
        
        #初始化检索策略
        self.strategies = {
            "vector": VectorRetrievalStrategy(vector_store),
            "keyword": KeywordRetrievalStrategy(vector_store)
        }
        
    def ingest_document_multi_vector(self, doc_id: str, text: str):
        """使用多向量方式摄取文档"""
        if self.document_store.exists(doc_id):
            logger.info(f"文档 {doc_id} 已存在，跳过")
            return
        
        try:
            # 保存原始文档
            self.document_store.save(doc_id, text)

            # 切块 - 多向量检索需要更大的块以生成高质量摘要和问题
            chunks = self.chunker.chunk(
                text,
                similariteshold=0.65,  # 提高阈值，减少分块
                max_chunk_size=1000        # 增加块大小，约500个汉字
            )

            # 生成多向量
            from app.processors.text_processor import MultiVectorOrchestrator, LLMTextGenerator, SummaryGenerator, HypotheticalQuestionGenerator
            
            text_gen = LLMTextGenerator()
            summary_gen = SummaryGenerator(text_gen)
            question_gen = HypotheticalQuestionGenerator(text_gen)
            
            orchestrator = MultiVectorOrchestrator(
                summary_generator=summary_gen,
                question_generator=question_gen
            )
            
            # 为每个块生成多向量
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                
                # 生成摘要和问题
                vectors = orchestrator.generate_multi_vectors(chunk)
                
                # 存储原始块
                self.vector_store.add_text(chunk_id, chunk)
                
                # 存储摘要向量
                if vectors.get("summary"):
                    summary_embedding = text_to_embedding(vectors["summary"])
                    self.vector_store.add_vector(f"{chunk_id}_summary", summary_embedding, {
                        "type": "summary",
                        "original_chunk_id": chunk_id,
                        "text": vectors["summary"]
                    })
                
                # 存储问题向量
                for j, question in enumerate(vectors.get("questions", [])):
                    question_embedding = text_to_embedding(question)
                    self.vector_store.add_vector(f"{chunk_id}_question_{j}", question_embedding, {
                        "type": "question", 
                        "original_chunk_id": chunk_id,
                        "text": question
                    })
                    
            logger.info(f"成功摄取文档 {doc_id}，共 {len(chunks)} 个块")
            
        except Exception as e:
            logger.error(f"摄取文档 {doc_id} 失败: {e}")
            raise
    
    def ingest_document(self, doc_id: str, text: str):
        """摄取文档到向量存储"""
        if self.document_store.exists(doc_id):
            logger.info(f"文档 {doc_id} 已存在，跳过")
            return
        
        try:
            # 保存原始文档
            self.document_store.save(doc_id, text)
            
            # 切块
            chunks = self.chunker.chunk(text)
            
            # 生成嵌入并存储
            embeddings = texts_to_embeddings([chunk for chunk in chunks])
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{doc_id}_chunk_{i}"
                self.vector_store.add_vector(chunk_id, embedding, {"text": chunk})
                
            logger.info(f"成功摄取文档 {doc_id}，共 {len(chunks)} 个块")
            
        except Exception as e:
            logger.error(f"摄取文档 {doc_id} 失败: {e}")
            raise
    
    def search(self, query: str, strategy: str = "vector", top_k: int = 5) -> List[Tuple[str, float]]:
        """搜索相关文档块"""
        if strategy not in self.strategies:
            raise ValueError(f"不支持的检索策略: {strategy}")
        
        return self.strategies[strategy].search(query, top_k)
    
    def hybrid_search(self, query: str, top_k: int = 5, vector_weight: float = 0.7) -> List[Tuple[str, float]]:
        """混合搜索：结合向量和关键词检索"""
        # 向量检索
        vector_results = self.search(query, "vector", top_k * 2)
        
        # 关键词检索
        keyword_results = self.search(query, "keyword", top_k * 2)
        
        # 合并和重排序
        merged_results = merge_and_rank(
            vector_results, 
            keyword_results, 
            vector_weight=vector_weight
        )
        
        return merged_results[:top_k]