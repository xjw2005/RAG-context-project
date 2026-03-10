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
                similarity_threshold=0.65,  # 提高阈值，减少分块
                max_chunk_size=1000        # 增加块大小，约500个汉字
            )

            # 生成多向量
            from app.processors.text_processor import MultiVectorOrchestrator, LLMTextGenerator, SummaryGenerator, HypotheticalQuestionGenerator
            
            text_gen = LLMTextGenerator()
            summary_gen = SummaryGenerator(text_gen)
            question_gen = HypotheticalQuestionGenerator(text_gen)
            orchestrator = MultiVectorOrchestrator(summary_gen, question_gen)
            
            multi_embeddings = orchestrator.generate_vectors(chunks)
            
            # 存储多向量
            self.vector_store.add_multi_vectors(doc_id, chunks, multi_embeddings)
            
            # 这里的vector_sotre的数据类型是：VectorStore
            
            logger.info(f"多向量文档 {doc_id} 已摄取，包含 {len(chunks)} 个块")
        except Exception as e:
            logger.error(f"多向量摄取文档 {doc_id} 失败: {e}")
            raise
        
    def ingest_document(self, doc_id: str, text: str):
        """摄取文档，进行切块和向量化

        Args:
            doc_id (str): 文档ID
            text (str): 文档内容
        """
        if self.document_store.exists(doc_id):
            logger.info(f"文档 {doc_id} 已存在，跳过")
            return
        
        try:
            # 保存原始文档
            self.document_store.save(doc_id, text)
            
            # 切块
            chunks = self.chunker.chunk(text)
            
            # 向量化
            embeddings = texts_to_embeddings(chunks)
            
            # 存储
            self.vector_store.add(doc_id, chunks, embeddings)
            
            logger.info(f"文档 {doc_id} 已摄取，包含 {len(chunks)} 个块")
        except Exception as e:
            logger.error(f"摄取文档 {doc_id} 失败: {e}")
            raise
        
    def generate_answer(self, question: str, contexts: List[Tuple[str, str]]) -> str:
        if not contexts:
            return "未检索到相关内容，请先导入文档或换个问题。"
        return f"基于已检索内容，关于「{question}」的回答：请先查看引用片段并人工确认。"

    def retrieve(self, question: str, top_k: int = 3) -> List[Tuple[str, str]]:
        """检索相关文档片段（使用组合检索）

        Args:
            question (str): 用户问题
            top_k (int): 返回的文档数量

        Returns:
            List[Tuple[str, str]]: (chunk_id, snippet) 列表
        """
        return self.retrieve_combined(question, top_k)
    
    def retrieve_combined(self, question: str, top_k: int = 5) -> List[Tuple[str, str]]:
        """结合向量和关键词向量检索

        Args:
            question (str): 用户提交的问题
            top_k (int, optional): 取前 top_k 个结果. Defaults to 5.

        Returns:
            List[Tuple[str, str]]: (chunk_id, snippet) 列表
        """
        try:
            # 向量检索
            vector_results = self.strategies["vector"].retrieve(question, top_k)
            # 关键词检索
            keyword_results = self.strategies["keyword"].retrieve(question, top_k)

            # 归一化结果
            vector_scores = normalize_search_results(vector_results)
            keyword_scores = normalize_search_results(keyword_results)

            # 合并结果，按分数排序
            merged_results = merge_and_rank(vector_scores, keyword_scores, top_k=top_k)
            
            return merged_results
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []
