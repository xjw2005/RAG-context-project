# app/dependencies.py

#初始化组件
from app.engine.rag_engine import RAGEngine
from app.processors.chunk_processor import SemanticAwareChunker
from app.storage.stores import DocumentStore, VectorStore
from app.engine.retrieval_strategy import MultiVectorRetrievalStrategy



document_store = DocumentStore()
vector_store = VectorStore(dimension=4096)  # Qwen3-Embedding-8B 的维度
chunker = SemanticAwareChunker()

# 创建 RAG 引擎实例
rag_engine = RAGEngine(
    chunker=chunker,
    vector_store=vector_store,
    document_store=document_store
)

# 添加多向量检索策略
multi_vector_strategy = MultiVectorRetrievalStrategy(vector_store)
rag_engine.strategies["multi_vector"] = multi_vector_strategy