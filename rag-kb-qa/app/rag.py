from typing import Dict, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

_STORE: Dict[str, str] = {}
# 全局向量化模型
_EMBEDDING_MODEL = None
# FAISS 索引和元数据
_FAISS_INDEX = None
_INDEX_TO_DOC_MAP = []  # 索引位置 -> (doc_id, chunk_text) 的映射


def ingest(doc_id: str, text: str):
    """摄取文档，进行切块和向量化

    Args:
        doc_id (str): 文档ID
        text (str): 文档内容
    """
    if doc_id in _STORE:
        print(f"文档 {doc_id} 已存在，跳过")
        return
    
    # 保存原始文档
    _STORE[doc_id] = text

    # 切块
    chunks = smart_chunk(text, chunk_size=500, chunk_overlap=100)

    # 添加到 FAISS 索引
    add_to_faiss_index(doc_id, chunks)

def retrieve(question: str, top_k: int = 3) -> List[Tuple[str, str]]:
    """检索相关文档片段

    Args:
        question (str): 用户问题
        top_k (int): 返回的文档数量

    Returns:
        List[Tuple[str, str]]: (doc_id, snippet) 列表
    """
    # 优先使用向量搜索
    vector_results = search_by_embedding(question, top_k)

    if vector_results:
        return vector_results

    # 如果向量搜索无结果，回退到关键词搜索
    hits = []
    q_words = [w.lower() for w in question.split() if w.strip()]
    for doc_id, text in _STORE.items():
        score = sum(1 for w in q_words if w in text.lower())
        if score > 0:
            hits.append((score, doc_id, text[:200]))
    hits.sort(reverse=True, key=lambda x: x[0])
    return [(doc_id, snippet) for _, doc_id, snippet in hits[:top_k]]

def generate_answer(question: str, contexts: List[Tuple[str, str]]) -> str:
    if not contexts:
        return "未检索到相关内容，请先导入文档或换个问题。"
    return f"基于已检索内容，关于「{question}」的回答：请先查看引用片段并人工确认。"

def smart_chunk(text: str, chunk_size=500, chunk_overlap=50):
    """用LangChain 的递归切块器

    Args:
        text (str): 传进来的等待切块的文本  
        chunk_size (int, optional): 每个块的大小. Defaults to 500.
        chunk_overlap (int, optional): 块与块之间的重叠大小. Defaults to 50.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

def get_embedding_model(model_name: str = "Qwen/Qwen3-Embedding-8B"):
    """获取向量化模型（懒加载）"""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            # 尝试使用 OpenAI 客户端
            from openai import OpenAI
            _EMBEDDING_MODEL = {
                'client': OpenAI(
                    api_key="sk-vugpnklghmbnqnmtnjpswxkxaddbsbxsjnqxlhwlqmezimuk",
                    base_url="https://api.siliconflow.cn/v1"
                ),
                'model_name': model_name,
                'type': 'openai'
            }
        except ImportError:
            print("OpenAI 库未安装，使用本地模型")
            _EMBEDDING_MODEL = {
                'client': SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'),
                'model_name': 'local',
                'type': 'sentence_transformer'
            }
    return _EMBEDDING_MODEL

def text_to_embedding(text: str) -> np.ndarray:
    """将文本转换为向量

    Args:
        text (str): 输入文本

    Returns:
        np.ndarray: 文本向量
    """
    model_info = get_embedding_model()

    try:
        # 如果是 OpenAI 客户端
        if model_info['type'] == 'openai':
            response = model_info['client'].embeddings.create(
                model=model_info['model_name'],
                input=text
            )
            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)
    except Exception as e:
        print(f"API 调用失败: {e}")

    # 回退到本地模型
    if model_info['type'] == 'sentence_transformer':
        embedding = model_info['client'].encode(text, convert_to_tensor=False)
        return np.array(embedding, dtype=np.float32)
    else:
        # 如果当前模型不是本地模型，重新加载本地模型
        local_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        embedding = local_model.encode(text, convert_to_tensor=False)
        return np.array(embedding, dtype=np.float32)

def texts_to_embeddings(texts: List[str]) -> np.ndarray:
    """批量将文本转换为向量

    Args:
        texts (List[str]): 文本列表

    Returns:
        np.ndarray: 向量矩阵
    """
    model_info = get_embedding_model()

    try:
        # 如果是 OpenAI 客户端
        if model_info['type'] == 'openai':
            response = model_info['client'].embeddings.create(
                model=model_info['model_name'],
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings, dtype=np.float32)
    except Exception as e:
        print(f"批量 API 调用失败: {e}")

    # 回退到本地模型或逐个调用
    if model_info['type'] == 'sentence_transformer':
        embeddings = model_info['client'].encode(texts, convert_to_tensor=False)
        return np.array(embeddings, dtype=np.float32)
    else:
        # 逐个调用
        embeddings = []
        for text in texts:
            embedding = text_to_embedding(text)
            embeddings.append(embedding)
        return np.array(embeddings, dtype=np.float32)


def init_faiss_index(dimension: int = 4096):
    """初始化 FAISS 索引

    Args:
        dimension (int): 向量维度，OpenAI text-embedding-3-small 为 1536 维
    """
    global _FAISS_INDEX, _INDEX_TO_DOC_MAP
    # 使用内积索引（适合归一化向量）
    _FAISS_INDEX = faiss.IndexFlatIP(dimension)
    _INDEX_TO_DOC_MAP = []

def add_to_faiss_index(doc_id: str, chunks: List[str]):
    """将文档块添加到 FAISS 索引

    Args:
        doc_id (str): 文档ID
        chunks (List[str]): 文档块列表
    """
    global _FAISS_INDEX, _INDEX_TO_DOC_MAP

    if _FAISS_INDEX is None:
        init_faiss_index()

    # 批量向量化
    embeddings = texts_to_embeddings(chunks)

    # 归一化向量（用于内积索引）
    faiss.normalize_L2(embeddings)

    # 添加到索引
    _FAISS_INDEX.add(embeddings)

    # 更新映射关系
    for chunk in chunks:
        _INDEX_TO_DOC_MAP.append((doc_id, chunk))


def search_by_embedding(query: str, top_k: int = 3) -> List[Tuple[str, str]]:
    """基于向量搜索相关文档

    Args:
        query (str): 用户查询
        top_k (int, optional): 返回的相关文档数量. Defaults to 3.

    Returns:
        List[Tuple[str, str]]: 相关文档列表 (doc_id, snippet)
    """
    global _FAISS_INDEX, _INDEX_TO_DOC_MAP

    # 检查索引是否存在
    if _FAISS_INDEX is None or _FAISS_INDEX.ntotal == 0:
        return []

    # 将查询转换为向量
    query_embedding = text_to_embedding(query).reshape(1, -1)

    # 归一化查询向量
    faiss.normalize_L2(query_embedding)

    # 在 FAISS 中搜索
    similarities, indices = _FAISS_INDEX.search(query_embedding, min(top_k, _FAISS_INDEX.ntotal))

    # 构建结果
    results = []
    for similarity, idx in zip(similarities[0], indices[0]):
        if idx != -1 and similarity > 0:  # 过滤无效结果
            doc_id, chunk_text = _INDEX_TO_DOC_MAP[idx]
            results.append((doc_id, chunk_text))

    return results