#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试向量化功能
运行命令: python test_embedding.py
"""

import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engine.rag_engine import text_to_embedding, texts_to_embeddings, smart_chunk

def test_single_embedding():
    """测试单个文本向量化"""
    print("=== 测试单个文本向量化 ===")

    text = "人工智能是计算机科学的一个分支"
    embedding = text_to_embedding(text)

    print(f"原文本: {text}")
    print(f"向量维度: {embedding.shape}")
    print(f"向量类型: {type(embedding)}")
    print(f"向量前5个值: {embedding[:5]}")
    print()

def test_batch_embedding():
    """测试批量文本向量化"""
    print("=== 测试批量文本向量化 ===")

    texts = [
        "人工智能是计算机科学的分支",
        "机器学习是AI的核心技术",
        "深度学习使用神经网络"
    ]

    embeddings = texts_to_embeddings(texts)

    print(f"文本数量: {len(texts)}")
    print(f"向量矩阵形状: {embeddings.shape}")
    print(f"每个向量维度: {embeddings.shape[1]}")

    for i, text in enumerate(texts):
        print(f"文本 {i+1}: {text}")
        print(f"  向量前3个值: {embeddings[i][:3]}")
    print()

def test_chunking_and_embedding():
    """测试切块 + 向量化的完整流程"""
    print("=== 测试切块 + 向量化流程 ===")

    long_text = """
    人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
    机器学习是人工智能的核心技术之一。它通过算法让计算机从数据中学习模式，而不需要明确编程每一个可能的情况。
    深度学习是机器学习的一个子集，使用神经网络来模拟人脑的工作方式。这种方法在图像识别、语音处理等领域取得了突破性进展。
    """

    # 步骤1: 切块
    chunks = smart_chunk(long_text, chunk_size=100, chunk_overlap=20)
    print(f"切块数量: {len(chunks)}")

    # 步骤2: 向量化
    embeddings = texts_to_embeddings(chunks)
    print(f"向量矩阵形状: {embeddings.shape}")

    # 显示结果
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        print(f"\n块 {i+1}:")
        print(f"  文本: {chunk[:50]}...")
        print(f"  向量维度: {embedding.shape}")
        print(f"  向量范围: [{embedding.min():.3f}, {embedding.max():.3f}]")

def test_similarity():
    """测试向量相似度"""
    print("\n=== 测试向量相似度 ===")

    texts = [
        "人工智能技术发展迅速",
        "AI技术进步很快",  # 相似
        "今天天气很好"      # 不相似
    ]

    embeddings = texts_to_embeddings(texts)

    # 计算余弦相似度
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    import numpy as np

    print("相似度矩阵:")
    for i in range(len(texts)):
        for j in range(len(texts)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"文本{i+1} vs 文本{j+1}: {sim:.3f}")

    print(f"\n文本1: {texts[0]}")
    print(f"文本2: {texts[1]}")
    print(f"文本3: {texts[2]}")

if __name__ == "__main__":
    try:
        test_single_embedding()
        test_batch_embedding()
        test_chunking_and_embedding()
        test_similarity()
        print("=== 所有测试完成 ===")
    except Exception as e:
        print(f"测试出错: {e}")
        import traceback
        traceback.print_exc()