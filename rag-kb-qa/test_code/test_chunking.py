#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 pytest 测试切块功能
运行命令: pytest test_chunking.py -v
"""

import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app.rag import smart_chunk
from chunking_methods import paragraph_chunk


def test_smart_chunk_basic():
    """测试 smart_chunk 基本功能"""
    # 测试文本
    test_text = """
    人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
    机器学习是人工智能的核心技术之一。它通过算法让计算机从数据中学习模式，而不需要明确编程每一个可能的情况。
    """

    chunks = smart_chunk(test_text, chunk_size=200, chunk_overlap=20)
    print(chunks)  # 输出切块结果以供调试
    # pytest 断言
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)


# def test_smart_chunk_parameters():
#     """测试不同参数"""
#     test_text = "这是一个测试文本。" * 20

#     chunks = smart_chunk(test_text, chunk_size=100, chunk_overlap=10)

#     assert len(chunks) > 0
#     # 大部分块长度应该合理
#     assert all(len(chunk) <= 150 for chunk in chunks)  # 允许一些误差


# def test_empty_text():
#     """测试空文本"""
#     result = smart_chunk("")
#     assert isinstance(result, list)


# def test_short_text():
#     """测试短文本"""
#     result = smart_chunk("短文本")
#     assert len(result) >= 1
#     assert isinstance(result[0], str)
    
