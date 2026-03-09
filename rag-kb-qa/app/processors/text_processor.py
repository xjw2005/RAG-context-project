#app/processors/text_processor.py
# 1. 抽象接口
from abc import ABC, abstractmethod
from typing import List, Dict

import numpy as np


# 1. 抽象接口
class TextGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 150) -> str:  
        pass

# 2. 实现类
class VectorProcessor(ABC):
    @abstractmethod
    def process(self, chunks: List[str]) -> np.ndarray:
        pass
    
# 单一职责的具体实现
class LLMTextGenerator(TextGenerator):
    def __init__(self):
        self._client = None
    
    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key="sk-vugpnklghmbnqnmtnjpswxkxaddbsbxsjnqxlhwlqmezimuk",
                base_url="https://api.siliconflow.cn/v1"
            )
        
        response = self._client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        # 安全检查
        if not response.choices:
            raise ValueError("API 返回空的 choices 列表")
        
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("API 返回的 content 为 None")
        
        return content.strip()
    
# 摘要生成器
class SummaryGenerator:
    def __init__(self, text_generator: TextGenerator):
        self.text_generator = text_generator

    def generate(self, text: str) -> str:
        prompt = f"请为以下文本生成摘要：\n{text}\n摘要："
        return self.text_generator.generate(prompt, 150)

# 假设问题生成器
class HypotheticalQuestionGenerator:
    def __init__(self, text_generator: TextGenerator):
        self.text_generator = text_generator

    def generate(self, text: str, num_questions: int = 2) -> List[str]:
        prompt = f"基于以下文本生成{num_questions}个问题：\n{text}" 
        response = self.text_generator.generate(prompt, 200)        
        return [q.strip() for q in response.split('\n') if q.strip()][:num_questions]


# 添加多向量协调器
class MultiVectorOrchestrator:
    def __init__(self, summary_generator: SummaryGenerator, question_generator: HypotheticalQuestionGenerator):
        self.summary_generator = summary_generator
        self.question_generator = question_generator
        
    def generate_multi_vectors(self, text: str) -> Dict[str, any]:
        """为单个文本块生成多向量表示"""
        result = {}
        
        # 生成摘要
        try:
            summary = self.summary_generator.generate(text)
            result["summary"] = summary
        except Exception as e:
            print(f"摘要生成失败: {e}")
            result["summary"] = None
            
        # 生成问题
        try:
            questions = self.question_generator.generate(text)
            result["questions"] = questions
        except Exception as e:
            print(f"问题生成失败: {e}")
            result["questions"] = []
            
        return result
        
    def generate_vectors(self, chunks: List[str]) -> dict:
        from app.utils.text_utils import texts_to_embeddings
        import logging

        logger = logging.getLogger(__name__)
        result = {} # {"original": np.ndarray, "summary": np.ndarray, "questions": np.ndarray, "question_mapping": List[int], "summary_texts": List[str], "question_texts": List[str]}

        # 1. 原始向量
        result['original'] = texts_to_embeddings(chunks)

        # 2. 摘要向量
        summaries = [self.summary_generator.generate(chunk) for chunk in chunks]
        result['summary'] = texts_to_embeddings(summaries)
        result['summary_texts'] = summaries  # 保存摘要文本

        # 打印前3个摘要样例
        logger.info(f"生成了 {len(summaries)} 个摘要，样例：")
        for i, summary in enumerate(summaries[:3]):
            logger.info(f"  摘要 {i}: {summary[:100]}...")

        # 3. 问题向量
        all_questions = []
        question_mapping = []
        for i, chunk in enumerate(chunks):
            questions = self.question_generator.generate(chunk)
            for question in questions:
                all_questions.append(question)
                question_mapping.append(i)

        result['questions'] = texts_to_embeddings(all_questions)
        result['question_mapping'] = question_mapping
        result['question_texts'] = all_questions  # 保存问题文本

        # 打印前5个问题样例
        logger.info(f"生成了 {len(all_questions)} 个问题，样例：")
        for i, question in enumerate(all_questions[:5]):
            logger.info(f"  问题 {i}: {question}")

        return result