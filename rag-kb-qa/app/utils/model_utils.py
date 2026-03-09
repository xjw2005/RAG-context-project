#app/utils/model_utils.py

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """所有模型的基础类"""
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型名称"""
        pass
    
class EmbeddingModel(BaseModel):
    """嵌入式模型"""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-8B"):
        self._model_name = model_name
        self.__api_key = "sk-vugpnklghmbnqnmtnjpswxkxaddbsbxsjnqxlhwlqmezimuk"
        self.__base_url = "https://api.siliconflow.cn/v1"
        self._client = None
    @property
    def model_name(self):
        return self._model_name
    
    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.__api_key, base_url=self.__base_url)
        return self._client
    
    def create_embeddings(self, texts: list[str]):
        return self.client.embeddings.create(model=self._model_name, input=texts)
    
    
    
