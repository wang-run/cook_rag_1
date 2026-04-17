"""
RAG系统配置文件
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RAGConfig:
    """RAG系统配置类
    """

    #路径配置
    data_path: str = "data/dishes"
    save_documents_path: str = "./documents/documents.pkl"
    index_save_path: str = "./vector_index"
    save_chunks_path: str = "./chunks/chunks.pkl"
    load_documents_path: str = "./documents/documents.pkl"

    #模型配置
    embedding_model : str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "kimi-k2-0711-preview"

    #检索配置
    top_k:int = 3

    #生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    def __post_init__(self):
        """初始化后的操作
        """
        pass

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        """从字典创建配置对象

        :param config_dict: 配置的字典，路径配置，模型配置，检索配置，生成配置
        :type config_dict: Dict[str, Any]
        :return: 用字典配置好的对象
        :rtype: RAGConfig
        """
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str , Any]:
        """
        将配置变为字典进行输出
        """
        return {
            'data_path' : self.data_path,
            'chunks_path' : self.chunks_path,
            'index_save_path' : self.index_save_path,
            'embedding_model' : self.embedding_model,
            'llm_model' : self.llm_model,
            'top_k' : self.top_k,
            'temperature' : self.temperature,
            'max_tokens' : self.max_tokens
        }
DEFAULT_CONFIG = RAGConfig()