"""
索引构建模块（构建向量库）
"""
import logging
from typing import List
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import torch
import os
import pickle


logger = logging.getLogger(__name__)#构建logging对象，这里在日志前面加上文件的名称

class IndexConstructionModule:
    """
    构建索引模块，负责向量化和索引构建
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", index_save_path: str = "./vector_index", save_chunks_path: str = "./chunks/chunks.pkl"):
        """初始化模块，指定使用BGE嵌入模型，指定存放地址

        :param model_name: 指定模型, defaults to "BAAI/bge-small-zh-v1.5"
        :type model_name: str, optional
        :param index_save_path: 存放地址, defaults to "./vector_index"
        :type index_save_path: str, optional
        """

        self.model_name = model_name
        self.save_chunks_path = save_chunks_path
        self.index_save_path = index_save_path
        self.embeddings = None
        self.vectorstore = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_embeddings()

    def setup_embeddings(self):
        """初始化嵌入模型
        """
        
        logger.info(f"正在初始化嵌入模型：{self.model_name}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name = self.model_name,
            model_kwargs = {"device" : self.device},
            encode_kwargs = {"normalize_embeddings" : True},
            show_progress = True
        )
        logger.info("嵌入模型初始化完成")

    def build_vector_index(self, chunks: List[Document]) ->FAISS:
        """构建向量索引

        :param chunks: 文档切块后的列表
        :type chunks: List[Document]
        :return: FAISS 向量储存对象
        :rtype: FAISS
        """
        logger.info("正在构建FAISS向量索引....")

        if not chunks:
            raise ValueError("文档块列表不能为空")
        
        self.vectorstore = FAISS.from_documents(
            documents = chunks,
            embedding = self.embeddings
        )

        logger.info(f"向量索引构建完成，包含{len(chunks)}个向量")
        return self.vectorstore
    

    def add_documents(self, new_chunks: List[Document]) -> FAISS:
        """在现有的索引基础上添加新的索引

        :param new_chunks: _description_
        :type new_chunks: List[Document]
        :return: 新的文档索引列表
        :rtype: FAISS
        """
        if not self.vectorstore:
            raise ValueError("请先构建向量索引才能添加向量索引")
        
        logger.info(f"开始将{len(new_chunks)}个向量添加")
        self.vectorstore.add_documents(new_chunks)
        logger.info(f"向量索引已经保存到{self.index_save_path}")

    def save_index(self):
        """将向量索引保存到指定路径中
        """
        if not self.vectorstore:
            raise ValueError("向量索引还未创建")
        
        #确保目录存在
        Path(self.index_save_path).mkdir(parents = True, exist_ok= True)

        #保存到本地
        self.vectorstore.save_local(self.index_save_path)
        logger.info(f"向量已经保存至{self.index_save_path}")

    def save_chunks(self, chunks: List[Document]):
        """将切分好的chunks保存到本地，方便后面BM25直接调用chunks

        :param chunks: 切分好的chunks
        :type chunks: List[Document]
        """
        logger.info(f"开始将{len(chunks)}个块保存到{self.save_chunks_path}")

        #保证参数合法
        if not chunks:
            raise ValueError(f"参数chunks不能为空")
        
        #确保目录存在
        Path(self.save_chunks_path).parent.mkdir(parents = True, exist_ok=True)

        #保存到本地
        with open(self.save_chunks_path, 'wb') as f:
            pickle.dump(chunks, f)
        logger.info(f"chunks已经保存到{self.save_chunks_path}中")

    def load_chunks(self):
        """从配置文件中加载向量索引
        """
        logger.info("开始加载chunks")
        try:
            if not Path(self.save_chunks_path).exists():
                raise FileExistsError(f"chunks路径不存在：{self.save_chunks_path},请更换路径")
            #打开保存好的chunks文件
            with open(self.save_chunks_path, "rb") as f:
                chunks = pickle.load(f)
            logger.info(f"chunks加载完毕,一共加载{len(chunks)}个块")
            return chunks
        except Exception as e:
            logger.warning(f"chunks加载失败:{e}")
            return None


    def load_index(self):
        """从配置文件中加载向量索引
        return :成功则返回向量库，失败则返回False
        """
        #加载的过程中需要用上embedding模型
        #这里先确认有无embedding
        if not self.embeddings:
            #没有的话这里直接创建
            self.setup_embeddings()
        
        #加载过程还需要使用到向量保存路径
        if not Path(self.index_save_path).exists():
            logger.info(f"路径不存在：{self.index_save_path}，请更换路径")
            return None
        
        #配置准备完成，开始尝试加载
        try:
            self.vectorstore = FAISS.load_local(
                folder_path=self.index_save_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"向量索引已经从路径{self.index_save_path}中加载")
            return self.vectorstore
        except Exception as e:
            logger.warning(f"加载向量失败：{e}")
            return None
        
    def load_local_documents(self, load_documents_path: str) -> List[Document]:
        """从本地的pickle中导入document父文档

        :param load_documents_path: 导入地址，一般与保存地址相同
        :type load_documents_path: str
        :return: 导入的父文档
        :rtype: List[Document]
        """
        logger.info(f"开始从{load_documents_path}中导入父文档...")
        try:
            if not Path(load_documents_path).exists():
                raise FileNotFoundError(f"{load_documents_path}路径无效")
            with open(load_documents_path, 'rb') as f:
                documents = pickle.load(f)
            logger.info(f"父文档加载成功，一共加载{len(documents)}个文档")
            return documents
        except Exception as e:
            logger.warning(f"documents导入失败：{e}")
            return []
        

    def similarity_search(self, query: str, k : int = 5) -> List[Document]:
        """对问题和向量库进行相似度搜查  后面还有检索优化，这里时最普通的检索方式

        :param query: 需要查询的文本
        :type query: str
        :param k: 返回相似结果数目, defaults to 5
        :type k: int, optional
        :return: 相似文档列表
        :rtype: List[Document]
        """
        if not self.vectorstore:
            raise ValueError("查询前请先构建向量索引")
        
        return self.vectorstore.similarity_search(query=query, k = k)

