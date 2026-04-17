"""
检索优化模块
"""

import logging
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class RetrievalOptimizationModule:
    """检索优化模块，混合检索，通过多个检索器检索的结果来进行优化
    """
    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        """初始化检索优化模块

        :param vectorstore: 构建好了的向量索引
        :type vectorstore: FAISS
        :param chunks: 文档块列表
        :type chunks: List[Document]
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.setup_retrievers()

    def setup_retrievers(self):
        """构建向量检索器和BM25检索器
        """
        logger.info("开始构建检索器...")

        #向量检索器
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type = "similarity",
            search_kwargs = {"k" : 5}
        )

        #BM25检索器
        #这里每次进行排序的时候都需要用到chunks，所以为了避免每次都切分文档加载chunks，直接保存到本地之后从本地加载
        self.bm25_retirever = BM25Retriever.from_documents(
            documents = self.chunks,
            k = 5
        )

        logger.info("检索器构建完成")

    def filtered_hybrid_search(self, query: str, top_k : int = 3, filters : dict = None) -> List[Document]:
        """使用RRF将两种检索器结果融合，混合检索

        :param query: 查询的文本
        :type query: str
        :param top_k: 相似文档个数, defaults to 3
        :type top_k: int, optional
        :param filters: 过滤条件
        :type filters: dict
        :return: 相似文档列表
        :rtype: List[Document]
        """
        #这里需要塞入filters，直接再setup_retrievers塞入的话每次询问都需要重新构建检索库
        if filters:
            #如果本次有过滤条件，就临时赛进检索器的配置里面
            self.vector_retriever.search_kwargs['filter'] = filters
        else:
            #没有过滤条件，要把上一次的残留清空
            if "filter" in self.vector_retriever.search_kwargs:
                del self.vector_retriever.search_kwargs['filter']

        #分别获取向量检索器和BM25检索器的结果
        vector_docs = self.vector_retriever.invoke(query)
        raw_bm25_docs = self.bm25_retirever.invoke(query)

        #先进行应用元数据过滤
        bm25_docs = []
        if filters:
            for doc in raw_bm25_docs:
                match = True
                #query:["category" : ["各种种类"]]这种格式是需要大模型进行提取的
                for key, value in filters.items():
                    if key in doc.metadata:
                        if isinstance(value, list):
                            if doc.metadata[key] not in value:
                                match = False
                                break
                        else:
                            if doc.metadata[key] != value:
                                match = False
                                break
                    else:
                        match = False
                        break
                if match:
                    bm25_docs.append(doc)
            logger.info(f"过滤完成，一共得到{len(vector_docs) + len(bm25_docs)}个文档")
        else:
            bm25_docs = raw_bm25_docs
        

        #使用RRF重排
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)
        return reranked_docs[:top_k]
    
    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k :int = 60) -> List[Document]:
        """构建RRF算法结构并实现重排

        :param vector_docs: 向量检索结果
        :type vector_docs: List[Document]
        :param bm25_docs: BM25检索结果
        :type bm25_docs: List[Document]
        :param k: RRF参数，用于平滑排名, defaults to 60
        :type k: int, optional
        :return: 重排之后的document列表
        :rtype: List[Document]
        """

        doc_scores = {}
        doc_objects = {}

        #计算向量检索结果的RRF分数
        for rank, doc in enumerate(vector_docs):
            #将文本内容转化为hash值作为文档的id
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            #RRF公式：1/(k + rank)
            rrf_score = 1.0 / (k + rank + 1)#这里没有排名为0的，但是索引有
            #这里用累加而不是覆盖，应为下面的BM25计算分数的时候也会用这个字典进行加分
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"向量检索 - 文档{rank + 1}: RRF分数 = {rrf_score:.4f}")

        #计算BM25检索结果的RRF分数
        for rank, doc in enumerate(bm25_docs):
            #用hash值生成id
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"BM25检索 - 文档{rank + 1}: RRF分数 = {rrf_score:.4f}")
        
        #按最终的RRF分数进行排序
        sorted_docs = sorted(doc_scores.items(), key = lambda x : x[1], reverse=True)#这里用items将字典打包成元组

        #构建最终结果
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                #将RRF分数添加到文档的元数据中
                doc.metadata["rrf_score"] = final_score
                reranked_docs.append(doc)
                logger.debug(f"最终排序 - 文档：{doc.page_content[:50]}... 最终分数为：{final_score:.4f}")
            
        logger.info(f"RRF重排结束：向量索引{len(vector_docs)}个文档，BM25检索{len(bm25_docs)}个文档，合并后{len(reranked_docs)}个文档")
        return reranked_docs
    
