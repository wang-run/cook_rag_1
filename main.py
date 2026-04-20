"""
RAG生成主程序
"""

import os
import sys
import logging
from pathlib import Path
from typing import List

#添加搜索文件路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule
)

#从默认目录下载入.env文件
load_dotenv("api.env")

#配置日志
logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecipeRAGSystem:
    """
    食谱RAG系统主类
    """
    def __init__(self, config:RAGConfig = None):
        """初始化RAG系统

        :param config: RAG系统配置，默认使用DEFAULT_CONFIG, defaults to None
        :type config: RAGConfig, optional
        """
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        #检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileExistsError(f"数据路径不存在：{self.config.data_path}")
        
        #检查API密钥是否配置好
        if not os.getenv("MOONSHOT_API_KEY"):
            raise ValueError(f"请先设置MOONSHOT_API_KEY变量")
        
    def initialize_system(self):
        """初始化数据模块(除了检索模块)
        """
        print("正在初始化RAG系统....")

        #1.初始化数据准备模块
        print("开始准备数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path, self.config.save_documents_path)

        #2.初始化索引模块...
        print("开始初始化索引模块...")
        self.index_module = IndexConstructionModule(model_name=self.config.embedding_model, index_save_path=self.config.index_save_path, save_chunks_path= self.config.save_chunks_path)

        #3.初始化生成集成模块
        print("开始初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(self.config.llm_model, self.config.temperature, self.config.max_tokens)

        print("系统数据模块初始化完成")


    def build_knowledge_base(self):
        """构建知识库
        """

        #1.尝试加载已经保存的索引库和chunks,documents
        vectorstore = self.index_module.load_index()
        chunks = self.index_module.load_chunks()
        self.data_module.documents = self.index_module.load_local_documents(self.config.load_documents_path)

        #如果之前已经保存过索引库
        #保存索引库的时候会同时保存chunks
        if all([vectorstore, chunks, self.data_module.documents]):
            print("成功加载索引库、chunks和父文档documents")
            
        else:
            print("未发现已保存的索引， 开始构建索引并保存chunks")

            #2.加载文档并保存文档
            print("加载文档...")
            self.data_module.load_documents()
            self.data_module.save_documents()

            #3.文本分块和保存
            print("进行文本分块并保存...")
            chunks = self.data_module.chunk_documents()
            self.index_module.save_chunks(chunks=chunks)
            

            #4.构建向量索引
            print("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks = chunks)

            #5.保存索引
            print("保存向量索引...")
            self.index_module.save_index()
        #6.初始化检索模块
        print("初始化检索模块...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore=vectorstore, chunks = chunks)#这里BM25会用到chunks

        #7.显示统计数据
        stats = self.data_module.get_statistics()
        print(f"\n知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   菜品分类: {list(stats['categories'].keys())}")
        print(f"   难度分布: {stats['difficulties']}")

        print("知识库构建完成！")

    def ask_question(self, question: str, stream: bool = False):
        """回答用户问题

        :param question: 用户的问题
        :type question: str
        :param stream: 是否使用流输出, defaults to False
        :type stream: bool, optional
        :return: 回答或者生成器
        :rtype: any
        """
        #生成回答的时候需要用上生成器和检索器
        if not all([self.generation_module, self.retrieval_module]):
            raise ValueError("请先构建知识库")
        
        print(f"\n 用户问题:{question}")

        #1.查询路由
        route_type = self.generation_module.query_router(query=question)
        print(f"查询类型：{route_type}")

        if route_type != 'unknown':
            #2.根据路由类型来进行重写(list不用重写，list意义明确)
            # if route_type == 'list':
            #     rewritten_query = question
            #     print(f"列表和未知查询保持原样：{rewritten_query}")
            # else:
            #     #其他类型进行重写
            #     print("智能分析查询....")
            #     rewritten_query = self.generation_module.query_rewrite(query=question)
            rewritten_query = self.generation_module.query_rewrite(query=question)
            question = rewritten_query['rewrite_query']
                
            print(f"重写的的查询为：{rewritten_query["rewrite_query"]}")
            print("查询相关文档...")
            #这里使用原查询进行元数据匹配的过滤操作
            filters = self._extract_filters_from_query(rewritten_query)
            
            print(f"应用过滤条件：{filters}")
            #这个函数里面包含检查filter是否有效，无效返回没有过滤的文档
            relevant_chunks = self.retrieval_module.filtered_hybrid_search(query = question, top_k = self.config.top_k, filters=filters)
            #显示检索到的子块信息
            if relevant_chunks:
                chunk_info = []
                for chunk in relevant_chunks:
                    dish_name = chunk.metadata.get("dish_name", "未知菜品")
                    #尝试从内容中提取章节标题
                    section_title = chunk.metadata.get('三级标题', chunk.metadata.get('二级标题', chunk.metadata.get('主表题', '内容片段')))
                    chunk_info.append(f"{dish_name}{section_title}")
                #这里直接显示找到了那个菜品的那个标题
                print(f"找到了{len(relevant_chunks)}个相关文档块：{','.join(chunk_info)}")
            else:
                print(f"找到{len(relevant_chunks)}个相关文档块")
                return "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"

            #5.根据路由类型选择回答方式
            if route_type == 'list':
                #列表查询:直接返回相关菜品名称列表
                print("生成菜品列表...")
                #这里直接找到对应的父文档
                relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
                #遍历父文档，拿到父文档的菜品名称
                doc_names = []
                for doc in relevant_chunks:
                    doc_name = doc.metadata.get('dish_name', '未知菜品')
                    doc_names.append(doc_name)
                if doc_names:
                    print(f"找到文档：{','.join(doc_names)}")
                #这里使用相关原始文档而不是使用名称，是因为只给菜名信息太少太单薄，大模型容易瞎猜，给全部文档有利于大模型回答问题
                return self.generation_module.generate_list_answer(query = question, context_docs = relevant_docs)
            else:
                # 详细查询：获取完整文档并生成详细回答
                print("获取完整文档...")
                relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

                # 显示找到的文档名称
                doc_names = []
                for doc in relevant_docs:
                    dish_name = doc.metadata.get('dish_name', '未知菜品')
                    doc_names.append(dish_name)

                if doc_names:
                    print(f"找到文档: {', '.join(doc_names)}")
                else:
                    print(f"对应 {len(relevant_docs)} 个完整文档")
                
                if route_type == "detail":
                    #详细查询使用分步指导模式
                    if stream:
                        return self.generation_module.generate_step_by_step_answer_stream(query = question, context_docs=relevant_docs)
                    else:
                        return self.generation_module.generate_step_by_step_answer(query = question, context_docs= relevant_chunks)
                else:
                    if stream:
                        return self.generation_module.generate_basic_answer_stream(query = question, context_docs = relevant_chunks)
                    else:
                        return self.generation_module.generate_basic_answer(query= question, context_docs=relevant_chunks)
        else:
            return self.generation_module.generate_unknown(query=question)


    def _extract_filters_from_query(self, re_write_query: str) -> dict:
        """在改写后的用户查询中提取相应的过滤关键词

        :param query: 改写后的用户的查询
        :type query: str
        :return: 过滤条件：菜品种类或困难度
        :rtype: dict
        """
        
        return re_write_query["filters"]
    
    def search_by_category(self, category : str, query: str = "") -> List[str]:
        """按分类搜寻菜品

        :param category: 菜品的种类
        :type category: str
        :param query: 可选的额外查询条件, defaults to ""
        :type query: str, optional
        :return: 菜品名称列表
        :rtype: List[str]
        """
        if not self.retrieval_module:
            raise ValueError("请先构建知识库")
        
        #使用元数据进行过滤
        search_query = query if query else category
        filters = {"category" : category}
        docs = self.retrieval_module.filtered_hybrid_search(query = search_query, top_k = 10, filters = filters)

        #提取菜品名称
        dish_names = []
        for doc in docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            if dish_name not in dish_names:
                dish_names.append(dish_name)
        
        return dish_names
    
    def get_ingredients_list(self, dish_name: str) -> str:
        """获取指定菜品的方法信息

        :param dish_name: 指定菜谱的名称
        :type dish_name: str
        :return: 相关菜谱的信息
        :rtype: str
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构造知识库和初始化")
        
        #搜索相关文档
        docs = self.retrieval_module.filtered_hybrid_search(query=dish_name, top_k = 3)
        #生成食材答案
        answer = self.generation_module.generate_basic_answer(f"{dish_name}需要什么食材？", docs)

        return answer
    
    def run_interactive(self):
        """运行交互式问答
        """
        print("=" * 60)
        print("🍽️  尝尝咸淡RAG系统 - 交互式问答  🍽️")
        print("=" * 60)
        print("💡 解决您的选择困难症，告别'今天吃什么'的世纪难题！")

        #初始化系统
        self.initialize_system()

        #构建知识库
        self.build_knowledge_base()

        print("\n交互式问答 (输入'退出'结束):")

        while True:
            try:
                user_input = input("\n您的问题：").strip()
                if user_input.lower() in ['退出', 'exist', 'quit', '']:
                    break
                stream_choice = input("是否使用流式输出？(y/n, 默认y)").lower().strip()
                use_stream = stream_choice != 'n'#这里把回答变为bool类型

                print("\n回答")
                if use_stream:
                    #流式输出
                    for chunk in self.ask_question(user_input, stream = True):
                        print(chunk, end = "", flush= True)
                    print("\n")
                else:
                    #普通输出
                    answer = self.ask_question(user_input, stream = False)
                    print(answer, "\n", end = "")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错：{e}")
        print("\n感谢使用尝尝咸淡RAG系统！")

def main():
    """主函数
    """
    try:
        #创建RAG系统
        rag_system = RecipeRAGSystem()

        #运行交互问答
        rag_system.run_interactive()
    except Exception as e:
        logger.error(f"系统运行出错：{e}")
        print(f"系统错误:{e}")
    
if __name__ == "__main__":
    main()







        

