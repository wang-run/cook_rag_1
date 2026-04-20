"""
生成集成模块，主要为路由模块
"""

import os
import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models.moonshot import MoonshotChat #使用kimi的聊天大模型
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    """生成集成模块 - 负责LLM集成和回答生成
    """

    def __init__(self, model_name : str = "kimi-k2-0711-preview", temperature : float = 0.1, max_tokens: int = 2048):
        """初始化生成集成模块，主要初始化模型

        :param model_name: 使用模型名称, defaults to "kimi-k2-0711-preview"
        :type model_name: str, optional
        :param temperature: 生成温度, defaults to 0.1
        :type temperature: float, optional
        :param max_tokens: 最大token数, defaults to 2048
        :type max_tokens: int, optional
        """
        self.modle_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()

    def setup_llm(self):
        """构建语言大模型
        """
        logger.info(f"开始构建llm：{self.modle_name}")

        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("请先设置MOONSHOT_API_KEY环境变量")
        
        self.llm = MoonshotChat(
            model = self.modle_name,
            temperature = self.temperature,
            max_tokens = self.max_tokens,
            moonshot_api_key = api_key
        )
        logger.info("语言模型初始化完成")

    def generate_basic_answer(self, query:str, context_docs:List[Document]) ->str:
        """根据检索出来的文档生成基础回答

        :param query: 查询问题
        :type query: str
        :param context_docs: 匹配到的上下文列表
        :type context_docs: List[Document]
        :return: 生成的回答
        :rtype: str
        """
        context = self._build_context(context_docs)#_build_context是用来构建上下文字符串的

        prompt = ChatPromptTemplate.from_template(
            """
            你是一位专业的烹饪助手。请根据以下食谱信息回答用户问题。

            用户问题：{question}

            相关食谱信息：
            {context}

            请提供详细、实用的回答。如果信息不足，请诚实说明。

            回答：
            """
        )

        #使用LCEL构建链
        chain = (
            {"question" : RunnablePassthrough(), "context" : lambda _ : context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response
    
    def generate_step_by_step_answer(self, query: str, context_docs : List[Document]) -> str:
        """生成分步骤回答

        :param query: 用户查询
        :type query: str
        :param context_docs: 匹配到的上下文档列表
        :type context_docs: List[Document]
        :return: 分步骤的详细回答
        :rtype: str
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template(
            """
            你是一维专业的烹饪导师。请根据食谱信息，为用户提供详细的分步骤指导。

            用户问题：{question}

            相关食谱信息：
            {context}

            请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

            ##🥘菜品介绍
            [简要介绍菜品特点和难度]

            ##🛒所需食材
            [列出主要食材和用量]

            ##👨‍🍳制作步骤
            [详细的分步骤说明，每一步包含具体操作和大概所需时间]

            ##💡制作技巧
            [仅在有实用技巧时包含。优先使用原文中的实用技巧，如果原文的"附加内容"与烹饪无关或为空，可以基于制作步骤总结关键要点，或者完全省略此部分]


            注意：
            -根据实际内容灵活调整结构
            -不要强行填充无关内容或重复制作堵步骤中的信息
            -重点突出实用性和可操作性
            -如果没有额外的技巧要分享，可以省略制作技巧部分

            回答：
            """
        )

        chain = (
            {"question" : RunnablePassthrough(), "context" : lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        response = chain.invoke(query)
        return response
        
    
    def query_rewrite(self, query: str) -> dict: # 建议返回类型标为 dict
        """智能查询重写 - 让大模型判断是否需要查询重写"""

        prompt = PromptTemplate(
            template="""
            任务：
            你是一个烹饪领域的查询分析专家。请分析用户的输入，提取元数据过滤条件（难度和种类），并根据查询的明确程度决定是否需要重写。
            原始查询：{query}

            第一阶段：元数据提取规则
            从查询中识别以下两项。如果用户完全没有提到相关信息，请将对应项设为 null，严禁强行猜测或联想。
            1. 难度映射标准：
            “一颗星 / 非常简单 / 极其简单” -> 非常简单
            “两颗星 / 简单 / 好做” -> 简单
            “三颗星 / 中等 / 普通难度” -> 中等
            “四颗星 / 困难 / 有挑战” -> 困难
            “五颗星 / 非常困难 / 厨神级” -> 非常困难
            2. 菜品种类映射标准：
            “饮料 / 喝的 / 饮品 / 水” -> 饮品
            “海鲜 / 鱼 / 虾 / 蟹” -> 水产
            其他分类直接映射：荤菜、素菜、汤品、甜品、早餐、主食、调料

            第二阶段：查询改写逻辑
            规则 1：直接返回（不重写）
            触发条件：包含具体菜品名称或具体制作技巧。
            动作：rewrite_query 保持用户原话。
            
            规则 2：重写增强（需重写）
            触发条件：查询涉及口语化表达或过于宽泛。
            动作：补全语义并替换为标准词。

            第三阶段：输出格式要求
            请严格按照 JSON 格式输出，不要包含任何多余的解释：
            {{
                "filters": {{
                    "difficulty": "映射后的标准难度名称或 null",
                    "category": "映射后的标准分类名称或 null"
                }},
                "rewrite_query": "重写后的查询字符串或用户原话",
                "is_clear": true
            }}
            """,
            input_variables=["query"]
        )

        # 注意：在 PromptTemplate 中，如果你想表示原始的大括号，需要写双大括号 {{ }}
        # 否则 Python 会把它当成变量占位符导致报错

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | JsonOutputParser()
        )

        try:
            response = chain.invoke(query)
            
            # 安全地获取值
            is_clear = response.get("is_clear", True)
            rewrite_query = response.get("rewrite_query", query)

            # 记录重写结果
            if not is_clear: # 如果不是明确查询，说明发生了重写
                logger.info(f"原查询 [{query}] 重写为 [{rewrite_query}]")
            else:
                logger.info(f"原查询无需重写：[{query}]")

            return response

        except Exception as e:
            logger.error(f"查询重写解析失败: {e}")
            # 发生异常时进行兜底，保证主流程不崩
            return {
                "filters": {"difficulty": None, "category": None},
                "rewrite_query": query,
                "is_clear": True
            }
    
    def query_router(self, query : str) -> str:
        """查询路由 - 根据查询类型选择不同的处理方式

        :param query: 用户查询
        :type query: str
        :return: 经过大模型分析后的路由类型('list', 'detail', 'general', 'unknown')
        :rtype: str
        """
        prompt = ChatPromptTemplate.from_template(
            """
            根据用户的问题，将其分类为以下四种类型之一：
            1、'list' - 用户想要获取菜品列表或推荐，只需要菜名
                例如：推荐几个素材、有什么川菜、给我3个简单的菜

            2、'detail' - 用户想要指导某个菜品的具体制作方法或详细信息
                例如：宫保鸡丁怎么做、制作步骤、需要什么食材

            3、'general' - 关于美食的一些一般性问题
                例如：什么是川菜、制作技巧、营养价值
            
            4、'unknown' - 与美食、菜谱、烹饪无关的问题，或无意义的字符。
                例如：写代码、问天气、asdkjfa（乱码）

            【特殊规则】
            当用户的查询中包含以上多个意图，请按照 detail > list > general > unknown 的优先级进行分类
            例如：“推荐几个川菜，另外回锅肉具体怎么做？” 请以'detail'为最高优先级进行返回

            请严格按照以下格式输出，不要包含任何标点符号，解释性文字或客套话：
            只返回分类结果：list、detail 或 general,unknown

            用户问题：{query}

            分类结果：
            """
        )
        chain = (
            {"query" : RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        result = chain.invoke(query).strip().lower()

        #确保返回的是有效的路由类型
        if result in ['list', 'detail', 'general', 'unknown']:
            return result
        else:
            return 'unknown' #默认类型
        

    def generate_list_answer(self, query : str, context_docs: List[Document]) -> str:
        """生成列表式回答 - 使用于推荐类查询（当路由为list时，需要推荐相关菜品名称）

        :param query: 用户关于菜品名称查询
        :type query: str
        :param context_docs: 匹配到的相关上下文档
        :type context_docs: List[Document]
        :return: 列表式回答
        :rtype: str
        """
        if not context_docs:
            return "抱歉，没有找到相关的菜品信息。"
        
        #根据匹配到的相关文档提取菜品名称
        dish_names = []
        for doc in context_docs:
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            #对匹配到的相同菜名的子文档进行去重
            if dish_name not in dish_names:
                dish_names.append(dish_name)

        #构建简洁的列表回答
        if len(dish_names) == 1:
            return f"为您推荐：{dish_names[0]}"
        elif len(dish_names) <= 3:
            return f"为您推荐以下菜品:\n" + "\n".join([f"{i + 1}. {dish_name}" for i, dish_name in enumerate(dish_names)])
        else:
            return f"为您推荐以下菜品：\n" + "\n".join([f"{i + 1}. {dish_name}" for i, dish_name in enumerate(dish_names)]) + f"\n\n还有其他{len(dish_names) - 3}个菜品可选择"
    
    def generate_basic_answer_stream(self, query:str, context_docs: List[Document]):
        """生成基础回答- 使用流式输出

        :param query: 用户关于general的查询
        :type query: str
        :param context_docs: 相关的上下文文档
        :type context_docs: List[Document]
        :return: 用于回答基础问题的str
        :rtype: str
        """

        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template(
            """
            你是一个专业的烹饪助手。请根据以下食谱的信息来回答用户问题

            用户问题：{question}

            相关食谱信息：
            {context}

            回答：
            """
        )
        
        chain = (
            {"question" : RunnablePassthrough(), "context" : lambda _ : context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def generate_step_by_step_answer_stream(self, query: str, context_docs : List[Document]):
        """生成细节回答 - 流式输出

        :param query: 用户关于细节的查询
        :type query: str
        :param context_docs: 相关上下文文档
        :type context_docs: List[Document]
        :return: 回答用户的str
        :rtype: str
        """
        context = self._build_context(context_docs)

        

        prompt = ChatPromptTemplate.from_template(
            """
            你是一位专业的烹饪导师。请根据食谱信息，为用户提供详细的分步骤指导。

            用户问题: {question}

            相关食谱信息:
            {context}

            请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

            ## 🥘 菜品介绍
            [简要介绍菜品特点和难度]

            ## 🛒 所需食材
            [列出主要食材和用量]

            ## 👨‍🍳 制作步骤
            [详细的分步骤说明，每步包含具体操作和大概所需时间]

            ## 💡 制作技巧
            [仅在有实用技巧时包含。如果原文的"附加内容"与烹饪无关或为空，可以基于制作步骤总结关键要点，或者完全省略此部分]

            注意：
            - 根据实际内容灵活调整结构
            - 不要强行填充无关内容
            - 重点突出实用性和可操作性

            回答：
            """
        )

        chain = (
            {"question" : RunnablePassthrough(), "context" : lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def generate_unknown(self, query:str):
        """对用户提出的与菜谱无关的问题的回答

        :param query: 用户查询
        :type query: str
        """
        prompt = ChatPromptTemplate.from_template(
            """
            你是一位乐于助人的美食与菜谱 AI 助手。
            用户目前问了一个与菜谱无关的问题。请根据你广泛的通用知识来尽量解答。
            当你真的不知道时，请礼貌地回答不知道，并引导用户问你关于做菜的问题。

            用户问题：{query}

            回答：
            """
        )
        chain = (
            {"query" : RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        for chunk in chain.stream(query):
            yield chunk
    
    def _build_context(self, docs: List[Document], max_length: int = 2000) -> str:
        """将多个document对象合并成字符串

        :param docs: 需要合并的document列表
        :type docs: List[Document]
        :param max_length: 最大长度, defaults to 2000
        :type max_length: int, optional
        :return: 合并后的上下文字符串
        :rtype: str
        """

        if not docs:
            return "暂无相关食谱信息"
        
        context_parts = []
        current_length = 0

        for i, doc in enumerate(docs):
            #提取文档的相关信息，后面一起放在字符串中
            metadata_info = f"【食谱{i}】"
            if "dish_name" in doc.metadata:
                metadata_info += f"名称:{doc.metadata['dish_name']}"
            if "category" in doc.metadata:
                metadata_info += f"| 分类：{doc.metadata['category']}"
            if "difficulty" in doc.metadata:
                metadata_info += f"| 难度等级：{doc.metadata['difficulty']}"

            #构建文档文本
            doc_text = f"{metadata_info}\n{doc.page_content}"

            #检查长度，防止超出模型的上下文长度
            if current_length + len(doc_text) > max_length:
                break
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        #这里返回的时候再将列表中的字符串拼接在一起
        return "\n" + "="*50 + "\n".join(context_parts)

    