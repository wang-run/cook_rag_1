"""
    数据准备模块
"""



from pathlib import Path
from typing import List, Dict, Any
import uuid
from langchain_core.documents import Document
import logging
from langchain_text_splitters import MarkdownHeaderTextSplitter
import pickle
import re

logger = logging.getLogger(__name__)




class DataPreparationModule:
    """数据准备模块， 负责数据加载，清洗和预处理"""
    # 统一维护的分类与难度配置，供外部复用，避免关键词重复定义
    CATEGORY_MAPPING = {
        'meat_dish': '荤菜',
        'vegetable_dish': '素菜',
        'soup': '汤品',
        'dessert': '甜品',
        'breakfast': '早餐',
        'staple': '主食',
        'aquatic': '水产',
        'condiment': '调料',
        'drink': '饮品'
    }
    CATEGORY_LABELS = list(set(CATEGORY_MAPPING.values()))
    DIFFICULTY_LABELS = ['非常简单', '简单', '中等', '困难', '非常困难']

    def __init__(self, data_path:str, save_documents_path: str):
        self.data_path = data_path
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.parent_child_map: Dict[str, str] = {}
        self.save_documents_path: str = save_documents_path


    def load_documents(self) -> List[Document]:
        """
        加载文档数据，保存document对象
        return:保存的document列表对象
        """
        logger.info(f"load_documents正在运行，正在从{self.data_path}加载文件")
        documents = []
        data_path_obj = Path(self.data_path)
        try:
            #开始对.md文档进行读取
            for md_file in data_path_obj.rglob("*.md"):
                with open(md_file, "r", encoding = "utf-8") as f:
                    content = f.read()
                #读取之后设置page_content和metadata
                parent_id = str(uuid.uuid4())

                doc = Document(
                    page_content = content,
                    metadata = {
                        "source" : str(md_file),#标记来源，md文件的目录
                        "parent_id" : parent_id,#标记唯一id
                        "doc_type" : "parent" #标记为父文档
                    }
                )
                documents.append(doc)
        except Exception as e:
            logger.warning(f"读取文件{md_file}失败:{e}")
        
        #这里对metadata进行了增强，在其中增加了一些检索过程中需要用到的元素
        for doc in documents:
            self._enhance_metadata(doc)
        
        self.documents = documents
        logger.info(f"成功加载{len(self.documents)}个文档")
        return documents
    
    def save_documents(self):
        """将document父文档保存到本地
        """
        if self.documents is None:
            raise ValueError("请先加载父文档为documents")
        logger.info(f"开始将{len(self.documents)}个父文档保存到{self.save_documents_path}")
        Path(self.save_documents_path).parent.mkdir(parents = True, exist_ok= True)
        with open(self.save_documents_path, 'wb') as f:
            pickle.dump(self.documents, f)
        logger.info(f"父文档已经保存到{self.save_documents_path}")
        


    def _enhance_metadata(self, doc:Document):
        """加强元数据

        :param doc: 需要加强的文档
        :type doc: Document
        """
        file_path = Path(doc.metadata.get("source", ""))#拿到文档的元数据
        path_parts = file_path.parts #这里通过.parts拿到路径的各个层级组件('C:\\', 'Users', '19039', 'Desktop', 'all-in-rag', 'data', 'test.txt')
        #不同的菜谱放在不同种类的文件夹当中
        #这里通过parts拿到菜谱对应的种类
        #提取菜品分类
        doc.metadata["category"] = "其他"#初始化种类名称
        for key, value in self.CATEGORY_MAPPING.items():
            if key in path_parts:
                doc.metadata["category"] = value#通过parts拿到种类
                break
        
        #提取菜品名称
        content = doc.page_content
        doc.metadata['dish_name'] = file_path.stem
        
        #分析难度等级
        stars_match = re.search(r'★{1,5}', content)

        if stars_match:
            star_count = len(stars_match.group()) # 数数有几个星星
            difficulty_map = {
                5: "非常困难",
                4: "困难",
                3: "中等",
                2: "简单",
                1: "非常简单"
            }
            doc.metadata['difficulty'] = difficulty_map.get(star_count, "未知")
        else:
            doc.metadata['difficulty'] = "未知"

        # if "★★★★★" in content:
        #     doc.metadata["difficulty"] = "非常困难"
        # elif '★★★★' in content:
        #     doc.metadata['difficulty'] = '困难'
        # elif '★★★' in content:
        #     doc.metadata['difficulty'] = '中等'
        # elif '★★' in content:
        #     doc.metadata['difficulty'] = '简单'
        # elif '★' in content:
        #     doc.metadata['difficulty'] = '非常简单'
        # else:
        #     doc.metadata['difficulty'] = '未知'
    
    @classmethod
    def get_supported_categories(cls) -> List[str]:
        """可直接提供支持的种类标签列表

        :return: _description_
        :rtype: List[str]
        """
        return cls.CATEGORY_LABELS
    
    @classmethod
    def get_supported_difficulties(cls) -> List[str]:
        """直接提供困难度标签列表

        :return: _description_
        :rtype: List[str]
        """
        return cls.DIFFICULTY_LABELS
    
    def chunk_documents(self) -> List[Document]:
        """Markdown结构感知切块

        :return: 分块后的文档列表
        :rtype: List[Document]
        """
        logger.info("正在进行markdown结构感知切块...")

        if not self.documents:
            raise ValueError("请先加载文档")
        
        #使用函数对document进行切分
        chunks = self._markdown_head_split()

        #为每个chunk添加基础元数据
        for i, chunk in enumerate(chunks):
            if "chunk_id" not in chunk.metadata:
                #如果切分器切分失败（很长文本但无切分点，很短文本未切分），生成一个id
                chunk.metadata["chunk_id"] = str(uuid.uuid4())
            chunk.metadata["batch_index"] = i #在当前批次中的索引
            chunk.metadata["chunk_size"] = len(chunk.page_content)#切分后文本的长度

        self.chunks = chunks
        logger.info(f"Markdown切分完成，并成功添加元数据，一共生成了{len(chunks)}个块")
        return chunks
    
    def _markdown_head_split(self) -> List[Document]:
        """使用markdown标题分割器进行结构化分割

        :return: 按标题结构切分的文档列表
        :rtype: List[Document]
        """
        #通过观察文档得知，最多问三级标题
        #定义要切分的标题层级
        headers_to_split_on = [
            ("#", "主标题"),  #某菜品的做法
            ("##", "二级标题"), #必备工具，计算，操作， 附加内容
            ("###", "三级标题") #附加内容中的一些其他的小标题
        ]

        #创建markdown分割器
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on= headers_to_split_on, #将刚刚设置好的标题切分点放入
            strip_headers = False #保留标题，便于理解上下文
        )
        
        all_chunks = []

        for doc in self.documents:
            try:
                #检查文档中是否含有markdown标题
                content_preview = doc.page_content[:200]
                has_headers = any(line.strip().startswith("#") for line in content_preview.split("\n"))#将前几行切分成列表，然后看列表中是否有#号开头的元素，有则说明有标题

                if not has_headers:
                    logger.warning(f"文档{doc.metadata.get("dish_name", "未知")}中没有发现markdown标题")
                    logger.debug(f"内容预览：{content_preview}")

                #对每个文档进行markdown切分
                md_chunks = markdown_splitter.split_text(doc.page_content)#这里是使用的doc.page_content进行的切分，没有继承doc的metadata

                logger.debug(f"文档{doc.metadata.get("dish_name", "未知")}被切分成了{len(md_chunks)}个chunk")

                #含有#但是文本内容比较怪异，并未完成正常切分，那么chunk应该是0或者1
                if len(md_chunks) <= 1:
                    logger.warning(f"文档{doc.metadata.get("dish_name", "未知")}内容有问题，未能按照标题分割")
                    logger.debug(f"文档预览：{content_preview}")

                #将每个块中添加父文档id和子文档id，可以通过子文档索引到父文档
                parent_id = doc.metadata.get("parent_id", "未知")

                for i, chunk in enumerate(md_chunks):
                    #为子文档分配id
                    child_id = str(uuid.uuid4())

                    #添加元数据
                    new_metadata = doc.metadata.copy()
                    new_metadata.update(chunk.metadata)
                    chunk.metadata = new_metadata#这里防止了子文档中的冲突内容被父文档覆盖
                    chunk.metadata.update(
                        {
                            "chunk_id" : child_id,
                            "parent_id" : parent_id,
                            "doc_type" : "child", #标记为子文档
                            "chunk_index" : i #在父文档中的位置
                        }
                    )

                    self.parent_child_map[child_id] = parent_id
                    

                all_chunks.extend(md_chunks)

            except Exception as e:
                logger.warning(f"文档{doc.metadata.get("dish_name", "未知")}Markdown分割失败：{e}")
                all_chunks.append(doc)

        logger.info(f"Markdown结构分割完成，生成 {len(all_chunks)} 个结构化块")
        return all_chunks


    def fillter_documents_by_category(self, category: str) ->List[Document]:
        """按分类过滤文档,将文档中为category的种类的doc返回为一个列表中

        :param category: 菜品种类
        :type category: str
        :return: 过滤后的文档
        :rtype: List[Document]
        """
        return [doc for doc in self.documents if doc.metadata.get("category", "") == category]
    
    def fillter_documents_by_difficulty(self, difficulty: str) -> List[Document]:
        """将困难度为difficulty的doc放在一个列表中

        :param difficulty: _description_
        :type difficulty: str
        :return: 返回同一等级的困难度doc
        :rtype: List[Document]
        """
        return [doc for doc in self.documents if doc.metadata.get("difficulty", "") == difficulty]
    
    def get_statistics(self) -> Dict[str, Any]:
        """统计各个种类，各个困难等级的总文档数

        :return: _description_
        :rtype: Dict[str, Any]
        """
        if not self.documents:
            return {}
        #初始化
        categories = {}
        difficulties = {}
        #开始遍历列表查看种类和困难度
        for doc in self.documents:
            category = doc.metadata.get("category", "未知")
            categories[category] = categories.get(category, 0) + 1
            difficulty = doc.metadata.get("difficulty", "未知")
            difficulties[difficulty] = doc.metadata.get(difficulty, 0) + 1

        return {
            "total_documents" : len(self.documents),
            "total_chunks" : len(self.chunks),
            "categories" : categories, #这个categories是字典类型，键是菜品种类，值是出现该菜品种类的次数
            "difficulties" : difficulties,
            "avg_chunk_size" : sum(chunk.metadata.get("chunk_size", 0) for chunk in self.chunks) if self.chunks else 0
        }
    
    def export_metadata(self, output_path: str):
        """导出数据到json文件

        :param output_path: 导出文件路径
        :type output_path: str
        """
        import json

        metadata_list = []
        for doc in self.documents:
            metadata_list.append({
                'source': doc.metadata.get('source'),
                'dish_name': doc.metadata.get('dish_name'),
                'category': doc.metadata.get('category'),
                'difficulty': doc.metadata.get('difficulty'),
                'content_length': len(doc.page_content)
            })

        with open(output_path, "w", encoding = "utf-8") as f:
            json.dump(metadata_list, f, ensure_ascii= False, indent=2)#将metadata_list加载到f文件中，同时中文不会编码为unicode，同时保持2的缩进
        
        logger.info(f"元数据已导出到{output_path}")

    def get_parent_documents(self, child_chunks: List[Document]) ->List[Document]:
        """根据子块回去对应的父文档

        :param child_chunks: _description_
        :type child_chunks: List[Document]
        :return: 子块对应的父文档，每个父文档只有一个，即多个子文档可能对应一个父文档
        :rtype: List[Document]
        """
        print(f"\n【DEBUG-照妖镜】")
        print(f"1. 第一个子块拿到的 parent_id 是: '{child_chunks[0].metadata.get('parent_id')}'")
        if self.documents:
            print(f"2. 内存里第一个父文档的 parent_id 是: '{self.documents[0].metadata.get('parent_id')}'")
        else:
            print(f"2. 💥 完蛋，self.documents 列表居然是空的！")
        print("-" * 40 + "\n")

        #统计每个父文档被匹配的次数
        parent_relevance = {}
        parent_docs_map = {}

        #收集对应的父文档和统计次数
        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id", "未知")
            if parent_id:
                #匹配到父文档则增加计数
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1
            #将匹配到的父文档根据id放入列表中
            if parent_id not in parent_docs_map:
                for doc in self.documents:
                    if doc.metadata.get("parent_id") == parent_id:
                        parent_docs_map[parent_id] = doc
                        break
            
        
        #对匹配列表根据次数列表进行排序，多的放在前面
        sorted_parent_ids = sorted(parent_relevance.keys(), key = lambda x : parent_relevance[x], reverse = True)

        parent_docs = []
        for parent_id in parent_relevance:
            if parent_id in parent_docs_map:
                parent_docs.append(parent_docs_map[parent_id])

        #收集父文档名称和相关性信息用于日志
        parent_info = []
        for doc in parent_docs:
            dish_name = doc.metadata.get("dish_name", "未知")
            parent_id = doc.metadata.get("parent_id", "")
            relevance_count = parent_relevance.get(parent_id, 0)
            parent_info.append(f"{dish_name}分为了{relevance_count}块")
        
        logger.info(f"从{len(child_chunks)}个子块中找到了{len(parent_docs)}个去重父文档：{", ".join(parent_info)}")
        return parent_docs


    
        

