import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # 使用开源模型
from langchain_chroma import Chroma

# --- 准备工作 ---
# 0. 创建一个示例文本文件 knowledge_lc.txt
knowledge_content = """
LangChain 是一个用于开发由语言模型驱动的应用程序的框架。
它使得构建复杂的应用程序变得简单，例如问答、聊天机器人和内容生成。
Chroma 是一个开源的嵌入数据库，可以轻松存储和检索向量嵌入。
Hugging Face 提供了大量的预训练模型，包括用于文本嵌入的模型。
all-MiniLM-L6-v2 是一个优秀的句子嵌入模型。
"""
with open("knowledge_lc.txt", "w", encoding="utf-8") as f:
    f.write(knowledge_content)

# 1. 定义持久化目录
persist_directory = "./chroma_db_lc" # 指定 ChromaDB 存储的目录

# --- 加载、分割、嵌入和存储 ---
print("\n--- LangChain 与 ChromaDB 示例 ---")

# 2. 加载文档
print("加载文档...")
loader = TextLoader("./knowledge_lc.txt", encoding="UTF-8")
documents = loader.load()

# 3. 分割文档
print("分割文档...")
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20) # 调整块大小和重叠
docs_split = text_splitter.split_documents(documents)
print(f"分割成 {len(docs_split)} 个文档块。")
# for i, doc_chunk in enumerate(docs_split):
# print(f"块 {i}: {doc_chunk.page_content[:50]}...")


# 4. 创建嵌入函数 (使用 HuggingFace 的开源模型)
print("初始化嵌入模型...")
# model_name = "sentence-transformers/all-MiniLM-L6-v2" # 完整模型名
model_name = "all-MiniLM-L6-v2" # LangChain 内部可能会自动补全
# 如果下载慢或遇到网络问题，可以先手动下载模型到本地，然后指定本地路径
# embedding_function = HuggingFaceEmbeddings(model_name="path/to/your/local/model")
try:
    embedding_function = HuggingFaceEmbeddings(model_name=model_name)
    print(f"使用模型: {embedding_function.model_name}")
except Exception as e:
    print(f"初始化 HuggingFaceEmbeddings 失败: {e}")
    print("请确保已安装 sentence-transformers 并且可以访问 Hugging Face 模型库。")
    exit()

# 5. 将文档加载到 ChromaDB 并持久化
# 如果持久化目录已存在且包含数据，下次会尝试加载
# 如果想强制重新创建，可以先手动删除 persist_directory
if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
    print(f"创建新的 ChromaDB 并存储文档于 '{persist_directory}'...")
    db = Chroma.from_documents(
        documents=docs_split,
        embedding=embedding_function,
        persist_directory=persist_directory
    )
    print("文档已存储到 ChromaDB。")
else:
    print(f"从 '{persist_directory}' 加载现有的 ChromaDB...")
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )
    print("ChromaDB 已加载。")

# --- 查询 ---
print("\n--- 查询 ChromaDB ---")
query1 = "LangChain 有什么用？"
query2 = "什么是 Chroma？"
query3 = "Hugging Face 是做什么的"

queries_to_test = [query1, query2, query3]

for query in queries_to_test:
    print(f"\n用户查询: {query}")
    # retriever = db.as_retriever(search_kwargs={"k": 2}) # 获取前2个最相关的
    # relevant_docs = retriever.invoke(query)
    relevant_docs = db.similarity_search(query, k=2) # k 是返回结果的数量
    if relevant_docs:
        for i, doc in enumerate(relevant_docs):
            print(f"  相关文档 {i+1}: {doc.page_content[:100]}...") # 打印部分内容
            # print(f"  元数据: {doc.metadata}")
    else:
        print("  未找到相关文档。")

