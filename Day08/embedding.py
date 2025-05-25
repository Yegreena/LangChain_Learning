import os
import openai
from openai import OpenAI
import numpy as np
from numpy.linalg import norm

# 确保你的 OpenAI API 密钥已设置在环境变量中
# 或者直接 client = OpenAI(api_key="YOUR_API_KEY")
try:
    client = OpenAI()
except openai.APIConnectionError as e:
    print("无法连接到 OpenAI API。请检查您的网络连接和 API 密钥。")
    print(e)
    exit()

# --- Embedding 函数 ---
def get_embedding(text, model="text-embedding-ada-002"):
    try:
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"获取嵌入时出错: {e}")
        return None

# --- 余弦相似度函数 ---
def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return 0
    # 确保 vec1 和 vec2 是 numpy 数组
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# --- 知识库和意图 ---
knowledge_base = [
    {
        "type": "qa",
        "question": "什么是人工智能？",
        "answer": "人工智能（Artificial Intelligence），英文缩写为AI。它是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
        "embedding": None # 稍后填充
    },
    {
        "type": "qa",
        "question": "OpenAI 是哪个国家的公司？",
        "answer": "OpenAI 是一家总部位于美国旧金山的人工智能研究和部署公司。",
        "embedding": None
    },
    {
        "type": "qa",
        "question": "什么是大型语言模型？",
        "answer": "大型语言模型（Large Language Model, LLM）是一种经过大量文本数据训练的深度学习模型，能够理解和生成人类语言。",
        "embedding": None
    },
    {
        "type": "intent",
        "intent_phrase": "预订会议室",
        "action": "好的，请告诉我您想预订哪个会议室，以及预订的日期和时间。",
        "embedding": None
    },
    {
        "type": "intent",
        "intent_phrase": "查询天气",
        "action": "请告诉我您想查询哪个城市的天气？",
        "embedding": None
    },
    {
        "type": "intent",
        "intent_phrase": "播放音乐",
        "action": "好的，您想听什么歌？",
        "embedding": None
    }
]

# --- 为知识库和意图预计算嵌入 ---
print("正在为知识库和意图生成嵌入...")
for item in knowledge_base:
    if item["type"] == "qa":
        item["embedding"] = get_embedding(item["question"])
    elif item["type"] == "intent":
        item["embedding"] = get_embedding(item["intent_phrase"])
print("嵌入生成完毕。")

# --- 问答和意图匹配函数 ---
def simple_assistant(user_query, similarity_threshold=0.75):
    query_embedding = get_embedding(user_query)
    if query_embedding is None:
        return "抱歉，处理您的请求时出现问题。"

    best_match = None
    max_similarity = -1

    for item in knowledge_base:
        if item["embedding"] is None: # 跳过无法生成嵌入的条目
            continue
        similarity = cosine_similarity(query_embedding, item["embedding"])
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = item

    if best_match and max_similarity >= similarity_threshold:
        if best_match["type"] == "qa":
            return f"回答: {best_match['answer']} (相似度: {max_similarity:.2f})"
        elif best_match["type"] == "intent":
            return f"意图匹配: {best_match['action']} (相似度: {max_similarity:.2f})"
    else:
        return f"抱歉，我不太理解您的问题或意图。 (最高相似度: {max_similarity:.2f})"

# --- 测试 ---
if __name__ == "__main__":
    print("\n--- 问答助手和意图匹配测试 ---")
    queries = [
        "AI 是什么东西？",
        "我想订个房间开会",
        "今天北京天气怎么样？",
        "我想听周杰伦的歌",
        "Python 怎么学？",
        "大型语言模型有哪些应用？"
    ]

    for query in queries:
        response = simple_assistant(query)
        print(f"\n用户: {query}")
        print(f"助手: {response}")