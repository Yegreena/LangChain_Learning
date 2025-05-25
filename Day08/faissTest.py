import faiss
import numpy as np
from openai import OpenAI

# 确保你的 OpenAI API 密钥已设置在环境变量中
try:
    client = OpenAI()
except Exception as e:
    print("无法连接到 OpenAI API。请检查您的网络连接和 API 密钥。")
    print(e)
    exit()

# --- Embedding 函数 ---
def get_embedding_faiss(text, model="text-embedding-ada-002"):
    try:
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"获取嵌入时出错 for '{text}': {e}")
        return None

# --- 示例文本 ---
print("\n--- FAISS 相似性搜索示例 ---")
sentences = [
    "我喜欢吃苹果。",
    "今天天气真好，适合散步。",
    "香蕉是一种美味的水果。",
    "编程是一项有趣的技能。",
    "人工智能正在改变世界。",
    "我喜欢在公园里跑步。",
    "橙子富含维生素C。"
]

# 1. 为所有句子生成嵌入
print("为句子生成嵌入...")
embeddings_list = []
valid_sentences = []
for sentence in sentences:
    emb = get_embedding_faiss(sentence)
    if emb:
        embeddings_list.append(emb)
        valid_sentences.append(sentence)

if not embeddings_list:
    print("未能成功生成任何嵌入，无法继续。")
    exit()

sentence_embeddings = np.array(embeddings_list).astype('float32') # FAISS 需要 float32
print(f"成功生成 {len(valid_sentences)} 个嵌入。")

# 2. 构建 FAISS 索引
dimension = sentence_embeddings.shape[1]  # 获取嵌入的维度
# 使用 IndexFlatL2 进行精确的 L2 距离搜索 (欧氏距离)
# 对于大数据集，通常使用近似索引，如 IndexIVFFlat, IndexHNSWFlat 等
index = faiss.IndexFlatL2(dimension)
print(f"FAISS 索引已创建，维度: {dimension}, 类型: IndexFlatL2")

# 3. 将嵌入添加到索引
index.add(sentence_embeddings)
print(f"已将 {index.ntotal} 个向量添加到索引中。")

# 4. 定义查询并进行搜索
query_sentence = "我喜欢的水果是什么？"
print(f"\n用户查询: {query_sentence}")

query_embedding = get_embedding_faiss(query_sentence)
if query_embedding is None:
    print("未能生成查询嵌入，无法搜索。")
    exit()

query_embedding_np = np.array([query_embedding]).astype('float32') # FAISS search 需要 2D 数组

k = 3  # 查找最相似的3个结果
distances, indices = index.search(query_embedding_np, k)

print(f"\n最相似的 {k} 个结果:")
for i in range(k):
    idx = indices[0][i]
    dist = distances[0][i]
    print(f"  排名 {i+1}: \"{valid_sentences[idx]}\" (L2 距离: {dist:.4f})")

query_sentence_2 = "户外活动有哪些？"
print(f"\n用户查询: {query_sentence_2}")
query_embedding_2 = get_embedding_faiss(query_sentence_2)
if query_embedding_2:
    query_embedding_np_2 = np.array([query_embedding_2]).astype('float32')
    distances_2, indices_2 = index.search(query_embedding_np_2, k)
    print(f"\n最相似的 {k} 个结果:")
    for i in range(k):
        idx = indices_2[0][i]
        dist = distances_2[0][i]
        print(f"  排名 {i+1}: \"{valid_sentences[idx]}\" (L2 距离: {dist:.4f})")
else:
    print("未能生成查询嵌入，无法搜索。")