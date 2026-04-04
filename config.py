import os
from dotenv import load_dotenv

# .env 文件里的配置，注入系统环境
load_dotenv()

class Config:
    # 模型路径
    EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH", "./models/bge-small-zh-v1.5")
    RERANK_MODEL_PATH = os.getenv("RERANK_MODEL_PATH", "./models/bge-reranker-base")

    # Ollama 配置
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3:8b")

    # 数据路径
    DATA_PATH = os.getenv("DATA_PATH", "./data/Journey to the West.txt")


    # 向量库参数
    BASE_DB_PATH = "./storage/base_knowledge"  # 基础知识库
    MEMORY_DB_PATH = "./storage/user_memory"  # 动态记忆库（读写聊天记录）
    VECTOR_DIM = 512 # BGE-small-zh-v1.5 的默认向量维度是 512，VECTOR_DIM 需要与之匹配，否则 FAISS 会报错.

    # 切分参数
    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 50
    OVERLAP_SENTENCE = 1

    # 检索策略参数
    # 写入阈值
    THRESHOLD = 0.6 # 记忆入库的距离阈值，只有当新记忆与库中最相似的记忆的距离大于该值时，才会被认为是新的、独特的记忆并存入库中，避免重复记忆。
    #读取阈值
    BASE_MAX_DISTANCE = 1.2  # 基础库的最大容忍距离（严格）
    MEMORY_MAX_DISTANCE = 1.5  # 记忆库的最大容忍距离（宽松）
    # 基础知识库
    BASE_RECALL_K = 10 # FAISS 粗排召回数量。
    BASE_RERANK_K = 3 # BGE 精排最终保留数量。
    # 动态记忆库
    MEMORY_RECALL_K = 3
    MEMORY_RERANK_K = 1
