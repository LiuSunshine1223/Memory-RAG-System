# 双塔 Bi-Encoder（bge-small-zh-v1.5）与FAISS天作之合
# Encoder-only模型：BERT, RoBERTa, bge-small-zh-v1.5, bge-reranker-base

# SentenceTransformer 和 bge-small-zh-v1.5 的配合
# 1. SentenceTransformer负责用内置Tokenizer将文本切碎，转化成 token_id 这种数字形式的张量。
# 2. 张量给到 BGE-small-zh-v1.5后，前向传播得到张量矩阵（包含每个字词的特征）。
# 3. SentenceTransformer将张量矩阵进行pooling操作，把每个字词的特征压缩成一个固定长度的向量（512维），这个向量就代表了整句话的语义信息。
# 4. SentenceTransformer再将512维度向量 L2归一化，为了让你后续在 FAISS 里算欧氏距离或者余弦相似度更准。
from sentence_transformers import SentenceTransformer
import torch

class EmbeddingModel:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_path, device=self.device)
        print(f" Embedding 加载成功 | 设备: {self.device}")

    # 接收texts(字符串或者包含多个字符串的列表)，转化成tensor进行推理，返回对应的嵌入向量（numpy数组即浮点数数组）
    # 不使用tensor而使用numpy数组是：一,为了方便后续与FAISS库的兼容；二，Tensor 默认还会呆在 GPU 的显存，有可能会爆显存
    def get_embeddings(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)