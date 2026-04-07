# 粗排序模型（双塔 Bi-Encoder）与FAISS天作之合

# nn.Embedding是词表级别的查表，没有上下文信息，存放所有token_id的向量

# Encoder-only模型：BERT, RoBERTa, bge-small-zh-v1.5, bge-reranker-base

# bge-small-zh-v1.5是句子级别的编码器，本质是预训练好的Transformer Encoder-only模型（最后加了池化层），推理将一整句话只能浓缩成“一个”向量。
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