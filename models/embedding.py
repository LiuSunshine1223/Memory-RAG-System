# 双塔 Bi-Encoder（bge-small-zh-v1.5）与FAISS天作之合
# Encoder-only模型：BERT, RoBERTa, bge-small-zh-v1.5, bge-reranker-base

# SentenceTransformer 和 bge-small-zh-v1.5 的配合
# 1. SentenceTransformer 作为外层封装框架，调用 bge-small-zh-v1.5 自带的 Tokenizer，将原始文本切分并转换成 token_id、attention_mask 等张量输入。
# 2. bge-small-zh-v1.5 接收这些张量后，先根据 token_id 查找内部已经训练好的token embedding 矩阵，得到每个 token 的初始向量表示。
# 3. bge-small-zh-v1.5 再通过 Transformer Encoder 进行前向传播，结合上下文信息，得到每个 token 的上下文语义表示矩阵。
# 4. SentenceTransformer 根据模型配置对 token 级别的表示进行 pooling，将多个 token 的特征压缩成一个固定长度的句向量。
#    对于 bge-small-zh-v1.5，最终得到的是 512 维向量。
# 5. SentenceTransformer 可根据 encode() 参数对句向量进行 L2 归一化，使向量长度变为 1，方便后续在 FAISS 中进行余弦相似度或内积检索。
from sentence_transformers import SentenceTransformer
import torch

class EmbeddingModel:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 使用 SentenceTransformer 加载 bge-small-zh-v1.5 embedding 模型
        # 这里的 SentenceTransformer 是通用封装框架，负责 encode、pooling、归一化等流程
        # bge-small-zh-v1.5 才是实际负责生成中文语义向量的模型
        self.model = SentenceTransformer(model_path, device=self.device)
        print(f" Embedding 加载成功 | 设备: {self.device}")

    # 接收texts(字符串或者包含多个字符串的列表)，转化成tensor进行推理，返回对应的嵌入向量（numpy数组即浮点数数组）
    # 不使用tensor而使用numpy数组是：一,为了方便后续与FAISS库的兼容；二，Tensor 默认还会呆在 GPU 的显存，有可能会爆显存
    def get_embeddings(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)