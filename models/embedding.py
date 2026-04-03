# 粗排序模型（双塔 Bi-Encoder）与FAISS天作之合

# nn.Embedding是词表级别的查表，没有上下文信息，存放所有token_id的向量
# sentence_transformers是句子级别的编码器，本质是预训练好的Transformer模型（如BERT、RoBERTa等）的封装，专门用于生成句子或文本的嵌入向量。
# 它通过对输入文本进行编码，捕捉上下文信息和语义关系，从而生成高质量的文本表示。这些嵌入向量可以用于各种自然语言处理任务，如文本分类、聚类、相似度计算等。
from sentence_transformers import SentenceTransformer
import torch

class EmbeddingModel:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 本地加载预训练好的 SentenceTransformer 模型，指定设备
        self.model = SentenceTransformer(model_path, device=self.device)
        print(f" Embedding 加载成功 | 设备: {self.device}")

    def get_embeddings(self, texts):
        # 接收texts(字符串或者包含多个字符串的列表)，转化成tensor进行推理，返回对应的嵌入向量（numpy数组即浮点数数组）
        # 不使用tensor而使用numpy数组是：一,为了方便后续与FAISS库的兼容；二，Tensor 默认还会呆在 GPU 的显存，会爆显存
        return self.model.encode(texts, convert_to_numpy=True)