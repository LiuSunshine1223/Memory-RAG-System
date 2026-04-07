# 向量存储库：负责管理FAISS(C++构建)索引和文本内容的持久化存储。

import faiss
import os
import numpy as np
import pickle

class VectorStore:
    def __init__(self, storage_path, dimension):
        self.storage_path = storage_path
        self.dimension = int(dimension)  # 强制转为整数，防止 FAISS 报错

        # “双轨制”存储
        # faiss.index存储向量内容，texts.pkl存储文本块(3-5句话)内容，二者通过索引位置一一对应
        self.index_file = os.path.join(storage_path, "faiss.index") # 向量索引的文件路径
        self.text_file = os.path.join(storage_path, "texts.pkl") # 文本内容的文件路径

        # 自动创建存储目录
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)

        # 初始化或加载现有索引
        if os.path.exists(self.index_file):
            print(f" 发现现有索引，正在从 {storage_path} 加载...")
            # index和texts全部读取到cpu内存
            self.index = faiss.read_index(self.index_file)
            with open(self.text_file, 'rb') as f:
                self.texts = pickle.load(f)
        else:
            print(f" 未发现现有索引，正在初始化新索引 (维度: {self.dimension})")
            # .IndexFlatL2(self.dimension): 在内存中申请分配一块全新的、空的 C++ 结构体，里面向量维度为 self.dimension，距离度量方式为 L2（欧氏距离）。
            self.index = faiss.IndexFlatL2(self.dimension)
            self.texts = []

    def add_texts(self, texts, embeddings):
        if not texts:
            return

        # 将粗排序模型生成的嵌入向量列表转换为 numpy 数组，并强制类型为 float32，以满足 FAISS 的输入要求。
        embeddings_np = np.array(embeddings).astype('float32')

        # 执行入库
        self.index.add(embeddings_np)
        self.texts.extend(texts)

        # 实时保存到硬盘（实现持久化记忆）
        self.save()

    # 粗筛召回：搜索最相似的 recall_k 条文本
    def search(self, query_vector, recall_k, max_distance=None):

        if self.index.ntotal == 0:
            return []

        query_vector = np.array(query_vector).astype('float32')
        # FAISS 搜索返回 距离(distances) 和 索引(indices)
        # distances = 各维度差的平方和，越小代表越相似；indices = 对应文本在 self.texts 中的索引位置
        distances, indices = self.index.search(query_vector, recall_k)
        results = []
        # 因为本模型的设计初衷就是一个问题，一个回答
        # 所以是indices[0]，而不是indices。
        for idx, i in enumerate(indices[0]): # 如[1024, 55, 33]
            if i != -1 and i < len(self.texts):
                # 重要修改：如果设置了最大容忍距离，且当前切片的距离大于该值，直接拒收！
                if max_distance is not None and distances[0][idx] > max_distance:
                    continue # 距离太远，判定为无关废话，跳过
                results.append(self.texts[i]) # 找第1024条文本，第55条文本，第33条文本添加到结果列表

        return results

    def save(self):
        # 将self.index对象保存到self.index_file路径下(D:\data\faiss.index)
        faiss.write_index(self.index, self.index_file)
        # 将文本存在self.text_file路径下(D:\data\texts.pkl)，使用pickle序列化
        with open(self.text_file, 'wb') as f:
            pickle.dump(self.texts, f)

    @property # 只读（Read-only），显示记忆容量
    def total_count(self):
        return self.index.ntotal