# 交叉编码器 Cross-Encoder（bge-reranker-base）

from FlagEmbedding import FlagReranker

class RerankerModel:
    def __init__(self, model_path):
        # “精度”指的就是张量（Tensor）里面，每一个具体数值的存储规格。
        # 数值精度越高（更精细的表示），模型性能不一定越好，甚至可能更差。
        # 默认模型的精度是fp32(单精度)
        # use_fp16=True：使用半精度（fp16）进行推理，减少显存占用和加速计算，但可能会略微降低精度。
        self.model = FlagReranker(model_path, use_fp16=True)
        print(" Rerank 模型加载完毕！")

    def rank(self, query, docs, rerank_k):
        """
        对候选文档进行重排序
        query: 用户问题
        docs: FAISS粗排召回的文档列表
        rerank_k: 最终想保留的文档条数
        return 排序后的 top-K 文档列表
        """
        if not docs:
            return []

        # 构造输入对
        # .strip(): 去除文档(字符串)前后的空格字符(还包含换行符\n、制表符\t)
        # 只有当 doc.strip() 不为空字符串时，才会被包含在 pairs 中，这样可以避免将纯空白的文档传递给模型进行评分。
        pairs = [[query, doc] for doc in docs if doc.strip()]
        if not pairs:
            return []

        # 计算分数
        # 将pairs插入特殊标记词如[CLS]、[SEP]，并进行编码(先tokenize分词, 再Embedding查表)，
        # 虽然是cross-encoder模型，但使用的是self-attention机制，新pairs与已知的Wq、Wk、Wv权重矩阵进行计算，进行前向传播(即推理)
        # 最终输出一个分数（score），表示query与doc的相关程度。
        scores = self.model.compute_score(pairs)

        # 兼容单条数据的返回值，使其变为列表形式，方便后续处理
        # isinstance(Variable, Type)：判断一个变量是不是某种特定的数据类型
        if isinstance(scores, float):
            scores = [scores]

        # 打包并按分数降序排列
        # 缝合数据
        doc_score_pairs = list(zip([p[1] for p in pairs], scores))
        # pairs 是 [[query, doc1], [query, doc2], ...] 的形式，scores 是 [score1, score2, ...] 的形式
        # 因为query是已经在上下文出现了，所以我们只关心文档部分p[1]
        # 通过列表推导式 [p[1] for p in pairs] 提取出文档部分，即：[doc1, doc2, ...]
        # zip() 函数把两个列表相同位置的元素一对一咬合起来，变成一个个元组，即：(doc1, score1), (doc2, score2)
        # list() 函数将 zip 对象转换为列表，即：[(doc1, score1), (doc2, score2), ...]

        # 洗牌降序排序
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        # sort() 方法对列表进行原地排序。默认从小到大，但通过指定 key 和 reverse 参数，我们可以自定义排序的方式。
        # key=lambda x: x[1] 表示按照每个元组的第二个元素（即分数）进行排序，reverse=True 表示降序排列。

        # 提取前 rerank_k 个文档
        top_docs = [doc for doc, score in doc_score_pairs[:rerank_k]]
        return top_docs