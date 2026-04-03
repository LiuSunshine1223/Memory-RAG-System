# 记忆管家
import numpy as np

class MemoryManager:
    def __init__(self, base_vs, memory_vs, embed_model, reranker_model, chunker, threshold, base_max_dist, memory_max_dist, base_recall_k, base_rerank_k, memory_recall_k, memory_rerank_k):
        self.base_vs = base_vs
        self.memory_vs = memory_vs
        self.embed_model = embed_model
        self.reranker_model = reranker_model
        self.chunker = chunker # 外部切分器
        self.threshold = threshold
        self.base_max_dist = base_max_dist
        self.memory_max_dist = memory_max_dist

        self.base_recall_k = base_recall_k
        self.base_rerank_k = base_rerank_k
        self.memory_recall_k = memory_recall_k
        self.memory_rerank_k = memory_rerank_k
        self.short_term_history = []

    # 短期记忆：只保留最近的 5 条对话，提供给 LLM 作为上下文参考，帮助它更好地理解当前对话环境和用户需求。
    def add_to_short_term(self, query, answer):
        self.short_term_history.append({"q": query, "a": answer})
        if len(self.short_term_history) > 5:
            # python中的列表功能很多，可以作为数组、栈、队列等数据结构来使用。
            # 这里的 pop(0) 就是把列表当成一个队列来用，弹出第一个元素（最早的对话），实现短期记忆的滚动更新。
            # 这是 FIFO 的策略
            self.short_term_history.pop(0)

    def memorize_to_long_term(self, query, answer):
        # llama3在阅读 记忆碎片 看到英文(User、Assistant)时，
        # 它的注意力会被带偏，更倾向于用英文回复
        # combined_text = f"User: {query} | Assistant: {answer}"
        combined_text = f"用户: {query} | AI: {answer}"
        chunks = self.chunker.split_text(combined_text)

        for chunk in chunks:
            # [0]: 一问一答的模式，该模型的缺陷处，如果一段话里有多个问题，它就处理不好了。
            new_vec = self.embed_model.get_embeddings([chunk])[0]
            # FAISS 欧氏距离判定
            # 为什么.reshape(1, -1)？因为FAISS要求你传入一个二维数组, 所以.reshape(1, -1)的作用就是把一个一维数组强行变成了一行多列的二维矩阵。
            q_vec = new_vec.astype('float32').reshape(1, -1)
            # 让 FAISS 去数据库里找“和当前这句话长得最像的唯一那 1 句话”

            # 从记忆库中查重
            if self.memory_vs.total_count > 0:
                # 常态：去库里查，算距离，判定是不是新知识
                distances, _ = self.memory_vs.index.search(q_vec, k=1)
                is_new = distances[0][0] > self.threshold
            else:
                # 冷启动：如果库是空的，别查了，它绝对是新知识！距离变量 distances 压根不需要存在！
                is_new = True

            # 只有当 L2 距离大于阈值（代表差异够大、是新信息）才存入库，避免重复记忆
            # 很有意思的地方：库里最相似的那条记忆与当前这条记忆的距离差距足够大，才说明当前这条记忆是新的、独特的，才有必要存入库里
            if is_new:
                self.memory_vs.add_texts([chunk], np.array([new_vec]))
                print(f" MemoryManager: 捕获到新知识，已存入动态记忆库中。")

    def get_chat_context(self):
        context = ""
        for turn in self.short_term_history:
            context += f"问：{turn['q']}\n答：{turn['a']}\n"
        return context

    def get_long_term_context(self, query):
        q_emb = self.embed_model.get_embeddings([query])

        # 任务 A：从黄金库调取权威资料
        base_docs = []
        if self.base_vs.total_count > 0:
            raw_base = self.base_vs.search(q_emb, self.base_recall_k, self.base_max_dist)
            if raw_base:
                base_docs = self.reranker_model.rank(query, raw_base, self.base_rerank_k)

        # 任务 B：从记忆库调取历史习惯
        memory_docs = []
        if self.memory_vs.total_count > 0:
            raw_memory = self.memory_vs.search(q_emb, self.memory_recall_k, self.memory_max_dist) # 粗排3
            if raw_memory:
                memory_docs = self.reranker_model.rank(query, raw_memory, self.memory_rerank_k) # 精排只留1

        # 把两个库的结果打包成字典返回给 Pipeline
        return {
            "base_context": "\n".join(base_docs) if base_docs else "无。",
            "memory_context": "\n".join(memory_docs) if memory_docs else "无。"
        }