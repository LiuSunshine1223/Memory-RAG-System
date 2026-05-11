# 记忆管家
import numpy as np

class MemoryManager:
    def __init__(self, base_vs, memory_vs, embed_model, reranker_model, chunker, threshold, base_max_dist, memory_max_dist, base_recall_k, base_rerank_k, memory_recall_k, memory_rerank_k):
        self.base_vs = base_vs
        self.memory_vs = memory_vs
        self.embed_model = embed_model
        self.reranker_model = reranker_model
        self.chunker = chunker # 外部切分器

        # 长期记忆写入时的新奇度阈值
        # distance > threshold 表示和已有记忆差异较大，可以视为新记忆
        self.threshold = threshold

        self.base_max_dist = base_max_dist
        self.memory_max_dist = memory_max_dist

        self.base_recall_k = base_recall_k
        self.base_rerank_k = base_rerank_k
        self.memory_recall_k = memory_recall_k
        self.memory_rerank_k = memory_rerank_k
        self.short_term_history = []

    # 1 写入路径：短期记忆写入 + 长期记忆沉淀
    # 1.1 短期记忆写入：只保留最近的 5 条对话
    def add_to_short_term(self, query, answer):
        self.short_term_history.append({"q": query, "a": answer})
        if len(self.short_term_history) > 5:
            # python中的列表功能很多，可以作为数组、栈、队列等数据结构来使用。
            # 这里的 pop(0) 就是把列表当成一个队列来用，弹出第一个元素（最早的对话），实现短期记忆的滚动更新。
            self.short_term_history.pop(0)

    # 1.2 长期记忆价值过滤：只判断“值不值得记”，不判断“新不新”
    def should_attempt_long_term_memory(self, query, answer):
        q = query.strip().lower()
        a = answer.strip()

        if not q:
            return False

        # 助手身份 / 闲聊类问题不写入长期记忆
        low_value_patterns = [
            "你是谁",
            "你叫什么",
            "你是什么",
            "你能做什么",
            "介绍一下你",
            "你是ai吗",
            "你是机器人吗",
            "你好",
            "在吗",
            "hello",
            "hi",
            "谢谢",
            "好的",
            "嗯",
            "哦",
            "哈哈",
        ]

        if any(pattern in q for pattern in low_value_patterns):
            return False

        # 明显被知识库带偏的错误回答，禁止写入长期记忆
        # 这是兜底，防止错误答案污染 Memory FAISS。
        bad_answer_patterns = [
            "我是泾河龙王",
            "泾河龙王的助手",
            "你不是秀士",
            "剐龙台",
        ]

        if any(pattern in a for pattern in bad_answer_patterns):
            return False

        # 用户个人长期状态相关信息：值得进入长期记忆候选
        memory_value_patterns = [
            "我叫",
            "我是",
            "我的",
            "我想",
            "我要",
            "我希望",
            "我计划",
            "我喜欢",
            "我不喜欢",
            "我正在",
            "我目前",
            "我以后",
            "记住",
            "帮我记",
            "别忘了",
            "目标",
            "计划",
            "求职",
            "实习",
            "简历",
            "论文",
            "项目",
            "导师",
            "学校",
            "研究方向",
            "职业规划",
            "银行",
            "大厂",
        ]

        if any(pattern in q for pattern in memory_value_patterns):
            return True

        # 普通知识问答默认不写入长期记忆
        # 例如“孙悟空是谁”由 Base FAISS 回答即可，不需要写进用户长期记忆。
        return False

    # 1.3 长期记忆新奇度判定：判断“新不新”
    def is_novel_memory(self, memory_vector):
        # FAISS 欧氏距离判定
        # 为什么.reshape(1, -1)？因为FAISS要求你传入一个二维数组, 所以.reshape(1, -1)的作用就是把一个一维数组强行变成了一行多列的二维矩阵。
        q_vec = memory_vector.astype('float32').reshape(1, -1)

        if self.memory_vs.total_count == 0:
            return True

        # 让 FAISS 去数据库里找“和当前这句话长得最像的唯一那 1 句话”
        # distances 是二维数组，形状为 (查询数量, top_k)
        # distances[0][0] 表示：第 0 个查询向量对应的 top-1 最近距离
        distances, _ = self.memory_vs.index.search(q_vec, k=1)
        nearest_distance = distances[0][0]
        return nearest_distance > self.threshold

    # 1.4 长期记忆写入流程
    def memorize_to_long_term(self, query, answer):
        # 第一道门：价值过滤
        if not self.should_attempt_long_term_memory(query, answer):
            print(" MemoryManager: 当前对话无长期记忆价值，跳过动态记忆写入。")
            return

        # 构造长期记忆文本
        # 局限：如果一轮对话里包含多个独立事实点或多个问题，
        # 它们可能会被压缩到同一个 chunk 向量中，导致检索和新奇度判断不够精细。
        memory_text = f"用户: {query} | AI: {answer}"
        chunks = self.chunker.split_text(memory_text)

        for chunk in chunks:
            # get_embeddings() 接收文本列表，即使这里只传入 1 个 chunk，
            # 也会返回一个 embedding 列表，因此用 [0] 取出当前 chunk 对应的唯一向量。
            new_vec = self.embed_model.get_embeddings([chunk])[0]
            # 转成 NumPy 的 float32 向量，满足 FAISS 检索和写入的输入格式要求
            new_vec = np.asarray(new_vec, dtype="float32")

            # 第二道门：新奇度判断
            if self.is_novel_memory(new_vec):
                self.memory_vs.add_texts([chunk], np.array([new_vec], dtype="float32"))
                print(" MemoryManager: 捕获到新知识，已存入动态记忆库中。")
            else:
                print(" MemoryManager: 与已有记忆相似，跳过重复写入。")

    # 2 检索路径：短期上下文 + 长期记忆 + 基础知识库
    # 2.1 短期上下文读取
    def get_chat_context(self):
        context = ""
        for turn in self.short_term_history:
            context += f"问：{turn['q']}\n答：{turn['a']}\n"
        return context

    # 2.2 特殊token（“我之前说过什么 / 你还记得我吗”）的历史记忆回忆检索策略
    def get_memory_recall_context(self, query):
        q_emb = self.embed_model.get_embeddings([query])

        memory_docs = []

        if self.memory_vs.total_count > 0:
            # 放宽召回距离阈值，尽量召回已有长期记忆
            raw_memory = self.memory_vs.search(q_emb, self.memory_recall_k, 4.0)
            if raw_memory:
                memory_docs = self.reranker_model.rank(query, raw_memory, self.memory_rerank_k)

        return {
            "base_context": "无。",
            "memory_context": "\n".join(memory_docs) if memory_docs else "无。",
        }

    # 2.3 正常 RAG 上下文检索策略
    def get_long_term_context(self, query):
        q_emb = self.embed_model.get_embeddings([query])

        # 任务 A：从基础库调取权威资料
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