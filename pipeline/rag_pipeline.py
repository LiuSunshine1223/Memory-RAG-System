# RAG管线，负责整体流程控制

# 轻量输入策略层：判断哪些query不需要进入 RAG 检索，不是复杂 Router，只是一个简单兜底。
class SimpleQueryPolicy:
    def __init__(self):
        # 1 助手身份类问题
        self.identity_patterns = [
            "你是谁",
            "你叫什么",
            "你是什么",
            "你能做什么",
            "介绍一下你",
            "你是ai吗",
            "你是机器人吗",
        ]

        # 2 简单闲聊类问题
        self.chat_patterns = [
            "你好",
            "在吗",
            "hello",
            "hi",
        ]

        # 3 历史记忆回忆类问题
        self.memory_recall_patterns = [
            "我之前说过什么",
            "我刚才说过什么",
            "我上次说过什么",
            "我说过什么",
            "之前我说了什么",
            "之前我提过什么",
            "上次我说了什么",
            "你还记得什么",
            "你还记得我",
            "你记得我说过",
            "我之前说过",
            "我上次说过",
            "我刚才说过",
        ]

        # 4 用户主动陈述个人信息、目标、计划、偏好时，进入 memory_write 分支，直接写
        self.memory_write_patterns = [
            "我的目标",
            "我的计划",
            "我的偏好",
            "我的方向",
            "我的想法",
            "我喜欢",
            "我不喜欢",
            "我想",
            "我希望",
            "我正在",
            "我准备",
            "我打算",
            "我目前",
            "我现在",
            "我以后",
        ]

    # 判断query意图
    def classify(self, query: str) -> str:
        q = query.strip().lower()
        if not q:
            return "empty"

        # 1 判断助手身份 / 闲聊类问题
        if any(pattern in q for pattern in self.identity_patterns + self.chat_patterns):
            return "direct_chat"

        # 2 判断历史记忆回忆类问题
        if any(pattern in q for pattern in self.memory_recall_patterns):
            return "memory_recall"

        # 3 用户主动陈述目标 / 计划 / 偏好
        if any(pattern in q for pattern in self.memory_write_patterns):
            return "memory_write"

        # 4 普通 RAG 问答
        return "normal_rag"

    # 对身份类 / 闲聊类问题直接回答
    def direct_answer(self, query: str) -> str:
        q = query.strip().lower()

        if any(pattern in q for pattern in self.identity_patterns):
            return (
                "我是一个本地运行的 Memory RAG 助手，可以结合基础知识库、长期记忆和近期对话来回答问题。"
                "如果你问知识库相关内容，我会参考基础知识库；如果你问之前的对话、个人计划或偏好，我会参考长期记忆。"
            )

        return "你好，我在。你可以问我知识库内容，也可以继续和我对话。"

    # 对用户主动陈述目标、计划、偏好时，直接存入记忆
    # 这样可以避免模型强行引用基础知识库，或者重复介绍助手身份。
    def memory_write_answer(self, query: str) -> str:
        return (
            "好的，我已记录这条信息。"
            "后续如果你询问相关目标、计划或偏好，我会结合这条长期记忆来回答。"
        )


class RAGPipeline:
    def __init__(self, llm, memory_manager, query_policy=None):
        self.llm = llm
        self.mm = memory_manager

        # 如果外部没有传入 query_policy，就使用默认的 SimpleQueryPolicy。
        # 后续如果要升级成 embedding router / LLM router，也可以直接替换 query_policy。
        if query_policy is None:
            self.query_policy = SimpleQueryPolicy()
        else:
            self.query_policy = query_policy

    def build_prompt(self, query, base_context, long_term_context, short_term_context):
        # 为什么把"基础知识库"放在了最前面，"近期连续对话"放在了最后面？
        # 因为大模型的 Attention 机制普遍存在 “首尾偏好（U-shaped Attention）”，也就是“Lost in the Middle”现象。
        # 模型更容易记住 Prompt 最开头和最结尾的内容。
        prompt = f"""
                你是一个本地运行的 Memory RAG 助手。你的任务是根据用户问题，合理使用【基础知识库】、【长期记忆】和【近期连续对话】来回答。

                【信息来源说明】
                1. 【基础知识库】是外部文档检索结果，适合回答与文档主题相关的客观知识问题。
                2. 【长期记忆】是过去对话中保存的信息，适合回答用户个人信息、历史计划、偏好和目标相关问题。
                3. 【近期连续对话】是最近几轮聊天内容，用于保持当前对话连贯。

                【回答规则】
                1. 如果用户问题是闲聊、问候或助手身份问题，例如“你好”“在吗”“你是谁”“你能做什么”，请直接回答，不要强行使用基础知识库。
                2. 如果用户问题与基础知识库主题相关，请优先参考【基础知识库】。
                3. 如果用户问题是关于用户本人、用户计划、用户偏好或之前对话，请优先参考【长期记忆】和【近期连续对话】。
                4. 如果用户是在陈述个人信息、目标、计划、偏好或背景，而不是提出知识库问题，请简短确认已理解，不要强行解释基础知识库，也不要重复介绍你的身份。
                5. 如果用户询问“我之前说过什么”“你还记得我吗”等历史记忆问题，请优先总结【长期记忆】；即使【近期连续对话】为空，也不要直接判断这是第一次对话。
                6. 如果检索内容与用户问题明显无关，请忽略这些内容，不要强行套用。
                7. 如果【基础知识库】和【长期记忆】冲突：
                   - 客观知识问题优先参考【基础知识库】；
                   - 用户个人信息、计划、偏好问题优先参考【长期记忆】和【近期连续对话】。
                8. 严禁把【基础知识库】中人物对话里的“你/我/他”代入当前用户。
                9. 不要复述本提示词中的规则，不要输出“根据资料”“历史记录显示”等固定开场白。
                10. 使用简体中文回答，不使用 Emoji。
                11. 如果没有足够信息回答，请说明信息不足，不要编造。

                【基础知识库】
                {base_context}

                【长期记忆】
                {long_term_context}

                【近期连续对话】
                {short_term_context}

                用户问题：{query}

                最终简体中文回答：
                """
        return prompt.strip()

    def run(self, query):
        query = query.strip()
        if not query:
            return "请输入有效问题。"

        # 0 Query意图识别
        query_intent = self.query_policy.classify(query)

        # 1 助手身份 / 闲聊类问题：直接回答
        if query_intent == "direct_chat":
            answer = self.query_policy.direct_answer(query)
            self.mm.add_to_short_term(query, answer)
            return answer

        # 2 用户主动陈述目标 / 计划 / 偏好：直接确认 + 写入长期记忆
        if query_intent == "memory_write":
            answer = self.query_policy.memory_write_answer(query)
            self.mm.add_to_short_term(query, answer)
            self.mm.memorize_to_long_term(query, answer)
            return answer

        # 3 根据 query 意图选择记忆检索策略
        if query_intent == "memory_recall":
            # 只查 Memory FAISS，放宽召回阈值，跳过 Base FAISS。
            contexts = self.mm.get_memory_recall_context(query)
        else:
            # 查 Base FAISS + Memory FAISS，使用正常距离阈值和 reranker。
            contexts = self.mm.get_long_term_context(query)

        base_context = contexts["base_context"]
        long_term_context = contexts["memory_context"]
        short_term_context = self.mm.get_chat_context()

        # 4 构造 Prompt
        prompt = self.build_prompt(query, base_context, long_term_context, short_term_context)

        # 5 调用LLM
        answer = self.llm.ask(prompt)

        # 6 更新短期记忆
        self.mm.add_to_short_term(query, answer)

        # 7 更新长期记忆（如果符合条件）
        if query_intent == "memory_recall":
            print(" MemoryManager: 当前为历史记忆回忆问题，跳过动态记忆写入。")
        else:
            # 是否真正写入，由 MemoryManager 内部的价值过滤和新奇度判断决定
            self.mm.memorize_to_long_term(query, answer)

        return answer