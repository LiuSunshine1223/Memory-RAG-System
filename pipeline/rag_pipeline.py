# RAG管线，负责整体流程控制

class RAGPipeline:
    def __init__(self, llm, memory_manager):
        self.llm = llm
        self.mm = memory_manager

    def run(self, query):
        # 1. 获取长期记忆和短期记忆
        # 长期记忆（FAISS 捞出来的东西）： 属于“背景资料”或“客观词典”。它应该被放在前面，作为底层的语境铺垫。
        # 短期记忆（最近 5 轮聊天）： 属于“当前工作台状态”。它承载着用户刚才的情绪和语境断点。
        contexts = self.mm.get_long_term_context(query)
        permanent_context = contexts["base_context"]
        long_term_context = contexts["memory_context"]
        short_term_context = self.mm.get_chat_context()

        # 2. 混合 Prompt 构造(工业级严厉约束)
        # 为什么把"权威参考资料"放在了最前面，"近期连续对话"放在了最后面？
        # 因为大模型的 Attention 机制普遍存在 “首尾偏好（U-shaped Attention）”，也就是“Lost in the Middle”现象。
        # 模型更容易记住 Prompt 最开头和最结尾的内容。

        prompt = f"""
        你是一个极其严谨、没有私人情感的 AI 知识库解答引擎。请严格遵循以下法则，任何违反都将导致系统崩溃：

        【绝对禁令（最高优先级）】
        1. 格式封死：严禁使用任何 Emoji 表情符号（如 🙅‍♂️、🚀 等）！严禁使用“根据资料”、“历史记录显示”等开场白！
        2. 语言封死：必须 100% 使用纯正的简体中文，严禁输出拼音、英文或任何小语种。
        3. 身份隔离：【权威参考资料】中的内容是书中人物的对话，严禁将其中的“你/我/他”代入到当前提问的用户身上！严禁对用户进行角色扮演或道德指责！

        【逻辑判定法则】
        如果【历史聊天记忆】中的事实与【权威参考资料】发生冲突，必须无条件判定【历史聊天记忆】是用户的错误幻觉，并极其坚定地以【权威参考资料】为准！

        【权威参考资料】（只读原著，绝对真实）:
        {permanent_context}

        【历史聊天记忆】（过去的对话，可能包含错误）:
        {long_term_context}

        【近期连续对话】:
        {short_term_context}

        用户问题: {query}
        最终简体中文回答:"""

        # 3. 请求 Ollama 并触发记忆更新
        answer = self.llm.ask(prompt)
        self.mm.add_to_short_term(query, answer)
        self.mm.memorize_to_long_term(query, answer)

        return answer