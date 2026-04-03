# 文本分割器
import re
# 1. 固定长度切分器：简单粗暴，直接按照固定长度切分文本，适合对文本结构不敏感的场景。
# 还有一个缺点：因为重叠部分，所以切出的数据量比原本多，会占用更多的资源。
class FixedLengthChunker:
    def __init__(self, chunk_size, chunk_overlap):
        # chunk_size: 每个文本块的最大长度
        # chunk_overlap: 相邻文本块之间的重叠部分长度。决定了扫描仪在往前挪动时，必须要倒退多少字，防止一句话被硬生生切断后。
        # chunk_overlap: 相邻文本块之间的重叠部分长度。决定了扫描仪在往前挪动时，必须要倒退多少字，防止一句话被硬生生切断后。
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):

        # 处理特殊情况：如果文本本身就短于 chunk_size，直接返回一个包含原文本的列表，避免不必要的切分。
        if len(text) <= self.chunk_size:
            return [text]
        # chunks作用：用于存储多个已经切好的重叠部分内容的文本块
        # start作用：记录当前扫描位置的起点（类似头指针）
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            # 切片溢出保护：Python 的切片在遇到右边界溢出时，会自动截断到字符串的最末尾。
            chunk = text[start:end]
            chunks.append(chunk)
            start += (self.chunk_size - self.chunk_overlap)
        return chunks

# 2. 递归切分器：优雅地处理各种文本结构，最大程度保留上下文连续性，适合对文本结构敏感的场景。
class RecursiveSplitter:
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size
        # 核心灵魂：降级切分符列表 (从大到小：双回车段落 -> 单回车换行 -> 句号 -> 逗号 -> 强行无情切断)
        self.separators = ["\n\n", "\n", "。", "，", ""]

    def split_text(self, text, sep_index=0):
        """真正的递归函数：不停地调用自己，直到切块满足大小限制"""

        # 递归的终止条件：如果当前文本已经小于限制，直接安全返回
        if len(text) <= self.chunk_size:
            return [text]

        # 拿到当前的“刀”（分隔符）
        current_sep = self.separators[sep_index]

        # 极端兜底：如果所有标点符号都用完了（到了 ""），只能像你之前的代码那样强行切片了
        if current_sep == "":
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        # 按照当前的标点符号（比如段落 \n\n）把文章大卸八块
        splits = text.split(current_sep)

        final_chunks = []
        for split in splits:
            if len(split) <= self.chunk_size:
                final_chunks.append(split)  # 完美，这段话没超标，保留！
            else:
                # 核心递归点：发现有个超级长的段落，超过了 400 字！
                # 怎么办？换一把更细的刀（sep_index + 1，比如换成按句号切），去专门对付这个长段落！
                sub_chunks = self.split_text(split, sep_index + 1)
                final_chunks.extend(sub_chunks)

        return final_chunks

# 3. 语义切分器：在保证语义完整的前提下，尽可能地切分文本，适合对文本结构和语义连续性都敏感的场景。
class SemanticChunker:
    def __init__(self, max_chunk_size, overlap_sentences):
        """
        max_chunk_size: 每个切片的最大字数（建议 300-400）
        overlap_sentences: 重叠的句子数量，而不是字数！
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences

    def split_text(self, text):
        if not text:
            return []

        # 1. 核心正则 Split：按句号、叹号、问号、省略号、换行切分，且保留标点！
        # (?<=...) 是正则的后行断言，意思是“在这个标点符号之后切开”，这样标点符号就会留在上一句的末尾。
        sentences = re.split(r'(?<=[。！？\n])', text)

        # 去除切分后可能产生的纯空格或空字符串
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk_sentences = []  # 存放当前切片里的句子列表
        current_length = 0

        # 2. 贪心 Merge：把短句打包成满足 max_chunk_size 的块
        for sentence in sentences:
            sentence_len = len(sentence)

            # 极端情况兜底：如果单单这一句话，就已经超过了最大限制
            if sentence_len > self.max_chunk_size:
                # 先把现有的装箱
                if current_chunk_sentences:
                    chunks.append("".join(current_chunk_sentences))
                    current_chunk_sentences = []
                    current_length = 0

                # 针对这句超长的话，只能退化为极其粗暴的固定切分
                for i in range(0, sentence_len, self.max_chunk_size):
                    chunks.append(sentence[i:i + self.max_chunk_size])
                continue

            # 正常情况判断：如果装下这句话就超载了，就把现有的装箱
            if current_length + sentence_len > self.max_chunk_size:
                chunks.append("".join(current_chunk_sentences))

                # 【核心逻辑】：处理语义级别的 Overlap（重叠上一块的最后 N 句话）
                if self.overlap_sentences > 0:
                    # 保留上一个块的最后几句话，作为新块的开头
                    current_chunk_sentences = current_chunk_sentences[-self.overlap_sentences:]
                    current_length = sum(len(s) for s in current_chunk_sentences)
                else:
                    current_chunk_sentences = []
                    current_length = 0

            # 把这句话装进当前盒子
            current_chunk_sentences.append(sentence)
            current_length += sentence_len

        # 收尾：把最后剩下的还没装箱的句子装箱
        if current_chunk_sentences:
            chunks.append("".join(current_chunk_sentences))

        return chunks