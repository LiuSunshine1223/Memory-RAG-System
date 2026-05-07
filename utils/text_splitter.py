# 文本分割器
import re
# 1. 固定长度切分器：简单粗暴，直接按照固定长度切分文本，适合对文本结构不敏感的场景。
# 还有一个缺点：因为重叠部分，所以切出的数据量比原本多，会占用更多的资源。
class FixedLengthChunker:
    def __init__(self, chunk_size, chunk_overlap):
        # chunk_size: 每个文本块的最大长度
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

# 2. 句子边界切分器：在保证语义完整的前提下，尽可能地切分文本，适合对文本结构和语义连续性都敏感的场景。
class SentenceBoundaryChunker:
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

        # 1. 终极正则 findall
        # 不使用 split的原因：因为 re.split(r'(?<=[！])', text)会将 "太棒了！！！" 中后续两个感叹号独立开，这个语义切分就失败了。
        # r : 正则表达式的起始标志。
        # [] : 定义一个字符类，匹配其中任意一个字符。
        # ^ : 在字符类的开头，表示取反，即匹配不在字符类中的字符。
        # + : 表示前面的字符类可以重复出现一次或多次。
        # [^。！？\.!\?\n]+ : 匹配一个或多个非句子结束符的文本。（英文标点.和?在正则中有特殊含义，所以加了 \ 转义）
        # (?: ... ) : 非捕获组，表示这个括号内的内容是一个整体，但不需要单独捕获它作为一个分组，节省内存。
        # $ : 匹配字符串的结尾，确保最后一个句子也能被正确切分出来。
        # (?:[。！？\.!\?\n]+|$) : 匹配一个或多个句子结束符，或者字符串的结尾。
        pattern = r'[^。！？\.!\?\n]+(?:[。！？\.!\?\n]+|$)'
        sentences = re.findall(pattern, text)

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