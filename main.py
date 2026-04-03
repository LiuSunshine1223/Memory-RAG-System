# 在底层加载 BGE 嵌入模型、FAISS 向量库和大模型等核心资源并注入给 RAG 业务流，随后接管所有的终端问答输入循环。
# 这个文件是整个系统的入口，负责启动时的环境准备、组件初始化、主对话循环以及最终的资源清理和显存监控数据可视化。

import transformers
# 屏蔽 transformers 的底层警告，只保留严重的报错
transformers.logging.set_verbosity_error()

import os
import sys

# 1. 强制清理环境变量，确保 WSL2 直连 Windows 上的 Ollama，不走代理
for var in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(var, None)

from models.embedding import EmbeddingModel
from models.reranker import RerankerModel
from models.llm import LLMModel
from retrieval.vector_store import VectorStore
from memory.memory_manager import MemoryManager
from utils.text_splitter import FixedLengthChunker, SemanticChunker
from pipeline.rag_pipeline import RAGPipeline
from config import Config
from monitor import VRAMMonitor


def main():
    print("\n" + "=" * 50)
    print(" Memory RAG 完整本地版 (GPU 加速已激活)")
    print("=" * 50)

    monitor = VRAMMonitor()

    try:
        monitor.start()

        # --- 初始化组件 ---
        # 加载 Embedding 模型 (BGE-small-zh)
        embed_model = EmbeddingModel(Config.EMBED_MODEL_PATH)

        # 加载 Reranker 模型 (BGE-reranker-base)
        reranker_model = RerankerModel(Config.RERANK_MODEL_PATH)

        # 加载 LLM (Ollama / Llama3)
        llm = LLMModel(Config.OLLAMA_URL, Config.LLM_MODEL)

        # 初始化切分器
        chunker = SemanticChunker(Config.CHUNK_SIZE, Config.OVERLAP_SENTENCE)

        # --- 初始知识入库  ---
        base_vs = VectorStore(Config.BASE_DB_PATH, Config.VECTOR_DIM)

        if os.path.exists(Config.DATA_PATH):
            with open(Config.DATA_PATH, 'r', encoding='utf-8') as f:
                raw_text = f.read()
                clean_text = " ".join(raw_text.split())

            if clean_text:
                # 只有当库里还没数据时，才进行初始入库，避免重复
                if base_vs.total_count == 0:
                    print(f" 正在为基础知识库建立索引: {Config.DATA_PATH}...")
                    chunks = chunker.split_text(clean_text)
                    embs = embed_model.get_embeddings(chunks)
                    base_vs.add_texts(chunks, embs)
                    print(f" VectorStore: 基础知识入库成功，共 {len(chunks)} 条知识。")
                else:
                    print(f" VectorStore: 检测到已有索引，跳过基础知识入库。")
            else:
                print(f" 警告: {Config.DATA_PATH} 是空文件，跳过入库。")
        else:
            print(f" 警告: 未找到初始文档 {Config.DATA_PATH}，将开启空白记忆模式。")

        # 动态记忆库（用户对话的读写库）
        memory_vs = VectorStore(Config.MEMORY_DB_PATH, Config.VECTOR_DIM)
        print(f" 动态记忆库: 加载完毕，当前拥有 {memory_vs.total_count} 条历史记忆。")

        # --- 实例化记忆管家和 Pipeline ---
        memory_manager = MemoryManager(
            base_vs = base_vs,
            memory_vs = memory_vs,
            embed_model = embed_model,
            reranker_model = reranker_model,
            chunker = chunker,
            threshold = Config.THRESHOLD,
            base_max_dist = Config.BASE_MAX_DISTANCE,
            memory_max_dist = Config.MEMORY_MAX_DISTANCE,
            base_recall_k = Config.BASE_RECALL_K,
            base_rerank_k = Config.BASE_RERANK_K,
            memory_recall_k = Config.MEMORY_RECALL_K,
            memory_rerank_k = Config.MEMORY_RERANK_K
        )
        rag = RAGPipeline(llm, memory_manager)

        print("\n[系统就绪] 请开始对话。输入 'exit' 或 'quit' 退出程序。")

        # --- 主对话循环 ---
        while True:
            query = input("\n[用户]: ").strip()

            # 退出指令判断
            if not query:
                continue
            if query.lower() in ['exit', 'quit', '退出']:
                print("\n正在保存记忆并退出系统... 再见！")
                break

            # 执行 RAG 流程
            try:
                answer = rag.run(query)
                print(f"[AI]: {answer}")
            except Exception as e:
                print(f"\n[错误]: 推理过程中出现问题: {e}")
                print("[提示]: 请检查 Ollama 是否已启动，或显存是否溢出。")

    except KeyboardInterrupt:
        # 处理 Ctrl+C 强制退出
        print("\n\n检测到中断指令，正在安全关闭...")
    except Exception as e:
        # 处理启动时的严重错误
        print(f"\n[崩溃]: 程序启动失败: {e}")
    finally:
        # 无论你是正常退出、报错崩溃、还是 Ctrl+C 强退，这里都会百分百执行
        if monitor and monitor.is_running:
            print("\n 正在整理压测数据并生成显存折线图...")
            output_dir = "outputs"  # 定义一个专门放输出文件的文件夹
            os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在，全自动创建
            # 放入路径：outputs/vram_test_result.png
            save_path = os.path.join(output_dir, "vram_test_result.png")
            monitor.stop_and_plot(save_path=save_path)
        sys.exit(0)


if __name__ == "__main__":
    main()