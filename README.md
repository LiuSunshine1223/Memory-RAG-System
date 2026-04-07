# Memory RAG System: 带有动态上下文记忆的高可用检索系统

## 项目简介
本项目是一个基于本地化大模型（Ollama + Llama3）与双引擎向量数据库（FAISS）构建的检索增强生成（RAG）系统。

核心创新点在于打破了传统 RAG 系统的“无状态”痛点，设计了独立的 **MemoryManager（动态记忆管理器）**。系统不仅能基于静态基础专业知识库进行精准问答，还能在多轮对话中动态捕获用户的个性化偏好，实现真正的“长期记忆”。

## 系统架构图 (Architecture)

### 1. 离线冷启动与知识建库 (Phase 1)
<div align="center">
  <img src="https://github.com/user-attachments/assets/e479385d-2e38-4cac-bc4b-a1d8a1c6f0eb" width="800">
  <p><em>图 1: 基础知识库 O(n) 向量化与硬写入流</em></p>
</div>

### 2. 在线并发问答与记忆沉淀闭环 (Phase 2)
<div align="center">
  <img src="https://github.com/user-attachments/assets/ec6d7e73-c470-41f8-838d-54a07009f75b" width="800">
  <p><em>图 2: RAG 核心引擎在线双轨调度流与动态记忆更新</em></p>
</div>

## 核心架构亮点
- **双轨向量检索架构**：物理隔离 `Base Knowledge`（高容量基础库）与 `User Memory`（动态记忆库），阻断核心知识被日常闲聊污染。
- **深度检索引擎**：融合双塔向量召回 (Bi-Encoder, `bge-small`) 粗排与交叉编码器 (Cross-Encoder, `bge-reranker`) 精排，极大提升长尾记忆的提取精度。
- **U型 Prompt 组装策略**：将客观知识置顶，短期会话置底，严格遵循系统级隔离，有效缓解大模型的 Attention 丢失（Lost in the Middle）现象。
- **全链路资源监控兜底**：系统内置后台显存监控线程，哪怕发生 OOM 崩溃，系统也会在临死前（`finally`）生成压测显存折线图，极大提升工程可维护性。
- **配置与代码解耦**：采用 `.env` 环境变量注入机制，实现本地算力调度与云端部署环境的安全隔离。

## 技术栈
- **核心模型**：`Llama3:8b` (基于 Ollama)
- **Embedding & Rerank**：`bge-small-zh-v1.5`, `bge-reranker-base`
- **向量数据库**：`FAISS` (支持并发检索)
- **基础框架**：`PyTorch`, `Transformers`, `Matplotlib`

## 核心目录结构

    MemoryRAG/
    ├── data/                  # 基础知识库文本存放处 (默认内置轻量级演示数据)
    ├── memory/
    │   └── memory_manager.py  # 核心：长短期记忆管家，负责查重阈值判定与双库路由
    ├── models/                # 模型加载层 (Embedding, Rerank, LLM 接口)
    ├── pipeline/
    │   └── rag_pipeline.py    # 组装层：调度记忆，生成三段式严密 Prompt
    ├── retrieval/
    │   └── vector_store.py    # FAISS 向量库底层原生封装
    ├── utils/
    │   └── text_splitter.py   # 语义切分器 (Semantic Chunker)
    ├── monitor.py             # 旁路系统：多线程显存监控与图表生成
    ├── config.py              # 全局配置中心 (环境变量解耦)
    └── main.py                # 系统启动入口

## 快速部署
**环境要求：** 建议使用 `Python 3.9+` 虚拟环境。

1. **安装核心依赖：**
   `pip install -r requirements.txt`

2. **配置环境变量：**
   复制配置模板并按需修改：
   `cp .env.example .env`

3. **准备数据：**
   请在 `data/` 目录下放置您的合法 `.txt` 文本文件，并在 `.env` 中指定 `DATA_PATH`。

4. **启动问答监听引擎：**
   `python main.py`
   *启动后，系统将自动进行冷启动知识入库，并在 `outputs/` 目录下生成显存性能图表。*

## 后续架构演进路线 (Future Work)
在当前系统基础上，规划了以下前沿维度的技术迭代路线：
- [ ] **外部 KV Buffer 内存容量优化**：探索通过构建结构化的外部 Key-Value 缓存机制，替代纯文本向量追加，以极低的时间复杂度解决动态记忆的冲突问题，并大幅提升系统长期记忆的容量上限与存取效率。
- [ ] **引入 Query Router (意图识别路由)**：利用轻量级 LLM 或分类模型作为前置路由，动态切分闲聊与专业提问，进一步抑制上下文注意力稀释。
- [ ] **向 Agentic RAG 跃迁**：将双库检索封装为独立 Tools，交由大模型进行自主规划与调度调用。

## 免责声明
本项目及其代码仅供学术研究与工程技术交流使用。严禁使用本系统对受著作权保护的文本进行非法索引、传播或商业化。开发者不对用户上传及生成的任何内容负责。