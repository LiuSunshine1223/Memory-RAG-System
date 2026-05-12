# Memory-RAG-System 自研多轮记忆知识库问答系统

## 项目简介

本项目是一个基于本地化大模型 `Ollama + Llama3:8B`、`BGE-small-zh-v1.5`、`BGE-reranker-base` 与 `FAISS` 向量库构建的本地化 Memory RAG / Agent Pipeline 原型系统。

系统面向多轮问答中的长期记忆管理问题，设计了“冷-温-热”数据分层结构：冷数据保存原始知识文档，温数据维护 `Base Knowledge` 与 `User Memory` 两个 FAISS 向量库，热数据保留当前会话的短期上下文窗口。通过这种方式，系统将静态知识、长期用户记忆和当前对话历史进行隔离管理，降低基础知识与用户记忆之间的交叉污染。

在在线问答阶段，系统在 `RAGPipeline` 前置轻量级 `QueryPolicy` 意图分流层，将用户输入划分为 `direct_chat`、`memory_write`、`memory_recall` 和 `normal_rag` 四类，并根据不同意图选择不同处理路径。系统不仅能够基于静态基础知识库进行问答，还能在多轮对话中选择性沉淀用户目标、偏好和计划，并支持程序重启后的长期记忆召回。

---

## 系统架构图 Architecture

### 1. 离线冷启动与知识建库

<div align="center">
  <img src="./assets/offline_cold_start.jpg" width="800">
  <p><em>图 1：离线冷启动：原始文档切分、向量化与基础知识库构建</em></p>
</div>

离线阶段主要负责冷数据处理，即对原始知识文档进行句子边界切分、Embedding 向量化，并写入 `Base Knowledge FAISS`，为后续在线 RAG 检索提供稳定的静态知识来源。

### 2. 在线问答、意图路由与记忆沉淀闭环

<div align="center">
  <img src="./assets/online_qa.jpg" width="800">
  <p><em>图 2：在线阶段：QueryPolicy 意图路由、双库召回、Prompt 组装与长期记忆写入闭环</em></p>
</div>

图 2 展示了在线问答阶段的完整闭环。用户输入首先经过 `QueryPolicy` 进行意图分流，划分为 `direct_chat`、`memory_write`、`memory_recall` 和 `normal_rag` 四类；其中 `normal_rag` 会进入 `Base FAISS` 与 `Memory FAISS` 双库召回，并通过 `bge-reranker-base` 进行二阶段精排。系统随后将基础知识、长期用户记忆、短期对话窗口与当前 Query 组装为最终 Prompt，交由 `Llama3:8B` 生成回答。回答生成后，`MemoryManager` 会根据记忆价值和新奇度判断，决定是否将当前 `[query, answer]` 写入 `User Memory`，从而形成长期记忆沉淀闭环。

---

## 核心架构亮点

- **冷-温-热数据分层架构**：将原始文档、双 FAISS 向量库和短期对话窗口分别作为冷数据、温数据和热数据管理，实现静态知识、长期用户记忆与当前会话历史的隔离。

- **轻量级 QueryPolicy 意图分流**：在 `RAGPipeline` 前置规则式意图判断，将用户输入划分为 `direct_chat`、`memory_write`、`memory_recall` 与 `normal_rag` 四类。身份闲聊类问题直接旁路回答，用户目标 / 计划 / 偏好类输入进入长期记忆写入流程，历史记忆回忆类问题走专门的 `User Memory` 检索路径，普通知识问答则进入完整 RAG 检索链路。

- **双库向量检索架构**：物理隔离 `Base Knowledge`（静态基础知识库）与 `User Memory`（动态长期记忆库），避免将基础知识、日常闲聊和用户长期记忆混入同一个向量空间，降低记忆污染风险。

- **动态记忆管理机制**：通过 `MemoryManager` 对用户目标、计划、偏好等信息进行价值过滤、新奇度判断和长期记忆写入控制，避免将普通知识问答或重复闲聊内容全部写入用户记忆库。

- **Bi-Encoder + Cross-Encoder 两阶段检索**：使用 `bge-small-zh-v1.5` 进行向量召回，再使用 `bge-reranker-base` 进行二阶段精排，提升基础知识库与长期记忆库的检索相关性。

- **历史记忆回忆检索策略**：针对“我之前说过什么 / 你还记得我吗”等泛化记忆查询，单独采用 `memory_recall` 检索路径，优先检索 `User Memory`，跳过 `Base Knowledge` 干扰，并避免将回忆类 Query 再次写入长期记忆库，提升重启后长期记忆召回稳定性。

- **U 型 Prompt 组装策略**：将基础知识库内容放在 Prompt 前部，将近期连续对话放在 Prompt 后部，结合大模型首尾注意力偏好，缓解部分 Lost in the Middle 现象。

- **显存监控与可视化**：系统内置后台显存监控线程，在运行结束或异常退出时生成显存变化折线图，便于观察本地模型运行时的资源占用情况。

- **配置与代码解耦**：采用 `.env` 环境变量注入机制，将模型路径、向量库路径、召回阈值等参数从业务代码中解耦，方便本地实验与后续迁移部署。

---

## 技术栈

- **LLM**：`Llama3:8b` via Ollama
- **Embedding Model**：`bge-small-zh-v1.5`
- **Reranker Model**：`bge-reranker-base`
- **Vector Store**：FAISS
- **Frameworks**：PyTorch, Transformers, Matplotlib
- **Runtime**：Python 3.9+

---

## 核心目录结构

```text
MemoryRAG/
├── assets/
│   ├── offline_cold_start.jpg       # 离线知识建库流程图
│   ├── online_qa.jpg                # 在线问答、意图路由与记忆沉淀流程图
│   ├── test_log1.png                # 测试日志截图 1
│   └── test_log2.png                # 测试日志截图 2
├── data/                            # 基础知识库文本存放处
├── docs/
│   ├── test_cases.md                # 轻量化功能测试用例
│   └── demo_log.md                  # Demo 运行日志
├── memory/
│   └── memory_manager.py            # 记忆管理层：短期记忆、长期记忆写入、普通 RAG 检索与历史回忆检索策略
├── models/                          # 模型加载层：Embedding、Reranker、LLM 接口
├── pipeline/
│   └── rag_pipeline.py              # 流程调度层：QueryPolicy 意图分流、检索策略选择与 Prompt 构造
├── retrieval/
│   └── vector_store.py              # FAISS 向量库封装
├── utils/
│   └── text_splitter.py             # 句子边界切分器 Sentence Boundary Chunker
├── monitor.py                       # 显存监控与图表生成
├── config.py                        # 全局配置中心
└── main.py                          # 系统启动入口
```

---

## 快速部署

### 1. 环境要求

建议使用 Python 3.9+ 虚拟环境。

```bash
conda create -n memory_rag python=3.9
conda activate memory_rag
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制配置模板并按需修改：

```bash
cp .env.example .env
```

### 4. 准备基础知识库文本

请在 `data/` 目录下放置合法的 `.txt` 文本文件，并在 `.env` 中指定 `DATA_PATH`。

### 5. 启动系统

```bash
python main.py
```

系统启动后会自动检查基础知识库索引。如果基础知识库尚未建立索引，系统会自动完成文本切分、向量化和入库。

---

## 功能测试

项目提供轻量化功能测试记录，覆盖 QueryPolicy 拦截、基础知识库检索、长期记忆写入、历史记忆回忆、重启后记忆持久化等核心链路。

- [测试用例说明](docs/test_cases.md)
- [Demo 运行日志](docs/demo_log.md)

核心测试场景包括：

| 编号 | 测试目标 | 示例输入 | 验证点 |
|---|---|---|---|
| TC-01 | 身份类 Query 拦截 | 你是谁 | 不进入知识库角色扮演，不写入长期记忆 |
| TC-02 | 基础知识库问答 | 孙悟空是谁 | 能从 Base Knowledge 检索相关知识 |
| TC-03 | 普通知识问答过滤 | 孙悟空是谁 | 普通知识问答不写入 User Memory |
| TC-04 | 用户目标写入长期记忆 | 我的目标是找一份 RAG 实习 | 用户目标被写入动态记忆库 |
| TC-05 | 当前会话历史记忆回忆 | 我之前说过什么 | 触发 memory_recall 检索策略 |
| TC-06 | 重启后长期记忆召回 | 重启后问：我之前说过什么 | 验证 Memory FAISS 持久化成功 |

---

## 后续架构演进路线 Future Work

在当前 V1.0 系统基础上，后续可以从数据治理、检索策略、记忆生命周期、底层推理优化和 Agent 化方向继续迭代。

- **数据治理与检索工程精细化**：真实业务场景中的原始文档通常包含噪声、重复段落、格式错乱和无效文本。后续可重构底层 Data ETL 管道，加入更细粒度的文本清洗、重复内容过滤和 Metadata 标注，从源头提升 Chunking 与向量检索质量。

- **Metadata 混合检索与时间衰减机制**：后续可为文本块增加来源、时间、类型等 Metadata 字段，并结合向量检索和元数据过滤。对于动态记忆库，可以引入时间衰减权重，让近期记忆在用户相关问题中拥有更高召回优先级；而基础知识库保持稳定召回，不做时间衰减，从而区分“客观知识”和“用户记忆”的不同生命周期。

- **记忆生命周期管理 Memory Lifecycle**：当前系统主要通过静态距离阈值进行新奇度判断。后续可引入检索频率、最近访问时间、记忆价值评分等因素，对长期未被激活的低价值记忆进行合并、降级或删除，降低长期记忆库膨胀带来的检索噪声。

- **结合长上下文研究做底层推理优化**：当前项目主要从外部记忆角度实现 Memory RAG，即通过 FAISS 向量库承接被挤出上下文窗口的长期信息。后续希望结合 KV Cache 和长文本记忆方向的研究，进一步探索推理阶段的选择性 KV 写入与驱逐策略，形成“内部 KV 工作记忆 + 外部 FAISS 长期记忆”的双层记忆结构。

- **向具备长期记忆管理能力的 Agentic RAG 演进**：当前版本已经实现轻量级 `SimpleQueryPolicy`，可以区分 `direct_chat`、`memory_write`、`memory_recall` 和 `normal_rag` 四类请求，解决所有 Query 都强行进入 RAG 检索链路的问题。但目前的 QueryPolicy 仍然基于关键词规则，泛化能力有限，且长期记忆的召回、写入与更新仍依赖固定流程控制。后续可以将规则路由升级为语义路由，例如使用轻量分类模型、Embedding 相似度路由或 LLM Router 判断用户意图。在此基础上，可以进一步将 `Base FAISS` 检索、`Memory FAISS` 检索、长期记忆写入、记忆更新与去重封装为工具能力，让大模型根据任务需求主动决定是否检索知识库、召回用户记忆或沉淀新记忆，从固定流程的 Pipeline RAG 演进到具备长期记忆管理能力的 Agentic RAG。

---

## 免责声明

本项目及其代码仅供学术研究与工程技术交流使用。请确保用于索引和检索的文本数据具备合法使用权限。开发者不对用户上传、索引及生成的任何内容承担责任。