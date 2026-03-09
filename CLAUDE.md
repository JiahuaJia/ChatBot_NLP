# CLAUDE.md — 电影 RAG 聊天机器人

## 项目概述
基于 Wikipedia 电影数据的 RAG 聊天机器人。
- UI: Streamlit（部署至 Streamlit Cloud）
- 向量库: Chroma (persist)，预构建索引存于 `chroma_db/`
- LLM/Embedding: OpenAI（GPT-4o-mini + text-embedding-3-small）
- 高级功能: Hybrid Search（BM25 语义 + RRF 融合）

---

## 禁止事项
- 不得切换向量库、LLM 提供商或 UI 框架（除非明确要求）
- 不得添加 FastAPI 或独立后端服务
- 不得提交 `.env` 或真实 API Key
- 不得在 `chroma_db/` 之外重复存储向量索引
- 不得盲目重试：先读日志、找根因、再修复

---

## 强制约定
- 分块 ID: `sha256(doc_id + str(chunk_index))[:16]`，保证幂等
- 向量集合名: `movies_rag_v1`（变更时须同步更新 `src/common/config.py`）
- 记忆上限: 最多 10 轮，超出时丢弃最旧一轮
- 每次回答 **MUST** 包含引用（电影标题 + 相关片段），无法引用则说明原因

---

## 模块边界（DRY 原则）
| 目录 | 职责 |
|------|------|
| `src/ingest/` | 数据抓取、分块、嵌入 |
| `src/retrieval/` | BM25、语义检索、RRF 融合 |
| `src/llm/` | 提示构建、OpenAI 客户端（streaming） |
| `src/memory/` | 滑动窗口对话记忆 |
| `src/common/` | 配置常量（唯一真相源） |
| `scripts/` | 一次性脚本：抓取数据、构建索引、冒烟测试 |
| `tests/` | 单元测试 |

公共工具放 `src/common/`，禁止跨模块复制逻辑。

---

## 验证命令（声称完成前必须全部通过）
```bash
make lint        # ruff check .
make test        # pytest -q tests/
make smoke       # python scripts/smoke.py
make build-index # python scripts/fetch_movies.py && python scripts/build_index.py
```

## 完成标准
1. `make lint` 无报错
2. `make test` 全绿
3. `make smoke` 输出包含 citations 且 top-K=5

---

## 版本控制
- Chroma 集合名：`movies_rag_v1`（Breaking change 时升级为 `v2`）
- Embedding 模型：`text-embedding-3-small`（变更时需重建全部索引）
- 数据文件：`data/movies/*.json`（SHA256 hash 存入 doc_id）

---

## Secrets & 安全
- API Key 仅通过 `.streamlit/secrets.toml`（本地）或 Streamlit Cloud Secrets（生产）注入
- 日志不得输出 API Key、用户输入原文（仅记录元数据）
