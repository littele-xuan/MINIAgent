# Migration Report

## 已完成

1. 新建 `src/miniagent` 模块化核心平台。
2. 将 OpenAI API 作为唯一新核心 LLM 后端。
3. 默认接入 Langfuse OpenAI drop-in tracing。
4. 新建 ToolRegistry 与工具模块。
5. 新建 ContextManager、turn summary、working checkpoint、compactor。
6. 新建文件记忆系统，包含 policy、recall、proposal、commit。
7. 新建真实 API 直接运行测试脚本。
8. 保留 `v0/` 原始代码。
9. 将原 LangGraph 实验实现整理到 `v1/`。

## 不再作为新核心使用

- 根目录旧 `agent.py` 已替换为 v2 CLI。
- 原 LangGraph 相关内容已归档到 `v1/`。
- 新核心不依赖 LangGraph、LangChain、MCP、A2A、多模型兼容层。

## 后续扩展点

- 向量记忆检索。
- 更严格 shell sandbox。
- OpenAI streaming events。
- Langfuse low-level spans 追踪每个工具调用。
- 使用模型摘要压缩 tool result。
