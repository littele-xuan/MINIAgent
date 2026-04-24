# MCP Tool Manager

[English](./README.md)

这是一个以英文 README 为默认入口、可切换到中文说明的 MCP 工具全生命周期管理参考实现。

这个项目不是简单地“暴露几个 MCP 工具”，而是把工具注册、治理、协议适配，以及一个 API-first 的 MCP/A2A Agent 放在同一个实现里。该 Agent 以严格 JSON 规划输出驱动 live MCP catalog，并可通过标准 A2A 协议与其他 Agent 协作，形成一个可演进的工具控制平面。

## 这个仓库实现了什么

- 用 `ToolRegistry` 作为工具元数据、别名、启停状态、版本历史和调用分发的唯一事实来源。
- 定义三层工具模型：
  - `INTERNAL_SYSTEM`：受保护的治理工具。
  - `INTERNAL_UTILITY`：受保护的内置工具。
  - `EXTERNAL`：运行时可管理的外部工具。
- 将治理能力本身也封装成工具，使 MCP 客户端或 LLM Agent 可以在会话中完成新增、更新、禁用、废弃、别名、合并、查询和删除外部工具。
- 使用 MCP Server 读取实时注册表，而不是在服务端硬编码工具清单。
- 使用 Pydantic 结构化输出驱动 Agent，避免脆弱的字符串解析。
- 提供分层测试：注册表逻辑、MCP 集成、可选的 LLM 演示。

## 设计理念

### 1. 以注册表为中心，而不是以 Server 为中心

MCP Server 在这里是协议适配层，不是系统核心。真正拥有状态和行为的是注册表，MCP 只负责把这些能力暴露出去。

### 2. 核心受保护，边界可演化

治理工具和基础内置工具不允许通过治理 API 被修改；只有 `EXTERNAL` 工具可以被全生命周期管理。这样可以保证控制平面的稳定，同时保留运行时扩展能力。

### 3. 生命周期元数据是模型的一部分

启用/禁用、废弃、别名、版本历史、标签、审计钩子都被建模为显式数据，而不是散落在各处的约定。这使系统更容易测试、导出和长期维护。

### 4. 给 Agent 自主性，但加上边界

Agent 可以通过治理工具管理外部工具，但无法越权修改受保护层。这是“允许自管理”与“保持系统稳定”之间的平衡。

### 5. 动态发现优先于硬编码

客户端通过实时 `list_tools()` 发现工具，而不是在 Agent 里重复维护一份工具定义。新增或更新工具时，不需要同步修改 Agent 代码。

## 架构概览

```text
用户 / LLM
    |
    v
结构化输出 Agent（ReAct + Pydantic）
    |
    v
MCP Client Session
    |
    v
MCP Server 适配层
    |
    v
ToolRegistry  <---->  GovernanceManager / registry_ops
    |
    +-- INTERNAL_SYSTEM 工具
    +-- INTERNAL_UTILITY 工具
    +-- EXTERNAL 工具
```

## 生命周期流程

1. 启动时把内置工具、治理工具和示例外部工具注册到注册表。
2. MCP Server 从当前注册表导出可用工具。
3. MCP 客户端或 Agent 通过注册表完成工具调用分发。
4. 治理工具只能修改 `EXTERNAL` 类别的工具。
5. 当工具治理发生变化时，Agent 刷新自己的工具视图。
6. 对破坏性更新保留版本历史，并暴露统计、搜索、导出等能力。

## 目录说明

| 路径 | 作用 |
|------|------|
| `agent.py` | API-first MCP/A2A Agent，使用严格 JSON 规划输出 |
| `agent_test.py` | 分层测试与可选 LLM 演示入口 |
| `mcp_lib/registry/` | 注册表模型与调用分发逻辑 |
| `mcp_lib/governance/` | 高层生命周期治理封装 |
| `mcp_lib/server/` | MCP 协议适配层 |
| `mcp_lib/tools/internal/` | 内置工具和治理工具 |
| `mcp_lib/tools/external/` | 启动时注册的示例外部工具 |
| `v0/` | 保留的早期原型，用于对比 |

## 内置能力

- 工具类能力：计算器、Python 执行、网页搜索、天气、文件操作。
- 治理类能力：列出、查看、添加、更新、启用、禁用、废弃、别名、合并、搜索、版本历史、注册表统计。
- 示例外部工具：随机笑话、UUID、时间戳、Base64、哈希。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行测试

```bash
python agent_test.py
```

默认会运行 Tier 1 和 Tier 2。

### 3. 运行可选的 LLM 演示

先设置环境变量：

```bash
export MCP_API_BASE="https://api.openai.com/v1"
export MCP_API_KEY="your_api_key"
export MCP_MODEL="gpt-4o-mini"
python agent_test.py --llm
```

## 安全说明

- 主测试入口中的明文 API 凭证已经移除。
- 仓库现在要求通过环境变量注入密钥。
- `.gitignore` 已排除常见本地环境与潜在敏感文件。
- 如果通过 `tool_add` 或 `tool_update` 注入了动态 Python 代码，在发布前应做安全审查。

## 这个实现的价值

很多 MCP 示例只展示“工具能调起来”。这个仓库更进一步，把工具当作有状态、可治理、可演化的资产来管理。这才是 MCP 从 demo 走向真实工具平台时更有价值的边界。

## License

本项目使用 [MIT License](./LICENSE)。

---

## LangGraph 科研级 Agent 基线入口

本版本新增 `langgraph_agent/` 作为主实现，真实 API 场景测试入口为：

```bash
python real_scenarios/run_research_agent_scenarios.py --reset --scenario all
```

详细说明见 `LANGGRAPH_AGENT_RESEARCH_BASELINE_PATCH.zh-CN.md`。
