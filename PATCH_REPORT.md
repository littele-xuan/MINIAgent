# MINIAgent MCP / Memory / Langfuse 修复报告（2026-04-22）

## 1. 这次实际修掉了什么

### 1.1 修掉了 planner 输出契约错误
旧版本 planner 允许模型输出：
- `response`
- `name`
- `tool`
- `arguments`
- `data`

这会导致：
- Pydantic 校验时 `mode="final"` 但没有 `final`，直接报错
- MCP 调用字段在 `name/tool/tool_name` 之间漂移
- 动态 dict / Any 字段进入 strict structured outputs 时，schema 本身就不稳定

现在统一为：
- `thought`
- `mode`
- `final`
- `mcp_calls`
- `tool_name`
- `arguments_json`
- `output_mode`
- `text`
- `data_json`

并保留旧字段输入兼容。

### 1.2 把 OpenAI 结构化输出调用改成 Responses API 风格
旧版本主要还是按 Chat Completions 的 `response_format` 在跑。
现在默认改成：
- 优先走 `client.responses.create(...)`
- 结构化输出走 `text.format = { type: "json_schema", name, schema, strict: true }`
- 只有在显式设置 `OPENAI_FORCE_CHAT_COMPLETIONS=1` 时才退回老接口

这就是针对你这次真实报错里 `text.format.name` 缺失而做的正面修复。

### 1.3 默认优先走 Langfuse
现在配置默认：
- 优先尝试 `langfuse.openai.AsyncOpenAI`
- 如果 Langfuse wrapper 不可用，再自动回退到原生 `openai.AsyncOpenAI`
- 可通过 `LANGFUSE_DISABLED=1` 显式关闭

也就是说：
- **默认偏向 Langfuse**
- **但不会因为本地没装 Langfuse 或配置不完整就直接炸死**

### 1.4 Memory 仍然保持为主循环能力，不伪装成工具
这版保留你要求的边界：
- 工具调用是 MCP
- memory 是 runtime-owned
- 记忆提取 / 检索 / 总结 / 注入上下文，不依赖“把记忆伪装成 MCP 工具”

同时继续允许读取旧 memory 形态：
- `name + memories + fact`
- `facts`
- 旧的 `add/remove` 操作名

### 1.5 提示词改成可执行契约，而不是抽象描述
已经重写：
- `prompts/agent/context_operating_contract.md`
- `prompts/agent/tool_usage_rules.md`
- `prompts/agent/planner_output_contract.md`

重点变化：
- 明确 canonical JSON envelope
- 明确 `mode=final` / `mode=mcp` 的互斥规则
- 明确禁止 legacy keys
- 明确 `arguments_json` 必须是 JSON object string，并由 runtime 在执行前解析和校验

## 2. 为什么这次要把 arguments/data 改成 *_json 字符串

这不是“倒退”，而是为了把 **planner 严格结构化输出** 和 **动态 MCP 工具 schema** 分开。

根本矛盾在于：
- planner 的输出 schema 必须对 OpenAI strict structured outputs 足够稳定
- 但 MCP 工具参数是动态的，不可能在 planner 的固定 Pydantic schema 里提前枚举完所有字段

所以现在采用两层做法：

### planner 层
输出固定 envelope：
- `tool_name`
- `arguments_json`

### runtime 层
收到后再：
1. `json.loads(arguments_json)`
2. 按 live MCP tool catalog 中该工具的 `input_schema` 去校验
3. 真正执行

这个边界更稳，也更便于后续替换工具治理方式。

`final.data_json` 同理：
- planner 保持 strict schema
- runtime 再把 `data_json` 解析成真实 JSON payload

## 3. 这次新增的关键模块/逻辑

### 3.1 `llm_runtime/schema_utils.py`
新增 strict schema sanitizer：
- inline `$ref`
- 删除 `title/default/examples` 等噪声字段
- 强制 object `additionalProperties=false`
- 自动补全 `required`
- 对开放 dict / Any 自动降级成 JSON string transport field

### 3.2 `llm_runtime/openai_client.py`
重做：
- 默认 Responses API
- 默认优先 Langfuse wrapper
- `output_text` 提取兼容
- strict schema transport
- repair fallback 保留

### 3.3 `agent_core/planners.py`
重做：
- `FinalAnswer` -> `text + data_json`
- `MCPToolCall` -> `tool_name + arguments_json`
- legacy `response / tool / name / parameters / data` 输入兼容
- top-level `extra='forbid'`

### 3.4 `llm_runtime/mcp_contract.py`
同步成同一套 envelope，避免 planner 和共享 MCP contract 再次分叉。

### 3.5 memory 输出模型收紧
- fact extraction
- retrieval selection
- answer envelope
- summary envelope

都改成 strict 输出模型，同时保留 legacy 输入兼容。

## 4. 本地验证结果

已执行：
- `python -m py_compile $(find . -name '*.py')`
- `pytest -q`

结果：
- **11 passed**

## 5. 真实 API 如何测

### 5.1 最小真实链路测试
```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-5.4
python real_api_contract_smoke_test.py
```

### 5.2 带 Langfuse
```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-5.4
export LANGFUSE_PUBLIC_KEY=...
export LANGFUSE_SECRET_KEY=...
export LANGFUSE_BASE_URL=https://cloud.langfuse.com
python real_api_contract_smoke_test.py
```

### 5.3 你的 memory 集成测试
```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-5.4
python agent_context_memory_test.py --reset
```

### 5.4 如果你想强制退回 Chat Completions
```bash
export OPENAI_FORCE_CHAT_COMPLETIONS=1
```

### 5.5 如果你想关闭 Langfuse
```bash
export LANGFUSE_DISABLED=1
```

## 6. 还没有替你做的事

这次我没有直接替你跑远端真实 API，因为当前环境里没有你的有效：
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `LANGFUSE_*`

所以我能确认的是：
- 本地结构、单测、schema、兼容层已经收拢
- 真实 API 的请求形状已经改到 Responses API / text.format 这一条正确轨道上

但最终是否 100% 跑过，还需要你在自己的密钥环境里再跑一次上面的脚本。

## 7. 这版更适合后续演进的地方

### 已经变得可替换
- LLM transport：OpenAI / Langfuse wrapper / 未来其他 provider adapter
- Planner JSON contract
- MCP tool registry / tool governance
- Context 拼接策略
- Memory 提取 / 检索 / 总结策略

### 建议你下一步继续做
1. 把 planner contract 和 memory contract 再抽成一个独立 `contracts/` 包
2. 给真实 API 跑一次 golden trace regression
3. 给 live MCP 工具参数校验再补一层错误分级
4. 把 Langfuse trace 中的 prompt/version/session 再标准化
