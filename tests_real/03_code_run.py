from __future__ import annotations

from _bootstrap import ROOT, assert_exit_ok, assert_tool_used, require_llm_env
from miniagent import MINIAgent

require_llm_env()
agent = MINIAgent.create_default(ROOT / "config" / "agent.yaml")
result = agent.run(
    "真实代码执行工具验证测试：必须调用 python_run，运行代码计算 sum(i*i for i in range(10))，"
    "验证输出是 285，然后给出一句简短结论。不要只心算。"
)
print(result.final_text)
print({"turns": result.turns, "exit": result.exit_reason, "session": result.session_id, "tools": [e.get("name") for e in result.tool_events]})
assert_exit_ok(result)
assert_tool_used(result, "python_run")
combined = result.final_text + "\n" + "\n".join(str(e.get("content") or "") for e in result.tool_events)
assert "285" in combined, combined
