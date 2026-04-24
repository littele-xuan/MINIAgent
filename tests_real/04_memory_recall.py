from __future__ import annotations

from _bootstrap import ROOT, assert_exit_ok, tool_names, require_llm_env
from miniagent import MINIAgent

require_llm_env()
token = "miniagent-real-memory-src-layout-20260424"
agent1 = MINIAgent.create_default(ROOT / "config" / "agent.yaml")
result1 = agent1.run(
    f"真实记忆工具验证测试：请调用 memory_commit_update，把长期偏好写入 facts 记忆："
    f"{token}: MINIAgent 项目默认使用 src layout。证据：用户明确要求写入长期偏好。"
)
print("FIRST RUN:")
print(result1.final_text)
print({"tools": tool_names(result1), "session": result1.session_id})
assert_exit_ok(result1)
assert "memory_commit_update" in tool_names(result1) or "memory_propose_update" in tool_names(result1), tool_names(result1)
facts = (ROOT / "memory" / "l2_facts.md").read_text(encoding="utf-8")
assert token in facts, "memory token was not committed to l2_facts.md"

agent2 = MINIAgent.create_default(ROOT / "config" / "agent.yaml")
result2 = agent2.run(f"请回忆 {token} 对应的项目默认代码布局偏好是什么，并说明它来自长期记忆。")
print("SECOND RUN:")
print(result2.final_text)
print({"tools": tool_names(result2), "session": result2.session_id})
assert_exit_ok(result2)
assert "src" in result2.final_text.lower(), result2.final_text
