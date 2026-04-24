from __future__ import annotations

from _bootstrap import ROOT, assert_exit_ok, reset_dir, tool_names, require_llm_env
from miniagent import MINIAgent

require_llm_env()
workspace = reset_dir(ROOT / "workspace" / "context_demo")
for i in range(1, 6):
    (workspace / f"module_{i}.py").write_text(f"VALUE = {i}\n", encoding="utf-8")
agent = MINIAgent.create_default(ROOT / "config" / "agent.yaml")
result = agent.run(
    "真实上下文管理验证测试：请分步骤检查 context_demo 下的 5 个模块，必须调用 update_working_checkpoint 记录关键发现，"
    "然后写 context_demo/summary.txt，汇总每个 module_i.py 的 VALUE，最后说明你如何保持上下文。"
)
print(result.final_text)
print({"turns": result.turns, "exit": result.exit_reason, "session": result.session_id, "tools": tool_names(result)})
assert_exit_ok(result)
used = tool_names(result)
assert "update_working_checkpoint" in used, f"checkpoint tool not used: {used}"
summary = workspace / "summary.txt"
assert summary.exists(), "summary.txt was not created"
text = summary.read_text(encoding="utf-8")
for i in range(1, 6):
    assert f"module_{i}" in text and str(i) in text, text
