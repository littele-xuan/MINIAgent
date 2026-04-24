from __future__ import annotations

from _bootstrap import ROOT, assert_exit_ok, reset_dir, tool_names, require_llm_env
from miniagent import MINIAgent

require_llm_env()
workspace = reset_dir(ROOT / "workspace" / "real_tests_patch")
(workspace / "patch_demo.py").write_text("def greet(name):\n    return 'bye ' + name\n", encoding="utf-8")
agent = MINIAgent.create_default(ROOT / "config" / "agent.yaml")
result = agent.run(
    "真实文件工具验证测试：必须先调用 file_read 读取 real_tests_patch/patch_demo.py，"
    "然后优先调用 file_patch 把 greet 修复为返回 'hello ' + name，最后再次读取文件确认。"
)
print(result.final_text)
print({"turns": result.turns, "exit": result.exit_reason, "session": result.session_id, "tools": tool_names(result)})
assert_exit_ok(result)
content = (workspace / "patch_demo.py").read_text(encoding="utf-8")
assert "return 'hello ' + name" in content and "return 'bye ' + name" not in content, content
used = tool_names(result)
assert "file_read" in used, f"file_read not used: {used}"
assert any(name in used for name in ("file_patch", "file_write")), f"no write/patch tool used: {used}"
