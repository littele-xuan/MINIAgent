from __future__ import annotations

import subprocess
import sys

from _bootstrap import ROOT, assert_exit_ok, reset_dir, tool_names, require_llm_env
from miniagent import MINIAgent

require_llm_env()
project = reset_dir(ROOT / "workspace" / "sample_project")
(project / "mathlib.py").write_text("def multiply(a, b):\n    return a + b\n", encoding="utf-8")
(project / "test_mathlib.py").write_text("from mathlib import multiply\nassert multiply(3, 4) == 12\nprint('tests passed')\n", encoding="utf-8")
agent = MINIAgent.create_default(ROOT / "config" / "agent.yaml")
result = agent.run(
    "真实端到端 agent 验证测试：请检查 sample_project，运行测试发现失败原因，修复代码，"
    "再次运行测试验证，最后总结修改文件和验证结果。必须使用文件工具和 python_run 或 shell_run。"
)
print(result.final_text)
print({"turns": result.turns, "exit": result.exit_reason, "session": result.session_id, "tools": tool_names(result)})
assert_exit_ok(result)
content = (project / "mathlib.py").read_text(encoding="utf-8")
assert "return a * b" in content or "return b * a" in content, content
completed = subprocess.run([sys.executable, "-S", "test_mathlib.py"], cwd=str(project), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
assert completed.returncode == 0, completed.stdout + completed.stderr
used = tool_names(result)
assert any(name in used for name in ("file_read", "read_many_files")), f"no file read tool used: {used}"
assert any(name in used for name in ("file_patch", "file_write")), f"no file edit tool used: {used}"
assert any(name in used for name in ("python_run", "shell_run")), f"no execution tool used: {used}"
