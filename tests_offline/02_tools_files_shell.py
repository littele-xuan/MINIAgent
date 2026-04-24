from __future__ import annotations

from _bootstrap import ROOT
from miniagent.memory import FileMemoryStore
from miniagent.runtime.workspace import Workspace
from miniagent.tools import create_default_tool_registry
from miniagent.tools.base import ToolContext

workspace = Workspace.create(ROOT / "workspace")
memory = FileMemoryStore.create(ROOT / "memory")
ctx = ToolContext(workspace=workspace, memory=memory, session_id="offline-tools", metadata={})
tools = create_default_tool_registry()

res = tools.dispatch("file_write", {"path": "offline_tests/demo.py", "content": "def add(a, b):\n    return a - b\n"}, ctx)
assert res.ok, res.content
res = tools.dispatch("file_read", {"path": "offline_tests/demo.py"}, ctx)
assert res.ok and "return a - b" in res.content
res = tools.dispatch("file_patch", {"path": "offline_tests/demo.py", "old_text": "return a - b", "new_text": "return a + b"}, ctx)
assert res.ok, res.content
res = tools.dispatch("grep_text", {"pattern": "return a \\+ b", "root": "offline_tests", "glob": "*.py"}, ctx)
assert res.ok and "demo.py" in res.content, res.content
res = tools.dispatch("python_run", {"code": "from offline_tests.demo import add\nprint(add(2, 3))", "timeout_seconds": 10}, ctx)
assert res.ok and "5" in res.content, res.content
print("tools/files/shell ok")
