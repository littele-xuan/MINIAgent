from __future__ import annotations

from _bootstrap import ROOT
from miniagent import MINIAgent
from miniagent.runtime.config import AgentConfig
from miniagent.runtime.diagnostics import collect_diagnostics
from miniagent.tools import create_default_tool_registry

config = AgentConfig.load(ROOT / "config" / "agent.yaml")
assert config.name == "MINIAgent"
assert config.llm.provider in {"openai-compatible", "openai"}
registry = create_default_tool_registry()
expected = {"list_dir", "file_read", "file_write", "file_patch", "grep_text", "python_run", "memory_recall"}
assert expected.issubset(set(registry.names())), registry.names()
assert MINIAgent.__name__ == "MINIAgent"
print("import/config ok")
print(collect_diagnostics(ROOT / "config" / "agent.yaml"))
