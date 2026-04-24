from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from _bootstrap import ROOT
from miniagent.context.manager import ContextManager
from miniagent.core.loop import AgentLoop
from miniagent.core.outcome import ToolCall
from miniagent.core.state import AgentState
from miniagent.llm.types import LLMResponse
from miniagent.memory import FileMemoryStore
from miniagent.runtime.workspace import Workspace
from miniagent.tools import create_default_tool_registry


@dataclass
class FakeLLM:
    calls: int = 0

    def create_response(self, *, instructions: str, input_items: list[dict[str, Any]], tools: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            call = ToolCall(id="fc_1", call_id="call_1", name="file_write", arguments={"path": "offline_loop/result.txt", "content": "loop ok\n", "mode": "overwrite"}, raw_arguments='{"path":"offline_loop/result.txt","content":"loop ok\\n","mode":"overwrite"}')
            return LLMResponse(text="", output_items=[{"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "file_write", "arguments": call.raw_arguments}], tool_calls=[call], usage={"fake_calls": self.calls})
        return LLMResponse(text="Fake loop completed after writing the file.", output_items=[{"type": "message", "content": [{"type": "output_text", "text": "Fake loop completed after writing the file."}]}], usage={"fake_calls": self.calls})


workspace = Workspace.create(ROOT / "workspace")
memory = FileMemoryStore.create(ROOT / "memory")
loop = AgentLoop(llm=FakeLLM(), tools=create_default_tool_registry(), context=ContextManager(), memory=memory, workspace=workspace, system_prompt="You are a fake test agent.", log_dir=str(ROOT / "workspace" / "logs"), max_turns=4)
result = loop.run(AgentState(user_input="write a result file"))
assert result.exit_reason == "final_answer", result.exit_reason
assert (ROOT / "workspace" / "offline_loop" / "result.txt").read_text(encoding="utf-8") == "loop ok\n"
assert result.tool_events and result.tool_events[0]["name"] == "file_write"
print("agent loop fake llm ok")
