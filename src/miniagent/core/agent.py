from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..context.manager import ContextManager
from ..context.compactor import ContextCompactor
from ..context.prompts import load_prompt
from ..llm.openai_client import OpenAIResponsesClient
from ..memory.store import FileMemoryStore
from ..runtime.config import AgentConfig
from ..runtime.workspace import Workspace
from ..tools import create_default_tool_registry
from ..tools.registry import ToolRegistry
from .loop import AgentLoop
from .outcome import AgentResult
from .state import AgentState


@dataclass(slots=True)
class GenericAgentCore:
    """Modular GenericAgent core: LLM + tools + context + memory + workspace."""

    config: AgentConfig
    llm: OpenAIResponsesClient
    tools: ToolRegistry
    context: ContextManager
    memory: FileMemoryStore
    workspace: Workspace
    system_prompt: str
    event_callback: Callable[[dict[str, Any]], None] | None = None

    def run(self, user_input: str, *, metadata: dict[str, Any] | None = None) -> AgentResult:
        state = AgentState(user_input=user_input, metadata=metadata or {})
        loop = AgentLoop(
            llm=self.llm,
            tools=self.tools,
            context=self.context,
            memory=self.memory,
            workspace=self.workspace,
            system_prompt=self.system_prompt,
            log_dir=self.config.log_dir,
            max_turns=self.config.max_turns,
            event_callback=self.event_callback,
        )
        return loop.run(state)


class MINIAgent(GenericAgentCore):
    """Concrete GenericAgentCore instance for this project."""

    @classmethod
    def create_default(
        cls,
        config_path: str | Path = "config/agent.yaml",
        *,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> "MINIAgent":
        config = AgentConfig.load(config_path)
        workspace = Workspace.create(config.workspace_dir)
        memory = FileMemoryStore.create(config.memory_dir)
        llm = OpenAIResponsesClient.create(config)
        tools = create_default_tool_registry()
        context = ContextManager(
            compactor=ContextCompactor(
                max_tool_output_chars=int(config.context.get("max_tool_output_chars", 12000)),
                keep_recent_summaries=int(config.context.get("keep_recent_summaries", 8)),
            )
        )
        system_prompt = load_prompt(config.system_prompt_path)
        system_prompt = system_prompt.rstrip() + (
            "\n\n### Runtime context\n"
            f"- Agent name: {config.name}\n"
            f"- Configured model: {config.llm_model()}\n"
            f"- API provider: {config.llm.provider}\n"
            f"- API mode: {config.llm.api_mode}\n"
            "- Runtime note: identify as MINIAgent unless the user specifically asks about the underlying configured model.\n"
        )
        return cls(
            config=config,
            llm=llm,
            tools=tools,
            context=context,
            memory=memory,
            workspace=workspace,
            system_prompt=system_prompt,
            event_callback=event_callback,
        )
