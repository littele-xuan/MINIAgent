from __future__ import annotations

__all__ = ["GenericAgentCore", "MINIAgent", "AgentResult", "StepOutcome", "ToolCall", "ToolResult"]


def __getattr__(name: str):
    if name in {"GenericAgentCore", "MINIAgent"}:
        from .agent import GenericAgentCore, MINIAgent
        return {"GenericAgentCore": GenericAgentCore, "MINIAgent": MINIAgent}[name]
    if name in {"AgentResult", "StepOutcome", "ToolCall", "ToolResult"}:
        from .outcome import AgentResult, StepOutcome, ToolCall, ToolResult
        return {"AgentResult": AgentResult, "StepOutcome": StepOutcome, "ToolCall": ToolCall, "ToolResult": ToolResult}[name]
    raise AttributeError(name)
