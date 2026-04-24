from __future__ import annotations

__all__ = ["GenericAgentCore", "MINIAgent", "AgentResult", "StepOutcome", "ToolResult"]


def __getattr__(name: str):
    if name in {"GenericAgentCore", "MINIAgent"}:
        from .core.agent import GenericAgentCore, MINIAgent
        return {"GenericAgentCore": GenericAgentCore, "MINIAgent": MINIAgent}[name]
    if name in {"AgentResult", "StepOutcome", "ToolResult"}:
        from .core.outcome import AgentResult, StepOutcome, ToolResult
        return {"AgentResult": AgentResult, "StepOutcome": StepOutcome, "ToolResult": ToolResult}[name]
    raise AttributeError(name)
