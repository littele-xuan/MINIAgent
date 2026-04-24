class MiniAgentError(Exception):
    """Base error for MINIAgent."""


class ToolNotFoundError(MiniAgentError):
    """Raised when the model calls an unknown tool."""


class WorkspaceSecurityError(MiniAgentError):
    """Raised when a path escapes the configured workspace."""


class LLMResponseError(MiniAgentError):
    """Raised when an LLM response cannot be parsed."""
