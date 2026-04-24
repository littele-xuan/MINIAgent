from .base import BaseTool, ToolContext, ToolSpec
from .registry import ToolRegistry
from .files import FilePatchTool, FileReadTool, FileWriteTool, GrepTextTool, ListDirTool, ReadManyFilesTool, SearchFilesTool
from .shell import PythonRunTool, ShellRunTool
from .context_tools import AskUserTool, UpdateWorkingCheckpointTool
from .memory_tools import MemoryCommitUpdateTool, MemoryProposeUpdateTool, MemoryRecallTool


def create_default_tool_registry() -> ToolRegistry:
    return ToolRegistry().register_many([
        ListDirTool(),
        SearchFilesTool(),
        FileReadTool(),
        ReadManyFilesTool(),
        FileWriteTool(),
        FilePatchTool(),
        GrepTextTool(),
        ShellRunTool(),
        PythonRunTool(),
        UpdateWorkingCheckpointTool(),
        MemoryRecallTool(),
        MemoryProposeUpdateTool(),
        MemoryCommitUpdateTool(),
        AskUserTool(),
    ])


__all__ = [
    "BaseTool",
    "ToolContext",
    "ToolRegistry",
    "ToolSpec",
    "create_default_tool_registry",
]
