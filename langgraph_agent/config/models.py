from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(slots=True)
class MCPServerConfig:
    name: str = 'default'
    transport: Literal['stdio', 'http'] = 'stdio'
    command: str = 'python'
    args: list[str] = field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    stateful_session: bool = False


@dataclass(slots=True)
class MemoryConfig:
    enabled: bool = True
    user_id: str = 'default-user'
    namespace: str = 'default'
    long_term_namespace: str = 'memories'
    store_backend: Literal['langgraph', 'legacy'] = 'langgraph'
    store_type: Literal['memory', 'postgres'] = 'memory'
    store_conn_string: str | None = None
    checkpointer_type: Literal['memory', 'sqlite'] = 'sqlite'
    sqlite_path: str | None = None
    extract_on_hot_path: bool = True
    retrieval_limit: int = 8
    record_assistant_turns: bool = True
    extractor: Literal['heuristic', 'llm', 'hybrid'] = 'hybrid'
    summarize_threads: bool = True
    event_log_limit: int = 20


@dataclass(slots=True)
class SkillConfig:
    enabled: bool = True
    skills_root: str | None = None
    auto_load: bool = True
    auto_select: bool = True
    policy: Literal['advisory', 'restrictive'] = 'advisory'
    implementation: Literal['filesystem', 'legacy'] = 'filesystem'


@dataclass(slots=True)
class A2AConfig:
    enabled: bool = False


@dataclass(slots=True)
class ContextConfig:
    history_window: int = 8
    max_tool_catalog_chars: int = 9000
    max_memory_chars: int = 6000
    max_observation_chars: int = 4000
    include_memory: bool = True
    include_history: bool = True
    include_visible_tools: bool = True


@dataclass(slots=True)
class ObservabilityConfig:
    provider: Literal['langfuse', 'none'] = 'langfuse'
    enabled: bool = True
    use_env_credentials: bool = True
    public_key: str | None = field(default_factory=lambda: os.getenv('LANGFUSE_PUBLIC_KEY'))
    secret_key: str | None = field(default_factory=lambda: os.getenv('LANGFUSE_SECRET_KEY'))
    base_url: str | None = field(default_factory=lambda: os.getenv('LANGFUSE_BASE_URL'))
    trace_name_prefix: str = 'langgraph-agent'


@dataclass(slots=True)
class ToolRuntimeConfig:
    call_timeout_seconds: float = 45.0
    max_parallel_calls: int = 4
    require_approval_for_risks: tuple[str, ...] = ('destructive',)


@dataclass(slots=True)
class LangGraphAgentConfig:
    name: str = 'langgraph-agent'
    description: str = 'LangGraph-based MCP/skill/memory agent.'
    role: str = 'general-purpose'
    verbose: bool = False
    model: str = field(default_factory=lambda: os.getenv('MCP_MODEL') or os.getenv('OPENAI_MODEL') or '')
    api_key: str = field(default_factory=lambda: os.getenv('MCP_API_KEY') or os.getenv('OPENAI_API_KEY') or '')
    api_base: str = field(default_factory=lambda: os.getenv('MCP_API_BASE') or os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1')
    temperature: float = 0.0
    max_steps: int = 6
    request_timeout_seconds: float = 90.0
    max_retries: int = 2
    provider: Literal['openai-compatible'] = 'openai-compatible'
    output_modes: tuple[str, ...] = ('text/plain', 'application/json')
    working_dir: str = field(default_factory=lambda: str(Path.cwd()))
    mcp_servers: list[MCPServerConfig] = field(default_factory=list)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    skills: SkillConfig = field(default_factory=SkillConfig)
    a2a: A2AConfig = field(default_factory=A2AConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    tools: ToolRuntimeConfig = field(default_factory=ToolRuntimeConfig)
