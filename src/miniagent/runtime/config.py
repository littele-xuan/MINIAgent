from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .env import load_dotenv_if_present

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return None


@dataclass(slots=True)
class LLMConfig:
    """Configuration for an OpenAI-compatible LLM endpoint.

    The runtime is intentionally named OpenAI-compatible because most MCP
    gateways expose an OpenAI SDK compatible /v1 endpoint.  MCP_* variables are
    the default source of truth, with OPENAI_* kept as fallback compatibility.
    """

    provider: str = "openai-compatible"
    model: str = "gpt-5.4"
    model_env: str = "MCP_MODEL"
    api_key_env: str = "MCP_API_KEY"
    base_url: str | None = None
    base_url_env: str = "MCP_API_BASE"
    fallback_model_env: str = "OPENAI_MODEL"
    fallback_api_key_env: str = "OPENAI_API_KEY"
    fallback_base_url_env: str = "OPENAI_BASE_URL"
    api_mode: str = "chat_completions"  # chat_completions | responses | auto
    temperature: float | None = None
    max_output_tokens: int | None = 4096
    request_timeout_seconds: float = 120.0
    max_retries: int = 2
    store: bool = False

    def resolved_model(self) -> str:
        return _first_env(self.model_env, self.fallback_model_env) or self.model

    def resolved_api_key(self) -> str | None:
        return _first_env(self.api_key_env, self.fallback_api_key_env)

    def resolved_base_url(self) -> str | None:
        return self.base_url or _first_env(self.base_url_env, self.fallback_base_url_env)


@dataclass(slots=True)
class ObservabilityConfig:
    provider: str = "langfuse"
    enabled: bool = True
    public_key_env: str = "LANGFUSE_PUBLIC_KEY"
    secret_key_env: str = "LANGFUSE_SECRET_KEY"
    base_url_env: str = "LANGFUSE_BASE_URL"
    default_base_url: str = "https://cloud.langfuse.com"
    capture_io: bool = True


@dataclass(slots=True)
class AgentConfig:
    name: str = "MINIAgent"
    max_turns: int = 40
    workspace_dir: str = "workspace"
    memory_dir: str = "memory"
    log_dir: str = "workspace/logs"
    system_prompt_path: str = "src/miniagent/prompts/system.md"
    llm: LLMConfig = field(default_factory=LLMConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    context: dict[str, Any] = field(default_factory=dict)
    tools: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = "config/agent.yaml") -> "AgentConfig":
        p = Path(path)
        load_dotenv_if_present(p.parent if p.parent else Path.cwd())
        load_dotenv_if_present(Path.cwd())
        if not p.exists():
            data: dict[str, Any] = {}
        elif yaml is not None:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        else:
            data = _simple_yaml_load(p.read_text(encoding="utf-8"))
        llm = LLMConfig(**(data.get("llm") or {}))
        obs = ObservabilityConfig(**(data.get("observability") or {}))
        return cls(
            name=data.get("name", "MINIAgent"),
            max_turns=int(data.get("max_turns", 40)),
            workspace_dir=data.get("workspace_dir", "workspace"),
            memory_dir=data.get("memory_dir", "memory"),
            log_dir=data.get("log_dir", "workspace/logs"),
            system_prompt_path=data.get("system_prompt_path", "src/miniagent/prompts/system.md"),
            llm=llm,
            observability=obs,
            context=data.get("context") or {},
            tools=data.get("tools") or {},
        )

    def llm_api_key(self) -> str | None:
        return self.llm.resolved_api_key()

    def llm_model(self) -> str:
        return self.llm.resolved_model()

    def llm_base_url(self) -> str | None:
        return self.llm.resolved_base_url()

    # Backward-compatible method name used by older code/tests.
    def openai_api_key(self) -> str | None:
        return self.llm_api_key()

    def langfuse_configured(self) -> bool:
        obs = self.observability
        return bool(
            obs.enabled
            and obs.provider == "langfuse"
            and os.getenv(obs.public_key_env)
            and os.getenv(obs.secret_key_env)
        )


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Tiny fallback parser for the project's simple config YAML.

    It supports top-level keys and one-level nested mappings; use PyYAML in
    production for full YAML semantics.
    """
    data: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if indent == 0:
            if value == "":
                current = {}
                data[key] = current
            else:
                data[key] = _parse_scalar(value)
                current = None
        elif current is not None:
            current[key] = _parse_scalar(value)
    return data


def _parse_scalar(value: str) -> Any:
    if value in {"null", "None", "~"}:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
