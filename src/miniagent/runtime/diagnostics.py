from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

from .config import AgentConfig
from .env import load_dotenv_if_present


def collect_diagnostics(config_path: str | Path = "config/agent.yaml") -> dict[str, Any]:
    config_path = Path(config_path)
    load_dotenv_if_present(config_path.resolve().parent if config_path.exists() else Path.cwd())
    load_dotenv_if_present(Path.cwd())
    config = AgentConfig.load(config_path)
    return {
        "config_path": str(config_path),
        "provider": config.llm.provider,
        "api_mode": config.llm.api_mode,
        "model": config.llm_model(),
        "model_env": config.llm.model_env,
        "model_env_present": bool(os.getenv(config.llm.model_env)),
        "api_base": config.llm_base_url(),
        "api_base_env": config.llm.base_url_env,
        "api_base_env_present": bool(os.getenv(config.llm.base_url_env)),
        "api_key_present": bool(config.llm_api_key()),
        "api_key_env": config.llm.api_key_env,
        "fallback_api_key_env": config.llm.fallback_api_key_env,
        "workspace_dir": config.workspace_dir,
        "memory_dir": config.memory_dir,
        "log_dir": config.log_dir,
        "langfuse_enabled_in_config": config.observability.enabled,
        "langfuse_keys_present": bool(os.getenv(config.observability.public_key_env) and os.getenv(config.observability.secret_key_env)),
        "langfuse_base_url": os.getenv(config.observability.base_url_env) or config.observability.default_base_url,
        "openai_package_installed": importlib.util.find_spec("openai") is not None,
        "langfuse_package_installed": importlib.util.find_spec("langfuse") is not None,
        "pyyaml_package_installed": importlib.util.find_spec("yaml") is not None,
    }


def format_diagnostics(diag: dict[str, Any]) -> str:
    lines = ["MINIAgent doctor"]
    for key, value in diag.items():
        lines.append(f"- {key}: {value}")
    if not diag.get("api_key_present"):
        lines.append("\n提示：真实 LLM 运行需要设置 MCP_API_KEY，或在项目根目录 .env 中写入 MCP_API_KEY=...")
    if not diag.get("model_env_present"):
        lines.append("提示：建议设置 MCP_MODEL；否则会使用 config/agent.yaml 里的默认 model。")
    if not diag.get("api_base_env_present"):
        lines.append("提示：建议设置 MCP_API_BASE；如果留空，将使用 OpenAI SDK 默认 endpoint。")
    if diag.get("langfuse_enabled_in_config") and not diag.get("langfuse_keys_present"):
        lines.append("提示：未检测到 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY，将自动回退到官方 OpenAI SDK。")
    return "\n".join(lines)
