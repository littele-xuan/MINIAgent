from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from miniagent.runtime.env import load_dotenv_if_present  # noqa: E402

load_dotenv_if_present(ROOT)


def require_llm_env() -> None:
    missing = []
    if not os.getenv("MCP_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        missing.append("MCP_API_KEY")
    if not os.getenv("MCP_MODEL") and not os.getenv("OPENAI_MODEL"):
        missing.append("MCP_MODEL")
    if missing:
        raise SystemExit(
            "Real LLM tests require MCP_API_KEY and MCP_MODEL. "
            "MCP_API_BASE is recommended for your OpenAI-compatible endpoint. "
            f"Missing: {', '.join(missing)}"
        )


def reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def tool_names(result) -> list[str]:
    return [event.get("name") for event in result.tool_events if event.get("event") == "tool_end"]


def assert_exit_ok(result) -> None:
    assert result.exit_reason == "final_answer", f"unexpected exit={result.exit_reason}, final={result.final_text!r}"
    assert (result.final_text or "").strip(), "final answer is empty"


def assert_tool_used(result, *names: str) -> None:
    used = tool_names(result)
    missing = [name for name in names if name not in used]
    assert not missing, f"missing tools={missing}; used={used}; final={result.final_text!r}"


# Backward-compatible alias for older tests.
def require_api_key() -> None:
    require_llm_env()
