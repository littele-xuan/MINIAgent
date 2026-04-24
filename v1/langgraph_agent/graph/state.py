from __future__ import annotations

from typing import Any, TypedDict


class AgentGraphState(TypedDict, total=False):
    query: str
    user_id: str
    thread_id: str
    run_id: str
    accepted_output_modes: list[str]
    max_steps: int
    step_count: int

    history: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    task_state: dict[str, Any]

    selected_skill: dict[str, Any] | None
    memory_context: dict[str, Any]
    visible_tools: list[dict[str, Any]]
    context_packet: dict[str, Any]

    plan: dict[str, Any] | None
    pending_mcp_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    delegated_result: dict[str, Any] | None
    final_response: dict[str, Any] | None
