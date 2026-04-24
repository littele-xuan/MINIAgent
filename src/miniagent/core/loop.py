from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..context.manager import ContextManager
from ..core.outcome import AgentResult, StepOutcome, ToolResult
from ..llm.openai_client import OpenAIResponsesClient
from ..memory.store import FileMemoryStore
from ..runtime.logging import JsonlRunLogger
from ..runtime.workspace import Workspace
from ..tools.base import ToolContext
from ..tools.registry import ToolRegistry
from .state import AgentState

EventCallback = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class AgentLoop:
    llm: OpenAIResponsesClient
    tools: ToolRegistry
    context: ContextManager
    memory: FileMemoryStore
    workspace: Workspace
    system_prompt: str
    log_dir: str = "workspace/logs"
    max_turns: int = 40
    event_callback: EventCallback | None = None
    max_empty_response_retries: int = 5

    def run(self, state: AgentState) -> AgentResult:
        memory_recall = self.memory.recall(state.user_input).format_for_prompt()
        packet = self.context.start_packet(
            system_prompt=self.system_prompt,
            user_input=state.user_input,
            memory_context=memory_recall,
            metadata=state.metadata,
        )
        state.input_items = self.context.initial_input_items(packet)
        logger = JsonlRunLogger.create(self.log_dir, state.session_id)
        logger.write("run_start", {"user_input": state.user_input, "memory_recall": memory_recall})
        last_text = ""
        empty_retries = 0
        while state.turn < self.max_turns:
            state.turn += 1
            self._emit({"event": "llm_start", "turn": state.turn})
            logger.write("llm_start", {"turn": state.turn})
            tool_events_start = len(state.tool_events)
            response = self.llm.create_response(
                instructions=packet.system_prompt,
                input_items=state.input_items,
                tools=self.tools.openai_schemas(),
                metadata={"session_id": state.session_id, "agent": "MINIAgent", "turn": str(state.turn)},
            )
            state.add_usage(response.usage)
            text = (response.text or "").strip()
            if text:
                last_text = text
            self._emit({"event": "llm_end", "turn": state.turn, "text": response.text, "tool_calls": [tc.name for tc in response.tool_calls]})
            logger.write(
                "llm_end",
                {"turn": state.turn, "text": response.text, "tool_calls": [_tool_call_dict(tc) for tc in response.tool_calls], "usage": response.usage, "raw_id": response.raw_id},
            )
            if response.output_items:
                state.input_items.extend(response.output_items)

            if not response.tool_calls:
                if text:
                    state.final_text = text
                    state.exit_reason = "final_answer"
                    new_tool_events = state.tool_events[tool_events_start:]
                    self.context.after_turn(turn=state.turn, user_input=state.user_input, assistant_text=text, tool_events=new_tool_events, metadata=state.metadata)
                    break
                empty_retries += 1
                if empty_retries <= self.max_empty_response_retries and state.turn < self.max_turns:
                    task_hint = (state.user_input[:400] + "...") if len(state.user_input) > 400 else state.user_input
                    checkpoint = self.context.working_checkpoint
                    parts = [
                        "Your previous response was empty — you must NOT return an empty response.",
                        f"Current task: {task_hint}",
                    ]
                    if checkpoint:
                        parts.append(f"Working checkpoint (resume from here): {checkpoint}")
                    parts.append(
                        "Either call the next required tool to continue making progress, "
                        "or write your final answer if all steps are already complete. "
                        "Never return empty."
                    )
                    msg = "\n".join(parts)
                    state.input_items.append({"role": "user", "content": msg})
                    self._emit({"event": "llm_empty_retry", "turn": state.turn, "attempt": empty_retries})
                    logger.write("llm_empty_retry", {"turn": state.turn, "attempt": empty_retries})
                    continue
                state.final_text = last_text or "The LLM endpoint returned an empty response and no tool calls."
                state.exit_reason = "empty_response"
                break

            empty_retries = 0
            step = self._execute_tools(state, response.tool_calls)
            state.input_items.extend(_tool_outputs_for_model(response.tool_calls, step.tool_results, max_chars=self.context.compactor.max_tool_output_chars))
            new_tool_events = state.tool_events[tool_events_start:]
            self.context.after_turn(turn=state.turn, user_input=state.user_input, assistant_text=text, tool_events=new_tool_events, metadata=state.metadata)
            logger.write("tool_results", {"turn": state.turn, "results": [r.to_model_output() for r in step.tool_results]})
            if any(not result.should_continue for result in step.tool_results):
                state.final_text = step.tool_results[-1].content
                state.exit_reason = "blocked_for_user"
                break
        else:
            state.final_text = last_text or "Max turns exceeded before a final answer was produced."
            state.exit_reason = "max_turns_exceeded"
        logger.write("run_end", {"exit_reason": state.exit_reason, "final_text": state.final_text, "usage": state.usage})
        self.memory.log_session_event(state.session_id, {"event": "run_end", "exit_reason": state.exit_reason, "final_text": state.final_text, "usage": state.usage})
        return AgentResult(final_text=state.final_text, turns=state.turn, exit_reason=state.exit_reason, tool_events=state.tool_events, session_id=state.session_id, usage=state.usage)

    def _execute_tools(self, state: AgentState, tool_calls) -> StepOutcome:
        ctx = ToolContext(workspace=self.workspace, memory=self.memory, session_id=state.session_id, metadata=state.metadata)
        results: list[ToolResult] = []
        for call in tool_calls:
            self._emit({"event": "tool_start", "turn": state.turn, "name": call.name, "arguments": call.arguments})
            try:
                result = self.tools.dispatch(call.name, call.arguments, ctx)
            except Exception as exc:
                result = ToolResult(False, f"Tool {call.name} failed: {type(exc).__name__}: {exc}", error=str(exc))
            results.append(result)
            event = {"event": "tool_end", "turn": state.turn, "name": call.name, "ok": result.ok, "content": result.content[:2000], "error": result.error}
            state.tool_events.append(event)
            self._emit(event)
        return StepOutcome(tool_calls=list(tool_calls), tool_results=results)

    def _emit(self, event: dict[str, Any]) -> None:
        if self.event_callback:
            self.event_callback(event)


def _tool_outputs_for_model(tool_calls, results: list[ToolResult], *, max_chars: int = 12000) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for call, result in zip(tool_calls, results):
        output = result.to_model_output()
        if len(output) > max_chars:
            half = max_chars // 2
            output = output[:half] + f"\n... <compacted {len(output) - max_chars} chars> ...\n" + output[-half:]
        items.append({"type": "function_call_output", "call_id": call.call_id, "output": output})
    return items


def _tool_call_dict(tc) -> dict[str, Any]:
    return {"id": tc.id, "call_id": tc.call_id, "name": tc.name, "arguments": tc.arguments}
