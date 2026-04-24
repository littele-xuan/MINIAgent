from __future__ import annotations

import json
import re
from typing import Any

from ..schemas import FinalResponse, MCPCall, PlanDecision


class HeuristicPlanner:
    """Debug-only fallback used when structured LLM planning fails."""

    def build_plan(
        self,
        *,
        query: str,
        visible_tools: list[dict[str, Any]],
        memory_context: dict[str, Any] | None,
        tool_results: list[dict[str, Any]],
    ) -> PlanDecision:
        visible = {tool.get('name'): tool for tool in visible_tools}
        if tool_results:
            latest = tool_results[-1]
            text = latest.get('text', '')
            # If the last step wrote a note and read_note exists, continue the explicit workflow.
            if latest.get('tool_name') == 'write_note' and 'read_note' in visible and ('读回来' in query or 'read' in query.lower()):
                return PlanDecision(thought='Read the note after writing it.', mode='mcp', mcp_calls=[MCPCall(tool_name='read_note', arguments={})])
            return PlanDecision(thought='Return latest MCP observation.', mode='final', final=FinalResponse(output_mode='text/plain', text=text))

        if 'write_note' in visible and ('write_note' in query or '写入' in query):
            content_match = re.search(r'`([^`]+)`', query, flags=re.DOTALL)
            content = content_match.group(1) if content_match else query
            return PlanDecision(
                thought='Use MCP write_note for the explicit write request.',
                mode='mcp',
                mcp_calls=[MCPCall(tool_name='write_note', arguments={'content': content})],
            )
        if 'read_note' in visible and ('read_note' in query or '读回来' in query):
            return PlanDecision(thought='Use MCP read_note for the explicit read request.', mode='mcp', mcp_calls=[MCPCall(tool_name='read_note', arguments={})])
        if 'huge_log' in visible and ('huge_log' in query or '很大的日志' in query or '大日志' in query):
            topic = 'demo'
            topic_match = re.search(r'topic\s*(?:设为|=|:)?\s*([A-Za-z0-9_.-]+)', query)
            if topic_match:
                topic = topic_match.group(1)
            return PlanDecision(
                thought='Use MCP huge_log for explicit log generation.',
                mode='mcp',
                mcp_calls=[MCPCall(tool_name='huge_log', arguments={'topic': topic})],
            )
        if 'calculator' in visible and any(token in query for token in ('计算', 'calculate', '*', '+', '-', '/')):
            expr_match = re.search(r'([0-9][0-9\s+\-*/().]+)', query)
            expression = expr_match.group(1).strip() if expr_match else query
            return PlanDecision(
                thought='Use calculator MCP tool for arithmetic.',
                mode='mcp',
                mcp_calls=[MCPCall(tool_name='calculator', arguments={'expression': expression})],
            )
        for name in visible:
            if name and name in query:
                return PlanDecision(thought=f'Execute explicitly named MCP tool {name}.', mode='mcp', mcp_calls=[MCPCall(tool_name=name, arguments={})])

        memories = []
        for item in (memory_context or {}).get('memories', []):
            if isinstance(item, dict):
                memories.append(item.get('text', ''))
            else:
                memories.append(str(item))
        if memories and any(token in query for token in ('记忆', '偏好', '约束', 'memory')):
            return PlanDecision(
                thought='Answer from retrieved memory context.',
                mode='final',
                final=FinalResponse(output_mode='text/plain', text='\n'.join(f'- {m}' for m in memories if m)),
            )
        return PlanDecision(thought='Answer directly without tool use.', mode='final', final=FinalResponse(output_mode='text/plain', text='我已收到请求，但没有必要调用工具。'))
