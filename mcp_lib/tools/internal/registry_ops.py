"""
mcp/tools/internal/registry_ops.py
────────────────────────────────────
Governance tools — INTERNAL_SYSTEM category.

These tools let the LLM manage the external tool set via MCP,
while remaining themselves immutable (protected = True).

Rules:
  ✓ Can list/inspect any tool (read-only on protected tools)
  ✓ Can add / update / remove / enable / disable EXTERNAL tools
  ✗ Cannot modify INTERNAL_SYSTEM or INTERNAL_UTILITY tools
  ✗ Cannot modify themselves

Each tool takes simple JSON arguments and returns human-readable strings,
making them easy to call from any LLM agent.
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Optional

from mcp_lib.registry.models import ToolCategory, ToolEntry, ToolStatus
from mcp_lib.tools.base import tool_def

if TYPE_CHECKING:
    from mcp_lib.registry.registry import ToolRegistry


# ── Handler factory (closure over registry) ──────────────────────────────────
# All governance handlers are closures: they capture the live registry
# so they always operate on the current state.

def _make_handlers(registry: "ToolRegistry") -> dict[str, Callable]:
    """Return a dict of {tool_name: handler_fn} bound to this registry."""

    # ── tool_list ────────────────────────────────────────────────────────────

    def tool_list(
        category: Optional[str] = None,
        tags: Optional[str] = None,
        include_disabled: bool = False,
        include_deprecated: bool = False,
    ) -> str:
        cat = ToolCategory(category) if category else None
        tag_list = [t.strip() for t in tags.split(",")] if tags else None

        tools = registry.list_tools(
            enabled_only=not include_disabled,
            category=cat,
            tags=tag_list,
            include_deprecated=include_deprecated,
            include_system=False,   # hide governance tools from LLM list
        )

        if not tools:
            return "（无工具匹配条件）"

        lines = [f"工具列表 ({len(tools)} 个):\n"]
        for t in sorted(tools, key=lambda x: x.name):
            state = "✓" if t.enabled else "✗"
            dep   = " [已废弃]" if t.deprecated else ""
            lines.append(
                f"  {state} [{t.category.label}] {t.name} v{t.version}{dep}\n"
                f"     {t.description[:80]}"
            )
            if t.tags:
                lines.append(f"     标签: {', '.join(t.tags)}")
            lines.append("")
        return "\n".join(lines)

    # ── tool_info ────────────────────────────────────────────────────────────

    def tool_info(name: str) -> str:
        try:
            t = registry.get(name)
        except Exception as e:
            return f"错误: {e}"
        d = t.to_dict(include_history=True)
        return json.dumps(d, indent=2, ensure_ascii=False, default=str)

    # ── tool_add ─────────────────────────────────────────────────────────────

    def tool_add(
        name:        str,
        description: str,
        code:        str,
        version:     str = "1.0.0",
        tags:        str = "",
        aliases:     str = "",
    ) -> str:
        """
        Register a new EXTERNAL tool with a Python handler.

        `code` must define a function named `handler(**kwargs) -> str`.
        Example:
            def handler(x: int, y: int) -> str:
                return str(x + y)
        """
        if registry.has(name):
            return f"错误: 工具 '{name}' 已存在。请先用 tool_remove 删除或用 tool_update 更新。"
        try:
            ns: dict[str, Any] = {}
            exec(compile(code, f"<tool:{name}>", "exec"), ns)
            fn = ns.get("handler")
            if fn is None or not callable(fn):
                return "错误: code 必须定义名为 'handler' 的可调用函数。"
        except Exception as e:
            return f"代码编译错误: {e}"

        entry = tool_def(
            name=name,
            description=description,
            handler=fn,
            category=ToolCategory.EXTERNAL,
            properties={},   # LLM can provide richer schema via tool_update
            version=version,
            tags=[t.strip() for t in tags.split(",") if t.strip()],
            aliases=[a.strip() for a in aliases.split(",") if a.strip()],
            created_by="llm_agent",
        )
        registry.register(entry)
        return f"✓ 工具 '{name}' v{version} 注册成功 [external]"

    # ── tool_update ──────────────────────────────────────────────────────────

    def tool_update(
        name:        str,
        description: Optional[str] = None,
        code:        Optional[str] = None,
        version:     Optional[str] = None,
        tags:        Optional[str] = None,
        changelog:   str           = "",
    ) -> str:
        kwargs: dict[str, Any] = {}
        if description is not None:
            kwargs["description"] = description
        if version is not None:
            kwargs["version"] = version
        if tags is not None:
            kwargs["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
        if code is not None:
            try:
                ns: dict[str, Any] = {}
                exec(compile(code, f"<tool:{name}>", "exec"), ns)
                fn = ns.get("handler")
                if fn is None or not callable(fn):
                    return "错误: code 必须定义名为 'handler' 的可调用函数。"
                kwargs["handler"] = fn
            except Exception as e:
                return f"代码编译错误: {e}"
        if not kwargs:
            return "错误: 没有提供任何更新字段。"
        try:
            registry.update(name, changelog=changelog, **kwargs)
            return f"✓ 工具 '{name}' 更新成功"
        except Exception as e:
            return f"错误: {e}"

    # ── tool_remove ──────────────────────────────────────────────────────────

    def tool_remove(name: str) -> str:
        try:
            registry.remove(name)
            return f"✓ 工具 '{name}' 已删除"
        except Exception as e:
            return f"错误: {e}"

    # ── tool_enable / tool_disable ────────────────────────────────────────────

    def tool_enable(name: str) -> str:
        try:
            registry.enable(name)
            return f"✓ 工具 '{name}' 已启用"
        except Exception as e:
            return f"错误: {e}"

    def tool_disable(name: str) -> str:
        try:
            registry.disable(name)
            return f"✓ 工具 '{name}' 已禁用"
        except Exception as e:
            return f"错误: {e}"

    # ── tool_deprecate ────────────────────────────────────────────────────────

    def tool_deprecate(name: str, replaced_by: Optional[str] = None) -> str:
        try:
            registry.deprecate(name, replaced_by=replaced_by)
            msg = f"✓ 工具 '{name}' 已标记为废弃"
            if replaced_by:
                msg += f"，建议使用 '{replaced_by}'"
            return msg
        except Exception as e:
            return f"错误: {e}"

    # ── tool_alias ────────────────────────────────────────────────────────────

    def tool_alias(name: str, alias: str) -> str:
        try:
            registry.add_alias(name, alias)
            return f"✓ 别名 '{alias}' → '{name}' 已添加"
        except Exception as e:
            return f"错误: {e}"

    # ── tool_merge ────────────────────────────────────────────────────────────

    def tool_merge(source: str, target: str, keep_source: bool = False) -> str:
        try:
            registry.merge(source, target, keep_source=keep_source)
            action = "保留（已禁用）" if keep_source else "删除"
            return f"✓ '{source}' 已合并至 '{target}'，源工具已{action}"
        except Exception as e:
            return f"错误: {e}"

    # ── tool_versions ─────────────────────────────────────────────────────────

    def tool_versions(name: str) -> str:
        try:
            versions = registry.list_versions(name)
            return json.dumps(versions, indent=2, ensure_ascii=False)
        except Exception as e:
            return f"错误: {e}"

    # ── tool_search ───────────────────────────────────────────────────────────

    def tool_search(query: str) -> str:
        results = registry.search(query)
        if not results:
            return f"没有找到包含 '{query}' 的工具"
        lines = [f"搜索 '{query}' 结果 ({len(results)} 个):"]
        for t in results:
            lines.append(f"  • {t.name}  [{t.category.label}]  {t.description[:60]}")
        return "\n".join(lines)

    # ── registry_stats ────────────────────────────────────────────────────────

    def registry_stats() -> str:
        s = registry.stats()
        return json.dumps(s, indent=2, ensure_ascii=False)

    return {
        "tool_list":       tool_list,
        "tool_info":       tool_info,
        "tool_add":        tool_add,
        "tool_update":     tool_update,
        "tool_remove":     tool_remove,
        "tool_enable":     tool_enable,
        "tool_disable":    tool_disable,
        "tool_deprecate":  tool_deprecate,
        "tool_alias":      tool_alias,
        "tool_merge":      tool_merge,
        "tool_versions":   tool_versions,
        "tool_search":     tool_search,
        "registry_stats":  registry_stats,
    }


# ── ToolEntry definitions ─────────────────────────────────────────────────────

_TOOL_SPECS: list[tuple[str, str, dict, list]] = [
    (
        "tool_list",
        "列出注册表中的工具。可按分类、标签过滤，可包含已禁用/已废弃工具。",
        {
            "category":          {"type": "string", "enum": ["internal_utility","external"], "description": "按分类过滤"},
            "tags":              {"type": "string", "description": "逗号分隔的标签过滤"},
            "include_disabled":  {"type": "boolean", "description": "包含已禁用工具 (default false)"},
            "include_deprecated":{"type": "boolean", "description": "包含已废弃工具 (default false)"},
        },
        [],
    ),
    (
        "tool_info",
        "获取某个工具的完整元数据（含版本历史）。",
        {"name": {"type": "string", "description": "工具名称"}},
        ["name"],
    ),
    (
        "tool_add",
        (
            "注册一个新的 EXTERNAL 工具。"
            "code 必须定义 `def handler(**kwargs) -> str` 函数。"
            "示例: def handler(x: int, y: int) -> str:\\n    return str(x + y)"
        ),
        {
            "name":        {"type": "string",  "description": "工具唯一名称"},
            "description": {"type": "string",  "description": "工具描述"},
            "code":        {"type": "string",  "description": "包含 handler 函数的 Python 代码"},
            "version":     {"type": "string",  "description": "版本号 (default '1.0.0')"},
            "tags":        {"type": "string",  "description": "逗号分隔的标签"},
            "aliases":     {"type": "string",  "description": "逗号分隔的别名"},
        },
        ["name", "description", "code"],
    ),
    (
        "tool_update",
        "更新已有 EXTERNAL 工具的描述、代码或标签。",
        {
            "name":        {"type": "string", "description": "工具名称"},
            "description": {"type": "string", "description": "新描述"},
            "code":        {"type": "string", "description": "新的 handler 代码"},
            "version":     {"type": "string", "description": "新版本号"},
            "tags":        {"type": "string", "description": "新标签（逗号分隔）"},
            "changelog":   {"type": "string", "description": "更新说明"},
        },
        ["name"],
    ),
    (
        "tool_remove",
        "删除一个 EXTERNAL 工具（不可删除 internal 工具）。",
        {"name": {"type": "string", "description": "工具名称"}},
        ["name"],
    ),
    (
        "tool_enable",
        "启用一个已禁用的工具。",
        {"name": {"type": "string", "description": "工具名称"}},
        ["name"],
    ),
    (
        "tool_disable",
        "禁用一个工具（不会删除，可随时重新启用）。",
        {"name": {"type": "string", "description": "工具名称"}},
        ["name"],
    ),
    (
        "tool_deprecate",
        "将工具标记为已废弃，可指定替代工具。",
        {
            "name":        {"type": "string", "description": "工具名称"},
            "replaced_by": {"type": "string", "description": "替代工具名称"},
        },
        ["name"],
    ),
    (
        "tool_alias",
        "为工具添加别名，使其可被多个名称调用。",
        {
            "name":  {"type": "string", "description": "工具名称"},
            "alias": {"type": "string", "description": "别名"},
        },
        ["name", "alias"],
    ),
    (
        "tool_merge",
        "将源工具的别名合并到目标工具，将源名称重定向到目标。",
        {
            "source":      {"type": "string",  "description": "源工具名称"},
            "target":      {"type": "string",  "description": "目标工具名称"},
            "keep_source": {"type": "boolean", "description": "保留源工具（禁用而非删除）"},
        },
        ["source", "target"],
    ),
    (
        "tool_versions",
        "列出工具的历史版本记录。",
        {"name": {"type": "string", "description": "工具名称"}},
        ["name"],
    ),
    (
        "tool_search",
        "按关键词搜索工具（搜索名称、描述、标签）。",
        {"query": {"type": "string", "description": "搜索关键词"}},
        ["query"],
    ),
    (
        "registry_stats",
        "返回注册表统计信息：总数、启用数、各分类计数等。",
        {},
        [],
    ),
]


def make_entries(registry: "ToolRegistry") -> list[ToolEntry]:
    """
    Create all governance ToolEntry objects, binding handlers to the registry.
    Called once at startup by the internal tools bootstrap.
    """
    handlers = _make_handlers(registry)
    entries = []
    for name, desc, props, required in _TOOL_SPECS:
        entries.append(
            tool_def(
                name=name,
                description=desc,
                handler=handlers[name],
                category=ToolCategory.INTERNAL_SYSTEM,
                properties=props,
                required=required,
                tags=["governance", "registry", "management"],
            )
        )
    return entries
