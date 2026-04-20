"""
mcp_lib/tools/external/seed_tools.py
──────────────────────────────────────
Seed external tools registered at startup.

These are EXTERNAL category tools — fully manageable via governance API.
They serve as examples and useful utilities.
"""

from __future__ import annotations

import random
import uuid as _uuid
from typing import TYPE_CHECKING

from mcp_lib.registry.models import ToolCategory
from mcp_lib.tools.base import tool_def

if TYPE_CHECKING:
    from mcp_lib.registry.registry import ToolRegistry


# ── random_joke ───────────────────────────────────────────────────────────────

_JOKES = [
    "为什么程序员总是分不清万圣节和圣诞节？因为 Oct 31 == Dec 25。",
    "一个SQL语句走进酒吧，走向两张桌子问道：'我可以JOIN你们吗？'",
    "世界上有10种人：懂二进制的和不懂二进制的。",
    "程序员的老婆让他去买一升牛奶，如果有鸡蛋就买一打。他回来带了12升牛奶。",
    "为什么程序员喜欢黑暗模式？因为光会吸引虫子（bug）。",
    "递归的定义：如果你还不理解递归，请参见'递归'。",
    "一个程序员去世后上了天堂，上帝说：你有两个选择，天堂或地狱。"
    "程序员说：先看看地狱吧。地狱里全是程序员在写代码。"
    "程序员说：这和天堂有什么区别？上帝说：地狱没有版本控制。",
    "Why do Java developers wear glasses? Because they don't C#.",
    "A programmer's wife says: 'Go to the store, get a gallon of milk, "
    "and if they have eggs, get a dozen.' He comes back with 12 gallons of milk.",
    "There are only 2 hard problems in CS: cache invalidation, naming things, "
    "and off-by-one errors.",
]


def _joke_handler() -> str:
    return random.choice(_JOKES)


# ── uuid_gen ──────────────────────────────────────────────────────────────────

def _uuid_handler(count: int = 1, version: int = 4) -> str:
    count = max(1, min(20, count))
    ids = []
    for _ in range(count):
        if version == 1:
            ids.append(str(_uuid.uuid1()))
        else:
            ids.append(str(_uuid.uuid4()))
    return "\n".join(ids)


# ── timestamp ─────────────────────────────────────────────────────────────────

def _timestamp_handler(format: str = "iso", timezone: str = "utc") -> str:
    from datetime import datetime, timezone as tz
    now = datetime.now(tz.utc)
    if format == "iso":
        return now.isoformat()
    elif format == "unix":
        return str(int(now.timestamp()))
    elif format == "human":
        return now.strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        try:
            return now.strftime(format)
        except Exception as e:
            return f"[timestamp ERROR] {e}"


# ── base64_tool ───────────────────────────────────────────────────────────────

def _base64_handler(text: str, action: str = "encode") -> str:
    import base64
    try:
        if action == "encode":
            return base64.b64encode(text.encode("utf-8")).decode("ascii")
        elif action == "decode":
            return base64.b64decode(text.encode("ascii")).decode("utf-8")
        else:
            return f"[base64 ERROR] Unknown action '{action}'. Use 'encode' or 'decode'."
    except Exception as e:
        return f"[base64 ERROR] {e}"


# ── hash_tool ─────────────────────────────────────────────────────────────────

def _hash_handler(text: str, algorithm: str = "sha256") -> str:
    import hashlib
    algo = algorithm.lower().replace("-", "")
    try:
        h = hashlib.new(algo)
        h.update(text.encode("utf-8"))
        return h.hexdigest()
    except ValueError:
        return f"[hash ERROR] Unsupported algorithm '{algorithm}'. Try: md5, sha1, sha256, sha512."


# ── make_entries ──────────────────────────────────────────────────────────────

def make_entries() -> list:
    return [
        tool_def(
            name="random_joke",
            description="Return a random programming joke or pun.",
            handler=_joke_handler,
            category=ToolCategory.EXTERNAL,
            properties={},
            required=[],
            tags=["fun", "joke", "demo"],
            aliases=["joke"],
            created_by="system_seed",
        ),
        tool_def(
            name="uuid_gen",
            description="Generate one or more UUIDs (v1 or v4).",
            handler=_uuid_handler,
            category=ToolCategory.EXTERNAL,
            properties={
                "count":   {"type": "integer", "description": "Number of UUIDs to generate (1-20, default 1)"},
                "version": {"type": "integer", "description": "UUID version: 1 or 4 (default 4)", "enum": [1, 4]},
            },
            required=[],
            tags=["utility", "uuid", "id", "demo"],
            aliases=["uuid"],
            created_by="system_seed",
        ),
        tool_def(
            name="timestamp",
            description="Get the current UTC timestamp in various formats.",
            handler=_timestamp_handler,
            category=ToolCategory.EXTERNAL,
            properties={
                "format": {
                    "type": "string",
                    "description": "Output format: 'iso' (default), 'unix', 'human', or strftime pattern",
                    "default": "iso",
                },
            },
            required=[],
            tags=["utility", "time", "datetime", "demo"],
            aliases=["now", "time"],
            created_by="system_seed",
        ),
        tool_def(
            name="base64",
            description="Encode or decode a string using Base64.",
            handler=_base64_handler,
            category=ToolCategory.EXTERNAL,
            properties={
                "text":   {"type": "string", "description": "Text to encode or decode"},
                "action": {"type": "string", "enum": ["encode", "decode"], "description": "Action: 'encode' or 'decode' (default 'encode')"},
            },
            required=["text"],
            tags=["utility", "encoding", "crypto"],
            aliases=["b64"],
            created_by="system_seed",
        ),
        tool_def(
            name="hash",
            description="Compute a cryptographic hash of a string (md5, sha1, sha256, sha512).",
            handler=_hash_handler,
            category=ToolCategory.EXTERNAL,
            properties={
                "text":      {"type": "string", "description": "Text to hash"},
                "algorithm": {"type": "string", "description": "Hash algorithm: md5, sha1, sha256 (default), sha512"},
            },
            required=["text"],
            tags=["utility", "crypto", "hash"],
            aliases=["digest"],
            created_by="system_seed",
        ),
    ]
