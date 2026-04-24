from __future__ import annotations

from _bootstrap import ROOT
from miniagent.context.manager import ContextManager
from miniagent.memory import FileMemoryStore, MemoryItem

store = FileMemoryStore.create(ROOT / "workspace" / "offline_memory")
item = MemoryItem(layer="facts", content="MINIAgent offline test prefers exact file_patch for small code edits.", evidence="offline test inserted this durable fact", tags=["offline", "tooling"])
ok, msg = store.commit_item(item)
assert ok, msg
recall = store.recall("file_patch small code edits")
assert recall.items, "expected at least one recalled memory item"
manager = ContextManager()
packet = manager.start_packet(system_prompt="system", user_input="test task", memory_context=recall.format_for_prompt(), metadata={"working_checkpoint": "checkpoint A"})
assert "checkpoint A" in packet.user_packet
manager.after_turn(turn=1, user_input="test task", assistant_text="used tools", tool_events=[{"event":"tool_end", "name":"file_read"}], metadata={})
assert manager.summaries, "summary missing"
print("memory/context ok")
