from __future__ import annotations

from _bootstrap import ROOT, require_llm_env
from miniagent import MINIAgent

require_llm_env()
agent = MINIAgent.create_default(ROOT / "config" / "agent.yaml")
expected = "MINIAGENT_REAL_SMOKE_20260424"
response = agent.llm.create_response(
    instructions=f"You are a strict smoke test. Reply with exactly this token and nothing else: {expected}",
    input_items=[{"role": "user", "content": "Return the exact smoke token."}],
    tools=[],
    metadata={"test": "01_llm_smoke"},
)
text = (response.text or "").strip()
print(text)
print({"langfuse_enabled": agent.llm.langfuse_enabled, "client": agent.llm.client_name, "api_mode": agent.llm.api_mode, "usage": response.usage})
assert expected in text, f"smoke token not found in response: {text!r}"
