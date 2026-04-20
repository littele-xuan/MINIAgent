---
name: weather-brief
description: Analyze weather-related requests and decide whether to use get_weather and web_search to answer.
when_to_use: Use this when the user asks about weather, forecast, city conditions, temperature, rain, weather impacts, 天气, 气温, 下雨, 城市天气, or current climate context.
allowed-tools:
  - get_weather
  - web_search
output-modes:
  - text/plain
  - application/json
accepted-output-modes:
  - text/plain
  - application/json
examples:
  - 帮我看一下上海现在天气
  - 给我一个简短天气摘要，并补充是否有出行影响
mcp:
  protocol: live-mcp
  selection: use get_weather first, then web_search only if broader context is needed
  batch_calls: false
a2a:
  enabled: false
---

# Weather Brief

Use this skill for weather-oriented requests.

## Workflow
1. Prefer `get_weather` for current conditions.
2. Use `web_search` only when the user asks for broader context, recent disruptions, travel impacts, or when current weather alone is insufficient.
3. Keep the answer concise and factual.
4. Do not use unrelated tools.

## Output
Return a short Chinese answer with the key conditions first, then any extra context.
