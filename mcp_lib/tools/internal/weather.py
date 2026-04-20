"""
mcp/tools/internal/weather.py
───────────────────────────────
Weather query using wttr.in (free, no API key).
Falls back to mock data if network unavailable.
Category: INTERNAL_UTILITY — immutable via governance API.
"""

import requests

from mcp_lib.registry.models import ToolCategory
from mcp_lib.tools.base import tool_def, make_error_result

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "curl/7.80.0"})
_TIMEOUT = 8


def _handle(city: str, lang: str = "zh") -> str:
    """
    Query weather for city via wttr.in (free, request-based, no API key).
    Returns a concise one-line weather summary.
    """
    try:
        # wttr.in supports ?format=j1 for JSON or ?format=2 for one-line text
        url = f"https://wttr.in/{requests.utils.quote(city)}?format=j1&lang={lang}"
        resp = _SESSION.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        cur = data["current_condition"][0]
        temp_c     = cur["temp_C"]
        feels_like = cur["FeelsLikeC"]
        humidity   = cur["humidity"]
        wind_kmph  = cur["windspeedKmph"]
        desc_list  = cur.get("lang_zh", cur.get("weatherDesc", [{}]))
        desc       = desc_list[0].get("value", "N/A") if desc_list else "N/A"

        # Nearest area
        area  = data.get("nearest_area", [{}])[0]
        aname = area.get("areaName", [{}])[0].get("value", city)
        cname = area.get("country",  [{}])[0].get("value", "")

        return (
            f"📍 {aname}{',' + cname if cname else ''}\n"
            f"🌤 {desc}\n"
            f"🌡 气温 {temp_c}°C（体感 {feels_like}°C）\n"
            f"💧 湿度 {humidity}%  💨 风速 {wind_kmph} km/h"
        )

    except requests.RequestException as e:
        # Graceful fallback to mock data
        return _mock_weather(city)
    except Exception as e:
        return make_error_result("get_weather", e)


def _mock_weather(city: str) -> str:
    mock = {
        "beijing":  "晴，28°C，湿度 40%，东南风 12 km/h",
        "shanghai": "多云，23°C，湿度 65%，东风 8 km/h",
        "london":   "小雨，14°C，湿度 80%，西风 20 km/h",
        "tokyo":    "晴转多云，22°C，湿度 55%，北风 10 km/h",
        "new york": "阴，18°C，湿度 70%，西南风 15 km/h",
    }
    key = city.lower()
    info = mock.get(key, f"晴，20°C，湿度 50%")
    return f"📍 {city} [离线模拟]\n🌤 {info}"


def make_entry():
    return tool_def(
        name="get_weather",
        description=(
            "Get current weather for any city using wttr.in (no API key). "
            "Returns temperature, feels-like, humidity, wind speed."
        ),
        handler=_handle,
        category=ToolCategory.INTERNAL_UTILITY,
        properties={
            "city": {
                "type": "string",
                "description": "City name in English or Chinese, e.g. 'Beijing', 'London'"
            },
            "lang": {
                "type": "string",
                "description": "Language code for weather description: 'zh' or 'en' (default 'zh')",
                "default": "zh",
            },
        },
        required=["city"],
        tags=["weather", "climate", "temperature", "utility"],
        aliases=["weather"],
    )
