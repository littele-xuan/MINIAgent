"""
mcp/tools/internal/web_search.py
──────────────────────────────────
Real web search using DuckDuckGo HTML — no API key required.
Falls back to Bing HTML if DDG is unavailable.
Category: INTERNAL_UTILITY — immutable via governance API.

Implementation: pure requests + html.parser (stdlib only).
"""

import re
import time
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import quote_plus, urlencode

import requests

from mcp_lib.registry.models import ToolCategory
from mcp_lib.tools.base import tool_def

# ── HTTP session (shared for connection pooling) ──────────────────────────────

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
})

_TIMEOUT = 12
_LAST_REQUEST = 0.0
_MIN_INTERVAL = 1.2   # seconds between requests (be polite)


def _throttle():
    global _LAST_REQUEST
    elapsed = time.time() - _LAST_REQUEST
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _LAST_REQUEST = time.time()


# ── DDG Lite HTML parser ──────────────────────────────────────────────────────

class _DDGLiteParser(HTMLParser):
    """
    Parses https://lite.duckduckgo.com/lite/ response.

    DDG Lite layout (table-based):
      <a class="result-link">Title</a>   → result title + redirect url
      <td class="result-snippet">…</td>  → snippet
      <span class="link-text">url</span> → display url
    """

    def __init__(self):
        super().__init__()
        self.results: list[dict] = []
        self._cur: Optional[dict] = None
        self._in_link = False
        self._in_snippet = False
        self._in_link_text = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        cls = attrs.get("class", "")
        href = attrs.get("href", "")

        if tag == "a" and "result-link" in cls:
            self._cur = {"title": "", "url": href, "snippet": ""}
            self._in_link = True

        elif tag == "td" and "result-snippet" in cls:
            self._in_snippet = True

        elif tag == "span" and "link-text" in cls:
            self._in_link_text = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        if self._in_link and self._cur is not None:
            self._cur["title"] += data
        elif self._in_snippet and self._cur is not None:
            self._cur["snippet"] += data + " "
        elif self._in_link_text and self._cur is not None:
            if not self._cur.get("display_url"):
                self._cur["display_url"] = data

    def handle_endtag(self, tag):
        if tag == "a" and self._in_link:
            self._in_link = False
            if self._cur and self._cur.get("title"):
                self.results.append(self._cur)
                self._cur = None
        elif tag == "td" and self._in_snippet:
            self._in_snippet = False
        elif tag == "span" and self._in_link_text:
            self._in_link_text = False


def _clean(text: str) -> str:
    """Remove residual HTML tags and normalize whitespace."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Primary: DuckDuckGo Lite ──────────────────────────────────────────────────

def _search_ddg(query: str, max_results: int) -> list[dict]:
    _throttle()
    url = "https://lite.duckduckgo.com/lite/"
    resp = _SESSION.post(
        url,
        data={"q": query, "kl": "cn-zh"},
        timeout=_TIMEOUT,
        allow_redirects=True,
    )
    resp.raise_for_status()

    parser = _DDGLiteParser()
    parser.feed(resp.text)

    results = []
    for r in parser.results[:max_results]:
        results.append({
            "title":   _clean(r.get("title", "")),
            "url":     r.get("display_url") or r.get("url", ""),
            "snippet": _clean(r.get("snippet", "")),
        })
    return results


# ── Fallback: Bing HTML ───────────────────────────────────────────────────────

def _search_bing(query: str, max_results: int) -> list[dict]:
    _throttle()
    url = f"https://www.bing.com/search?q={quote_plus(query)}&cc=CN&setlang=zh-CN"
    resp = _SESSION.get(url, timeout=_TIMEOUT)
    resp.raise_for_status()

    # Extract title + url + snippet via regex on Bing's li.b_algo structure
    title_pattern   = re.compile(r'<h2[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
    snippet_pattern = re.compile(r'<p class="b_lineclamp\d+"[^>]*>(.*?)</p>', re.DOTALL)

    titles   = title_pattern.findall(resp.text)
    snippets = snippet_pattern.findall(resp.text)

    results = []
    for i, (href, title) in enumerate(titles[:max_results]):
        snippet = snippets[i] if i < len(snippets) else ""
        results.append({
            "title":   _clean(title),
            "url":     href,
            "snippet": _clean(snippet),
        })
    return results


# ── Main handler ──────────────────────────────────────────────────────────────

def _handle(query: str, max_results: int = 5, engine: str = "auto") -> str:
    """
    Search the web and return formatted results.

    Args:
        query:       Search query string
        max_results: Number of results to return (1-10)
        engine:      "auto" | "ddg" | "bing"
    """
    max_results = max(1, min(10, max_results))

    results: list[dict] = []
    errors:  list[str]  = []

    engines_to_try = (
        [_search_ddg, _search_bing]  if engine == "auto" else
        [_search_ddg]                if engine == "ddg"  else
        [_search_bing]
    )

    for search_fn in engines_to_try:
        try:
            results = search_fn(query, max_results)
            if results:
                break
        except Exception as e:
            errors.append(f"{search_fn.__name__}: {e}")

    if not results:
        err_detail = "; ".join(errors) if errors else "no results found"
        return f"[web_search] 未找到结果。错误: {err_detail}"

    lines = [f"🔍 搜索「{query}」— {len(results)} 条结果\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        if r.get("url"):
            lines.append(f"   {r['url']}")
        if r.get("snippet"):
            lines.append(f"   {r['snippet']}")
        lines.append("")

    return "\n".join(lines).strip()


def make_entry():
    return tool_def(
        name="web_search",
        description=(
            "Search the web using DuckDuckGo (fallback: Bing). "
            "No API key required. Returns titles, URLs, and snippets."
        ),
        handler=_handle,
        category=ToolCategory.INTERNAL_UTILITY,
        properties={
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "max_results": {
                "type": "integer",
                "description": "Number of results to return (1-10, default 5)",
                "default": 5,
            },
            "engine": {
                "type": "string",
                "enum": ["auto", "ddg", "bing"],
                "description": "Search engine: auto (try DDG first, fallback Bing)",
                "default": "auto",
            },
        },
        required=["query"],
        tags=["search", "web", "internet", "query"],
        aliases=["search", "google", "bing"],
    )
