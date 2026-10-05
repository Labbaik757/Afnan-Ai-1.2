"""Web research — search → filter → open → extract.

A small workflow layer over the BrowserController (the only
browser driver): run a search on a known engine, recognize and
extract the results page into structured results (title, URL,
snippet, source), then open a chosen result in its own tab —
tagged with a task purpose — and extract its content.

Results are kept in a session object so multi-step plans can
search once and open results later, and every step returns the
structured outputs (with observation summaries) the Executor
records into AgentState.  Failures (no results, unknown index,
unreachable page) are structured BrowserExceptions, exactly
like the rest of the browser layer.
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, unquote_plus, urlparse

from afnan_ai.browser.base import BrowserErrorCode, BrowserException
from afnan_ai.browser.extraction import clean_document
from afnan_ai.log_config import get_logger
from afnan_ai.redaction import redact_text

logger = get_logger(__name__)

ENGINES = {
    "duckduckgo": "https://duckduckgo.com/html/?q={query}",
    "google": "https://www.google.com/search?q={query}",
    "bing": "https://www.bing.com/search?q={query}",
}

# Engine-specific result-link selectors, tried in order.  These
# are plain CSS, resolved through the controller's locator
# system, so no second driver exists anywhere.
_RESULT_SELECTORS = {
    "duckduckgo": ("a.result__a", ".result__a"),
    "google": ("a:has(h3)", "a h3"),
    "bing": ("li.b_algo h2 a", ".b_algo a"),
}
_SNIPPET_SELECTORS = {
    "duckduckgo": (".result__snippet",),
    "google": (".VwiC3b",),
    "bing": (".b_caption p",),
}


def _unwrap(url: str) -> str:
    """Resolve a search-engine redirect to the real destination."""
    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        for key in ("uddg", "q", "u"):
            if params.get(key):
                target = unquote_plus(params[key][0])
                if target.startswith("http"):
                    return target
        if parsed.path.startswith("/l/") and params.get("kh"):
            return url
    except Exception:
        pass
    return url


class WebResearch:
    """Search + result-extraction session on a BrowserController."""

    def __init__(self, controller):
        self.controller = controller
        self.last_query = ""
        self.last_engine = ""
        self.last_results: list[dict[str, Any]] = []

    # -- search -----------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        engine: str = "duckduckgo",
        max_results: int = 10,
        tab_id: str | None = None,
    ) -> dict[str, Any]:
        """Run *query* on *engine* and extract structured results."""
        query = str(query or "").strip()
        if not query:
            raise BrowserException(
                "A search query is required",
                code=BrowserErrorCode.OPERATION_FAILED,
            )
        if engine not in ENGINES:
            raise BrowserException(
                f"Unknown search engine {engine!r}; "
                f"expected one of {sorted(ENGINES)}",
                code=BrowserErrorCode.OPERATION_FAILED,
                details={"engine": engine},
            )
        from urllib.parse import quote_plus
        url = ENGINES[engine].format(query=quote_plus(query))
        page = self.controller.navigate(url, tab_id=tab_id)
        results = self._extract_results(
            engine, max(1, min(int(max_results), 25))
        )
        if not results:
            raise BrowserException(
                f"No search results could be extracted from the "
                f"{engine} results page (the page may be a "
                f"consent/CAPTCHA wall or the search returned "
                f"nothing)",
                code=BrowserErrorCode.OPERATION_FAILED,
                details={"engine": engine, "query": query,
                         "url": page.get("url", "")},
            )
        self.last_query = query
        self.last_engine = engine
        self.last_results = results
        logger.info(
            "browser search %r on %s: %d results",
            query, engine, len(results),
        )
        return {
            "query": query,
            "engine": engine,
            "tab_id": page["tab_id"],
            "url": page["url"],
            "results": results,
            "count": len(results),
            "observation": {
                "type": "search_results",
                "summary": (
                    f"Search {query!r} on {engine}: "
                    f"{len(results)} results; top: "
                    f"{results[0]['title'][:80]!r}"
                ),
                "engine": engine,
                "count": len(results),
            },
        }

    def _extract_results(self, engine: str, max_results: int) -> list[dict]:
        links: list[dict[str, Any]] = []
        for selector in _RESULT_SELECTORS[engine]:
            try:
                links = self.controller.find_elements(
                    {"selector": selector}, limit=max_results * 2
                )
            except BrowserException:
                links = []
            if links:
                break
        snippets: list[str] = []
        for selector in _SNIPPET_SELECTORS[engine]:
            try:
                found = self.controller.find_elements(
                    {"selector": selector}, limit=max_results * 2
                )
                snippets = [str(el.get("text", "")) for el in found]
            except BrowserException:
                snippets = []
            if snippets:
                break
        results: list[dict[str, Any]] = []
        for el in links:
            attrs = el.get("attributes") or {}
            url = _unwrap(str(attrs.get("href", "")))
            title = re.sub(r"\s+", " ", str(el.get("text", ""))).strip()
            if not url.startswith("http") or not title:
                continue
            idx = len(results)
            results.append({
                "rank": idx + 1,
                "title": redact_text(title),
                "url": redact_text(url),
                "snippet": redact_text(
                    snippets[idx] if idx < len(snippets) else ""
                ),
                "source": engine,
            })
            if len(results) >= max_results:
                break
        return results

    # -- open + extract -----------------------------------------------------
    def open_result(
        self,
        index: int,
        *,
        purpose: str | None = None,
        max_chars: int = 4000,
    ) -> dict[str, Any]:
        """Open a stored search result in a new, purpose-tagged
        tab and extract its content."""
        if not self.last_results:
            raise BrowserException(
                "No search results are stored; run a search first",
                code=BrowserErrorCode.OPERATION_FAILED,
            )
        try:
            result = self.last_results[int(index)]
        except (IndexError, ValueError, TypeError):
            raise BrowserException(
                f"No stored search result at index {index!r}; "
                f"{len(self.last_results)} result(s) available",
                code=BrowserErrorCode.ELEMENT_NOT_FOUND,
                details={"index": index,
                         "available": len(self.last_results)},
            ) from None
        tab = self.controller.new_tab(
            result["url"],
            purpose=purpose or f"Search result: {result['title'][:60]}",
        )
        doc = self.controller.page_document(tab_id=tab["tab_id"])
        content = clean_document(doc, max_chars=max_chars)
        return {
            "result": result,
            "tab_id": tab["tab_id"],
            "purpose": tab.get("purpose", ""),
            "content": content,
            "observation": {
                "type": "opened_result",
                "summary": (
                    f"Opened search result {result['title'][:60]!r} "
                    f"({result['url']}) in tab {tab['tab_id']}: "
                    f"{content['counts']['paragraphs']} paragraphs, "
                    f"{content['counts']['tables']} tables, "
                    f"{content['chunk_count']} chunk(s)"
                ),
                "url": result["url"],
                "tab_id": tab["tab_id"],
                "counts": content["counts"],
            },
        }
