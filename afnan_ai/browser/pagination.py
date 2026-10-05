"""Infinite-scroll and pagination collection for long pages.

:class:`Paginator` walks a long list — either by scrolling (infinite
scroll feeds) or by clicking a Next control (classic pagination) — and
collects deduplicated items.  Collection always terminates:

* ``max_items`` caps the number of collected items;
* ``max_pages`` caps the number of page snapshots;
* a snapshot with no new items ends collection (``exhausted``);
* a missing / stalled Next control ends it (``next_unavailable``).

Dynamic content is awaited through the controller's condition-based
``wait_for_stable`` — never a blind fixed sleep.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Paginator"]


class Paginator:
    """Collects items across scrolling or paginated pages."""

    def __init__(self, controller: Any) -> None:
        self._controller = controller

    def collect(
        self,
        *,
        mode: str = "scroll",
        item_selector: str | None = None,
        max_items: int = 100,
        max_pages: int = 10,
        tab_id: str | None = None,
    ) -> dict[str, Any]:
        if mode not in ("scroll", "next"):
            return {
                "success": False,
                "error": (
                    f"Unknown pagination mode {mode!r}; "
                    "use 'scroll' or 'next'."
                ),
                "items": [],
            }
        max_items = max(1, int(max_items))
        max_pages = max(1, int(max_pages))

        seen: dict[str, dict[str, Any]] = {}
        duplicates = 0
        snapshots = 0
        stopped_reason: str | None = None

        while True:
            snapshots += 1
            new_items = 0
            for item in self._snapshot_items(item_selector, tab_id):
                key = item["key"]
                if key in seen:
                    duplicates += 1
                    continue
                seen[key] = item
                new_items += 1
                if len(seen) >= max_items:
                    stopped_reason = "max_items"
                    break

            if stopped_reason:
                break
            if snapshots >= max_pages:
                stopped_reason = "max_pages"
                break
            if snapshots >= 2 and new_items == 0:
                stopped_reason = "exhausted"
                break

            if mode == "scroll":
                try:
                    self._controller.scroll_page(dy=1500, tab_id=tab_id)
                except Exception as exc:
                    return {
                        "success": False,
                        "error": f"Scrolling failed: {exc}",
                        "items": list(seen.values()),
                    }
            else:
                if not self._click_next(tab_id):
                    stopped_reason = "next_unavailable"
                    break
            self._stable(tab_id)

        items = list(seen.values())[:max_items]
        return {
            "success": True,
            "mode": mode,
            "items": items,
            "count": len(items),
            "pages_visited": snapshots,
            "duplicates_skipped": duplicates,
            "stopped_reason": stopped_reason or "exhausted",
            "observation": {
                "type": "pagination",
                "summary": (
                    f"Collected {len(items)} items over {snapshots} "
                    f"page(s); stopped: {stopped_reason or 'exhausted'}."
                ),
                "items_collected": len(items),
            },
        }

    # -- helpers -----------------------------------------------------------

    def _snapshot_items(
        self, item_selector: str | None, tab_id: str | None
    ) -> list[dict[str, Any]]:
        if item_selector:
            elements = self._controller.find_elements(
                {"selector": item_selector},
                tab_id=tab_id,
                limit=500,
            )
        else:
            observation = self._controller.observe(tab_id=tab_id)
            elements = observation.get("elements") or []
        items: list[dict[str, Any]] = []
        for element in elements:
            text = str(element.get("text") or "").strip()
            url = str(
                (element.get("attributes") or {}).get("href") or ""
            )
            if not text and not url:
                continue
            items.append(
                {
                    "key": url or f"{element.get('tag')}:{text}",
                    "text": text,
                    "url": url or None,
                    "tag": element.get("tag"),
                }
            )
        return items

    def _click_next(self, tab_id: str | None) -> bool:
        candidates: list[dict[str, Any]] = []
        for locator in (
            {"text": "Next"},
            {"role": "link", "name": "Next"},
            {"text": "next"},
        ):
            try:
                candidates = self._controller.find_elements(
                    locator, tab_id=tab_id, limit=3
                )
            except Exception:
                candidates = []
            if candidates:
                break
        if not candidates:
            return False
        try:
            self._controller.click(
                {"ref": candidates[0]["ref"]}, tab_id=tab_id
            )
        except Exception:
            return False
        return True

    def _stable(self, tab_id: str | None) -> None:
        """Wait for async content to settle; instability is not fatal."""
        try:
            self._controller.wait_for_stable(
                tab_id=tab_id, timeout_ms=2000
            )
        except Exception:
            pass
