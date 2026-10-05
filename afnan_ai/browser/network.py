"""Network/request awareness for the BrowserController.

The driver reports the requests a page made (failures, HTTP error
statuses, timeouts, blocked resources).  :func:`summarize_network`
turns those raw events into a small, safe diagnostic summary:

* counts of failed / timed-out / blocked / HTTP-error requests,
* whether the page looks healthy,
* the most recent failures with **redacted** URLs.

This is diagnostics only — there is deliberately no API here (or
anywhere in the browser layer) that fires arbitrary network
requests; the agent can observe what the page did, never craft
its own traffic.
"""

from __future__ import annotations

from typing import Any

from ..redaction import redact_text

__all__ = ["summarize_network"]


def _is_timeout(event: dict[str, Any]) -> bool:
    failure = str(event.get("failure") or "").lower()
    return (
        "timeout" in failure
        or "timed out" in failure
        or "timed_out" in failure
    )


def _is_blocked(event: dict[str, Any]) -> bool:
    failure = str(event.get("failure") or "").lower()
    return (
        "blocked" in failure
        or "err_blocked" in failure
        or event.get("status") in (401, 403)
    )


def summarize_network(
    events: list[dict[str, Any]], *, page_url: str = ""
) -> dict[str, Any]:
    """Summarize raw driver network events for one page."""
    failures: list[dict[str, Any]] = []
    timeouts = 0
    blocked = 0
    http_errors = 0
    rate_limited = 0
    for event in events or []:
        status = event.get("status")
        failure = event.get("failure")
        if status == 429:
            rate_limited += 1
        if not failure and not (
            isinstance(status, int) and status >= 400
        ):
            continue
        if _is_timeout(event):
            timeouts += 1
        if _is_blocked(event):
            blocked += 1
        if isinstance(status, int) and status >= 400:
            http_errors += 1
        failures.append(
            {
                "url": redact_text(str(event.get("url") or "")),
                "method": event.get("method") or "GET",
                "status": status,
                "failure": failure,
                "resource_type": event.get("resource_type") or "",
            }
        )
    return {
        "page_url": redact_text(page_url),
        "total_requests": len(events or []),
        "failed_requests": len(failures),
        "timeouts": timeouts,
        "blocked": blocked,
        "http_errors": http_errors,
        "rate_limited_responses": rate_limited,
        "healthy": not failures,
        "recent_failures": failures[-10:],
    }
