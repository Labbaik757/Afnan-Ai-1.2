"""CAPTCHA / human-check detection for the BrowserController.

Detection only — this module never solves, bypasses or works around a
CAPTCHA.  When a human check is detected, browser actions stop with a
structured ``human_required`` error so the task pauses for human
intervention instead of failing blindly or looping.

Signals are read from the controller's structured observation (URL,
title, visible text and interactive elements): challenge URLs, known
challenge phrases, and reCAPTCHA/hCaptcha markers on elements.  No page
content beyond what ``observe()`` already returns is used, and nothing
here touches credentials, cookies or tokens.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ChallengeDetector", "detect_challenge"]

#: URL fragments that strongly suggest a human-check page.
_URL_SIGNALS = (
    "/captcha",
    "captcha",
    "challenge-platform",
    "human-verification",
    "verify-you-are-human",
    "bot-check",
)

#: Phrases shown by common human-check pages.
_TEXT_SIGNALS = (
    "verify you are human",
    "i'm not a robot",
    "im not a robot",
    "are you a robot",
    "human verification",
    "checking your browser",
    "prove you're human",
    "prove you are human",
    "security check",
    "bot detection",
    "captcha",
)

#: Element attribute fragments identifying captcha widgets.
_ELEMENT_SIGNALS = ("recaptcha", "hcaptcha", "captcha", "turnstile")


class ChallengeDetector:
    """Detects human-check pages from a page observation."""

    def detect(self, observation: dict[str, Any]) -> dict[str, Any]:
        url = str(observation.get("url") or "")
        title = str(observation.get("title") or "")
        text = str(observation.get("text") or "")
        haystack = f"{title}\n{text}".lower()
        signals: list[str] = []

        lowered_url = url.lower()
        for fragment in _URL_SIGNALS:
            if fragment in lowered_url:
                signals.append(f"url:{fragment}")

        kind = "generic_human_check"
        for phrase in _TEXT_SIGNALS:
            if phrase in haystack:
                signals.append(f"text:{phrase}")

        for element in observation.get("elements") or []:
            attrs = {
                str(k).lower(): str(v).lower()
                for k, v in (element.get("attributes") or {}).items()
            }
            blob = " ".join(
                [str(element.get("text") or "").lower()]
                + list(attrs.values())
            )
            for marker in _ELEMENT_SIGNALS:
                if marker in blob:
                    signals.append(f"element:{marker}")
                    if marker in ("recaptcha", "hcaptcha", "turnstile"):
                        kind = marker

        if "recaptcha" in " ".join(signals):
            kind = "recaptcha"
        elif "hcaptcha" in " ".join(signals):
            kind = "hcaptcha"

        detected = bool(signals)
        strong = any(
            s.startswith(("url:", "text:")) for s in signals
        )
        return {
            "detected": detected,
            "type": kind if detected else None,
            "confidence": 0.9 if strong else (0.7 if detected else 0.0),
            "signals": sorted(set(signals)),
            "human_required": detected,
        }


def detect_challenge(
    controller: Any, tab_id: str | None = None
) -> dict[str, Any]:
    """Detect a human check on the active (or given) tab's page."""
    observation = controller.observe(tab_id=tab_id)
    result = ChallengeDetector().detect(observation)
    result["url"] = observation.get("url")
    result["tab_id"] = observation.get("tab_id")
    return result
