"""Hybrid Brain Router — three lanes, one interface.

Afnan has two brains: a small local model (Ollama, fast, works
offline, private) and a big cloud model (Groq 120b, smart, needs
internet).  Running *everything* through the cloud is slow and
leaks private data; running everything locally gives the "10-minute
thinking" hangs the 1b model hits on hard planning.

The router picks the right brain per message, implementing the
standard :class:`LLMProvider` interface so the agent's code does not
change — the router *wraps* the two providers:

* **INSTANT lane** → local model: greetings, chitchat, simple Q&A.
  Fast, works offline.
* **GENIUS lane** → cloud model: tasks, planning, browser reasoning,
  deep questions.  Smart, needs internet.
* **PRIVATE lane** → local model, always: personal markers (password,
  bank, CNIC, medical, ...).  Never leaves the PC, even if the cloud
  is faster.

If the cloud does not answer within ``max_cloud_latency_s``, the
router degrades to the local model with an ``[offline mode]`` prefix
rather than hanging the user.
"""

from __future__ import annotations

import concurrent.futures
import re
import threading
from collections.abc import Iterator, Sequence
from typing import Any, Optional

from afnan_ai.llm.base import (
    ChatMessage,
    LLMConnectionError,
    LLMError,
    LLMProvider,
    LLMUnavailableError,
)

#: Warning prefix added when the cloud was unreachable and the
#: local model answered instead.
OFFLINE_PREFIX = "[offline mode] "

# ---------------------------------------------------------------------------
# Intent patterns (English + Roman Urdu + Urdu script)
# ---------------------------------------------------------------------------

# Greetings and small talk — answered instantly by the local model.
_INSTANT_HINTS = (
    # English
    "hello", "hi ", "hey", "good morning", "good evening", "good night",
    "how are you", "how r u", "thanks", "thank you", "thankyou",
    "welcome", "bye", "goodbye", "see you", "nice to meet",
    "what's up", "whats up", "sup ",
    # Roman Urdu
    "assalam", "salam", "aoa", "walaikum", "w salam",
    "kya haal", "kia haal", "kesay ho", "kese ho", "kaisay ho",
    "kaise ho", "kesi ho", "kaisi ho", "kya hal", "kia hal",
    "shukriya", "khuda hafiz", "alvida", "theek ho", "theek hun",
    "aur sunao", "kya ho raha", "kia ho raha",
    # Urdu script
    "سلام", "ہیلو", "وعلیکم", "کیا حال", "کیسے ہو", "کیسی ہو",
    "شکریہ", "خدا حافظ", "الوداع", "ٹھیک ہو",
)

# Short factual Q&A the local model can handle ("time kya hai" style).
# These are sentence-start patterns for simple questions.
_SIMPLE_Q_HINTS = (
    "time kya", "time kia", "waqt kya", "date kya", "aaj kya",
    "what time", "what is the time", "what date", "what day",
    "tumhara naam", "tumhara name", "your name",
    "tum kaun", "tum kon", "who are you",
)

# Action verbs: the user wants something DONE → genius lane.
# (Extends AfnanAgent._TASK_HINTS with planning/deep verbs.)
_GENIUS_HINTS = (
    # English actions
    "open", "launch", "start", "close", "search", "find", "look up",
    "play", "download", "send", "type", "click", "press", "screenshot",
    "volume", "mute", "shutdown", "restart", "lock", "delete",
    "create", "make", "fill", "signup", "sign up", "register",
    "buy", "book", "order", "pay", "remind", "schedule",
    # English deep-thinking verbs
    "explain", "compare", "analyze", "analyse", "summarize",
    "summarise", "translate", "write a", "write an", "compose",
    "plan", "strategy", "pros and cons", "difference between",
    "how do i", "how to ", "step by step", "in detail",
    # Roman Urdu actions
    "kholo", "khol", "band karo", "talash", "dhoondo", "dhundo",
    "chalao", "bajao", "bhejo", "likho", "dabao",
    "bnao", "banao", "bnawo", "bharo", "yaad",
    "kharido", "mangwao", "likhwao",
    # Roman Urdu deep verbs
    "samjhao", "samjha", "wazahat", "tafseel", "muqabla",
    "faraq", "farq", "kyun", "kaise", "kesay",
    # Urdu script actions
    "کھولو", "کھول", "بند", "تلاش", "ڈھونڈ", "چلاؤ",
    "بجاؤ", "بھیجو", "لکھو", "دباؤ", "بناؤ", "بناو", "بھرو",
    # Urdu script deep verbs
    "سمجھاؤ", "وضاحت", "تفصیل", "موازنہ", "فرق", "کیوں", "کیسے",
)

# Personal markers: these MUST stay on the PC.  Checked first —
# a private marker beats a task hint ("open my bank account").
_PRIVATE_HINTS = (
    "password", "passwd", "pin code", "otp",
    "bank", "account number", "iban", "credit card", "debit card",
    "cnic", "شناختی کارڈ", "passport",
    "medical", "doctor", "disease", "bimari", "بیماری", "hospital",
    "private", "secret", "confidential", "personal",
    "پرائیویٹ", "خفیہ", "ذاتی",
    "salary", "tankhah", "تنخواہ",
)

# Question words that, combined with length, suggest a deep question
# needing the big model ("why is the sky blue in detail...").
_DEEP_QUESTION_WORDS = (
    "why", "how come", "kyun", "کیوں",
)
_DEEP_QUESTION_MIN_LEN = 60


def classify_intent(text: str) -> str:
    """Classify *text* into a brain lane.

    Returns ``"instant"``, ``"genius"`` or ``"private"``.  Robust to
    English, Roman Urdu and Urdu script.  Private markers always win.
    """
    lowered = text.lower().strip()
    if not lowered:
        return "instant"

    # PRIVATE first — personal data never leaves the PC.
    if any(h in lowered for h in _PRIVATE_HINTS):
        return "private"

    # GENIUS — actions, planning, deep questions.
    if any(h in lowered for h in _GENIUS_HINTS):
        return "genius"
    if (
        len(lowered) >= _DEEP_QUESTION_MIN_LEN
        and any(w in lowered for w in _DEEP_QUESTION_WORDS)
    ):
        return "genius"
    # Long multi-sentence input usually needs real reasoning.
    if len(lowered) > 200 and lowered.count(" ") > 25:
        return "genius"

    # INSTANT — greetings, simple Q&A, everything short and simple.
    return "instant"


def _is_simple_question(lowered: str) -> bool:
    """True for short factual Q&A the local model can answer."""
    return any(lowered.startswith(h) for h in _SIMPLE_Q_HINTS)


class BrainRouter(LLMProvider):
    """Route each message to the right brain: local or cloud.

    Implements :class:`LLMProvider`, wrapping a local provider
    (Ollama) and a cloud provider (Groq).  The agent uses the router
    exactly like any other provider::

        router = BrainRouter(local, cloud)
        router.chat(messages)

    ``router.last_lane`` records which lane handled the last call
    (``"instant"`` / ``"genius"`` / ``"private"``), useful for the UI.
    """

    name: str = "router"
    display_name: str = "Afnan Brain"
    model: str = "hybrid"

    def __init__(
        self,
        local_provider: LLMProvider,
        cloud_provider: LLMProvider,
        max_cloud_latency_s: float = 30.0,
    ) -> None:
        self.local = local_provider
        self.cloud = cloud_provider
        self.max_cloud_latency_s = max_cloud_latency_s
        self.last_lane: str = "instant"
        self._lock = threading.Lock()

    # -- routing --------------------------------------------------------

    @staticmethod
    def classify(text: str) -> str:
        """Classify *text* into ``"instant"``/``"genius"``/``"private"``."""
        return classify_intent(text)

    def route(self, text: str, context_hint: Optional[str] = None) -> LLMProvider:
        """Return the provider that should handle *text*.

        *context_hint* may force a lane (``"instant"``, ``"genius"``
        or ``"private"``); otherwise the text is classified.
        """
        lane = context_hint if context_hint in (
            "instant", "genius", "private") else classify_intent(text)
        with self._lock:
            self.last_lane = lane
        if lane == "genius":
            return self.cloud
        # "instant" and "private" both stay on the PC.
        if self.local is None:
            if lane == "private":
                raise LLMUnavailableError(
                    "No local provider: private content cannot be "
                    "sent to the cloud.")
            return self.cloud
        return self.local

    # -- provider interface -----------------------------------------------

    def _last_user_text(self, messages: Sequence[ChatMessage]) -> str:
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                return content if isinstance(content, str) else str(content)
        return ""

    def chat(self, messages: Sequence[ChatMessage]) -> str:
        """Route on the last user message and chat with that brain.

        Cloud calls are bounded by ``max_cloud_latency_s``; on
        timeout or connection failure the local model answers with
        an ``[offline mode]`` prefix instead of hanging.
        """
        text = self._last_user_text(messages)
        provider = self.route(text)
        lane = self.last_lane

        if provider is self.cloud:
            try:
                return self._chat_cloud_with_timeout(messages)
            except (LLMError, concurrent.futures.TimeoutError,
                    TimeoutError, Exception) as exc:
                # Degrade gracefully — never hang the user.
                if self.local is None:
                    raise LLMConnectionError(
                        f"Cloud brain failed and no local fallback: {exc}"
                    ) from exc
                with self._lock:
                    self.last_lane = "instant"
                try:
                    local_reply = self.local.chat(messages)
                except LLMError:
                    raise
                except Exception as exc2:
                    raise LLMConnectionError(
                        f"Both brains failed (cloud: {exc}; "
                        f"local: {exc2})"
                    ) from exc2
                return OFFLINE_PREFIX + local_reply

        # Local lanes (instant / private): direct call, no timeout
        # wrapper — the local model is the fallback itself.
        try:
            return provider.chat(messages)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMConnectionError(
                f"Local brain failed on {lane} lane: {exc}") from exc

    def _chat_cloud_with_timeout(
        self, messages: Sequence[ChatMessage]
    ) -> str:
        """Run the cloud chat bounded by ``max_cloud_latency_s``."""
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="brain-cloud"
        ) as pool:
            future = pool.submit(self.cloud.chat, messages)
            return future.result(timeout=self.max_cloud_latency_s)

    def chat_stream(
        self, messages: Sequence[ChatMessage]
    ) -> Iterator[str]:
        """Stream from the routed brain; fall back on cloud failure."""
        text = self._last_user_text(messages)
        provider = self.route(text)
        try:
            yield from provider.chat_stream(messages)
        except (LLMError, Exception) as exc:
            if provider is self.cloud and self.local is not None:
                with self._lock:
                    self.last_lane = "instant"
                yield OFFLINE_PREFIX
                yield from self.local.chat_stream(messages)
            else:
                raise LLMConnectionError(
                    f"Brain stream failed: {exc}") from exc

    def generate(self, prompt: str) -> str:
        return self.chat([{"role": "user", "content": prompt}])

    @property
    def is_available(self) -> bool:
        return (self.local is not None and self.local.is_available) or (
            self.cloud is not None and self.cloud.is_available
        )

    def status(self) -> dict[str, Any]:
        """Lane + provider health, for the UI status card."""
        return {
            "router": self.name,
            "last_lane": self.last_lane,
            "local": {
                "name": getattr(self.local, "name", "?"),
                "model": getattr(self.local, "model", "?"),
                "available": self.local.is_available
                if self.local else False,
            },
            "cloud": {
                "name": getattr(self.cloud, "name", "?"),
                "model": getattr(self.cloud, "model", "?"),
                "available": self.cloud.is_available
                if self.cloud else False,
            },
            "max_cloud_latency_s": self.max_cloud_latency_s,
        }
