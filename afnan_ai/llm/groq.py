"""Groq cloud provider — big models, fast inference, free tier.

Uses Groq's OpenAI-compatible API with stdlib-only HTTP (no new
dependencies).  Get a free API key at https://console.groq.com and
set ``AFNAN_GROQ_API_KEY``.  Select with ``AFNAN_LLM_PROVIDER=groq``.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Iterator, Sequence

from afnan_ai.llm.base import (
    ChatMessage,
    LLMConnectionError,
    LLMInvalidResponseError,
    LLMProvider,
    LLMUnavailableError,
)

_API_URL = "https://api.groq.com/openai/v1/chat/completions"
_MODELS_URL = "https://api.groq.com/openai/v1/models"
# Preferred chat models, in order.  Resolved against the live
# /models endpoint so retired names never break us.
_PREFERRED_MODELS = (
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
)
_DEFAULT_MODEL = "openai/gpt-oss-20b"  # last-resort fallback
_TIMEOUT = 60


class GroqProvider(LLMProvider):
    """Talk to Groq's hosted models via their OpenAI-compatible API."""

    name = "groq"
    display_name = "Groq"

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        *,
        api_key: str | None = None,
    ) -> None:
        self.model = model or _DEFAULT_MODEL
        self.api_key = (
            api_key or os.environ.get("AFNAN_GROQ_API_KEY", "")
        ).strip()
        self._model_resolved = False

    def _resolve_model(self) -> None:
        """Pick the best available model from the live endpoint.

        Runs once; retired model names fall back gracefully instead
        of 404ing on every call.
        """
        if self._model_resolved:
            return
        self._model_resolved = True
        # If the user pinned a specific model, trust it.
        if os.environ.get("AFNAN_GROQ_MODEL"):
            self.model = os.environ["AFNAN_GROQ_MODEL"].strip()
            return
        try:
            req = urllib.request.Request(
                _MODELS_URL, headers=self._headers(), method="GET")
            with urllib.request.urlopen(req,
                                        timeout=_TIMEOUT) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            available = {
                m.get("id", "") for m in body.get("data", [])
            }
            for candidate in _PREFERRED_MODELS:
                if candidate in available:
                    self.model = candidate
                    return
            # Requested default gone and nothing preferred: keep
            # constructor default; the API will say what's wrong.
        except Exception:
            pass  # offline / blocked: try the default anyway

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise LLMUnavailableError(
                "Groq API key missing — set AFNAN_GROQ_API_KEY "
                "(free at https://console.groq.com)"
            )
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Cloudflare (error 1010) blocks Python-urllib's default
            # UA; a browser-like UA passes the integrity check.
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        }

    def _post(self, payload: dict[str, Any]) -> Any:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            _API_URL, data=data, headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req,
                                        timeout=_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            raise LLMConnectionError(
                f"Groq API error {e.code}: {body}"
            ) from e
        except Exception as e:
            raise LLMConnectionError(
                f"Could not reach Groq: {e}"
            ) from e

    def chat(self, messages: Sequence[ChatMessage]) -> str:
        self._resolve_model()
        body = self._post({
            "model": self.model,
            "messages": [dict(m) for m in messages],
        })
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMInvalidResponseError(
                f"Unexpected Groq response shape: {e}"
            ) from e

    def chat_stream(
        self, messages: Sequence[ChatMessage]
    ) -> Iterator[str]:
        self._resolve_model()
        data = json.dumps({
            "model": self.model,
            "messages": [dict(m) for m in messages],
            "stream": True,
        }).encode("utf-8")
        req = urllib.request.Request(
            _API_URL, data=data, headers=self._headers(),
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=_TIMEOUT)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            raise LLMConnectionError(
                f"Groq API error {e.code}: {body}"
            ) from e
        except Exception as e:
            raise LLMConnectionError(
                f"Could not reach Groq: {e}"
            ) from e
        yielded_any = False
        try:
            with resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        content = chunk["choices"][0]["delta"].get(
                            "content", "")
                    except (KeyError, IndexError, ValueError):
                        continue
                    if content:
                        yielded_any = True
                        yield content
        except LLMConnectionError:
            raise
        except Exception as e:
            raise LLMConnectionError(
                f"Groq stream failed: {e}"
            ) from e
        if not yielded_any:
            raise LLMInvalidResponseError(
                "Groq stream yielded no content")
