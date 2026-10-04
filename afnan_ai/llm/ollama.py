"""OllamaProvider — the existing local Ollama integration.

This is the Ollama code that used to live inside the core agent,
moved behind the :class:`~afnan_ai.llm.base.LLMProvider` interface
without changing its behaviour:

* model ``llama3`` by default
* one ``chat`` call with the user's prompt as a single user message
* the reply is ``response["message"]["content"]``

The ``ollama`` Python package is imported lazily, and a client can
be injected (``OllamaProvider(client=...)``), which is how the
tests exercise success, connection-failure and invalid-response
paths without a running Ollama server.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from afnan_ai.llm.base import (
    ChatMessage,
    LLMConnectionError,
    LLMError,
    LLMInvalidResponseError,
    LLMProvider,
    LLMUnavailableError,
)


class OllamaProvider(LLMProvider):
    """Talk to a local Ollama server through the LLMProvider interface."""

    name = "ollama"
    display_name = "Ollama"

    def __init__(
        self,
        model: str = "llama3",
        client: Any = None,
        *,
        host: str | None = None,
    ):
        self.model = model
        self.host = host
        self._client = client
        self._client_resolved = client is not None

    # -- client handling -------------------------------------------------
    def _get_client(self) -> Any:
        """Return the Ollama client, importing the package lazily."""
        if not self._client_resolved:
            self._client_resolved = True
            try:
                import ollama

                self._client = (
                    ollama.Client(host=self.host) if self.host else ollama
                )
            except Exception:
                self._client = None
        if self._client is None:
            raise LLMUnavailableError(
                "Ollama is not installed. Install the 'ollama' package "
                "and pull a model, e.g. `ollama pull llama3`."
            )
        return self._client

    @property
    def is_available(self) -> bool:
        try:
            self._get_client()
        except LLMUnavailableError:
            return False
        return True

    # -- LLMProvider interface --------------------------------------------
    def chat(self, messages: Sequence[ChatMessage]) -> str:
        client = self._get_client()
        try:
            response = client.chat(
                model=self.model,
                messages=[dict(m) for m in messages],
            )
        except LLMError:
            raise
        except Exception as e:
            # Connection refused, server not running, network errors and
            # any other client failure all mean the same thing to the
            # agent: Ollama could not be reached.
            raise LLMConnectionError(
                f"Could not reach Ollama (model={self.model}): {e}"
            ) from e
        return self._extract_content(response)

    # -- response parsing ---------------------------------------------------
    @staticmethod
    def _extract_content(response: Any) -> str:
        """Pull ``message.content`` out of an Ollama response.

        Supports the dict shape the Ollama client returns
        (``{"message": {"content": "..."}}``) and, defensively, an
        object shape with ``.message.content`` attributes.  Anything
        else is an invalid response, not a silent empty reply.
        """
        content: Any = None
        if isinstance(response, Mapping):
            message = response.get("message")
            if isinstance(message, Mapping):
                content = message.get("content")
        else:
            message = getattr(response, "message", None)
            if message is not None:
                content = getattr(message, "content", None)

        if not isinstance(content, str) or not content.strip():
            raise LLMInvalidResponseError(
                "Ollama returned an unexpected response: "
                f"expected message.content text, got {response!r:.200}"
            )
        return content
