"""LLMProvider interface — the only way the agent talks to a model.

The core agent depends on this interface, never on a concrete model
client (Ollama, a cloud API, ...).  Adding a new local or cloud
model means writing one new provider class; the agent's code does
not change.

Provider contract
-----------------
* ``chat(messages)`` takes a list of chat messages, each a mapping
  with ``role`` and ``content`` (the OpenAI/Ollama style), and
  returns the assistant's reply as a plain ``str``.
* ``generate(prompt)`` is a convenience wrapper for a single user
  prompt — providers get it for free from ``chat``.
* Failures are reported with the typed errors below, never with a
  provider-specific exception leaking through:

  - :class:`LLMUnavailableError` — the provider's client/runtime is
    not installed or not configured at all.
  - :class:`LLMConnectionError` — the client is installed but the
    model service could not be reached (not running, network, ...).
  - :class:`LLMInvalidResponseError` — the service answered, but the
    answer did not have the expected shape.

* Providers must be safe to construct when their backend is missing:
  construction never raises for a missing client; the error surfaces
  on the first ``chat``/``generate`` call as
  :class:`LLMUnavailableError`.  This keeps ``python main.py``
  starting normally on machines without a model installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any


class LLMError(Exception):
    """Base class for every error raised through the LLMProvider interface."""


class LLMUnavailableError(LLMError):
    """The provider's client or runtime is not installed/configured."""


class LLMConnectionError(LLMError):
    """The model service could not be reached."""


class LLMInvalidResponseError(LLMError):
    """The model service answered in an unexpected shape."""


# A chat message, e.g. {"role": "user", "content": "hello"}
ChatMessage = Mapping[str, Any]


class LLMProvider(ABC):
    """Interface every language-model provider implements."""

    #: short machine name, e.g. "ollama"
    name: str = "llm"
    #: human name used in user-facing messages, e.g. "Ollama"
    display_name: str = "AI"
    #: model identifier the provider will call, e.g. "llama3"
    model: str = ""

    @abstractmethod
    def chat(self, messages: Sequence[ChatMessage]) -> str:
        """Send chat *messages* and return the assistant reply text.

        Raises:
            LLMUnavailableError: client/runtime not installed.
            LLMConnectionError: model service unreachable.
            LLMInvalidResponseError: unexpected response shape.
        """

    def generate(self, prompt: str) -> str:
        """Single-prompt convenience wrapper around :meth:`chat`."""
        return self.chat([{"role": "user", "content": prompt}])

    @property
    def is_available(self) -> bool:
        """Whether the provider's client/runtime appears usable.

        The default assumes a constructed provider is usable;
        providers with an optional client override this.  It must
        not call the model service.
        """
        return True

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r}, model={self.model!r})"
