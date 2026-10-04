"""Language-model providers for Afnan AI.

The agent talks to models only through :class:`LLMProvider`; the
concrete Ollama integration lives in :class:`OllamaProvider`.
"""

from afnan_ai.llm.base import (
    ChatMessage,
    LLMConnectionError,
    LLMError,
    LLMInvalidResponseError,
    LLMProvider,
    LLMUnavailableError,
)
from afnan_ai.llm.factory import (
    DEFAULT_PROVIDER,
    available_providers,
    create_provider,
    get_default_provider,
    register_provider,
)
from afnan_ai.llm.ollama import OllamaProvider

__all__ = [
    "ChatMessage",
    "DEFAULT_PROVIDER",
    "LLMConnectionError",
    "LLMError",
    "LLMInvalidResponseError",
    "LLMProvider",
    "LLMUnavailableError",
    "OllamaProvider",
    "available_providers",
    "create_provider",
    "get_default_provider",
    "register_provider",
]
