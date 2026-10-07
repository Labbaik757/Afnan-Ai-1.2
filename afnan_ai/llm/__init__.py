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
from afnan_ai.llm.groq import GroqProvider
from afnan_ai.llm.ollama import OllamaProvider
from afnan_ai.llm.router import BrainRouter, classify_intent

__all__ = [
    "BrainRouter",
    "ChatMessage",
    "DEFAULT_PROVIDER",
    "GroqProvider",
    "LLMConnectionError",
    "LLMError",
    "LLMInvalidResponseError",
    "LLMProvider",
    "LLMUnavailableError",
    "OllamaProvider",
    "available_providers",
    "classify_intent",
    "create_provider",
    "get_default_provider",
    "register_provider",
]
