"""Provider registry — how the agent gets its default model.

Ollama is the default local provider.  Registering another local
or cloud provider here makes it selectable via
:func:`create_provider` without the agent's code changing at all.
"""

from __future__ import annotations

from typing import Any

from afnan_ai.llm.base import LLMProvider, LLMUnavailableError
from afnan_ai.llm.groq import GroqProvider
from afnan_ai.llm.ollama import OllamaProvider

_PROVIDERS: dict[str, type[LLMProvider]] = {
    OllamaProvider.name: OllamaProvider,
    GroqProvider.name: GroqProvider,
}

#: Provider used when nothing is configured — preserves the
#: original behaviour (local Ollama, llama3).
DEFAULT_PROVIDER = OllamaProvider.name


def available_providers() -> list[str]:
    return sorted(_PROVIDERS)


def register_provider(provider_cls: type[LLMProvider]) -> None:
    """Register a new provider class under its ``name``."""
    _PROVIDERS[provider_cls.name] = provider_cls


def create_provider(name: str = DEFAULT_PROVIDER, **kwargs: Any) -> LLMProvider:
    """Create a provider by name, e.g. ``create_provider("ollama")``."""
    try:
        provider_cls = _PROVIDERS[name.lower()]
    except KeyError:
        raise LLMUnavailableError(
            f"Unknown LLM provider {name!r}. "
            f"Available providers: {', '.join(available_providers())}"
        ) from None
    return provider_cls(**kwargs)


def get_default_provider(**kwargs: Any) -> LLMProvider:
    """Return the configured provider.

    ``AFNAN_LLM_PROVIDER`` selects it (``ollama`` default, ``groq``
    for hosted big models).  Extra kwargs (model, host) are passed
    through; providers ignore what they don't need.
    """
    import os

    name = os.environ.get("AFNAN_LLM_PROVIDER", DEFAULT_PROVIDER)
    provider_cls = _PROVIDERS.get(name.lower())
    if provider_cls is None:
        raise LLMUnavailableError(
            f"Unknown LLM provider {name!r}. "
            f"Available providers: {', '.join(available_providers())}"
        )
    # Groq takes model + api_key, not host.
    if provider_cls is GroqProvider:
        kwargs.pop("host", None)
        kwargs["api_key"] = os.environ.get("AFNAN_GROQ_API_KEY", "")
        if not kwargs.get("model") or kwargs["model"] in (
            "llama3", "llama3.2:1b",
        ):
            kwargs["model"] = "llama-3.3-70b-versatile"
    return provider_cls(**kwargs)
