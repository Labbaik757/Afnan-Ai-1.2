"""Central configuration for Afnan AI.

One small, explicit place for the tunables that used to be
scattered as literals: the wake word, the orchestration limits
and the default model.  Components keep their own defaults (e.g.
``Agent.DEFAULT_MAX_ITERATIONS`` mirrors ``max_iterations`` here);
this dataclass is what the *application* passes in when it wants
to change them, optionally from environment variables.

Environment overrides (``AgentConfig.from_env()``)::

    AFNAN_WAKE_WORD, AFNAN_MAX_ITERATIONS,
    AFNAN_MAX_RECOVERY_ATTEMPTS, AFNAN_LLM_PROVIDER,
    AFNAN_LLM_MODEL, AFNAN_GIF_PATH
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


@dataclass(frozen=True)
class AgentConfig:
    """Runtime configuration for the assistant + Agent pipeline."""

    wake_word: str = "afnan"
    max_iterations: int = 10
    max_recovery_attempts: int = 2
    llm_provider: str = "ollama"
    llm_model: str = "llama3"
    gif_path: str = "afnan_animation.gif"

    @classmethod
    def from_env(cls, prefix: str = "AFNAN_") -> "AgentConfig":
        """Build a config from environment variables (falling back
        to the defaults for anything unset or invalid)."""
        defaults = cls()
        return cls(
            wake_word=os.environ.get(
                f"{prefix}WAKE_WORD", defaults.wake_word
            ),
            max_iterations=_int_from_env(
                f"{prefix}MAX_ITERATIONS", defaults.max_iterations
            ),
            max_recovery_attempts=_int_from_env(
                f"{prefix}MAX_RECOVERY_ATTEMPTS",
                defaults.max_recovery_attempts,
            ),
            llm_provider=os.environ.get(
                f"{prefix}LLM_PROVIDER", defaults.llm_provider
            ),
            llm_model=os.environ.get(
                f"{prefix}LLM_MODEL", defaults.llm_model
            ),
            gif_path=os.environ.get(
                f"{prefix}GIF_PATH", defaults.gif_path
            ),
        )
