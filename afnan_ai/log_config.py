"""Structured logging for Afnan AI (standard library only).

Every component logs through ``get_logger(__name__)`` and stays
silent unless the application configures logging — a library must
not print on its own.  Call :func:`configure_logging` once at
startup (``main.py`` / the voice loop does) to see structured
lines like::

    2026-10-04 23:50:01 INFO    afnan_ai.orchestrator | task 3f2a.. started: goal='Open Chrome'
    2026-10-04 23:50:02 WARNING afnan_ai.recovery | recovery attempt 1 (execution_failed, step step_1): replanned

Logging is observation only: it never changes behaviour, and no
component logs secrets or raw user audio — goals and step ids are
enough to trace a task end to end.
"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Library default: silent until the app configures logging
logging.getLogger("afnan_ai").addHandler(logging.NullHandler())

_configured = False


def configure_logging(
    level: int | str = logging.INFO,
    *,
    stream=None,
) -> logging.Logger:
    """Configure the ``afnan_ai`` logger tree with one structured
    handler.  Safe to call more than once (the handler is
    replaced, not stacked)."""
    global _configured
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger("afnan_ai")
    root.handlers[:] = [handler]
    root.setLevel(level)
    root.propagate = False
    _configured = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Return the logger for a component (``__name__``)."""
    return logging.getLogger(name)


def is_configured() -> bool:
    return _configured
