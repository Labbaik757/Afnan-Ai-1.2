"""Platform abstraction layer for Afnan AI.

All operating-system specific behaviour lives in this package.
The core agent (:mod:`afnan_ai.agent`) only talks to the
:class:`~afnan_ai.platform.base.PlatformAdapter` interface, so
Windows, macOS and Linux differences are isolated in one adapter
per platform and the correct adapter is selected automatically
at runtime.
"""

from afnan_ai.platform.base import PlatformAdapter
from afnan_ai.platform.factory import get_adapter, detect_system_name

__all__ = ["PlatformAdapter", "get_adapter", "detect_system_name"]
