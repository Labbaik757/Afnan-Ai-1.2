"""Afnan Web UI — modern browser-based control center.

Backend lives in :mod:`afnan_ai.webui.server` (stdlib-only HTTP +
SSE); the single-page frontend lives in ``afnan_ai/webui/static/``.
"""

from afnan_ai.webui.server import WebUIServer, create_server

__all__ = ["WebUIServer", "create_server"]
