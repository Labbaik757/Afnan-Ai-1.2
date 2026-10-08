"""Afnan AI — Web UI entry point.

Starts the modern browser-based control center (chat, live browser
view, tasks, goals, memory, activity) at http://localhost:5000/.

Usage::

    python afnan_web.py [--port 5000] [--no-browser]

New entry point — reuses AfnanAgent as-is, modifies no existing
module.
"""

from __future__ import annotations

import argparse
import time

from afnan_ai.agent import AfnanAgent
from afnan_ai.config import AgentConfig
from afnan_ai.platform import get_adapter
from afnan_ai.webui.server import WebUIServer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Afnan AI web control center")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-browser", action="store_true",
                        help="don't auto-open the frontend")
    args = parser.parse_args()

    agent = AfnanAgent(adapter=get_adapter(),
                       config=AgentConfig.from_env())
    server = WebUIServer(agent, port=args.port)
    server.start(open_browser=not args.no_browser)
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
