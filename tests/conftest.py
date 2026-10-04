"""Compatibility checks for Afnan AI on Windows, macOS and Linux.

Run with:
    python -m pytest tests/ -q
or, without pytest:
    python -m unittest discover -s tests -v

These tests never actually open apps, speak, or use a microphone:
every OS call is mocked, and ``platform.system`` is simulated, so
the whole suite runs safely on any host OS.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
