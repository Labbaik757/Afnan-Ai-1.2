"""Computer Use layer — safe desktop interaction outside the
browser.

The architecture mirrors the browser stack:

    AgentLoop -> Tools -> ComputerController -> ComputerBackend
    (validated actions)   (observe/locate/act)  (OS-specific)

The controller never orchestrates: it only observes the
desktop (windows, active application, accessibility metadata,
visual fallback through ScreenObserver), locates semantic
targets with confidence scores, and executes validated
actions in an Observe -> Validate -> Execute -> Observe
Again -> Verify flow.  Blind coordinate clicking is never the
default strategy: coordinates are used only as a validated
fallback for accessibility/visual targets.

Safety mirrors the browser layer: sensitive actions (closing
applications, typing credentials, destructive hotkeys, file
moves/renames) require human approval through the same
fail-safe pattern — no approver means they do not run.
Secrets never reach logs, AgentState or long-term memory.
"""

from afnan_ai.computer.backend import ComputerBackend
from afnan_ai.computer.controller import ComputerController
from afnan_ai.computer.errors import ComputerError
from afnan_ai.computer.files import FileService
from afnan_ai.computer.models import (
    AppInfo,
    ComputerElement,
    ComputerObservation,
    WindowInfo,
)

__all__ = [
    "AppInfo",
    "ComputerBackend",
    "ComputerController",
    "ComputerElement",
    "ComputerError",
    "ComputerObservation",
    "FileService",
    "WindowInfo",
]
