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
from afnan_ai.computer.clipboard import ClipboardController
from afnan_ai.computer.controller import ComputerController
from afnan_ai.computer.dialogs import DialogHandler
from afnan_ai.computer.display import DisplayManager
from afnan_ai.computer.errors import ComputerError
from afnan_ai.computer.files import FileService
from afnan_ai.computer.integration import (
    WorkspaceComputerScope,
    attach_activity_center,
    attach_emergency_stop,
    emit_result,
)
from afnan_ai.computer.managers import (
    ApplicationManager,
    InputController,
    WindowManager,
)
from afnan_ai.computer.models import (
    AppInfo,
    ComputerAction,
    ComputerActionResult,
    ComputerElement,
    ComputerObservation,
    DialogInfo,
    MonitorInfo,
    WindowInfo,
)
from afnan_ai.computer.recovery import CrashRecovery
from afnan_ai.computer.runtime import (
    ComputerRuntime,
    ComputerRuntimeError,
    RemoteComputerRuntime,
)
from afnan_ai.computer.screen import ScreenObserver
from afnan_ai.computer.wait import (
    WaitResult,
    wait_for,
    wait_for_stable,
)

__all__ = [
    "AppInfo",
    "ApplicationManager",
    "ClipboardController",
    "ComputerAction",
    "ComputerActionResult",
    "ComputerBackend",
    "ComputerController",
    "ComputerElement",
    "ComputerError",
    "ComputerObservation",
    "ComputerRuntime",
    "ComputerRuntimeError",
    "CrashRecovery",
    "DialogHandler",
    "DialogInfo",
    "DisplayManager",
    "FileService",
    "InputController",
    "MonitorInfo",
    "RemoteComputerRuntime",
    "ScreenObserver",
    "WaitResult",
    "WindowInfo",
    "WindowManager",
    "WorkspaceComputerScope",
    "attach_activity_center",
    "attach_emergency_stop",
    "emit_result",
    "wait_for",
    "wait_for_stable",
]
