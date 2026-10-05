"""Computer Use + file operations as registry Tools.

The Planner picks browser, computer and file tools from the
same ToolRegistry, so one task can research in the browser,
detect the downloaded file, open it in a desktop application
and verify the result — all through the existing
AgentLoop/Executor/Verifier machinery.  Failures come back
as structured ToolResults with the computer error code; the
tools themselves never crash the agent.
"""

from __future__ import annotations

from typing import Any

from afnan_ai.computer.controller import ComputerController
from afnan_ai.computer.errors import ComputerError
from afnan_ai.computer.files import FileService
from afnan_ai.tools.base import Tool, ToolExecutionError


class _ComputerTool(Tool):
    def __init__(self, controller: ComputerController):
        self.controller = controller

    def _call(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return operation(*args, **kwargs)
        except ComputerError as exc:
            raise ToolExecutionError(
                exc.message,
                tool=self.name,
                details={"computer_error": exc.to_dict()},
            ) from exc


def _obj(properties: dict[str, Any], required: list[str] | None = None):
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


class ComputerObserveTool(_ComputerTool):
    name = "computer_observe"
    description = (
        "Observe the desktop: active application/window, all "
        "windows and visible UI elements (accessibility first, "
        "visual fallback). Returns structured state only — "
        "never raw screenshots."
    )
    input_schema = _obj({
        "include_visual": {"type": "boolean"},
    })

    def run(self, arguments):
        return self._call(
            self.controller.observe,
            include_visual=arguments.get("include_visual", True),
        )


class ComputerLocateTool(_ComputerTool):
    name = "computer_locate"
    description = (
        "Find a desktop UI target by description (e.g. 'Save "
        "button', 'Chrome address bar', 'Settings window'). "
        "Low-confidence matches come back without an element_id "
        "and cannot be clicked."
    )
    input_schema = _obj(
        {"description": {"type": "string"}}, ["description"]
    )

    def run(self, arguments):
        return self._call(
            self.controller.locate, arguments["description"]
        )


class ComputerClickTool(_ComputerTool):
    name = "computer_click"
    description = (
        "Click a located desktop element (element_id from "
        "computer_locate/computer_observe). Raw coordinates "
        "need human approval; prefer element_id."
    )
    input_schema = _obj({
        "element_id": {"type": "string"},
        "x": {"type": "integer"},
        "y": {"type": "integer"},
    })

    def run(self, arguments):
        return self._call(
            self.controller.act, "click",
            element_id=arguments.get("element_id"),
            x=arguments.get("x"), y=arguments.get("y"),
        )


class ComputerTypeTool(_ComputerTool):
    name = "computer_type"
    description = (
        "Type text, optionally into a located element first. "
        "Typing into password fields requires human approval "
        "and the text is never echoed back."
    )
    input_schema = _obj(
        {
            "text": {"type": "string"},
            "element_id": {"type": "string"},
        },
        ["text"],
    )

    def run(self, arguments):
        return self._call(
            self.controller.act, "type",
            element_id=arguments.get("element_id"),
            text=arguments.get("text"),
        )


class ComputerKeyPressTool(_ComputerTool):
    name = "computer_key_press"
    description = "Press a single key (e.g. Return, Escape, Tab)."
    input_schema = _obj({"key": {"type": "string"}}, ["key"])

    def run(self, arguments):
        return self._call(
            self.controller.act, "key_press",
            key=arguments.get("key"),
        )


class ComputerHotkeyTool(_ComputerTool):
    name = "computer_hotkey"
    description = (
        "Press a key combination (e.g. ['ctrl', 's']). "
        "Destructive combos require human approval."
    )
    input_schema = _obj(
        {"keys": {"type": "array", "items": {"type": "string"}}},
        ["keys"],
    )

    def run(self, arguments):
        return self._call(
            self.controller.act, "hotkey",
            keys=arguments.get("keys"),
        )


class ComputerScrollTool(_ComputerTool):
    name = "computer_scroll"
    description = "Scroll vertically (dy) or horizontally (dx), optionally over an element."
    input_schema = _obj({
        "dy": {"type": "integer"},
        "dx": {"type": "integer"},
        "element_id": {"type": "string"},
    })

    def run(self, arguments):
        return self._call(
            self.controller.act, "scroll",
            element_id=arguments.get("element_id"),
            dx=int(arguments.get("dx", 0)),
            dy=int(arguments.get("dy", 0)),
        )


class ComputerDragTool(_ComputerTool):
    name = "computer_drag"
    description = "Drag from a located element (or x,y) to (to_x, to_y)."
    input_schema = _obj(
        {
            "element_id": {"type": "string"},
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "to_x": {"type": "integer"},
            "to_y": {"type": "integer"},
        },
        ["to_x", "to_y"],
    )

    def run(self, arguments):
        return self._call(
            self.controller.act, "drag",
            element_id=arguments.get("element_id"),
            x=arguments.get("x"), y=arguments.get("y"),
            to_x=arguments.get("to_x"), to_y=arguments.get("to_y"),
        )


class ComputerMoveMouseTool(_ComputerTool):
    name = "computer_move_mouse"
    description = "Move the mouse cursor to (x, y) without clicking."
    input_schema = _obj(
        {"x": {"type": "integer"}, "y": {"type": "integer"}},
        ["x", "y"],
    )

    def run(self, arguments):
        return self._call(
            self.controller.act, "move_mouse",
            x=arguments.get("x"), y=arguments.get("y"),
        )


class ComputerFocusWindowTool(_ComputerTool):
    name = "computer_focus_window"
    description = "Bring a window (window_id from computer_observe) to the front."
    input_schema = _obj(
        {"window_id": {"type": "string"}}, ["window_id"]
    )

    def run(self, arguments):
        return self._call(
            self.controller.focus_window, arguments["window_id"]
        )


class ComputerOpenApplicationTool(_ComputerTool):
    name = "computer_open_application"
    description = (
        "Launch a desktop application by name and report its "
        "window once it appears."
    )
    input_schema = _obj({"name": {"type": "string"}}, ["name"])

    def run(self, arguments):
        return self._call(
            self.controller.open_application, arguments["name"]
        )


class ComputerCloseApplicationTool(_ComputerTool):
    name = "computer_close_application"
    description = (
        "Close a desktop application. Destructive: requires "
        "human approval."
    )
    input_schema = _obj({"name": {"type": "string"}}, ["name"])

    def run(self, arguments):
        return self._call(
            self.controller.close_application, arguments["name"]
        )


class ComputerListApplicationsTool(_ComputerTool):
    name = "computer_list_applications"
    description = "List discoverable desktop applications."
    input_schema = _obj({})

    def run(self, arguments):
        return self._call(self.controller.list_applications)


class ComputerScreenshotTool(_ComputerTool):
    name = "computer_screenshot"
    description = (
        "Capture a desktop screenshot to a file path (the path "
        "is returned, never the image bytes)."
    )
    input_schema = _obj({"path": {"type": "string"}})

    def run(self, arguments):
        return self._call(
            self.controller.screenshot, arguments.get("path")
        )


class ComputerWaitForChangeTool(_ComputerTool):
    name = "computer_wait_for_ui_change"
    description = "Wait until the desktop UI changes (or timeout)."
    input_schema = _obj({"timeout_s": {"type": "number"}})

    def run(self, arguments):
        return self._call(
            self.controller.wait_for_ui_change,
            timeout_s=float(arguments.get("timeout_s", 5.0)),
        )


class _FileTool(Tool):
    def __init__(self, service: FileService):
        self.service = service

    def _call(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return operation(*args, **kwargs)
        except ComputerError as exc:
            raise ToolExecutionError(
                exc.message,
                tool=self.name,
                details={"computer_error": exc.to_dict()},
            ) from exc


class FileListTool(_FileTool):
    name = "file_list"
    description = "List files/folders in a directory (names, paths, sizes)."
    input_schema = _obj({"path": {"type": "string"}}, ["path"])

    def run(self, arguments):
        return self._call(self.service.list_dir, arguments["path"])


class FileCreateFolderTool(_FileTool):
    name = "file_create_folder"
    description = "Create a folder (parents included)."
    input_schema = _obj({"path": {"type": "string"}}, ["path"])

    def run(self, arguments):
        return self._call(
            self.service.create_folder, arguments["path"]
        )


class FileCopyTool(_FileTool):
    name = "file_copy"
    description = "Copy a file. Overwriting needs human approval."
    input_schema = _obj(
        {
            "source": {"type": "string"},
            "destination": {"type": "string"},
        },
        ["source", "destination"],
    )

    def run(self, arguments):
        return self._call(
            self.service.copy_file,
            arguments["source"], arguments["destination"],
        )


class FileMoveTool(_FileTool):
    name = "file_move"
    description = "Move a file/folder. Destructive: requires human approval."
    input_schema = _obj(
        {
            "source": {"type": "string"},
            "destination": {"type": "string"},
        },
        ["source", "destination"],
    )

    def run(self, arguments):
        return self._call(
            self.service.move_file,
            arguments["source"], arguments["destination"],
        )


class FileRenameTool(_FileTool):
    name = "file_rename"
    description = "Rename a file in place. Requires human approval."
    input_schema = _obj(
        {
            "path": {"type": "string"},
            "new_name": {"type": "string"},
        },
        ["path", "new_name"],
    )

    def run(self, arguments):
        return self._call(
            self.service.rename_file,
            arguments["path"], arguments["new_name"],
        )


class FileOpenTool(_FileTool):
    name = "file_open"
    description = "Open a file with the platform's default application."
    input_schema = _obj({"path": {"type": "string"}}, ["path"])

    def run(self, arguments):
        return self._call(self.service.open_file, arguments["path"])


class FileSaveTextTool(_FileTool):
    name = "file_save_text"
    description = (
        "Save text to a file (export a result). Overwriting "
        "needs human approval; content is never echoed."
    )
    input_schema = _obj(
        {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        ["path", "content"],
    )

    def run(self, arguments):
        return self._call(
            self.service.save_text,
            arguments["path"], arguments["content"],
        )


class FileFindDownloadsTool(_FileTool):
    name = "file_find_downloads"
    description = (
        "Detect recently downloaded files in the downloads "
        "folders (name, path, size, age)."
    )
    input_schema = _obj({"max_age_s": {"type": "number"}})

    def run(self, arguments):
        return self._call(
            self.service.find_downloads,
            max_age_s=float(arguments.get("max_age_s", 3600.0)),
        )


def create_computer_tools(
    controller: ComputerController, service: FileService
) -> list[Tool]:
    computer_tools: list[Tool] = [
        ComputerObserveTool(controller),
        ComputerLocateTool(controller),
        ComputerClickTool(controller),
        ComputerTypeTool(controller),
        ComputerKeyPressTool(controller),
        ComputerHotkeyTool(controller),
        ComputerScrollTool(controller),
        ComputerDragTool(controller),
        ComputerMoveMouseTool(controller),
        ComputerFocusWindowTool(controller),
        ComputerOpenApplicationTool(controller),
        ComputerCloseApplicationTool(controller),
        ComputerListApplicationsTool(controller),
        ComputerScreenshotTool(controller),
        ComputerWaitForChangeTool(controller),
    ]
    file_tools: list[Tool] = [
        FileListTool(service),
        FileCreateFolderTool(service),
        FileCopyTool(service),
        FileMoveTool(service),
        FileRenameTool(service),
        FileOpenTool(service),
        FileSaveTextTool(service),
        FileFindDownloadsTool(service),
    ]
    return computer_tools + file_tools
