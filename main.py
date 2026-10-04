"""Afnan AI — cross-platform voice assistant (entry point).

All platform-specific code lives in :mod:`afnan_ai.platform`
(Windows / macOS / Linux adapters, selected automatically at
runtime).  This file is a thin, backwards-compatible wrapper:
``python main.py`` still works exactly as before, and the old
module-level functions (``speak``, ``process_command``, ...)
are preserved for anyone importing them.

Usage:
    Windows:  python main.py
    macOS/Linux: python3 main.py
"""

from afnan_ai.agent import AfnanAgent, create_agent
from afnan_ai.executor import Executor
from afnan_ai.llm import LLMProvider, OllamaProvider, create_provider, get_default_provider
from afnan_ai.planner import Planner, TaskPlan
from afnan_ai.platform import get_adapter
from afnan_ai.state import AgentState
from afnan_ai.tools import Tool, ToolRegistry, create_default_registry

# Default agent + adapter for this machine (auto-selected at runtime)
adapter = get_adapter()
_agent = AfnanAgent(adapter=adapter)

# Backwards-compatible module state (previous main.py exposed these)
SYSTEM = adapter.name
IS_WINDOWS = adapter.name == "windows"
IS_MACOS = adapter.name == "macos"
IS_LINUX = adapter.name == "linux"
GIF_PATH = "afnan_animation.gif"
recognizer = _agent.recognizer


def speak(text):
    _agent.speak(text)


def show_startup_gif():
    _agent.show_startup_gif()


def introduce_yourself():
    _agent.introduce_yourself()


def open_folder_anywhere(foldername):
    _agent.open_folder_anywhere(foldername)


def open_path(path):
    adapter.open_path(path)


def launch_app(app_key):
    return adapter.launch_app(app_key)


def ask_local_ai(prompt):
    return _agent.ask_local_ai(prompt)


def ask_ai(prompt):
    return _agent.ask_ai(prompt)


def get_llm_provider():
    """Return the LLMProvider the agent talks to (Ollama by default)."""
    return _agent.llm


def get_tool_registry():
    """Return the central ToolRegistry (dynamic register/get/execute)."""
    return _agent.tools


def list_tools():
    return _agent.list_tools()


def execute_tool(name, arguments=None, **kwargs):
    """Execute a registered tool; failures come back as a
    structured ToolResult (success=False, error=ToolError)."""
    return _agent.execute_tool(name, arguments, **kwargs)


def get_planner():
    """Return the Planner (plans only — it never executes tools)."""
    return _agent.planner


def create_plan(goal, state=None):
    """Create a structured TaskPlan for a goal, without executing
    any tool."""
    return _agent.create_plan(goal, state=state)


def get_executor():
    """Return the Executor (runs TaskPlans through the ToolRegistry)."""
    return _agent.executor


def execute_plan(plan, state=None):
    """Execute a TaskPlan step by step through the ToolRegistry;
    every result is recorded in AgentState and a failed step is
    reported as failed, never as successful."""
    return _agent.execute_plan(plan, state=state)


def plan_and_execute(goal, state=None):
    """Plan a goal, then execute the resulting TaskPlan."""
    return _agent.plan_and_execute(goal, state=state)


def listen_command(timeout=5, phrase_time=6):
    return _agent.listen_command(timeout=timeout, phrase_time=phrase_time)


def play_song(command):
    _agent.play_song(command)


def take_screenshot():
    return _agent.take_screenshot()


def process_command(command):
    _agent.process_command(command)


def start_afnan():
    _agent.start()


def get_state():
    """Return the current centralized AgentState (or None)."""
    return _agent.state


def start_task(goal):
    return _agent.start_task(goal)


if __name__ == "__main__":
    try:
        start_afnan()
    except KeyboardInterrupt:
        print("\nAfnan AI stopped by user")
