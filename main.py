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
from afnan_ai.llm import LLMProvider
from afnan_ai.planner import Planner, TaskPlan
from afnan_ai.platform import get_adapter
from afnan_ai.state import AgentState
from afnan_ai.tools import Tool, ToolRegistry
from afnan_ai.verifier import Verifier

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


def get_verifier():
    """Return the Verifier (judges results — it never runs tools)."""
    return _agent.verifier


def verify_plan(plan, execution, state=None):
    """Verify an executed plan's steps against their expected
    results, using only the execution results and AgentState."""
    return _agent.verify_plan(plan, execution, state=state)


def execute_and_verify(plan, state=None):
    """Execute a TaskPlan, then verify every step's outcome."""
    return _agent.execute_and_verify(plan, state=state)


def get_orchestrator():
    """Return the central Agent orchestrator (Planner + Executor +
    Verifier connected, managing the complete task lifecycle)."""
    return _agent.orchestrator


def run_task(goal, state=None, max_iterations=None):
    """Run a complete orchestrated task: create/update state,
    generate a plan, execute and verify step by step, and finish
    or stop at the maximum-iteration limit."""
    return _agent.run_task(goal, state=state, max_iterations=max_iterations)


def get_recovery():
    """Return the RecoveryManager used by the orchestrator
    (replans after failed/uncertain steps, attempts recorded
    in AgentState)."""
    return _agent.recovery


def get_browser_controller():
    """Return the BrowserController whose operations are exposed
    as browser_* tools in the registry (browser launches only
    when such a tool runs)."""
    return _agent.browser


def get_browser_reliability():
    """Return the BrowserReliability layer (page observations +
    recovery advice feeding the Verifier), or None when browser
    tools are disabled."""
    return _agent.browser_reliability


def get_screen_observer():
    """Return the ScreenObserver (structured visual observations
    of the desktop/browser screen), or None when screen tools
    are disabled."""
    return _agent.screen_observer


def get_web_research():
    """Return the WebResearch session (stored search results for
    browser_search/browser_open_result), or None when browser
    tools are disabled."""
    return getattr(_agent.browser, "research_session", None)


def get_download_manager():
    """Return the browser DownloadManager (download tracking,
    integrity verification, unsafe-payload flags)."""
    return _agent.browser.download_manager


def get_session_manager():
    """Return the browser SessionManager (navigation history and
    save/load/restore of task browsing sessions)."""
    return _agent.browser.session_manager


def get_approval_gate():
    """Return the browser ApprovalGate (sensitive-action policy
    plus recorded approval decisions)."""
    return _agent.browser.approval_gate


def set_browser_approver(approver):
    """Set the human approver for sensitive browser actions
    (a callable ApprovalRequest -> bool; None restores the
    fail-safe refusal)."""
    _agent.set_browser_approver(approver)


def set_challenge_handler(handler):
    """Set the human-check handler for CAPTCHA pages: a callable
    detection -> bool.  True means the human solves the check in
    the browser and the action resumes once it clears."""
    _agent.set_challenge_handler(handler)


def listen_command(timeout=5, phrase_time=6):
    return _agent.listen_command(timeout=timeout, phrase_time=phrase_time)


def play_song(command):
    _agent.play_song(command)


def take_screenshot():
    return _agent.take_screenshot()


def handle_request(request):
    """Handle one natural-language request (voice-transcribed or
    typed).  The actual task handling is delegated to the central
    Agent orchestrator (Agent.run()); this wrapper adds no
    planning/execution/verification/recovery logic of its own."""
    return _agent.handle_request(request)


def process_command(command):
    return _agent.process_command(command)


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
