"""Core Afnan AI agent — platform-agnostic and model-agnostic.

This module contains *only* assistant behaviour: wake word, command
routing, web search, music, screenshots and local-AI fallback.  Every
operating-system specific action is delegated to a
:class:`~afnan_ai.platform.base.PlatformAdapter`, selected at runtime
by :func:`afnan_ai.platform.get_adapter`, and every language-model
call goes through the :class:`~afnan_ai.llm.LLMProvider` interface
(default: :class:`~afnan_ai.llm.OllamaProvider`), and every
capability (open URL/app, search, screenshot) runs through the
central :class:`~afnan_ai.tools.ToolRegistry`.  There is
intentionally no OS-specific launching, searching or speech code,
and no concrete model-client call, in this file.
"""

from __future__ import annotations

import os
import webbrowser
from pathlib import Path

from afnan_ai import speech as _speech
from afnan_ai.executor import ExecutionReport, Executor
from afnan_ai.llm import LLMProvider, get_default_provider
from afnan_ai.llm.base import (
    LLMConnectionError,
    LLMInvalidResponseError,
    LLMUnavailableError,
)
from afnan_ai.platform import get_adapter
from afnan_ai.platform.base import PlatformAdapter
from afnan_ai.planner import Planner, TaskPlan
from afnan_ai.state import AgentState
from afnan_ai.tools import ToolRegistry, ToolResult, create_default_registry

try:
    import speech_recognition as sr
except Exception:  # pragma: no cover - optional at import time in tests
    sr = None  # type: ignore

try:
    import pywhatkit
except Exception:
    pywhatkit = None

try:
    import pyautogui
except Exception:
    pyautogui = None


GIF_PATH = "afnan_animation.gif"


class AfnanAgent:
    """Platform-agnostic voice assistant."""

    def __init__(
        self,
        adapter: PlatformAdapter | None = None,
        state: AgentState | None = None,
        *,
        track_state: bool = True,
        llm_provider: LLMProvider | None = None,
        tool_registry: ToolRegistry | None = None,
        planner: Planner | None = None,
        executor: Executor | None = None,
    ):
        self.adapter = adapter or get_adapter()
        self.recognizer = sr.Recognizer() if sr is not None else None
        # Capabilities run through the central ToolRegistry
        # (open_url, open_application, search_google, ...).  Pass a
        # registry to add/replace tools without changing agent code.
        self.tools: ToolRegistry = tool_registry or create_default_registry(
            self.adapter,
            screenshot_capture=self._capture_screenshot,
        )
        # The agent talks to a model only through the LLMProvider
        # interface.  By default that is the local Ollama provider
        # (llama3), exactly as before; pass any other provider
        # (local or cloud) and no agent code changes.
        self.llm: LLMProvider = llm_provider or get_default_provider()
        # Backwards-compatible alias
        self.llm_provider = self.llm
        # Planner uses the same LLM + tools, but only ever plans —
        # planning never executes a tool
        self.planner: Planner = planner or Planner(self.llm, self.tools)
        # Executor runs a TaskPlan's steps through the same registry,
        # recording every result in AgentState
        self.executor: Executor = executor or Executor(self.tools)
        # Centralized, serializable task state.  Components may pass
        # their own AgentState, read ``agent.state``, or ignore it —
        # existing behaviour is unchanged when they do.
        self.track_state = track_state
        self.state: AgentState | None = state

    # -- state helpers ---------------------------------------------------
    def start_task(self, goal: str) -> AgentState:
        """Start tracking a new task and return its AgentState."""
        self.state = AgentState.create(goal)
        self.state.start_task()
        return self.state

    def _state_begin(self, command: str) -> AgentState | None:
        if not self.track_state:
            return None
        if self.state is None or self.state.is_terminal:
            self.state = AgentState.create(command or "voice command")
            self.state.start_task()
        self.state.add_observation(command, source="command")
        self.state.start_step(command or "voice command")
        return self.state

    def _state_succeed(self, result=None) -> None:
        if self.state is not None and self.track_state:
            # complete the running step, but keep the task running so
            # the agent can take the next command in the same session
            if self.state.current_step:
                self.state.complete_step(self.state.current_step, result=result)

    def _state_fail(self, error: str) -> None:
        if self.state is not None and self.track_state:
            if self.state.current_step:
                self.state.fail_step(self.state.current_step, error)
            self.state.add_tool_result("process_command", success=False, error=error)

    # -- tool helpers ------------------------------------------------------
    def execute_tool(self, name: str, arguments=None, **kwargs) -> ToolResult:
        """Execute a registered tool and record it in AgentState.

        Structured failures come back as ``ToolResult(success=False,
        error=ToolError(...))``; this never raises for an unknown
        tool, missing arguments or a tool failure.
        """
        result = self.tools.execute(name, arguments, **kwargs)
        if self.track_state:
            if self.state is None or self.state.is_terminal:
                self.state = AgentState.create(f"Execute tool {name}")
                self.state.start_task()
            self.state.add_tool_result(
                name,
                success=result.success,
                output=result.output if result.success else None,
                error=result.error.message if result.error else None,
            )
        return result

    def get_tool(self, name: str):
        return self.tools.get(name)

    def list_tools(self):
        return self.tools.definitions()

    # -- planning (never executes tools) -------------------------------------
    def create_plan(
        self, goal: str, state: AgentState | None = None
    ) -> TaskPlan:
        """Create a structured TaskPlan for *goal* without executing
        any tool.  Raises PlanningError on invalid/failed planning.

        The plan is recorded in AgentState as an observation (and in
        its metadata) — recording is not execution.
        """
        effective_state = state if state is not None else self.state
        plan = self.planner.plan(goal, state=effective_state)
        if self.track_state and effective_state is not None:
            effective_state.add_observation(
                f"Plan created for goal: {goal} "
                f"({len(plan.steps)} step(s))",
                source="planner",
            )
            effective_state.metadata["last_plan"] = plan.to_dict()
        return plan

    # Alias matching the Planner's naming
    plan_task = create_plan

    # -- execution (through the ToolRegistry, recorded in AgentState) --------
    def execute_plan(
        self, plan: TaskPlan, state: AgentState | None = None
    ) -> ExecutionReport:
        """Execute a TaskPlan's steps, one by one, through the
        ToolRegistry and return an ExecutionReport.

        Every step result is recorded in AgentState; a failed step
        makes the report (and the task) failed — it is never
        silently treated as successful.
        """
        effective_state = state if state is not None else self.state
        if effective_state is None or effective_state.is_terminal:
            effective_state = (
                AgentState.create(plan.goal, task_id=plan.task_id)
                if getattr(plan, "task_id", None)
                else AgentState.create(plan.goal)
            )
        self.state = effective_state
        report = self.executor.execute_plan(plan, state=effective_state)
        self.state = effective_state
        return report

    def plan_and_execute(
        self, goal: str, state: AgentState | None = None
    ) -> ExecutionReport:
        """Plan *goal* with the Planner, then execute the plan with
        the Executor.  Planning errors raise PlanningError; step
        failures come back on the ExecutionReport."""
        plan = self.create_plan(goal, state=state)
        return self.execute_plan(plan, state=state)

    def _open_url(self, url: str) -> bool:
        return self.execute_tool("open_url", {"url": url}).success

    def _open_application(self, application: str) -> bool:
        return self.execute_tool(
            "open_application", {"application": application}
        ).success

    def _capture_screenshot(self):
        # Read the module global at call time so tests/hosts can
        # substitute the capture backend
        if pyautogui is None:
            return None
        return pyautogui.screenshot()

    # -- speech --------------------------------------------------------
    def speak(self, text: str) -> None:
        _speech.speak(text, self.adapter)

    # -- startup animation ---------------------------------------------
    def show_startup_gif(self, gif_path: str = GIF_PATH) -> None:
        try:
            gif_absolute_path = os.path.abspath(gif_path)
            if not os.path.exists(gif_absolute_path):
                print(f"❌ GIF file not found: {gif_absolute_path}")
                print("💡 Continuing without animation...")
                return

            file_url = Path(gif_absolute_path).as_uri()
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Afnan AI</title>
    <style>
        body {{ margin: 0; padding: 0; background: black; display: flex;
               justify-content: center; align-items: center; height: 100vh;
               overflow: hidden; }}
        .afnan-gif {{ max-width: 90vw; max-height: 90vh; }}
    </style>
</head>
<body>
    <div class="afnan-container">
        <img src="{file_url}" alt="Afnan AI Animation" class="afnan-gif">
    </div>
</body>
</html>
            """
            html_file = "afnan_animation.html"
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(html_content)
            webbrowser.open(Path(os.path.abspath(html_file)).as_uri())
            print("✅ Afnan AI animation opened in browser")
        except Exception as e:
            print(f"❌ GIF Error: {e}")
            print("💡 Continuing without animation...")

    # -- introduction ----------------------------------------------------
    def introduce_yourself(self) -> None:
        self.speak(
            """
Hello! I am Afnan.

Created by Afnan.

I am not just a simple assistant — I am smart, fast, and always ready to help.

I can control your system, search anything, play music and write code,
and assist you like a real AI companion.

What do you want me to do?
"""
        )

    # -- folders ---------------------------------------------------------
    def open_folder_anywhere(self, foldername: str) -> None:
        try:
            foldername = (foldername or "").strip()
            if not foldername:
                self.speak("Please tell me the folder name boss")
                return
            path = self.adapter.find_folder(foldername)
            if path:
                self.speak("Opening folder")
                self.adapter.open_path(path)
            else:
                self.speak("Folder not found boss")
        except Exception as e:
            print("Folder Search Error:", e)
            self.speak("Error while opening folder")

    # -- AI (through the LLMProvider interface only) -----------------------
    def ask_ai(self, prompt: str) -> str:
        """Ask the configured LLM provider and return its reply text.

        The agent never calls a concrete model client directly, so
        swapping Ollama for a future local or cloud provider needs
        no change here.  User-facing failure messages are preserved:
        with the default Ollama provider they are the exact strings
        Afnan has always spoken.
        """
        provider = self.llm
        try:
            reply = provider.generate(prompt)
        except LLMUnavailableError as e:
            self._record_llm_failure(provider, str(e))
            return (
                f"Sorry boss, AI is not available. "
                f"{provider.display_name} is not installed."
            )
        except (LLMConnectionError, LLMInvalidResponseError) as e:
            print(f"{provider.display_name} Error:", e)
            self._record_llm_failure(provider, str(e))
            return (
                f"Sorry boss, AI is not responding. "
                f"Make sure {provider.display_name} is running."
            )
        except Exception as e:  # provider broke the interface contract
            print(f"{provider.display_name} Error:", e)
            self._record_llm_failure(provider, str(e))
            return (
                f"Sorry boss, AI is not responding. "
                f"Make sure {provider.display_name} is running."
            )

        if self.state is not None and self.track_state:
            self.state.add_tool_result(
                provider.name, success=True, output=reply
            )
        return reply

    def _record_llm_failure(self, provider: LLMProvider, error: str) -> None:
        if self.state is not None and self.track_state:
            self.state.add_tool_result(
                provider.name, success=False, error=error
            )

    def ask_local_ai(self, prompt: str) -> str:
        """Backwards-compatible name for :meth:`ask_ai`.

        Kept so existing callers (``main.ask_local_ai``, scripts and
        tests) keep working unchanged; with the default provider the
        model is still the local Ollama llama3.
        """
        return self.ask_ai(prompt)

    # -- listening -----------------------------------------------------------
    def listen_command(self, timeout: int = 5, phrase_time: int = 6) -> str:
        if sr is None or self.recognizer is None:
            return ""
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("Listening...")
                audio = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time
                )
            return self.recognizer.recognize_google(audio, language="en-IN")
        except Exception:
            return ""

    # -- music -----------------------------------------------------------------
    def play_song(self, command: str) -> None:
        try:
            song = command.lower().replace("play", "", 1).strip()
            if not song:
                self.speak("Please tell me the song name.")
                return
            self.speak(f"Playing {song} on YouTube")
            if pywhatkit is not None:
                pywhatkit.playonyt(song)
            else:
                self.execute_tool("search_youtube", {"query": song})
        except Exception:
            self.speak("Sorry boss")

    # -- screenshot --------------------------------------------------------------
    def take_screenshot(self) -> str | None:
        result = self.execute_tool("take_screenshot")
        if not result.success:
            # Preserve the original spoken message when capture is
            # unavailable (pyautogui missing / no screen)
            error_text = result.error.message if result.error else ""
            if "not available" in error_text:
                self.speak("Sorry boss, screenshot is not available")
            return None
        return result.output

    # -- command routing (existing behaviour preserved) ---------------------------
    def process_command(self, command: str) -> None:
        raw_command = command or ""
        command = raw_command.lower().strip()
        self._state_begin(command)
        try:
            if "open visual studio code" in command or "open vs code" in command:
                self.speak("Opening Visual Studio Code")
                if not self._open_application("vscode"):
                    self.speak("Visual Studio Code not found boss")

            elif "open safari" in command:
                if not self.adapter.supports_app("safari"):
                    self.speak(
                        "Safari is not available on this system, "
                        "opening your default browser"
                    )
                    self._open_url("https://www.apple.com/safari/")
                else:
                    self.speak("Opening Safari")
                    self._open_application("safari")

            elif "open chrome" in command or "open google chrome" in command:
                self.speak("Opening Chrome")
                if not self._open_application("chrome"):
                    self._open_url("https://www.google.com")

            elif "open edge" in command or "open microsoft edge" in command:
                self.speak("Opening Microsoft Edge")
                if not self._open_application("edge"):
                    self.speak("Microsoft Edge not found boss")

            elif "open youtube" in command:
                self.speak("Opening YouTube")
                self._open_url("https://youtube.com")

            elif "open whatsapp" in command:
                self.speak("Opening WhatsApp")
                if not self._open_application("whatsapp"):
                    self._open_url("https://web.whatsapp.com")

            elif (
                "tell me about yourself" in command
                or "introduce yourself" in command
                or "who are you" in command
            ):
                self.introduce_yourself()

            elif "folder" in command and command.startswith("open"):
                foldername = command.replace("open", "").replace("folder", "").strip()
                self.open_folder_anywhere(foldername)

            elif "search youtube for" in command:
                query = command.replace("search youtube for", "").strip()
                self.execute_tool("search_youtube", {"query": query})

            elif "search google for" in command:
                query = command.replace("search google for", "").strip()
                self.execute_tool("search_google", {"query": query})

            elif command.startswith("play "):
                self.play_song(command)

            elif "screenshot" in command:
                self.take_screenshot()
                self.speak("Screenshot taken")

            elif "stop afnan" in command:
                self.speak("Goodbye boss")
                raise SystemExit

            else:
                self.speak("Thinking boss")
                reply = self.ask_local_ai(command)
                self.speak(reply)

            self._state_succeed(result="ok")

        except SystemExit:
            # "stop afnan" is a successful stop, not a failure
            self._state_succeed(result="stopped")
            raise
        except Exception as e:
            print("Command Error:", e)
            self._state_fail(str(e))
            self.speak("Error boss")

    # -- main loop ------------------------------------------------------------------
    def start(self) -> None:
        self.show_startup_gif()
        self.speak("Afnan is activated")
        while True:
            try:
                word = self.listen_command(timeout=5, phrase_time=3)
                if not word:
                    continue
                if "afnan" in word.lower():
                    self.speak("Yes boss")
                    command = self.listen_command(timeout=7, phrase_time=8)
                    if command:
                        self.process_command(command)
            except SystemExit:
                break
            except KeyboardInterrupt:
                break
            except Exception:
                pass


def create_agent(
    adapter: PlatformAdapter | None = None,
    state: AgentState | None = None,
    llm_provider: LLMProvider | None = None,
    tool_registry: ToolRegistry | None = None,
    planner: Planner | None = None,
    executor: Executor | None = None,
) -> AfnanAgent:
    return AfnanAgent(
        adapter=adapter,
        state=state,
        llm_provider=llm_provider,
        tool_registry=tool_registry,
        planner=planner,
        executor=executor,
    )
