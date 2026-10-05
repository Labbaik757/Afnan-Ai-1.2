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
from afnan_ai.browser import BrowserController, register_browser_tools
from afnan_ai.browser.runtime import AfnanBrowserRuntime
from afnan_ai.browser.reliability import BrowserReliability
from afnan_ai.browser.security import ApprovalGate
from afnan_ai.browser.workflow import BrowserWorkflow
from afnan_ai.checkpointing import CheckpointManager
from afnan_ai.config import AgentConfig
from afnan_ai.executor import ExecutionReport, Executor
from afnan_ai.llm import LLMProvider, get_default_provider
from afnan_ai.log_config import get_logger
from afnan_ai.llm.base import (
    LLMConnectionError,
    LLMInvalidResponseError,
    LLMUnavailableError,
)
from afnan_ai.orchestrator import Agent as OrchestratorAgent
from afnan_ai.screen import (
    FunctionCaptureSource,
    ScreenObserver,
    register_screen_tools,
)
from afnan_ai.orchestrator import OrchestrationResult, OrchestrationStatus
from afnan_ai.platform import get_adapter
from afnan_ai.platform.base import PlatformAdapter
from afnan_ai.planner import Planner, TaskPlan
from afnan_ai.state import AgentState
from afnan_ai.tools import ToolRegistry, ToolResult, create_default_registry
from afnan_ai.verifier import VerificationReport, VerificationResult, Verifier

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

logger = get_logger(__name__)


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
        verifier: Verifier | None = None,
        orchestrator: OrchestratorAgent | None = None,
        max_iterations: int | None = None,
        max_recovery_attempts: int | None = None,
        config: AgentConfig | None = None,
        browser_controller: BrowserController | None = None,
        enable_browser_tools: bool = True,
        browser_approver=None,
        security_policy=None,
        challenge_handler=None,
        checkpoint_dir=None,
        browser_runtime_dir=None,
        screen_observer: ScreenObserver | None = None,
        enable_screen_tools: bool = True,
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
        # Browser control (Phase 2): a BrowserController bound to
        # browser Tools in the same registry, so the Planner/Agent
        # can launch, navigate and read pages like any other
        # capability.  Constructing it launches nothing; the
        # browser starts only when a browser tool runs.  By
        # default the controller runs on an AfnanBrowserRuntime
        # with a persistent runtime directory, so browser
        # profiles and session state survive restarts.
        if browser_controller is not None:
            self.browser = browser_controller
        else:
            runtime_dir = (
                browser_runtime_dir
                or (config.browser_runtime_dir if config else None)
                or str(
                    Path.home() / ".afnan-ai" / "browser-runtime"
                )
            )
            self.browser = BrowserController(
                runtime=AfnanBrowserRuntime(
                    runtime_dir=runtime_dir
                )
            )
        # Human approval for sensitive browser actions (purchases,
        # sends, destructive clicks, uploads...).  Without an
        # approver, sensitive actions are refused with a
        # structured approval_required error — never auto-allowed.
        if browser_approver is not None or security_policy is not None:
            self.browser.approval_gate = ApprovalGate(
                policy=security_policy, approver=browser_approver
            )
        # Human-in-the-loop CAPTCHA flow: on a human check, the
        # handler is asked; the human solves it in the browser
        # and the action resumes automatically once it clears.
        if challenge_handler is not None:
            self.browser.challenge_handler = challenge_handler
        if enable_browser_tools:
            register_browser_tools(self.tools, self.browser)
        # Screen observation: structured visual information
        # (dimensions, UI elements, regions, confidence scores)
        # from the desktop (the agent's existing capture) or from
        # browser pages (DOM first, pixel detection as fallback).
        # Low-confidence elements are gated to the Verifier/human
        # approval; the observer never clicks anything itself.
        self.screen_observer: ScreenObserver | None = screen_observer
        if self.screen_observer is None and enable_screen_tools:
            self.screen_observer = ScreenObserver(
                FunctionCaptureSource(self._capture_screenshot),
                browser_controller=(
                    self.browser if enable_browser_tools else None
                ),
            )
        if enable_screen_tools and self.screen_observer is not None:
            register_screen_tools(self.tools, self.screen_observer)
        # The perception layer's visual fallback uses the same
        # ScreenObserver as the screen tools (one observer, one
        # set of visual refs).
        if enable_browser_tools and self.screen_observer is not None:
            perception = getattr(self.browser, "perception", None)
            if perception is not None:
                perception.screen_observer = self.screen_observer
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
        # Browser reliability: fresh post-action page observations
        # (confirmation against actual page state) plus structured
        # recovery advice for browser failures.  It only feeds the
        # Verifier's generic provider hooks — browser details stay
        # inside afnan_ai/browser, the Verifier stays generic.
        self.browser_reliability: BrowserReliability | None = (
            BrowserReliability(self.browser) if enable_browser_tools else None
        )
        # Verifier judges executed steps against their expected
        # results — it analyzes results/state, never re-executes.
        # For browser steps it also sees the actual page state.
        self.verifier: Verifier = verifier or Verifier(
            observation_provider=self._verifier_observation_provider,
            failure_advisor=(
                self.browser_reliability.advise
                if self.browser_reliability
                else None
            ),
        )
        # Central orchestration layer: the Agent that connects
        # AgentState + Planner + Executor + Verifier and manages a
        # complete task lifecycle (goal → plan → step-by-step
        # execute/verify → completion), capped by max_iterations.
        # Voice commands keep their existing direct behaviour.
        # Central configuration (wake word, limits, default
        # model...).  Explicit arguments win over the config.
        self.config = config or AgentConfig()
        resolved_iterations = (
            max_iterations
            if max_iterations is not None
            else self.config.max_iterations
        )
        resolved_recovery = (
            max_recovery_attempts
            if max_recovery_attempts is not None
            else self.config.max_recovery_attempts
        )
        self.max_iterations = resolved_iterations
        # Task checkpointing: with a checkpoint directory, task
        # snapshots (redacted state + plan + browser session) are
        # persisted as the task runs, so an interrupted task can
        # resume from its last valid checkpoint (resume_task).
        self.checkpointer: CheckpointManager | None = (
            CheckpointManager(checkpoint_dir)
            if checkpoint_dir
            else None
        )
        self.orchestrator: OrchestratorAgent = orchestrator or OrchestratorAgent(
            planner=self.planner,
            executor=self.executor,
            verifier=self.verifier,
            max_iterations=resolved_iterations,
            max_recovery_attempts=resolved_recovery,
            checkpointer=self.checkpointer,
            checkpoint_extra_provider=self._checkpoint_extra,
        )
        if orchestrator is not None and self.checkpointer is not None:
            orchestrator.checkpointer = self.checkpointer
            orchestrator.checkpoint_extra_provider = (
                self._checkpoint_extra
            )
        # Unified autonomous browser workflow: browser goals run
        # through the same orchestrator with a browser briefing;
        # the workflow plans/executes/judges nothing itself.
        self.browser_workflow: BrowserWorkflow | None = (
            BrowserWorkflow(
                orchestrator=self.orchestrator, controller=self.browser
            )
            if self.browser is not None
            else None
        )
        # Recovery manager (owned by the orchestrator): replans
        # after failed/uncertain steps, attempts recorded in state
        self.recovery = self.orchestrator.recovery
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
        return self.tools.get_or_none(name)

    def _verifier_observation_provider(self, step=None):
        """Route the Verifier's observation hook by step type:
        browser steps get fresh page state, screen steps get a
        fresh screen observation, everything else gets None (the
        step's own output is judged)."""
        tool_name = str(getattr(step, "tool_name", "")) if step else ""
        if tool_name.startswith("browser_") and self.browser_reliability:
            return self.browser_reliability.observe_state(step)
        if tool_name.startswith("screen_") and self.screen_observer:
            return self.screen_observer.verifier_observation(step)
        return None
        return self.tools.get_or_none(name)

    def set_browser_approver(self, approver) -> None:
        """Set the human approver for sensitive browser actions.

        ``approver`` is a callable taking an ApprovalRequest and
        returning True to allow the action.  Setting it to None
        restores the fail-safe default: sensitive browser actions
        are refused until a human approves them.
        """
        self.browser.approval_gate.approver = approver
        return self.tools.get(name)

    def set_challenge_handler(self, handler) -> None:
        """Set the human-check handler (CAPTCHA flow).

        ``handler`` is a callable taking the challenge detection
        dict and returning True when the human will solve the
        check in the browser; the blocked browser action then
        waits and resumes automatically once the check clears.
        The agent never solves a challenge itself.  None restores
        the default: actions pause with human_required.
        """
        self.browser.challenge_handler = handler

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

    # -- verification (analyzes results — never re-executes tools) ------------
    def verify_step(
        self,
        step,
        execution_result=None,
        state: AgentState | None = None,
    ) -> VerificationResult:
        """Verify one executed step against its expected_result
        using the execution result and AgentState evidence."""
        effective_state = state if state is not None else self.state
        return self.verifier.verify_step(
            step, execution_result, state=effective_state
        )

    def verify_plan(
        self,
        plan: TaskPlan,
        execution,
        state: AgentState | None = None,
    ) -> VerificationReport:
        """Verify every step of an executed plan and record the
        judgements in AgentState."""
        effective_state = state if state is not None else self.state
        return self.verifier.verify_plan(
            plan, execution, state=effective_state
        )

    def execute_and_verify(
        self,
        plan: TaskPlan,
        state: AgentState | None = None,
    ) -> tuple[ExecutionReport, VerificationReport]:
        """Execute a plan, then verify each step's actual result
        against its expected_result (verification itself runs no
        tool again)."""
        report = self.execute_plan(plan, state=state)
        verification = self.verify_plan(plan, report, state=state)
        return report, verification

    # -- central orchestration (complete task lifecycle) ----------------------
    def run_task(
        self,
        goal: str,
        state: AgentState | None = None,
        max_iterations: int | None = None,
    ) -> OrchestrationResult:
        """Run a complete orchestrated task for *goal*.

        Lifecycle: state is created/updated, the Planner generates
        a plan, then steps are executed and verified one at a time
        until the task completes, fails, or hits the mandatory
        maximum-iteration limit.  The resulting AgentState becomes
        this agent's current state.
        """
        effective_state = state if state is not None else self.state
        result = self.orchestrator.run(
            goal, state=effective_state, max_iterations=max_iterations
        )
        self.state = result.state
        return result

    # -- checkpointing (crash-safe task persistence) --------------------

    def _checkpoint_extra(self) -> dict:
        """Layer extras for a checkpoint snapshot: the browser
        session (tabs + purposes + redacted history).  Never
        fails the task being checkpointed."""
        try:
            if self.browser is not None:
                return {
                    "browser": self.browser.session_manager.export()
                }
        except Exception:
            pass
        return {}

    def save_checkpoint(self):
        """Persist a checkpoint of the current/last task now."""
        return self.orchestrator.save_checkpoint()

    def resume_task(self, checkpoint_source) -> OrchestrationResult:
        """Resume an interrupted task from a checkpoint.

        Loads and validates the checkpoint (by path or task_id),
        restores the checkpointed browser tabs on a best-effort
        basis, then continues the task: steps already completed
        before the interruption are not executed again.
        """
        from pathlib import Path

        if self.checkpointer is not None:
            data = self.checkpointer.load(checkpoint_source)
        else:
            path = Path(str(checkpoint_source)).expanduser()
            manager = CheckpointManager(
                path.parent if str(path.parent) else Path(".")
            )
            data = manager.load(path)
        browser_data = (data.get("extra") or {}).get("browser") or {}
        if browser_data.get("tabs") and self.browser is not None:
            try:
                self.browser.session_manager.restore_data(browser_data)
            except Exception:
                pass  # best effort; missing pages are reported there
        result = self.orchestrator.run(
            data["goal"], resume_from=data
        )
        self.state = result.state
        return result

    # -- autonomous browser workflow ------------------------------------

    def run_browser_goal(
        self,
        goal: str,
        *,
        profile: str | None = None,
        max_iterations: int | None = None,
        max_duration_s: float | None = None,
        loop: bool = False,
    ):
        """Run one autonomous browser goal end to end.

        The existing orchestrator plans, executes, re-observes,
        verifies and recovers; the BrowserWorkflow adds the
        browser briefing (tabs, current page, profile) before
        acting and composes the final answer from recorded
        evidence (search results, extracted content, verified
        steps, downloads).  ``loop=True`` switches execution to
        the observation-driven loop (small re-decided batches
        instead of one long plan).
        """
        if self.browser_workflow is None:
            raise RuntimeError(
                "Browser tools are disabled for this agent"
            )
        outcome = self.browser_workflow.run_goal(
            goal,
            profile=profile,
            max_iterations=max_iterations,
            max_duration_s=max_duration_s,
            loop=loop,
        )
        if outcome.state is not None:
            self.state = outcome.state
        return outcome

    def get_browser_workflow(self):
        """The BrowserWorkflow (None when browser tools are off)."""
        return self.browser_workflow

    def get_browser_runtime(self):
        """The AfnanBrowserRuntime under the browser controller
        (None when browser tools are off)."""
        if self.browser is None:
            return None
        return self.browser.runtime

    def get_browser_perception(self):
        """The unified BrowserPerception layer (accessibility /
        DOM / visual observation + computer actions), or None
        when browser tools are off."""
        if self.browser is None:
            return None
        return getattr(self.browser, "perception", None)

    # Alias in goal vocabulary
    run_goal = run_task

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
    def show_startup_gif(self, gif_path: str | None = None) -> None:
        gif_path = gif_path or self.config.gif_path
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
            logger.error("folder search failed: %s", e)
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
            logger.error("%s provider failed: %s", provider.display_name, e)
            self._record_llm_failure(provider, str(e))
            return (
                f"Sorry boss, AI is not responding. "
                f"Make sure {provider.display_name} is running."
            )
        except Exception as e:  # provider broke the interface contract
            logger.error("%s provider failed: %s", provider.display_name, e)
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

    # -- user requests (voice-transcribed or typed) -------------------------
    def handle_request(self, request: str) -> OrchestrationResult | None:
        """Handle one natural-language user request.

        This is the single entry point for everything the user
        says after the wake word or types as text: the actual task
        handling is delegated to the central Agent orchestrator
        (``Agent.run()`` — state → plan → execute → verify →
        recover/complete).  No planning, execution, verification
        or recovery logic lives here.

        Preserved behaviour around that delegation:

        * session-control requests ("stop afnan", "introduce
          yourself") are not tasks and are handled directly;
        * if the model cannot plan at all (e.g. it is offline),
          the request falls back to the isolated legacy routing
          below, so every existing command keeps working;
        * the outcome is spoken back through the usual TTS path.

        Returns the OrchestrationResult for orchestrated requests,
        or None when a session/legacy path handled the request.
        ("stop afnan" still raises SystemExit, as before.)
        """
        text = (request or "").strip()
        if not text:
            return None
        lowered = text.lower()

        # 1) Session control — preserved, handled directly
        if (
            "stop afnan" in lowered
            or "tell me about yourself" in lowered
            or "introduce yourself" in lowered
            or "who are you" in lowered
        ):
            self._handle_legacy_command(lowered)
            return None

        # 2) Delegate the actual task to the central Agent.  The
        # orchestrator runs with its own task state; the assistant's
        # session state is only adopted once a plan actually
        # exists, so a planning failure (e.g. model offline) leaves
        # the session untouched for the legacy fallback below.
        result = self.orchestrator.run(text)
        if result.status == OrchestrationStatus.PLANNING_FAILED:
            # The model could not plan (offline, invalid output…).
            # Fall back to the isolated legacy routing so existing
            # behaviour survives; if nothing matches there, use
            # the original conversational AI fallback.
            if self._handle_legacy_command(lowered):
                return None
            self._legacy_chat_fallback(lowered)
            return None

        self.state = result.state
        if result.success:
            self.speak("Done boss")
        else:
            self.speak("Sorry boss, I could not complete that task")
        return result

    def process_command(self, command: str) -> OrchestrationResult | None:
        """Backwards-compatible name for :meth:`handle_request`.

        Existing callers (the voice loop, ``main.process_command``,
        scripts and tests) keep working unchanged; the request is
        handled by the central Agent orchestrator with the legacy
        routing as an isolated fallback.
        """
        return self.handle_request(command)

    # -- LEGACY direct routing (isolated, not yet migrated to tools) --------
    # The command patterns below predate the Tool system.  Some of
    # them use capabilities that do not exist as Tools yet (opening
    # a folder by name, playing a song, the introduction), so they
    # are kept here — clearly isolated — instead of being broken.
    # Each one should move into a Tool over time; until then this
    # handler is used for session control and as the fallback when
    # the model cannot plan.  It deliberately contains no planning,
    # verification or recovery logic.
    @staticmethod
    def _is_legacy_command(command: str) -> bool:
        return (
            "open visual studio code" in command
            or "open vs code" in command
            or "open safari" in command
            or "open chrome" in command
            or "open google chrome" in command
            or "open edge" in command
            or "open microsoft edge" in command
            or "open youtube" in command
            or "open whatsapp" in command
            or "tell me about yourself" in command
            or "introduce yourself" in command
            or "who are you" in command
            or ("folder" in command and command.startswith("open"))
            or "search youtube for" in command
            or "search google for" in command
            or command.startswith("play ")
            or "screenshot" in command
            or "stop afnan" in command
        )

    def _handle_legacy_command(self, command: str) -> bool:
        """Run one legacy direct command.  Returns True when a
        known pattern matched (including the stop command, which
        raises SystemExit as it always has)."""
        if not self._is_legacy_command(command):
            return False
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

            self._state_succeed(result="ok")
            return True

        except SystemExit:
            # "stop afnan" is a successful stop, not a failure
            self._state_succeed(result="stopped")
            raise
        except Exception as e:
            logger.error("command handling failed: %s", e)
            self._state_fail(str(e))
            self.speak("Error boss")
            return True

    def _legacy_chat_fallback(self, command: str) -> None:
        """The original conversational fallback: no known command
        matched, so the request is answered by the LLM provider
        directly (preserved exactly as before)."""
        self._state_begin(command)
        try:
            self.speak("Thinking boss")
            reply = self.ask_local_ai(command)
            self.speak(reply)
            self._state_succeed(result="ok")
        except Exception as e:
            logger.error("command handling failed: %s", e)
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
                if self.config.wake_word in word.lower():
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
    verifier: Verifier | None = None,
    orchestrator: OrchestratorAgent | None = None,
    max_iterations: int | None = None,
    max_recovery_attempts: int | None = None,
    config: AgentConfig | None = None,
    browser_controller: BrowserController | None = None,
    enable_browser_tools: bool = True,
) -> AfnanAgent:
    return AfnanAgent(
        adapter=adapter,
        state=state,
        llm_provider=llm_provider,
        tool_registry=tool_registry,
        planner=planner,
        executor=executor,
        verifier=verifier,
        orchestrator=orchestrator,
        max_iterations=max_iterations,
        max_recovery_attempts=max_recovery_attempts,
        config=config,
        browser_controller=browser_controller,
        enable_browser_tools=enable_browser_tools,
    )
