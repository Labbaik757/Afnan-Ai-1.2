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
import re
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


# Sentence terminators for streaming speech: Latin . ! ? plus the
# Urdu full stop U+06D4 (۔).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?۔])\s+")


def _split_complete_sentences(buffer: str) -> tuple[list[str], str]:
    """Split finished sentences off the front of a stream buffer.

    Returns (complete_sentences, remainder): every sentence ending
    in a terminator followed by whitespace is complete; the tail
    (possibly an unfinished sentence) stays in the buffer.
    """
    parts = _SENTENCE_SPLIT_RE.split(buffer)
    complete = [p.strip() for p in parts[:-1]]
    complete = [p for p in complete if p]
    return complete, parts[-1]


# Invisible characters that speech-to-text engines sprinkle into
# transcripts — Google's Urdu STT in particular inserts zero-width
# joiners between words. They are invisible on the console, so a
# heard command can *look* like a known phrase while failing every
# substring match and silently falling through to the slow LLM
# planning path. Normalize them (plus stray whitespace) in every
# heard/typed command before matching.
#
# U+200B/U+200C/U+200D act as word separators in STT output, so they
# become plain spaces; U+FEFF/U+00AD are removed outright.
_INVISIBLE_SEPARATOR_RE = re.compile("[\u200b\u200c\u200d]")
_INVISIBLE_REMOVE_RE = re.compile("[\ufeff\u00ad]")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_command_text(text: str) -> str:
    """Normalize STT output for command matching.

    Turns invisible word separators into spaces and collapses
    whitespace, so "براؤزر‌اوپن‌کرو" (zero-width joiners, as Urdu
    STT often returns) matches the same legacy patterns as
    "براؤزر اوپن کرو".
    """
    text = _INVISIBLE_SEPARATOR_RE.sub(" ", text or "")
    text = _INVISIBLE_REMOVE_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


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
        memory_dir=None,
        screen_observer: ScreenObserver | None = None,
        enable_screen_tools: bool = True,
        computer_backend=None,
        computer_approver=None,
        enable_computer_tools: bool = True,
        downloads_dir=None,
        connector_registry=None,
        connector_approver=None,
        enable_connector_tools: bool = True,
        connector_credentials=None,
        context_budget=None,
        skill_registry=None,
        skill_approver=None,
        enable_skill_tools: bool = True,
        artifact_workspace_dir=None,
        artifact_approver=None,
        enable_artifact_tools: bool = True,
        proactive_config=None,
        proactive_approver=None,
        enable_proactive: bool = True,
        security_approver=None,
        enable_security_center: bool = True,
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
        # Browser control: a BrowserController bound to
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
        # Computer Use: validated desktop observation/actions
        # (mouse, keyboard, windows, applications) plus safe
        # file operations, through the same ToolRegistry.
        # Sensitive desktop/file actions need a human approver
        # (fail-safe: without one they do not run).
        self.computer = None
        self.file_service = None
        if enable_computer_tools:
            from afnan_ai.computer.command_backend import (
                CommandComputerBackend,
            )
            from afnan_ai.computer.controller import (
                ComputerController,
            )
            from afnan_ai.computer.files import FileService
            from afnan_ai.computer.policy import ComputerApprovalGate
            from afnan_ai.computer.tools import create_computer_tools

            backend = computer_backend or CommandComputerBackend()
            gate = ComputerApprovalGate(computer_approver)
            self.computer = ComputerController(
                backend,
                screen_observer=self.screen_observer,
                gate=gate,
            )
            self.file_service = FileService(
                gate=gate,
                open_path=self.adapter.open_path,
                downloads_dirs=(
                    [downloads_dir] if downloads_dir else None
                ),
            )
            self.tools.register_many(
                create_computer_tools(self.computer, self.file_service)
            )
        # The perception layer's visual fallback uses the same
        # ScreenObserver as the screen tools (one observer, one
        # set of visual refs).
        if enable_browser_tools and self.screen_observer is not None:
            perception = getattr(self.browser, "perception", None)
            if perception is not None:
                perception.screen_observer = self.screen_observer
        # Central configuration (wake word, limits, default
        # model...).  Explicit arguments win over the config.
        # Resolved early: the LLM provider below needs the model/host.
        self.config = config or AgentConfig()
        # The agent talks to a model only through the LLMProvider
        # interface.  By default that is the local Ollama provider
        # (llama3), exactly as before; pass any other provider
        # (local or cloud) and no agent code changes.
        # Model and host come from the config so AFNAN_LLM_MODEL /
        # AFNAN_LLM_HOST can switch them without code changes
        # (e.g. a remote Ollama with a GPU, like Google Colab).
        self.llm: LLMProvider = llm_provider or get_default_provider(
            model=self.config.llm_model,
            host=self.config.llm_host,
        )
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
        # (self.config was resolved earlier, before the LLM provider.)
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
        # Persistent memory / goals / task queue: long-term
        # state that survives tasks (verified facts, user
        # preferences, managed goals, long-running tasks).
        # Files are created lazily under the memory directory;
        # memory content is safety-gated (no secrets, trusted
        # sources only) inside the stores themselves.
        from pathlib import Path as _Path

        from afnan_ai.goal_manager import GoalManager
        from afnan_ai.memory_store import LocalMemoryStore
        from afnan_ai.scheduler import TaskScheduler
        from afnan_ai.task_manager import TaskManager

        memory_base = (
            _Path(str(memory_dir)).expanduser()
            if memory_dir
            else _Path.home() / ".afnan-ai"
        )
        self.memory_store = LocalMemoryStore(
            str(memory_base / "memory.json")
        )
        self.goal_manager = GoalManager(
            str(memory_base / "goals.json")
        )
        self.task_manager = TaskManager(
            str(memory_base / "tasks.json")
        )
        self.scheduler = TaskScheduler(
            str(memory_base / "schedules.json"), self.task_manager
        )
        self._task_worker = None
        self._background_runner = None
        # Deferred: the SecurityCenter is built near the
        # end of __init__ (after every tool family is
        # registered); consumers that accept it earlier
        # receive None and get wired when it exists.
        self.security_center = None
        # Long-context & trajectory reasoning: the per-task
        # TrajectoryStore persists every run's trajectory
        # (observations, decisions, actions, verifications,
        # recoveries, approvals, checkpoints) for restart
        # recovery; each AgentLoop run gets a fresh
        # ContextManager that keeps the unified working
        # context inside a configurable budget.
        from afnan_ai.context.trajectory import TrajectoryStore

        self._context_budget = context_budget
        self.trajectory_store = TrajectoryStore(
            str(memory_base / "trajectories.json")
        )
        # Connector System: external services (email, calendar,
        # cloud storage, chat, ...) through a registry of
        # self-contained Connector implementations.  The agent
        # knows only the generic interface; new services are
        # added by registering a Connector, never by changing
        # core code.  Secrets travel only through the credential
        # store (provide_credentials), never through tools.
        self.connector_service = None
        self.connector_registry = None
        if enable_connector_tools:
            from afnan_ai.connectors import (
                ConnectorApprovalGate,
                ConnectorRegistry,
                ConnectorService,
                MemoryCredentialStore,
                create_connector_tools,
            )

            self.connector_registry = (
                connector_registry or ConnectorRegistry()
            )
            # Best-effort entry-point discovery: installed
            # distributions can contribute connectors without
            # any agent change.
            self.connector_registry.discover()
            gate = ConnectorApprovalGate(connector_approver)
            self.connector_service = ConnectorService(
                self.connector_registry,
                credential_store=(
                    connector_credentials or MemoryCredentialStore()
                ),
                gate=gate,
                audit_path=str(memory_base / "connector_audit.jsonl"),
            )
            self.tools.register_many(
                create_connector_tools(self.connector_service)
            )
        # Dynamic Tool & Skill Builder: reusable workflows
        # composed of *existing* registered tools.  A Skill is
        # never a Tool subclass; the registry bridges active
        # skills into the ToolRegistry one-way as skill_<id>
        # adapters, so Planner/Executor/Verifier/Recovery and
        # the AgentLoop handle them with zero special cases —
        # skills get no separate orchestration loop.
        from afnan_ai.skills import (
            SkillExecutor,
            SkillGenerator,
            SkillLearner,
            SkillRegistry,
        )

        self.skill_registry: SkillRegistry = (
            skill_registry
            or SkillRegistry(
                tool_registry=self.tools,
                audit_path=str(
                    memory_base / "skill_audit.jsonl"
                ),
            )
        )
        # Keep the registry pointed at the live tool registry
        # even when a pre-built one was injected.
        self.skill_registry._tool_registry = self.tools
        self.skill_learner = SkillLearner()
        self.skill_generator = SkillGenerator(
            self.tools, self.skill_registry
        )
        self.skill_executor = SkillExecutor(
            self.skill_registry,
            self.tools,
            approver=skill_approver,
            audit_path=str(memory_base / "skill_audit.jsonl"),
        )
        if enable_skill_tools:
            self.register_skill_tools()
        # Multi-agent / subagent architecture: complex goals
        # divide into specialized, least-privilege subagents.
        # Each subagent runs the *existing* AgentLoop against
        # a scoped tool view — no new orchestration layer.
        # Permissions can never exceed the parent's; results
        # return as structured handoffs the parent verifies.
        from afnan_ai.skills.models import SkillRisk
        from afnan_ai.subagents import (
            SubAgentManager,
            SubAgentSecurity,
        )

        parent_connector_ids: set[str] = set()
        if self.connector_registry is not None:
            try:
                parent_connector_ids = {
                    c.connector_id
                    for c in self.connector_registry.list()
                }
            except Exception:
                parent_connector_ids = set()
        self.subagent_security = SubAgentSecurity(
            parent_tool_names=set(self.tools.names()),
            parent_connector_ids=parent_connector_ids,
            parent_max_risk=SkillRisk.DESTRUCTIVE,
        )
        self.subagent_manager = SubAgentManager(
            tool_registry=self.tools,
            loop_factory=self._build_subagent_loop,
            security=self.subagent_security,
            audit_path=str(memory_base / "subagent_audit.jsonl"),
            security_center=self.security_center,
        )
        # Artifact System: research/task results become real,
        # versioned, verified deliverables.  The manager is a
        # controlled interface — the agent builds artifacts
        # through data-driven builders, never by executing
        # arbitrary code against the filesystem.
        from afnan_ai.artifacts import (
            ArtifactManager,
            ArtifactWorkspace,
            create_artifact_tools,
        )

        self.artifact_workspace = ArtifactWorkspace(
            artifact_workspace_dir
            or str(memory_base / "artifacts")
        )
        self.artifact_manager = ArtifactManager(
            self.artifact_workspace,
            approver=artifact_approver,
            audit_path=str(memory_base / "artifact_audit.jsonl"),
        )
        if enable_artifact_tools:
            self.tools.register_many(
                create_artifact_tools(self.artifact_manager)
            )
        # Proactive intelligence: a controlled layer that
        # surfaces evidence-based ideas from authorized
        # state.  It never executes by itself — accepted
        # ideas become normal tasks for the existing loop.
        from afnan_ai.proactive import (
            ProactiveConfig,
            ProactiveEngine,
        )

        self.proactive_engine: ProactiveEngine | None = None
        if enable_proactive:
            self.proactive_engine = ProactiveEngine(
                memory_store=self.memory_store,
                goal_manager=self.goal_manager,
                task_manager=self.task_manager,
                scheduler=self.scheduler,
                config=proactive_config or ProactiveConfig(),
                store_path=str(
                    memory_base / "proactive.json"
                ),
                approver=proactive_approver,
                task_runner=self._run_proactive_task,
            )
        # Security Center: built after every tool family is
        # registered so capability grants cover them all.
        # From here on, every tool call on this registry is
        # authorized through the center.
        self.security_center = (
            self._build_security_center(
                security_approver, memory_base
            )
            if enable_security_center
            else None
        )
        if self.security_center is not None:
            # Managers constructed before the center existed.
            self.subagent_manager.set_security_center(
                self.security_center
            )
            if self.connector_service is not None:
                self.connector_service.set_security_center(
                    self.security_center
                )
            # Browser runtime: emergency stop closes pages
            # and refuses new engine calls until reset.
            runtime = getattr(
                self.browser, "runtime", None
            )
            if runtime is not None and hasattr(
                runtime, "set_security_center"
            ):
                runtime.set_security_center(
                    self.security_center
                )
            # Computer controller: emergency stop refuses
            # new desktop actions until reset.
            computer = getattr(self, "computer", None)
            if computer is not None and hasattr(
                computer, "set_security_center"
            ):
                computer.set_security_center(
                    self.security_center
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
        # Real-time autonomous loop (observe → decide → validate
        # → act → observe → verify); built lazily, wired to the
        # browser perception layer for fresh observations.
        self._agent_loop = None
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
        if tool_name.startswith("computer_") and self.computer:
            return self.computer.verifier_observation(step)
        if tool_name.startswith("connector_") and self.connector_service:
            return self.connector_service.verifier_observation(step)
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

    # -- real-time autonomous loop --------------------------------
    def _loop_observation(self, state=None):
        """Fresh unified browser observation for the AgentLoop
        (None when no browser page is available)."""
        perception = self.get_browser_perception()
        if perception is None:
            return None
        try:
            observed = perception.observe()
        except Exception:
            return None
        return {
            "kind": observed.get("kind"),
            "url": observed.get("url", ""),
            "title": observed.get("title", ""),
            "text": observed.get("text", ""),
            "tab_id": observed.get("tab_id", ""),
            "elements": [
                {
                    "role": e.get("role", ""),
                    "accessible_name": e.get("accessible_name", ""),
                    "text": e.get("text", ""),
                }
                for e in observed.get("elements") or []
            ],
        }

    def get_agent_loop(self):
        """The real-time AgentLoop driving this agent's
        Planner/Executor/Verifier/Recovery with fresh browser
        observations, long-term memory and active goals."""
        if self._agent_loop is None:
            self._agent_loop = self._build_agent_loop()
        return self._agent_loop

    # -- multi-agent / subagents ------------------------------------
    def get_subagent_manager(self):
        """The SubAgentManager: create and supervise
        least-privilege subagents for complex goals."""
        return self.subagent_manager

    # -- artifact system ------------------------------------------
    def get_artifact_manager(self):
        """The ArtifactManager: versioned, verified
        deliverables (documents, reports, data, ...)."""
        return self.artifact_manager

    def set_artifact_approver(self, approver) -> None:
        """Set the human approver for destructive artifact
        operations (None restores fail-safe refusal)."""
        self.artifact_manager.set_approver(approver)

    # -- proactive intelligence ---------------------------------------
    def get_proactive_engine(self):
        """The ProactiveEngine: evidence-based ideas from
        authorized state (None when disabled)."""
        return self.proactive_engine

    def set_proactive_approver(self, approver) -> None:
        """Set the human approver for sensitive proactive
        actions (None restores fail-safe refusal)."""
        if self.proactive_engine is not None:
            self.proactive_engine.set_approver(approver)

    def run_proactive_sweep(self):
        """One background-style sweep: detect, expire and
        queue ideas.  Never executes actions by itself."""
        if self.proactive_engine is None:
            return []
        return self.proactive_engine.run_sweep()

    # -- security center ------------------------------------------
    def get_security_center(self):
        """The central SecurityCenter: permissions, risk
        classification, approvals, vault, audit (None when
        disabled)."""
        return self.security_center

    def set_security_approver(self, approver) -> None:
        """Set the human approver for sensitive/irreversible
        actions (None restores fail-safe refusal)."""
        if self.security_center is not None:
            self.security_center.set_approver(approver)

    def _run_proactive_task(self, task_id: str):
        """Auto-execution path for explicitly configured
        read-only low-risk ideas: claim exactly this task
        and run it through the normal managed-task runner,
        recording the outcome like the background worker."""
        from datetime import datetime, timezone

        def _iso():
            return datetime.now(timezone.utc).isoformat()

        task = self.task_manager.get(task_id)
        if task is None:
            raise ValueError(f"unknown task {task_id!r}")
        if task.status != "pending":
            raise ValueError(
                f"task {task_id!r} is {task.status}, "
                "not pending"
            )
        task.attempts += 1
        task.started_at = _iso()
        task.status = "running"
        task.updated_at = _iso()
        self.task_manager._save()
        try:
            result = self._run_managed_task(task, None)
        except Exception as exc:  # noqa: BLE001
            self.task_manager.fail(
                task_id, f"Runner error: {exc}", retry=False
            )
            raise
        status = getattr(result, "status", None)
        status_value = getattr(status, "value", status)
        if status_value == "completed":
            return self.task_manager.complete(
                task_id, f"Proactive task {task_id} completed"
            )
        error = getattr(result, "error", None) or {}
        self.task_manager.fail(
            task_id,
            str(error.get("message") or "task failed"),
            retry=False,
        )
        return self.task_manager.get(task_id)

    # -- security center --------------------------------------------
    def _build_security_center(self, approver, memory_base):
        """Central, mandatory security authority.  Every
        tool call on this agent's registry is authorized
        through it."""
        from afnan_ai.security import (
            PolicyProfile,
            SecurityCenter,
        )

        center = SecurityCenter(
            audit_path=str(memory_base / "security_audit.jsonl"),
            approver=approver,
        )
        # Explicit owner profile (default Standard): the
        # profile grants the agent exactly the capabilities
        # its tool families need — no unrestricted defaults.
        # Legacy domain wildcards below keep dynamically
        # registered tools working; capability ids cover
        # known tools with resource-level policy.
        profile = getattr(
            self, "_security_profile", PolicyProfile.STANDARD
        )
        center.apply_profile(profile)
        domains = set()
        for name in self.tools.names():
            domain, _, _ = str(name).partition("_")
            if domain:
                domains.add(f"{domain}.*")
        # Built-in/tool-less capabilities the loop needs.
        domains.update({"memory.*", "note.*"})
        center.grant_agent_capabilities(*sorted(domains))
        self.tools.security_center = center
        return center

    def apply_security_profile(
        self, profile: "PolicyProfile | str"
    ) -> dict[str, Any]:
        """Owner-controlled profile switch: Restricted /
        Standard / Advanced / Fully Authorized.  Publishes a
        new policy version; background snapshots are
        revalidated on next claim."""
        if self.security_center is None:
            raise RuntimeError(
                "security center is disabled"
            )
        return self.security_center.apply_profile(
            profile,
            changelog="owner profile change",
        )

    def _build_subagent_loop(
        self, spec, scoped_tools, context_text, checkpointer=None
    ):
        """Build a scoped AgentLoop for one subagent.

        Reuses the existing Planner/Executor/Verifier/
        RecoveryManager/AgentLoop machinery against the
        subagent's restricted tool view, with its own
        isolated AgentState.  Subagents share the parent's
        LLM provider and read-only browser observation, but
        never the parent's AgentState, credentials, or
        approval authority.
        """
        from afnan_ai.agent_loop import AgentLoop
        from afnan_ai.executor import Executor
        from afnan_ai.orchestrator import Agent as OrchestratorAgent
        from afnan_ai.planner import Planner
        from afnan_ai.recovery import RecoveryManager
        from afnan_ai.verifier import Verifier

        planner = Planner(self.llm, scoped_tools)
        executor = Executor(scoped_tools)
        verifier = Verifier(
            observation_provider=self._verifier_observation_provider,
        )
        orchestrator = OrchestratorAgent(
            planner=planner,
            executor=executor,
            verifier=verifier,
            max_iterations=min(
                10, max(3, spec.limits.max_steps)
            ),
            max_recovery_attempts=1,
            checkpointer=(
                checkpointer._inner
                if checkpointer is not None
                and getattr(checkpointer, "_inner", None)
                is not None
                else None
            ),
        )
        # The capturing wrapper still records snapshots for
        # approval-pause resume; give the orchestrator the
        # wrapper when there is no inner checkpointer.
        if (
            orchestrator.checkpointer is None
            and checkpointer is not None
        ):
            orchestrator.checkpointer = checkpointer

        def _scoped_context():
            return context_text

        return AgentLoop(
            orchestrator,
            observation_provider=self._loop_observation,
            memory_store=self.memory_store,
            system_context_provider=_scoped_context,
            security_center=self.security_center,
        )

    def _build_agent_loop(self):
        """A fresh AgentLoop (background tasks each get their
        own so concurrent runs stay isolated)."""
        from afnan_ai.agent_loop import AgentLoop
        from afnan_ai.context.manager import ContextManager

        budget = self._context_budget

        def _context_factory():
            return ContextManager(budget=budget)

        def _system_context():
            # Capability-agnostic: connectors + skills each
            # contribute a short section; the loop itself
            # knows nothing about either.
            parts = []
            if self.connector_service is not None:
                try:
                    section = (
                        self.connector_service.context_section()
                    )
                except Exception:
                    section = None
                if section:
                    parts.append(section)
            try:
                skill_section = (
                    self.skill_registry.context_section()
                )
            except Exception:
                skill_section = ""
            if skill_section:
                parts.append(skill_section)
            return "\n".join(parts) or None

        return AgentLoop(
            self.orchestrator,
            observation_provider=self._loop_observation,
            memory_store=self.memory_store,
            goal_manager=self.goal_manager,
            system_context_provider=_system_context,
            context_manager_factory=_context_factory,
            trajectory_store=self.trajectory_store,
            skill_learner=self.skill_learner,
            security_center=self.security_center,
        )

    def run_agent_loop(self, goal, *, state=None, resume_from=None,
                       control=None, on_event=None, goal_id=None,
                       **limit_kwargs):
        """Run *goal* through the real-time autonomous loop:
        observe → decide (small batch) → validate → execute →
        fresh observation → verify → continue/replan/complete.
        Extra keyword arguments map to LoopLimits fields
        (max_steps, batch_limit, max_replans, max_llm_calls,
        max_duration_s, max_browser_actions...)."""
        from afnan_ai.agent_loop import LoopLimits

        limits = LoopLimits(**{
            k: v for k, v in limit_kwargs.items() if v is not None
        }) if limit_kwargs else None
        return self.get_agent_loop().run(
            goal,
            state=state,
            limits=limits,
            resume_from=resume_from,
            control=control,
            on_event=on_event,
            goal_id=goal_id,
        )

    # -- computer use -------------------------------------------------
    def get_computer_controller(self):
        """The ComputerController for validated desktop
        observation and actions (None when disabled)."""
        return self.computer

    def get_file_service(self):
        """The safe file-operations service (destructive
        operations need human approval)."""
        return self.file_service

    def set_computer_approver(self, approver) -> None:
        """Set the human approver for sensitive desktop/file
        actions (None restores the fail-safe refusal)."""
        if self.computer is not None:
            self.computer.gate.set_approver(approver)

    # -- connector system ---------------------------------------
    def get_connector_registry(self):
        """The ConnectorRegistry (None when connector tools are
        disabled).  Register new service integrations here —
        no core agent change needed."""
        return self.connector_registry

    def get_connector_service(self):
        """The ConnectorService: credentials, scopes, approval,
        execution, audit (None when disabled)."""
        return self.connector_service

    def set_connector_approver(self, approver) -> None:
        """Set the human approver for sensitive/irreversible
        connector operations (None restores the fail-safe
        refusal: they do not run without a human yes)."""
        if self.connector_service is not None:
            self.connector_service.set_approver(approver)

    # -- long-context & trajectory reasoning ------------------
    def get_trajectory_store(self):
        """The persistent per-task TrajectoryStore (restart
        recovery for execution trajectories)."""
        return self.trajectory_store

    # -- dynamic tool & skill builder -------------------------
    def get_skill_registry(self):
        """The versioned SkillRegistry (register / discover /
        update / disable skills)."""
        return self.skill_registry

    def get_skill_learner(self):
        """The SkillLearner (verified workflows → candidates)."""
        return self.skill_learner

    def get_skill_generator(self):
        """The SkillGenerator (capability search → draft →
        validated skill)."""
        return self.skill_generator

    def set_skill_approver(self, approver) -> None:
        """Set the human approver for sensitive/destructive
        skill execution (None restores fail-safe refusal)."""
        self.skill_executor.set_approver(approver)

    def register_skill_tools(self) -> int:
        """Expose active skills as skill_<id> tools in the
        agent's ToolRegistry (call again after registering or
        updating skills)."""
        return self.skill_registry.register_skill_tools(
            self.tools, executor=self.skill_executor
        )

    # -- persistent memory / goals / tasks --------------------------
    def get_memory_store(self):
        """The persistent MemoryStore (verified facts, user
        preferences, task summaries — secrets are refused)."""
        return self.memory_store

    def get_goal_manager(self):
        """The persistent GoalManager (long-lived goals with
        milestones, dependencies and verified progress)."""
        return self.goal_manager

    def get_task_manager(self):
        """The persistent TaskManager (long-running task
        queue)."""
        return self.task_manager

    def _run_managed_task(self, task, resume_from):
        """Managed-task runner: a fresh AgentLoop per task
        (isolation), with the task's timeout/step budget."""
        from afnan_ai.agent_loop import LoopLimits

        limit_kwargs: dict = {}
        if task.timeout_s is not None:
            limit_kwargs["max_duration_s"] = task.timeout_s
        if task.metadata.get("max_steps"):
            limit_kwargs["max_steps"] = int(
                task.metadata["max_steps"]
            )
        limits = LoopLimits(**limit_kwargs) if limit_kwargs else None
        return self._build_agent_loop().run(
            task.goal_text,
            limits=limits,
            resume_from=resume_from,
            goal_id=task.goal_id or None,
        )

    def get_task_worker(self):
        """The explicitly-invoked TaskWorker over the task
        queue (runs tasks through the AgentLoop; starts no
        background execution by itself)."""
        if self._task_worker is None:
            from afnan_ai.task_manager import TaskWorker

            self._task_worker = TaskWorker(
                self.task_manager,
                self._run_managed_task,
                checkpointer=self.checkpointer,
            )
        return self._task_worker

    # -- scheduling + background execution --------------------------
    def get_scheduler(self):
        """The persistent TaskScheduler (one-time + recurring
        tasks; fires into the TaskManager queue)."""
        return self.scheduler

    def get_background_runner(self):
        """The BackgroundTaskRunner over the task queue +
        scheduler.  Nothing runs until start_background_runner
        is called explicitly."""
        if self._background_runner is None:
            from afnan_ai.background_runner import (
                AuditLog,
                BackgroundTaskRunner,
            )

            audit_path = (
                self.memory_store._store.path.parent
                / "audit.jsonl"
            )
            self._background_runner = BackgroundTaskRunner(
                self.task_manager,
                self._run_managed_task,
                scheduler=self.scheduler,
                checkpointer=self.checkpointer,
                audit_log=AuditLog(str(audit_path)),
                security_center=self.security_center,
            )
        return self._background_runner

    def start_background_runner(self):
        """Start background execution (explicit opt-in).

        Crashed tasks recover from their checkpoints; approval
        gates, limits and retries all stay enforced."""
        runner = self.get_background_runner()
        runner.start()
        return runner

    def stop_background_runner(self):
        """Stop background execution cleanly."""
        if self._background_runner is not None:
            self._background_runner.stop()

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
        _speech.speak(
            text,
            self.adapter,
            urdu_voice=self.config.tts_urdu_voice,
        )

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
            # Standalone app-like window via the platform adapter
            # (Chrome --app mode on Windows, default browser else).
            self.adapter.open_app_window(
                Path(os.path.abspath(html_file)).as_uri()
            )
            print("✅ Afnan AI animation opened in app window")
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
    def _llm_failure_message(self, exc: Exception) -> str:
        """User-facing message for an LLM failure.

        Preserves the exact strings Afnan has always spoken; shared
        by the blocking :meth:`ask_ai` path and the streaming path.
        """
        provider = self.llm
        self._record_llm_failure(provider, str(exc))
        if isinstance(exc, LLMUnavailableError):
            return (
                f"Sorry boss, AI is not available. "
                f"{provider.display_name} is not installed."
            )
        logger.error(
            "%s provider failed: %s", provider.display_name, exc
        )
        return (
            f"Sorry boss, AI is not responding. "
            f"Make sure {provider.display_name} is running."
        )

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
        except Exception as e:  # includes provider contract breaks
            return self._llm_failure_message(e)

        if self.state is not None and self.track_state:
            self.state.add_tool_result(
                provider.name, success=True, output=reply
            )
        return reply

    def _speak_streaming(self, prompt: str) -> None:
        """Speak the LLM reply sentence-by-sentence as it generates.

        Chunks stream from the provider; every finished sentence is
        spoken immediately while the rest is still generating, so
        the user hears the answer start much sooner than waiting
        for the full reply.
        """
        provider = self.llm
        messages = [{"role": "user", "content": prompt}]
        try:
            stream = provider.chat_stream(messages)
            buffer = ""
            spoke_any = False
            for chunk in stream:
                buffer += chunk
                complete, buffer = _split_complete_sentences(
                    buffer
                )
                for sentence in complete:
                    spoke_any = True
                    self.speak(sentence)
        except Exception as e:
            self.speak(self._llm_failure_message(e))
            return
        remainder = buffer.strip()
        if remainder:
            self.speak(remainder)
        elif not spoke_any:
            # Defensive: streamed nothing and raised nothing.
            self.speak(
                self._llm_failure_message(
                    LLMInvalidResponseError("empty reply")
                )
            )

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
        except Exception as e:
            # Silence within the timeout is normal; a real mic
            # failure (busy/missing device) is printed so it is
            # never a silent infinite "Listening..." loop.
            is_timeout = sr is not None and isinstance(
                e, sr.WaitTimeoutError
            )
            if not is_timeout:
                print(
                    "microphone issue "
                    f"({type(e).__name__}): mic busy or unavailable"
                )
            return ""
        try:
            command = self.recognizer.recognize_google(
                audio, language=self.config.stt_language
            )
        except Exception:
            return ""
        if command:
            print(f"heard: {command}")
            # Let the audio driver fully release the mic before
            # any speech: on some systems speaking too soon
            # after closing the input stream produces silence.
            import time

            time.sleep(0.5)
        return command

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
        text = _normalize_command_text(request)
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

        # 2) Fast path (config): known direct commands and chitchat
        # skip LLM planning entirely — instant on slow hardware
        # where planning a trivial command takes minutes.
        if self.config.fast_path:
            if self._handle_legacy_command(lowered):
                return None
            if self._is_chitchat(lowered):
                self._legacy_chat_fallback(lowered)
                return None

        # 3) Free conversation vs. task: if the user is just talking
        # (no action words), answer directly with the LLM — fast,
        # no planning.  The user can say anything they want; only
        # task-like requests go to the orchestrator.
        if not self._looks_like_task(lowered):
            self._legacy_chat_fallback(lowered)
            return None

        # 4) Delegate the actual task to the central Agent.  The
        # orchestrator runs with its own task state; the assistant's
        # session state is only adopted once a plan actually
        # exists, so a planning failure (e.g. model offline) leaves
        # the session untouched for the legacy fallback below.
        #
        # This path is dark on the console and slow on weak
        # hardware, so say so explicitly instead of leaving the
        # user staring at a dead "heard:" line.
        print(
            "task detected — asking the local AI "
            "(slow on this PC, please wait)..."
        )
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
            or "open browser" in command
            or "browser kholo" in command
            or "browser open karo" in command
            or "براؤزر کھولو" in command
            or "براؤزر اوپن کرو" in command
            or "براوزر کھولو" in command
            or "open youtube" in command
            or "youtube kholo" in command
            or "یوٹیوب کھولو" in command
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

    @staticmethod
    def _is_chitchat(command: str) -> bool:
        """True for greetings/small talk: answered directly by the
        LLM with no planning overhead."""
        return any(p in command for p in AfnanAgent._CHAT_PATTERNS)

    # Words that signal the user wants something DONE (a task for
    # the orchestrator) rather than just talked about.  Anything
    # else is free conversation and goes straight to the LLM —
    # fast, no planning.  The user can say anything they want.
    _TASK_HINTS = (
        # English actions
        "open", "launch", "start", "close", "search", "find",
        "play", "download", "send", "type", "click", "press",
        "screenshot", "volume", "mute", "shutdown", "restart",
        "lock", "delete", "create", "make",
        # Urdu actions (Latin + script)
        "kholo", "khol", "band karo", "talash", "dhoondo",
        "chalao", "bajao", "bhejo", "likho", "dabao",
        "کھولو", "کھول", "بند", "تلاش", "ڈھونڈ", "چلاؤ",
        "بجاؤ", "بھیجو", "لکھو", "دباؤ",
    )

    @staticmethod
    def _looks_like_task(lowered: str) -> bool:
        """True when the input asks for an action, not conversation."""
        return any(h in lowered for h in AfnanAgent._TASK_HINTS)

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

            elif (
                "open browser" in command
                or "browser kholo" in command
                or "browser open karo" in command
                or "براؤزر کھولو" in command
                or "براؤزر اوپن کرو" in command
                or "براوزر کھولو" in command
            ):
                self.speak("Opening browser")
                self._open_url("https://www.google.com")

            elif (
                "youtube kholo" in command
                or "یوٹیوب کھولو" in command
            ):
                self.speak("Opening YouTube")
                self._open_url("https://youtube.com")

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
            if self.config.stream_responses:
                self._speak_streaming(command)
            else:
                reply = self.ask_local_ai(command)
                self.speak(reply)
            self._state_succeed(result="ok")
        except Exception as e:
            logger.error("command handling failed: %s", e)
            self._state_fail(str(e))
            self.speak("Error boss")

    # -- main loop ------------------------------------------------------------------
    # Phrases that end conversation mode and return to the
    # wake-word loop.
    _GOODBYE_PHRASES = (
        "khuda hafiz",
        "alvida",
        "goodbye",
    )
    # Chitchat: answered directly by the LLM without planning.
    # Keeps greetings and small talk instant on slow hardware.
    _CHAT_PATTERNS = (
        "assalam",
        "salam",
        "سلام",
        "hello",
        "ہیلو",
        "aoa",
        "وعلیکم",
        "kya haal",
        "kia haal",
        "kesay ho",
        "kese ho",
        "kaisay ho",
        "kaise ho",
        "how are you",
        "کیا حال",
        "کیسے ہو",
        "کیسی ہو",
        "shukriya",
        "شکریہ",
        "thanks",
        "thank you",
        "khuda hafiz",
        "alvida",
        "goodbye",
    )

    def _conversation_loop(self) -> None:
        """Stay in dialogue after a single wake word.

        Follow-up commands don't need the wake word again: keep
        listening and answering until ``conversation_timeout``
        seconds of silence pass, or the user says a goodbye
        phrase.  Then return to the wake-word loop.
        """
        import time

        print(
            "conversation mode: speak freely, "
            "no wake word needed"
        )
        last_active = time.time()
        while True:
            if (
                time.time() - last_active
                >= self.config.conversation_timeout
            ):
                print("conversation mode ended (silence)")
                return
            command = self.listen_command(
                timeout=7, phrase_time=8
            )
            if not command:
                # Brief pause so a failing mic can never spin
                # this into a tight loop hammering the audio
                # driver.
                time.sleep(0.5)
                continue
            if any(
                phrase in command.lower()
                for phrase in self._GOODBYE_PHRASES
            ):
                self.speak("Theek hai boss")
                return
            last_active = time.time()
            try:
                self.process_command(command)
            except Exception:
                # Never fail silently here: an unexpected error
                # used to vanish into start()'s bare except and
                # leave only a dead "heard:" line behind.
                import traceback

                traceback.print_exc()
                print("command failed — listening again")

    def start(self) -> None:
        self.show_startup_gif()
        self.speak("Afnan is activated")
        from afnan_ai.wakeword import create_detector

        # On-device wake word when a model is configured;
        # otherwise the previous cloud loop is the fallback.
        detector = create_detector(
            self.config.wakeword_model,
            threshold=self.config.wakeword_threshold,
        )
        if detector is not None:
            print("on-device wake word active")
        while True:
            try:
                if detector is not None:
                    if not self._wait_for_wake_local(detector):
                        continue
                else:
                    word = self.listen_command(
                        timeout=5, phrase_time=3
                    )
                    if not word:
                        continue
                    if self.config.wake_word not in word.lower():
                        continue
                # Let the audio driver fully release the mic before
                # speaking (see listen_command): otherwise "Yes boss"
                # can come out silent on some systems.
                import time

                time.sleep(0.5)
                self.speak("Yes boss")
                if self.config.conversation_mode:
                    self._conversation_loop()
                    continue
                command = self.listen_command(
                    timeout=7, phrase_time=8
                )
                if command:
                    self.process_command(command)
                else:
                    # Wake word fired on background noise but no
                    # command followed: pause before listening for
                    # the wake word again so a noisy room doesn't
                    # spam "Yes boss" in a tight loop.
                    import time

                    time.sleep(
                        self.config.wakeword_miss_cooldown
                    )
            except SystemExit:
                break
            except KeyboardInterrupt:
                break
            except Exception:
                # Last-resort net: print instead of swallowing.
                # Silent failures here once left the user with a
                # dead "heard:" line and no clue what broke.
                import traceback

                traceback.print_exc()
                print("(unexpected error — listening again)")

    def _wait_for_wake_local(
        self, detector, timeout_s: float = 120.0
    ) -> bool:
        """Stream mic frames through the on-device detector.

        Returns True on wake, False on timeout/error (the
        caller then falls back to the cloud loop).
        """
        if sr is None or self.recognizer is None:
            return False
        import time

        try:
            with sr.Microphone(sample_rate=16000) as source:
                self.recognizer.adjust_for_ambient_noise(
                    source, duration=0.5
                )
                print("Listening (on-device wake word)...")
                stream = source.stream
                frame_bytes = detector.frame_samples * 2
                deadline = time.time() + timeout_s
                while time.time() < deadline:
                    try:
                        chunk = stream.read(
                            detector.frame_samples
                        )
                    except Exception:
                        return False
                    if len(chunk) < frame_bytes:
                        continue
                    if detector.is_wake(
                        chunk[:frame_bytes],
                        self.config.wakeword_threshold,
                    ):
                        return True
        except Exception:
            return False
        return False


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
