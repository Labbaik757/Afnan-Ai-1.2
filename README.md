# 🤖 Afnan AI 1.2 (Windows + macOS + Linux Edition)

Afnan AI is a personal voice assistant built with Python and powered by Ollama. It can understand voice commands, open applications, search the web, play music, capture screenshots, and assist you with everyday tasks using natural voice interaction.

> **Version:** 1.2  
> **Platform:** Windows 10/11, macOS, Linux  
> **Language:** Python

---

# ✨ Features

- 🎤 Voice Recognition
- 🤖 Local AI Chat using Ollama (Llama 3)
- 🗣️ Text-to-Speech Responses (Windows SAPI / macOS `say`, or pyttsx3)
- 💻 Open Visual Studio Code
- 🌐 Open Google Chrome / Microsoft Edge
- 🧭 Open Safari (macOS) — on Windows/Linux it opens your default browser instead
- 💬 Open WhatsApp (app on macOS, WhatsApp/protocol or WhatsApp Web on Windows)
- ▶️ Open YouTube
- 📂 Smart Folder Search (Downloads, Desktop, Documents, Pictures, Music, Videos)
- 🎵 Play Songs on YouTube
- 🔎 Google Search
- 📺 YouTube Search
- 📸 Screenshot Capture
- 🎬 Startup GIF Animation (in your browser, on every platform)
- 🎯 Wake Word Detection ("Afnan")
- ⚡ Fast Voice Command Processing

---

# 🎙️ Available Voice Commands

| Voice Command | Action |
|---------------|--------|
| Afnan | Activate the assistant |
| Open Visual Studio Code / Open VS Code | Opens VS Code |
| Open Chrome | Opens Google Chrome |
| Open Edge | Opens Microsoft Edge (Windows) |
| Open Safari | Opens Safari (macOS only) |
| Open YouTube | Opens YouTube |
| Open WhatsApp | Opens WhatsApp |
| Open Folder Downloads | Opens Downloads folder |
| Open Folder Desktop | Opens Desktop folder |
| Open Folder Documents | Opens Documents folder |
| Play Believer | Plays the requested song on YouTube |
| Search Google for Python | Searches Google |
| Search YouTube for AI | Searches YouTube |
| Screenshot | Captures a screenshot |
| Tell me about yourself | Afnan introduces itself |
| Introduce yourself | Afnan introduces itself |
| Who are you | Afnan introduces itself |
| Stop Afnan | Closes Afnan AI |

---

# 🛠️ Technologies Used

- Python
- SpeechRecognition
- Ollama
- PyWhatKit
- PyAutoGUI
- Webbrowser
- pyttsx3 (cross-platform text-to-speech, with OS speech as fallback)

---

# 📦 Requirements

- Python 3.10 or later
- Windows 10/11, macOS, or Linux
- Ollama Installed, with the Llama 3 model: `ollama pull llama3`
- Working Microphone (and microphone permission on Windows/macOS)
- Internet Connection (for Google speech recognition and online features)

### Windows note for PyAudio

On Windows, `pip install PyAudio` sometimes needs a wheel. If it fails, install it with:

```powershell
pip install pipwin
pipwin install pyaudio
```

or install a matching PyAudio wheel for your Python version, then run
`pip install -r requirements.txt` again.

---

# 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/Labbaik757/Afnan-Ai-1.2.git
cd Afnan-Ai-1.2
```

### Install dependencies

Windows (PowerShell / CMD):

```powershell
python -m pip install -r requirements.txt
```

macOS / Linux:

```bash
pip3 install -r requirements.txt
```

The macOS-only packages in `requirements.txt` are marked with
`sys_platform == 'darwin'`, so pip automatically skips them on Windows
and Linux — and the Windows-only `pywin32` is skipped on macOS/Linux.

### Run Afnan AI

Windows:

```powershell
python main.py
```

macOS / Linux:

```bash
python3 main.py
```

---

# 📄 requirements.txt (core)

```text
SpeechRecognition
PyAudio
PyAutoGUI
pywhatkit
ollama
pyttsx3
```

---

# 📁 Project Structure

```
Afnan-Ai-1.2
│
├── main.py                  (thin entry point, backwards compatible)
├── afnan_ai/
│   ├── agent.py             (core agent — no OS-specific code, no
│   │                         direct model-client calls)
│   ├── state.py             (centralized AgentState — goal, steps,
│   │                         observations, tool results, status)
│   ├── planner.py           (Planner — goal + AgentState + tools →
│   │                         validated TaskPlan, never executes)
│   ├── executor.py          (Executor — runs a TaskPlan step by step
│   │                         via ToolRegistry, records AgentState)
│   ├── verifier.py          (Verifier — checks actual results vs
│   │                         expected_result, never re-executes)
│   ├── orchestrator.py      (Agent — central orchestrator: state +
│   │                         planner + executor + verifier, full
│   │                         task lifecycle, max-iteration limit)
│   ├── recovery.py          (RecoveryManager — replans after a
│   │                         failed/uncertain step, never repeats
│   │                         a failed action, attempts recorded)
│   ├── config.py            (AgentConfig — wake word, limits,
│   │                         default model; env overridable)
│   ├── log_config.py        (structured logging, stdlib only)
│   ├── browser/             (BrowserController — launch/connect,
│   │   │                     tabs, navigation, page state, shutdown)
│   │   ├── base.py          (tab/page types + structured errors)
│   │   ├── backend.py       (BrowserBackend interface +
│   │   │                     PlaywrightBackend)
│   │   ├── controller.py    (session/tab management, validation)
│   │   └── tools.py         (browser_* Tools for the registry)
│   ├── llm/
│   │   ├── base.py          (LLMProvider interface + typed errors)
│   │   ├── ollama.py        (OllamaProvider — local Ollama, llama3)
│   │   └── factory.py       (provider registry / default provider)
│   ├── tools/
│   │   ├── base.py          (Tool interface + structured errors)
│   │   ├── registry.py      (ToolRegistry — register/get/execute)
│   │   └── builtin.py       (open_url, open_application,
│   │                         search_google, take_screenshot, ...)
│   ├── speech.py            (pyttsx3 first, adapter speech as fallback)
│   └── platform/
│       ├── base.py          (PlatformAdapter interface)
│       ├── factory.py       (auto-selects the adapter at runtime)
│       ├── windows.py       (PowerShell speech, startfile, cmd start)
│       ├── macos.py         (say, open / open -a, mdfind)
│       └── linux.py         (espeak/spd-say, xdg-open)
├── tests/                   (unit + integration + end-to-end tests,
│                             run on any host OS)
├── tools/check_compatibility.py
├── requirements.txt
├── README.md
├── afnan_animation.gif
├── afnan_animation.html
└── screenshots/   (created when you take a screenshot)
```

# 🏗️ Architecture — OS Abstraction Layer

Platform-specific code is isolated behind one interface, so the
core agent never touches an OS command directly:

- `afnan_ai/platform/base.py` — `PlatformAdapter`: `speak_system()`,
  `open_path()`, `launch_app()`, `find_folder()`
- `afnan_ai/platform/windows.py` / `macos.py` / `linux.py` — the
  only files containing Windows, macOS or Linux specific calls
- `afnan_ai/platform/factory.py` — `get_adapter()` reads
  `platform.system()` at runtime and returns the Windows, macOS
  or Linux adapter automatically — no configuration needed
- `afnan_ai/agent.py` — wake word, command routing, search, music,
  screenshots and Ollama fallback, all written against the adapter

A test in `tests/test_agent_commands.py` fails if an OS-specific
call ever leaks back into the core agent.

# 🧠 AgentState — Centralized Task State

`afnan_ai/state.py` provides one serializable `AgentState` per task.
Every component (voice agent, platform adapters, tests, a future UI)
can read and update the same object:

```python
from afnan_ai.state import AgentState

state = AgentState.create("Open Chrome and search for Python")
state.start_step("open chrome")
state.add_observation("User said: open chrome", source="microphone")
state.add_tool_result("chrome", success=True, output="opened")
state.complete_step("open chrome", result="opened")
state.complete_task()

data = state.to_json()              # save / send anywhere
restored = AgentState.from_json(data)  # lossless on any OS
state.save("state.json")            # or AgentState.load("state.json")
```

It tracks the task **goal**, **current step**, **completed steps**,
**failed steps** (with errors), **observations**, **tool results**
and **status** (`pending`, `running`, `paused`, `completed`,
`failed`, `cancelled`).  It uses only the Python standard library
and `pathlib`, so a state saved on Windows loads identically on
macOS and Linux.

The voice agent updates it automatically for every command
(`agent.state`), you can start an explicit task with
`agent.start_task(goal)`, share one state between components by
passing `AfnanAgent(state=...)`, or turn tracking off with
`track_state=False` — existing behaviour is unchanged either way.

Tests for creation, updating, JSON/file serialization and
failure-state handling live in `tests/test_agent_state.py`.

# 🤖 LLMProvider — Model Abstraction

The agent never calls Ollama (or any model client) directly.  It
only talks to the `LLMProvider` interface in `afnan_ai/llm/base.py`:

- `chat(messages) -> str` — send chat messages, get the reply text
- `generate(prompt) -> str` — single-prompt convenience wrapper
- Typed failures: `LLMUnavailableError` (not installed),
  `LLMConnectionError` (service unreachable) and
  `LLMInvalidResponseError` (unexpected response shape) — no
  provider-specific exception leaks into the agent

The existing Ollama integration lives in
`afnan_ai/llm/ollama.py` as `OllamaProvider`, unchanged in
behaviour: local Ollama, model `llama3`, reply taken from
`response["message"]["content"]`, and the same spoken messages on
failure ("AI is not available. Ollama is not installed." /
"AI is not responding. Make sure Ollama is running.").

Adding a future local or cloud model means writing one new
provider class and registering it in `afnan_ai/llm/factory.py` —
the agent's code does not change:

```python
from afnan_ai.agent import AfnanAgent
from afnan_ai.llm import create_provider

agent = AfnanAgent(llm_provider=create_provider("ollama", model="llama3"))
# future: create_provider("some-cloud-provider", ...)
```

`agent.ask_ai(prompt)` is the interface-based entry point;
`agent.ask_local_ai(prompt)` is kept as a backwards-compatible
alias, and `main.py` exposes both plus `get_llm_provider()`.

Tests for a successful response, a connection failure, an invalid
response, an unavailable client and swapping in a completely
different provider without touching the agent live in
`tests/test_llm_provider.py`.  A test there also fails if a direct
`ollama` call ever leaks back into `agent.py` or `main.py`.

# 🧰 Tools — Generic Tool Interface & ToolRegistry

Every capability is a `Tool` in `afnan_ai/tools/base.py` with a
**name**, a **description**, a JSON-Schema-style **input schema**
and an **execute** method.  The central `ToolRegistry`
(`afnan_ai/tools/registry.py`) registers, finds and runs them:

```python
from afnan_ai.tools import create_default_registry

registry = create_default_registry(adapter)   # adapter from afnan_ai.platform
registry.register(my_custom_tool)             # dynamic registration
tool = registry.get("search_google")          # raises a structured error if unknown

result = registry.execute("search_google", {"query": "python"})
result.success   # True
result.output    # {"query": "python", "url": "https://...", "opened": True}

bad = registry.execute("search_google", {})   # missing argument — no crash
bad.error.code   # ToolErrorCode.MISSING_ARGUMENTS
```

Built-in tools preserve the exact behaviour Afnan already had:
`open_url`, `open_application` (Chrome/VS Code/Edge/WhatsApp/
Safari via the platform adapter), `search_google`, `search_youtube`
and `take_screenshot`.  The voice agent routes its commands through
the registry (`agent.execute_tool(...)`, `agent.list_tools()`),
records every execution in `AgentState`, and falls back exactly as
before (e.g. Chrome not launching opens google.com instead).
`registry.definitions()` returns serializable tool descriptions
ready to hand to an LLM for function calling.

Failures are structured, never bare crashes: an unknown tool gives
`tool_not_found`, missing arguments give `missing_arguments`, a
wrong-typed argument gives `invalid_arguments`, and a tool that
fails while running gives `execution_failed` — all as
`ToolResult(success=False, error=ToolError(code, message, tool,
details))`, with matching exception forms (`ToolNotFoundError`,
`ToolValidationError`, `ToolExecutionError`) for callers who prefer
`try/except`.

Tests for registration, dynamic lookup, execution, invalid tools,
missing/invalid arguments and execution failures live in
`tests/test_tool_registry.py`.

# 🗺️ Planner — Structured Task Plans (No Execution)

`afnan_ai/planner.py` contains a `Planner` that turns a user goal,
the current `AgentState` and the registered tools into a
structured `TaskPlan` — and nothing more.  The Planner never
executes a tool; it only reads tool definitions from the
`ToolRegistry` to know what it may plan with.

```python
from afnan_ai.planner import Planner

planner = Planner(agent.llm, agent.tools)   # or: agent.planner
plan = planner.plan("Open Chrome and search for Python", state=agent.state)

plan.goal            # "Open Chrome and search for Python"
plan.steps[0].step_id         # "step_1"
plan.steps[0].description     # "Open the Chrome browser"
plan.steps[0].tool_name       # "open_application"
plan.steps[0].arguments       # {"application": "chrome"}  (schema-validated)
plan.steps[0].expected_result # "Chrome is launched"
data = plan.to_json()         # TaskPlan.from_json(data) restores it
```

The LLM (through the `LLMProvider` interface, Ollama by default)
is asked to reply with exactly one JSON object
`{"goal": ..., "steps": [...]}` and that output is validated
strictly before a plan is returned: valid JSON, exactly the
required fields on every step (`step_id`, `description`,
`tool_name`, `arguments`, `expected_result`), unique `step_id`
values, `tool_name` must be one of the available tools, and
`arguments` must satisfy that tool's input schema.

Invalid or failed planning is handled safely with structured
`PlanningError`s — never a bare crash and never a half-valid plan:
non-JSON output (`invalid_llm_output`), an unknown tool or bad
arguments (`invalid_plan`), an unreachable model
(`llm_connection_failed`), a missing model (`llm_unavailable`),
an empty goal (`empty_goal`) or no tools to plan with
(`no_tools_available`).  The agent exposes the same thing as
`agent.create_plan(goal)` (and `main.create_plan(goal)`); a
successful plan is recorded in `AgentState` as an observation,
which is bookkeeping, not execution.

Tests for successful planning, invalid LLM output (bad JSON,
missing/extra fields, unknown tool, bad arguments, duplicate
`step_id`), failed planning (connection failure, unavailable
model, empty goal, no tools) and the never-executes-tools
guarantee live in `tests/test_planner.py`.
`tools/check_compatibility.py` additionally AST-checks that
`planner.py` contains no tool-execution call.

# ⚙️ Executor — Running a TaskPlan

`afnan_ai/executor.py` carries out a `TaskPlan` produced by the
Planner.  For every step, in order, the `Executor`:

1. checks the named tool **exists** in the ToolRegistry,
2. **validates the step's arguments** against that tool's input schema,
3. executes the tool **through the ToolRegistry only**, and
4. records the result (tool result + completed/failed step) in `AgentState`.

```python
from afnan_ai.executor import Executor

executor = Executor(agent.tools, state=agent.state)  # or: agent.executor
report = executor.execute_plan(plan)

report.success                 # True only if every step really worked
report.status                  # "completed" / "failed"
report.step_results[0].output  # the tool's real output
report.step_results[0].error   # structured ToolError dict on failure
agent.state.status             # completed / failed — results are recorded
```

Two safety rules are enforced and tested:

- **A failed action is never silently successful.** An unknown tool
  (`tool_not_found`), bad arguments (`missing_arguments` /
  `invalid_arguments`), a crashing tool (`execution_failed`) — each
  makes that step `success=False`, fails the task in `AgentState`,
  and by default stops the plan so later steps are reported as
  `skipped`, not as done. (`Executor(..., stop_on_failure=False)`
  attempts every step, but the report is still `failed`.)
- **No arbitrary code execution.** A plan step can only name an
  already-registered tool; tool names and arguments are treated
  strictly as data, never as Python code, a shell command or an
  import. `tools/check_compatibility.py` AST-checks that
  `executor.py` dispatches only via `registry.execute(...)`.

The agent ties both halves together:

```python
agent.execute_plan(plan)                    # execute an existing plan
report = agent.plan_and_execute("Open Chrome and search for Python")
```

Tests for successful execution, failed and crashing tools,
invalid tools, invalid arguments, skipped steps, AgentState
updates and the no-arbitrary-code guarantee live in
`tests/test_executor.py`.

# 🔍 Verifier — Did the Step Actually Do What Was Expected?

`afnan_ai/verifier.py` contains an independent `Verifier` that
checks each executed step's **actual** result against that step's
**expected_result**.  It holds no ToolRegistry and no model, so it
cannot re-run anything — it only analyzes the structured execution
result (from the Executor), plus any evidence already in
`AgentState` (completed/failed step records, tool results,
observations).

```python
from afnan_ai.verifier import Verifier, VerificationStatus

verifier = Verifier()  # or: agent.verifier
result = verifier.verify_step(step, execution_result, state=state)
result.status      # VerificationStatus.VERIFIED / FAILED / UNCERTAIN
result.reason      # human-readable explanation
result.confidence  # 0.0–1.0 evidence strength

report = verifier.verify_plan(plan, execution_report, state=state)
report.status      # failed if any step failed, verified only if all verified
```

- **verified** — the execution succeeded and the actual output
  confirms the expected outcome (e.g. expected "Chrome is launched",
  output `{"application": "chrome", "launched": true}`)
- **failed** — the execution failed or was skipped, the output flags
  a non-outcome (`launched: false`), a different application actually
  ran, or the output shares nothing with the expected outcome
- **uncertain** — no result/evidence, no or vague `expected_result`,
  success with no observable output, only partial keyword overlap,
  or conflicting evidence between the execution result and
  `AgentState`

Every judgement is recorded in `AgentState` — a `verifier`
observation, a structured entry in
`state.metadata["verifications"]`, and an annotation on the step's
record — without adding a tool result, because recording a
judgement is not executing a tool.  The analysis is deterministic
(no LLM call), so the same result always verifies the same way.

The agent ties the three phases together without merging them:

```python
report, verification = agent.execute_and_verify(plan)  # Executor, then Verifier
```

Tests for verified, failed and uncertain outcomes, contradictions,
AgentState recording and the never-re-executes guarantee live in
`tests/test_verifier.py`; `tools/check_compatibility.py`
additionally AST-checks that `verifier.py` contains no execution
call.

# 🧭 Agent — Central Orchestrator (Complete Task Lifecycle)

`afnan_ai/orchestrator.py` contains the `Agent` — the central
orchestration layer that connects `AgentState`, the `Planner`, the
`Executor` and the `Verifier` and manages a complete task from
goal to completion:

```
user goal received
  → AgentState created / updated
  → Planner generates a TaskPlan
  → Executor executes the next step
  → Verifier verifies that step's result
  → AgentState updated
  → next step … or task completion / failure
```

```python
from afnan_ai import Agent

result = agent.run_task("Open Chrome and search for Python scripting")
result.status       # OrchestrationStatus.COMPLETED / FAILED /
                    # PLANNING_FAILED / MAX_ITERATIONS_EXCEEDED
result.success      # True only when the task genuinely completed
result.iterations   # step executions actually performed
result.plan         # the TaskPlan that was executed
result.execution    # aggregated ExecutionReport (unrun steps = skipped)
result.verification # aggregated VerificationReport
result.state        # final AgentState (inspect, save, serialize…)
```

Rules the orchestrator enforces:

- **Maximum iteration limit is mandatory.** Every step execution
  counts as one iteration; reaching `max_iterations` (default 10,
  validated as a positive integer, overridable per run but never
  removable) stops the task with a `max_iterations_exceeded`
  outcome — the agent can never loop forever.
- **Failures stop the task honestly.** A planning failure
  (`planning_failed`), an execution failure, or a failed
  verification ends the run; remaining steps are reported as
  *skipped*, never as done. An *uncertain* verification is
  recorded but does not block an otherwise successful execution
  unless `strict_verification=True`.
- **No responsibility is duplicated.** The Agent plans only via
  the Planner, executes only via the Executor (one step at a time,
  through `Executor.execute_step`) and judges only via the
  Verifier; `tools/check_compatibility.py` AST-checks that it
  never touches the registry, a tool or a platform adapter itself.

The voice assistant uses it as its orchestration layer
(`agent.orchestrator`, `agent.run_task(goal)`, also exposed as
`main.run_task(goal)`), while the existing voice-command behaviour
is unchanged. Tests for successful completion, planning failure,
execution failure, unknown tools, the maximum-iteration stop and
voice-assistant integration live in `tests/test_orchestrator.py`.

Every request the user speaks (after the wake word) or types goes
through this orchestrator: `agent.handle_request(text)` (also
`main.handle_request(text)`, and `agent.process_command(text)` for
backwards compatibility) delegates the actual task to
`Agent.run()` and speaks the outcome.  Session control
("stop afnan", "introduce yourself") is handled directly, as
before.  If the model cannot plan at all (e.g. it is offline),
the request falls back to the clearly isolated legacy routing in
`agent._handle_legacy_command` — the pre-Tool command patterns,
some of which use capabilities that are not Tools yet (opening a
folder by name, playing a song).  Nothing is broken by the
migration; those patterns are marked for later migration into
Tools.  `main.py` itself stays a thin set of delegates with no
planning, execution, verification or recovery logic.

# 🩹 Recovery — Replanning After a Failed or Uncertain Step

`afnan_ai/recovery.py` contains the `RecoveryManager`.  When a
step fails, or the Verifier cannot confirm it (uncertain/failed),
the orchestrator does **not** blindly run the same action again.
Instead, Recovery hands the Planner:

- the original goal,
- the failure reason,
- the current `AgentState` (completed steps, observations,
  earlier attempts), and
- the exact previous attempt (tool + arguments) that did not work,

and asks for a *different* plan for the remaining work.

- **No blind repetition** — a recovery plan containing any
  already-failed action (same tool, same arguments) is rejected
  and never executed.
- **Limited attempts** — `max_recovery_attempts` (default 2;
  0 disables recovery) caps replanning, and recovered steps still
  count against the orchestrator's maximum-iteration limit.
- **Fully recorded** — every attempt (replanned, planner-failed,
  rejected, limit-reached) is stored in
  `state.metadata["recovery_attempts"]`, noted as a `recovery`
  observation, and returned on
  `result.recovery_attempts`.
- **No duplicated responsibilities** — Recovery only replans via
  the Planner; execution stays with the Executor, judgement with
  the Verifier, and the decision of when to recover or give up
  with the Agent.  `tools/check_compatibility.py` AST-checks that
  `recovery.py` never executes a tool.

Tests for successful recovery, blind-repeat rejection,
planner-replan failure, the recovery-attempt limit and the
iteration cap live in `tests/test_recovery.py`; voice-independent
Agent invocation and entry-point/backward-compatibility
integration tests live in `tests/test_entry_points.py`.

# 🌐 BrowserController — Programmatic Browser Control (Phase 2)

`afnan_ai/browser/` is a modular browser-control system that
works the same on Windows, macOS and Linux:

- **`BrowserController`** owns a browser session: launch or
  connect (CDP), create/select/close tabs, navigate by URL,
  back/forward/reload, read the current page state (URL + title),
  and shut down.  Navigating with no tab open creates one.
- **`BrowserBackend`** is the driver boundary.  The shipped
  `PlaywrightBackend` drives Chromium/Chrome/Edge/Firefox/WebKit
  via Playwright (lazily imported — installing nothing is fine
  until a browser actually launches).  Tests and other hosts can
  inject any backend without touching controller or agent code.
- **Tools** — every operation is a registry Tool (`browser_launch`,
  `browser_connect`, `browser_new_tab`, `browser_list_tabs`,
  `browser_select_tab`, `browser_close_tab`, `browser_navigate`,
  `browser_current_page`, `browser_back`, `browser_forward`,
  `browser_reload`, `browser_shutdown`), registered on the agent's
  ToolRegistry by default, so the Planner can plan browser tasks
  and the Executor runs them like any other capability.  Disable
  with `AfnanAgent(..., enable_browser_tools=False)`.
- **Page interaction tools** — `browser_find_elements` (inspect
  elements: identity `ref` + tag/text/attributes/visible/enabled),
  `browser_inspect_element`, `browser_click`, `browser_type`,
  `browser_clear`, `browser_select_option`, `browser_press_key`,
  and `browser_scroll` (page scroll or scroll-element-into-view).
  Targets are either a `ref` from `browser_find_elements` or a
  locator: stable **selector / test id first**, with accessibility
  fallbacks (**role + name**, label, placeholder, visible text).
  Every interaction validates its target first — element must
  exist, be current (a ref from before the last navigation is a
  `stale_element` error; find it again), visible, enabled, and of
  the right kind (typing only into editable fields, options only
  on `<select>`) — so an action never lands on the wrong element.
- **Structured errors only** — browser unavailable (install with
  `pip install playwright` + `playwright install chromium`),
  connection failed, browser not started, invalid tab,
  navigation failures, `element_not_found`, `invalid_element`,
  `stale_element` and `timeout` all come back as
  ``ToolResult(success=False)`` with the browser error code in
  `error.details["browser_error"]["code"]`; nothing crashes and
  nothing silently "succeeds".

```python
agent.execute_tool("browser_launch", {})
agent.execute_tool("browser_navigate", {"url": "https://example.com"})
page = agent.execute_tool("browser_current_page", {})
page.output  # {"tab_id": "tab_1", "url": "https://example.com/", "title": "Example Domain"}
found = agent.execute_tool("browser_find_elements", {"role": "button", "name": "Sign in"})
agent.execute_tool("browser_click", {"ref": found.output["elements"][0]["ref"]})
agent.execute_tool("browser_type", {"selector": "#username", "text": "afnan"})
```

Unit tests (fake in-memory backend) live in
`tests/test_browser_controller.py`; interaction tests (login-form
DOM: successful interactions, element-not-found, stale/invalid
element, timeout and execution-failure cases) live in
`tests/test_browser_interactions.py`; registry/Agent integration
tests live in `tests/test_browser_tools.py`;
`tools/check_compatibility.py` verifies the browser tools exist
and that driver code stays inside `afnan_ai/browser/`.

# 🏁 Phase 1 Status — Clean, Tested, Cross-Platform

Phase 1 (the core agent architecture) is complete and verified
end to end:

- **Full pipeline tested** — `tests/test_end_to_end.py` drives a
  goal through AgentState → Planner → Executor → Verifier →
  Recovery/Replanning → Completion/Failure, through the voice and
  text entry points, and on all three platform adapters
  (Windows/macOS/Linux, with OS calls mocked).
- **246 automated tests, all passing**, plus
  `tools/check_compatibility.py` AST guards that keep the
  component responsibilities separate (Planner never executes,
  Executor dispatches only via ToolRegistry, Verifier and
  Recovery never execute, orchestrator never bypasses Executor).
- **Backward compatibility verified** — wake word, speech
  recognition, text input, TTS and every pre-existing voice
  command behave as before; the legacy-only commands are isolated
  in `agent._handle_legacy_command` for later Tool migration.
- **Clean project** — components log through structured stdlib
  logging (`afnan_ai/log_config.py`, silent until configured),
  tunables live in `AgentConfig` (`afnan_ai/config.py`,
  environment-overridable), dead code (the unused tkinter
  `gif_viewer.py`) and unused dependencies/imports were removed,
  and `requirements.txt` contains only what the code imports or
  a feature requires.

Phase 2 (browser / computer-use agent development) builds on
this foundation: new capabilities are added as registered Tools
without touching the core pipeline, starting with the
BrowserController above.

# ✅ Compatibility Tests

The tests simulate all three operating systems (mocking
`platform.system`, `subprocess`, speech and the microphone), so
they run safely on any host:

```bash
python -m unittest discover -s tests -v
# or
python tools/check_compatibility.py
```

They cover adapter selection for Windows/macOS/Linux, each
adapter's speech/open/launch behaviour, known-folder lookup, and
that every existing voice command still routes to the same feature
as before the refactor.

---

# ⚙️ How It Works

1. Launch Afnan AI.
2. The startup animation will appear in your browser.
3. Afnan activates the microphone.
4. Say **"Afnan"** to wake the assistant.
5. Afnan replies **"Yes Boss"**.
6. Speak your command.
7. Afnan processes and executes your request, or asks the local
   Ollama Llama 3 model when no command matches.

At startup the platform factory detects your OS and picks the
matching adapter, so the same `main.py` opens apps, folders and
files natively on Windows, macOS and Linux without any manual
configuration. Folder search uses Spotlight on macOS and a
home-folder search on Windows/Linux.

---

# 💬 Example

```
You: Afnan

Afnan: Yes Boss

You: Open Chrome

Afnan: Opening Chrome
```

---

# ⚠️ Notes

- Windows, macOS and Linux are supported in this version.
- On Windows, speech works out of the box with the built-in SAPI
  voices (via pyttsx3 / PowerShell), no `say` command needed.
- Safari is a macOS app; on Windows/Linux "Open Safari" opens your
  default browser instead.
- Ollama must be installed and running for AI chat functionality.
- Make sure your microphone permission is enabled in your OS settings.

---

# 🚀 Afnan AI 1.7 — Coming Soon

Afnan AI 1.7 is currently under active development and will introduce a major upgrade over version 1.2.

### Planned Features

- 🧠 Smarter AI Engine
- ⚡ Faster Performance
- 🎨 Modern User Interface
- 🎙️ Improved Voice Recognition
- 🤖 Advanced AI Automation
- 🔥 More Powerful Voice Commands
- 💎 Exclusive Premium Features

Stay tuned for future updates.

Thank you for supporting Afnan AI! ❤️

---

# 👨‍💻 Author

Developed with ❤️ by **Afnan**

If you like this project, please consider giving it a ⭐ on GitHub.

---

# 📄 License

This project is licensed under the MIT License.
