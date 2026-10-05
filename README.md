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
- **`BrowserBackend`** is the driver boundary (now the
  `BrowserEngineAdapter` interface — see the Runtime section
  below).  The default `ChromiumAdapter` drives a real
  Chromium-family browser over the Chrome DevTools Protocol
  (standard library only); `PlaywrightAdapter` remains as the
  fallback/development adapter (lazily imported — installing
  nothing is fine until a browser actually launches).  Tests
  and other hosts can inject any adapter without touching
  controller or agent code.
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
- **Observation tools** — `browser_observe_page` returns the
  page's structured state (URL, title, visible text, the
  interactive elements with usable refs, and whether the page
  changed since the last observation); `browser_wait_for`
  waits for dynamic content by condition (element present/hidden,
  text present, URL/title contains — driver-level condition
  waits, never a fixed sleep) and times out with a structured
  error; `browser_screenshot` captures a PNG of the page to
  disk.  Observation outputs carry an `observation` summary
  that the Executor records in `AgentState` alongside the tool
  result, so what the agent saw is part of the task record.
  Pages the site itself opens (popups/new-window links) are
  adopted as regular tabs and appear in `browser_list_tabs`.
  A failed observation is an error — never an empty "success".

```python
agent.execute_tool("browser_launch", {})
agent.execute_tool("browser_navigate", {"url": "https://example.com"})
page = agent.execute_tool("browser_current_page", {})
page.output  # {"tab_id": "tab_1", "url": "https://example.com/", "title": "Example Domain"}
found = agent.execute_tool("browser_find_elements", {"role": "button", "name": "Sign in"})
agent.execute_tool("browser_click", {"ref": found.output["elements"][0]["ref"]})
agent.execute_tool("browser_type", {"selector": "#username", "text": "afnan"})
obs = agent.execute_tool("browser_observe_page", {})
shot = agent.execute_tool("browser_screenshot", {})
shot.output["path"]  # screenshots/screenshot_20261005_....png
```

Unit tests (fake in-memory backend) live in
`tests/test_browser_controller.py`; interaction tests (login-form
DOM: successful interactions, element-not-found, stale/invalid
element, timeout and execution-failure cases) live in
`tests/test_browser_interactions.py`; observation tests (page
state, dynamic-content waits, popup adoption, screenshot
recording in AgentState) live in
`tests/test_browser_observation.py`; registry/Agent integration
tests live in `tests/test_browser_tools.py`;
`tools/check_compatibility.py` verifies the browser tools exist
and that driver code stays inside `afnan_ai/browser/`.

## Reliability: state-grounded verification + recovery

Browser success is not trusted blindly; it is confirmed against
the page itself:

- **Post-action confirmation** — `BrowserReliability` (in
  `afnan_ai/browser/reliability.py`) is wired into the Verifier's
  generic observation-provider hook. After every `browser_*`
  step, the Verifier judges the expected result against a fresh
  `browser_observe_page` snapshot (URL, title, text, elements) —
  a click that "succeeds" but leaves the page unchanged is
  **failed/uncertain**, not verified. Non-browser steps are
  untouched, and the Verifier stays browser-agnostic (enforced
  by `tools/check_compatibility.py`).
- **Structured recovery strategies** — on a failed/uncertain
  browser step, the failure advisor classifies the problem
  (navigation failure, element-not-found, timeout, stale
  element, invalid tab, unexpected popup/new tab, page-state
  mismatch) and records a strategy in AgentState with the
  verification: relocate from the observed elements (with
  candidate locators taken from the real page), wait for a
  condition before acting, re-find after the page changed,
  adopt/select the popup tab, choose a new route, and so on.
- **Replanning, never blind repetition** — the advice and the
  observed page travel with the verification into AgentState,
  so the existing RecoveryManager/Planner generate an
  alternative action; a recovery plan that repeats the failed
  action (same tool + same arguments) is rejected and never
  executed, and recovery stays within the existing attempt and
  iteration limits.

End-to-end tests (successful sign-in, failure → recovery →
completion, blind-repeat rejection, navigation-failure
recovery, unexpected-popup recovery, dynamic-content wait) live
in `tests/test_browser_reliability.py`.

## Afnan Browser Runtime (Phase 3 foundation)

The agent never depends on a browser automation library.  The
layering is:

```
Afnan Agent → Browser Tools → BrowserController
    → AfnanBrowserRuntime → BrowserEngineAdapter
    → ChromiumAdapter → Chromium            (current)
    → PlaywrightAdapter → Chromium          (fallback / development)
    → Native Afnan Chromium Adapter         (future)
        → Customized Afnan Browser
```

- **`ChromiumAdapter`** (`afnan_ai/browser/chromium_adapter.py`)
  is the current engine: it launches a Chromium-family browser
  (Chromium, Chrome or Edge — set `AFNAN_CHROMIUM_EXECUTABLE`
  to choose the binary) and drives it over the Chrome DevTools
  Protocol using only the Python standard library.  Each Afnan
  profile is its own Chromium process with its own
  user-data directory, so profiles (persistent or isolated)
  never share cookies/storage; tabs, navigation, screenshots,
  element interaction and page observation all flow through
  CDP, with accessibility from the modern CDP Accessibility
  domain (the same normalized tree the Playwright adapter
  produces from `aria_snapshot()`).  The debug port binds to
  127.0.0.1 only, and profile data is never read back into
  AgentState, logs or screenshots.
- **`PlaywrightAdapter`** (`afnan_ai/browser/backend.py`) is
  preserved as the fallback/development adapter
  (`BrowserBackend`/`PlaywrightBackend` remain as compatibility
  aliases).  Both adapters return identical normalized results
  — `tests/test_chromium_adapter.py` runs the same controller
  workflow through both and through a fake adapter and compares
  the observations.  Swapping engines (including the future
  native Afnan Chromium build) means implementing
  `BrowserEngineAdapter` once; runtime, controller, tools,
  Planner, Executor, Verifier and Recovery stay untouched.

- **`AfnanBrowserRuntime`** (`afnan_ai/browser/runtime.py`) is
  the actual owner of browser state: lifecycle
  (start/stop/restart/connect), the engine-level session
  (session id, active profile, tab records with URLs/titles),
  persistent profiles (a registry + per-profile storage
  directories under a runtime dir; credential-shaped
  preferences are stripped and never logged), session
  persistence (`session.json`, redacted, reloadable after a
  restart), an event stream (`browser_started`, `tab_created`,
  `navigation_completed`, `popup_detected`,
  `download_completed`, `browser_crashed`, ... — subscribable
  and logged, ready for AgentState consumers), capability
  discovery (Afnan-level names via the `browser_capabilities`
  tool — never raw engine features), and crash detection +
  recovery: a dead engine flips the session to `crashed`,
  `recover()` restarts it and reports the recoverable tabs
  instead of blindly restarting the task.  By default the
  agent gives the runtime a persistent home
  (`~/.afnan-ai/browser-runtime`, override with
  `browser_runtime_dir=` / `AgentConfig.browser_runtime_dir` /
  `AFNAN_BROWSER_RUNTIME_DIR`), so profiles and session state
  survive restarts; supplying your own controller keeps full
  control.
- **`BrowserEngineAdapter`** (`afnan_ai/browser/engine.py`) is
  the only interface an engine implements; handles stay opaque
  and no engine types cross it.  **`ChromiumAdapter`** is the
  default implementation (real Chromium via CDP);
  **`PlaywrightAdapter`** (`backend.py`) is the fallback
  (Chromium via Playwright, persistent profile contexts
  included); `BrowserBackend`/`PlaywrightBackend` remain as
  compatibility aliases.  The runtime also exposes runtime
  health (`runtime.health()` — engine alive, session valid,
  active tab, page responsive), tab↔task association and a
  permission extension point (`check_permission` with
  normal/sensitive/destructive/approval_required levels; human
  approval itself stays with the controller's ApprovalGate).
- **Models** (`afnan_ai/browser/models.py`) — `BrowserSession`,
  `BrowserProfile`, `BrowserTab`, `BrowserWindow`,
  `BrowserPage`, `BrowserObservation`, `BrowserElement`,
  `BrowserAction`, `BrowserResult`: engine-independent,
  serializable records; AgentState and checkpoints only ever
  hold these (URLs redacted), never engine objects.
- **Errors** are Afnan codes end to end: `browser_unavailable`,
  `startup_failed`, `navigation_failed`, `tab_not_found`,
  `element_not_found`, `timeout`, `browser_crashed`,
  `session_expired`, `profile_error`, `unsupported_operation`.
  Engine exceptions never leak past the adapter/runtime
  boundary, and the runtime never executes model-generated
  code.

Tests: `tests/test_browser_runtime.py` drives a purpose-built
`FakeBrowserAdapter` (startup, tabs, navigation, events,
profiles, sessions, crash recovery, serialization, controller
and AgentState integration) with no real browser launched
(41 browser tools total).  `tests/test_chromium_adapter.py`
covers the ChromiumAdapter against a fake CDP connection,
runs real-Chromium integration tests (launch, navigation,
tabs, screenshots, profile persistence, restart, crash
recovery) when a Chromium binary is installed, and compares
the normalized controller output of the Chromium, Playwright
and fake adapters.

## Autonomous browser workflow

`run_browser_goal(goal)` (main / `AfnanAgent.run_browser_goal`,
backed by `afnan_ai/browser/workflow.py`) runs one complex
browser goal end to end on the existing pipeline — no layer
gained a second job:

* before acting, the browser is briefed into the task state
  (open tabs with purposes, current page, active profile,
  network health), so the Planner works from what the browser
  actually looks like;
* every action is executed through the ToolRegistry and
  re-verified against a fresh observation — a click that
  changed nothing is never counted as success;
* failures and uncertain states go to the RecoveryManager with
  the full state (completed work is not redone, failed actions
  are never blindly repeated, sensitive actions still pass the
  human approval gate);
* step, recovery and now also wall-clock limits
  (`max_iterations`, `max_duration_s`) bound the run;
* the final answer is composed from recorded evidence —
  search results, extracted page text, verified steps and
  downloads — not from what the actions merely claimed.

End-to-end scenarios live in `tests/test_browser_workflow.py`
(search → open → extract, multi-tab comparison, form login,
pagination, downloads, dynamic SPA pages, limits, recovery,
profile/session continuity).

## Observation-driven loop (Phase 3)

`run_browser_goal(goal, loop=True)` — or
`Orchestrator.run_loop(...)` directly — runs the task as a
true autonomous loop instead of one long pre-generated plan:

* each cycle the Planner sees the **current** AgentState
  (fresh observations, completed/failed steps, tool results)
  and proposes only the next few actions (`batch_limit`,
  default 3), or reports the goal complete via the
  `{"complete": true}` plan form;
* every action is executed and verified against a fresh
  observation before the next decision; a failed or uncertain
  step stops the batch immediately — the remaining steps of
  that plan are discarded, recovery replans from the current
  state, and the Planner re-decides. A stale plan is never
  followed blindly;
* hard limits bound the loop: `max_steps`,
  `max_duration_s`, `max_replans`, `max_llm_calls`,
  `max_repeated_actions` (the same tool with the same
  arguments may not repeat endlessly) plus the existing
  recovery-attempt limits;
* the Planner additionally self-repairs one invalid reply
  (bounded `max_parse_retries`, default 1) before failing.

Accessibility now uses the modern Playwright API:
`page.aria_snapshot()` is parsed into the normalized tree
(the deprecated `page.accessibility` API remains only as a
fallback for very old Playwright versions, and the DOM-derived
tree after that). Semantic matches in the 0.5–0.8 confidence
band are gated: acting on them requires human approval
(category `uncertain_target`). Uploads are verified after the
fact (the file input must show the uploaded file; a mismatch
is a structured failure), and dialogs a page raises are
recorded, redacted and dismissed by the driver and surfaced
in page observations. Tests: `tests/test_architecture_upgrade.py`.

## Operations: network, checkpoints, rate limits, profiles

- **Network awareness**: pages report their request health
  (failed requests, timeouts, blocked resources, HTTP 429s);
  `browser_network_status` summarizes it for diagnostics and
  recovery advice. Observation only: nothing can fire arbitrary
  network requests.
- **Task checkpointing**: with a `checkpoint_dir`, tasks persist
  redacted, checksum-protected checkpoints (goal, plan, state,
  recovery history, browser session) after every step.
  `resume_task` continues an interrupted task from its last
  checkpoint; completed steps are never executed twice, and a
  corrupted checkpoint is rejected, never trusted.
- **Rate-limit awareness**: rate-limit pages, bot-block signals
  and HTTP 429 responses stop actions with a structured
  `rate_limited` error; optional per-host pacing pauses instead
  of hammering, and `browser_rate_limit` takes one controlled
  backoff. There are no aggressive retries.
- **Approval records**: every sensitive-action decision
  (approved, denied, timed out, blocked) is recorded with its
  outcome; `browser_approvals` lists them, and a policy can set
  `approval_timeout_s` so a late answer never runs the action.
- **Browser profiles**: isolated profiles with separate cookies,
  storage and tabs (`browser_profiles`); one profile is active
  at a time, switching stashes the other profile's tabs, and
  profile records never hold credential-shaped preferences.

## Security: approval gate + secret hygiene

Irreversible browser actions are gated by a human, and secrets
stay out of every record:

- **Sensitive-action classification** (`afnan_ai/browser/security.py`)
  — purchases, payments, message/email sends, account changes,
  destructive clicks, form submits, file uploads and
  credential/payment-field entry are classified from the tool,
  the target element (its text, type and attributes) and the
  page before anything runs.
- **Configurable human approval** — an `ApprovalGate` with a
  `SecurityPolicy` decides: safe actions run; sensitive ones run
  only when a human approver says yes. With no approver
  configured they **do not run at all** — they fail with the
  structured `approval_required`/`approval_denied` browser
  errors, recorded like any other failure. Set the approver via
  `main.set_browser_approver(fn)` / `agent.set_browser_approver(fn)`,
  the policy via the `security_policy=` argument. A new
  `browser_upload_file` tool is gated the same way, and locators
  can target iframes (`frame` key; CSS pierces open shadow DOM).
- **Redaction** (`afnan_ai/redaction.py`) — password fields
  report `***` instead of their contents, and AgentState tool
  results, step/plan records, recovery context, approval
  requests and planner prompts pass through the redactor, so
  credentials, tokens, cookies and card numbers never land in
  state, logs or model prompts. Execution always uses the real
  values; only records are sanitized.
- **Stress-tested** — long multi-step tasks, failed clicks,
  stale elements, popups, session redirects, timeouts, failed
  downloads/uploads, auth-failure recovery, iframe/shadow-DOM,
  approval enforcement, secret scans and backend crashes are
  covered in `tests/test_browser_stress.py` and
  `tests/test_browser_security.py`, including an acceptance
  suite for the six guarantees (tasks complete; failures
  recover or terminate; no sensitive action without approval;
  no secret exposure; failures never crash the agent;
  Windows/Linux/macOS architecture intact).

## ScreenObserver: structured visual observation

For everything the DOM cannot see — canvas apps, remote
desktops, custom-drawn UI — the `ScreenObserver`
(`afnan_ai/screen/`) describes the actual screen:

- **Structured, never raw pixels** — observations carry screen
  dimensions, visible UI elements, regions (bounding boxes with
  centers) and **confidence scores**. The Planner and AgentState
  only ever see this structure; a screenshot file is saved only
  when explicitly requested.
- **DOM first, pixels as fallback** — browser observations fuse
  the BrowserController's DOM elements (confidence 1.0) with
  pixel-detected ones; when the DOM is missing, the visual
  elements (marked `source="visual"`) carry the page. Desktop
  captures reuse the agent's existing screenshot capture, and
  PNG decoding/detection is pure stdlib, so the layer has no
  platform or driver code of its own.
- **Never act on a guess** — `screen_assess_action` tiers every
  element: high confidence may proceed, medium must be
  confirmed by the Verifier (fresh screen observations feed the
  same observation-provider hook the browser layer uses), and
  low confidence requires human approval. The observer itself
  clicks nothing.
- Tools: `screen_observe`, `screen_find_elements`,
  `screen_assess_action` (disable with
  `enable_screen_tools=False`; access via
  `main.get_screen_observer()`). Screen-change detection makes
  stale refs fail with a structured `element_not_found` instead
  of acting on an old screen. Tests:
  `tests/test_screen_observer.py`.

## Accessibility, semantics, tabs, search and extraction

Five capabilities layered on the same BrowserController (no
second driver, no browser logic in the core agent):

- **Accessibility tree** (`browser_accessibility_tree`) — the
  browser's accessibility structure (Playwright's modern
  `page.aria_snapshot()`, parsed and normalized) into
  structured nodes (role, name, value, heading level); when a
  driver cannot provide one, an equivalent tree is derived
  from DOM interactive elements with live refs. Accessibility
  first, DOM second, pixels (ScreenObserver) last.
- **Semantic locator** (`browser_find_semantic`) — "Login
  button", "Search field", "Next page link": candidates are
  ranked by role fit + name similarity with confidence scores.
  Matches below 0.5 are returned with **no element reference**,
  so a low-confidence guess can never be acted on
  automatically; matches in the 0.5–0.8 band require human
  approval before an action runs on them.
- **Multi-tab task manager** — tabs carry task purposes
  (`browser_new_tab` with `purpose`, `browser_set_tab_purpose`,
  purposes shown by `browser_list_tabs` and recorded into
  AgentState), so parallel research lines never act on the
  wrong tab.
- **Web research** (`browser_search`, `browser_open_result`) —
  search DuckDuckGo/Google/Bing, get structured results
  (rank, title, URL, snippet, source; engine redirects
  unwrapped), then open a result by index into its own
  purpose-tagged tab with content attached.
- **Page content extraction** (`browser_extract_content`) —
  headings, paragraphs, lists, links and tables (row/column
  structure preserved) in clean normalized form: boilerplate
  filtered, large pages chunked with token estimates, and
  secrets redacted from text and URLs before anything reaches
  AgentState or the model. Tests:
  `tests/test_browser_accessibility.py` and
  `tests/test_browser_research.py` (30 browser tools total).

## SPA awareness, pagination, CAPTCHA, downloads and sessions

Five more advanced capabilities, same BrowserController and
Tool architecture (35 browser tools total):

- **JavaScript / SPA awareness** (`browser_wait_for_stable`) —
  a page probe (URL, title, `readyState`, text/element counts,
  content hash, detected React/Next.js/Vue/Angular) is polled
  until it stops changing, so client-side routing and async
  rendering are awaited by condition, never by blind sleeps.
- **Infinite scroll / pagination** (`browser_collect_items`) —
  scrolls feeds or clicks Next controls, deduplicates items,
  and always terminates: `max_items`, `max_pages`, an
  exhausted list, or a missing/stalled Next control.
- **CAPTCHA detection** (`browser_check_challenge`,
  `browser_wait_challenge`) — human checks are detected from
  URL/title/text/widget signals and reported as
  `human_required`; clicks and typing on a challenge page stop
  with the same structured error. A CAPTCHA is **never solved
  or bypassed by the agent**: with a challenge handler
  registered (`set_challenge_handler`), the human is asked,
  solves the check in the browser themselves, and the blocked
  action resumes automatically once the page clears; without
  one, the task pauses for the user, and recovery advice says
  exactly that. Interactive `main` registers a console prompt
  handler by default: the user solves the check in the browser
  window, presses Enter, and the task resumes.
- **Download manager** (`browser_downloads`) — downloads are
  tracked centrally (state, filename, type, destination,
  size); finished files are verified for existence and basic
  integrity (magic bytes). Executable payloads are flagged
  unsafe and need human approval; nothing is ever opened or
  executed automatically.
- **History / session manager** (`browser_session`) — every
  navigation is recorded (tab, purpose, timestamp, redacted
  URL); sessions can be inspected, saved to JSON and restored
  later so a new task recovers the previous task's browsing
  context. Tests: `tests/test_browser_advanced.py`.

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
