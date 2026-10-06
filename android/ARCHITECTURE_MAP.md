# Afnan AI — Android Productization: Architecture Map

Read-only repository-first audit of `~/workspace/afnan-refactor`
(Python project `Labbaik757/Afnan-Ai-1.2`).
No repo code was changed. All references are `file:line`.

---

## 1. Real application entry — the exact call chain

`python main.py` / `python3 main.py`:

| Step | Location | What happens |
|---|---|---|
| 1 | `main.py:1-24` | Thin wrapper; imports `AfnanAgent, create_agent` from `afnan_ai.agent` |
| 2 | `main.py:27` | `adapter = get_adapter()` — auto-selects Windows/macOS/Linux (`afnan_ai/platform/factory.py:13-17`; **no Android adapter exists**) |
| 3 | `main.py:28` | `_agent = AfnanAgent(adapter=adapter)` — the single agent instance |
| 4 | `main.py:323-327` | `start_afnan()` → `set_challenge_handler(console_challenge_handler)` → `_agent.start()` |
| 5 | `main.py:336-337` | `if __name__ == "__main__": start_afnan()` |
| 6 | `afnan_ai/agent.py:1767` | `AfnanAgent.start()` — wake-word loop: `create_detector(...)` (`afnan_ai/wakeword.py:94`) → `_wait_for_wake_local` → `listen_command` → `process_command` |
| 7 | `afnan_ai/agent.py:1623` | `process_command(command)` — legacy routing wrapper |
| 8 | `afnan_ai/agent.py:1562` | `handle_request(request)` — **the single entry point** for every user request: session-control shortcuts handled directly; everything else delegated to the central `Agent` orchestrator (`Agent.run()`: state → plan → execute → verify → recover/complete) |
| 9 | `main.py:1128` (agent.py) | `_build_agent_loop()` builds the real-time `AgentLoop` (observe→decide→validate→act→observe→verify) |

Key facts for Android:

- There is **no `AgentLoop` duplication point** — `handle_request()` is the narrow waist. An Android client must NOT reimplement this; it reaches it via the Control Plane (`START_TASK`/`execute_command`) or not at all.
- `AfnanAgent.start()` is a **blocking voice loop** (`while True`, mic + speaker). It is not a service API. Android must not call it; the PC runs it (or the Control Plane server), Android talks to the Control Plane.
- `afnan_ai/agent.py:53,123,1522,1819` — microphone input is `speech_recognition` + `PyAudio` (`sr.Microphone()`). Both are **desktop-only** (PyAudio needs PortAudio; no Android wheels).

---

## 2. Top-level packages under `afnan_ai/` — one-line purpose each

| Package / module | Purpose |
|---|---|
| `agent.py` | `AfnanAgent` — composition root; wires every subsystem; voice loop |
| `agent_loop.py` | Real-time autonomous AgentLoop (sole reasoning/orchestration authority) |
| `planner.py` | Strictly-validated Planner — plans only, never executes |
| `executor.py` | Executor — invokes registered tools, never plans |
| `verifier.py` | Deterministic Verifier — judges results, never re-executes |
| `recovery.py` | RecoveryManager — bounded replans, rejects blind repeats |
| `orchestrator.py` | `Agent` orchestrator connecting Planner/Executor/Verifier/Recovery |
| `state.py` | Central `AgentState` |
| `tools/` | `ToolRegistry` + 89 built-in tools; execution gateway with auth |
| `llm/` | `LLMProvider` abstraction (`base.py`), `factory.py` registry, `ollama.py` (lazy `import ollama`, HTTP to Ollama server) |
| `memory_store.py` | Persistent local memory (`~/.afnan-ai/`) |
| `goal_manager.py` | `GoalManager` — goal source of truth |
| `task_manager.py` | `TaskManager` — task lifecycle authority |
| `scheduler.py` | `TaskScheduler` (once/interval/daily/weekly) |
| `background_runner.py` | `BackgroundTaskRunner` — claim/execute/record, crash recovery |
| `checkpointing.py` | Checkpoint save/restore for long tasks |
| `browser/` | BrowserController → AfnanBrowserRuntime → ChromiumAdapter (CDP over stdlib WebSocket) or Playwright fallback; perception, security/approval gate, downloads, sessions |
| `computer/` | ComputerRuntime/Controller — desktop observe/act via `xdotool`/`wmctrl` (Linux), `osascript`/`cliclick` (macOS), PowerShell (Windows); `FileService` |
| `screen/` | Pure-stdlib ScreenObserver (pixel region detection, no Pillow) |
| `connectors/` | Subclassable `Connector` + `ConnectorRegistry`/`ConnectorService`; risk-graded, approval-gated |
| `artifacts/` | Versioned `ArtifactManager`, 8 `artifact_*` tools |
| `security/` | `SecurityCenter` (central auth authority), `CredentialVault` (central secret authority), PolicyEngine, hash-chained audit |
| `activity/` | `ActivityCenter` (event bus), `ApprovalCenter`, `AgentStatusTracker`, `NotificationService` |
| `control/` | Remote & multi-device Control Plane (see §4); `client/` = Python SDK |
| `context/` | Long-horizon context: tiered ContextManager, TrajectoryStore, EntityRegistry, FailureLearner |
| `research/` | Research & Evidence Intelligence: EvidenceGraph, citation validator |
| `evaluation/` | Evaluation & Benchmarking: versioned suites, regression detection |
| `skills/` | Production Skill system: manifests, trust levels, integrity HMAC |
| `subagents/` | SubAgentManager, role templates, verified handoffs |
| `proactive/` | ProactiveEngine — 6 evidence-based detectors, ideas/notifications |
| `workspace/` | Secure Workspace: 10-state lifecycle, FS isolation, snapshots, e-stop |
| `platform/` | OS adapters: `base.py`, `factory.py`, `linux.py`, `macos.py`, `windows.py` (**no `android.py`**) |
| `speech.py` | TTS chain: edge-tts (ur-PK neural) → gTTS (ur) → pyttsx3 → platform adapter |
| `wakeword.py` | `create_detector()` (`wakeword.py:94`): numpy JSON classifier (`wakeword_models/afnan.json`, stdlib+numpy) → custom ONNX → openWakeWord → cloud fallback |
| `config.py`, `persistence.py`, `redaction.py`, `log_config.py` | Config, JSON persistence, secret redaction, logging |

---

## 3. Dependency analysis — what is NOT Android-compatible

`requirements.txt` (full contents read):

| Dependency | Android verdict |
|---|---|
| `SpeechRecognition==3.16.0` | **No** — desktop mic via PyAudio; Android must use native `SpeechRecognizer` / ML Kit |
| `PyAudio==0.2.14` | **No** — needs PortAudio C build; no Android wheels |
| `PyAutoGUI==0.9.54` | **No** — desktop input automation; meaningless on Android |
| `pywhatkit==5.4` | **No** — desktop automation helpers |
| `ollama==0.6.1` | **Bridge only** — it's an HTTP client; the Ollama *server* + models need GBs of RAM/GPU. Android points at the PC's Ollama over LAN, or a cloud provider via `llm/factory.py` |
| `playwright>=1.40` | **No** — downloads desktop Chromium binaries; Android uses the Control Plane's PC browser instead |
| `pyttsx3==2.99` | **No** — needs SAPI/NSSpeech/espeak OS engines; Android uses native Android TTS (Urdu voice available) |
| `pywin32`, `pyobjc-*` | Platform-gated already; irrelevant on Android |

Native/binary deps found by import scan:

- `afnan_ai/speech.py:85` — `from gtts import gTTS` (lazy; **not in requirements.txt** — network MP3 API, works anywhere with internet but not vendored)
- `afnan_ai/speech.py:62` — edge-tts via subprocess (`_speak_with_edge_tts`); **not in requirements.txt** either
- `afnan_ai/wakeword.py:85` — `import numpy as np` (lazy, only for the JSON classifier path); `wakeword_models/afnan.json` is numpy-based. NumPy *has* Android builds via Chaquopy but adds ~15 MB; the model could alternatively be ported to TFLite
- `afnan_ai/wakeword.py:68` — `openwakeword` (optional, pip-only, needs onnxruntime — **not Android-viable**)
- `afnan_ai/browser/backend.py:146` — `playwright.sync_api` (lazy import; desktop only)
- `afnan_ai/browser/chromium_adapter.py` — **stdlib-only CDP WebSocket** (no dependency!), but it still needs a real Chromium *binary*, which Android doesn't ship
- `afnan_ai/computer/command_backend.py:89-120` — shells out to `xdotool`/`wmctrl`/`osascript`/`cliclick`/PowerShell: **PC-only by nature**

**Portable core (pure stdlib, runs anywhere Python runs):**
planner, executor, verifier, recovery, orchestrator, state, tools/registry,
memory_store, goal_manager, task_manager, scheduler, background_runner,
checkpointing, security/* (vault/audit/policy), activity/*, control/*,
context/*, artifacts/* (logic), research/*, evaluation/*, skills/*,
subagents/*, proactive/*, workspace/* (logic; FS paths need an Android adapter).

---

## 4. Control Plane wire protocol — endpoint table for a Kotlin client

Base: `http(s)://<pc-host>:<port>` · protocol version `v1` · JSON everywhere.
Auth: `Authorization: Bearer <token>` header. TLS in production
(TLS 1.2 min); plaintext only with `allow_insecure=True` on loopback.
Every error JSON carries `X-Request-Id` (header + body `request_id`).

| # | Method & path | Auth | Request body | Success response | Errors |
|---|---|---|---|---|---|
| 1 | `GET /v1/health` | none | — | `200 {"ok":true,"protocol_version":"v1","uptime_s","devices","sessions_live","emergency_tripped"}` | — |
| 2 | `POST /v1/pair/request` | none (rate-limited 10/min/IP) | `{"device_id" (required), "device_name", "platform", "platform_version", "client_version", "capabilities":[]}` | `200 {"pairing_id","code"}` — 6-digit code, **shown once**, hash-only server-side | `400` bad metadata / revoked device; `429` rate limit |
| 3 | `POST /v1/pair/redeem` | none (rate-limited) | `{"pairing_id","code"}` | `200 {"device_id","session_id","token","capabilities":[],"expires_at"}` | `400` wrong/expired/replayed/locked code; `404` unknown pairing |
| 4 | `POST /v1/commands` | Bearer (rate-limited 240/min/IP) | `RemoteCommand`: `{"command_id","command_type","payload":{},"idempotency_key","created_at","expires_at","correlation_id","protocol_version":"v1"}` | `200 RemoteCommandResult {"command_id","status","result":{},"error":"","error_code":"","verification":{},"completed_at","correlation_id","protocol_version"}` — note: `status` may be `denied`/`failed` with HTTP 200; check it | `401` bad/expired/revoked token; `403` forbidden; `400` malformed/unknown command |
| 5 | `POST /v1/session/heartbeat` | Bearer | `{}` | `200 {"session_id","state","at"}` | `401` |
| 6 | `POST /v1/session/rotate` | Bearer | `{"token":"<old token>"}` | `200 {"token":"<new>","session_id"}` — old token dies immediately, session id stable | `401` |
| 7 | `GET /v1/events` | Bearer | query `?cursor=<seq>` (optional) | `200 text/event-stream`: `data: {event}\n\n` frames; `: keep-alive` comments; explicit `stream.gap` event + snapshot on history loss | `401` |
| 8 | `GET /v1/stream` | Bearer header preferred; `?token=` compatibility fallback (never logged) | WS upgrade (`Sec-WebSocket-Version: 13`) | `101 Switching Protocols` | `401`; `426` wrong upgrade; protocol violations → close `1002`/`1007`/`1009`; revocation → close `1008` |

**WebSocket frames (RFC 6455; client frames MUST be masked):**

- Client → server: `{"type":"heartbeat"}` → server `{"type":"heartbeat_ack"}`; or a `RemoteCommand` body → server `{"type":"command_result","result":{RemoteCommandResult}}`
- Server → client: `{"type":"event","event":{ControlEvent}}`, `{"type":"error","error":"<safe message>"}`
- `ControlEvent`: `{"event_id","category","seq","at","device_id","session_id","correlation_id","data":{sanitized},"protocol_version":"v1"}` — monotonically increasing `seq` per session buffer; never contains secrets or chain-of-thought
- `command_type` values are lowercase strings (`afnan_ai/control/models.py:152-...`): `get_agent_status`, `list_tasks`, `get_task`, `start_task`, `pause_task`, `resume_task`, `cancel_task`, `retry_task`, `list_goals`, `get_goal`, `pause_goal`, `resume_goal`, `cancel_goal`, `list_artifacts`, `get_artifact`, `list_approvals`, `get_approval`, `approve_action`, `deny_action`, `list_devices`, `get_device`, `list_sessions`, `query_activity`, `get_metrics`, `health_check`, `browser_navigate`, `browser_screenshot`, `browser_state`, `computer_observe`, `computer_act`, `list_schedules`, `manage_schedule`, `emergency_stop`, `heartbeat`, `revoke_device`, `revoke_session`, `rotate_credentials`
- Status codes: `401` auth · `403` authz · `404` missing · `405` bad method · `400` validation · `409` conflict · `429` rate limit · `500` never leaks internals

---

## 5. What is inherently PC/server-side vs portable

**MUST run on the PC (or a server) — cannot move to the phone:**

| Subsystem | Why it can't run on Android |
|---|---|
| Ollama LLM (`afnan_ai/llm/ollama.py`) | Model weights need GBs of RAM/compute; `ollama` package is only an HTTP client — the *server* stays on PC |
| Chromium browser (`browser/chromium_adapter.py`, `backend.py`) | Needs a desktop Chromium binary + CDP; Playwright downloads desktop builds. Android has WebView, not CDP-controllable Chromium |
| ComputerRuntime (`computer/command_backend.py`) | `xdotool`/`wmctrl`/`osascript`/PowerShell are desktop OS automation; meaningless on Android |
| PyAudio + SpeechRecognition mic (`agent.py:1522,1819`) | No Android wheels; Android has its own audio APIs |
| pyttsx3 / edge-tts / gTTS (`speech.py`) | OS speech engines / network MP3 hacks; Android has native TTS with Urdu voices |
| `AfnanAgent.start()` voice loop (`agent.py:1767`) | Blocking mic/speaker loop designed for a PC microphone |
| Wake-word continuous listening | Android Doze/background restrictions kill permanent mic loops; needs foreground service or push-to-talk |

**CAN run on Android (pure-stdlib portable logic):**
The entire protocol/client layer (`control/client/`), plus — *if* an embedded Python (Chaquopy) is used — planner/executor/verifier/tools/memory/goals/tasks/scheduler/security/activity/artifacts logic. But running the full agent on-device is pointless without an LLM: the LLM call is the brain, and it lives on the PC (or approved cloud).

**Needs an Android adapter (interface preserved, implementation swapped):**
`platform/` (new `android.py` adapter), voice input/output (Android SpeechRecognizer + Android TTS), secure storage (Android Keystore instead of `~/.afnan-ai/` files), notifications (FCM / local), wake word (TFLite port of `afnan.json` or push-to-talk), file/workspace access (scoped storage).

---

## 6. Python SDK — the protocol reference for the Kotlin port

`afnan_ai/control/client/` (9 files, stdlib-only). The Kotlin client should mirror this 1:1:

| Class | File | Responsibility |
|---|---|---|
| `ControlClient` | `base.py:39` | `base_url`/`timeout_s`; `request()`/`get()`/`post()` via urllib; maps 401→`AuthenticationError`, 403→`AuthorizationError`, 400/404/409→`ProtocolError`, transport failures→`ConnectionError`; `redact_token_text()`; `close()` |
| `PairingClient` | `pairing.py:24` | `request_pairing(metadata) -> (pairing_id, code)`; `redeem(pairing_id, code) -> {device_id, session_id, token, capabilities, expires_at}` |
| `SessionClient` | `session.py:16` | In-memory token custody (`set_token`/`clear`); `heartbeat()`; `rotate_token()` |
| `CommandClient` | `commands.py:44` | `execute(token, command_type, payload, idempotency_key, timeout_s)` — validates against real `CommandType` enum, auto idempotency keys; `cancel(token, task_id)` |
| `EventClient` | `events.py:139` | `stream_sse(token, cursor, on_event, timeout_s)` generator; `connect_websocket(...)` → `WebSocketConnection` |
| `WebSocketConnection` | `events.py:295` | Raw-socket WS: masked client frames, client-side frame decoder (`_decode_server_frame`, `events.py:80`), ping/pong, close handshake, daemon reader thread |
| `ReconnectManager` | `reconnect.py:26` | Exponential backoff (1s→60s, jitter), `run(fn)`, stops on auth failure, pluggable cursor store |
| `RemoteStateStore` | `state.py:49` | Sanitized in-memory snapshot (`snapshot()` → `{"protocol_version":"v1","agent","tasks"}`); `sanitize()` drops secret-looking keys |
| `ProtocolError` + subclasses | `errors.py:10` | `code` attribute; `AuthenticationError`, `AuthorizationError`, `ConnectionError` |

---

## 7. Pairing approval CLI — gap confirmed

**There is NO owner-side pairing approval CLI or UI in the repo.**

- Backend exists: `ControlPlane.pair_approve(pairing_id, approved_by=...)` (`afnan_ai/control/plane.py:238`), `PairingManager.approve()` (`afnan_ai/control/pairing.py:116`), and `clear_device_revocation()` (`plane.py:309`)
- There is **no HTTP endpoint** for approval (only request/redeem are exposed in `transport.py:391-393`) and **no CLI command** anywhere (`grep` over `afnan_ai/` finds only the API and tests)
- The prompt's §19 (PC approval UI) is therefore **new work**, not a reuse: build a small owner-side tool (CLI and/or tray/desktop prompt) that lists pending `PairingRequest`s (id, device name, platform, requested capabilities, expiry) and calls `pair_approve`/`pair_reject`. It must run on the PC next to the agent, reusing `ControlPlane` directly — no new pairing logic

---

## 8. Forced architecture (what the audit dictates)

1. **The agent brain stays on the PC.** Ollama, Chromium/CDP, xdotool-class automation, and the mic/speaker loop are PC-only by nature. The APK is a *remote client*, not a port of the agent.
2. **The Control Plane is the phone↔PC bridge** — already production-hardened (TLS, RFC 6455, pairing, revocation, audit). The Kotlin app implements §4's endpoint table; `control/client/` is the line-by-line reference.
3. **No Python runtime needed on Android at all** if the app is a pure remote client (recommended): 100% Kotlin, no Chaquopy weight, no dependency risk. Embedded Python only buys offline logic with no LLM — not worth it.
4. **Voice = Android-native**: `SpeechRecognizer` (Urdu `ur-PK`) for input, Android TTS (Urdu voice) for output. `speech.py`/`wakeword.py` desktop chains are not reused; the *Urdu-first voice contract* is.
5. **Browser on Android = remote view**: `browser_navigate`/`browser_screenshot`/`browser_state` commands drive the PC's Chromium; the app renders screenshots + state. Never WebView-pretend.
6. **Computer control = remote only**: `computer_observe`/`computer_act` go to the PC's ComputerRuntime behind its approval gate. Nothing executes on the phone.
7. **Approvals flow through `ApprovalCenter`**: Android shows `list_approvals` → `approve_action`/`deny_action` with action-bound context; notifications deep-link to the approval screen.
8. **Pairing**: QR or manual 6-digit code → `POST /v1/pair/request` → **new PC owner-approval tool required** (§7 gap) → `POST /v1/pair/redeem` → token in **Android Keystore**, rotation via `/v1/session/rotate`.
9. **LLM stays behind `LLMProvider`**: Android never bundles a model; the PC's Ollama serves it. A cloud provider is only added via `llm/factory.py` with explicit user approval.
10. **Background work**: persistent SSE/WS with `ReconnectManager` semantics (backoff, cursor resume); FCM for approval/task notifications; no permanent mic loop (Doze).
11. **New code allowed**: Kotlin UI, `platform/android.py` adapter (only if any Python ships), PC pairing-approval tool, Android Keystore/WorkManager/notification glue. Everything else is reuse.
