# Afnan AI — Remote & Multi-Device Control Plane

Production-grade remote control for the Afnan AI agent runtime.
Authorized clients and devices (local device, remote computer,
mobile, desktop, web, future voice) can securely monitor and control
running agents, tasks, sessions, approvals, artifacts and activity —
without a second agent implementation.

## Architecture

```
Local Agent Runtime
        |
Secure Control Plane          (afnan_ai/control/)
        |
Authenticated Client Sessions
        |
Device / Session Registry
        |
Tasks / Goals / Approvals / Activity
        |
Secure Commands
        |
Existing Agent Runtime
```

The control plane is **not** an autonomous reasoning engine.  Every
remote command flows through one pipeline:

```
REMOTE CLIENT -> AUTHENTICATE -> AUTHORIZE -> SEND COMMAND
    -> VALIDATE -> IDEMPOTENCY -> RATE LIMIT
    -> EXISTING RUNTIME API -> VERIFY
    -> EVENT / RESULT -> CLIENT -> AUDIT
```

### Reuse map (nothing duplicated)

| Control-plane piece | Existing component reused |
|---|---|
| Authorization | `SecurityCenter` / `PermissionManager` / `CapabilityManager` |
| Secrets | `CredentialVault` (hashes/refs only, never plaintext) |
| Audit | `SecurityAuditCenter` (hash-chained, tamper-evident) |
| Rate limiting | `SecurityCenter.rate_limiter` |
| Emergency stop | `SecurityCenter.emergency` (`trip_emergency`) |
| Tasks | `TaskManager` (source of truth) |
| Goals | `GoalManager` (source of truth) |
| Schedules | `TaskScheduler` |
| Checkpoints | `CheckpointManager` (via TaskManager refs) |
| Activity events | `ActivityCenter` (`subscribe` + `sync` replay) |
| Approvals | `ApprovalCenter` (`request`/`decide`, action-bound, expiring) |
| Artifacts | `ArtifactManager` |
| Browser ops | `BrowserController` (never raw engine APIs) |
| Computer ops | `ComputerRuntime.execute()` / `ComputerController.act()` |
| Monitoring views | `AgentState.summary()`-style sanitized projections |

### Module layout

| Module | Purpose |
|---|---|
| `models.py` | Protocol models: sessions, devices, commands, results, events |
| `capabilities.py` | `remote.*` capability definitions (registered, least-privilege defaults) |
| `devices.py` | `DeviceRegistry` — persistent identity, trust, revocation |
| `sessions.py` | `ClientSessionManager` — strict session state machine |
| `auth.py` | `AuthProvider` — vault-backed tokens, rotation, revocation |
| `pairing.py` | `PairingManager` — one-time short-lived pairing codes |
| `idempotency.py` | `IdempotencyStore` — replay-resistant commands |
| `views.py` | Sanitized remote projections (never raw `AgentState`) |
| `commands.py` | `RemoteCommandRouter` — validate/authorize/dispatch/verify/audit |
| `events.py` | `EventStream` — live delivery + replay cursors over ActivityCenter |
| `approvals.py` | `ApprovalGateway` — remote approval transport |
| `audit.py` | `ControlPlaneAuditAdapter` — remote actions into the audit chain |
| `metrics.py` | `MetricsCollector` — observability |
| `transport.py` | Stdlib HTTP + SSE + WebSocket server, TLS / dev-only insecure |
| `plane.py` | `ControlPlane` — wiring facade |
| `server.py` | `ControlPlaneServer` — lifecycle (`from_agent`) |

## Device pairing

1. New device calls `POST /v1/pair/request` with its metadata.
   The plane registers the device (`trust: pending`) and returns a
   one-time 6-digit code (shown once; only its hash is stored).
2. The user verifies the request out-of-band and approves it
   (owner action).
3. The device calls `POST /v1/pair/redeem` with the code.  On success
   the device becomes `paired`, receives the least-privilege
   monitoring capability set, and gets a session + bearer token.

Codes expire after 5 minutes, are single-use, non-replayable and
rate-limited.  Too many wrong attempts reject the request.

## Authentication

- Bearer tokens (`afn_...`), issued per session, stored in the
  `CredentialVault`; the plane keeps only SHA-256 hashes.
- Tokens expire (default 12h), rotate (`POST /v1/session/rotate`
  revokes the old token immediately) and revoke (per token, per
  session, per device).
- No long-lived static master token.  Device trust is re-checked on
  every authentication.  LAN membership grants no trust.

## Authorization

Capability-based, through the existing `PermissionManager`.
Newly paired devices get monitoring only:

`remote.status.read`, `remote.tasks.read`, `remote.goals.read`,
`remote.activity.read`, `remote.artifacts.read`,
`remote.devices.read`, `remote.metrics.read`

Mutating capabilities (`remote.tasks.control`, `remote.approvals.decide`,
`remote.browser.control`, `remote.computer.control`,
`remote.safety.emergency_stop`, `remote.admin`, ...) are granted
explicitly and audited.  Admin is never granted by default.

## Command lifecycle

`RemoteCommand` carries `command_id`, `session_id`, `device_id`,
`command_type`, `target`, `payload`, `requested_capability`,
`created_at`, `expires_at`, `idempotency_key`, `correlation_id` and
`protocol_version` (`v1`).

- **Validation:** unknown types, malformed ids, oversized payloads
  and expired commands are rejected before anything else.
- **Idempotency:** retries with the same key return the stored
  result (`status: duplicate`) instead of executing twice.
- **Offline policy:** read-only commands are `safe_queueable`;
  mutating/sensitive commands require a live session;
  approvals, revocations, emergency stop and computer actions are
  `never_queueable` — they are never silently executed later.
- **Conflict policy:** the existing managers are authoritative.
  `TaskManager` validates every state transition, so conflicting
  pause/cancel/approve commands resolve deterministically against
  real task state instead of racing.

## Event stream

- `GET /v1/events` — Server-Sent Events.
- `GET /v1/stream` — WebSocket (token via `?token=`).
- Categories: `agent.status`, `task.*`, `approval.requested/resolved`,
  `browser.state_changed`, `computer.state_changed`,
  `artifact.created/updated`, `security.alert`, `device.connected/
  disconnected`, `checkpoint.created`, `research.updated`,
  `evaluation.completed`, `session.revoked`, `command.completed`.
- Reconnect: the client sends its last cursor; missed events replay
  from `ActivityCenter.sync(after_seq=...)`, then the live stream
  resumes.  Duplicates are safe to ignore (event ids + cursors).
- Backpressure: per-session bounded buffers; overflow drops oldest
  with a `stream.gap` marker — a slow client never stalls the agent.

## Approvals

When the agent needs approval, `ApprovalCenter` creates the request
(action-bound, expiring, single-decision) and the gateway pushes it
to remote sessions.  The remote decision flows back through
`ApprovalCenter.decide()` — the same enforcement as local approvals.
A stale approval cannot authorize a different action, and generic
"approve everything" does not exist.

## Emergency stop

`remote.safety.emergency_stop` capability required.  Trips the
existing `SecurityCenter.emergency`: idempotent, audited, propagated
to halt callbacks immediately.  A lost remote connection never
disables the local stop.

## Remote browser / computer control

Routed exclusively through `BrowserController` and
`ComputerRuntime.execute()` (observe → resolve → validate →
permission → execute → re-observe → verify).  Remote navigation
re-checks the existing browser URL resource policy; screenshots stay
server-side and are referenced, not embedded.  No raw process or OS
API is exposed.

## State snapshots

`views.py` projects `RemoteAgentView`, `RemoteTaskView`,
`RemoteGoalView`, `RemoteApprovalView`, `RemoteDeviceView`.
Internal `AgentState` is never serialized.  Model reasoning,
observations and secrets never cross the boundary.

## Transport security

- TLS when a certificate/key is configured.
- Plaintext only with `allow_insecure=True` (explicit development
  opt-in) and only on loopback; non-loopback insecure bind is
  refused.
- Credentials, vault secrets and session secrets are never
  transmitted or logged.

## Deployment

```python
from afnan_ai.control import ControlPlaneServer

server = ControlPlaneServer.from_agent(
    agent,
    host="127.0.0.1",
    port=8765,
    tls_cert="/path/to/cert.pem",   # production
    tls_key="/path/to/key.pem",
    allow_insecure=True,            # dev only, loopback only
    device_path="~/.afnan-ai/control/devices.json",
)
server.start()   # plane maintenance + transport
# ...
server.stop()
```

Without `from_agent`, construct `ControlPlane(...)` directly and
inject the components you have; unwired commands report `NOT_WIRED`
instead of failing silently.  Local-only operation is unaffected —
the control plane is an optional deployment component.

## Client integration

1. `POST /v1/pair/request` → `{pairing_id, code}` (show code to user)
2. User approves (owner UI / CLI)
3. `POST /v1/pair/redeem` → `{device_id, session_id, token, capabilities}`
4. `POST /v1/commands` with `Authorization: Bearer <token>` and a
   `RemoteCommand` body → `RemoteCommandResult`
5. `GET /v1/events` (SSE) or `GET /v1/stream?token=...` (WebSocket)
   for real-time events; reconnect with the last cursor.
6. `POST /v1/session/heartbeat` to keep the session warm;
   `POST /v1/session/rotate` to rotate the token.

## Testing

`tests/test_control_plane.py` covers: device identity, session state
machine, command validation, authorization, idempotency, event
ordering/cursors, conflict handling, authentication (valid/invalid/
expired/revoked/replayed), pairing expiry/replay/brute-force,
authorization denials and escalation attempts, reconnection and
event replay, approval binding/expiry/replay, emergency stop,
multi-device coexistence and revocation.  Security and routing logic
execute for real — only the network transport is substituted in
tests.
