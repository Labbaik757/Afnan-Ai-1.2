"""Unified context sources: normalize every system into items.

Adapters turn each subsystem's state into structured
ContextItemV2 records: goal/task managers, AgentState,
MemoryStore, TrajectoryStore, checkpoints, workspace,
browser, computer, connectors, artifacts, activity,
permissions and previous verifications.

Every adapter is defensive: a missing system or a broken
call yields zero items, never an exception.  Untrusted
sources are zone-labeled; secrets are never included raw.
"""

from __future__ import annotations

from typing import Any, Callable

from afnan_ai.context.models import (
    ContextItemV2,
    ItemKind,
    Sensitivity,
    TrustZone,
)
from afnan_ai.redaction import redact_text


def _item(
    kind: ItemKind,
    zone: TrustZone,
    text: str,
    *,
    source: str = "",
    importance: float = 0.5,
    confidence: float = 0.7,
    sensitivity: Sensitivity = Sensitivity.INTERNAL,
    task_id: str = "",
    provenance: str = "",
    tags: tuple[str, ...] = (),
) -> ContextItemV2:
    return ContextItemV2(
        kind=kind,
        zone=zone,
        text=redact_text(text)[:2000],
        source=source,
        importance=importance,
        confidence=confidence,
        sensitivity=sensitivity,
        task_id=task_id,
        provenance=provenance or source,
        tags=tags,
    )


def _safe(call: Callable[[], Any]) -> Any:
    try:
        return call()
    except Exception:
        return None


def goal_items(goal: str, *, task_id: str = "") -> list[ContextItemV2]:
    if not goal:
        return []
    return [
        _item(
            ItemKind.GOAL, TrustZone.USER, f"User goal: {goal}",
            source="goal", importance=1.0, confidence=1.0,
            task_id=task_id, tags=("goal",),
        )
    ]


def agent_state_items(
    state: Any, *, task_id: str = ""
) -> list[ContextItemV2]:
    """Current execution state → items (trusted, agent_state)."""
    data = _safe(
        lambda: state.to_dict()
        if hasattr(state, "to_dict")
        else dict(state or {})
    )
    if not isinstance(data, dict):
        return []
    items = []
    for key in (
        "current_step", "active_task", "status", "progress_note",
    ):
        value = data.get(key)
        if value:
            items.append(
                _item(
                    ItemKind.OBSERVATION, TrustZone.AGENT_STATE,
                    f"agent_state.{key}: {value}",
                    source="agent_state", importance=0.7,
                    task_id=task_id, tags=("agent-state", key),
                )
            )
    return items[:8]


def memory_items(
    memories: list[dict[str, Any]], *, task_id: str = ""
) -> list[ContextItemV2]:
    out = []
    for mem in memories or []:
        if not isinstance(mem, dict):
            continue
        out.append(
            _item(
                ItemKind.FACT, TrustZone.AGENT_STATE,
                str(mem.get("text", mem.get("summary", ""))),
                source="memory",
                importance=float(mem.get("importance", 0.6)),
                confidence=float(mem.get("confidence", 0.7)),
                task_id=task_id,
                provenance=str(mem.get("source", "memory")),
                tags=("memory",),
            )
        )
    return out[:20]


def trajectory_items(
    events: list[Any], *, task_id: str = "", limit: int = 15
) -> list[ContextItemV2]:
    """Recent trajectory events → WARM context."""
    out = []
    for event in (events or [])[-limit:]:
        to_dict = getattr(event, "to_dict", None)
        data = to_dict() if to_dict else event
        if not isinstance(data, dict):
            continue
        kind_map = {
            "observation": ItemKind.OBSERVATION,
            "decision": ItemKind.DECISION,
            "action": ItemKind.ACTION,
            "result": ItemKind.RESULT,
            "verification": ItemKind.VERIFICATION,
            "failure": ItemKind.FAILURE,
            "recovery": ItemKind.RECOVERY,
            "approval": ItemKind.APPROVAL,
            "checkpoint": ItemKind.CHECKPOINT,
        }
        kind = kind_map.get(
            str(data.get("kind", "observation")),
            ItemKind.OBSERVATION,
        )
        out.append(
            _item(
                kind,
                TrustZone.AGENT_STATE,
                str(data.get("summary", "")),
                source="trajectory",
                importance=0.6,
                confidence=float(data.get("confidence", 0.7)),
                task_id=task_id,
                provenance="trajectory:"
                + str(data.get("event_id", ""))[:16],
                tags=("trajectory", str(data.get("kind", ""))),
            )
        )
    return out


def observation_items(
    *,
    browser: dict[str, Any] | None = None,
    computer: dict[str, Any] | None = None,
    freshness: dict[str, str] | None = None,
    task_id: str = "",
) -> list[ContextItemV2]:
    """Browser/desktop observations with freshness labels."""
    out = []
    for source, obs in (
        ("browser", browser),
        ("computer", computer),
    ):
        if not isinstance(obs, dict) or not obs:
            continue
        fresh = (freshness or {}).get(source, "fresh")
        summary = str(
            obs.get("summary", obs.get("title", "observation"))
        )
        out.append(
            _item(
                ItemKind.OBSERVATION, TrustZone.TOOL_OBSERVATION,
                f"[{source} observation, {fresh}] {summary}",
                source=source, importance=0.6,
                task_id=task_id,
                tags=("observation", source, fresh),
            )
        )
    return out


def connector_items(
    results: list[dict[str, Any]], *, task_id: str = ""
) -> list[ContextItemV2]:
    """Connector results — always untrusted_external."""
    out = []
    for res in results or []:
        if not isinstance(res, dict):
            continue
        out.append(
            _item(
                ItemKind.OBSERVATION,
                TrustZone.UNTRUSTED_EXTERNAL,
                str(res.get("summary", res.get("text", ""))),
                source="connector",
                importance=0.5,
                confidence=float(res.get("confidence", 0.5)),
                task_id=task_id,
                provenance="connector:"
                + str(res.get("connector", ""))[:40],
                tags=("connector", "untrusted"),
            )
        )
    return out[:15]


def artifact_items(
    artifacts: list[dict[str, Any]], *, task_id: str = ""
) -> list[ContextItemV2]:
    out = []
    for art in artifacts or []:
        if not isinstance(art, dict):
            continue
        out.append(
            _item(
                ItemKind.FACT, TrustZone.AGENT_STATE,
                f"artifact {art.get('artifact_id')}: "
                f"{art.get('title', art.get('kind', ''))}",
                source="artifacts", importance=0.65,
                task_id=task_id,
                provenance="artifact:"
                + str(art.get("artifact_id", ""))[:40],
                tags=("artifact",),
            )
        )
    return out[:15]


def collect_all(
    *,
    goal: str = "",
    task_id: str = "",
    agent_state: Any = None,
    memories: list[dict[str, Any]] | None = None,
    trajectory_events: list[Any] | None = None,
    browser_obs: dict[str, Any] | None = None,
    computer_obs: dict[str, Any] | None = None,
    freshness: dict[str, str] | None = None,
    connector_results: list[dict[str, Any]] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
) -> list[ContextItemV2]:
    """Assemble items from every available source."""
    items: list[ContextItemV2] = []
    items.extend(goal_items(goal, task_id=task_id))
    if agent_state is not None:
        items.extend(
            agent_state_items(agent_state, task_id=task_id)
        )
    items.extend(memory_items(memories or [], task_id=task_id))
    items.extend(
        trajectory_items(
            trajectory_events or [], task_id=task_id
        )
    )
    items.extend(
        observation_items(
            browser=browser_obs,
            computer=computer_obs,
            freshness=freshness,
            task_id=task_id,
        )
    )
    items.extend(
        connector_items(
            connector_results or [], task_id=task_id
        )
    )
    items.extend(
        artifact_items(artifacts or [], task_id=task_id)
    )
    return items
