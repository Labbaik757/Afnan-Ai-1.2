"""Sanitized remote views.

Remote clients never see raw ``AgentState`` or other internal
objects.  These views project runtime state into small, stable,
explicitly-permitted shapes.  When ``AgentState`` evolves, the views
stay the same — accidental leakage is impossible by construction.
"""

from __future__ import annotations

from typing import Any


def agent_view(
    *,
    status: str,
    current_task: dict[str, Any] | None = None,
    current_goal: dict[str, Any] | None = None,
    waiting_for_approval: bool = False,
    pending_approvals: int = 0,
    error: str | None = None,
    connected_devices: int = 0,
    active_sessions: int = 0,
    emergency_tripped: bool = False,
    runtime_version: str = "",
) -> dict[str, Any]:
    """Concise, action-oriented agent snapshot.

    Deliberately excludes model reasoning, observations and tool
    outputs — only operational state crosses the boundary.
    """
    return {
        "status": status,
        "current_task": current_task,
        "current_goal": current_goal,
        "waiting_for_approval": waiting_for_approval,
        "pending_approvals": pending_approvals,
        "error": (error or "")[:300],
        "connected_devices": connected_devices,
        "active_sessions": active_sessions,
        "emergency_tripped": emergency_tripped,
        "runtime_version": runtime_version,
    }


def task_view(task) -> dict[str, Any]:
    """Sanitized projection of a ManagedTask."""
    get = lambda name, default=None: getattr(task, name, default)  # noqa: E731
    return {
        "task_id": get("task_id", ""),
        "goal": get("goal_text", "") or get("goal", ""),
        "status": get("status", ""),
        "priority": get("priority", 0),
        "attempts": get("attempts", 0),
        "max_retries": get("max_retries", 0),
        "created_at": get("created_at", 0.0),
        "updated_at": get("updated_at", 0.0),
        "summary": (get("summary", "") or "")[:500],
        "error": (get("error", "") or "")[:500],
        "has_checkpoint": bool(get("checkpoint_ref")),
        "goal_id": get("goal_id", ""),
    }


def goal_view(goal) -> dict[str, Any]:
    """Sanitized projection of a ManagedGoal."""
    get = lambda name, default=None: getattr(goal, name, default)  # noqa: E731
    milestones = get("milestones", []) or []
    return {
        "goal_id": get("goal_id", ""),
        "description": (get("description", "") or "")[:500],
        "status": get("status", ""),
        "priority": get("priority", 0),
        "progress": get("progress", 0.0),
        "milestones": [
            {
                "title": str(m.get("title", ""))[:200],
                "completed": bool(m.get("completed", False)),
            }
            for m in milestones
            if isinstance(m, dict)
        ][:50],
        "created_at": get("created_at", 0.0),
        "updated_at": get("updated_at", 0.0),
    }


def approval_view(approval) -> dict[str, Any]:
    """Sanitized projection of a TrackedApproval."""
    get = lambda name, default=None: getattr(approval, name, default)  # noqa: E731
    return {
        "approval_id": get("approval_id", ""),
        "task_id": get("task_id", ""),
        "action": get("action", ""),
        "reason": (get("reason", "") or "")[:500],
        "target": get("target", ""),
        "risk_level": get("risk_level", ""),
        "status": get("status", ""),
        "requested_capability": get("requested_capability", ""),
        "scope": get("scope", ""),
        "expires_at": get("expires_at", 0.0),
        "is_expired": bool(get("is_expired", lambda: False)()),
        "created_at": get("created_at", 0.0),
    }


def device_view(device) -> dict[str, Any]:
    """Sanitized projection of a DeviceInfo (no secret hash)."""
    if hasattr(device, "to_dict"):
        return device.to_dict()
    return {
        "device_id": getattr(device, "device_id", ""),
        "device_name": getattr(device, "device_name", ""),
        "platform": getattr(device, "platform", ""),
        "trust": str(getattr(device, "trust", "")),
        "connection_state": str(
            getattr(device, "connection_state", "")
        ),
        "last_seen": getattr(device, "last_seen", 0.0),
    }


def session_view(session) -> dict[str, Any]:
    """Sanitized projection of a ClientSession (no token material)."""
    if hasattr(session, "to_dict"):
        return session.to_dict()
    return {
        "session_id": getattr(session, "session_id", ""),
        "device_id": getattr(session, "device_id", ""),
        "state": str(getattr(session, "state", "")),
        "principal": getattr(session, "principal", ""),
        "created_at": getattr(session, "created_at", 0.0),
        "last_activity": getattr(session, "last_activity", 0.0),
        "expires_at": getattr(session, "expires_at", 0.0),
    }


def artifact_view(artifact) -> dict[str, Any]:
    """Sanitized projection of an artifact record."""
    get = lambda name, default=None: getattr(artifact, name, default)  # noqa: E731
    return {
        "artifact_id": get("artifact_id", ""),
        "name": get("name", ""),
        "artifact_type": get("artifact_type", ""),
        "status": str(get("status", "")),
        "current_version": get("current_version", 0),
        "project_id": get("project_id", ""),
        "sensitive": bool(get("sensitive", False)),
        "created_at": get("created_at", ""),
        "updated_at": get("updated_at", ""),
    }
