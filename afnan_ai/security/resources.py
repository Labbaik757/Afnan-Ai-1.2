"""Resource-level permissions.

A capability alone is not enough — it must also cover
the resource:

* filesystem.write → only inside allowed roots
* browser access   → only allowed domains (block-list wins)
* connector access → only the scoped account/service
* computer control → only allowed applications/windows

The resource check runs inside the policy engine for
every action; a capability without a matching resource
is denied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


@dataclass
class ResourcePolicy:
    # Filesystem roots a capability may touch.
    filesystem_roots: tuple[str, ...] = ()
    # Browser: allow-list; empty = no restriction, but the
    # block-list always wins.
    browser_allowed_domains: tuple[str, ...] = ()
    browser_blocked_domains: tuple[str, ...] = ()
    # Connector scopes: {"email": {"account": "..."}, ...}
    connector_scopes: dict[str, dict[str, str]] = field(
        default_factory=dict
    )
    # Computer: allowed application/window names.
    computer_allowed_apps: tuple[str, ...] = ()
    # Sandbox/download directories.
    download_dir: str = ""
    sandbox_fs_root: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "filesystem_roots": list(self.filesystem_roots),
            "browser_allowed_domains": list(
                self.browser_allowed_domains
            ),
            "browser_blocked_domains": list(
                self.browser_blocked_domains
            ),
            "connector_scopes": {
                k: dict(v)
                for k, v in self.connector_scopes.items()
            },
            "computer_allowed_apps": list(
                self.computer_allowed_apps
            ),
            "download_dir": self.download_dir,
            "sandbox_fs_root": self.sandbox_fs_root,
        }


def _under(path: str, roots: tuple[str, ...]) -> bool:
    try:
        resolved = str(
            Path(str(path)).expanduser().resolve()
        )
    except Exception:
        resolved = str(path)
    for root in roots:
        try:
            r = str(
                Path(str(root)).expanduser().resolve()
            )
        except Exception:
            r = str(root)
        if resolved == r or resolved.startswith(r + "/"):
            return True
    return False


def check_filesystem(
    path: str, policy: ResourcePolicy
) -> tuple[bool, str]:
    # Unconfigured = unrestricted (owner configures roots
    # to restrict).  Configured = strictly enforced.
    if not policy.filesystem_roots:
        return True, ""
    if _under(path, policy.filesystem_roots):
        return True, ""
    return (
        False,
        f"{path!r} is outside the allowed filesystem roots",
    )


def check_browser_url(
    url: str, policy: ResourcePolicy
) -> tuple[bool, str]:
    try:
        host = (urlparse(str(url)).hostname or "").lower()
    except Exception:
        return False, f"unparseable url {url!r}"
    blocked = {d.lower() for d in policy.browser_blocked_domains}
    if any(host == d or host.endswith("." + d) for d in blocked):
        return False, f"domain {host!r} is blocked"
    allowed = {
        d.lower() for d in policy.browser_allowed_domains
    }
    if allowed and not any(
        host == d or host.endswith("." + d) for d in allowed
    ):
        return False, f"domain {host!r} is not allowed"
    return True, ""


def check_connector_scope(
    service: str, account: str, policy: ResourcePolicy
) -> tuple[bool, str]:
    # Unconfigured = unrestricted; configured = enforced.
    scope = policy.connector_scopes.get(str(service))
    if scope is None:
        return True, ""
    allowed_account = scope.get("account", "")
    if allowed_account and allowed_account != str(account):
        return (
            False,
            f"account {account!r} is outside the allowed "
            f"scope for {service!r}",
        )
    return True, ""


def check_computer_app(
    app: str, policy: ResourcePolicy
) -> tuple[bool, str]:
    # Unconfigured = unrestricted; configured = enforced.
    if not policy.computer_allowed_apps:
        return True, ""
    wanted = str(app or "").lower()
    for allowed in policy.computer_allowed_apps:
        if wanted == str(allowed).lower():
            return True, ""
    return False, f"application {app!r} is not allowed"


def check_resource(
    capability_id: str,
    resource: dict[str, Any],
    policy: ResourcePolicy,
) -> tuple[bool, str]:
    """Dispatch a resource check by capability family."""
    cap = str(capability_id or "")
    if cap.startswith("filesystem."):
        path = str(
            resource.get("path") or resource.get("target")
            or ""
        )
        return check_filesystem(path, policy)
    if cap.startswith("browser."):
        url = str(
            resource.get("url") or resource.get("target")
            or ""
        )
        if not url:
            return True, ""  # no url → nothing to scope
        return check_browser_url(url, policy)
    if cap.startswith("connector."):
        service = str(resource.get("service") or "")
        account = str(resource.get("account") or "")
        if not service:
            return True, ""
        return check_connector_scope(service, account, policy)
    if cap.startswith("computer."):
        app = str(
            resource.get("app") or resource.get("window")
            or ""
        )
        if not app:
            return True, ""
        return check_computer_app(app, policy)
    return True, ""
