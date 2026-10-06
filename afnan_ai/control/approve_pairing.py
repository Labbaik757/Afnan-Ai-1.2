"""Owner-side pairing approval for the Afnan AI Control Plane.

The control plane server holds pairing requests in memory, so the
owner's approval must go through the *running* server — not a fresh
``ControlPlane`` in this process.  This tool talks to the server's
owner-only endpoints (``/v1/owner/pairings*``) over loopback,
authorized by the owner token the server minted into
``<control-dir>/owner_token`` (mode 0600) on first start.

Usage:
    python -m afnan_ai.control.approve_pairing --list
    python -m afnan_ai.control.approve_pairing --approve <pairing_id>
    python -m afnan_ai.control.approve_pairing --deny <pairing_id>
    python -m afnan_ai.control.approve_pairing --watch   # interactive loop
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_CONTROL_DIR = os.path.expanduser("~/.afnan-ai/control")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


class ServerUnreachable(Exception):
    """The control plane server is not answering."""


class OwnerClient:
    """Minimal stdlib client for the owner-only endpoints."""

    def __init__(
        self,
        *,
        control_dir: str = DEFAULT_CONTROL_DIR,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        owner: str = "owner",
    ) -> None:
        self.base = f"http://{host}:{port}"
        self.owner = owner
        token_path = os.path.join(
            os.path.expanduser(control_dir), "owner_token"
        )
        try:
            with open(token_path, "r", encoding="utf-8") as handle:
                token = handle.read().strip()
        except OSError:
            raise ServerUnreachable(
                f"owner token not found at {token_path}; "
                "is the control plane server running?"
            )
        if not token:
            raise ServerUnreachable(
                f"owner token at {token_path} is empty."
            )
        self._token = token

    def _request(
        self, method: str, path: str, body: dict | None = None
    ) -> dict:
        data = (
            json.dumps(body).encode("utf-8")
            if body is not None
            else None
        )
        req = urllib.request.Request(
            self.base + path, data=data, method=method
        )
        req.add_header("Authorization", "Bearer " + self._token)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode("utf-8"))
                message = detail.get("error", "")
            except Exception:
                message = ""
            raise ServerUnreachable(
                f"server refused the request "
                f"(HTTP {exc.code}"
                + (f": {message}" if message else "")
                + ")"
            )
        except OSError as exc:
            raise ServerUnreachable(
                f"cannot reach the control plane server at "
                f"{self.base}: {exc}"
            )
        try:
            result = json.loads(raw)
        except ValueError:
            raise ServerUnreachable(
                "server returned a non-JSON response"
            )
        if not isinstance(result, dict):
            raise ServerUnreachable(
                "server returned an unexpected response"
            )
        return result

    def list_pending(self) -> list[dict]:
        result = self._request("GET", "/v1/owner/pairings")
        pairings = result.get("pairings", [])
        return pairings if isinstance(pairings, list) else []

    def approve(self, pairing_id: str) -> dict:
        return self._request(
            "POST",
            "/v1/owner/pairings/approve",
            {"pairing_id": pairing_id, "approved_by": self.owner},
        )

    def deny(self, pairing_id: str) -> dict:
        return self._request(
            "POST",
            "/v1/owner/pairings/deny",
            {"pairing_id": pairing_id, "approved_by": self.owner},
        )


def _print_pairings(pairings: list[dict]) -> None:
    if not pairings:
        print("No pending pairing requests.")
        return
    print(f"{len(pairings)} pending pairing request(s):\n")
    for item in pairings:
        print(f"  pairing_id : {item.get('pairing_id', '?')}")
        print(
            f"  device     : "
            f"{item.get('device_name') or '(unnamed)'}"
        )
        print(f"  device_id  : {item.get('device_id', '?')}")
        print(f"  platform   : {item.get('platform') or '(unknown)'}")
        print(f"  expires in : {item.get('expires_in_s', '?')}s")
        print()


def cmd_list(client: OwnerClient) -> int:
    _print_pairings(client.list_pending())
    return 0


def cmd_approve(client: OwnerClient, pairing_id: str) -> int:
    result = client.approve(pairing_id)
    print(
        f"Approved pairing for device "
        f"'{result.get('requesting_device_id', pairing_id)}'."
    )
    print("The device can now redeem its code and connect.")
    return 0


def cmd_deny(client: OwnerClient, pairing_id: str) -> int:
    client.deny(pairing_id)
    print("Pairing request denied.")
    return 0


def cmd_watch(client: OwnerClient) -> int:
    print("Watching for pairing requests (Ctrl+C to stop)...")
    seen: set[str] = set()
    try:
        while True:
            for item in client.list_pending():
                pairing_id = str(item.get("pairing_id", ""))
                if not pairing_id or pairing_id in seen:
                    continue
                seen.add(pairing_id)
                print("\n=== New pairing request ===")
                print(
                    f"  device     : "
                    f"{item.get('device_name') or '(unnamed)'}"
                )
                print(f"  device_id  : {item.get('device_id', '?')}")
                print(
                    f"  platform   : "
                    f"{item.get('platform') or '(unknown)'}"
                )
                print(f"  pairing_id : {pairing_id}")
                print(f"  expires in : "
                      f"{item.get('expires_in_s', '?')}s")
                answer = (
                    input("Approve? [y/N]: ").strip().lower()
                )
                if answer in ("y", "yes"):
                    cmd_approve(client, pairing_id)
                else:
                    cmd_deny(client, pairing_id)
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Approve or deny Afnan AI device pairing requests "
        "on the running control plane server."
    )
    parser.add_argument(
        "--control-dir",
        default=DEFAULT_CONTROL_DIR,
        help="Control plane state dir holding owner_token "
        "(same dir as the server's devices.json).",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Control plane server host (loopback only).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Control plane server port.",
    )
    parser.add_argument(
        "--owner",
        default="owner",
        help="Name recorded as the approver in the audit trail.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true")
    group.add_argument("--approve", metavar="PAIRING_ID")
    group.add_argument("--deny", metavar="PAIRING_ID")
    group.add_argument("--watch", action="store_true")
    args = parser.parse_args(argv)

    try:
        client = OwnerClient(
            control_dir=args.control_dir,
            host=args.host,
            port=args.port,
            owner=args.owner,
        )
    except ServerUnreachable as exc:
        print(f"error: {exc}")
        return 1

    try:
        if args.list:
            return cmd_list(client)
        if args.approve:
            return cmd_approve(client, args.approve)
        if args.deny:
            return cmd_deny(client, args.deny)
        return cmd_watch(client)
    except ServerUnreachable as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
