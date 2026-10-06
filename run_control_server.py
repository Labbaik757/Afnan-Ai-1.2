"""Start the Afnan Remote Control Plane so the Android app can pair.

Quick start (no certificates, local network only)::

    python run_control_server.py --insecure-lan

Then in the app's setup screen enter ``http://<PC LAN IP>:8765``
with TLS switched off and tap "Test connection & continue".

TLS mode (recommended outside your own network)::

    openssl req -x509 -newkey rsa:2048 \\
        -keyout afnan-key.pem -out afnan-cert.pem \\
        -days 825 -nodes -subj "/CN=192.168.1.10" \\
        -addext "subjectAltName=IP:192.168.1.10"
    python run_control_server.py \\
        --tls-cert afnan-cert.pem --tls-key afnan-key.pem

(Replace 192.168.1.10 with your PC's LAN IP from ``ipconfig``.)

Pairing: when the phone shows a pairing code, approve it on this PC::

    python -m afnan_ai.control.approve_pairing

Keep this window open while the phone is connected. Ctrl+C stops it.
"""

from __future__ import annotations

import argparse
import sys
import threading


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the Afnan Remote Control Plane for phone pairing."
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Interface to bind (default: all interfaces).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to listen on (default: 8765).",
    )
    parser.add_argument(
        "--tls-cert",
        default=None,
        help="Path to the PEM certificate file (TLS mode).",
    )
    parser.add_argument(
        "--tls-key",
        default=None,
        help="Path to the PEM private-key file (TLS mode).",
    )
    parser.add_argument(
        "--insecure-lan",
        action="store_true",
        help="Serve plaintext HTTP on the local network without TLS. "
        "Development only: anyone on this network can read traffic. "
        "No certificate needed.",
    )
    args = parser.parse_args(argv)
    if not args.insecure_lan and not (args.tls_cert and args.tls_key):
        parser.error("pass --insecure-lan or both --tls-cert and --tls-key")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    from afnan_ai.agent import AfnanAgent
    from afnan_ai.platform import get_adapter
    from afnan_ai.control import ControlPlaneServer

    agent = AfnanAgent(adapter=get_adapter())
    server = ControlPlaneServer.from_agent(
        agent,
        host=args.host,
        port=args.port,
        tls_cert=args.tls_cert,
        tls_key=args.tls_key,
        allow_insecure_lan=args.insecure_lan,
    )
    server.start()
    print(f"Afnan Control Plane listening on {server.url}")
    print("Pair a phone from the app, then approve with:")
    print("    python -m afnan_ai.control.approve_pairing")
    print("Press Ctrl+C to stop.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
