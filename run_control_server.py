"""Start the Afnan Remote Control Plane so the Android app can pair.

The phone reaches the PC over the local network, so TLS is required
(insecure mode is refused on non-loopback addresses).

1. Find your PC's LAN IP, e.g. ``ipconfig`` on Windows.
2. Create a self-signed certificate with that IP in the SAN
   (Git Bash / Linux / macOS)::

       openssl req -x509 -newkey rsa:2048 \\
           -keyout afnan-key.pem -out afnan-cert.pem \\
           -days 825 -nodes -subj "/CN=192.168.1.10" \\
           -addext "subjectAltName=IP:192.168.1.10"

   (Replace 192.168.1.10 with your PC's LAN IP.)
3. Run this script from the repository root::

       python run_control_server.py \\
           --tls-cert afnan-cert.pem --tls-key afnan-key.pem

4. Install ``afnan-cert.pem`` on the phone
   (Settings -> Security -> Install certificate) so the app trusts it.
5. In the app's setup screen enter ``https://<PC LAN IP>:8765``,
   keep TLS on, and tap "Test connection & continue".
6. When the phone shows a pairing code, approve it on this PC::

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
        required=True,
        help="Path to the PEM certificate file.",
    )
    parser.add_argument(
        "--tls-key",
        required=True,
        help="Path to the PEM private-key file.",
    )
    return parser.parse_args(argv)


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
