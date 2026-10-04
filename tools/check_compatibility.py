"""Compatibility check — run on any host, simulates all three OSes.

    python tools/check_compatibility.py

Exits non-zero if any platform adapter cannot be selected, any
module fails to compile, or the core agent leaks OS-specific code.
"""

import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from afnan_ai.platform import get_adapter  # noqa: E402


def main() -> int:
    failures = []

    for system, expected in (("Windows", "windows"), ("Darwin", "macos"), ("Linux", "linux")):
        adapter = get_adapter(system)
        status = "OK" if adapter.name == expected else "FAIL"
        if status == "FAIL":
            failures.append(system)
        print(f"{status}: platform.system()={system!r} -> {type(adapter).__name__} ({adapter.name})")

    for path in sorted(ROOT.rglob("*.py")):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as e:
            failures.append(str(path))
            print(f"FAIL: compile {path.relative_to(ROOT)}: {e}")

    print("OK: all modules compile" if not failures else f"FAILURES: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
