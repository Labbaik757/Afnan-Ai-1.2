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
from afnan_ai.planner import Planner  # noqa: E402
from afnan_ai.tools import create_default_registry  # noqa: E402


def main() -> int:
    failures = []

    for system, expected in (("Windows", "windows"), ("Darwin", "macos"), ("Linux", "linux")):
        adapter = get_adapter(system)
        status = "OK" if adapter.name == expected else "FAIL"
        if status == "FAIL":
            failures.append(system)
        print(f"{status}: platform.system()={system!r} -> {type(adapter).__name__} ({adapter.name})")

    registry = create_default_registry(get_adapter("Linux"))
    required_tools = ("open_url", "open_application", "search_google", "take_screenshot")
    missing_tools = [name for name in required_tools if not registry.has(name)]
    if missing_tools:
        failures.append(f"tools:{missing_tools}")
        print(f"FAIL: tool registry missing {missing_tools}")
    else:
        print(f"OK: tool registry has {registry.names()}")

    # The Planner must never execute tools — guard the core promise
    # statically so a future refactor cannot silently break it.
    # (AST-based, so docstrings/comments mentioning execution are fine.)
    import ast

    assert isinstance(Planner, type)
    planner_tree = ast.parse(
        (ROOT / "afnan_ai" / "planner.py").read_text(encoding="utf-8")
    )
    forbidden_calls = [
        node.func.attr
        for node in ast.walk(planner_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("execute", "execute_or_raise", "run")
    ]
    if forbidden_calls:
        failures.append(f"planner executes tools ({forbidden_calls})")
        print(f"FAIL: planner.py calls {forbidden_calls} — Planner must only plan")
    else:
        print("OK: planner never executes tools")

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
