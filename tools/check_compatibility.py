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

    # The Executor must never evaluate arbitrary/model-generated code —
    # AST-check that executor.py has no such call and only dispatches
    # through the registry's execute method.
    executor_tree = ast.parse(
        (ROOT / "afnan_ai" / "executor.py").read_text(encoding="utf-8")
    )
    called_names = {
        node.func.id
        for node in ast.walk(executor_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        node.func.attr
        for node in ast.walk(executor_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    dangerous = {"eval", "exec", "compile", "__import__", "system", "popen", "Popen"}
    found_dangerous = sorted(dangerous & called_names)
    if found_dangerous:
        failures.append(f"executor evaluates code ({found_dangerous})")
        print(f"FAIL: executor.py calls {found_dangerous} — plans must run via ToolRegistry only")
    elif "execute" not in called_names:
        failures.append("executor does not dispatch via registry.execute")
        print("FAIL: executor.py does not call registry execute")
    else:
        print("OK: executor dispatches only via ToolRegistry, no code evaluation")

    # The Verifier must never re-execute a tool — AST-check that
    # verifier.py contains no execution/dispatch call at all.
    verifier_tree = ast.parse(
        (ROOT / "afnan_ai" / "verifier.py").read_text(encoding="utf-8")
    )
    verifier_calls = {
        node.func.attr
        for node in ast.walk(verifier_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    } | {
        node.func.id
        for node in ast.walk(verifier_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    execution_calls = {"execute", "execute_or_raise", "run", "eval", "exec", "compile"}
    found_execution = sorted(execution_calls & verifier_calls)
    if found_execution:
        failures.append(f"verifier executes tools ({found_execution})")
        print(f"FAIL: verifier.py calls {found_execution} — Verifier must only analyze results")
    else:
        print("OK: verifier never executes tools")

    # The central Agent orchestrator must drive tools only through
    # the Executor — it must never reach the ToolRegistry, a tool,
    # or the platform adapter directly.
    orchestrator_tree = ast.parse(
        (ROOT / "afnan_ai" / "orchestrator.py").read_text(encoding="utf-8")
    )
    orchestrator_attr_calls = {
        node.func.attr
        for node in ast.walk(orchestrator_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    orchestrator_name_calls = {
        node.func.id
        for node in ast.walk(orchestrator_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    forbidden_orchestrator = {
        "execute", "execute_or_raise", "launch_app", "open_path",
        "speak_system", "find_folder", "eval", "exec", "compile",
        "__import__",
    }
    found_forbidden = sorted(
        forbidden_orchestrator
        & (orchestrator_attr_calls | orchestrator_name_calls)
    )
    if found_forbidden:
        failures.append(f"orchestrator bypasses Executor ({found_forbidden})")
        print(f"FAIL: orchestrator.py calls {found_forbidden} — Agent must execute only via Executor")
    elif "execute_step" not in orchestrator_attr_calls:
        failures.append("orchestrator never calls Executor.execute_step")
        print("FAIL: orchestrator.py does not dispatch through Executor.execute_step")
    else:
        print("OK: orchestrator dispatches only via Executor, no direct tool/platform calls")

    # The RecoveryManager must only re-plan — AST-check that
    # recovery.py never executes a tool and only asks the Planner.
    recovery_tree = ast.parse(
        (ROOT / "afnan_ai" / "recovery.py").read_text(encoding="utf-8")
    )
    recovery_attr_calls = {
        node.func.attr
        for node in ast.walk(recovery_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    recovery_name_calls = {
        node.func.id
        for node in ast.walk(recovery_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    forbidden_recovery = {
        "execute", "execute_or_raise", "launch_app", "open_path",
        "speak_system", "eval", "exec", "compile", "__import__",
    }
    found_recovery = sorted(
        forbidden_recovery & (recovery_attr_calls | recovery_name_calls)
    )
    if found_recovery:
        failures.append(f"recovery executes tools ({found_recovery})")
        print(f"FAIL: recovery.py calls {found_recovery} — Recovery must only re-plan via the Planner")
    elif "plan" not in recovery_attr_calls:
        failures.append("recovery never calls Planner.plan")
        print("FAIL: recovery.py does not replan through Planner.plan")
    else:
        print("OK: recovery only replans via Planner, never executes tools")

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
