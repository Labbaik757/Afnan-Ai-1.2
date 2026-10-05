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

    # ---- Verifier must stay browser-agnostic ------------------------------
    # Browser knowledge lives in afnan_ai/browser (BrowserReliability);
    # the Verifier only sees generic observation/advisor callables.
    verifier_source = (
        ROOT / "afnan_ai" / "verifier.py"
    ).read_text(encoding="utf-8")
    if "afnan_ai.browser" in verifier_source or "BrowserController" in verifier_source:
        failures.append("verifier imports browser code")
        print("FAIL: verifier.py references browser code — it must stay browser-agnostic")
    else:
        print("OK: verifier stays browser-agnostic (browser details live in afnan_ai/browser)")

    # ---- Security layer: gate wired + platform/driver agnostic -----------
    controller_source = (
        ROOT / "afnan_ai" / "browser" / "controller.py"
    ).read_text(encoding="utf-8")
    if "approval_gate.check" not in controller_source:
        failures.append("controller does not consult the approval gate")
        print("FAIL: controller.py never calls approval_gate.check — sensitive actions are ungated")
    else:
        print("OK: controller consults the approval gate before sensitive actions")

    security_sources = {
        "security.py": (ROOT / "afnan_ai" / "browser" / "security.py").read_text(encoding="utf-8"),
        "redaction.py": (ROOT / "afnan_ai" / "redaction.py").read_text(encoding="utf-8"),
    }
    leaked = [
        name for name, src in security_sources.items()
        if "playwright" in src or "sys.platform" in src or "startfile" in src
    ]
    if leaked:
        failures.append(f"security layer is not platform/driver agnostic: {leaked}")
        print(f"FAIL: {leaked} contain driver/OS-specific code")
    else:
        print("OK: security + redaction layers are platform/driver agnostic")

    # ---- ScreenObserver: isolated, tool surface fixed ---------------------
    screen_dir = ROOT / "afnan_ai" / "screen"
    coupling = []
    for path in sorted(screen_dir.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for bad in ("import pyautogui", "import playwright",
                    "sys.platform", "startfile"):
            if bad in source:
                coupling.append(f"{path.name}:{bad}")
    if coupling:
        failures.append(f"screen layer has platform/driver coupling: {coupling}")
        print(f"FAIL: screen layer coupling: {coupling}")
    else:
        print("OK: screen layer is platform/driver agnostic (capture is injected)")

    try:
        from afnan_ai.screen import ScreenObserver, create_screen_tools
        observer = ScreenObserver()
        screen_names = sorted(t.name for t in create_screen_tools(observer))
    except Exception as e:  # noqa: BLE001 - report, don't crash the checker
        failures.append(f"screen tools unavailable: {e}")
        print(f"FAIL: screen tools unavailable: {e}")
    else:
        expected_screen = [
            "screen_assess_action", "screen_find_elements", "screen_observe",
        ]
        if screen_names != expected_screen:
            failures.append(f"screen tools mismatch: {screen_names}")
            print(f"FAIL: screen tools are {screen_names}")
        else:
            print(f"OK: screen tools are {screen_names} (observer never clicks)")

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

    # The browser tools must be creatable without launching a real
    # browser, and browser driver code must stay inside the
    # browser package (core modules never import it directly).
    try:
        from afnan_ai.browser import BrowserController, create_browser_tools
        from afnan_ai.browser.backend import BrowserBackend

        class _StubBackend(BrowserBackend):
            name = "stub"
            def start(self, browser, headless): pass
            def connect(self, endpoint): pass
            def new_page(self): return object()
            def close_page(self, handle): pass
            def goto(self, handle, url): pass
            def go_back(self, handle): pass
            def go_forward(self, handle): pass
            def reload(self, handle): pass
            def page_url(self, handle): return ""
            def page_title(self, handle): return ""
            def stop(self): pass

        browser_tool_names = sorted(
            t.name
            for t in create_browser_tools(
                BrowserController(backend=_StubBackend())
            )
        )
        expected_browser_tools = {
            "browser_launch", "browser_connect", "browser_new_tab",
            "browser_list_tabs", "browser_select_tab",
            "browser_close_tab", "browser_navigate",
            "browser_current_page", "browser_back",
            "browser_forward", "browser_reload", "browser_shutdown",
            "browser_find_elements", "browser_inspect_element",
            "browser_click", "browser_type", "browser_clear",
            "browser_select_option", "browser_press_key",
            "browser_scroll", "browser_observe_page",
            "browser_wait_for", "browser_screenshot",
            "browser_upload_file",
            "browser_accessibility_tree", "browser_find_semantic",
            "browser_set_tab_purpose", "browser_extract_content",
            "browser_search", "browser_open_result",
            "browser_wait_for_stable", "browser_collect_items",
            "browser_check_challenge", "browser_wait_challenge",
            "browser_downloads", "browser_network_status",
            "browser_rate_limit", "browser_approvals",
            "browser_profiles",
            "browser_session",
            "browser_capabilities",
        }
        if set(browser_tool_names) != expected_browser_tools:
            failures.append(f"browser tools mismatch: {browser_tool_names}")
            print(f"FAIL: browser tools are {browser_tool_names}")
        else:
            print(f"OK: browser tools are {browser_tool_names}")
    except Exception as e:
        failures.append(f"browser tools check crashed: {e}")
        print(f"FAIL: browser tools check crashed: {e}")

    driver_importers = []
    for py_file in (ROOT / "afnan_ai").rglob("*.py"):
        if "browser" in py_file.parts:
            continue
        if "playwright" in py_file.read_text(encoding="utf-8"):
            driver_importers.append(str(py_file.relative_to(ROOT)))
    if driver_importers:
        failures.append(f"playwright leaked outside browser package: {driver_importers}")
        print(f"FAIL: playwright used outside afnan_ai/browser: {driver_importers}")
    else:
        print("OK: browser driver code isolated in afnan_ai/browser")

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
