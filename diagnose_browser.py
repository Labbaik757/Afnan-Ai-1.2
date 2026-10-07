"""Diagnose browser automation step by step. Run: python diagnose_browser.py"""
import sys, traceback

print("=== Step 1: Find Chromium executable ===")
try:
    from afnan_ai.browser.chromium_adapter import ChromiumAdapter
    adapter = ChromiumAdapter()
    exe = adapter.find_executable("chromium")
    print(f"chromium: {exe}")
    exe2 = adapter.find_executable("chrome")
    print(f"chrome: {exe2}")
    if not exe and not exe2:
        print("FAIL: No browser found!")
        sys.exit(1)
    print("OK: Browser found")
except Exception as e:
    print(f"FAIL: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== Step 2: Launch browser (visible) ===")
try:
    from afnan_ai.browser.controller import BrowserController
    controller = BrowserController(headless=False)
    result = controller.launch(browser="chromium")
    print(f"Launch result: {result}")
    print("OK: Browser launched - DO YOU SEE A WINDOW?")
except Exception as e:
    print(f"FAIL: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== Step 3: Navigate to google.com ===")
try:
    result = controller.new_tab(url="https://www.google.com")
    print(f"Navigate result: {result.get('url', result)}")
    print("OK: Navigation worked")
except Exception as e:
    print(f"FAIL: {e}")
    traceback.print_exc()

print("\n=== Step 4: Take screenshot ===")
try:
    result = controller.screenshot()
    print(f"Screenshot: {type(result)}")
    print("OK: Screenshot worked")
except Exception as e:
    print(f"FAIL: {e}")

print("\n=== Step 5: Shutdown ===")
try:
    controller.shutdown()
    print("OK: Shutdown clean")
except Exception as e:
    print(f"FAIL: {e}")

print("\n=== DIAGNOSIS COMPLETE ===")
