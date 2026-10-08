"""Bisect: find the exact Popen difference that kills Chrome.
Run: python bisect_launch.py — reports which variation works."""
import subprocess, tempfile, time, os, sys

EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BASE_ARGS = [
    "--remote-debugging-address=127.0.0.1",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-dev-shm-usage",
    "--mute-audio",
    "--no-sandbox",
    "--disable-gpu",
    "--disable-software-rasterizer",
    "about:blank",
]

def try_launch(label, popen_kwargs, extra_args=None):
    tmp = tempfile.mkdtemp(prefix="afnan-bisect-")
    args = [EXE, "--remote-debugging-port=9334",
            f"--user-data-dir={tmp}"] + BASE_ARGS + (extra_args or [])
    print(f"\n--- {label} ---")
    try:
        p = subprocess.Popen(args, **popen_kwargs)
        time.sleep(3)
        code = p.poll()
        if code is None:
            print(f"  ALIVE after 3s (pid {p.pid}) — SUCCESS")
            p.terminate()
            try: p.wait(timeout=5)
            except Exception: p.kill()
            return True
        else:
            print(f"  exited with code {code} — FAIL")
            return False
    except Exception as e:
        print(f"  exception: {e} — FAIL")
        return False

results = {}
# 1: exactly like the adapter (DEVNULL stdout, PIPE stderr, text=True)
results["adapter-style"] = try_launch(
    "1. adapter-style (stdout=DEVNULL, stderr=PIPE, text=True)",
    dict(stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True))
# 2: inherit stdio like the manual PowerShell command
results["inherit-stdio"] = try_launch(
    "2. inherit stdio (like manual & command)", {})
# 3: DEVNULL both (original adapter style before stderr capture)
results["devnull-both"] = try_launch(
    "3. stdout=DEVNULL, stderr=DEVNULL",
    dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
# 4: shell=True (PowerShell-style parsing)
results["shell-true"] = try_launch(
    "4. shell=True",
    dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
    extra_args=None)
# redo 4 properly with string command for shell=True
if not results["shell-true"]:
    tmp = tempfile.mkdtemp(prefix="afnan-bisect-")
    cmd = (f'"{EXE}" --remote-debugging-port=9334 '
           f'--remote-debugging-address=127.0.0.1 '
           f'--user-data-dir="{tmp}" --no-first-run '
           f'--no-default-browser-check --disable-dev-shm-usage '
           f'--mute-audio --no-sandbox --disable-gpu '
           f'--disable-software-rasterizer about:blank')
    print("\n--- 4b. shell=True with string command ---")
    try:
        p = subprocess.Popen(cmd, shell=True,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        time.sleep(3)
        code = p.poll()
        print(f"  {'ALIVE — SUCCESS' if code is None else f'exited {code} — FAIL'}")
        if code is None:
            results["shell-true-string"] = True
            p.terminate()
        else:
            results["shell-true-string"] = False
    except Exception as e:
        print(f"  exception: {e} — FAIL")
        results["shell-true-string"] = False

print("\n=== SUMMARY ===")
for k, v in results.items():
    print(f"  {k}: {'WORKS' if v else 'fails'}")
