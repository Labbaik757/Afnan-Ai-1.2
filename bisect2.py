"""Focused bisect: os.system vs Popen, same args, same profile dir."""
import os, subprocess, tempfile, time

EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
# Use the SAME fixed dir as the manual test (already has profile data)
FIXED_DIR = os.path.join(os.environ["TEMP"], "afnan-manual-test")

def check_port(port, timeout=5):
    import urllib.request, json
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/json/version", timeout=2) as r:
                return True
        except Exception:
            time.sleep(0.3)
    return False

print("=== Test A: os.system with EXACT manual command ===")
cmd = (f'"{EXE}" --remote-debugging-port=9335 '
       f'--remote-debugging-address=127.0.0.1 '
       f'--user-data-dir="{FIXED_DIR}" --no-first-run '
       f'--no-default-browser-check --disable-dev-shm-usage '
       f'--mute-audio --no-sandbox --disable-gpu '
       f'--disable-software-rasterizer about:blank')
print(f"Running: {cmd[:80]}...")
ret = os.system(f'start /b "" {cmd}')
print(f"os.system returned: {ret}")
time.sleep(3)
print(f"Debug port 9335 live: {check_port(9335)}")

print("\n=== Test B: Popen with SAME fixed dir (not fresh mkdtemp) ===")
args = [EXE, "--remote-debugging-port=9336",
        "--remote-debugging-address=127.0.0.1",
        f"--user-data-dir={FIXED_DIR}",
        "--no-first-run", "--no-default-browser-check",
        "--disable-dev-shm-usage", "--mute-audio",
        "--no-sandbox", "--disable-gpu",
        "--disable-software-rasterizer", "about:blank"]
p = subprocess.Popen(args, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
time.sleep(3)
code = p.poll()
print(f"Popen poll: {code} ({'ALIVE' if code is None else 'EXITED'})")
print(f"Debug port 9336 live: {check_port(9336)}")
if code is None:
    p.terminate()

print("\n=== Test C: Popen with FRESH dir (like adapter) ===")
fresh = tempfile.mkdtemp(prefix="afnan-bisect2-")
args2 = [EXE, "--remote-debugging-port=9337",
         "--remote-debugging-address=127.0.0.1",
         f"--user-data-dir={fresh}",
         "--no-first-run", "--no-default-browser-check",
         "--disable-dev-shm-usage", "--mute-audio",
         "--no-sandbox", "--disable-gpu",
         "--disable-software-rasterizer", "about:blank"]
p2 = subprocess.Popen(args2, stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL)
time.sleep(3)
code2 = p2.poll()
print(f"Popen poll: {code2} ({'ALIVE' if code2 is None else 'EXITED'})")
print(f"Debug port 9337 live: {check_port(9337)}")
if code2 is None:
    p2.terminate()

print("\nDone. Kill any leftover chrome test windows manually.")
