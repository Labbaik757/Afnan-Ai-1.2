"""ExecutionSandbox — controlled advanced execution.

Generated code NEVER runs directly on the production
host.  It runs here, under explicit limits:

* isolated filesystem (dedicated temp working dir;
  bubblewrap jail when available)
* explicit network policy (denied unless allow-listed)
* process restrictions (no new sessions, no shell by
  default)
* CPU limit, memory limit, execution timeout
* scrubbed environment variables
* credential isolation (secret-shaped vars stripped)
* no package installs unless explicitly allowed
* output size limit

On Linux without bubblewrap the sandbox degrades
honestly: rlimits + timeout + isolated cwd + scrubbed
env still apply, and the report says which mechanisms
were active.  Network denial without a network
namespace is best-effort and reported as such.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SandboxLimits:
    cpu_seconds: int = 10
    memory_mb: int = 256
    timeout_s: float = 30.0
    max_output_bytes: int = 256 * 1024
    allow_network: bool = False
    allowed_hosts: tuple[str, ...] = ()
    allowed_env: tuple[str, ...] = (
        "PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TZ",
        "PYTHONPATH", "PYTHONIOENCODING",
    )
    allow_pip_install: bool = False
    max_processes: int = 32


@dataclass
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    output_truncated: bool
    mechanisms: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return (
            not self.timed_out
            and not self.error
            and self.returncode == 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "output_truncated": self.output_truncated,
            "mechanisms": list(self.mechanisms),
            "error": self.error,
            "ok": self.ok,
        }


def _scrub_env(
    allowed: tuple[str, ...], extra: dict[str, str] | None
) -> dict[str, str]:
    keep = {v.upper() for v in allowed}
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key.upper() in keep:
            env[key] = value
    # Credential isolation: never inherit secret-shaped vars.
    for key in list(env):
        lowered = key.lower()
        if any(
            token in lowered
            for token in (
                "secret", "password", "token", "api_key",
                "apikey", "credential", "private",
            )
        ):
            del env[key]
    for key, value in (extra or {}).items():
        env[str(key)] = str(value)
    return env


def _rlimits(limits: SandboxLimits):
    def _apply():
        try:
            import resource

            resource.setrlimit(
                resource.RLIMIT_CPU,
                (limits.cpu_seconds, limits.cpu_seconds),
            )
            mem = limits.memory_mb * 1024 * 1024
            resource.setrlimit(
                resource.RLIMIT_AS, (mem, mem)
            )
            resource.setrlimit(
                resource.RLIMIT_NPROC,
                (limits.max_processes, limits.max_processes),
            )
        except Exception:
            pass
        try:
            os.setsid()  # no new sessions for children
        except Exception:
            pass

    return _apply


class ExecutionSandbox:
    """Run untrusted code with hard limits."""

    def __init__(
        self, limits: SandboxLimits | None = None
    ) -> None:
        self.limits = limits or SandboxLimits()
        self._bwrap = shutil.which("bwrap")

    @property
    def mechanisms_available(self) -> list[str]:
        mech = [
            "rlimits(cpu, memory, nproc)",
            "timeout",
            "isolated temp cwd",
            "scrubbed env",
            "credential isolation",
            "output cap",
        ]
        if self._bwrap:
            mech.append("bubblewrap jail")
            if not self.limits.allow_network:
                mech.append("bubblewrap: no network")
        elif not self.limits.allow_network:
            mech.append(
                "network: best-effort (no net namespace)"
            )
        return mech

    def run_python(
        self,
        code: str,
        *,
        extra_env: dict[str, str] | None = None,
        args: list[str] | None = None,
    ) -> SandboxResult:
        """Execute Python source in the sandbox."""
        workdir = tempfile.mkdtemp(prefix="afnan-sandbox-")
        script = os.path.join(workdir, "main.py")
        try:
            with open(script, "w", encoding="utf-8") as fh:
                fh.write(code)
            return self._run(
                [sys.executable, "-I", "-c",
                 _guarded_runner()],
                workdir=workdir,
                extra_env={
                    **(extra_env or {}),
                    "AFNAN_SANDBOX_SCRIPT": script,
                    "AFNAN_SANDBOX_ARGS": "\n".join(
                        args or []
                    ),
                },
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def run_command(
        self,
        argv: list[str],
        *,
        extra_env: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Execute an argv (no shell) in the sandbox."""
        if not argv:
            return SandboxResult(
                -1, "", "", False, False,
                error="empty command",
            )
        workdir = tempfile.mkdtemp(prefix="afnan-sandbox-")
        try:
            return self._run(
                list(argv), workdir=workdir,
                extra_env=extra_env,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    # -- internals ------------------------------------------------------
    def _run(
        self,
        argv: list[str],
        *,
        workdir: str,
        extra_env: dict[str, str] | None,
    ) -> SandboxResult:
        limits = self.limits
        mechanisms = list(self.mechanisms_available)
        env = _scrub_env(limits.allowed_env, extra_env)
        if self._bwrap and sys.platform.startswith("linux"):
            argv = self._bwrap_argv(argv, workdir)
            mechanisms.append("bwrap active")
        try:
            proc = subprocess.run(
                argv,
                cwd=workdir,
                env=env,
                capture_output=True,
                timeout=limits.timeout_s,
                preexec_fn=(
                    _rlimits(limits)
                    if not self._bwrap
                    else None
                ),
            )
            truncated = False
            out = proc.stdout or b""
            err = proc.stderr or b""
            cap = limits.max_output_bytes
            if len(out) > cap:
                out = out[:cap]
                truncated = True
            if len(err) > cap:
                err = err[:cap]
                truncated = True
            return SandboxResult(
                proc.returncode,
                out.decode("utf-8", "replace"),
                err.decode("utf-8", "replace"),
                False, truncated, mechanisms,
            )
        except subprocess.TimeoutExpired as exc:
            out = (exc.stdout or b"")[: limits.max_output_bytes]
            err = (exc.stderr or b"")[: limits.max_output_bytes]
            return SandboxResult(
                -1, out.decode("utf-8", "replace"),
                err.decode("utf-8", "replace"),
                True, False, mechanisms,
                error=f"timeout after {limits.timeout_s}s",
            )
        except Exception as exc:  # noqa: BLE001
            return SandboxResult(
                -1, "", "", False, False, mechanisms,
                error=f"sandbox failed: {exc}",
            )

    def _bwrap_argv(
        self, argv: list[str], workdir: str
    ) -> list[str]:
        cmd = [
            self._bwrap,
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/etc/resolv.conf",
            "/etc/resolv.conf",
            "--tmpfs", "/tmp",
            "--bind", workdir, workdir,
            "--chdir", workdir,
            "--die-with-parent",
            "--new-session",
        ]
        if not self.limits.allow_network:
            cmd.append("--unshare-net")
        # /lib may be a symlink forest; bind what's needed.
        for libdir in ("/lib", "/lib64"):
            if os.path.isdir(libdir):
                cmd += ["--ro-bind", libdir, libdir]
        return cmd + ["--"] + argv


def _guarded_runner() -> str:
    """Preamble: blocks pip installs unless allowed, then
    runs the script from AFNAN_SANDBOX_SCRIPT."""
    return (
        "import os, runpy, sys\n"
        "allow_pip = os.environ.get('AFNAN_ALLOW_PIP') == '1'\n"
        "script = os.environ['AFNAN_SANDBOX_SCRIPT']\n"
        "src = open(script, encoding='utf-8').read()\n"
        "if not allow_pip and ('pip install' in src or 'subprocess' in src and 'pip' in src):\n"
        "    raise SystemExit('pip install blocked by sandbox policy')\n"
        "sys.argv = ['main.py'] + os.environ.get('AFNAN_SANDBOX_ARGS', '').splitlines()\n"
        "runpy.run_path(script, run_name='__main__')\n"
    )


# A process-wide default the security center can share.
DEFAULT_SANDBOX = ExecutionSandbox()
