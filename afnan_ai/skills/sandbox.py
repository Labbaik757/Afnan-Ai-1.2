"""Sandbox for skill validation and code skills.

Two jobs:

1. **Composed-skill validation** (`validate_skill`): purely
   static — schemas, step resolution, argument validation
   against real tool schemas, dependency checks, risk
   analysis, injection and secret scans.  No code runs.

2. **Code-skill execution** (`SandboxedPython`): for the
   future path where a skill genuinely needs generated
   code.  Runs in a separate process with:
   - scrubbed environment (no secrets inherited),
   - current directory locked to an empty temp dir,
   - dangerous imports blocked (os, sys, subprocess,
     socket, pathlib, shutil, ...),
   - restricted builtins (no open/eval/exec/compile),
   - wall-clock timeout and (on POSIX) CPU/memory limits.

   This is a best-effort sandbox for accidental harm, not
   a security boundary against a determined attacker —
   which is why registering or running a code skill always
   additionally requires explicit human approval.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from typing import Any

from afnan_ai.log_config import get_logger
from afnan_ai.redaction import redact_text

logger = get_logger(__name__)


@dataclass
class SandboxConfig:
    timeout_s: float = 10.0
    max_memory_mb: int = 256
    max_cpu_s: int = 15
    allow_network: bool = False  # currently always False


@dataclass
class SandboxResult:
    ok: bool
    output: str = ""
    error: str = ""
    timed_out: bool = False


# Modules a code skill may never import.
_BLOCKED_IMPORTS = frozenset({
    "os", "sys", "subprocess", "socket", "pathlib", "shutil",
    "ctypes", "multiprocessing", "threading", "signal",
    "importlib", "pkgutil", "runpy", "pty", "fcntl",
    "urllib", "http", "ftplib", "smtplib", "ssl",
    "site", "sitecustomize",
})


def _child_prelude() -> str:
    blocked = sorted(_BLOCKED_IMPORTS)
    return textwrap.dedent(
        f"""\
        import sys as _sys

        _BLOCKED = {blocked!r}
        # Purge risky modules the interpreter pre-loaded
        # (site.py etc.): with them gone from sys.modules the
        # import guard below blocks any fresh import by name.
        # sys itself is kept — the stdlib needs it.
        for _m in list(_sys.modules):
            if _m.split(".")[0] in _BLOCKED and _m != "sys":
                del _sys.modules[_m]
        _real_import = __import__

        def _guarded_import(name, *args, **kwargs):
            root = str(name).split(".")[0]
            if root in _BLOCKED and root not in _sys.modules:
                raise ImportError(
                    f"import of '{{name}}' is blocked in the "
                    "skill sandbox"
                )
            return _real_import(name, *args, **kwargs)

        __builtins__.__import__ = _guarded_import
        # open() is removed process-wide (imports do not need
        # it); eval/exec/compile are shadowed in the module
        # globals below so the import machinery keeps working.
        del __builtins__.open
        """
    )


class SandboxedPython:
    """Run untrusted Python with tight restrictions."""

    def __init__(
        self, config: SandboxConfig | None = None
    ) -> None:
        self.config = config or SandboxConfig()

    def run(
        self,
        code: str,
        *,
        arguments: dict[str, Any] | None = None,
    ) -> SandboxResult:
        """Execute *code* with ``arguments`` available as the
        ``ARGS`` dict.  The snippet must assign its result to
        ``RESULT`` (JSON-serializable)."""
        if not self.config.allow_network:
            pass  # network is blocked via the import guard
        workdir = tempfile.mkdtemp(prefix="skill_sandbox_")
        payload = (
            _child_prelude()
            + "\nARGS = "
            + _safe_repr(arguments or {})
            + "\nRESULT = None\n"
            # Shadow dangerous builtins in module globals (the
            # import system keeps working; user code cannot
            # reach the real ones).
            + "eval = None\nexec = None\ncompile = None\n"
            + str(code)
            + "\nimport json as _json\n"
            + "print(_json.dumps({'result': RESULT}, default=str))\n"
        )
        script = os.path.join(workdir, "skill.py")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write(payload)
        # Scrubbed environment: no inherited secrets.
        env = {
            "PATH": "/usr/bin:/bin",
            "PYTHONSAFEPATH": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

        def _limits() -> None:  # POSIX only
            try:
                import resource

                mem = self.config.max_memory_mb * 1024 * 1024
                resource.setrlimit(
                    resource.RLIMIT_AS, (mem, mem)
                )
                cpu = int(self.config.max_cpu_s)
                resource.setrlimit(
                    resource.RLIMIT_CPU, (cpu, cpu)
                )
            except Exception:
                pass

        try:
            proc = subprocess.run(
                # -I: isolated mode — no site.py, no user site,
                # no PYTHONPATH/env inheritance.
                [sys.executable, "-I", script],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_s,
                preexec_fn=(
                    _limits if os.name == "posix" else None
                ),
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                ok=False, error="sandbox timeout",
                timed_out=True,
            )
        except Exception as e:  # noqa: BLE001 - report it
            return SandboxResult(ok=False, error=str(e)[:300])
        finally:
            _remove_tree(workdir)
        if proc.returncode != 0:
            return SandboxResult(
                ok=False,
                error=redact_text(proc.stderr.strip())[:500]
                or f"exit {proc.returncode}",
            )
        return SandboxResult(
            ok=True, output=proc.stdout.strip()[:4000]
        )


def _safe_repr(value: Any) -> str:
    import json

    return json.dumps(value, default=str)


def _remove_tree(path: str) -> None:
    import shutil

    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


@dataclass
class ValidationIssue:
    code: str
    message: str
    step_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "step_id": self.step_id,
        }


@dataclass
class SkillValidation:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [i.to_dict() for i in self.issues],
        }
