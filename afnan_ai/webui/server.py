"""Afnan Web UI — HTTP + SSE backend (stdlib only).

Serves the modern web interface for the Afnan assistant:

* ``GET /`` — the single-page frontend (``afnan_ai/webui/static/``)
* ``POST /api/chat`` — send text; the agent works in a background
  thread and replies stream back over SSE
* ``GET /api/events`` — Server-Sent Events: ``chat``, ``activity``
  and ``status`` messages as JSON
* ``/api/browser/*`` — drive the automation Chromium
* ``/api/tasks``, ``/api/goals``, ``/api/memory`` — managers
* ``/api/speak``, ``/api/voice/toggle`` — TTS / wake-word loop

Threading model: ``ThreadingHTTPServer`` handles each request in its
own thread.  Agent work always runs in daemon background threads so a
slow LLM or browser never blocks the server.  SSE fan-out is guarded
by a lock; a slow or dead client never stalls the broadcast.

Only the standard library is used — no Flask, no websockets.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

# ----------------------------------------------------------------------
# Small JSON helpers
# ----------------------------------------------------------------------


def _json(obj: Any) -> bytes:
    """Serialize *obj* to UTF-8 JSON, tolerating odd values."""
    return json.dumps(obj, ensure_ascii=False,
                      default=str).encode("utf-8")


def _safe_call(fn: Callable[[], Any]) -> dict[str, Any]:
    """Run *fn*, returning ``{"ok": True, "data": ...}`` or an error."""
    try:
        return {"ok": True, "data": fn()}
    except Exception as e:  # never let agent code crash the server
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ----------------------------------------------------------------------
# Server
# ----------------------------------------------------------------------


class WebUIServer:
    """HTTP + SSE backend for the Afnan web interface."""

    def __init__(self, agent: Any, port: int = 5000,
                 static_dir: str | Path | None = None) -> None:
        self.agent = agent
        self.port = int(port)
        self.static_dir = Path(
            static_dir
            if static_dir is not None
            else Path(__file__).resolve().parent / "static"
        )
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

        # SSE clients: each entry is a queue fed by broadcast().
        self._sse_lock = threading.Lock()
        self._sse_clients: set[queue.Queue] = set()

        # Voice wake-word loop state (best-effort toggle).
        self._voice_on = False
        self._voice_thread: threading.Thread | None = None
        self._voice_lock = threading.Lock()

        self._hook_agent()

    # -- lifecycle ------------------------------------------------------
    def start(self, open_browser: bool = True) -> None:
        """Start the server in a daemon thread (idempotent)."""
        if self._httpd is not None:
            return
        server = self

        class _Handler(_RequestHandler):
            pass

        _Handler.server_ref = server  # type: ignore[attr-defined]
        self._httpd = ThreadingHTTPServer(
            ("127.0.0.1", self.port), _Handler)
        # Allow quick restarts.
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="afnan-webui", daemon=True)
        self._thread.start()
        url = f"http://localhost:{self.port}/"
        print(f"Afnan Web UI: {url}")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass  # headless environments

    def stop(self) -> None:
        """Shut the server down (idempotent)."""
        httpd, self._httpd = self._httpd, None
        if httpd is not None:
            try:
                httpd.shutdown()
                httpd.server_close()
            except Exception:
                pass

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}/"

    # -- SSE ------------------------------------------------------------
    def broadcast(self, event_type: str,
                  data: Any = None) -> None:
        """Send an event to every connected SSE client (thread-safe)."""
        payload = _json({"type": event_type, "data": data})
        with self._sse_lock:
            dead: list[queue.Queue] = []
            for q in self._sse_clients:
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    dead.append(q)  # slow client: drop it
            for q in dead:
                self._sse_clients.discard(q)

    def _sse_register(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=256)
        with self._sse_lock:
            self._sse_clients.add(q)
        return q

    def _sse_unregister(self, q: queue.Queue) -> None:
        with self._sse_lock:
            self._sse_clients.discard(q)

    # -- agent hooks (no existing module is modified) --------------------
    def _hook_agent(self) -> None:
        """Mirror spoken replies + tool calls onto the SSE bus."""
        agent = self.agent
        orig_speak = agent.speak

        def speak_and_broadcast(text: str) -> None:
            try:
                self.broadcast(
                    "chat", {"who": "afnan", "text": text})
            except Exception:
                pass
            return orig_speak(text)

        agent.speak = speak_and_broadcast  # type: ignore[method-assign]

        registry = agent.tools
        orig_execute = registry.execute

        def execute_and_broadcast(*args: Any,
                                  **kwargs: Any) -> Any:
            tool_name = (args[0] if args
                         else kwargs.get("name", "?"))
            try:
                self.broadcast(
                    "activity",
                    {"kind": "tool", "tool": str(tool_name)})
            except Exception:
                pass
            return orig_execute(*args, **kwargs)

        registry.execute = execute_and_broadcast  # type: ignore[method-assign]

    # -- chat -------------------------------------------------------------
    def handle_chat(self, text: str) -> None:
        """Run one request in a background thread; replies go over SSE."""
        text = (text or "").strip()
        if not text:
            return
        self.broadcast("chat", {"who": "you", "text": text})
        self.broadcast("status", {"state": "working"})

        def _run() -> None:
            try:
                self.agent.handle_request(text)
            except SystemExit:
                self.broadcast("status", {"state": "stopping"})
            except Exception as e:
                self.broadcast(
                    "chat",
                    {"who": "afnan",
                     "text": f"Sorry boss, something went wrong: {e}"})
            finally:
                self.broadcast("status", {"state": "ready"})

        threading.Thread(target=_run, name="afnan-chat",
                         daemon=True).start()

    # -- voice --------------------------------------------------------------
    def toggle_voice(self) -> dict[str, Any]:
        """Toggle the wake-word loop (best-effort stop).

        Starting spawns a daemon thread running ``agent.start()``.
        Stopping is best-effort: a thread blocked inside the mic
        listen call cannot be interrupted safely, so the flag takes
        effect when the current listen cycle ends.
        """
        with self._voice_lock:
            self._voice_on = not self._voice_on
            on = self._voice_on
        if on:
            def _loop() -> None:
                try:
                    self.agent.start()
                except Exception as e:
                    self.broadcast(
                        "activity",
                        {"kind": "voice",
                         "message": f"voice loop ended: {e}"})
                finally:
                    with self._voice_lock:
                        self._voice_on = False
                    self.broadcast("status",
                                   {"voice": False})

            self._voice_thread = threading.Thread(
                target=_loop, name="afnan-voice", daemon=True)
            self._voice_thread.start()
            self.broadcast("status", {"voice": True})
            return {"voice": True}
        self.broadcast("status", {"voice": False})
        return {"voice": False,
                "note": "takes effect after the current listen ends"}

    # -- status ---------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        agent = self.agent
        cfg = getattr(agent, "config", None)
        llm = getattr(agent, "llm", None)
        try:
            browser_running = bool(
                getattr(agent.browser, "_running", False))
        except Exception:
            browser_running = False
        with self._voice_lock:
            voice = self._voice_on
        return {
            "model": getattr(llm, "model", None)
            or getattr(cfg, "llm_model", "?"),
            "provider": getattr(llm, "display_name",
                                getattr(llm, "name", "?")),
            "wake_word": getattr(cfg, "wake_word", "afnan"),
            "voice": voice,
            "browser_running": browser_running,
            "platform": getattr(getattr(agent, "adapter", None),
                                "name", "?"),
        }


# ----------------------------------------------------------------------
# HTTP request handler
# ----------------------------------------------------------------------


class _RequestHandler(BaseHTTPRequestHandler):
    """Routes for the Web UI.  Every handler is exception-proof."""

    server_ref: WebUIServer  # set by WebUIServer.start()
    server_version = "AfnanWebUI/1.0"

    # -- plumbing -------------------------------------------------------
    def log_message(self, *args: Any) -> None:
        pass  # keep the console for the assistant, not HTTP logs

    def _send_json(self, obj: Any,
                   status: int = 200) -> None:
        body = _json(obj)
        self.send_response(status)
        self.send_header("Content-Type",
                         "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > 1_000_000:
            return {}
        try:
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    # -- routing ----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        try:
            self._route_get()
        except Exception:
            traceback.print_exc()
            try:
                self._send_json({"ok": False,
                                 "error": "internal error"},
                                status=500)
            except Exception:
                pass

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._route_post()
        except Exception:
            traceback.print_exc()
            try:
                self._send_json({"ok": False,
                                 "error": "internal error"},
                                status=500)
            except Exception:
                pass

    def _route_get(self) -> None:
        srv = self.server_ref
        path = urlparse(self.path).path

        if path == "/":
            return self._serve_static("index.html",
                                      "text/html; charset=utf-8")
        if path == "/api/events":
            return self._serve_sse()
        if path.startswith("/static/"):
            return self._serve_static(path[len("/static/"):])
        if path == "/api/status":
            return self._send_json(
                _safe_call(srv.status))
        if path == "/api/tasks":
            return self._send_json(
                _safe_call(self._list_tasks))
        if path == "/api/goals":
            return self._send_json(
                _safe_call(self._list_goals))
        if path == "/api/memory":
            return self._send_json(
                _safe_call(self._list_memory))
        if path == "/api/browser/screenshot":
            return self._serve_screenshot()
        self._send_json({"ok": False, "error": "not found"},
                        status=404)

    def _route_post(self) -> None:
        srv = self.server_ref
        path = urlparse(self.path).path
        body = self._read_json()

        if path == "/api/chat":
            text = str(body.get("text", ""))
            if not text.strip():
                return self._send_json(
                    {"ok": False, "error": "empty text"})
            # Background thread; replies stream over SSE.
            threading.Thread(
                target=srv.handle_chat, args=(text,),
                name="afnan-chat-dispatch", daemon=True).start()
            return self._send_json({"ok": True,
                                    "accepted": True})

        if path == "/api/browser/launch":
            return self._send_json(_safe_call(
                lambda: self._tool("browser_launch", {})))
        if path == "/api/browser/navigate":
            url = str(body.get("url", "")).strip()
            if not url:
                return self._send_json(
                    {"ok": False, "error": "missing url"})
            if "://" not in url:
                url = "https://" + url
            return self._send_json(_safe_call(
                lambda: self._tool("browser_navigate",
                                   {"url": url})))
        if path in ("/api/browser/back", "/api/browser/forward",
                    "/api/browser/reload"):
            action = path.rsplit("/", 1)[-1]
            return self._send_json(_safe_call(
                lambda: self._tool(f"browser_{action}", {})))
        if path == "/api/browser/tabs":
            return self._send_json(_safe_call(
                lambda: self._tool("browser_list_tabs", {})))
        if path == "/api/browser/screenshot":
            return self._serve_screenshot()

        if path == "/api/tasks/add":
            title = str(body.get("title", "")).strip()
            if not title:
                return self._send_json(
                    {"ok": False, "error": "missing title"})

            def _add() -> dict[str, Any]:
                task = srv.agent.task_manager.enqueue(
                    goal_text=title)
                return {"task_id": getattr(task, "task_id",
                                           "")}

            return self._send_json(_safe_call(_add))
        if path == "/api/tasks/complete":
            task_id = str(body.get("id", "")).strip()
            if not task_id:
                return self._send_json(
                    {"ok": False, "error": "missing id"})
            return self._send_json(_safe_call(
                lambda: srv.agent.task_manager.complete(task_id)
                and True))

        if path == "/api/speak":
            text = str(body.get("text", ""))
            if not text.strip():
                return self._send_json(
                    {"ok": False, "error": "empty text"})

            def _speak() -> bool:
                # Speak without re-broadcasting as chat (already
                # shown by the caller); call the wrapped original.
                srv.agent.speak(text)
                return True

            threading.Thread(
                target=lambda: _safe_call(_speak),
                name="afnan-speak", daemon=True).start()
            return self._send_json({"ok": True,
                                    "accepted": True})

        if path == "/api/voice/toggle":
            return self._send_json(_safe_call(srv.toggle_voice))

        self._send_json({"ok": False, "error": "not found"},
                        status=404)

    # -- static files -------------------------------------------------------
    _MIME = {
        ".html": "text/html; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".svg": "image/svg+xml",
        ".ico": "image/x-icon",
        ".json": "application/json",
    }

    def _serve_static(self, rel: str,
                      content_type: str | None = None) -> None:
        srv = self.server_ref
        target = (srv.static_dir / rel).resolve()
        # No path traversal outside the static dir.
        try:
            target.relative_to(srv.static_dir.resolve())
        except ValueError:
            return self._send_json(
                {"ok": False, "error": "forbidden"}, status=403)
        if not target.is_file():
            return self._send_json(
                {"ok": False, "error": "not found"}, status=404)
        ctype = (content_type
                 or self._MIME.get(target.suffix.lower(),
                                   "application/octet-stream"))
        try:
            data = target.read_bytes()
        except OSError:
            return self._send_json(
                {"ok": False, "error": "read failed"}, status=500)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # -- SSE ------------------------------------------------------------------
    def _serve_sse(self) -> None:
        srv = self.server_ref
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        q = srv._sse_register()
        try:
            # Initial hello so the client knows the stream is live.
            self._sse_write(b"data: " + _json(
                {"type": "hello", "data": srv.status()}) + b"\n\n")
            last_ping = time.monotonic()
            while True:
                try:
                    payload = q.get(timeout=15)
                    self._sse_write(b"data: " + payload + b"\n\n")
                except queue.Empty:
                    pass
                # Keep-alive comment every ~15s.
                if time.monotonic() - last_ping > 15:
                    self._sse_write(b": ping\n\n")
                    last_ping = time.monotonic()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            srv._sse_unregister(q)

    def _sse_write(self, data: bytes) -> None:
        self.wfile.write(data)
        self.wfile.flush()

    # -- API helpers ------------------------------------------------------------
    def _tool(self, name: str,
              arguments: dict[str, Any]) -> Any:
        srv = self.server_ref
        res = srv.agent.tools.execute(name, arguments)
        if getattr(res, "success", False):
            out = getattr(res, "output", None)
            return out if out is not None else True
        err = getattr(res, "error", None)
        raise RuntimeError(
            getattr(err, "message", None) or f"{name} failed")

    def _list_tasks(self) -> list[dict[str, Any]]:
        tasks = self.server_ref.agent.task_manager.list()
        out = []
        for t in tasks or []:
            if hasattr(t, "to_dict"):
                out.append(t.to_dict())
            else:
                out.append({
                    "task_id": getattr(t, "task_id", ""),
                    "title": getattr(t, "title",
                                     getattr(t, "goal_text",
                                             "?")),
                    "status": str(getattr(t, "status", "?")),
                })
        return out

    def _list_goals(self) -> list[dict[str, Any]]:
        goals = self.server_ref.agent.goal_manager.list()
        out = []
        for g in goals or []:
            if hasattr(g, "to_dict"):
                out.append(g.to_dict())
            else:
                out.append({
                    "goal_id": getattr(g, "goal_id", ""),
                    "title": getattr(g, "title",
                                     getattr(g, "name", "?")),
                    "description": str(
                        getattr(g, "description", ""))[:200],
                })
        return out

    def _list_memory(self) -> list[dict[str, Any]]:
        store = self.server_ref.agent.memory_store
        try:
            items = store.list(kind=None)  # type: ignore[call-arg]
        except TypeError:
            items = store.list()
        out = []
        for item in (items or [])[-30:]:
            out.append({
                "id": str(getattr(item, "memory_id",
                                  getattr(item, "id", ""))),
                "text": str(getattr(item, "text",
                                    getattr(item, "content",
                                            item)))[:500],
                "kind": str(getattr(item, "kind", "")),
            })
        return out

    def _serve_screenshot(self) -> None:
        def _capture() -> bytes:
            res = self.server_ref.agent.tools.execute(
                "browser_screenshot", {})
            if not getattr(res, "success", False):
                err = getattr(res, "error", None)
                raise RuntimeError(
                    getattr(err, "message", None)
                    or "screenshot failed")
            out = getattr(res, "output", None) or {}
            path = out.get("path") if isinstance(out, dict) \
                else None
            if not path:
                raise RuntimeError("no screenshot path")
            data = Path(str(path)).read_bytes()
            if not data:
                raise RuntimeError("empty screenshot")
            return data

        result = _safe_call(_capture)
        if not result["ok"]:
            return self._send_json(result, status=500)
        data: bytes = result["data"]
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass


# ----------------------------------------------------------------------
# Entry point helper
# ----------------------------------------------------------------------


def create_server(agent: Any,
                  port: int = 5000) -> WebUIServer:
    """Build (but do not start) a Web UI server for *agent*."""
    return WebUIServer(agent, port=port)
