"""Tests for the Afnan Web UI backend (afnan_ai/webui/server.py).

Uses a fake agent plus live HTTP against a real WebUIServer on a
high test port.  stdlib only.
"""

from __future__ import annotations

import json
import queue
import socket
import threading
import time
import unittest
import urllib.request
from unittest import mock

from afnan_ai.webui.server import WebUIServer

_PORT = 18780


def _fake_agent() -> mock.MagicMock:
    agent = mock.MagicMock()
    agent.config.llm_model = "llama3"
    agent.config.wake_word = "afnan"
    agent.llm.model = "llama3"
    agent.llm.display_name = "Ollama"
    agent.adapter.name = "linux"
    agent.browser._running = False
    return agent


class _ServerCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent = _fake_agent()
        cls.srv = WebUIServer(cls.agent, port=_PORT)
        cls.srv.start(open_browser=False)
        time.sleep(0.4)
        cls.base = f"http://localhost:{_PORT}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.srv.stop()

    def _get(self, path: str):
        with urllib.request.urlopen(
                self.base + path, timeout=10) as r:
            return r.status, r.read()

    def _post(self, path: str, data: dict):
        req = urllib.request.Request(
            self.base + path, data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())


class StaticTests(_ServerCase):
    def test_index_served(self):
        status, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"Afnan", body)

    def test_static_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("/static/does-not-exist.js")
        self.assertEqual(ctx.exception.code, 404)

    def test_path_traversal_blocked(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("/static/../../server.py")
        self.assertIn(ctx.exception.code, (403, 404))


class StatusTests(_ServerCase):
    def test_status(self):
        _status, body = self._get("/api/status")
        data = json.loads(body)
        self.assertTrue(data["ok"])
        self.assertEqual(data["data"]["model"], "llama3")
        self.assertEqual(data["data"]["wake_word"], "afnan")
        self.assertFalse(data["data"]["voice"])


class ChatTests(_ServerCase):
    def test_chat_accepted_and_dispatched(self):
        self.agent.handle_request = mock.MagicMock()
        _status, data = self._post("/api/chat", {"text": "hi"})
        self.assertTrue(data["ok"])
        self.assertTrue(data["accepted"])
        time.sleep(0.5)
        self.agent.handle_request.assert_called_with("hi")

    def test_chat_empty_rejected(self):
        _status, data = self._post("/api/chat", {"text": "  "})
        self.assertFalse(data["ok"])


class SseTests(_ServerCase):
    def _open_sse(self) -> socket.socket:
        sock = socket.create_connection(("localhost", _PORT),
                                        timeout=10)
        sock.sendall(b"GET /api/events HTTP/1.1\r\n"
                     b"Host: x\r\n"
                     b"Accept: text/event-stream\r\n\r\n")
        sock.settimeout(6)
        return sock

    def _read_event(self, sock: socket.socket) -> bytes:
        data = b""
        while data.count(b"\n\n") < 1:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
        return data.split(b"\r\n\r\n", 1)[-1]

    def test_hello_framed(self):
        sock = self._open_sse()
        try:
            body = self._read_event(sock)
            self.assertTrue(body.startswith(b"data: "))
            self.assertIn(b'"type": "hello"', body)
        finally:
            sock.close()

    def test_broadcast_reaches_client(self):
        sock = self._open_sse()
        try:
            self._read_event(sock)  # hello
            self.srv.broadcast("activity", {"kind": "ping"})
            data = b""
            while b"ping" not in data:
                data += sock.recv(4096)
            self.assertIn(b"ping", data)
        finally:
            sock.close()

    def test_speak_hook_broadcasts_chat(self):
        sock = self._open_sse()
        try:
            self._read_event(sock)  # hello
            self.agent.speak("hello boss")
            data = b""
            while b"hello boss" not in data:
                data += sock.recv(4096)
            self.assertIn(b'"who": "afnan"', data)
        finally:
            sock.close()


class VoiceTests(_ServerCase):
    def test_toggle_on_off(self):
        stop_ev = threading.Event()
        self.agent.start.side_effect = lambda: stop_ev.wait(30)
        _status, data = self._post("/api/voice/toggle", {})
        self.assertTrue(data["ok"])
        self.assertTrue(data["data"]["voice"])
        _status, data = self._post("/api/voice/toggle", {})
        self.assertTrue(data["ok"])
        self.assertFalse(data["data"]["voice"])
        stop_ev.set()


class ManagerTests(_ServerCase):
    def test_tasks(self):
        task = mock.MagicMock()
        task.to_dict.return_value = {
            "task_id": "task_1", "title": "demo",
            "status": "pending"}
        self.agent.task_manager.list.return_value = [task]
        _status, body = self._get("/api/tasks")
        data = json.loads(body)
        self.assertTrue(data["ok"])
        self.assertEqual(data["data"][0]["task_id"], "task_1")

        self.agent.task_manager.enqueue.return_value = mock.MagicMock(
            task_id="task_2")
        _status, data = self._post("/api/tasks/add",
                                   {"title": "x"})
        self.assertTrue(data["ok"])

        _status, data = self._post("/api/tasks/complete",
                                   {"id": "task_1"})
        self.assertTrue(data["ok"])

    def test_goals(self):
        goal = mock.MagicMock()
        goal.to_dict.return_value = {"goal_id": "g1"}
        self.agent.goal_manager.list.return_value = [goal]
        _status, body = self._get("/api/goals")
        self.assertTrue(json.loads(body)["ok"])

    def test_memory(self):
        rec = mock.MagicMock()
        rec.text = "hi"
        rec.memory_id = "m1"
        rec.kind = "fact"
        self.agent.memory_store.list.return_value = [rec]
        _status, body = self._get("/api/memory")
        data = json.loads(body)
        self.assertTrue(data["ok"])
        self.assertEqual(data["data"][0]["text"], "hi")


class BrowserApiTests(_ServerCase):
    def setUp(self):
        res = mock.MagicMock()
        res.success = True
        res.output = {"url": "https://example.com"}
        self.agent.tools.execute.return_value = res

    def test_endpoints(self):
        for path, payload in [
            ("/api/browser/launch", {}),
            ("/api/browser/navigate", {"url": "example.com"}),
            ("/api/browser/back", {}),
            ("/api/browser/forward", {}),
            ("/api/browser/reload", {}),
            ("/api/browser/tabs", {}),
        ]:
            _status, data = self._post(path, payload)
            self.assertTrue(data["ok"], (path, data))

    def test_navigate_needs_url(self):
        _status, data = self._post("/api/browser/navigate", {})
        self.assertFalse(data["ok"])


class RobustnessTests(_ServerCase):
    def test_unknown_route_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("/api/nope")
        self.assertEqual(ctx.exception.code, 404)

    def test_malformed_json_no_crash(self):
        req = urllib.request.Request(
            self.base + "/api/chat", data=b"{not json",
            headers={"Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req,
                                        timeout=10) as r:
                data = json.loads(r.read())
            self.assertFalse(data["ok"])
        except urllib.error.HTTPError as e:
            self.assertIn(e.code, (400, 500))

    def test_slow_client_dropped(self):
        q: queue.Queue = queue.Queue(maxsize=1)
        q.put_nowait(b"x")  # full
        self.srv._sse_clients.add(q)
        self.srv.broadcast("activity", {"k": 1})  # must not block
        self.assertNotIn(q, self.srv._sse_clients)

    def test_server_idempotent_start_stop(self):
        srv2 = WebUIServer(self.agent, port=_PORT + 1)
        srv2.start(open_browser=False)
        srv2.start(open_browser=False)  # no-op
        srv2.stop()
        srv2.stop()  # no-op


if __name__ == "__main__":
    unittest.main()
