"""Afnan AI — Control Center (GUI).

A desktop GUI for the Afnan assistant: type commands like you chat,
watch live activity, and see the browser automation happen.

This is a *new* entry point — it reuses AfnanAgent as-is and does not
modify any existing module.  Run with::

    python afnan_gui.py
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import scrolledtext

from afnan_ai.agent import AfnanAgent
from afnan_ai.config import AgentConfig
from afnan_ai.platform import get_adapter


class AfnanGUI:
    """Chat + live-activity control center for the Afnan assistant."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Afnan AI — Control Center")
        self.root.geometry("1000x650")

        # Thread-safe message queues (background threads -> GUI).
        self.chat_q: queue.Queue = queue.Queue()
        self.activity_q: queue.Queue = queue.Queue()

        # Agent (same as main.py, no modifications to its code).
        self.adapter = get_adapter()
        self.agent = AfnanAgent(
            adapter=self.adapter, config=AgentConfig.from_env()
        )
        self._hook_agent()

        self.voice_on = False
        self.voice_thread: threading.Thread | None = None

        self._build_ui()
        self._pump_queues()
        self.log_activity("Control Center ready. Type a command below, "
                          "or toggle voice.")

    # -- UI ---------------------------------------------------------
    def _build_ui(self) -> None:
        bg, fg = "#1e1e1e", "#e0e0e0"
        self.root.configure(bg=bg)

        # Top bar: controls.
        bar = tk.Frame(self.root, bg="#2d2d2d")
        bar.pack(fill=tk.X, padx=5, pady=5)
        self.voice_btn = tk.Button(
            bar, text="🎤 Voice: OFF", command=self.toggle_voice,
            bg="#3d3d3d", fg=fg, relief=tk.FLAT, padx=10,
        )
        self.voice_btn.pack(side=tk.LEFT, padx=5)
        tk.Button(
            bar, text="🌐 Open Browser", command=self.open_browser,
            bg="#3d3d3d", fg=fg, relief=tk.FLAT, padx=10,
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            bar, text="⏹ Stop", command=self.root.quit,
            bg="#5d2d2d", fg=fg, relief=tk.FLAT, padx=10,
        ).pack(side=tk.RIGHT, padx=5)

        # Main split: chat (left) + activity (right).
        main = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=bg)
        main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        chat_frame = tk.Frame(main, bg=bg)
        tk.Label(chat_frame, text="💬 Chat", bg=bg, fg=fg,
                 font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)
        self.chat = scrolledtext.ScrolledText(
            chat_frame, bg="#252525", fg=fg, font=("Segoe UI", 10),
            wrap=tk.WORD, state=tk.DISABLED,
        )
        self.chat.pack(fill=tk.BOTH, expand=True)
        main.add(chat_frame, minsize=450)

        act_frame = tk.Frame(main, bg=bg)
        tk.Label(act_frame, text="⚡ Live Activity", bg=bg, fg=fg,
                 font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)
        self.activity = scrolledtext.ScrolledText(
            act_frame, bg="#252525", fg="#a0d0a0", font=("Consolas", 9),
            wrap=tk.WORD, state=tk.DISABLED,
        )
        self.activity.pack(fill=tk.BOTH, expand=True)
        main.add(act_frame, minsize=300)

        # Bottom: input.
        bottom = tk.Frame(self.root, bg=bg)
        bottom.pack(fill=tk.X, padx=5, pady=5)
        self.entry = tk.Entry(bottom, bg="#2d2d2d", fg=fg,
                              font=("Segoe UI", 11), relief=tk.FLAT)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True,
                        padx=(0, 5), ipady=8)
        self.entry.bind("<Return>", lambda _e: self.send())
        tk.Button(bottom, text="Send ➤", command=self.send,
                  bg="#0d5c0d", fg="white", relief=tk.FLAT,
                  padx=15).pack(side=tk.RIGHT)

    # -- agent hooks (no existing code modified) ---------------------
    def _hook_agent(self) -> None:
        # Show every spoken reply in the chat panel too.
        orig_speak = self.agent.speak

        def speak_and_show(text: str) -> None:
            self.chat_q.put(("afnan", text))
            orig_speak(text)

        self.agent.speak = speak_and_show  # type: ignore[method-assign]

        # Log every tool execution to the activity panel.
        registry = self.agent.tools
        orig_execute = registry.execute

        def execute_and_log(*args, **kwargs):  # type: ignore[no-untyped-def]
            tool_name = args[0] if args else kwargs.get("name", "?")
            self.activity_q.put(f"🔧 tool: {tool_name}")
            return orig_execute(*args, **kwargs)

        registry.execute = execute_and_log  # type: ignore[method-assign]

    # -- actions ------------------------------------------------------
    def send(self) -> None:
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        self.chat_q.put(("you", text))
        self.log_activity(f"📨 you: {text}")
        threading.Thread(target=self._run_request, args=(text,),
                         daemon=True).start()

    def _run_request(self, text: str) -> None:
        try:
            self.activity_q.put("⏳ working...")
            self.agent.handle_request(text)
            self.activity_q.put("✅ done")
        except SystemExit:
            self.activity_q.put("👋 stopping")
            self.root.quit()
        except Exception as e:  # never kill the GUI on a task error
            self.activity_q.put(f"❌ error: {e}")

    def open_browser(self) -> None:
        self.log_activity("🌐 launching browser...")

        def _launch() -> None:
            try:
                ctrl = self.agent.browser
                if ctrl is None:
                    self.activity_q.put("❌ no browser controller")
                    return
                ctrl.launch()
                self.activity_q.put("🌐 browser launched (visible window)")
            except Exception as e:
                self.activity_q.put(f"❌ browser failed: {e}")

        threading.Thread(target=_launch, daemon=True).start()

    def toggle_voice(self) -> None:
        self.voice_on = not self.voice_on
        self.voice_btn.config(
            text=f"🎤 Voice: {'ON' if self.voice_on else 'OFF'}")
        if self.voice_on:
            self.log_activity("🎤 voice wake word ON — say 'Afnan'")
            self.voice_thread = threading.Thread(
                target=self._voice_loop, daemon=True)
            self.voice_thread.start()
        else:
            self.log_activity("🎤 voice OFF")

    def _voice_loop(self) -> None:
        try:
            self.agent.start()
        except Exception as e:
            self.activity_q.put(f"🎤 voice loop ended: {e}")

    # -- thread-safe GUI updates --------------------------------------
    def log_activity(self, msg: str) -> None:
        self.activity_q.put(msg)

    def _pump_queues(self) -> None:
        try:
            while True:
                who, text = self.chat_q.get_nowait()
                self._append(self.chat, f"{who}: {text}\n",
                             "#7ec8ff" if who == "afnan" else "#ffd27e")
        except queue.Empty:
            pass
        try:
            while True:
                self._append(self.activity,
                             self.activity_q.get_nowait() + "\n", None)
        except queue.Empty:
            pass
        self.root.after(200, self._pump_queues)

    def _append(self, widget: scrolledtext.ScrolledText, text: str,
                color: str | None) -> None:
        widget.config(state=tk.NORMAL)
        if color:
            tag = f"tag_{color}"
            widget.tag_config(tag, foreground=color)
            widget.insert(tk.END, text, tag)
        else:
            widget.insert(tk.END, text)
        widget.see(tk.END)
        widget.config(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    AfnanGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
