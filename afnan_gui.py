"""Afnan AI — Control Center (full GUI).

Dashboard + Chat + Browser + Tasks + Goals + Memory + Activity +
Settings, all in one desktop app.  Type commands like you chat,
watch live activity, drive the browser visibly.

New entry point — reuses AfnanAgent as-is, modifies no existing
module.  Run with::

    python afnan_gui.py
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk

from afnan_ai.agent import AfnanAgent
from afnan_ai.config import AgentConfig
from afnan_ai.platform import get_adapter

# -- theme ---------------------------------------------------------------
BG = "#1a1a1a"
PANEL = "#242424"
CARD = "#2d2d2d"
FG = "#e8e8e8"
DIM = "#909090"
ACCENT = "#4da3ff"
GREEN = "#4caf50"
AMBER = "#ffb300"
RED = "#e05252"


class AfnanGUI:
    """Full control center: sidebar navigation + section frames."""

    SECTIONS = [
        ("📊", "Dashboard"),
        ("💬", "Chat"),
        ("🌐", "Browser"),
        ("✅", "Tasks"),
        ("🎯", "Goals"),
        ("🧠", "Memory"),
        ("⚡", "Activity"),
        ("⚙️", "Settings"),
    ]

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Afnan AI — Control Center")
        self.root.geometry("1200x720")
        self.root.configure(bg=BG)

        self.chat_q: queue.Queue = queue.Queue()
        self.activity_q: queue.Queue = queue.Queue()

        self.adapter = get_adapter()
        self.agent = AfnanAgent(
            adapter=self.adapter, config=AgentConfig.from_env()
        )
        self._hook_agent()

        self.voice_on = False
        self.frames: dict[str, tk.Frame] = {}
        self.current = "Dashboard"

        self._build_chrome()
        self._build_sidebar()
        self._build_sections()
        self.show("Dashboard")
        self._pump_queues()
        self.log_activity("Control Center ready.")

    # -- window chrome ----------------------------------------------------
    def _build_chrome(self) -> None:
        top = tk.Frame(self.root, bg=CARD, height=44)
        top.pack(fill=tk.X)
        top.pack_propagate(False)
        tk.Label(top, text="🤖 Afnan AI", bg=CARD, fg=FG,
                 font=("Segoe UI", 13, "bold")).pack(
            side=tk.LEFT, padx=12)
        self.status_dot = tk.Label(top, text="●", bg=CARD, fg=GREEN,
                                   font=("Segoe UI", 14))
        self.status_dot.pack(side=tk.LEFT)
        self.status_lbl = tk.Label(top, text="ready", bg=CARD, fg=DIM,
                                   font=("Segoe UI", 10))
        self.status_lbl.pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="⏹ Quit", command=self.root.quit,
                  bg="#5d2d2d", fg=FG, relief=tk.FLAT).pack(
            side=tk.RIGHT, padx=10, pady=6)

    def _build_sidebar(self) -> None:
        side = tk.Frame(self.root, bg=PANEL, width=150)
        side.pack(side=tk.LEFT, fill=tk.Y)
        side.pack_propagate(False)
        self.nav_btns: dict[str, tk.Button] = {}
        for icon, name in self.SECTIONS:
            btn = tk.Button(
                side, text=f"{icon}  {name}", anchor=tk.W,
                bg=PANEL, fg=FG, relief=tk.FLAT,
                font=("Segoe UI", 11), padx=14, pady=10,
                command=lambda n=name: self.show(n),
            )
            btn.pack(fill=tk.X, pady=1)
            self.nav_btns[name] = btn

    def _build_sections(self) -> None:
        self.container = tk.Frame(self.root, bg=BG)
        self.container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        builders = {
            "Dashboard": self._build_dashboard,
            "Chat": self._build_chat,
            "Browser": self._build_browser,
            "Tasks": self._build_tasks,
            "Goals": self._build_goals,
            "Memory": self._build_memory,
            "Activity": self._build_activity,
            "Settings": self._build_settings,
        }
        for name, builder in builders.items():
            frame = tk.Frame(self.container, bg=BG)
            builder(frame)
            self.frames[name] = frame

    def show(self, name: str) -> None:
        self.current = name
        for fname, frame in self.frames.items():
            frame.pack_forget()
        self.frames[name].pack(fill=tk.BOTH, expand=True,
                               padx=10, pady=10)
        for bname, btn in self.nav_btns.items():
            btn.configure(
                bg=CARD if bname == name else PANEL,
                fg=ACCENT if bname == name else FG,
            )
        # Refresh data sections on show.
        if name == "Tasks":
            self.refresh_tasks()
        elif name == "Goals":
            self.refresh_goals()
        elif name == "Memory":
            self.refresh_memory()
        elif name == "Dashboard":
            self.refresh_dashboard()

    # -- Dashboard ---------------------------------------------------------
    def _build_dashboard(self, f: tk.Frame) -> None:
        tk.Label(f, text="📊 Dashboard", bg=BG, fg=FG,
                 font=("Segoe UI", 16, "bold")).pack(anchor=tk.W,
                                                     pady=(0, 10))
        cards = tk.Frame(f, bg=BG)
        cards.pack(fill=tk.X)
        self.dash_cards: dict[str, tk.Label] = {}
        for title in ("Assistant", "Browser", "Voice", "Model"):
            card = tk.Frame(cards, bg=CARD, padx=16, pady=12)
            card.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            tk.Label(card, text=title, bg=CARD, fg=DIM,
                     font=("Segoe UI", 10)).pack(anchor=tk.W)
            val = tk.Label(card, text="—", bg=CARD, fg=FG,
                           font=("Segoe UI", 13, "bold"))
            val.pack(anchor=tk.W)
            self.dash_cards[title] = val

        tk.Label(f, text="Quick actions", bg=BG, fg=FG,
                 font=("Segoe UI", 12, "bold")).pack(
            anchor=tk.W, pady=(16, 6))
        qa = tk.Frame(f, bg=BG)
        qa.pack(fill=tk.X)
        for label, cmd in (
            ("💬 New chat", lambda: self.show("Chat")),
            ("🌐 Open browser", self.open_browser),
            ("🎤 Toggle voice", self.toggle_voice),
            ("📸 Screenshot", self.take_screenshot),
        ):
            tk.Button(qa, text=label, command=cmd, bg=CARD, fg=FG,
                      relief=tk.FLAT, padx=14, pady=8).pack(
                side=tk.LEFT, padx=5)

        tk.Label(f, text="Recent activity", bg=BG, fg=FG,
                 font=("Segoe UI", 12, "bold")).pack(
            anchor=tk.W, pady=(16, 6))
        self.dash_feed = scrolledtext.ScrolledText(
            f, bg=PANEL, fg=DIM, font=("Consolas", 9),
            height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.dash_feed.pack(fill=tk.BOTH, expand=True)

    def refresh_dashboard(self) -> None:
        try:
            model = getattr(self.agent.config, "llm_model", "?")
        except Exception:
            model = "?"
        self.dash_cards["Assistant"].configure(text="● Online",
                                               fg=GREEN)
        try:
            running = bool(
                getattr(self.agent.browser, "_running", False))
        except Exception:
            running = False
        self.dash_cards["Browser"].configure(
            text="● Running" if running else "○ Stopped",
            fg=GREEN if running else DIM)
        self.dash_cards["Voice"].configure(
            text="● ON" if self.voice_on else "○ OFF",
            fg=GREEN if self.voice_on else DIM)
        self.dash_cards["Model"].configure(text=str(model)[:24])

    # -- Chat ---------------------------------------------------------------
    def _build_chat(self, f: tk.Frame) -> None:
        tk.Label(f, text="💬 Chat — type anything", bg=BG, fg=FG,
                 font=("Segoe UI", 14, "bold")).pack(anchor=tk.W,
                                                     pady=(0, 8))
        self.chat = scrolledtext.ScrolledText(
            f, bg=PANEL, fg=FG, font=("Segoe UI", 11),
            wrap=tk.WORD, state=tk.DISABLED)
        self.chat.pack(fill=tk.BOTH, expand=True)
        bottom = tk.Frame(f, bg=BG)
        bottom.pack(fill=tk.X, pady=(8, 0))
        self.entry = tk.Entry(bottom, bg=CARD, fg=FG,
                              font=("Segoe UI", 11), relief=tk.FLAT)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True,
                        padx=(0, 6), ipady=9)
        self.entry.bind("<Return>", lambda _e: self.send())
        tk.Button(bottom, text="Send ➤", command=self.send,
                  bg="#0d5c0d", fg="white", relief=tk.FLAT,
                  padx=18).pack(side=tk.RIGHT)
        self.voice_btn = tk.Button(
            bottom, text="🎤 Voice: OFF", command=self.toggle_voice,
            bg=CARD, fg=FG, relief=tk.FLAT, padx=12)
        self.voice_btn.pack(side=tk.RIGHT, padx=6)

    def send(self) -> None:
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        self.chat_q.put(("you", text))
        self.set_status("working...")
        threading.Thread(target=self._run_request, args=(text,),
                         daemon=True).start()

    def _run_request(self, text: str) -> None:
        try:
            self.agent.handle_request(text)
        except SystemExit:
            self.activity_q.put("👋 stopping")
            self.root.quit()
        except Exception as e:
            self.activity_q.put(f"❌ error: {e}")
        finally:
            self.set_status("ready")

    # -- Browser --------------------------------------------------------------
    def _build_browser(self, f: tk.Frame) -> None:
        tk.Label(f, text="🌐 Browser", bg=BG, fg=FG,
                 font=("Segoe UI", 14, "bold")).pack(anchor=tk.W,
                                                     pady=(0, 8))
        nav = tk.Frame(f, bg=BG)
        nav.pack(fill=tk.X, pady=(0, 8))
        for label, cmd in (("◀", "back"), ("▶", "forward"),
                           ("↻", "reload")):
            tk.Button(nav, text=label,
                      command=lambda c=cmd: self.browser_nav(c),
                      bg=CARD, fg=FG, relief=tk.FLAT,
                      width=4).pack(side=tk.LEFT, padx=2)
        self.url_entry = tk.Entry(nav, bg=CARD, fg=FG,
                                  font=("Segoe UI", 10), relief=tk.FLAT)
        self.url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True,
                            padx=6, ipady=7)
        self.url_entry.bind("<Return>", lambda _e: self.browser_go())
        tk.Button(nav, text="Go ➤", command=self.browser_go,
                  bg="#0d5c0d", fg="white", relief=tk.FLAT,
                  padx=14).pack(side=tk.LEFT)
        ctl = tk.Frame(f, bg=BG)
        ctl.pack(fill=tk.X, pady=(0, 8))
        for label, cmd in (
            ("🚀 Launch", self.open_browser),
            ("📸 Screenshot", self.take_screenshot),
            ("📑 Tabs", self.browser_tabs),
            ("📄 Page info", self.browser_info),
        ):
            tk.Button(ctl, text=label, command=cmd, bg=CARD, fg=FG,
                      relief=tk.FLAT, padx=12).pack(side=tk.LEFT,
                                                    padx=4)
        self.browser_log = scrolledtext.ScrolledText(
            f, bg=PANEL, fg=FG, font=("Consolas", 10),
            wrap=tk.WORD, state=tk.DISABLED, height=18)
        self.browser_log.pack(fill=tk.BOTH, expand=True)

    def _browser_tool(self, name: str, args: dict | None = None) -> None:
        def _run() -> None:
            try:
                res = self.agent.tools.execute(name, args or {})
                ok = getattr(res, "ok", True)
                self.activity_q.put(
                    f"{'✅' if ok else '❌'} browser_{name}")
                if name == "browser_current_page" and ok:
                    data = getattr(res, "data", res)
                    self.browser_q.put(str(data)[:2000])
            except Exception as e:
                self.activity_q.put(f"❌ browser {name}: {e}")

        threading.Thread(target=_run, daemon=True).start()

    browser_q: queue.Queue = queue.Queue()  # type: ignore[assignment]

    def browser_go(self) -> None:
        url = self.url_entry.get().strip()
        if not url:
            return
        if "://" not in url:
            url = "https://" + url
        self._browser_tool("browser_navigate", {"url": url})
        self.blog(f"→ {url}")

    def browser_nav(self, action: str) -> None:
        self._browser_tool(f"browser_{action}")

    def browser_tabs(self) -> None:
        self._browser_tool("browser_list_tabs")

    def browser_info(self) -> None:
        self._browser_tool("browser_current_page")

    def blog(self, msg: str) -> None:
        self.browser_log.config(state=tk.NORMAL)
        self.browser_log.insert(tk.END, msg + "\n")
        self.browser_log.see(tk.END)
        self.browser_log.config(state=tk.DISABLED)

    # -- Tasks ------------------------------------------------------------------
    def _build_tasks(self, f: tk.Frame) -> None:
        tk.Label(f, text="✅ Tasks", bg=BG, fg=FG,
                 font=("Segoe UI", 14, "bold")).pack(anchor=tk.W,
                                                     pady=(0, 8))
        top = tk.Frame(f, bg=BG)
        top.pack(fill=tk.X, pady=(0, 8))
        self.task_entry = tk.Entry(top, bg=CARD, fg=FG,
                                   font=("Segoe UI", 10),
                                   relief=tk.FLAT)
        self.task_entry.pack(side=tk.LEFT, fill=tk.X, expand=True,
                             padx=(0, 6), ipady=7)
        self.task_entry.bind("<Return>", lambda _e: self.add_task())
        tk.Button(top, text="+ Add task", command=self.add_task,
                  bg="#0d5c0d", fg="white", relief=tk.FLAT,
                  padx=14).pack(side=tk.LEFT)
        tk.Button(top, text="↻ Refresh", command=self.refresh_tasks,
                  bg=CARD, fg=FG, relief=tk.FLAT,
                  padx=10).pack(side=tk.LEFT, padx=6)
        self.task_list = tk.Frame(f, bg=BG)
        self.task_list.pack(fill=tk.BOTH, expand=True)

    def refresh_tasks(self) -> None:
        for w in self.task_list.winfo_children():
            w.destroy()
        try:
            tasks = self.agent.task_manager.list()
        except Exception as e:
            tk.Label(self.task_list, text=f"Error: {e}", bg=BG,
                     fg=RED).pack(anchor=tk.W)
            return
        if not tasks:
            tk.Label(self.task_list, text="No tasks yet.", bg=BG,
                     fg=DIM).pack(anchor=tk.W)
            return
        for t in tasks:
            row = tk.Frame(self.task_list, bg=CARD, pady=4)
            row.pack(fill=tk.X, pady=2, padx=2)
            status = getattr(t, "status", "?")
            color = GREEN if status in ("done", "completed") else FG
            tk.Label(row, text=f"● {getattr(t, 'title', '?')}",
                     bg=CARD, fg=color,
                     font=("Segoe UI", 10)).pack(side=tk.LEFT,
                                                  padx=10)
            tk.Label(row, text=str(status), bg=CARD, fg=DIM,
                     font=("Segoe UI", 9)).pack(side=tk.LEFT)
            tid = getattr(t, "task_id", "")
            if status not in ("done", "completed"):
                tk.Button(row, text="✓ Done",
                          command=lambda i=tid: self.complete_task(i),
                          bg=CARD, fg=GREEN, relief=tk.FLAT
                          ).pack(side=tk.RIGHT, padx=8)

    def add_task(self) -> None:
        title = self.task_entry.get().strip()
        if not title:
            return
        self.task_entry.delete(0, tk.END)

        def _run() -> None:
            try:
                self.agent.task_manager.enqueue(title=title)
                self.activity_q.put(f"✅ task added: {title}")
            except Exception as e:
                self.activity_q.put(f"❌ task add failed: {e}")
            self.root.after(0, self.refresh_tasks)

        threading.Thread(target=_run, daemon=True).start()

    def complete_task(self, task_id: str) -> None:
        def _run() -> None:
            try:
                self.agent.task_manager.complete(task_id)
                self.activity_q.put("✅ task completed")
            except Exception as e:
                self.activity_q.put(f"❌ complete failed: {e}")
            self.root.after(0, self.refresh_tasks)

        threading.Thread(target=_run, daemon=True).start()

    # -- Goals ------------------------------------------------------------------
    def _build_goals(self, f: tk.Frame) -> None:
        tk.Label(f, text="🎯 Goals", bg=BG, fg=FG,
                 font=("Segoe UI", 14, "bold")).pack(anchor=tk.W,
                                                     pady=(0, 8))
        tk.Button(f, text="↻ Refresh", command=self.refresh_goals,
                  bg=CARD, fg=FG, relief=tk.FLAT).pack(anchor=tk.W,
                                                       pady=(0, 8))
        self.goal_list = tk.Frame(f, bg=BG)
        self.goal_list.pack(fill=tk.BOTH, expand=True)

    def refresh_goals(self) -> None:
        for w in self.goal_list.winfo_children():
            w.destroy()
        try:
            goals = self.agent.goal_manager.list()
        except Exception as e:
            tk.Label(self.goal_list, text=f"Error: {e}", bg=BG,
                     fg=RED).pack(anchor=tk.W)
            return
        if not goals:
            tk.Label(self.goal_list, text="No goals yet.", bg=BG,
                     fg=DIM).pack(anchor=tk.W)
            return
        for g in goals:
            card = tk.Frame(self.goal_list, bg=CARD, padx=12,
                            pady=8)
            card.pack(fill=tk.X, pady=4)
            title = getattr(g, "title", getattr(g, "name", "?"))
            tk.Label(card, text=title, bg=CARD, fg=FG,
                     font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)
            desc = getattr(g, "description", "")
            if desc:
                tk.Label(card, text=str(desc)[:120], bg=CARD,
                         fg=DIM, font=("Segoe UI", 9),
                         wraplength=700, justify=tk.LEFT).pack(
                    anchor=tk.W)

    # -- Memory -------------------------------------------------------------------
    def _build_memory(self, f: tk.Frame) -> None:
        tk.Label(f, text="🧠 Memory", bg=BG, fg=FG,
                 font=("Segoe UI", 14, "bold")).pack(anchor=tk.W,
                                                     pady=(0, 8))
        tk.Button(f, text="↻ Refresh", command=self.refresh_memory,
                  bg=CARD, fg=FG, relief=tk.FLAT).pack(anchor=tk.W,
                                                       pady=(0, 8))
        self.memory_view = scrolledtext.ScrolledText(
            f, bg=PANEL, fg=FG, font=("Segoe UI", 10),
            wrap=tk.WORD, state=tk.DISABLED)
        self.memory_view.pack(fill=tk.BOTH, expand=True)

    def refresh_memory(self) -> None:
        self.memory_view.config(state=tk.NORMAL)
        self.memory_view.delete("1.0", tk.END)
        try:
            items = self.agent.memory_store.recent(limit=30)
        except Exception:
            try:
                items = self.agent.memory_store.list()[-30:]
            except Exception as e:
                items = []
                self.memory_view.insert(tk.END, f"Error: {e}")
        for item in items or []:
            text = getattr(item, "text",
                           getattr(item, "content", str(item)))
            self.memory_view.insert(tk.END, f"• {text}\n\n")
        if not (items or []):
            self.memory_view.insert(tk.END, "No memories yet.")
        self.memory_view.config(state=tk.DISABLED)

    # -- Activity -------------------------------------------------------------------
    def _build_activity(self, f: tk.Frame) -> None:
        tk.Label(f, text="⚡ Live Activity", bg=BG, fg=FG,
                 font=("Segoe UI", 14, "bold")).pack(anchor=tk.W,
                                                     pady=(0, 8))
        self.activity = scrolledtext.ScrolledText(
            f, bg=PANEL, fg="#a0d0a0", font=("Consolas", 10),
            wrap=tk.WORD, state=tk.DISABLED)
        self.activity.pack(fill=tk.BOTH, expand=True)

    # -- Settings ---------------------------------------------------------------------
    def _build_settings(self, f: tk.Frame) -> None:
        tk.Label(f, text="⚙️ Settings", bg=BG, fg=FG,
                 font=("Segoe UI", 14, "bold")).pack(anchor=tk.W,
                                                     pady=(0, 12))
        cfg = self.agent.config
        rows = [
            ("Model", getattr(cfg, "llm_model", "?")),
            ("Wake word", getattr(cfg, "wake_word", "?")),
            ("STT language", getattr(cfg, "stt_language", "?")),
            ("Fast path", str(getattr(cfg, "fast_path", "?"))),
            ("Conversation mode",
             str(getattr(cfg, "conversation_mode", "?"))),
            ("Stream responses",
             str(getattr(cfg, "stream_responses", "?"))),
            ("Platform", self.adapter.name),
        ]
        for label, value in rows:
            row = tk.Frame(f, bg=BG)
            row.pack(fill=tk.X, pady=3)
            tk.Label(row, text=label, bg=BG, fg=DIM,
                     font=("Segoe UI", 10), width=20,
                     anchor=tk.W).pack(side=tk.LEFT)
            tk.Label(row, text=str(value), bg=CARD, fg=FG,
                     font=("Segoe UI", 10), anchor=tk.W,
                     padx=10).pack(side=tk.LEFT, fill=tk.X,
                                    expand=True)
        tk.Label(f, text="Restart the app to apply .env changes.",
                 bg=BG, fg=DIM, font=("Segoe UI", 9)).pack(
            anchor=tk.W, pady=(16, 0))

    # -- shared actions ----------------------------------------------------------
    def open_browser(self) -> None:
        self.log_activity("🌐 launching browser...")
        self.set_status("launching browser...")

        def _launch() -> None:
            try:
                self.agent.browser.launch()
                self.activity_q.put("🌐 browser launched")
            except Exception as e:
                self.activity_q.put(f"❌ browser failed: {e}")
            finally:
                self.set_status("ready")

        threading.Thread(target=_launch, daemon=True).start()

    def take_screenshot(self) -> None:
        def _run() -> None:
            try:
                path = self.agent.take_screenshot()
                self.activity_q.put(f"📸 screenshot: {path}")
            except Exception as e:
                self.activity_q.put(f"❌ screenshot failed: {e}")

        threading.Thread(target=_run, daemon=True).start()

    def toggle_voice(self) -> None:
        self.voice_on = not self.voice_on
        label = f"🎤 Voice: {'ON' if self.voice_on else 'OFF'}"
        if hasattr(self, "voice_btn"):
            self.voice_btn.config(text=label)
        self.log_activity(f"🎤 voice {'ON — say Afnan' if self.voice_on else 'OFF'}")
        self.refresh_dashboard()
        if self.voice_on:
            threading.Thread(target=self._voice_loop,
                             daemon=True).start()

    def _voice_loop(self) -> None:
        try:
            self.agent.start()
        except Exception as e:
            self.activity_q.put(f"🎤 voice loop ended: {e}")

    # -- agent hooks (no existing code modified) --------------------------------------
    def _hook_agent(self) -> None:
        orig_speak = self.agent.speak

        def speak_and_show(text: str) -> None:
            self.chat_q.put(("afnan", text))
            orig_speak(text)

        self.agent.speak = speak_and_show  # type: ignore[method-assign]

        registry = self.agent.tools
        orig_execute = registry.execute

        def execute_and_log(*args, **kwargs):  # type: ignore[no-untyped-def]
            tool_name = args[0] if args else kwargs.get("name", "?")
            self.activity_q.put(f"🔧 tool: {tool_name}")
            return orig_execute(*args, **kwargs)

        registry.execute = execute_and_log  # type: ignore[method-assign]

    # -- thread-safe GUI updates ---------------------------------------------------------
    def log_activity(self, msg: str) -> None:
        self.activity_q.put(msg)

    def set_status(self, msg: str) -> None:
        self.root.after(0, lambda: self.status_lbl.config(text=msg))

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
                msg = self.activity_q.get_nowait()
                self._append(self.activity, msg + "\n", None)
                if hasattr(self, "dash_feed"):
                    self._append(self.dash_feed, msg + "\n", None)
        except queue.Empty:
            pass
        try:
            while True:
                self.blog("ℹ " + self.browser_q.get_nowait())
        except queue.Empty:
            pass
        self.root.after(200, self._pump_queues)

    def _append(self, widget: scrolledtext.ScrolledText, text: str,
                color: str | None) -> None:
        try:
            widget.config(state=tk.NORMAL)
            if color:
                tag = f"tag_{color}"
                widget.tag_config(tag, foreground=color)
                widget.insert(tk.END, text, tag)
            else:
                widget.insert(tk.END, text)
            widget.see(tk.END)
            widget.config(state=tk.DISABLED)
        except tk.TclError:
            pass  # widget not built yet


def main() -> None:
    root = tk.Tk()
    AfnanGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
