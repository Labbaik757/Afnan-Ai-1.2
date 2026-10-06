# 🤖 Afnan AI 1.2 (Windows + macOS + Linux Edition)

Afnan AI is a personal voice assistant built with Python and powered by Ollama. It can understand voice commands, open applications, search the web, play music, capture screenshots, and assist you with everyday tasks using natural voice interaction.

> **Version:** 1.2  
> **Platform:** Windows 10/11, macOS, Linux  
> **Language:** Python

---

# ✨ Features

- 🎤 Voice Recognition
- 🤖 Local AI Chat using Ollama (Llama 3)
- 🗣️ Text-to-Speech Responses (Windows SAPI / macOS `say`, or pyttsx3)
- 💻 Open Visual Studio Code
- 🌐 Open Google Chrome / Microsoft Edge
- 🧭 Open Safari (macOS) — on Windows/Linux it opens your default browser instead
- 💬 Open WhatsApp (app on macOS, WhatsApp/protocol or WhatsApp Web on Windows)
- ▶️ Open YouTube
- 📂 Smart Folder Search (Downloads, Desktop, Documents, Pictures, Music, Videos)
- 🎵 Play Songs on YouTube
- 🔎 Google Search
- 📺 YouTube Search
- 📸 Screenshot Capture
- 🎬 Startup GIF Animation (in your browser, on every platform)
- 🎯 Wake Word Detection ("Afnan") — on-device, private
- 🗣️ Urdu Voice (sunta aur bolta hai Urdu mein)
- 🌐 Browser Automation (navigate, search, extract)
- 🖥️ Desktop Control (verified computer-use actions)
- 🔬 Deep Research (evidence, citations, contradictions)
- 🧠 Memory, Goals, Tasks & Scheduling
- 🛡️ Safety First (sensitive actions need your approval)
- 📡 Remote Control (paired devices: monitor & control securely)
- 📱 Android App (native APK: pair, chat, tasks, approvals from your phone)
- ⚡ Fast Voice Command Processing

---

# 🚀 Advanced Features

| Feature | Power |
|---|---|
| 🏗️ OS Abstraction | Same code on Windows, macOS, Linux |
| 🧠 AgentState | One central task state |
| 🤖 LLMProvider | Swap AI models anytime |
| 🧰 ToolRegistry | Safe, validated tool execution |
| 🗺️ Planner | Builds verified task plans |
| ⚙️ Executor | Runs plans step by step |
| 🔍 Verifier | Confirms results, never assumes |
| 🧭 Agent | Manages full task lifecycle |
| 🩹 Recovery | Replans after any failure |
| 🌐 Browser Control | Automated web browsing |
| 🖥️ Computer Use | Verified desktop actions |
| 🎯 Wake Word | On-device "Afnan" detection |
| 🗣️ Urdu Voice | Listens and speaks Urdu |
| 🔬 Deep Research | Evidence, citations, contradictions |
| 🧠 Memory & Goals | Remembers, tracks, schedules |
| 🧩 Skills | Reusable capabilities |
| 🤝 Subagents | Parallel delegated work |
| 🔌 Connectors | External service integrations |
| 📄 Artifacts | Generated docs and files |
| 📊 Self-Evaluation | Benchmarks its own quality |
| 🛡️ Security | Approvals, vault, audit trail |
| 📡 Remote Control | Paired devices monitor & control securely |
| 📱 Android App | Native APK: pair, chat, tasks, approvals from your phone |

---

# 🎙️ Available Voice Commands

> **Urdu-first:** Afnan ab Urdu mein sunta aur bolta hai.
> Speech recognition `ur-PK` pe hoti hai aur jawab Urdu
> neural voice mein aata hai. English text pe purana
> system-engine behaviour rehta hai.

| Voice Command | Action |
|---------------|--------|
| Afnan | Activate the assistant |
| Open Visual Studio Code / Open VS Code | Opens VS Code |
| Open Chrome | Opens Google Chrome |
| Open Edge | Opens Microsoft Edge (Windows) |
| Open Safari | Opens Safari (macOS only) |
| Open YouTube | Opens YouTube |
| Open WhatsApp | Opens WhatsApp |
| Open Folder Downloads | Opens Downloads folder |
| Open Folder Desktop | Opens Desktop folder |
| Open Folder Documents | Opens Documents folder |
| Play Believer | Plays the requested song on YouTube |
| Search Google for Python | Searches Google |
| Search YouTube for AI | Searches YouTube |
| Screenshot | Captures a screenshot |
| Tell me about yourself | Afnan introduces itself |
| Introduce yourself | Afnan introduces itself |
| Who are you | Afnan introduces itself |
| Stop Afnan | Closes Afnan AI |

---

# 🛠️ Technologies Used

- Python
- SpeechRecognition
- Ollama
- PyWhatKit
- PyAutoGUI
- Webbrowser
- pyttsx3 (cross-platform text-to-speech, with OS speech as fallback)

---

# 📦 Requirements

- Python 3.10 or later
- Windows 10/11, macOS, or Linux
- Ollama Installed, with the Llama 3 model: `ollama pull llama3`
- Working Microphone (and microphone permission on Windows/macOS)
- Internet Connection (for Google speech recognition and online features)

### Windows note for PyAudio

On Windows, `pip install PyAudio` sometimes needs a wheel. If it fails, install it with:

```powershell
pip install pipwin
pipwin install pyaudio
```

or install a matching PyAudio wheel for your Python version, then run
`pip install -r requirements.txt` again.

---

# 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/Labbaik757/Afnan-Ai-1.2.git
cd Afnan-Ai-1.2
```

### Install dependencies

Windows (PowerShell / CMD):

```powershell
python -m pip install -r requirements.txt
```

macOS / Linux:

```bash
pip3 install -r requirements.txt
```

The macOS-only packages in `requirements.txt` are marked with
`sys_platform == 'darwin'`, so pip automatically skips them on Windows
and Linux — and the Windows-only `pywin32` is skipped on macOS/Linux.

### Run Afnan AI

Windows:

```powershell
python main.py
```

macOS / Linux:

```bash
python3 main.py
```

---

# 📄 requirements.txt (core)

```text
SpeechRecognition
PyAudio
PyAutoGUI
pywhatkit
ollama
pyttsx3
```

---

# 📁 Project Structure

```
Afnan-Ai-1.2
│
├── main.py                  (thin entry point, backwards compatible)
├── afnan_ai/
│   ├── agent.py             (core agent — no OS-specific code, no
│   │                         direct model-client calls)
│   ├── state.py             (centralized AgentState — goal, steps,
│   │                         observations, tool results, status)
│   ├── planner.py           (Planner — goal + AgentState + tools →
│   │                         validated TaskPlan, never executes)
│   ├── executor.py          (Executor — runs a TaskPlan step by step
│   │                         via ToolRegistry, records AgentState)
│   ├── verifier.py          (Verifier — checks actual results vs
│   │                         expected_result, never re-executes)
│   ├── orchestrator.py      (Agent — central orchestrator: state +
│   │                         planner + executor + verifier, full
│   │                         task lifecycle, max-iteration limit)
│   ├── recovery.py          (RecoveryManager — replans after a
│   │                         failed/uncertain step, never repeats
│   │                         a failed action, attempts recorded)
│   ├── config.py            (AgentConfig — wake word, limits,
│   │                         default model; env overridable)
│   ├── log_config.py        (structured logging, stdlib only)
│   ├── browser/             (BrowserController — launch/connect,
│   │   │                     tabs, navigation, page state, shutdown)
│   │   ├── base.py          (tab/page types + structured errors)
│   │   ├── backend.py       (BrowserBackend interface +
│   │   │                     PlaywrightBackend)
│   │   ├── controller.py    (session/tab management, validation)
│   │   └── tools.py         (browser_* Tools for the registry)
│   ├── llm/
│   │   ├── base.py          (LLMProvider interface + typed errors)
│   │   ├── ollama.py        (OllamaProvider — local Ollama, llama3)
│   │   └── factory.py       (provider registry / default provider)
│   ├── tools/
│   │   ├── base.py          (Tool interface + structured errors)
│   │   ├── registry.py      (ToolRegistry — register/get/execute)
│   │   └── builtin.py       (open_url, open_application,
│   │                         search_google, take_screenshot, ...)
│   ├── speech.py            (pyttsx3 first, adapter speech as fallback)
│   ├── wakeword.py          (on-device "Afnan" wake-word detection)
│   ├── memory_store.py      (persistent memory)
│   ├── goal_manager.py      (goals & milestones)
│   ├── task_manager.py      (long-running tasks)
│   ├── scheduler.py         (scheduled & background tasks)
│   ├── skills/              (dynamic skill system)
│   ├── subagents/           (multi-agent delegation)
│   ├── connectors/          (external service integrations)
│   ├── artifacts/           (generated documents & files)
│   ├── context/             (long-context & trajectory reasoning)
│   ├── research/            (research with evidence & citations)
│   ├── evaluation/          (self-evaluation & benchmarking)
│   ├── security/            (permissions, vault, audit)
│   ├── activity/            (activity center & approvals)
│   ├── workspace/           (secure isolated task workspaces)
│   ├── computer/            (desktop automation runtime)
│   ├── proactive/           (proactive suggestions)
│   ├── control/             (remote & multi-device control plane)
│   └── platform/
│       ├── base.py          (PlatformAdapter interface)
│       ├── factory.py       (auto-selects the adapter at runtime)
│       ├── windows.py       (PowerShell speech, startfile, cmd start)
│       ├── macos.py         (say, open / open -a, mdfind)
│       └── linux.py         (espeak/spd-say, xdg-open)
├── android/                (native Android APK client)
├── tests/                   (unit + integration + end-to-end tests,
│                             run on any host OS)
├── tools/check_compatibility.py
├── requirements.txt
├── README.md
├── afnan_animation.gif
├── afnan_animation.html
└── screenshots/   (created when you take a screenshot)
```

# ✅ Compatibility Tests

The tests simulate all three operating systems (mocking
`platform.system`, `subprocess`, speech and the microphone), so
they run safely on any host:

```bash
python -m unittest discover -s tests -v
# or
python tools/check_compatibility.py
```

They cover adapter selection for Windows/macOS/Linux, each
adapter's speech/open/launch behaviour, known-folder lookup, and
that every existing voice command still routes to the same feature
as before the refactor.

---

# ⚙️ How It Works

1. Launch Afnan AI.
2. The startup animation will appear in your browser.
3. Afnan activates the microphone.
4. Say **"Afnan"** to wake the assistant.
5. Afnan replies **"Yes Boss"**.
6. Speak your command.
7. Afnan processes and executes your request, or asks the local
   Ollama Llama 3 model when no command matches.

At startup the platform factory detects your OS and picks the
matching adapter, so the same `main.py` opens apps, folders and
files natively on Windows, macOS and Linux without any manual
configuration. Folder search uses Spotlight on macOS and a
home-folder search on Windows/Linux.

---

# 💬 Example

```
You: Afnan

Afnan: Yes Boss

You: Open Chrome

Afnan: Opening Chrome
```

---

# ⚠️ Notes

- Windows, macOS and Linux are supported in this version.
- On Windows, speech works out of the box with the built-in SAPI
  voices (via pyttsx3 / PowerShell), no `say` command needed.
- Safari is a macOS app; on Windows/Linux "Open Safari" opens your
  default browser instead.
- Ollama must be installed and running for AI chat functionality.
- Make sure your microphone permission is enabled in your OS settings.

---

# 🚀 Afnan AI 1.7 — Coming Soon

Afnan AI 1.7 is currently under active development and will introduce a major upgrade over version 1.2.

### Planned Features

- 🧠 Smarter AI Engine
- ⚡ Faster Performance
- 🎨 Modern User Interface
- 🎙️ Improved Voice Recognition
- 🤖 Advanced AI Automation
- 🔥 More Powerful Voice Commands
- 💎 Exclusive Premium Features

Stay tuned for future updates.

Thank you for supporting Afnan AI! ❤️

---

# 👨‍💻 Author

Developed with ❤️ by **Afnan**

If you like this project, please consider giving it a ⭐ on GitHub.

---

# 📄 License

This project is licensed under the MIT License.
