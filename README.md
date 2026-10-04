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
- 🎯 Wake Word Detection ("Afnan")
- ⚡ Fast Voice Command Processing

---

# 🎙️ Available Voice Commands

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
Pillow
pyttsx3
```

---

# 📁 Project Structure

```
Afnan-Ai-1.2
│
├── main.py                  (thin entry point, backwards compatible)
├── afnan_ai/
│   ├── agent.py             (core agent — no OS-specific code)
│   ├── speech.py            (pyttsx3 first, adapter speech as fallback)
│   └── platform/
│       ├── base.py          (PlatformAdapter interface)
│       ├── factory.py       (auto-selects the adapter at runtime)
│       ├── windows.py       (PowerShell speech, startfile, cmd start)
│       ├── macos.py         (say, open / open -a, mdfind)
│       └── linux.py         (espeak/spd-say, xdg-open)
├── gif_viewer.py
├── tests/                   (compatibility tests, run on any host OS)
├── tools/check_compatibility.py
├── requirements.txt
├── README.md
├── afnan_animation.gif
├── afnan_animation.html
└── screenshots/   (created when you take a screenshot)
```

# 🏗️ Architecture — OS Abstraction Layer

Platform-specific code is isolated behind one interface, so the
core agent never touches an OS command directly:

- `afnan_ai/platform/base.py` — `PlatformAdapter`: `speak_system()`,
  `open_path()`, `launch_app()`, `find_folder()`
- `afnan_ai/platform/windows.py` / `macos.py` / `linux.py` — the
  only files containing Windows, macOS or Linux specific calls
- `afnan_ai/platform/factory.py` — `get_adapter()` reads
  `platform.system()` at runtime and returns the Windows, macOS
  or Linux adapter automatically — no configuration needed
- `afnan_ai/agent.py` — wake word, command routing, search, music,
  screenshots and Ollama fallback, all written against the adapter

A test in `tests/test_agent_commands.py` fails if an OS-specific
call ever leaks back into the core agent.

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
