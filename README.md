# Afnan AI 1.2

A cross-platform (Windows / macOS / Linux) personal voice assistant in Python, powered by a local LLM. Say **"Afnan"** to wake it, speak your command, and it responds — in Urdu or English.

## Features

- Voice wake word ("Afnan") with on-device detection
- Urdu speech recognition (`ur-PK`) and Urdu neural text-to-speech
- Local AI chat (Ollama / Llama 3) — no cloud account needed
- Open apps, folders and files by voice
- Web and YouTube search, play songs on YouTube
- Screenshot capture
- Autonomous agent: plans tasks, uses tools, verifies results, recovers from failures
- Browser automation and desktop (computer-use) control
- Research with evidence, citations and contradiction detection
- Skills, subagents, memory, goals and scheduled tasks
- Security-first: human approval for sensitive actions, secrets stay in a vault, untrusted web content is treated as data

## Requirements

- Python 3.10+
- Windows 10/11, macOS, or Linux
- Ollama with the Llama 3 model (`ollama pull llama3`)
- Microphone (with OS permission enabled)
- Internet connection (for speech recognition and online features)

## Installation

```bash
git clone https://github.com/Labbaik757/Afnan-Ai-1.2.git
cd Afnan-Ai-1.2
pip install -r requirements.txt
```

On Windows, if `PyAudio` fails to build, install it via `pipwin` or a matching wheel first, then re-run the command above. macOS-only and Windows-only packages in `requirements.txt` are platform-marked, so pip skips the ones that don't apply.

## Usage

```bash
python main.py        # Windows
python3 main.py      # macOS / Linux
```

1. Say **"Afnan"** — it replies "Yes Boss".
2. Speak your command, e.g. "Open Chrome", "Search Google for Python", "Screenshot".
3. Anything it doesn't recognize as a command goes to the local AI model.

## Configuration

Key environment variables (all optional):

| Variable | Purpose | Default |
|---|---|---|
| `AFNAN_STT_LANGUAGE` | Speech recognition language | `ur-PK` |
| `AFNAN_TTS_URDU_VOICE` | Urdu TTS voice | `ur-PK-GulNawazNeural` |
| `AFNAN_WAKEWORD_MODEL` | Path to a custom wake-word model | built-in model |
| `AFNAN_WAKEWORD_THRESHOLD` | Wake-word sensitivity | `0.5` |
| `AFNAN_BROWSER_RUNTIME_DIR` | Browser profile storage | `~/.afnan-ai/browser-runtime` |

## Project structure

```text
main.py                  # entry point
afnan_ai/
  agent.py               # central orchestrator
  agent_loop.py          # autonomous observe→decide→act→verify loop
  planner.py / executor.py / verifier.py / recovery.py
  state.py               # central task state
  tools/                 # generic tool interface + registry
  browser/               # browser automation runtime
  computer/              # desktop automation runtime
  skills/                # dynamic skill system
  subagents/             # multi-agent orchestration
  connectors/            # external service integrations
  artifacts/             # generated documents and files
  context/               # long-context and trajectory reasoning
  research/              # research & evidence intelligence
  evaluation/            # self-evaluation and benchmarking
  security/              # permissions, vault, audit
  activity/              # activity center and approvals
  memory_store.py / goal_manager.py / task_manager.py
  scheduler.py           # scheduled and background tasks
  speech.py / wakeword.py
  platform/              # Windows / macOS / Linux adapters
tests/                   # automated test suite
```

Module-level documentation lives next to the code it describes (e.g. `afnan_ai/research/RESEARCH.md`, `afnan_ai/evaluation/EVALUATION.md`).

## Tests

```bash
python -m unittest discover -s tests
```

## Notes

- Speech recognition uses Google's free speech API; the wake word itself is detected on-device.
- Sensitive actions (purchases, deletions, credential use) require explicit human approval — without an approver they never run.
- The assistant never solves CAPTCHAs; it hands them to you and resumes afterwards.

## License

MIT License — see `LICENSE` for details.
