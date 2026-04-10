# 🎙️ VoiceAgent AI — Voice-Controlled Local AI Agent

> **Mem0 AI/ML Intern Assignment** — Build a voice-controlled agent that transcribes speech, classifies intent, executes local tools, and shows results in a clean UI.

---

## 📸 Demo

> **Video Demo**: [https://www.loom.com/share/477a2e16f1144f91800194819ea06c35]  
> **Technical Article**: [https://dev.to/manushree_patil_22e650fcc/building-a-voice-controlled-local-ai-agent-with-whisper-groq-streamlit-3dfj]

---

## 🏗️ Architecture

```
Audio Input (.wav/.mp3 or mic)
        │
        ▼
┌───────────────────┐
│  STT Module       │  ← HuggingFace Whisper (local) or Groq API
│  (stt.py)         │
└────────┬──────────┘
         │ transcript
         ▼
┌───────────────────┐
│  Intent Module    │  ← Ollama (local LLM) or Groq/OpenAI API
│  (intent.py)      │  Returns: intent + structured JSON
└────────┬──────────┘
         │ intent_data
         ▼
┌───────────────────┐
│  Tool Executor    │  ← File ops | Code gen | Summarize | Chat
│  (tools.py)       │  All file writes → output/ folder only
└────────┬──────────┘
         │ result
         ▼
┌───────────────────┐
│  Streamlit UI     │  ← Shows transcript, intent, action, output
│  (app.py)         │
└───────────────────┘
```

### Supported Intents
| Intent | Description |
|--------|-------------|
| `create_file` | Create a new file in `output/` |
| `write_code` | Generate code via LLM → save to `output/` |
| `summarize` | Summarize provided text |
| `general_chat` | Conversational Q&A with memory |
| `compound` | Multiple commands in one utterance |

---

## ⚡ Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/voice-agent-ai.git
cd voice-agent-ai
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

**CPU-only machines** (no CUDA GPU):
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers requests streamlit groq openai python-dotenv
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 5. Set up LLM (choose one)

**Option A — Ollama (recommended, fully local)**
```bash
# Install Ollama: https://ollama.com/download
ollama serve
ollama pull llama3       # or: mistral, phi3, gemma2
```

**Option B — Groq API (free, fast cloud)**
```
Get key at: https://console.groq.com
Add to .env: GROQ_API_KEY=your_key
```

### 6. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🧠 Model Choices

### Speech-to-Text
| Model | Where | Notes |
|-------|-------|-------|
| `openai/whisper-base` | Local (HuggingFace) | ~150MB, fast on CPU |
| `openai/whisper-small` | Local (HuggingFace) | ~250MB, better accuracy |
| `openai/whisper-medium` | Local (HuggingFace) | ~800MB, best local accuracy |
| `groq-api` | Cloud (Groq) | Fastest, needs API key |

**Hardware Note**: On machines without a GPU, `whisper-base` runs in ~5-10 seconds on CPU. If your machine is too slow, set `groq-api` in the UI — it's free and processes audio in under 1 second.

### LLM (Intent + Generation)
| Backend | Model | Notes |
|---------|-------|-------|
| Ollama | `llama3` (default) | Fully local, private, free |
| Groq API | `llama3-70b-8192` | Fast cloud, free tier |
| OpenAI | `gpt-4o-mini` | Most capable, paid |

---

## 📁 Project Structure

```
voice-agent-ai/
├── app.py              # Streamlit UI (main entry point)
├── stt.py              # Speech-to-Text module
├── intent.py           # Intent classification via LLM
├── tools.py            # Tool execution (files, code, summarize, chat)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
├── .gitignore
├── output/             # All generated files go here (git-ignored)
└── README.md
```

---

## ✨ Features

### Core
- [x] Upload `.wav`, `.mp3`, `.m4a`, `.ogg` audio files
- [x] Local STT via HuggingFace Whisper
- [x] Cloud STT fallback via Groq
- [x] Local LLM intent classification via Ollama
- [x] Cloud LLM fallback (Groq / OpenAI)
- [x] `create_file`, `write_code`, `summarize`, `general_chat` intents
- [x] All file ops restricted to `output/` folder
- [x] Clean Streamlit UI showing full pipeline

### Bonus Implemented
- [x] **Compound commands** — e.g., *"Generate a bubble sort function and save it as bubble.py, then summarize what it does"*
- [x] **Human-in-the-loop** — Confirmation prompt before any file operation (toggleable)
- [x] **Graceful degradation** — Error handling for unintelligible audio, LLM failures, unmapped intents
- [x] **Session memory** — Chat context preserved across turns within a session
- [x] **Text override** — Skip STT and type directly for testing

---

## 🧪 Example Commands to Test

```
"Create a Python file with a retry decorator function"
"Write JavaScript code for a debounce function and save it as utils.js"
"Summarize this text: The quick brown fox jumps over the lazy dog..."
"What is the difference between a list and a tuple in Python?"
"Create a Python file with merge sort, then summarize what the algorithm does"
```

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| `Ollama not running` | Run `ollama serve` in a separate terminal |
| `GROQ_API_KEY not set` | Copy `.env.example` to `.env` and add your key |
| Whisper slow on CPU | Use `whisper-base` model or switch to `groq-api` |
| `transformers` not found | `pip install transformers torch torchaudio` |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |

---

## 📝 Submission Links

- **GitHub**: https://github.com/YOUR_USERNAME/voice-agent-ai
- **Video Demo**: [YouTube Unlisted]
- **Article**: [Medium / Dev.to / Substack]
- **Submission Form**: https://forms.gle/5x32P7zr4NvyRgK6A

---

## 📄 License

MIT License — free to use and modify.
