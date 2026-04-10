import streamlit as st
import os
import json
import tempfile
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from datetime import datetime

# Import modules
from stt import transcribe_audio
from intent import classify_intent
from tools import execute_tool

# Page config
st.set_page_config(
    page_title="VoiceAgent AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

:root {
    --bg: #0a0a0f;
    --card: #13131a;
    --border: #1e1e2e;
    --accent: #7c3aed;
    --accent2: #06b6d4;
    --green: #10b981;
    --red: #ef4444;
    --text: #e2e8f0;
    --muted: #64748b;
}

html, body, .stApp {
    background-color: var(--bg) !important;
    font-family: 'Sora', sans-serif;
    color: var(--text);
}

.stApp > header { background: transparent !important; }

.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #7c3aed, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
}

.subtitle {
    color: var(--muted);
    font-size: 0.9rem;
    font-family: 'Space Mono', monospace;
    margin-top: 4px;
}

.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}

.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.5rem;
}

.card-value {
    font-size: 1rem;
    color: var(--text);
    font-weight: 400;
}

.intent-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.badge-create_file { background: #1a3a2a; color: #10b981; border: 1px solid #10b981; }
.badge-write_code { background: #1a1a3a; color: #7c3aed; border: 1px solid #7c3aed; }
.badge-summarize { background: #1a2a3a; color: #06b6d4; border: 1px solid #06b6d4; }
.badge-general_chat { background: #2a1a1a; color: #f59e0b; border: 1px solid #f59e0b; }
.badge-compound { background: #2a1a3a; color: #ec4899; border: 1px solid #ec4899; }

.output-code {
    background: #0d0d14;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #a5f3fc;
    white-space: pre-wrap;
    word-break: break-word;
}

.success-msg {
    background: #0d2a1e;
    border: 1px solid #10b981;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #10b981;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
}

.error-msg {
    background: #2a0d0d;
    border: 1px solid #ef4444;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #ef4444;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
}

.history-item {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
}

.confirm-box {
    background: #1a1500;
    border: 1px solid #f59e0b;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}

/* Streamlit overrides */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #6d28d9, #4c1d95) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
}

.stFileUploader {
    background: var(--card) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
}

[data-testid="stSidebar"] {
    background: var(--card) !important;
    border-right: 1px solid var(--border) !important;
}

.stSelectbox > div > div {
    background: var(--card) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

div[data-testid="stMarkdownContainer"] p {
    color: var(--text);
}

.stAudio { filter: invert(0.9) hue-rotate(180deg); }

.pipeline-step {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 0.6rem 0;
    border-bottom: 1px solid var(--border);
}

.step-num {
    background: var(--accent);
    color: white;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 2px;
}

.step-text { color: var(--text); font-size: 0.9rem; }
.step-label { color: var(--muted); font-size: 0.75rem; font-family: 'Space Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# Ensure output dir
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "pending_confirmation" not in st.session_state:
    st.session_state.pending_confirmation = None
if "chat_context" not in st.session_state:
    st.session_state.chat_context = []

# Header
st.markdown('<h1 class="main-title">🎙️ VoiceAgent AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">// speech → intent → action → result</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown('<div class="card-title">⚙️ Configuration</div>', unsafe_allow_html=True)

    stt_model = st.selectbox(
        "STT Model",
        ["openai/whisper-base", "openai/whisper-small", "openai/whisper-medium", "groq-api"],
        help="Choose speech-to-text model"
    )

    llm_backend = st.selectbox(
        "LLM Backend",
        ["ollama (local)", "openai-api", "groq-api"],
        help="LLM for intent & generation"
    )

    ollama_model = st.text_input("Ollama Model", value="llama3", placeholder="llama3, mistral, phi3...")
    confirm_file_ops = st.checkbox("Human-in-the-loop (confirm file ops)", value=True)

    st.markdown("---")
    st.markdown('<div class="card-title">📁 Output Folder</div>', unsafe_allow_html=True)
    output_files = list(OUTPUT_DIR.iterdir()) if OUTPUT_DIR.exists() else []
    if output_files:
        for f in sorted(output_files)[-5:]:
            st.markdown(f'<div style="font-family:Space Mono;font-size:0.75rem;color:#64748b;">📄 {f.name}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#64748b;font-size:0.8rem;">No files yet</div>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.session_state.chat_context = []
        st.rerun()

# Main columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="card-title">🎤 Audio Input</div>', unsafe_allow_html=True)

    input_method = st.radio("Input Method", ["Upload Audio File", "Record via Microphone"], horizontal=True)

    audio_data = None
    audio_path = None

    if input_method == "Upload Audio File":
        uploaded = st.file_uploader("Drop .wav or .mp3 file", type=["wav", "mp3", "m4a", "ogg"])
        if uploaded:
            # Save to temp
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                audio_path = tmp.name
            st.audio(uploaded)
    else:
        st.markdown('<div class="card" style="text-align:center;padding:2rem;">', unsafe_allow_html=True)
        st.markdown("🎙️ **Microphone recording** requires running locally with `streamlit run app.py`")
        st.markdown("Use the `record_audio()` helper or st-audiorec component")
        st.markdown('</div>', unsafe_allow_html=True)

        # Try st-audiorec if available
        try:
            from st_audiorec import st_audiorec
            wav_audio_data = st_audiorec()
            if wav_audio_data:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(wav_audio_data)
                    audio_path = tmp.name
        except ImportError:
            st.info("Install `st-audiorec` for microphone support: `pip install st-audiorec`")

    # Manual text override
    st.markdown("---")
    st.markdown('<div class="card-title">✏️ Or Type Directly (for testing)</div>', unsafe_allow_html=True)
    manual_text = st.text_area("Skip STT — type your command", placeholder='e.g. "Create a Python file with a bubble sort function"', height=80)

    run_btn = st.button("⚡ Run Agent", use_container_width=True)

with col2:
    st.markdown('<div class="card-title">📊 Pipeline Output</div>', unsafe_allow_html=True)

    result_placeholder = st.empty()

    # Show pending confirmation
    if st.session_state.pending_confirmation:
        pending = st.session_state.pending_confirmation
        st.markdown(f"""
        <div class="confirm-box">
        ⚠️ <b>Confirm File Operation</b><br>
        Intent: <b>{pending['intent']}</b><br>
        Action: {pending['description']}
        </div>
        """, unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("✅ Confirm & Execute"):
            result = execute_tool(pending['intent'], pending['data'], OUTPUT_DIR)
            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "transcript": pending['transcript'],
                "intent": pending['intent'],
                "result": result
            })
            st.session_state.pending_confirmation = None
            st.rerun()
        if c2.button("❌ Cancel"):
            st.session_state.pending_confirmation = None
            st.rerun()

# Process on run
if run_btn:
    transcript = ""
    intent_result = {}
    tool_result = {}

    with st.spinner(""):
        # Step 1: STT
        if manual_text.strip():
            transcript = manual_text.strip()
        elif audio_path:
            with result_placeholder.container():
                st.markdown('<div class="pipeline-step"><div class="step-num">1</div><div><div class="step-label">STEP 1 — STT</div><div class="step-text">🔊 Transcribing audio...</div></div></div>', unsafe_allow_html=True)
            try:
                transcript = transcribe_audio(audio_path, stt_model)
            except Exception as e:
                with result_placeholder.container():
                    st.markdown(f'<div class="error-msg">STT Error: {e}</div>', unsafe_allow_html=True)
                st.stop()
        else:
            with result_placeholder.container():
                st.markdown('<div class="error-msg">⚠️ Please upload an audio file or type a command.</div>', unsafe_allow_html=True)
            st.stop()

        # Step 2: Intent
        with result_placeholder.container():
            st.markdown(f'<div class="pipeline-step"><div class="step-num">2</div><div><div class="step-label">STEP 2 — INTENT</div><div class="step-text">🧠 Classifying intent...</div></div></div>', unsafe_allow_html=True)

        try:
            intent_result = classify_intent(transcript, llm_backend, ollama_model, st.session_state.chat_context)
        except Exception as e:
            with result_placeholder.container():
                st.markdown(f'<div class="error-msg">Intent Error: {e}</div>', unsafe_allow_html=True)
            st.stop()

        # Update chat context
        st.session_state.chat_context.append({"role": "user", "content": transcript})

        # Step 3: Tool execution
        intent = intent_result.get("intent", "general_chat")
        needs_confirmation = confirm_file_ops and intent in ["create_file", "write_code", "compound"]

        if needs_confirmation:
            st.session_state.pending_confirmation = {
                "intent": intent,
                "data": intent_result,
                "transcript": transcript,
                "description": intent_result.get("description", "File operation")
            }
            st.rerun()
        else:
            with result_placeholder.container():
                st.markdown(f'<div class="pipeline-step"><div class="step-num">3</div><div><div class="step-label">STEP 3 — EXECUTION</div><div class="step-text">⚙️ Executing tool...</div></div></div>', unsafe_allow_html=True)
            tool_result = execute_tool(intent, intent_result, OUTPUT_DIR)

        # Save history
        st.session_state.history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "transcript": transcript,
            "intent": intent,
            "intent_data": intent_result,
            "result": tool_result
        })

        if intent == "general_chat":
            st.session_state.chat_context.append({
                "role": "assistant",
                "content": tool_result.get("output", "")
            })

    # Display results
    with col2:
        with result_placeholder.container():
            # Transcript
            st.markdown(f"""
            <div class="card">
                <div class="card-title">📝 Transcription</div>
                <div class="card-value">{transcript}</div>
            </div>
            """, unsafe_allow_html=True)

            # Intent
            badge_class = f"badge-{intent}"
            st.markdown(f"""
            <div class="card">
                <div class="card-title">🎯 Detected Intent</div>
                <div class="card-value">
                    <span class="intent-badge {badge_class}">{intent.replace('_',' ')}</span>
                    <br><br>
                    <span style="color:#64748b;font-size:0.85rem;">{intent_result.get('description','')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Result
            if tool_result:
                output_text = tool_result.get("output", "")
                success = tool_result.get("success", True)
                filepath = tool_result.get("filepath", "")

                status_class = "success-msg" if success else "error-msg"
                status_icon = "✅" if success else "❌"
                status_text = tool_result.get("message", "Done")

                st.markdown(f'<div class="{status_class}">{status_icon} {status_text}</div>', unsafe_allow_html=True)

                if filepath:
                    st.markdown(f'<div style="color:#64748b;font-size:0.8rem;margin:0.5rem 0;">📁 Saved to: <code style="color:#7c3aed;">{filepath}</code></div>', unsafe_allow_html=True)

                if output_text:
                    st.markdown("**Output:**")
                    st.code(output_text, language="python" if intent == "write_code" else None)

# History
st.markdown("---")
st.markdown('<div class="card-title">📜 Session History</div>', unsafe_allow_html=True)

if st.session_state.history:
    for item in reversed(st.session_state.history[-10:]):
        intent_color = {"create_file":"#10b981","write_code":"#7c3aed","summarize":"#06b6d4","general_chat":"#f59e0b","compound":"#ec4899"}.get(item.get("intent",""), "#64748b")
        st.markdown(f"""
        <div class="history-item">
            <span style="color:#64748b;font-family:Space Mono;font-size:0.7rem;">{item['time']}</span>
            <span class="intent-badge badge-{item.get('intent','')}" style="margin:0 8px;font-size:0.7rem;">{item.get('intent','?').replace('_',' ')}</span>
            <span style="font-size:0.85rem;">{item.get('transcript','')[:80]}{'...' if len(item.get('transcript',''))>80 else ''}</span>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown('<div style="color:#64748b;font-size:0.9rem;">No actions yet. Run the agent to see history.</div>', unsafe_allow_html=True)
