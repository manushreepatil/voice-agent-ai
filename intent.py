"""
Intent classification module.
Supports:
  - Ollama (local LLM via HTTP)
  - OpenAI API
  - Groq API
"""

import json
import os
import re
from typing import Optional


INTENT_SYSTEM_PROMPT = """You are an intent classification assistant for a voice-controlled AI agent.

Analyze the user's command and return a JSON object with the following structure:

{
  "intent": "<one of: create_file, write_code, summarize, general_chat, compound>",
  "description": "<short human-readable description of what to do>",
  "filename": "<target filename if applicable, else null>",
  "language": "<programming language if write_code intent, else null>",
  "content": "<text to summarize if summarize intent, else null>",
  "sub_tasks": [
    // only for compound intent: list of sub-task objects with same fields
  ]
}

Intent definitions:
- create_file: User wants to create a new file or folder (may or may not have content)
- write_code: User wants code generated and saved to a file
- summarize: User wants text summarized
- general_chat: Casual conversation, questions, or anything else
- compound: Multiple distinct actions in one command (e.g., "generate code AND save it AND summarize it")

Rules:
- Always return ONLY valid JSON, no extra text or markdown fences
- filename should include extension (.py, .txt, .md, etc.)
- If no filename mentioned, infer a sensible one from context
- For compound intents, break into ordered sub_tasks
- description should be concise (under 80 chars)
"""


def classify_intent(
    transcript: str,
    backend: str = "ollama (local)",
    model: str = "llama3",
    chat_context: Optional[list] = None,
) -> dict:
    """
    Classify the intent of a transcribed command.

    Args:
        transcript: The transcribed text from the user.
        backend: LLM backend to use.
        model: Model name for Ollama.
        chat_context: Previous conversation turns for context.

    Returns:
        dict with intent, description, filename, language, content, sub_tasks.
    """
    messages = _build_messages(transcript, chat_context or [])

    if backend == "ollama (local)":
        raw = _call_ollama(messages, model)
    elif backend == "openai-api":
        raw = _call_openai(messages)
    elif backend == "groq-api":
        raw = _call_groq(messages)
    else:
        raw = _call_ollama(messages, model)

    return _parse_response(raw)


def _build_messages(transcript: str, context: list) -> list:
    messages = [{"role": "system", "content": INTENT_SYSTEM_PROMPT}]
    # Include last 6 turns of context
    for turn in context[-6:]:
        messages.append(turn)
    messages.append({"role": "user", "content": transcript})
    return messages


def _call_ollama(messages: list, model: str = "llama3") -> str:
    import requests

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }
    try:
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Ollama not running. Start it with: ollama serve\n"
            "Then pull a model: ollama pull llama3"
        )
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")


def _call_openai(messages: list) -> str:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content
    except ImportError:
        raise ImportError("openai not installed. Run: pip install openai")


def _call_groq(messages: list) -> str:
    try:
        from groq import Groq

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content
    except ImportError:
        raise ImportError("groq not installed. Run: pip install groq")


def _parse_response(raw: str) -> dict:
    """Parse and validate the LLM JSON response."""
    try:
        # Strip markdown fences if present
        clean = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(clean)

        # Defaults
        data.setdefault("intent", "general_chat")
        data.setdefault("description", "")
        data.setdefault("filename", None)
        data.setdefault("language", None)
        data.setdefault("content", None)
        data.setdefault("sub_tasks", [])

        # Validate intent
        valid_intents = {"create_file", "write_code", "summarize", "general_chat", "compound"}
        if data["intent"] not in valid_intents:
            data["intent"] = "general_chat"

        return data

    except (json.JSONDecodeError, KeyError) as e:
        print(f"[Intent] Parse error: {e}. Raw: {raw[:200]}")
        return {
            "intent": "general_chat",
            "description": "Could not parse intent, falling back to chat",
            "filename": None,
            "language": None,
            "content": None,
            "sub_tasks": [],
        }
