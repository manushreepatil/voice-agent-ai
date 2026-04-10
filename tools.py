"""
Tool execution module.
Handles: create_file, write_code, summarize, general_chat, compound
All file writes are restricted to the output/ directory.
"""

import os
import json
import re
from pathlib import Path
from typing import Optional
from datetime import datetime


# ─── LLM helper ──────────────────────────────────────────────────────────────

def _llm_generate(prompt: str, system: str = "", backend: str = "ollama", model: str = "llama3") -> str:
    """Simple LLM text generation for code/summary/chat tasks."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # Try Ollama first
    try:
        import requests
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.3},
        }
        resp = requests.post("http://localhost:11434/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception:
        pass

    # Fallback: Groq
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        pass

    # Fallback: OpenAI
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"All LLM backends failed. Last error: {e}")


# ─── Safe path helper ─────────────────────────────────────────────────────────

def _safe_path(output_dir: Path, filename: str) -> Path:
    """Ensure the file path stays inside output_dir."""
    # Strip any path traversal attempts
    safe_name = Path(filename).name  # Only keep the filename, not any directory parts
    # Add timestamp to avoid overwrites
    stem = Path(safe_name).stem
    suffix = Path(safe_name).suffix or ".txt"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_name = f"{stem}_{ts}{suffix}"
    return output_dir / final_name


# ─── Tool handlers ────────────────────────────────────────────────────────────

def _tool_create_file(data: dict, output_dir: Path) -> dict:
    """Create an empty file or file with basic content."""
    filename = data.get("filename") or "new_file.txt"
    description = data.get("description", "")
    filepath = _safe_path(output_dir, filename)

    # Generate minimal content if it's not just an empty file
    content = f"# {filename}\n# Created by VoiceAgent AI\n# {description}\n"
    filepath.write_text(content)

    return {
        "success": True,
        "message": f"File created: {filepath.name}",
        "filepath": str(filepath),
        "output": content,
    }


def _tool_write_code(data: dict, output_dir: Path) -> dict:
    """Generate code using LLM and save to file."""
    description = data.get("description", "")
    filename = data.get("filename") or "generated_code.py"
    language = data.get("language") or _infer_language(filename)

    system = f"""You are an expert {language} programmer. 
Write clean, well-commented, production-quality code.
Return ONLY the code, no explanations, no markdown fences."""

    prompt = f"Write {language} code for: {description}"
    code = _llm_generate(prompt, system=system)

    # Strip markdown code fences if LLM added them
    code = re.sub(r"^```[\w]*\n?", "", code, flags=re.MULTILINE)
    code = re.sub(r"\n?```$", "", code, flags=re.MULTILINE).strip()

    filepath = _safe_path(output_dir, filename)
    filepath.write_text(code)

    return {
        "success": True,
        "message": f"Code generated and saved: {filepath.name}",
        "filepath": str(filepath),
        "output": code,
    }


def _tool_summarize(data: dict, output_dir: Path) -> dict:
    """Summarize provided text using LLM."""
    content = data.get("content") or data.get("description", "")
    if not content:
        return {
            "success": False,
            "message": "No text provided to summarize.",
            "output": "",
        }

    system = """You are a professional summarizer. 
Create a concise, accurate summary that captures the key points.
Format: Start with a one-sentence TL;DR, then 3-5 bullet points."""

    prompt = f"Summarize this text:\n\n{content}"
    summary = _llm_generate(prompt, system=system)

    # Optionally save
    filename = data.get("filename")
    filepath = None
    if filename:
        fp = _safe_path(output_dir, filename)
        fp.write_text(f"SUMMARY\n{'='*40}\n\nOriginal:\n{content}\n\n{'='*40}\n\nSummary:\n{summary}")
        filepath = str(fp)

    return {
        "success": True,
        "message": "Text summarized" + (f" and saved to {Path(filepath).name}" if filepath else ""),
        "filepath": filepath,
        "output": summary,
    }


def _tool_general_chat(data: dict, context: list = None) -> dict:
    """Handle general conversation."""
    description = data.get("description", "")
    
    system = """You are a helpful, concise AI assistant integrated into a voice agent system.
Answer clearly and helpfully. Keep responses focused and not too long."""

    response = _llm_generate(description or "Hello!", system=system)

    return {
        "success": True,
        "message": "Response generated",
        "filepath": None,
        "output": response,
    }


def _tool_compound(data: dict, output_dir: Path) -> dict:
    """Handle compound commands by executing sub-tasks in order."""
    sub_tasks = data.get("sub_tasks", [])
    if not sub_tasks:
        # Try to handle as the primary intent
        return _tool_general_chat(data)

    results = []
    all_success = True

    for i, task in enumerate(sub_tasks):
        intent = task.get("intent", "general_chat")
        result = _dispatch(intent, task, output_dir)
        results.append({
            "step": i + 1,
            "intent": intent,
            "description": task.get("description", ""),
            "result": result,
        })
        if not result.get("success", True):
            all_success = False

    # Build combined output
    combined_output = "\n\n".join([
        f"[Step {r['step']}] {r['description']}\n{r['result'].get('output','')}"
        for r in results
    ])

    return {
        "success": all_success,
        "message": f"Compound command: {len(results)} steps executed",
        "filepath": None,
        "output": combined_output,
        "steps": results,
    }


# ─── Dispatcher ───────────────────────────────────────────────────────────────

def _dispatch(intent: str, data: dict, output_dir: Path) -> dict:
    if intent == "create_file":
        return _tool_create_file(data, output_dir)
    elif intent == "write_code":
        return _tool_write_code(data, output_dir)
    elif intent == "summarize":
        return _tool_summarize(data, output_dir)
    elif intent == "compound":
        return _tool_compound(data, output_dir)
    else:
        return _tool_general_chat(data)


def execute_tool(intent: str, intent_data: dict, output_dir: Path) -> dict:
    """
    Main entry point for tool execution.

    Args:
        intent: Classified intent string
        intent_data: Full intent data dict from LLM
        output_dir: Safe output directory path

    Returns:
        dict with success, message, filepath, output fields
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return _dispatch(intent, intent_data, output_dir)
    except Exception as e:
        return {
            "success": False,
            "message": f"Tool execution failed: {str(e)}",
            "filepath": None,
            "output": "",
        }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _infer_language(filename: str) -> str:
    ext_map = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
        ".java": "Java", ".cpp": "C++", ".c": "C", ".go": "Go",
        ".rs": "Rust", ".rb": "Ruby", ".sh": "Bash", ".sql": "SQL",
        ".html": "HTML", ".css": "CSS", ".json": "JSON",
    }
    ext = Path(filename).suffix.lower()
    return ext_map.get(ext, "Python")
