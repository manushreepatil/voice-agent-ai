"""
Speech-to-Text module.
Supports:
  - HuggingFace Whisper (local)
  - Groq Whisper API (cloud fallback)
"""

import os
from pathlib import Path


def transcribe_audio(audio_path: str, model_id: str = "openai/whisper-base") -> str:
    """
    Transcribe an audio file to text.

    Args:
        audio_path: Path to the audio file (.wav, .mp3, etc.)
        model_id: HuggingFace model ID or 'groq-api'

    Returns:
        Transcribed text string.
    """

    if model_id == "groq-api":
        return _transcribe_groq(audio_path)
    else:
        return _transcribe_hf(audio_path, model_id)


def _transcribe_hf(audio_path: str, model_id: str) -> str:
    """Local transcription using HuggingFace transformers pipeline."""
    try:
        from transformers import pipeline
        import torch

        device = "cuda" if _cuda_available() else "cpu"
        print(f"[STT] Loading {model_id} on {device}...")

        asr = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=0 if device == "cuda" else -1,
            chunk_length_s=30,          # handle long audio
            stride_length_s=5,
            return_timestamps=False,
        )
        result = asr(audio_path)
        transcript = result["text"].strip()
        print(f"[STT] Transcript: {transcript}")
        return transcript

    except ImportError:
        raise ImportError(
            "transformers not installed. Run: pip install transformers torch torchaudio"
        )
    except Exception as e:
        raise RuntimeError(f"HuggingFace STT failed: {e}")


def _transcribe_groq(audio_path: str) -> str:
    """Cloud transcription via Groq Whisper API."""
    try:
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Add it to your .env file or environment."
            )

        client = Groq(api_key=api_key)
        with open(audio_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(Path(audio_path).name, f.read()),
                model="whisper-large-v3",
                response_format="text",
                language="en",
            )
        transcript = transcription.strip()
        print(f"[STT] Groq transcript: {transcript}")
        return transcript

    except ImportError:
        raise ImportError("groq not installed. Run: pip install groq")
    except Exception as e:
        raise RuntimeError(f"Groq STT failed: {e}")


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
