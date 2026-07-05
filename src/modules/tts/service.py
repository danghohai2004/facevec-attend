"""Offline text-to-speech via Piper.

Runs fully locally (no network call, no API key, no per-request cost) — the
same "self-host the model, work offline" pattern already used for the
kiosk's in-browser MediaPipe assets. Chosen over browser Web Speech API
because installed voices vary wildly by OS/browser, so the kiosk's audio
greeting was inconsistent (silent on machines with no Vietnamese voice,
robotic on others). Synthesizing server-side gives every kiosk the same
voice regardless of what's installed on it.
"""

import io
import logging
import os
import threading
import wave

from piper import PiperVoice

logger = logging.getLogger(__name__)

_DEFAULT_VOICE_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "models", "piper", "vi_VN-vais1000-medium.onnx",
    )
)
VOICE_PATH = os.getenv("PIPER_VOICE_PATH", _DEFAULT_VOICE_PATH)

# /api/tts is unauthenticated (kiosk calls it without an API key, same as the
# other public GET endpoints) — cap input length so it can't be used to force
# arbitrarily long synthesis runs.
MAX_TEXT_LENGTH = 200

_voice: PiperVoice | None = None
_lock = threading.Lock()


def _get_voice() -> PiperVoice:
    global _voice
    if _voice is None:
        with _lock:
            if _voice is None:  # double-checked: avoid the lock on the hot path
                logger.info("Loading Piper voice from %s", VOICE_PATH)
                _voice = PiperVoice.load(VOICE_PATH)
    return _voice


def synthesize_wav(text: str) -> bytes:
    """Text -> WAV bytes. Blocking/CPU-bound; call via a thread from async code."""
    voice = _get_voice()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    return buf.getvalue()
