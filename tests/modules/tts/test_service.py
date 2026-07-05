import io
import wave

from src.modules.tts.service import synthesize_wav


def test_synthesize_wav_produces_valid_playable_audio():
    wav_bytes = synthesize_wav("Xin chào Hải.")

    assert len(wav_bytes) > 0
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.getnframes() > 0
        assert wav_file.getsampwidth() == 2  # 16-bit PCM
        assert wav_file.getnchannels() == 1
