from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.modules.tts import api as tts_api


def make_client():
    app = FastAPI()
    app.include_router(tts_api.router)
    return TestClient(app)


def test_tts_returns_wav_audio():
    client = make_client()
    with patch.object(tts_api, "synthesize_wav", return_value=b"RIFF....WAVEfmt ") as mock_synth:
        response = client.get("/api/tts", params={"text": "Xin chào Hải."})

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == b"RIFF....WAVEfmt "
    mock_synth.assert_called_once_with("Xin chào Hải.")


def test_tts_rejects_empty_text():
    client = make_client()
    response = client.get("/api/tts", params={"text": ""})
    assert response.status_code == 422


def test_tts_rejects_text_over_max_length():
    client = make_client()
    response = client.get("/api/tts", params={"text": "a" * 201})
    assert response.status_code == 422
