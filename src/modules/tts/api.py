import asyncio

from fastapi import APIRouter, Query, Response

from src.modules.tts.service import MAX_TEXT_LENGTH, synthesize_wav

router = APIRouter(prefix="/api/tts", tags=["TTS"])


@router.get("")
async def api_tts(text: str = Query(..., min_length=1, max_length=MAX_TEXT_LENGTH)):
    # synthesize_wav is CPU-bound and blocking (onnxruntime); run off the
    # event loop so one synthesis doesn't stall every other request.
    wav_bytes = await asyncio.to_thread(synthesize_wav, text)
    return Response(content=wav_bytes, media_type="audio/wav")
