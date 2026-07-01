import hmac
import os

from fastapi import Header, HTTPException


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise HTTPException(503, "Server auth not configured")
    if not x_api_key or not hmac.compare_digest(
        x_api_key.encode(),
        api_key.encode(),
    ):
        raise HTTPException(401, "Invalid or missing API key")
