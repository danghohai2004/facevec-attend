import os

from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

COLLECTION_NAME = "face_embeddings"
_VECTOR_SIZE = 512

_client: AsyncQdrantClient | None = None


def get_qdrant_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            api_key=os.getenv("QDRANT_API_KEY"),
            # qdrant-client auto-enables HTTPS once api_key is set, but local
            # Qdrant serves plaintext HTTP → default to http. Set QDRANT_HTTPS=true
            # when Qdrant sits behind TLS in production.
            https=os.getenv("QDRANT_HTTPS", "false").lower() == "true",
        )
    return _client


async def ensure_collection() -> None:
    client = get_qdrant_client()
    collections = await client.get_collections()
    names = [c.name for c in collections.collections]
    if COLLECTION_NAME not in names:
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
        )
