from qdrant_client import AsyncQdrantClient

from src.platform.db.qdrant import COLLECTION_NAME


async def identify_face(
    qdrant: AsyncQdrantClient,
    embedding: list[float],
    threshold: float = 0.6,
) -> dict | None:
    """Return payload {emp_id, name, emp_code} if cosine similarity >= threshold, else None.

    Qdrant cosine: score=1 means identical, score=0 means orthogonal.
    score_threshold = 1 - threshold means "at least `threshold` similar".
    """
    results = await qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        limit=1,
        score_threshold=1.0 - threshold,
    )
    if not results or results[0].score < (1.0 - threshold):
        return None
    return {**results[0].payload, "score": results[0].score}
