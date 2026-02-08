from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    IsNullCondition,
    MatchValue,
    PayloadField,
    Range,
)

from config import settings

logger = logging.getLogger(__name__)


class QdrantStore:
    """Async Qdrant client wrapper."""

    def __init__(self) -> None:
        self._client: Optional[AsyncQdrantClient] = None

    async def connect(self) -> None:
        self._client = AsyncQdrantClient(
            url=settings.qdrant_endpoint,
            api_key=settings.qdrant_api_key,
        )
        logger.info("Connected to Qdrant at %s", settings.qdrant_endpoint)

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            logger.info("Qdrant connection closed")

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            raise RuntimeError("QdrantStore not connected. Call connect() first.")
        return self._client

    async def search(
        self,
        vector: List[float],
        qdrant_filter: Optional[Filter] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector search and return list of {id, score, payload}.
        """
        results = await self.client.query_points(
            collection_name=settings.qdrant_collection,
            query=vector,
            query_filter=qdrant_filter,
            limit=limit,
        )

        hits = []
        for point in results.points:
            hits.append(
                {
                    "id": point.payload.get("program_id", str(point.id)),
                    "score": point.score,
                    "payload": point.payload,
                }
            )
        return hits


# Singleton instance
qdrant_store = QdrantStore()
