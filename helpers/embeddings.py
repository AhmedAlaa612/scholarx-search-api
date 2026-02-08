from __future__ import annotations

import logging
from typing import List

import httpx
import numpy as np

from config import settings

logger = logging.getLogger(__name__)

# Reusable async HTTP client (created once, closed on shutdown)
_http_client: httpx.AsyncClient | None = None


async def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=30.0)
    return _http_client


async def close_http_client() -> None:
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


async def get_jina_embedding(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings from Jina API.
    Returns list of embedding vectors.
    """
    client = await get_http_client()

    response = await client.post(
        settings.jina_endpoint,
        json={
            "model": settings.jina_model,
            "input": texts,
        },
        headers={
            "Authorization": f"Bearer {settings.jina_api_key}",
            "Content-Type": "application/json",
        },
    )
    response.raise_for_status()

    data = response.json()["data"]
    embeddings = [d["embedding"] for d in data]

    logger.debug("Generated %d embeddings via Jina", len(embeddings))
    return embeddings
