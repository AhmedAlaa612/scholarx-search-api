from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from models.requests import SearchRequest
from models.responses import SearchResponse
from controllers.search import search_opportunities

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Search"])


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic search for opportunities",
    description=(
        "Accepts a natural-language query (any language including Arabic), "
        "parses it with an LLM, performs vector search on Qdrant, "
        "and returns full opportunity data from PostgreSQL."
    ),
)
async def search(request: SearchRequest) -> SearchResponse:
    try:
        result = await search_opportunities(
            query=request.query,
            lang=request.lang.value,
            limit=request.limit,
        )
        return result
    except Exception as e:
        logger.exception("Search failed for query: %s", request.query)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed. Please try again.",
        )
