from __future__ import annotations

import logging
from typing import Any, Dict, List

from helpers.embeddings import get_jina_embedding
from helpers.query_parser import understand_query
from helpers.filters import build_qdrant_filter
from stores import qdrant_store, pg_store
from models.responses import SearchResponse, SearchHit, ParsedQueryInfo
from config import settings

logger = logging.getLogger(__name__)


async def search_opportunities(
    query: str,
    lang: str = "en",
    limit: int = 10,
) -> SearchResponse:
    """
    Full search pipeline:
    1. LLM parses user query → semantic_query + constraints
    2. Embed semantic_query via Jina
    3. Search Qdrant with vector + filters
    4. Join results with PostgreSQL for full data
    5. Return combined results
    """
    # Step 1: Parse query
    parsed = await understand_query(query)
    logger.info(
        "Original query: '%s' | Parsed query: semantic='%s', filters=%s",
        query,
        parsed.semantic_query,
        parsed.constraints.model_dump(exclude_none=True, exclude_defaults=True),
    )

    # Step 2: Embed
    embeddings = await get_jina_embedding([parsed.semantic_query])
    vector = embeddings[0]

    # Step 3: Qdrant search
    qdrant_filter = build_qdrant_filter(parsed.constraints)
    hits = await qdrant_store.search(
        vector=vector,
        qdrant_filter=qdrant_filter,
        limit=limit,
    )

    if not hits:
        return SearchResponse(
            results=[],
            total=0,
            parsed_query=_build_debug_info(parsed) if settings.app_env != "production" else None,
        )

    # Step 4: Join with PostgreSQL
    program_ids = [hit["id"] for hit in hits]
    pg_data = await pg_store.get_opportunities_by_ids(program_ids, lang=lang)

    # Step 5: Combine
    results = []
    for hit in hits:
        pid = hit["id"]
        opportunity_data = pg_data.get(pid)
        if opportunity_data is None:
            logger.warning("Opportunity %s found in Qdrant but missing in PG", pid)
            continue

        results.append(
            SearchHit(
                id=pid,
                score=round(hit["score"], 4),
                opportunity=opportunity_data,
            )
        )

    return SearchResponse(
        results=results,
        total=len(results),
        parsed_query=_build_debug_info(parsed) if settings.app_env != "production" else None,
    )


def _build_debug_info(parsed) -> ParsedQueryInfo:
    """Build debug info for non-production responses."""
    return ParsedQueryInfo(
        semantic_query=parsed.semantic_query,
        filters_applied=parsed.constraints.model_dump(exclude_none=True, exclude_defaults=True),
    )
