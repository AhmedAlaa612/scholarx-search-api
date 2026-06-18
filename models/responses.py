from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ParsedQueryInfo(BaseModel):
    """Debug info about how the query was parsed (only in non-production)."""

    semantic_query: str
    filters_applied: Dict[str, Any] = {}


class SearchHit(BaseModel):
    """A single search result."""

    id: str
    score: float
    opportunity: Dict[str, Any]


class SearchResponse(BaseModel):
    """Response for POST /api/search"""

    results: List[SearchHit]
    total: int
    parsed_query: Optional[ParsedQueryInfo] = None


class OpportunityDetail(BaseModel):
    """A single opportunity in list view."""

    id: str
    data: Dict[str, Any]


class PaginationMeta(BaseModel):
    page: int
    per_page: int
    total: int
    total_pages: int


class OpportunityListResponse(BaseModel):
    """Response for GET /api/opportunities"""

    opportunities: List[OpportunityDetail]
    pagination: PaginationMeta
