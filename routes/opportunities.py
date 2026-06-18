from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status

from models.requests import Language
from models.responses import OpportunityListResponse, OpportunityDetail
from controllers.opportunities import list_opportunities, get_opportunity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Opportunities"])


@router.get(
    "/opportunities",
    response_model=OpportunityListResponse,
    summary="List all opportunities with pagination",
    description="Paginated listing of all opportunities. Supports optional filters by category, subtype, country, and fund_type.",
)
async def get_opportunities(
    lang: Language = Query(default=Language.EN, description="Response language"),
    page: int = Query(default=1, ge=1, description="Page number"),
    per_page: int = Query(default=12, ge=1, le=50, description="Items per page"),
    q: Optional[str] = Query(default=None, max_length=200, description="Title search (fuzzy/partial match)"),
    category: Optional[str] = Query(default=None, description="academic or non_academic"),
    subtype: Optional[str] = Query(default=None, description="masters, bachelor, phd, internship, etc."),
    country: Optional[str] = Query(default=None, description="Filter by country"),
    fund_type: Optional[str] = Query(default=None, description="fully_funded or partially_funded"),
    target_segment: Optional[str] = Query(default=None, description="high school, undergraduate, or graduate"),
    is_remote: Optional[bool] = Query(default=None, description="Filter by remote/in-person"),
) -> OpportunityListResponse:
    try:
        result = await list_opportunities(
            lang=lang.value,
            page=page,
            per_page=per_page,
            category=category,
            subtype=subtype,
            country=country,
            fund_type=fund_type,
            target_segment=target_segment,
            is_remote=is_remote,
            q=q,
        )
        return result
    except Exception as e:
        logger.exception("Failed to list opportunities")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch opportunities.",
        )


@router.get(
    "/opportunities/{opportunity_id}",
    response_model=OpportunityDetail,
    summary="Get a single opportunity by ID",
    description="Fetch full opportunity details by UUID. Useful for shareable links and bookmarks.",
)
async def get_opportunity_by_id(
    opportunity_id: str,
    lang: Language = Query(default=Language.EN, description="Response language"),
) -> OpportunityDetail:
    try:
        result = await get_opportunity(opportunity_id, lang=lang.value)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Opportunity {opportunity_id} not found.",
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get opportunity %s", opportunity_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch opportunity.",
        )
