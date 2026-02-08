from __future__ import annotations

import logging
from math import ceil
from typing import Optional

from stores import pg_store
from models.responses import (
    OpportunityListResponse,
    OpportunityDetail,
    PaginationMeta,
)

logger = logging.getLogger(__name__)


async def list_opportunities(
    lang: str = "en",
    page: int = 1,
    per_page: int = 12,
    category: Optional[str] = None,
    subtype: Optional[str] = None,
    country: Optional[str] = None,
    fund_type: Optional[str] = None,
) -> OpportunityListResponse:
    """
    Paginated listing of all opportunities with optional filters.
    """
    items, total = await pg_store.list_opportunities(
        lang=lang,
        page=page,
        per_page=per_page,
        category=category,
        subtype=subtype,
        country=country,
        fund_type=fund_type,
    )

    total_pages = ceil(total / per_page) if total > 0 else 0

    return OpportunityListResponse(
        opportunities=[
            OpportunityDetail(id=item["id"], data=item["data"])
            for item in items
        ],
        pagination=PaginationMeta(
            page=page,
            per_page=per_page,
            total=total,
            total_pages=total_pages,
        ),
    )


async def get_opportunity(
    opportunity_id: str,
    lang: str = "en",
) -> Optional[OpportunityDetail]:
    """
    Fetch a single opportunity by ID.
    """
    result = await pg_store.get_opportunity_by_id(opportunity_id, lang=lang)
    if result is None:
        return None

    return OpportunityDetail(id=result["id"], data=result["data"])
