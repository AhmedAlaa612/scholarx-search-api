from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Language(str, Enum):
    EN = "en"
    AR = "ar"


class SearchRequest(BaseModel):
    """Body for POST /api/search"""

    query: str = Field(..., min_length=1, max_length=500, description="User search query (any language)")
    lang: Language = Field(default=Language.EN, description="Preferred response language")
    limit: int = Field(default=10, ge=1, le=50, description="Max results to return")


class OpportunitiesQueryParams(BaseModel):
    """Query parameters for GET /api/opportunities"""

    lang: Language = Field(default=Language.EN)
    page: int = Field(default=1, ge=1)
    per_page: int = Field(default=12, ge=1, le=50)

    # Optional filters
    category: Optional[str] = Field(default=None, description="academic or non_academic")
    subtype: Optional[str] = Field(default=None, description="masters, bachelor, phd, internship, etc.")
    country: Optional[str] = Field(default=None, description="Filter by country")
    fund_type: Optional[str] = Field(default=None, description="fully_funded or partially_funded")
