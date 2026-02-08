from __future__ import annotations

import json
import logging
from typing import List, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Pydantic models for parsed query
# ──────────────────────────────────────────────

class QueryConstraints(BaseModel):
    subtype: List[str] = Field(default_factory=list, description="masters, bachelor, phd, internship, camp, conference, exchange, volunteering, workshop")
    category: List[str] = Field(default_factory=list, description="academic or non_academic")
    country: List[str] = Field(default_factory=list)
    fund_type: List[str] = Field(default_factory=list, description="fully_funded or partially_funded")
    target_segment: List[str] = Field(default_factory=list, description="high school, undergraduate, graduate")
    documents_not_required: List[str] = Field(default_factory=list, description="Documents user does NOT want required")
    language_requirements: List[str] = Field(default_factory=list, description="IELTS, TOEFL, etc.")
    eligible_nationalities: List[str] = Field(default_factory=list, description="User nationality to filter eligible opportunities")
    is_remote: Optional[bool] = None
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    gpa: Optional[float] = None
    has_document_requirements: Optional[bool] = None
    has_language_requirements: Optional[bool] = None
    has_fee: Optional[bool] = None


class ParsedQuery(BaseModel):
    semantic_query: str
    constraints: QueryConstraints


# ──────────────────────────────────────────────
#  System prompt
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a query understanding engine for a scholarship search system. Your job is to:
1. Rewrite the user's query into clear, search-optimized English
2. Extract structured filters from the query
3. Return results as valid JSON

DATABASE CONTEXT:
- All opportunities in the database are currently active (ignore words like "open now", "available", "currently")
- Opportunities include both academic (degrees) and non_academic (internships, volunteering, workshops, etc.)

USER INPUT NOTES:
- Users may use Egyptian Arabic slang - interpret intent carefully
- Common abbreviations: "free" = fully_funded, "resume" = CV

EXTRACTION RULES:

**semantic_query (rewritten search text):**
- Rewrite into clear, concise English
- Remove redundant terms already captured in filters
- Do Not mention language requirements, document requirements, or fees if captured in filters
- Keep the core intent and keywords for semantic search

**category:**
- "academic" = degree-seeking programs (Bachelor, Masters, PhD)
- "non_academic" = internships, volunteering, workshops, camps, conferences, exchanges, research programs
- OMIT if ambiguous
- Summer programs are mostly non_academic

**subtype:**
- academic: "bachelor", "masters", "phd"
- non_academic: "internship", "conference", "camp", "volunteering", "workshop", "exchange"
- ONLY set if user explicitly wants that TYPE
- Do NOT infer from eligibility

**target_segment:**
- "high_school", "undergraduate", "graduate"
- Only if user specifies eligibility

**country:**
- Use short forms: "UAE", "USA", "UK"
- Use full names otherwise
- For regions, expand to relevant countries

**fund_type:**
- "fully_funded" or "partially_funded"

**eligible_nationalities:**
- If user mentions their nationality

**documents_not_required:**
- Documents user says "no X required"

**has_language_requirements:**
- false = no language certificate needed

**has_fee:**
- false = no fees

**has_document_requirements:**
- false = no documents required at all

**age & GPA:**
- Only include if user mentions specific numbers

**is_remote:**
- true = online/virtual only
- false = in-person only

IMPORTANT: Only include fields with actual values. No null, empty arrays, or empty strings.
Return ONLY valid JSON.

EXAMPLES:

Query: "fully funded masters in USA no IELTS and my GPA is 3.2"
{
  "semantic_query": "fully funded masters program USA",
  "filters": {
    "subtype": ["masters"],
    "category": "academic",
    "country": ["USA"],
    "fund_type": ["fully_funded"],
    "has_language_requirements": false,
    "gpa": 3.2
  }
}

Query: "منح للمصريين في اوروبا"
{
  "semantic_query": "scholarships in Europe",
  "filters": {
    "country": ["Germany", "France", "UK", "Spain", "Italy", "Netherlands", ... all European countries...],
    "eligible_nationalities": ["Egypt"]
  }
}
"""


def _build_prompt(user_query: str) -> str:
    return f"""Return JSON with this structure (only include fields with actual values):
{{
  "semantic_query": "rewritten search query",
  "filters": {{
    "subtype": ["only if specified"],
    "category": "only if clear",
    "country": ["only if mentioned"],
    "fund_type": ["only if mentioned"],
    "target_segment": ["only if specified"],
    "eligible_nationalities": ["only if user mentions nationality"],
    "documents_not_required": ["only if user says no X required"],
    "language_requirements": ["only if specific tests mentioned"],
    "min_age": "number - only if mentioned",
    "max_age": "number - only if mentioned",
    "gpa": "number - only if mentioned",
    "has_document_requirements": "boolean - only if mentioned",
    "has_language_requirements": "boolean - only if mentioned",
    "has_fee": "boolean - only if mentioned",
    "is_remote": "boolean - only if mentioned"
  }}
}}

User query: "{user_query}"
"""


# ──────────────────────────────────────────────
#  LLM clients with round-robin failover
# ──────────────────────────────────────────────

_groq_client: AsyncOpenAI | None = None
_cerebras_client: AsyncOpenAI | None = None
_call_count = 0


def _get_llm_clients():
    global _groq_client, _cerebras_client
    if _groq_client is None:
        _groq_client = AsyncOpenAI(
            api_key=settings.grok_api,
            base_url="https://api.groq.com/openai/v1",
        )
    if _cerebras_client is None:
        _cerebras_client = AsyncOpenAI(
            api_key=settings.cerebras_api,
            base_url="https://api.cerebras.ai/v1/",
        )
    return [
        (_groq_client, "openai/gpt-oss-120b"),
        (_cerebras_client, "gpt-oss-120b"),
    ]


async def _call_llm(user_query: str) -> ParsedQuery | None:
    global _call_count
    clients = _get_llm_clients()
    primary_idx = _call_count % 2
    _call_count += 1

    for idx in [primary_idx, 1 - primary_idx]:
        client, model = clients[idx]
        try:
            response = await client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_prompt(user_query)},
                ],
            )
            raw = response.choices[0].message.content
            data = json.loads(raw)

            filters = data.get("filters", {})
            category = filters.get("category")
            if category and isinstance(category, str):
                category = [category]
            elif not category:
                category = []

            constraints = QueryConstraints(
                subtype=filters.get("subtype") or [],
                category=category,
                country=filters.get("country") or [],
                fund_type=filters.get("fund_type") or [],
                target_segment=filters.get("target_segment") or [],
                documents_not_required=filters.get("documents_not_required") or [],
                language_requirements=filters.get("language_requirements") or [],
                eligible_nationalities=filters.get("eligible_nationalities") or [],
                is_remote=filters.get("is_remote"),
                min_age=filters.get("min_age"),
                max_age=filters.get("max_age"),
                gpa=filters.get("gpa"),
                has_document_requirements=filters.get("has_document_requirements"),
                has_language_requirements=filters.get("has_language_requirements"),
                has_fee=filters.get("has_fee"),
            )

            return ParsedQuery(
                semantic_query=data.get("semantic_query", user_query),
                constraints=constraints,
            )

        except Exception as e:
            logger.warning("LLM client %d failed: %s", idx + 1, e)
            continue

    return None


async def understand_query(user_query: str) -> ParsedQuery:
    """
    Parse user query via LLM into semantic query + structured constraints.
    Falls back to raw query if LLM fails.
    """
    parsed = await _call_llm(user_query)
    if parsed:
        return parsed

    logger.warning("All LLM clients failed, falling back to raw query")
    return ParsedQuery(
        semantic_query=user_query,
        constraints=QueryConstraints(),
    )
