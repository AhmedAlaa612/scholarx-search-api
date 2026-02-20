from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from config import settings
from .countries import normalize_countries

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
    language_score_requirements: Dict[str, float] = Field(default_factory=dict, description="User's exam scores: e.g. {'ielts': 6.5, 'toefl': 90}")
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
- Remove redundant terms already captured in filters (e.g., "no IELTS" -> don't include in semantic query if has_language_requirements=false)
- Do Not mention language requirements, document requirements, or fees in the semantic query if those are captured in filters
- Keep the core intent and keywords for semantic search

**category:**
- "academic" = degree-seeking programs (Bachelor, Masters, PhD)
- "non_academic" = internships, volunteering, workshops, camps, conferences, exchanges, research programs, short courses
- OMIT if: user mentions only a program name, query is ambiguous, or could be either type (e.g., "travel opportunities")
- if user mentions summer programs they mostly mean non_academic

**subtype:**
- academic: "bachelor", "masters", "phd"
- non_academic: "internship", "conference", "camp", "volunteering", "workshop", "exchange"
- ONLY set if user explicitly wants that TYPE of program
- Do NOT infer from eligibility (e.g., "for undergraduates" ≠ subtype "bachelor")
- OMIT if user just mentions a program name

**target_segment:**
- Add to list ONLY if user specifies eligibility: "high school", "undergraduate", "graduate"
- "Undergraduate" or "student" refers to WHO can apply, not the degree type
- OMIT if not specified

**country:**
- Extract mentioned countries
- Use short forms: "UAE", "USA", "UK"
- Use full names otherwise: "Egypt", "Germany", "Canada"
- For regions (e.g., "Europe", "Asia"), expand to relevant countries
- OMIT if not mentioned

**fund_type:**
- "fully_funded" = 100% covered, no cost to student
- "partially_funded" = some financial support
- OMIT if not specified

**eligible_nationalities:**
- If user mentions their nationality or who the opportunity is for (e.g., "for Egyptians", "as a Jordanian", "for Arab students")
- Extract country names: "Egypt", "Jordan", etc.
- This filters opportunities that accept these nationalities (or accept "all")
- OMIT if not mentioned

**documents_not_required:**
- If user says "no CV", "without transcript", "IELTS not needed" -> ADD that document to this list
- This filters for opportunities that do NOT require these documents
- Common values: "cv", "transcript", "motivation_letter", "recommendation_letter", "language_certificate"
- OMIT if not mentioned

**language_score_requirements:**
- If user mentions a SPECIFIC exam WITH a score or threshold
- Extract as {"exam_name_lowercase": score_as_number}
- Normalize exam names to lowercase: "ielts", "toefl", "duolingo", "pte", etc.
- Examples:
  - "IELTS less than 6.5" or "IELTS 6.5" or "my IELTS is 6.5" → {"ielts": 6.5}
  - "TOEFL 90" → {"toefl": 90}
  - "IELTS 6.5 and TOEFL 90" → {"ielts": 6.5, "toefl": 90}
- This filters: opportunities requiring that exam with score ≤ user's score, or not requiring that exam at all
- OMIT if no specific exam+score is mentioned
- If user says "no IELTS" or "no language requirements" WITHOUT a score, use has_language_requirements: false instead

**has_language_requirements:**
- false = no language certificate needed at all (overrides language_score_requirements)
- true = language certificate required
- OMIT if not specified

**has_fee:**
- false = no application/program fees
- true = has fees
- OMIT if not specified

**has_document_requirements:**
- false = no documents required at all
- true = documents are required
- OMIT if not specified

**age & GPA:**
- Only include if user mentions specific numeric thresholds
- min_age/max_age: age range
- gpa: maximum GPA user has (to find programs they qualify for)

**is_remote:**
- true = online/virtual opportunities only
- false = in-person only
- OMIT if not specified

IMPORTANT OUTPUT RULES:
- Only include fields that have actual values
- Do NOT include fields with null, empty arrays [], empty objects {}, or empty strings ""
- If a filter is not mentioned or unclear, simply don't include it in the output
- Return ONLY valid JSON

EXAMPLES:

Query: "research program for undergrads open now"
{
  "semantic_query": "undergraduate research opportunities",
  "filters": {
    "category": "non_academic",
    "target_segment": ["undergraduate"]
  }
}

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

Query: "masters in Germany, my IELTS is 6.5"
{
  "semantic_query": "masters program Germany",
  "filters": {
    "subtype": ["masters"],
    "category": "academic",
    "country": ["Germany"],
    "language_score_requirements": {"ielts": 6.5}
  }
}

Query: "scholarship without CV requirement for a 16 year old"
{
  "semantic_query": "scholarship for a 16 year old",
  "filters": {
    "documents_not_required": ["cv"],
    "min_age": 16,
    "max_age": 16
  }
}

Query: "منح للمصريين في اوروبا"
{
  "semantic_query": "scholarships in Europe",
  "filters": {
    "country": ["Germany", "France", "UK", "Spain", "Italy", "Netherlands"],
    "eligible_nationalities": ["Egypt"]
  }
}

Query: "عايز اتفسح في اوروبا ببلاش"
{
  "semantic_query": "opportunity in Europe",
  "filters": {
    "country": ["Germany", "France", "UK", "Spain", "Italy", "Netherlands"],
    "fund_type": ["fully_funded"],
    "category": "non_academic"
  }
}

Query: "منحة ماجستير في كندا الايلتس بتاعي 6"
{
  "semantic_query": "masters scholarship Canada",
  "filters": {
    "subtype": ["masters"],
    "category": "academic",
    "country": ["Canada"],
    "language_score_requirements": {"ielts": 6.0}
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
    "language_score_requirements": {{"exam_name": score_number, "only if user mentions exam + score"}},
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
                country=normalize_countries(filters.get("country") or []),
                fund_type=filters.get("fund_type") or [],
                target_segment=filters.get("target_segment") or [],
                documents_not_required=filters.get("documents_not_required") or [],
                language_score_requirements=filters.get("language_score_requirements") or {},
                eligible_nationalities=normalize_countries(filters.get("eligible_nationalities") or []),
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
