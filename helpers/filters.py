from __future__ import annotations

import logging
from typing import Optional

from qdrant_client.models import (
    FieldCondition,
    Filter,
    IsNullCondition,
    MatchValue,
    Nested,
    NestedCondition,
    PayloadField,
    Range,
)

from .query_parser import QueryConstraints

logger = logging.getLogger(__name__)


def build_qdrant_filter(constraints: QueryConstraints) -> Optional[Filter]:
    """
    Convert parsed query constraints into a Qdrant Filter object.
    """
    must_conditions = []
    must_not_conditions = []

    for field, value in constraints.model_dump().items():
        if value is None or (isinstance(value, list) and len(value) == 0) or (isinstance(value, dict) and len(value) == 0):
            continue

        # Documents user does NOT want required → must_not against "documents_required"
        if field == "documents_not_required" and isinstance(value, list):
            for doc in value:
                must_not_conditions.append(
                    FieldCondition(
                        key="documents_required",
                        match=MatchValue(value=doc),
                    )
                )
            continue

        # Language score requirements → nested filter on exam_scores
        # Exclude opportunities that require an exam with score > user's score
        # Keeps: no requirements, different exams, same exam with lower/equal score
        if field == "language_score_requirements" and isinstance(value, dict):
            for exam_name, user_score in value.items():
                must_not_conditions.append(
                    NestedCondition(
                        nested=Nested(
                            key="exam_scores",
                            filter=Filter(
                                must=[
                                    FieldCondition(
                                        key="name",
                                        match=MatchValue(value=exam_name.lower()),
                                    ),
                                    FieldCondition(
                                        key="score",
                                        range=Range(gt=float(user_score)),
                                    ),
                                ]
                            ),
                        )
                    )
                )
            continue

        # Eligible nationalities: match "all" OR specific nationality
        if field == "eligible_nationalities" and isinstance(value, list):
            should_conditions = [
                FieldCondition(key="eligible_nationalities", match=MatchValue(value="all"))
            ]
            for nat in value:
                should_conditions.append(
                    FieldCondition(key="eligible_nationalities", match=MatchValue(value=nat))
                )
            must_conditions.append(Filter(should=should_conditions))
            continue

        # List fields (KEYWORD type in Qdrant)
        if isinstance(value, list):
            if len(value) == 1:
                must_conditions.append(
                    FieldCondition(key=field, match=MatchValue(value=value[0]))
                )
            else:
                should_conditions = [
                    FieldCondition(key=field, match=MatchValue(value=v))
                    for v in value
                ]
                must_conditions.append(Filter(should=should_conditions))

        # Boolean fields
        elif isinstance(value, bool):
            must_conditions.append(
                FieldCondition(key=field, match=MatchValue(value=value))
            )

        # min_age: opportunity.min_age <= user_age OR null
        elif field == "min_age" and isinstance(value, int):
            must_conditions.append(
                Filter(
                    should=[
                        FieldCondition(key="min_age", range=Range(lte=value)),
                        IsNullCondition(is_null=PayloadField(key="min_age")),
                    ]
                )
            )

        # max_age: opportunity.max_age >= user_age OR null
        elif field == "max_age" and isinstance(value, int):
            must_conditions.append(
                Filter(
                    should=[
                        FieldCondition(key="max_age", range=Range(gte=value)),
                        IsNullCondition(is_null=PayloadField(key="max_age")),
                    ]
                )
            )

        # gpa: opportunity.gpa <= user_gpa OR null
        elif field == "gpa" and isinstance(value, (int, float)):
            must_conditions.append(
                Filter(
                    should=[
                        FieldCondition(key="gpa", range=Range(gte=0, lte=value)),
                        IsNullCondition(is_null=PayloadField(key="gpa")),
                    ]
                )
            )

    if not must_conditions and not must_not_conditions:
        return None

    return Filter(
        must=must_conditions if must_conditions else None,
        must_not=must_not_conditions if must_not_conditions else None,
    )
