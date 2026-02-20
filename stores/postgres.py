from __future__ import annotations

import json
import logging
from math import ceil
from typing import Any, Dict, List, Optional, Tuple

import asyncpg

from config import settings

logger = logging.getLogger(__name__)


class PostgresStore:
    """Async PostgreSQL connection pool wrapper."""

    def __init__(self) -> None:
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            min_size=2,
            max_size=10,
            statement_cache_size=0,  # required for pgbouncer/supabase pooler
        )
        logger.info("PostgreSQL pool created (%s:%s)", settings.db_host, settings.db_port)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            logger.info("PostgreSQL pool closed")

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgresStore not connected. Call connect() first.")
        return self._pool

    async def get_opportunities_by_ids(
        self,
        ids: List[str],
        lang: str = "en",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch opportunities by UUIDs, return {id: data} mapping.
        Uses data_en or data_ar based on lang.
        """
        if not ids:
            return {}

        column = "data_en" if lang == "en" else "data_ar"

        query = f"""
            SELECT id::text, {column} AS data
            FROM opportunities
            WHERE id = ANY($1::uuid[])
        """
        rows = await self.pool.fetch(query, ids)

        result = {}
        for row in rows:
            data = row["data"]
            if isinstance(data, str):
                data = json.loads(data)
            result[row["id"]] = data

        return result

    async def list_opportunities(
        self,
        lang: str = "en",
        page: int = 1,
        per_page: int = 12,
        category: Optional[str] = None,
        subtype: Optional[str] = None,
        country: Optional[str] = None,
        fund_type: Optional[str] = None,
        target_segment: Optional[str] = None,
        is_remote: Optional[bool] = None,
        q: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Paginated listing with optional filters on proper columns
        and optional trigram/ILIKE title search.
        Returns (items, total_count).
        """
        column = "data_en" if lang == "en" else "data_ar"
        title_col = "title_en" if lang == "en" else "title_ar"

        where_clauses = []
        params: list = []
        param_idx = 1

        # ── Title search (trigram + ILIKE) ──
        if q:
            where_clauses.append(
                f"({title_col} ILIKE '%' || ${param_idx} || '%'"
                f" OR {title_col} % ${param_idx})"
            )
            params.append(q)
            param_idx += 1

        # ── Filter columns (proper types, indexed) ──
        if category:
            where_clauses.append(f"category = ${param_idx}")
            params.append(category)
            param_idx += 1

        if subtype:
            where_clauses.append(f"subtype @> ARRAY[${param_idx}]::text[]")
            params.append(subtype)
            param_idx += 1

        if country:
            where_clauses.append(f"country @> ARRAY[${param_idx}]::text[]")
            params.append(country)
            param_idx += 1

        if fund_type:
            where_clauses.append(f"fund_type @> ARRAY[${param_idx}]::text[]")
            params.append(fund_type)
            param_idx += 1

        if target_segment:
            where_clauses.append(f"target_segment @> ARRAY[${param_idx}]::text[]")
            params.append(target_segment)
            param_idx += 1

        if is_remote is not None:
            where_clauses.append(f"is_remote = ${param_idx}")
            params.append(is_remote)
            param_idx += 1

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        # ── Count query ──
        count_query = f"SELECT COUNT(*) FROM opportunities {where_sql}"
        total = await self.pool.fetchval(count_query, *params)

        # ── Data query ──
        offset = (page - 1) * per_page

        # If title search is active, order by relevance first
        if q:
            order_sql = f"ORDER BY similarity({title_col}, ${param_idx}) DESC, created_at DESC"
            data_params = [*params, q]  # extra param for similarity()
            param_idx += 1
        else:
            order_sql = "ORDER BY created_at DESC"
            data_params = list(params)

        data_query = f"""
            SELECT id::text, {column} AS data
            FROM opportunities
            {where_sql}
            {order_sql}
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        data_params.extend([per_page, offset])

        rows = await self.pool.fetch(data_query, *data_params)

        items = []
        for row in rows:
            data = row["data"]
            if isinstance(data, str):
                data = json.loads(data)
            items.append({"id": row["id"], "data": data})

        return items, total

    # ──────────────────────────────────────────────
    #  Fetch single
    # ──────────────────────────────────────────────

    async def get_opportunity_by_id(
        self,
        opportunity_id: str,
        lang: str = "en",
    ) -> Optional[Dict[str, Any]]:
        """Fetch a single opportunity by UUID."""
        column = "data_en" if lang == "en" else "data_ar"
        query = f"""
            SELECT id::text, {column} AS data
            FROM opportunities
            WHERE id = $1::uuid
        """
        row = await self.pool.fetchrow(query, opportunity_id)
        if not row:
            return None

        data = row["data"]
        if isinstance(data, str):
            data = json.loads(data)
        return {"id": row["id"], "data": data}


# Singleton instance
pg_store = PostgresStore()
