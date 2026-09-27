import asyncio
import logging

from collections import defaultdict
from time import perf_counter
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import ValidationError

import ollama
from sqlalchemy import ARRAY, JSON, Text, cast, exists, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import City, Locality, Project, ProjectType, ProjectUnitType, UnitType
from app.schemas.rag import (
    AgentQueryAnswer,
    AgentQueryResponse,
    QueryIntent,
    SearchResult,
    StructuredProjectResult,
    StructuredSearchCriteria,
    QueryType, 
    QueryClassification
)
from app.services.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class DocumentInput:
    tenant_id: int
    name: str
    source_url: str
    checksum: str
    chunks: Sequence[dict[str, object]]
    project_id: int | None = None


class RagService:
    def __init__(self, session: AsyncSession, embeddings: EmbeddingProvider) -> None:
        self.session = session
        self.embeddings = embeddings

    async def ingest(self, document: DocumentInput) -> int:
        started_at = perf_counter()
        contents = [str(chunk["content"]) for chunk in document.chunks]
        logger.info(
            "RAG document ingestion started",
            extra={
                "tenant_id": document.tenant_id,
                "project_id": document.project_id,
                "chunk_count": len(contents),
            },
        )
        vectors = await self.embeddings.embed(contents)

        if len(vectors) != len(contents):
            raise ValueError("Embedding provider returned an unexpected number of vectors")

        document_id = await self._upsert_document(document)

        await self.session.execute(
            text('DELETE FROM "sales-agent".document_chunks WHERE document_id = :document_id'),
            {"document_id": document_id},
        )

        for index, (chunk, vector) in enumerate(zip(document.chunks, vectors, strict=True)):
            await self.session.execute(
                text(
                    'INSERT INTO "sales-agent".document_chunks '
                    "(tenant_id, document_id, project_id, chunk_index, content, "
                    "page_number, section, embedding) "
                    "VALUES (:tenant_id, :document_id, :project_id, :chunk_index, "
                    ":content, :page_number, :section, CAST(:embedding AS vector))"
                ),
                {
                    "tenant_id": document.tenant_id,
                    "document_id": document_id,
                    "project_id": document.project_id,
                    "chunk_index": index,
                    "content": chunk["content"],
                    "page_number": chunk.get("page_number"),
                    "section": chunk.get("section"),
                    "embedding": str(vector),
                },
            )

        await self.session.execute(
            text('UPDATE "sales-agent".documents SET status = \'ready\' WHERE id = :id'),
            {"id": document_id},
        )

        await self.session.commit()
        logger.info(
            "RAG document ingestion completed",
            extra={
                "tenant_id": document.tenant_id,
                "project_id": document.project_id,
                "document_id": document_id,
                "chunk_count": len(contents),
                "duration_ms": round((perf_counter() - started_at) * 1000, 2),
            },
        )
        return document_id

    async def search(
        self,
        tenant_id: int,
        query: str,
        project_ids: list[int] | None = None,
        limit: int = 5,
        similarity_threshold: float = 0.50,
    ) -> list[SearchResult]:
        started_at = perf_counter()

        logger.info(
            "Semantic project search started",
            extra={
                "tenant_id": tenant_id,
                "project_count": len(project_ids) if project_ids else 0,
                "project_filter_supplied": project_ids is not None,
                "limit": limit,
                "similarity_threshold": similarity_threshold,
            },
        )

        # ---------------------------------------------------------
        # 1. Generate query embedding
        # ---------------------------------------------------------
        query_vector = await self.embeddings.embed([query])
        vector = query_vector[0]

        if isinstance(vector, list) and vector and isinstance(vector[0], list):
            vector = vector[0]

        # ---------------------------------------------------------
        # 2. Semantic search
        #
        # We:
        # - Search ONLY candidate projects
        # - Apply similarity threshold
        # - Return the BEST matching chunk per project
        # ---------------------------------------------------------
        sql = text(
            """
            WITH ranked_chunks AS (
                SELECT
                    dc.id AS chunk_id,
                    dc.document_id,
                    dc.project_id,
                    p.name AS project_name,
                    dc.content,
                    dc.page_number,
                    dc.section,

                    -- pgvector cosine distance -> cosine similarity
                    1 - (
                        dc.embedding <=> CAST(:query_vector AS vector)
                    ) AS similarity_score,

                    ROW_NUMBER() OVER (
                        PARTITION BY dc.project_id
                        ORDER BY
                            dc.embedding <=> CAST(:query_vector AS vector)
                    ) AS project_rank

                FROM "sales-agent".document_chunks dc

                INNER JOIN "sales-agent".projects p
                    ON p.id = dc.project_id

                WHERE
                    dc.tenant_id = :tenant_id

                    -- Search only projects returned by structured search
                    AND (
                        CAST(:project_ids AS bigint[]) IS NULL
                        OR dc.project_id = ANY(
                            CAST(:project_ids AS bigint[])
                        )
                    )
            )

            SELECT
                chunk_id,
                document_id,
                project_id,
                project_name,
                content,
                page_number,
                section,
                similarity_score

            FROM ranked_chunks

            -- Only the best chunk for each project
            WHERE project_rank = 1 AND similarity_score >= :similarity_threshold
            ORDER BY similarity_score DESC

            LIMIT :limit
            """
        )

        result = await self.session.execute(
            sql,
            {
                "query_vector": str(vector),
                "query": query,
                "tenant_id": tenant_id,
                "project_ids": project_ids if project_ids else None,
                "similarity_threshold": similarity_threshold,
                "limit": limit,
            },
        )

        rows = result.mappings().all()

        search_results = [
            SearchResult(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                project_id=row["project_id"],
                project_name=row["project_name"],
                content=row["content"],
                page_number=row["page_number"],
                section=row["section"],
                score=float(row["similarity_score"]),
            )
            for row in rows
        ]

        logger.info(
            "Semantic project search completed",
            extra={
                "tenant_id": tenant_id,
                "result_count": len(search_results),
                "duration_ms": round(
                    (perf_counter() - started_at) * 1000,
                    2,
                ),
                "results": [
                    {
                        "project_id": result.project_id,
                        "project_name": result.project_name,
                        "score": result.score,
                    }
                    for result in search_results
                ],
            },
        )

        return search_results

    async def search_projects(
        self,
        tenant_id: int,
        criteria: StructuredSearchCriteria,
        limit: int = 10,
    ) -> list[StructuredProjectResult]:
        started_at = perf_counter()
        criteria_fields = [
            field
            for field in (
                "city",
                "locality",
                "project_type",
                "unit_types",
                "budget_min",
                "budget_max",
            )
            if getattr(criteria, field) is not None
        ]
        
        logger.info(
            "Structured project search started",
            extra={
                "tenant_id": tenant_id,
                "criteria_fields": criteria_fields,
                "city" : criteria.city,
                "locality" : criteria.locality,
                "project_type" : criteria.project_type,
                "unit_types" : criteria.unit_types,
                "budget_min" : criteria.budget_min,
                "budget_max" : criteria.budget_max,
                "limit": limit,
            },
        )
        project = Project.__table__
        locality = Locality.__table__
        city = City.__table__
        project_type = ProjectType.__table__
        project_unit_type = ProjectUnitType.__table__
        unit_type = UnitType.__table__

        unit_json = func.json_agg(
            func.json_build_object(
                "id",
                unit_type.c.id,
                "code",
                unit_type.c.code,
                "name",
                unit_type.c.name,
                "size_sqft",
                project_unit_type.c.size_sqft,
                "available",
                project_unit_type.c.available,
            )
        ).filter(unit_type.c.id.is_not(None))

        statement = (
            select(
                project.c.id.label("project_id"),
                project.c.name.label("project_name"),
                project.c.description,
                city.c.name.label("city_name"),
                locality.c.name.label("locality_name"),
                project_type.c.name.label("project_type_name"),
                project.c.status,
                project.c.price_from,
                project.c.price_to,
                func.coalesce(unit_json, cast("[]", JSON)).label("units"),
            )
            .select_from(
                project
                .join(locality, locality.c.id == project.c.locality_id)
                .join(city, city.c.id == locality.c.city_id)
                .outerjoin(project_type, project_type.c.id == project.c.project_type_id)
                .outerjoin(
                    project_unit_type,
                    project_unit_type.c.project_id == project.c.id,
                )
                .outerjoin(unit_type, unit_type.c.id == project_unit_type.c.unit_type_id)
            )
            .where(project.c.tenant_id == tenant_id)
            .group_by(
                project.c.id,
                city.c.name,
                locality.c.name,
                project_type.c.name,
            )
            .order_by(project.c.id)
            .limit(limit)
        )

        if criteria.city is not None:
            statement = statement.where(func.lower(city.c.name) == func.lower(criteria.city))
        if criteria.locality is not None:
            statement = statement.where(
                func.lower(locality.c.name) == func.lower(criteria.locality)
            )
        if criteria.project_type is not None:
            statement = statement.where(
                func.lower(project_type.c.name) == func.lower(criteria.project_type)
            )
        if criteria.unit_types:
            project_unit_type_sub = project_unit_type.alias("put_sub")
            unit_type_sub = unit_type.alias("ut_sub")
            term = func.unnest(cast(criteria.unit_types, ARRAY(Text))).table_valued("term", name ="term")
            term_subquery = select(1).select_from(term).where(
                or_(
                    func.lower(unit_type_sub.c.code).like(
                        "%" + func.lower(term.c.term) + "%"
                    ),
                    func.lower(unit_type_sub.c.name).like(
                        "%" + func.lower(term.c.term) + "%"
                    ),
                    func.lower(func.replace(unit_type_sub.c.name, " ", "")).like(
                        "%"
                        + func.lower(func.replace(term.c.term, " ", ""))
                        + "%"
                    ),
                )
            )
            statement = statement.where(
                exists(
                    select(1)
                    .select_from(
                        project_unit_type_sub.join(
                            unit_type_sub,
                            unit_type_sub.c.id == project_unit_type_sub.c.unit_type_id,
                        )
                    )
                    .where(
                        project_unit_type_sub.c.project_id == project.c.id,
                        exists(term_subquery),
                    )
                )
            )
        if criteria.budget_min is not None:
            statement = statement.where(project.c.price_to >= criteria.budget_min)
        if criteria.budget_max is not None:
            statement = statement.where(project.c.price_from <= criteria.budget_max)

        result = await self.session.execute(statement)
        projects = [StructuredProjectResult.model_validate(dict(row)) for row in result.mappings()]
        logger.info(
            "Structured project search completed",
            extra={
                "tenant_id": tenant_id,
                "criteria_fields": criteria_fields,
                "result_count": len(projects),
                "duration_ms": round((perf_counter() - started_at) * 1000, 2),
            },
        )
        return projects

    async def _upsert_document(self, document: DocumentInput) -> int:
        result = await self.session.execute(
            text(
                'INSERT INTO "sales-agent".documents '
                "(tenant_id, project_id, name, source_url, checksum, status) "
                "VALUES (:tenant_id, :project_id, :name, :source_url, "
                ":checksum, 'processing') "
                "ON CONFLICT (tenant_id, source_url) DO UPDATE SET "
                "name = EXCLUDED.name, "
                "source_url = EXCLUDED.source_url, "
                "project_id = EXCLUDED.project_id, "
                "checksum = EXCLUDED.checksum, "
                "status = 'processing' "
                "RETURNING id"
            ),
            {
                "tenant_id": document.tenant_id,
                "project_id": document.project_id,
                "name": document.name,
                "source_url": document.source_url,
                "checksum": document.checksum,
            },
        )

        return int(result.scalar_one())
