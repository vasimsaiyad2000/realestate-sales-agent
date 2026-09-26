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

    async def _classify_query(self, user_query: str) -> QueryIntent:
        started_at = perf_counter()
        logger.info("Query intent classification started")
        
        prompt = f"""
        You are a real estate search query analyzer.

        TASK:
        Analyze the customer's query and extract structured and semantic search criteria.

        STRUCTURED QUERY FIELD SPECIFICATIONS:
        - `city`: Standard city name (e.g., "Ahmedabad", "Mumbai", "Bangalore").
        - `locality`: Neighborhood, area, landmark, sector, or road name (e.g., "SG Highway", "Whitefield", "Bandra", "Sector 62").
        - `project_type`: Type of property (e.g., "Residential", "Commercial", "Villa", "Plot", "Apartment").
        - `unit_types`: List of unit configurations requested (e.g., ["2 BHK", "3 BHK"], "Penthouse"). Preserve exact wording like "2 BHK".
        - `budget_min`: Lower numeric budget limit in absolute numbers (e.g., "above 50 lakhs" -> 5000000).
        - `budget_max`: Upper numeric budget limit in absolute numbers (e.g., "under 80 lakh" -> 8000000, "1.5 CR" -> 15000000).
            
        Also determine whether the query contains semantic preferences, such as peaceful, green, family-friendly, luxury, near schools, good connectivity, investment potential, etc.
        
        SEMANTIC QUERY FIELD SPECIFICATIONS:
        - Lifestyle: peaceful, luxury, premium, affordable, family-friendly, senior-friendly, pet-friendly, modern, contemporary, traditional
        - Environment: green, lake-view, sea-view, low-noise, pollution-free, eco-friendly, scenic, natural surroundings
        - Connectivity: metro, railway, airport, highway, good connectivity, walkable, near public transport, near office hubs, near hospitals, near shopping malls
        - Community: gated community, clubhouse, children play area, safe neighborhood
        - Education & Daily Needs: near schools, hospitals, shopping mall, supermarket, daily needs, convenience store, pharmacy, daycare, library
        - Amenities: swimming pool, gym, spa, sports facilities, jogging track, park,
        - Investment: investment potential, rental yield, appreciation, upcoming area, high ROI, resale value, future development

        CLASSIFICATION RULE:
        - Set `semantic_search` to true if the query contains semantic preferences (e.g., "peaceful", "green", "family-friendly").
        - Set `semantic_search` to false if the query does not contain any semantic preferences.
        - Set `structured_search` to true if ANY criteria field (city, locality, project_type, unit_types, budget_min, or budget_max) is present or strongly implied.
        - Set `structured_search` to false ONLY if the query is strictly descriptive (asking for brochure PDFs, swimming pool details, builder reputation, possession timelines without criteria).
        
        
        Return JSON only with this exact shape:
        {{
        "semantic_query": string or null,
        "criteria": {{
            "city": string or null,
            "locality": string or null,
            "project_type": string or null,
            "unit_types": array of strings (e.g., ["1BHK", "1 Bedroom", "1 BHK"]),
            "budget_min": number or null,
            "budget_max": number or null
        }} or null
        }}
    
        EXAMPLES:

        Query: "Looking for 2 BHK near SG Highway under 80 lakhs"
        JSON:
        {{
        "semantic_query": null,
        "criteria": {{
            "city": null,
            "locality": "SG Highway",
            "project_type": null,
            "unit_types": ["2 BHK", "1 Bedroom", "3BHK"],
            "budget_min": null,
            "budget_max": 8000000
        }}
        }}

        Query: "Looking for luxury properties with good connectivity?"
        JSON:
        {{
        "semantic_query": "luxury and good connectivity",
        "criteria": null
        }}

        Customer query: {user_query}
        """
        try:
            client = ollama.AsyncClient(host=settings.ollama_base_url)
            response = await client.generate(
                model=settings.ollama_llm_model or "deepseek-r1:7b",
                prompt=prompt,
                format="json",
            )
            
            # Safely extract response body whether dictionary or object
            raw_content = getattr(response, "response", None) or response.get("response", "")
            logger.info("Raw intent classification response received", extra={"raw_response": raw_content})

            # Clean markdown wrappers if Ollama includes them
            cleaned_content = raw_content.strip()
            if cleaned_content.startswith("```"):
                lines = cleaned_content.splitlines()
                cleaned_content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

            intent = QueryIntent.model_validate_json(cleaned_content)
            logger.info("Intent classification response received", 
                        extra={"raw_response": cleaned_content},
            )
            
            criteria = intent.criteria
            has_structured = False

            if criteria is not None:
                extracted_fields = [
                    field
                    for field in (
                        "city",
                        "locality",
                        "project_type",
                        "unit_types",
                        "budget_min",
                        "budget_max",
                    )
                    if getattr(intent.criteria, field, None) is not None
                ]
                if not extracted_fields:
                    return QueryIntent()

                has_structured = any([
                    criteria.city,
                    criteria.locality,
                    criteria.project_type,
                    criteria.unit_types,
                    criteria.budget_min is not None,
                    criteria.budget_max is not None,
                ])
             
            has_semantic = bool(
                intent.semantic_query and intent.semantic_query.strip()
            )
                
            intent.structured_search = has_structured
            intent.semantic_search = has_semantic
            
            if not has_structured:
                intent.criteria = None
            
            logger.info("Query intent classification completed",
                extra={
                    "has_structured": has_structured,
                    "has_semantic": has_semantic,
                    "semantic_query": intent.semantic_query if has_semantic else None,
                    "has_criteria": bool(criteria),
                    "structured_criteria": criteria.model_dump() if criteria else None,
                    "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                },
            )  
            return intent;  
        except (KeyError, OSError, TypeError, ValueError, ValidationError, Exception) as exc:
            logger.error(
                "Query intent classification failed; using structured search",
                extra={
                    "error_type": type(exc).__name__,
                    "error_details": str(exc),  # Include exact message to debug easily
                    "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                },
                exc_info=True  # Prints full stack trace in logs
            )

        logger.info("Falling back to default QueryIntent with structured_search=False and criteria=None")
        return QueryIntent();

    async def ask_agent(
        self,
        tenant_id: int,
        user_query: str,
        project_id: int | None = None,
        criteria: StructuredSearchCriteria | None = None,
        limit: int = 5,
    ) -> AgentQueryResponse:
        
        started_at = perf_counter()
        logger.info("RAG ask started")
        intent = await self._classify_query(user_query)
        
        structured_projects = []
        semantic_results = []
        
        # ----------------------------
        # PostgreSQL structured search
        # ----------------------------
        if intent.structured_search and intent.criteria is not None:
            structured_projects = await self.search_projects(
                tenant_id=tenant_id,
                criteria=intent.criteria,
                limit=limit,
            )
        
        logger.info("structured search completed", extra={"structured_project_count": len(structured_projects)})
        # Fetch project_ids from structured search results for semantic search filtering
        project_ids = [p.project_id for p in structured_projects] if structured_projects else None
        
        # ----------------------------
        # pgvector semantic search
        # ----------------------------
        if intent.semantic_search and intent.semantic_query:
            semantic_results = await self.search(
                tenant_id=tenant_id,
                query=intent.semantic_query,
                project_ids=project_ids,
                limit=limit,
            )
        
        # ----------------------------
        # Build merged project context
        # ----------------------------
        project_map = {}

        # Add SQL results
        for p in structured_projects:
            units = getattr(p, "units", []) or []

            units_text = ", ".join(
                f"{u.name} {u.code} {u.size_sqft} sqft"
                for u in units
                if getattr(u, "id", None)
            )

            context = (
                f"{p.project_name}. "
                f"{p.description}. "
                f"{p.locality_name}, {p.city_name}. "
                f"{p.project_type_name}. "
                f"Price {p.price_from} to {p.price_to}. "
                f"{units_text}"
            )

            project_map[p.project_id] = {
                "project_id": p.project_id,
                "project_name": p.project_name,
                "context": context,
                "sources": [],
            }

        # Merge vector results
        grouped = defaultdict(list)
        for r in semantic_results:
            grouped[r.project_id].append(r)

        for pid, docs in grouped.items():
            semantic_context = "\n\n".join(d.content for d in docs)

            if pid in project_map:
                project_map[pid]["context"] += (
                    f"\n\nSemantic information:\n{semantic_context}"
                )
                project_map[pid]["sources"].extend(docs)
            else:
                project_map[pid] = {
                    "project_id": pid,
                    "project_name": docs[0].project_name,
                    "context": semantic_context,
                    "sources": docs,
                }

        if not project_map:
            return AgentQueryResponse(answers=[], projects=[])

        project_data = list(project_map.values())

        # ----------------------------
        # Generate answers
        # ----------------------------
        async def generate_answer(item):
            prompt = f"""
        You are an expert Real Estate AI Sales Agent.

        Answer ONLY from the provided context.

        If the information is unavailable, say:
        "I couldn't find that information in the provided records."

        Context:
        {item["context"]}

        User Question:
        {user_query}

        Give a concise answer.
        """

            client = ollama.AsyncClient(
                host=settings.ollama_base_url
            )

            response = await client.generate(
                model=settings.ollama_llm_model,
                prompt=prompt,
            )

            return AgentQueryAnswer(
                project_id=item["project_id"],
                project_name=item["project_name"],
                answer=response["response"].strip(),
                sources=item["sources"],
            )

        answers = await asyncio.gather(
            *(generate_answer(item) for item in project_data)
        )

        return AgentQueryResponse(
            answers=list(answers),
            projects=structured_projects,
        )
