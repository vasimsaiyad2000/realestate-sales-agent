import hashlib
import json
import logging
from collections.abc import Mapping

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.chunking import chunk_text
from app.services.embeddings import EmbeddingProvider
from app.services.rag import DocumentInput, RagService

logger = logging.getLogger(__name__)


class ProjectIndexer:
    def __init__(self, session: AsyncSession, embeddings: EmbeddingProvider) -> None:
        self.session = session
        self.rag = RagService(session, embeddings)

    async def sync_all(self) -> int:
        logger.info("Project indexer query started")
        result = await self.session.execute(
            text(
                'SELECT p.id AS project_id, p.tenant_id, p.name AS project_name, '
                'p.status, p.price_from, p.price_to, p.brochure_url, p.location_url, '
                'p.description, c.name AS city_name, l.name AS locality_name, '
                'pt.name AS project_type_name, '
                "COALESCE(json_agg(json_build_object('name', ut.name, 'code', ut.code, 'size_sqft', "
                "put.size_sqft,  " 
                "'available',   CASE WHEN put.available = TRUE THEN 'Yes' ELSE 'No' END) "
                "ORDER BY ut.id) FILTER (WHERE ut.id IS NOT NULL), '[]'::json) AS units "
                'FROM "sales-agent".projects p '
                'JOIN "sales-agent".locality l ON l.id = p.locality_id '
                'JOIN "sales-agent".cities c ON c.id = l.city_id '
                'LEFT JOIN "sales-agent".project_types pt ON pt.id = p.project_type_id '
                'LEFT JOIN "sales-agent".project_unit_types put ON put.project_id = p.id '
                'LEFT JOIN "sales-agent".unit_types ut ON ut.id = put.unit_type_id '
                'GROUP BY p.id, c.name, l.name, pt.name '
                'ORDER BY p.tenant_id, p.id'
            )
        )
        projects = list(result.mappings())
        logger.info("Project indexer query completed", extra={"project_count": len(projects)})
        count = 0
        for project in projects:
            logger.info(
                "Project indexing started",
                extra={
                    "tenant_id": project["tenant_id"],
                    "project_id": project["project_id"],
                },
            )
            await self.sync_project(project)
            count += 1
        logger.info("Project indexing completed", extra={"project_count": count})
        return count

    async def sync_project(self, project: Mapping[str, object]) -> int:
        content = self._project_content(project)
        checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
        chunks = [
            {"content": chunk.content, "section": "Project details"}
            for chunk in chunk_text(content)
        ]
        return await self.rag.ingest(
            DocumentInput(
                tenant_id=int(project["tenant_id"]),
                project_id=int(project["project_id"]),
                name=f"Project: {project['project_name']}",
                source_url=f"project://{project['tenant_id']}/{project['project_id']}",
                checksum=checksum,
                chunks=chunks,
            )
        )

    @staticmethod
    def _project_content(project: Mapping[str, object]) -> str:
        units = project["units"]
        if isinstance(units, str):
            units = json.loads(units)
        unit_lines = [
            f"{unit['name']}: {unit.get('size_sqft') or 'size unavailable'} sqft, "
            f"{'available' if unit.get('available') else 'not available'}"
            for unit in units
        ]
        return "\n".join(
            [
                f"Project: {project['project_name']}",
                f"City: {project['city_name']}",
                f"Locality: {project['locality_name']}",
                f"Project type: {project['project_type_name'] or 'not specified'}",
                f"Status: {project['status'] or 'not specified'}",
                (
                    f"Price range: {project['price_from'] or 'not specified'} to "
                    f"{project['price_to'] or 'not specified'}"
                ),
                f"Description: {project['description'] or 'not available'}",
                f"Brochure: {project['brochure_url'] or 'not available'}",
                f"Location: {project['location_url'] or 'not available'}",
                "Units:",
                *unit_lines,
            ]
        )
