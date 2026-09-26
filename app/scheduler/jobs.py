import logging

from app.core.config import settings
from app.db.base import session_factory
from app.services.embeddings import create_embedding_provider
from app.services.project_indexer import ProjectIndexer

logger = logging.getLogger(__name__)


async def sync_projects_job() -> None:
    logger.info("Starting scheduled project RAG sync job")
    provider = create_embedding_provider()
    async with session_factory() as session:
        try:
            count = await ProjectIndexer(session, provider).sync_all()
            logger.info(
                "Project RAG sync completed successfully",
                extra={
                    "synced_project_count": count,
                    "provider": provider.__class__.__name__,
                },
            )
        except Exception:
            await session.rollback()
            logger.exception("Project RAG sync failed")


def scheduler_settings() -> dict[str, int]:
    return {"seconds": settings.rag_sync_interval_seconds}
