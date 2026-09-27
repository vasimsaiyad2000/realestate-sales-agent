import logging

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.scheduler import router as scheduler_router
from app.core.config import settings
from app.core.logger import configure_logging
from app.scheduler.jobs import sync_projects_job

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        sync_projects_job,
        "interval",
        seconds=settings.rag_sync_interval_seconds,
        id="project-rag-sync",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
        next_run_time=datetime.now(UTC),
    )
    scheduler.start()
    _.state.scheduler = scheduler
    logger.info(
        "Project RAG scheduler started",
        extra={"interval_seconds": settings.rag_sync_interval_seconds},
    )
    yield
    scheduler.shutdown(wait=False)


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.include_router(health_router)
app.include_router(scheduler_router)
app.include_router(chat_router)
