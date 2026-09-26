import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_session
from app.schemas.rag import AgentQueryRequest, AgentQueryResponse, SearchRequest, SearchResult
from app.services.embeddings import create_embedding_provider
from app.services.rag import RagService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])
SessionDependency = Annotated[AsyncSession, Depends(get_session)]


@router.post("/search", response_model=list[SearchResult])
async def search_documents(
    request: SearchRequest, session: SessionDependency
) -> list[SearchResult]:
    logger.info(
        "RAG search request received",
        extra={
            "tenant_id": request.tenant_id,
            "project_id": request.project_id,
            "limit": request.limit,
        },
    )
    try:
        service = RagService(session, create_embedding_provider())
        # Convert single project_id to list for updated search method
        project_ids = [request.project_id] if request.project_id is not None else None
        results = await service.search(
            tenant_id=request.tenant_id,
            query=request.query,
            project_ids=project_ids,
            limit=request.limit,
        )
        logger.info(
            "RAG search request completed",
            extra={"tenant_id": request.tenant_id, "result_count": len(results)},
        )
        return results
    except RuntimeError as exc:
        logger.exception(
            "RAG search runtime error",
            extra={
                "tenant_id": request.tenant_id,
                "project_id": request.project_id,
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "Unexpected error in RAG search",
            extra={
                "tenant_id": request.tenant_id,
                "project_id": request.project_id,
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(status_code=500, detail="RAG search failed") from None


@router.post("/ask", response_model=AgentQueryResponse)
async def ask_agent(
    request: AgentQueryRequest,
    session: SessionDependency,
):
    logger.info(
        "RAG ask request received",
        extra={
            "tenant_id": request.tenant_id,
            "project_id": request.project_id,
            "criteria_supplied": request.criteria is not None,
            "limit": request.limit,
        },
    )
    try:
        service = RagService(session, create_embedding_provider())
        response = await service.ask_agent(
            tenant_id=request.tenant_id,
            user_query=request.query,
            project_id=request.project_id,
            criteria=request.criteria,
            limit=request.limit,
        )
        logger.info(
            "RAG ask request completed",
            extra={
                "tenant_id": request.tenant_id,
                "answer_count": len(response.answers),
                "project_count": len(response.projects),
            },
        )
        return response
    except RuntimeError as exc:
        logger.exception(
            "RAG ask runtime error",
            extra={
                "tenant_id": request.tenant_id,
                "project_id": request.project_id,
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "Unexpected error in RAG ask",
            extra={
                "tenant_id": request.tenant_id,
                "project_id": request.project_id,
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(status_code=500, detail="Search request failed") from None