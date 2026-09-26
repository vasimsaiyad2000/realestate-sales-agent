import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_session
from app.schemas.rag import ChatRequest, ChatResponse
from app.services.chat import ChatService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

SessionDependency = Annotated[
    AsyncSession,
    Depends(get_session),
]


@router.post("", response_model=ChatResponse)
async def whatsapp_chat(
    request: ChatRequest,
    session: SessionDependency,
):

    try:
        service = ChatService(session)
        return await service.handle_message(request)
    except Exception:
        logger.exception("Chat failed")
        raise HTTPException(
            status_code=500,
            detail="Chat failed",
        )