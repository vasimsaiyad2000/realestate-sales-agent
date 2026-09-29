import logging
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.base import get_session
from app.schemas.rag import ChatRequest, ChatResponse
from app.services.chat import ChatService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])
webhook_router = APIRouter(tags=["whatsapp-webhook"])

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


@webhook_router.get("/webhook")
async def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == settings.whatsapp_verify_token:
        return PlainTextResponse(content=challenge or "")

    raise HTTPException(status_code=403, detail="Verification failed")


async def send_whatsapp_message(to: str, message: str, phone_number_id: str):
    graph_url = (
        f"https://graph.facebook.com/{settings.whatsapp_graph_version}/"
        f"{phone_number_id}/messages"
    )
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message},
    }

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(graph_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


@webhook_router.post("/webhook")
async def receive_webhook(request: Request, session: SessionDependency):
    body = await request.json()
    service = ChatService(session)

    for entry in body.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            phone_number_id = value.get("metadata", {}).get("phone_number_id")

            for message in value.get("messages", []):
                if message.get("type") != "text" or not phone_number_id:
                    continue

                sender = message.get("from")
                provider_message_id = message.get("id")
                text = message.get("text", {}).get("body")
                if not sender or not provider_message_id or not text:
                    continue

                try:
                    result = await service.handle_message(
                        ChatRequest(
                            whatsapp_phone_id=phone_number_id,
                            customer_phone=sender,
                            provider_message_id=provider_message_id,
                            message=text,
                        )
                    )
                    if result.reply:
                        await send_whatsapp_message(
                            sender,
                            result.reply,
                            phone_number_id,
                        )
                except Exception:
                    logger.exception("WhatsApp webhook message processing failed")
                    raise HTTPException(
                        status_code=500,
                        detail="WhatsApp message processing failed",
                    )

    return JSONResponse({"status": "ok"})