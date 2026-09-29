import json

import pytest
from starlette.requests import Request

from app.api import chat
from app.schemas.rag import ChatResponse


@pytest.mark.asyncio
async def test_webhook_passes_message_to_chat_and_sends_reply(monkeypatch) -> None:
    received_requests = []
    sent_messages = []

    class FakeChatService:
        def __init__(self, session):
            pass

        async def handle_message(self, request):
            received_requests.append(request)
            return ChatResponse(conversation_id=3, reply="Here are some homes.")

    async def fake_send(to: str, message: str, phone_number_id: str):
        sent_messages.append((to, message, phone_number_id))

    monkeypatch.setattr(chat, "ChatService", FakeChatService)
    monkeypatch.setattr(chat, "send_whatsapp_message", fake_send)

    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "metadata": {"phone_number_id": "business-number-id"},
                            "messages": [
                                {
                                    "from": "15551234567",
                                    "id": "wamid.inbound-1",
                                    "type": "text",
                                    "text": {"body": "Show me apartments"},
                                }
                            ],
                        }
                    }
                ]
            }
        ]
    }
    raw_body = json.dumps(payload).encode()

    async def receive():
        return {"type": "http.request", "body": raw_body, "more_body": False}

    request = Request(
        {"type": "http", "method": "POST", "path": "/webhook", "headers": []},
        receive,
    )

    response = await chat.receive_webhook(request, session=object())

    assert response.status_code == 200
    assert received_requests[0].whatsapp_phone_id == "business-number-id"
    assert received_requests[0].customer_phone == "15551234567"
    assert received_requests[0].provider_message_id == "wamid.inbound-1"
    assert received_requests[0].message == "Show me apartments"
    assert sent_messages == [
        ("15551234567", "Here are some homes.", "business-number-id")
    ]