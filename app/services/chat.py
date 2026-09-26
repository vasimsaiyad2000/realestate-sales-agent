import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

import ollama
from sqlalchemy import select
from typing import Optional
from app.core.config import settings
from app.db.models import Conversation, Message, Tenant
from app.schemas.rag import (
    ChatRequest,
    ChatResponse,
    StructuredSearchCriteria,
)
from app.services.embeddings import create_embedding_provider
from app.services.rag import RagService

logger = logging.getLogger(__name__)


class ChatService:

    def __init__(self, session):
        self.session = session
        self.rag = RagService(session, create_embedding_provider())

    async def handle_message(self, request: ChatRequest) -> ChatResponse:

        # Getting tenant details using whatsapp_phone_id
        tenant = await self.get_tenant_by_whatsapp_id(request.whatsapp_phone_id)
        
        # 2. Check if tenant exists and is active
        if not tenant or not tenant.active:
            return {
                "status": "error",
                "message": (
                    "This real estate company is not registered with us "
                "or their account is currently inactive."
            )
        }
        
        conversation = await self.get_or_create_conversation(
            tenant_id=tenant.id,
            phone=request.customer_phone,
        )

        # Ignore duplicate webhook
        if await self.message_exists(
            tenant.id,
            request.provider_message_id,
        ):
            return ChatResponse(
                conversation_id=conversation.id,
                reply="",
            )

        await self.save_message(
            tenant_id=tenant.id,
            conversation_id=conversation.id,
            provider_message_id=request.provider_message_id,
            direction="inbound",
            body=request.message,
        )

        history = await self.load_recent_messages(conversation.id)
        state = conversation.search_state or {}

        # Reuse your existing classifier
        intent = await self.rag._classify_query(request.message)
        state = self.merge_state(state, intent)
        conversation.search_state = state

        structured_projects = []
        semantic_results = []

        if intent.structured_search and intent.criteria:
            criteria = StructuredSearchCriteria(
                city=state.get("city"),
                locality=state.get("locality"),
                project_type=state.get("project_type"),
                unit_types=state.get("unit_types"),
                budget_min=state.get("budget_min"),
                budget_max=state.get("budget_max"),
            )

            structured_projects = await self.rag.search_projects(
                tenant_id=tenant.id,
                criteria=criteria,
                limit=5,
            )

            project_ids = [
                p.project_id
                for p in structured_projects
            ]

            if state.get("semantic_query") and project_ids:
                semantic_results = await self.rag.search(
                    tenant_id=tenant.id,
                    query=state["semantic_query"],
                    project_ids=project_ids,
                    limit=5,
                )

            state["candidate_project_ids"] = project_ids
            conversation.search_state = state

        
        reply = await self.generate_reply(
            user_message=request.message,
            history=history,
            state=state,
            structured_projects=structured_projects,
            semantic_results=semantic_results,
            tenant=tenant
        )

        await self.save_message(
            tenant_id=tenant.id,
            conversation_id=conversation.id,
            provider_message_id=f"assistant-{uuid4()}",
            direction="outbound",
            body=reply,
        )

        conversation.last_message_at = datetime.now(timezone.utc)
        await self.session.commit()

        return ChatResponse(
            conversation_id=conversation.id,
            reply=reply,
        )

    async def get_or_create_conversation(
        self,
        tenant_id: int,
        phone: str,
    ) -> Conversation:

        result = await self.session.execute(
            select(Conversation).where(
                Conversation.tenant_id == tenant_id,
                Conversation.phone_number == phone,
            )
        )

        conversation = result.scalar_one_or_none()

        if conversation:
            return conversation

        conversation = Conversation(
            tenant_id=tenant_id,
            phone_number=phone,
            search_state={},
        )

        self.session.add(conversation)
        await self.session.flush()

        return conversation

    async def message_exists(
        self,
        tenant_id: int,
        provider_message_id: str,
    ) -> bool:

        result = await self.session.execute(
            select(Message).where(
                Message.tenant_id == tenant_id,
                Message.provider_message_id == provider_message_id,
            )
        )

        return result.scalar_one_or_none() is not None

    async def save_message(
        self,
        *,
        tenant_id: int,
        conversation_id: int,
        provider_message_id: str,
        direction: str,
        body: str,
    ):

        self.session.add(
            Message(
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                provider_message_id=provider_message_id,
                direction=direction,
                message_type="text",
                body=body,
                created_at=datetime.now(timezone.utc),
            )
        )

    async def load_recent_messages(self, conversation_id: int):

        result = await self.session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(10)
        )

        messages = list(result.scalars().all())
        messages.reverse()

        return messages

    def merge_state(self, state: dict, intent):
        state = dict(state)
        if intent.criteria:
            c = intent.criteria
            
            if c.city:
                state["city"] = c.city

            if c.locality:
                state["locality"] = c.locality

            if c.project_type:
                state["project_type"] = c.project_type

            if c.unit_types:
                state["unit_types"] = c.unit_types

            if c.budget_min is not None:
                state["budget_min"] = c.budget_min

            if c.budget_max is not None:
                state["budget_max"] = c.budget_max

        if intent.semantic_query:
            state["semantic_query"] = intent.semantic_query

        return state

    async def get_tenant_by_whatsapp_id(
        self, 
        whatsapp_phone_id: str
    ) -> Optional[Tenant]:
        """
        Fetch tenant details using whatsapp_phone_id.
        """
        result = await self.session.execute(
            select(Tenant).where(
                Tenant.whatsapp_phone_id == whatsapp_phone_id,
                Tenant.active == True
            )
        )

        return result.scalar_one_or_none()

    async def generate_reply(
        self,
        *,
        user_message,
        history,
        state,
        structured_projects,
        semantic_results,
        tenant
    ):

        history_text = "\n".join(
            f"{m.direction}: {m.body}"
            for m in history
        )

        project_text = ""

        for p in structured_projects:
            units = getattr(p, "units", []) or []

            units_text = ", ".join(
                f"{u.name} {u.code} {u.size_sqft} sqft"
                for u in units
                if getattr(u, "id", None)
            )
            
            project_text += f"""
Project: {p.project_name}
City: {p.city_name}
Locality: {p.locality_name}
Price: {p.price_from} - {p.price_to}
Description: {p.description}
Units: {units_text}
"""

        for r in semantic_results:
            project_text += f"\nRelevant info:\n{r.content}\n"

        prompt = f"""
You are a friendly and professional Real Estate Sales Agent for {tenant.name}.

CONVERSATION INSTRUCTIONS:
1. Primary Goal: Help the customer search for, evaluate, and view real estate properties.
2. TOPIC DRIFT / OFF-TOPIC HANDLING:
   - If the user asks about an unrelated domain (health, coding, weather, sports, general trivia, personal advice):
   - Step A: Briefly and politely decline the off-topic query in ONE short sentence.
   - Step B: Immediately pivot back to real estate using their CURRENT USER SEARCH STATE stored above.
3. RAG CONTEXT USAGE:
   - Use the provided search results to answer valid property questions.
   - Do NOT attempt to answer non-real estate queries using general knowledge.
4. GREETING RULES: 
   - If the user is new (no prior messages), greet them and ask how you can assist with their property search for given tenant. 
   - Provide a brief overview of the types of properties available and how you can help.
   - welcome them to the service and express enthusiasm for assisting with their property search for given tenant.
    
Conversation State:
{json.dumps(state)}

Real Estate Company Name:
{tenant.name}

History:
{history_text}

Projects:
{project_text}

Customer:
{user_message}

Reply naturally in WhatsApp style.
Do not invent any information.
"""

        client = ollama.AsyncClient(
            host=settings.ollama_base_url,
        )

        response = await client.generate(
            model=settings.ollama_llm_model,
            prompt=prompt,
        )

        return response["response"].strip()
    
    