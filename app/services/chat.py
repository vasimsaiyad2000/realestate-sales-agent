import json
import logging
import ollama

from app.core.config import settings
from openai import OpenAI

from datetime import datetime, timezone
from uuid import uuid4
from app.core.redis import redis_client
from time import perf_counter

from sqlalchemy import select
from typing import Optional
from app.core.config import settings
from app.db.models import Conversation, Message, Tenant
from app.schemas.rag import (
    ChatRequest,
    ChatResponse,
    StructuredSearchCriteria,
    QueryType,
    QueryIntent,
    QueryClassification
)
from app.services.embeddings import create_embedding_provider
from app.services.rag import RagService
from app.services.catalog import TenantCatalogService

logger = logging.getLogger(__name__)


class ChatService:

    def __init__(self, session):
        self.session = session
        self.redis = redis_client
        self.rag = RagService(session, create_embedding_provider())
        self.catalog = TenantCatalogService(session, self.redis)

    async def handle_message(self, request: ChatRequest) -> ChatResponse:
        
        # Getting tenant details using whatsapp_phone_id
        tenant = await self.get_tenant_by_whatsapp_id(request.whatsapp_phone_id)
        
        # 2. Check if tenant exists and is active
        if not tenant or not tenant.active:
            return ChatResponse(
                reply=(
                    "This real estate company is not registered with us "
                    "or their account is currently inactive."
                ),
            )
        
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

        # Getting the query type classification
        query_type = await self._classify_query_type(request.message)
        logger.info("Query routed",
            extra={
                "query_type": query_type,
                "user_query": request.message,
            },
        )

         # 1. Greeting
        if query_type == QueryType.GREETING:
            return ChatResponse(
                conversation_id=conversation.id,
                reply=(
                    "Hello! 👋 I’m your real estate assistant." 
                    "I can help you find the right property based on your location, budget, BHK, property type, and preferences."
                    "How can I help you today?"

                ),
        )
            
        # 2. Non-real-estate query
        if query_type == QueryType.OTHER:
            return ChatResponse(
                conversation_id=conversation.id,
                reply=(
                    "I understand your request. My assistance is focused on real estate, including property searches, projects, apartments, villas, commercial spaces, locations, budgets, amenities, and related information."
                    "Please let me know how I can assist you with your real estate requirements."

                ),
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

        # Calling real estate query intent classfier
        intent = await self._classify_real_estate_query(request.message)
        state = self.merge_state(state, intent)
        conversation.search_state = state

        if self._is_broad_property_request(request.message, intent):
            city_counts = await self.catalog.get_city_project_counts(tenant.id)
            state["conversation_stage"] = "qualification"
            state["pending_slot"] = "city"
            conversation.search_state = state

            reply = self._build_city_question(city_counts)
            await self.save_message(
                tenant_id=tenant.id,
                conversation_id=conversation.id,
                provider_message_id=f"assistant-{uuid4()}",
                direction="outbound",
                body=reply,
            )
            conversation.last_message_at = datetime.now(timezone.utc)
            await self.session.commit()

            return ChatResponse(conversation_id=conversation.id, reply=reply)

        structured_projects = []
        semantic_results = []
        project_ids = None

        if intent.structured_search and intent.criteria:
            criteria = StructuredSearchCriteria(
                city=state.get("city"),
                locality=state.get("locality"),
                project_type=state.get("project_type"),
                unit_types=state.get("unit_types"),
                budget_min=state.get("budget_min"),
                budget_max=state.get("budget_max"),
            )

            structured_projects = await self.catalog.filter_projects(
                tenant_id=tenant.id,
                criteria=criteria,
                limit=5,
            )

            project_ids = (
                [p.id for p in structured_projects]
                if structured_projects else None
            )

            state["candidate_project_ids"] = project_ids
            conversation.search_state = state

        if intent.semantic_search and intent.semantic_query:
            semantic_results = await self.rag.search(
                tenant_id=tenant.id,
                query=intent.semantic_query or request.message,
                project_ids=project_ids,
                limit=5,
            )

        
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

    @staticmethod
    def _is_broad_property_request(query: str, intent: QueryIntent) -> bool:
        return not (intent.structured_search
            or intent.semantic_search
            or intent.criteria
        )

    @staticmethod
    def _build_city_question(city_counts: dict[str, int]) -> str:
        if not city_counts:
            return "I can help you find a home. Which city are you interested in?"

        if len(city_counts) == 1:
            city = next(iter(city_counts))
            return (
                f"I can help you find a home in {city}. "
                "Which locality or area are you interested in?"
            )

        options = ", ".join(
            f"{city} ({count} {'project' if count == 1 else 'projects'})"
            for city, count in city_counts.items()
        )

        return (
            f"I can help you find a home. We have {options}. "
            "Which city are you interested in?"
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
Project: {p.name}
City: {p.city_name}
Locality: {p.locality_name}
Project Type :{p.project_type_name}
Price: {p.price_from} - {p.price_to}
Description: {p.description}
Units: {units_text}
Status: {p.status}
"""

        for r in semantic_results:
            project_text += f"\nRelevant info:\n{r.content}\n"

        prompt = f"""
You are a friendly and professional Real Estate Sales Agent for {tenant.name}.
    
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

        client = OpenAI(
            api_key=settings.open_router_api_key,
            base_url=settings.open_router_base_url
        )
            
        response = client.chat.completions.create(
            model="typesafe/jev-router",  # replace with the exact OpenRouter model slug
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
         
        raw_content = response.choices[0].message.content or ""
        return raw_content.strip()
    
    async def _classify_real_estate_query(self, user_query: str) -> QueryIntent:
        started_at = perf_counter()
        logger.info("Query intent classification started for real estate query",
            extra = {
                "user_query" : user_query
            }
        )
        
        prompt = f"""
        You are a real estate search query analyzer.

        TASK:
        Analyze the customer's query and extract structured and semantic search criteria.

        STRUCTURED QUERY FIELD SPECIFICATIONS:
        - `city`: Standard city name (e.g., "Ahmedabad", "Mumbai", "Bangalore").
        - `locality`: Neighborhood, area, landmark, sector, or road name (e.g., "SG Highway", "Whitefield", "Bandra", "Sector 62").
        - `project_type`: Type of property (e.g., "Residential", "Commercial").
        - `unit_types`: List of unit configurations requested (e.g., ["2 BHK", "3 BHK"], "Penthouse").
            - Return a list of canonical unit type codes.
            - The user may use different wording such as "2 BHK", "2bhk", "2 bedroom", or "two bedroom".
            - Normalize these to the matching canonical code.
            - Never return the user's original wording.
            - Only return values from the available unit type codes.
            - Normalization examples:
                - "1 BHK" -> "1BHK"
                - "1bhk" -> "1BHK"
                - "1 bedroom" -> "1BHK"
                - "2 BHK" -> "2BHK"
                - "2bhk" -> "2BHK"
                - "2 bedroom" -> "2BHK"
                - "one bedroom" -> "1BHK"
                - "two bedroom" -> "2BHK"
                - "penthouse" -> "PENTHOUSE"
                
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
            "unit_types": array of strings or empty array (e.g., ["1BHK", "1 Bedroom", "1 BHK"]),
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
            client = OpenAI(
                api_key=settings.open_router_api_key,
                base_url=settings.open_router_base_url
            )
            
            response = client.chat.completions.create(
                model="typesafe/jev-router",  # replace with the exact OpenRouter model slug
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            
            # Safely extract response body whether dictionary or object
            raw_content = response.choices[0].message.content or ""
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

    async def _classify_query_type(self, user_query: str) -> QueryType:
        started_at = perf_counter()
        logger.info("Query Type classification started", 
                    extra = {"user_query" : user_query}
        )
        
        prompt = f"""
        You are a query classifier for a real estate AI agent.

        Classify the customer's message into exactly one category:

        1. greeting
        Examples:
        - hi
        - hello
        - hey
        - good morning
        - good afternoon
        - good evening
        - how are you?

        2. real_estate
        Any query related to:
        - buying property
        - renting property
        - residential property
        - commercial property
        - apartments
        - flats
        - villas
        - bungalows
        - plots
        - projects
        - builders
        - property prices
        - BHK
        - property amenities
        - project details
        - location/locality for property
        - property availability
        - possession
        - investment in real estate
        - property search

        3. other
        Anything unrelated to real estate.

        Important:
        A real-estate query does NOT need to contain a city, budget,
        BHK, or other structured criteria.

        For example:
        - "Do you have a project with a gym?" -> real_estate
        - "Tell me about project ABC" -> real_estate
        - "What is the possession date?" -> real_estate
        - "Looking for something peaceful in Ahmedabad" -> real_estate

        Return JSON only:

        {{
            "query_type": "greeting" | "real_estate" | "other"
        }}

        Customer query:
        {user_query}
        """

        logger.info("base url", extra = {"url" : settings.open_router_base_url})
        try:    
            client = OpenAI(
                api_key=settings.open_router_api_key,
                base_url=settings.open_router_base_url
            )

            
            response = client.chat.completions.create(
                model="typesafe/jev-router",  # replace with the exact OpenRouter model slug
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )

            raw_content = response.choices[0].message.content or ""
            classification = QueryClassification.model_validate_json(
                raw_content.strip()
            )

            logger.info(
                "Query type classification completed",
                extra={
                    "query_type": classification.query_type,
                    "duration_ms": round(
                        (perf_counter() - started_at) * 1000,
                        2,
                    ),
                },
            )

            return classification.query_type

        except Exception as exc:
            logger.error(
                "Query type classification failed",
                extra={
                    "error_type": type(exc).__name__,
                    "error_details": str(exc),
                    "duration_ms": round(
                        (perf_counter() - started_at) * 1000,
                        2,
                    ),
                },
                exc_info=True,
            )

            # Conservative fallback
            return QueryType.REAL_ESTATE