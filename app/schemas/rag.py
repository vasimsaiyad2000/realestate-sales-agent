from enum import Enum

from pydantic import BaseModel, Field, model_validator

class IngestChunk(BaseModel):
    content: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    section: str | None = None


class IngestDocumentRequest(BaseModel):
    tenant_id: int = Field(gt=0)
    name: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    checksum: str = Field(min_length=1)
    project_id: int | None = None
    chunks: list[IngestChunk] = Field(min_length=1)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    tenant_id: int = Field(gt=0)
    project_id: int | None = Field(default=None, gt=0)
    limit: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    chunk_id: int
    document_id: int
    project_id: int | None
    project_name: str | None = None
    content: str
    page_number: int | None
    section: str | None
    score: float

class StructuredSearchCriteria(BaseModel):
    city: str | None = Field(default=None, min_length=1)
    locality: str | None = Field(default=None, min_length=1)
    project_type: str | None = Field(default=None, min_length=1)
    unit_types: list[str] = Field(default_factory=list)
    budget_min: float | None = Field(default=None, ge=0)
    budget_max: float | None = Field(default=None, ge=0)
    available_only: bool = True

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, values):
        if isinstance(values, dict):
            values["unit_types"] = values.get("unit_types") or []

        return values
    
class QueryIntent(BaseModel):
    structured_search: bool = False
    semantic_search: bool = False
    semantic_query : str | None = None
    criteria: StructuredSearchCriteria | None = None


class StructuredUnitResult(BaseModel):
    id: int
    name: str
    code: str
    size_sqft: float | None = None
    available: bool


class StructuredProjectResult(BaseModel):
    project_id: int
    project_name: str
    city_name: str
    locality_name: str
    project_type_name: str | None = None
    description: str | None = None
    status: str | None = None
    price_from: float | None = None
    price_to: float | None = None
    units: list[StructuredUnitResult]


# Request and Response schemas for conversational AI query
class AgentQueryRequest(BaseModel):
    tenant_id: int = Field(gt=0)
    project_id: int | None = Field(default=None, gt=0)
    query: str = Field(min_length=1)
    criteria: StructuredSearchCriteria | None = None
    limit: int = Field(default=5, ge=1, le=20)

class AgentQueryAnswer(BaseModel):
    project_id: int 
    project_name: str
    answer: str

class AgentQueryResponse(BaseModel):
    answers: list[AgentQueryAnswer]
    projects: list[StructuredProjectResult] = Field(default_factory=list)
    
class ChatRequest(BaseModel):
    whatsapp_phone_id: str  
    customer_phone: str
    provider_message_id: str
    message: str

class ChatResponse(BaseModel):
    conversation_id: int | None = None
    reply: str
    
class QueryType(str, Enum):
    GREETING = "greeting"
    REAL_ESTATE = "real_estate"
    OTHER = "other"
    
class QueryClassification(BaseModel):
    query_type: QueryType