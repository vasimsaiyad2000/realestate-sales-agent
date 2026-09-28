from datetime import date, datetime, time

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    String,
    Time,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB  # <-- Added missing import

class Base(DeclarativeBase):
    pass


class City(Base):
    __tablename__ = "cities"
    __table_args__ = (
        UniqueConstraint("name", name="uq_cities_name"),
        {"schema": "sales-agent"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)


class Locality(Base):
    __tablename__ = "localities"
    __table_args__ = (
        UniqueConstraint("city_id", "name", name="uq_locality_city_name"),
        Index("idx_locality_city_id", "city_id"),
        {"schema": "sales-agent"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
    )
    city_id: Mapped[int] = mapped_column(
        ForeignKey(
            "sales-agent.cities.id",
            name="fk_locality_city",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)


class ProjectType(Base):
    __tablename__ = "project_types"
    __table_args__ = (
        UniqueConstraint("name", name="uq_project_types_name"),
        {"schema": "sales-agent"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)


class UnitType(Base):
    __tablename__ = "unit_types"
    __table_args__ = (
        UniqueConstraint("code", name="uq_unit_types_code"),
        {"schema": "sales-agent"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
    )
    code: Mapped[str] = mapped_column(String(100), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)


class Tenant(Base):
    __tablename__ = "tenants"
    __table_args__ = (
        UniqueConstraint("whatsapp_phone_id", name="uq_tenants_whatsapp_phone_id"),
        {"schema": "sales-agent"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    whatsapp_phone_id: Mapped[str | None] = mapped_column(Text)
    calendar_id: Mapped[str | None] = mapped_column(Text)
    meta_access_token: Mapped[str | None] = mapped_column(Text)
    timezone: Mapped[str] = mapped_column(String(64), default="UTC", nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class Project(Base):
    __tablename__ = "projects"
    __table_args__ = (
        CheckConstraint(
            "price_from IS NULL OR price_to IS NULL OR price_from <= price_to",
            name="chk_projects_price_range",
        ),
        Index("idx_projects_locality_id", "locality_id"),
        Index("idx_projects_price_range", "price_from", "price_to"),
        Index("idx_projects_project_type_id", "project_type_id"),
        Index("idx_projects_status", "status"),
        Index("idx_projects_tenant_id", "tenant_id"),
        {"schema": "sales-agent"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
    )
    tenant_id: Mapped[int] = mapped_column(
        ForeignKey(
            "sales-agent.tenants.id",
            name="fk_projects_tenant",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    locality_id: Mapped[int] = mapped_column(
        ForeignKey(
            "sales-agent.localities.id",
            name="fk_projects_locality",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    project_type_id: Mapped[int | None] = mapped_column(
        ForeignKey(
            "sales-agent.project_types.id",
            name="fk_projects_project_type",
            ondelete="SET NULL",
        )
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str | None] = mapped_column(Text)
    price_from: Mapped[int | None] = mapped_column(BigInteger)
    price_to: Mapped[int | None] = mapped_column(BigInteger)
    brochure_url: Mapped[str | None] = mapped_column(Text)
    location_url: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

class ProjectUnitType(Base):
    __tablename__ = "project_unit_types"
    __table_args__ = (
        Index("idx_project_unit_types_project_id", "project_id"),
        Index("idx_project_unit_types_unit_type_id", "unit_type_id"),
        Index("idx_project_unit_types_project_unit", "project_id", "unit_type_id"),
        {"schema": "sales-agent"},
    )

    project_id: Mapped[int] = mapped_column(
        ForeignKey(
            "sales-agent.projects.id",
            name="fk_project_unit_types_project",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    unit_type_id: Mapped[int] = mapped_column(
        ForeignKey(
            "sales-agent.unit_types.id",
            name="fk_project_unit_types_unit_type",
            onupdate="CASCADE",
            ondelete="RESTRICT",
        ),
        primary_key=True,
    )
    size_sqft: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,      # ← changed
    )
    available: Mapped[bool] = mapped_column(Boolean, nullable=False)


class Lead(Base):
    __tablename__ = "leads"
    __table_args__ = (
        CheckConstraint("budget IS NULL OR budget >= 0", name="chk_leads_budget"),
        Index("idx_leads_created_at", "created_at"),
        Index("idx_leads_phone", "phone"),
        Index("idx_leads_project_id", "project_id"),
        Index("idx_leads_stage", "stage"),
        Index("idx_leads_tenant_id", "tenant_id"),
        {"schema": "sales-agent"},
    )

    tenant_id: Mapped[int] = mapped_column(
        ForeignKey("sales-agent.tenants.id", name="fk_leads_tenant", ondelete="CASCADE"),
        nullable=False,
    )
    project_id: Mapped[int | None] = mapped_column(
        ForeignKey("sales-agent.projects.id", name="fk_leads_project", ondelete="SET NULL")
    )
    name: Mapped[str | None] = mapped_column(Text)
    phone: Mapped[str] = mapped_column(Text, nullable=False)
    budget: Mapped[int | None] = mapped_column(BigInteger)
    stage: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default="WhatsApp",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(),
        nullable=False,
        server_default=func.now(),
    )
    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
    )


class Appointment(Base):
    __tablename__ = "appointments"
    __table_args__ = (
        Index("idx_appointments_google_event_id", "google_event_id"),
        Index("idx_appointments_status", "status"),
        Index("idx_appointments_visit_date", "visit_date"),
        {"schema": "sales-agent"},
    )

    google_event_id: Mapped[str | None] = mapped_column(Text)
    visit_date: Mapped[date] = mapped_column(Date, nullable=False)
    visit_time: Mapped[time] = mapped_column(Time, nullable=False)
    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default="scheduled",
    )
    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
    )
    lead_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        ForeignKey("sales-agent.leads.id", name="fk_appointments_leads"),
        nullable=False,
    )


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (
        UniqueConstraint("tenant_id", "phone_number", name="uq_conversations_tenant_phone"),
        {"schema": "sales-agent"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("sales-agent.tenants.id", ondelete="CASCADE"))
    phone_number: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(String(32), default="qualifying", nullable=False)
    opted_out: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    last_message_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    search_state: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )
    tenant: Mapped[Tenant] = relationship()


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("tenant_id", "provider_message_id", name="uq_messages_provider_id"),
        Index("idx_messages_conversation_created", "conversation_id", "created_at"),
        {"schema": "sales-agent"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("sales-agent.tenants.id", ondelete="CASCADE"))
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("sales-agent.conversations.id", ondelete="CASCADE")
    )
    provider_message_id: Mapped[str] = mapped_column(Text, nullable=False)
    direction: Mapped[str] = mapped_column(String(16), nullable=False)
    message_type: Mapped[str] = mapped_column(String(32), nullable=False)
    body: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
