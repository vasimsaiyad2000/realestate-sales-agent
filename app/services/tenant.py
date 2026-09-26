from dataclasses import dataclass


@dataclass(frozen=True)
class TenantContext:
    tenant_id: int
    whatsapp_phone_id: str


def require_tenant(context: TenantContext | None) -> TenantContext:
    if context is None:
        raise ValueError("Tenant context is required")
    return context
