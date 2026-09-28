from pydantic import BaseModel, Field


class CatalogUnit(BaseModel):
    unit_type_id: int
    code: str
    name: str
    size_sqft: int
    available: bool


class CatalogProject(BaseModel):
    id: int
    name: str
    description: str | None = None

    city_id: int
    city_name: str

    locality_id: int
    locality_name: str

    project_type_id: int | None = None
    project_type_name: str | None = None

    status: str | None = None
    price_from: int | None = None
    price_to: int | None = None

    units: list[CatalogUnit] = Field(default_factory=list)