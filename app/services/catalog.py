import json
import logging

from redis.exceptions import RedisError

from sqlalchemy import select, func, text

from app.db.models import (
    City,
    Locality,
    Project,
    ProjectType,
    ProjectUnitType,
    UnitType,
)
from app.schemas.rag import StructuredSearchCriteria
from app.schemas.catalog import CatalogProject, CatalogUnit

logger = logging.getLogger(__name__)


class TenantCatalogService:

    PROJECTS_KEY = "tenant:{tenant_id}:projects"

    def __init__(self, session, redis_client):
        self.session = session
        self.redis = redis_client

    def _projects_key(self, tenant_id: int) -> str:
        return self.PROJECTS_KEY.format(tenant_id=tenant_id)

    async def load_catalog(self, tenant_id: int) -> list[CatalogProject]:

        p = Project.__table__
        l = Locality.__table__
        c = City.__table__
        pt = ProjectType.__table__
        put = ProjectUnitType.__table__
        ut = UnitType.__table__

        # Build the aggregated JSON object for each unit row
        unit_json = func.json_build_object(
            'unit_type_id', ut.c.id,
            'code', ut.c.code,
            'name', ut.c.name,
            'size_sqft', put.c.size_sqft,
            'available', put.c.available
        )
        
        statement = (
            select(
                p.c.id.label("project_id"),
                p.c.name.label("project_name"),
                p.c.description,

                c.c.id.label("city_id"),
                c.c.name.label("city_name"),

                l.c.id.label("locality_id"),
                l.c.name.label("locality_name"),

                pt.c.id.label("project_type_id"),
                pt.c.name.label("project_type_name"),

                p.c.status,
                p.c.price_from,
                p.c.price_to,

                func.coalesce(
                    func.json_agg(unit_json).filter(put.c.unit_type_id.is_not(None)),
                    text("'[]'::json")
                ).label("units")
            )
            .select_from(
                p
                .join(l, l.c.id == p.c.locality_id)
                .join(c, c.c.id == l.c.city_id)
                .outerjoin(pt, pt.c.id == p.c.project_type_id)
                .outerjoin(
                    put,
                    put.c.project_id == p.c.id,
                )
                .outerjoin(
                    ut,
                    ut.c.id == put.c.unit_type_id,
                )
            )
            .where(
                    p.c.tenant_id == tenant_id,
                    p.c.active.is_(True),
            )
            .group_by(p.c.id, c.c.id, l.c.id, pt.c.id) 
            .order_by(p.c.id)
        )

        result = await self.session.execute(statement)
        projects: dict[int, CatalogProject] = {}

        for row in result.mappings():
            project_id = row["project_id"]

            if project_id not in projects:
                projects[project_id] = CatalogProject(
                    id=project_id,
                    name=row["project_name"],
                    description=row["description"],
                    city_id=row["city_id"],
                    city_name=row["city_name"],
                    locality_id=row["locality_id"],
                    locality_name=row["locality_name"],
                    project_type_id=row["project_type_id"],
                    project_type_name=row["project_type_name"],
                    status=row["status"],
                    price_from=row["price_from"],
                    price_to=row["price_to"],
                    units=[CatalogUnit(**u) for u in row["units"]] # Fast conversion from native JSON list
                )

        project_list = list(projects.values())

        try:
            await self.redis.set(
                self._projects_key(tenant_id),
                json.dumps(
                    [project.model_dump() for project in project_list],
                    default=str,
                ),
            )
        except RedisError:
            logger.warning("Redis unavailable; catalog cache was not updated")

        return project_list

    async def get_projects(
        self,
        tenant_id: int,
    ) -> list[CatalogProject]:

        try:
            cached = await self.redis.get(self._projects_key(tenant_id))
        except RedisError:
            logger.warning("Redis unavailable; loading catalog from the database")
            cached = None

        if cached:
            data = json.loads(cached)

            return [
                CatalogProject.model_validate(item)
                for item in data
            ]

        return await self.load_catalog(tenant_id)

    async def get_cities(
        self,
        tenant_id: int,
    ) -> list[str]:

        projects = await self.get_projects(tenant_id)

        cities = {
            project.city_name
            for project in projects
            if project.city_name
        }

        return sorted(cities)

    async def get_city_project_counts(
        self,
        tenant_id: int,
    ) -> dict[str, int]:
        projects = await self.get_projects(tenant_id)
        counts: dict[str, int] = {}

        for project in projects:
            counts[project.city_name] = counts.get(project.city_name, 0) + 1

        return dict(sorted(counts.items()))

    async def filter_projects(
        self,
        tenant_id: int,
        criteria: StructuredSearchCriteria,
        limit: int = 10,
    ) -> list[CatalogProject]:

        projects = await self.get_projects(tenant_id)

        requested_unit_types = {
            self._normalize_unit_type(value)
            for value in criteria.unit_types
        }

        filtered: list[CatalogProject] = []

        for project in projects:

            # City
            if criteria.city:
                if project.city_name.lower() != criteria.city.lower():
                    continue

            # Locality
            if criteria.locality:
                if project.locality_name.lower() != criteria.locality.lower():
                    continue

            # Project type
            if criteria.project_type:
                if not project.project_type_name:
                    continue

                if (
                    project.project_type_name.lower()
                    != criteria.project_type.lower()
                ):
                    continue

            # Unit type
            if requested_unit_types:

                project_unit_types = {
                    self._normalize_unit_type(unit.code)
                    for unit in project.units
                }

                if not requested_unit_types.intersection(
                    project_unit_types
                ):
                    continue

            # Budget
            if criteria.budget_min is not None:

                if (
                    project.price_to is not None
                    and project.price_to < criteria.budget_min
                ):
                    continue

            if criteria.budget_max is not None:

                if (
                    project.price_from is not None
                    and project.price_from > criteria.budget_max
                ):
                    continue

            filtered.append(project)

            if len(filtered) >= limit:
                break

        return filtered

    async def get_project(
        self,
        tenant_id: int,
        project_id: int,
    ) -> CatalogProject | None:

        projects = await self.get_projects(tenant_id)

        for project in projects:
            if project.id == project_id:
                return project

        return None

    async def invalidate(
        self,
        tenant_id: int,
    ) -> None:

        try:
            await self.redis.delete(self._projects_key(tenant_id))
        except RedisError:
            logger.warning("Redis unavailable; catalog cache was not invalidated")

    @staticmethod
    def _normalize_unit_type(value: str) -> str:

        return (
            value
            .lower()
            .replace(" ", "")
            .replace("-", "")
        )