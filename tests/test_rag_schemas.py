from app.schemas.rag import AgentQueryRequest, StructuredProjectResult


def test_agent_query_accepts_structured_search_criteria() -> None:
    request = AgentQueryRequest(
        tenant_id=1,
        query="Tell me about the amenities",
        criteria={
            "city": "Ahmedabad",
            "locality": "Shela",
            "project_type": "Residential",
            "unit_type": "2 BHK",
            "budget_max": 8_000_000,
        },
    )

    assert request.criteria is not None
    assert request.criteria.city == "Ahmedabad"
    assert request.criteria.available_only is True


def test_structured_project_result_keeps_matching_units() -> None:
    project = StructuredProjectResult(
        project_id=42,
        project_name="Lake View",
        city_name="Ahmedabad",
        locality_name="Shela",
        units=[
            {"id": 7, "name": "2 BHK", "price": 7_500_000, "available": True}
        ],
    )

    assert project.units[0].name == "2 BHK"
    assert project.units[0].available is True