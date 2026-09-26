from unittest.mock import AsyncMock, patch

import pytest

from app.schemas.rag import QueryIntent, StructuredSearchCriteria
from app.services.rag import RagService


@pytest.mark.asyncio
async def test_classify_query_extracts_structured_criteria() -> None:
    response = {
        "response": (
            '{"structured_search": true, "criteria": {'
            '"city": "Ahmedabad", "locality": "Shela", '
            '"project_type": "Residential", "unit_type": "2 BHK", '
            '"budget_min": null, "budget_max": 8000000, '
            '"available_only": true}}'
        )
    }
    client = AsyncMock()
    client.generate.return_value = response
    service = RagService(None, None)  # type: ignore[arg-type]

    with patch("app.services.rag.ollama.AsyncClient", return_value=client):
        intent = await service._classify_query("2 BHK in Shela Ahmedabad under 80 lakh")

    assert intent == QueryIntent(
        structured_search=True,
        criteria=StructuredSearchCriteria(
            city="Ahmedabad",
            locality="Shela",
            project_type="Residential",
            unit_type="2 BHK",
            budget_max=8_000_000,
        ),
    )
    client.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_classify_query_falls_back_to_vector_search_for_descriptive_query() -> None:
    response = {"response": '{"structured_search": false, "criteria": null}'}
    client = AsyncMock()
    client.generate.return_value = response
    service = RagService(None, None)  # type: ignore[arg-type]

    with patch("app.services.rag.ollama.AsyncClient", return_value=client):
        intent = await service._classify_query("Does the project have a gym?")

    assert intent == QueryIntent()


@pytest.mark.asyncio
async def test_classify_query_routes_when_criteria_exists_without_intent_flag() -> None:
    response = {
        "response": (
            '{"structured_search": false, "criteria": {'
            '"city": null, "locality": "S G Highway", '
            '"project_type": null, "unit_type": null, '
            '"budget_min": null, "budget_max": null, '
            '"available_only": true}}'
        )
    }
    client = AsyncMock()
    client.generate.return_value = response
    service = RagService(None, None)  # type: ignore[arg-type]

    with patch("app.services.rag.ollama.AsyncClient", return_value=client):
        intent = await service._classify_query("I am looking for home in S G Highway")

    assert intent.structured_search is True
    assert intent.criteria is not None
    assert intent.criteria.locality == "S G Highway"