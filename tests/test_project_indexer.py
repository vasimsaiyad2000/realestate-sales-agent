from app.services.project_indexer import ProjectIndexer


def test_project_content_contains_inventory_context() -> None:
    content = ProjectIndexer._project_content(
        {
            "project_name": "Lake View",
            "city_name": "Pune",
            "locality_name": "Baner",
            "project_type_name": "Commercial",
            "status": "Ready",
            "price_from": 100,
            "price_to": 200,
            "description": "Near transit",
            "brochure_url": "https://example.test/brochure.pdf",
            "location_url": "https://maps.example.test/project",
            "units": [{"name": "Office", "size_sqft": 500, "price": 100, "available": True}],
        }
    )

    assert "Project: Lake View" in content
    assert "Locality: Baner" in content
    assert "Office: 500 sqft" in content
    assert "available" in content
