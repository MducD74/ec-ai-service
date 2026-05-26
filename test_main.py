from fastapi.testclient import TestClient
import pandas as pd
import pytest

import recommendation
from main import app


client = TestClient(app)


def test_recommend_existing_user_returns_200_and_recommendations():
    response = client.get("/recommend/u1")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["recommended_product_ids"], list)
    assert len(data["recommended_product_ids"]) > 0
    assert all(isinstance(product_id, int) for product_id in data["recommended_product_ids"])


def test_recommend_cold_start_user_returns_empty_list():
    response = client.get("/recommend/new-user")

    assert response.status_code == 200
    data = response.json()
    assert data["recommended_product_ids"] == []


def test_recommend_response_has_expected_json_keys():
    response = client.get("/recommend/u1")

    assert response.status_code == 200
    assert set(response.json().keys()) == {
        "status",
        "user_id",
        "recommended_product_ids",
    }


def test_recommend_similar_products_returns_top_matches(monkeypatch):
    products = pd.DataFrame(
        [
            {
                "id": 1,
                "name": "Phone A",
                "category": "Smartphone",
                "specifications": {"screen": "oled", "ram": "8gb"},
            },
            {
                "id": 2,
                "name": "Phone B",
                "category": "Smartphone",
                "specifications": {"screen": "oled", "ram": "12gb"},
            },
            {
                "id": 3,
                "name": "Laptop",
                "category": "Laptop",
                "specifications": {"cpu": "intel", "ram": "16gb"},
            },
        ]
    )

    monkeypatch.setattr(recommendation, "fetch_products", lambda: products)

    response = client.get("/recommend/similar/1")

    assert response.status_code == 200
    data = response.json()
    assert data["product_id"] == 1
    assert data["recommended_product_ids"][0] == 2
    assert 1 not in data["recommended_product_ids"]


def test_recommend_similar_products_returns_404_when_database_empty(monkeypatch):
    monkeypatch.setattr(recommendation, "fetch_products", lambda: pd.DataFrame())

    response = client.get("/recommend/similar/1")

    assert response.status_code == 404
    assert response.json()["detail"] == "Product database is empty"


def test_recommend_similar_products_returns_404_when_product_missing(monkeypatch):
    products = pd.DataFrame(
        [
            {
                "id": 1,
                "name": "Phone A",
                "category": "Smartphone",
                "specifications": {"screen": "oled"},
            }
        ]
    )
    monkeypatch.setattr(recommendation, "fetch_products", lambda: products)

    response = client.get("/recommend/similar/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Product not found"


def test_recommend_hybrid_reads_cached_recommendations(monkeypatch):
    monkeypatch.setattr(recommendation, "fetch_cached_recommendations", lambda user_id: [2, 3])
    monkeypatch.setattr(
        recommendation,
        "get_cold_start_recommendations",
        lambda: pytest.fail("cold start fallback should not be called"),
    )

    response = client.get("/recommend/hybrid/1")

    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert data["recommended_product_ids"] == [2, 3]


def test_recommend_hybrid_cold_start_falls_back_to_popular_products(monkeypatch):
    products = pd.DataFrame(
        [
            {
                "id": 1,
                "name": "Phone A",
                "category": "Smartphone",
                "specifications": {"screen": "oled", "ram": "8gb"},
            },
            {
                "id": 2,
                "name": "Phone B",
                "category": "Smartphone",
                "specifications": {"screen": "oled", "ram": "12gb"},
            },
            {
                "id": 3,
                "name": "Laptop",
                "category": "Laptop",
                "specifications": {"cpu": "intel"},
            },
        ]
    )
    interactions = pd.DataFrame(
        [
            {"user_id": 1, "product_id": 1, "action_type": "VIEW"},
            {"user_id": 2, "product_id": 1, "action_type": "VIEW"},
            {"user_id": 2, "product_id": 2, "action_type": "PURCHASE"},
        ]
    )

    monkeypatch.setattr(recommendation, "fetch_cached_recommendations", lambda user_id: None)
    monkeypatch.setattr(recommendation, "fetch_products", lambda: products)
    monkeypatch.setattr(recommendation, "fetch_user_interactions", lambda: interactions)

    response = client.get("/recommend/hybrid/999")

    assert response.status_code == 200
    data = response.json()
    assert 2 in data["recommended_product_ids"]
    assert len(data["recommended_product_ids"]) <= 5


def test_train_refreshes_recommendation_cache(monkeypatch):
    products = pd.DataFrame(
        [
            {
                "id": 1,
                "name": "Phone A",
                "category": "Smartphone",
                "specifications": {"screen": "oled"},
            },
            {
                "id": 2,
                "name": "Phone B",
                "category": "Smartphone",
                "specifications": {"screen": "oled"},
            },
            {
                "id": 3,
                "name": "Laptop",
                "category": "Laptop",
                "specifications": {"cpu": "intel"},
            },
        ]
    )
    interactions = pd.DataFrame(
        [
            {"user_id": 1, "product_id": 2, "action_type": "PURCHASE"},
            {"user_id": 2, "product_id": 2, "action_type": "ADD_TO_CART"},
            {"user_id": 3, "product_id": 1, "action_type": "VIEW"},
        ]
    )
    saved_recommendations: dict[int, list[int]] = {}

    monkeypatch.setattr(recommendation, "fetch_products", lambda: products)
    monkeypatch.setattr(recommendation, "fetch_user_interactions", lambda: interactions)
    monkeypatch.setattr(
        recommendation,
        "replace_recommendation_cache",
        lambda user_recommendations: saved_recommendations.update(user_recommendations),
    )

    response = client.post("/train")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["trained_users"] == 3
    assert set(saved_recommendations.keys()) == {1, 2, 3}
