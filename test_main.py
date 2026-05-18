from fastapi.testclient import TestClient

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
