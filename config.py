import httpx

from db import get_backend_url


ACTION_SCORES = {
    "VIEW": 1,
    "ADD_TO_CART": 2,
    "PURCHASE": 3,
}
BRAND_AFFINITY_BONUS_WEIGHT = 1.3
DEFAULT_AI_WEIGHTS = (0.4, 0.3, 0.3)


def fetch_ai_weights() -> tuple[float, float, float]:
    try:
        import os

        headers = {}
        token = os.getenv("BACKEND_ADMIN_TOKEN")

        if token:
            headers["Authorization"] = f"Bearer {token}"

        response = httpx.get(
            f"{get_backend_url()}/api/v1/admin/ai-config",
            headers=headers,
            timeout=2.0,
        )
        response.raise_for_status()
        payload = response.json()
        config = payload.get("data") or payload.get("config") or payload

        collaborative_weight = float(config["collaborativeWeight"])
        content_weight = float(config["contentWeight"])
        brand_weight = float(config["brandWeight"])
        return collaborative_weight, content_weight, brand_weight
    except Exception:
        return DEFAULT_AI_WEIGHTS
