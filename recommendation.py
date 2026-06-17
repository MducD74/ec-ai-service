import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import httpx
import psycopg2
from fastapi import APIRouter, HTTPException
from psycopg2.extras import Json, RealDictCursor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

router = APIRouter(prefix="/recommend", tags=["recommendations"])
train_router = APIRouter(tags=["training"])

ACTION_SCORES = {
    "VIEW": 1,
    "ADD_TO_CART": 2,
    "PURCHASE": 3,
}
BRAND_AFFINITY_BONUS_WEIGHT = 1.3
DEFAULT_AI_WEIGHTS = (0.4, 0.3, 0.3)
DEFAULT_BACKEND_URL = "http://localhost:5000"


def read_database_url_from_env_file(path: Path) -> str | None:
    if not path.exists():
        return None

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)

        if key.strip() == "DATABASE_URL":
            return value.strip().strip('"').strip("'")

    return None


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")

    if database_url:
        return database_url

    service_dir = Path(__file__).resolve().parent
    database_url = (
        read_database_url_from_env_file(service_dir / ".env")
        or read_database_url_from_env_file(service_dir.parent / "backend" / ".env")
    )

    if database_url:
        return database_url

    raise RuntimeError("DATABASE_URL is not configured")


def get_backend_url() -> str:
    return os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL).rstrip("/")


def fetch_ai_weights() -> tuple[float, float, float]:
    try:
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


def normalize_database_url_for_psycopg(database_url: str) -> str:
    parsed_url = urlsplit(database_url)
    query_params = [
        (key, value)
        for key, value in parse_qsl(parsed_url.query, keep_blank_values=True)
        if key != "schema"
    ]

    return urlunsplit(
        (
            parsed_url.scheme,
            parsed_url.netloc,
            parsed_url.path,
            urlencode(query_params),
            parsed_url.fragment,
        )
    )


def fetch_products() -> pd.DataFrame:
    query = """
        SELECT
            p.id,
            p.name,
            COALESCE(b.name, '') AS brand,
            COALESCE(c.name, '') AS category,
            p.specifications
        FROM "Product" p
        LEFT JOIN "Brand" b ON b.id = p."brandId"
        LEFT JOIN "Category" c ON c.id = p."categoryId"
    """

    with psycopg2.connect(normalize_database_url_for_psycopg(get_database_url())) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

    return pd.DataFrame(rows, columns=["id", "name", "brand", "category", "specifications"])


def fetch_product_price(product_id: int) -> float:
    query = """
        SELECT price
        FROM "Product"
        WHERE id = %s
    """

    with psycopg2.connect(normalize_database_url_for_psycopg(get_database_url())) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (product_id,))
            row = cursor.fetchone()

    if not row:
        raise LookupError("Product not found")

    return float(row["price"])


def fetch_active_vouchers() -> list[dict[str, Any]]:
    query = """
        SELECT
            id,
            code,
            "discountType" AS discount_type,
            "discountValue" AS discount_value,
            "minOrderValue" AS min_order_value,
            "maxDiscountValue" AS max_discount_value,
            "endDate" AS end_date,
            "usageLimit" AS usage_limit,
            "usedCount" AS used_count
        FROM "Voucher"
        WHERE
            "isActive" = TRUE
            AND "startDate" <= NOW()
            AND "endDate" > NOW()
            AND "usedCount" < "usageLimit"
    """

    with psycopg2.connect(normalize_database_url_for_psycopg(get_database_url())) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

    return [dict(row) for row in rows]


def fetch_user_interactions() -> pd.DataFrame:
    query = """
        SELECT
            "userId" AS user_id,
            "productId" AS product_id,
            "actionType" AS action_type,
            "createdAt" AS created_at
        FROM "UserInteraction"
        WHERE "userId" IS NOT NULL
    """

    with psycopg2.connect(normalize_database_url_for_psycopg(get_database_url())) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

    return pd.DataFrame(
        rows,
        columns=["user_id", "product_id", "action_type", "created_at"],
    )


def fetch_cached_recommendations(user_id: int) -> list[int] | None:
    query = """
        SELECT "recommendedProductIds" AS recommended_product_ids
        FROM "RecommendationCache"
        WHERE "userId" = %s
    """

    with psycopg2.connect(normalize_database_url_for_psycopg(get_database_url())) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (user_id,))
            row = cursor.fetchone()

    if not row:
        return None

    recommended_product_ids = row["recommended_product_ids"]

    if isinstance(recommended_product_ids, str):
        recommended_product_ids = json.loads(recommended_product_ids)

    if not isinstance(recommended_product_ids, list):
        return []

    return [
        int(product_id)
        for product_id in recommended_product_ids
    ]


def replace_recommendation_cache(user_recommendations: dict[int, list[int]]) -> None:
    with psycopg2.connect(normalize_database_url_for_psycopg(get_database_url())) as connection:
        with connection.cursor() as cursor:
            cursor.execute('DELETE FROM "RecommendationCache"')

            for user_id, recommended_product_ids in user_recommendations.items():
                cursor.execute(
                    """
                    INSERT INTO "RecommendationCache"
                        ("userId", "recommendedProductIds", "updatedAt")
                    VALUES (%s, %s, NOW())
                    """,
                    (user_id, Json(recommended_product_ids)),
                )

        connection.commit()


def normalize_specifications(specifications: Any) -> str:
    if specifications is None:
        return ""

    if isinstance(specifications, str):
        return specifications

    return json.dumps(specifications, ensure_ascii=False, sort_keys=True)


def build_product_features(products: pd.DataFrame):
    if products.empty:
        raise ValueError("Product database is empty")

    documents = (
        products["category"].fillna("").astype(str)
        + " "
        + products["specifications"].apply(normalize_specifications)
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    return vectorizer.fit_transform(documents)


def build_user_item_matrix(interactions: pd.DataFrame) -> pd.DataFrame:
    if interactions.empty:
        return pd.DataFrame()

    scored_interactions = interactions.copy()
    scored_interactions["score"] = scored_interactions["action_type"].map(ACTION_SCORES).fillna(0)
    scored_interactions = scored_interactions[scored_interactions["score"] > 0]

    if scored_interactions.empty:
        return pd.DataFrame()

    return scored_interactions.pivot_table(
        index="user_id",
        columns="product_id",
        values="score",
        aggfunc="sum",
        fill_value=0,
    )


def normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}

    max_score = max(scores.values())

    if max_score <= 0:
        return scores

    return {
        product_id: score / max_score
        for product_id, score in scores.items()
    }


def get_product_brand_lookup(products: pd.DataFrame) -> dict[int, str]:
    if products.empty or "brand" not in products.columns:
        return {}

    brand_lookup: dict[int, str] = {}

    for _index, product in products.iterrows():
        brand = product.get("brand")

        if not isinstance(brand, str) or not brand.strip():
            continue

        brand_lookup[int(product["id"])] = brand.strip()

    return brand_lookup


def get_user_favorite_brand(
    user_id: int,
    interactions: pd.DataFrame,
    products: pd.DataFrame,
) -> str | None:
    if interactions.empty:
        return None

    required_columns = {"user_id", "product_id", "action_type"}

    if not required_columns.issubset(interactions.columns):
        return None

    brand_by_product_id = get_product_brand_lookup(products)

    if not brand_by_product_id:
        return None

    user_interactions = interactions[interactions["user_id"] == user_id].copy()

    if user_interactions.empty:
        return None

    user_interactions["brand"] = user_interactions["product_id"].map(brand_by_product_id)
    user_interactions["score"] = user_interactions["action_type"].map(ACTION_SCORES).fillna(0)
    user_interactions = user_interactions[
        user_interactions["brand"].notna() & (user_interactions["score"] > 0)
    ]

    if user_interactions.empty:
        return None

    brand_scores = user_interactions.groupby("brand")["score"].sum().sort_values(ascending=False)

    if brand_scores.empty:
        return None

    return str(brand_scores.index[0])


def apply_brand_affinity_bonus(
    scores: dict[int, float],
    products: pd.DataFrame,
    favorite_brand: str | None,
    bonus_weight: float = BRAND_AFFINITY_BONUS_WEIGHT,
) -> dict[int, float]:
    if not scores or not favorite_brand or bonus_weight <= 0:
        return scores

    brand_by_product_id = get_product_brand_lookup(products)

    if not brand_by_product_id:
        return scores

    normalized_favorite_brand = favorite_brand.strip().casefold()

    return {
        product_id: (
            score + bonus_weight
            if brand_by_product_id.get(product_id, "").casefold() == normalized_favorite_brand
            else score
        )
        for product_id, score in scores.items()
    }


def get_popular_product_scores(interactions: pd.DataFrame) -> dict[int, float]:
    if interactions.empty:
        return {}

    scored_interactions = interactions.copy()
    scored_interactions["score"] = scored_interactions["action_type"].map(ACTION_SCORES).fillna(0)
    scored_interactions = scored_interactions[scored_interactions["score"] > 0]

    if scored_interactions.empty:
        return {}

    popularity = scored_interactions.groupby("product_id")["score"].sum().sort_values(ascending=False)

    return {
        int(product_id): float(score)
        for product_id, score in popularity.items()
    }


def get_item_based_cf_scores(
    user_id: int,
    user_item_matrix: pd.DataFrame,
    limit: int = 20,
) -> dict[int, float]:
    if user_item_matrix.empty or user_id not in user_item_matrix.index:
        return {}

    item_similarity_matrix = cosine_similarity(user_item_matrix.T)
    item_similarity = pd.DataFrame(
        item_similarity_matrix,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns,
    )
    user_scores = user_item_matrix.loc[user_id]
    interacted_products = user_scores[user_scores > 0]
    recommendation_scores: dict[int, float] = {}

    for product_id, interaction_score in interacted_products.items():
        similar_items = item_similarity[product_id].drop(index=product_id)

        for similar_product_id, similarity_score in similar_items.items():
            if similar_product_id in interacted_products.index or similarity_score <= 0:
                continue

            recommendation_scores[int(similar_product_id)] = recommendation_scores.get(
                int(similar_product_id),
                0.0,
            ) + float(similarity_score * interaction_score)

    return dict(
        sorted(
            recommendation_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]
    )


def get_content_based_scores(
    products: pd.DataFrame,
    seed_product_ids: list[int],
    excluded_product_ids: set[int] | None = None,
    limit_per_seed: int = 10,
) -> dict[int, float]:
    if products.empty or not seed_product_ids:
        return {}

    excluded_product_ids = excluded_product_ids or set()
    feature_matrix = build_product_features(products)
    similarity_matrix = cosine_similarity(feature_matrix)
    product_index_by_id = {
        int(product["id"]): index
        for index, product in products.iterrows()
    }
    scores: dict[int, float] = {}

    for seed_product_id in seed_product_ids:
        seed_index = product_index_by_id.get(int(seed_product_id))

        if seed_index is None:
            continue

        similar_products = sorted(
            enumerate(similarity_matrix[seed_index]),
            key=lambda item: item[1],
            reverse=True,
        )

        for similar_index, similarity_score in similar_products[: limit_per_seed + 1]:
            similar_product_id = int(products.iloc[similar_index]["id"])

            if similar_product_id == seed_product_id or similar_product_id in excluded_product_ids:
                continue

            scores[similar_product_id] = max(
                scores.get(similar_product_id, 0.0),
                float(similarity_score),
            )

    return scores


def get_top_user_seed_product_ids(
    user_id: int,
    user_item_matrix: pd.DataFrame,
    limit: int = 3,
) -> list[int]:
    if user_item_matrix.empty or user_id not in user_item_matrix.index:
        return []

    user_scores = user_item_matrix.loc[user_id]

    return [
        int(product_id)
        for product_id, _score in user_scores[user_scores > 0].sort_values(ascending=False).head(limit).items()
    ]


def calculate_hybrid_recommendations(
    user_id: int,
    products: pd.DataFrame,
    interactions: pd.DataFrame,
    user_item_matrix: pd.DataFrame,
    limit: int = 5,
) -> list[int]:
    collaborative_weight, content_weight, brand_weight = fetch_ai_weights()
    product_ids = {int(product_id) for product_id in products["id"].tolist()}
    weighted_scores: dict[int, float] = {}

    if not user_item_matrix.empty and user_id in user_item_matrix.index:
        user_interacted_product_ids = {
            int(product_id)
            for product_id, score in user_item_matrix.loc[user_id].items()
            if score > 0
        }
        cf_scores = normalize_scores(get_item_based_cf_scores(user_id, user_item_matrix))
        seed_product_ids = get_top_user_seed_product_ids(user_id, user_item_matrix)
        cbf_scores = normalize_scores(
            get_content_based_scores(
                products,
                seed_product_ids,
                excluded_product_ids=user_interacted_product_ids,
            )
        )

        for product_id, score in cf_scores.items():
            if product_id in product_ids:
                weighted_scores[product_id] = (
                    weighted_scores.get(product_id, 0.0) + score * collaborative_weight
                )

        for product_id, score in cbf_scores.items():
            if product_id in product_ids:
                weighted_scores[product_id] = (
                    weighted_scores.get(product_id, 0.0) + score * content_weight
                )
    else:
        popular_scores = normalize_scores(get_popular_product_scores(interactions))
        seed_product_ids = list(popular_scores.keys())[:3]
        cbf_scores = normalize_scores(get_content_based_scores(products, seed_product_ids))

        for product_id, score in popular_scores.items():
            if product_id in product_ids:
                weighted_scores[product_id] = (
                    weighted_scores.get(product_id, 0.0) + score * collaborative_weight
                )

        for product_id, score in cbf_scores.items():
            if product_id in product_ids:
                weighted_scores[product_id] = (
                    weighted_scores.get(product_id, 0.0) + score * content_weight
                )

    if not weighted_scores:
        return [int(product_id) for product_id in products["id"].head(limit).tolist()]

    favorite_brand = get_user_favorite_brand(user_id, interactions, products)
    weighted_scores = apply_brand_affinity_bonus(
        weighted_scores,
        products,
        favorite_brand,
        bonus_weight=brand_weight,
    )

    return [
        product_id
        for product_id, _score in sorted(
            weighted_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]
    ]


def get_cold_start_recommendations(limit: int = 5) -> list[int]:
    products = fetch_products()

    if products.empty:
        raise ValueError("Product database is empty")

    interactions = fetch_user_interactions()
    user_item_matrix = build_user_item_matrix(interactions)

    return calculate_hybrid_recommendations(
        user_id=-1,
        products=products,
        interactions=interactions,
        user_item_matrix=user_item_matrix,
        limit=limit,
    )


def train_recommendation_cache(limit: int = 5) -> dict[str, Any]:
    products = fetch_products()

    if products.empty:
        raise ValueError("Product database is empty")

    interactions = fetch_user_interactions()

    if interactions.empty:
        replace_recommendation_cache({})
        return {
            "trained_users": 0,
            "message": "No user interactions found",
        }

    user_item_matrix = build_user_item_matrix(interactions)
    user_ids = [
        int(user_id)
        for user_id in sorted(interactions["user_id"].dropna().unique())
    ]
    user_recommendations = {
        user_id: calculate_hybrid_recommendations(
            user_id=user_id,
            products=products,
            interactions=interactions,
            user_item_matrix=user_item_matrix,
            limit=limit,
        )
        for user_id in user_ids
    }

    replace_recommendation_cache(user_recommendations)

    return {
        "trained_users": len(user_recommendations),
        "message": "Recommendation cache refreshed",
    }


def train_model(limit: int = 5) -> dict[str, Any]:
    return train_recommendation_cache(limit=limit)


def get_similar_product_ids(product_id: int, limit: int = 5) -> list[int]:
    products = fetch_products()

    if products.empty:
        raise ValueError("Product database is empty")

    matching_indexes = products.index[products["id"] == product_id].tolist()

    if not matching_indexes:
        raise LookupError("Product not found")

    feature_matrix = build_product_features(products)
    similarity_matrix = cosine_similarity(feature_matrix)
    product_index = matching_indexes[0]

    similar_products = sorted(
        enumerate(similarity_matrix[product_index]),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        int(products.iloc[index]["id"])
        for index, _score in similar_products
        if index != product_index
    ][:limit]


def calculate_voucher_discount(price: float, voucher: dict[str, Any]) -> float:
    discount_value = float(voucher["discount_value"])

    if voucher["discount_type"] == "PERCENTAGE":
        discount_amount = price * discount_value / 100
        max_discount_value = voucher.get("max_discount_value")

        if max_discount_value is not None:
            discount_amount = min(discount_amount, float(max_discount_value))
    else:
        discount_amount = discount_value

    return max(0.0, min(discount_amount, price))


def rank_context_aware_vouchers(product_id: int, limit: int = 3) -> list[dict[str, Any]]:
    price = fetch_product_price(product_id)
    vouchers = fetch_active_vouchers()
    ranked_vouchers: list[dict[str, Any]] = []

    for voucher in vouchers:
        min_order_value = float(voucher["min_order_value"])

        if min_order_value > price * 2.5:
            continue

        is_upsell = price < min_order_value and (min_order_value - price) / min_order_value < 0.3
        discount_amount = calculate_voucher_discount(price, voucher)

        ranked_vouchers.append(
            {
                "id": int(voucher["id"]),
                "code": voucher["code"],
                "discount_type": voucher["discount_type"],
                "discount_value": float(voucher["discount_value"]),
                "min_order_value": min_order_value,
                "max_discount_value": (
                    float(voucher["max_discount_value"])
                    if voucher.get("max_discount_value") is not None
                    else None
                ),
                "discount_amount": discount_amount,
                "is_upsell": is_upsell,
                "label": "upsell_target" if is_upsell else "best_value",
            }
        )

    return sorted(
        ranked_vouchers,
        key=lambda voucher: (voucher["discount_amount"], voucher["is_upsell"]),
        reverse=True,
    )[:limit]


@router.get("/similar/{product_id}")
def recommend_similar_products(product_id: int) -> dict[str, Any]:
    try:
        product_ids = get_similar_product_ids(product_id)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except psycopg2.Error as error:
        raise HTTPException(status_code=503, detail="Database connection failed") from error

    return {
        "status": "success",
        "product_id": product_id,
        "recommended_product_ids": product_ids,
    }


@router.get("/vouchers/{product_id}")
def recommend_vouchers(product_id: int) -> dict[str, Any]:
    try:
        vouchers = rank_context_aware_vouchers(product_id)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except psycopg2.Error as error:
        raise HTTPException(status_code=503, detail="Database connection failed") from error

    return {
        "status": "success",
        "product_id": product_id,
        "recommended_vouchers": vouchers,
    }


@router.get("/hybrid/{user_id}")
def recommend_hybrid_products(user_id: int) -> dict[str, Any]:
    try:
        product_ids = fetch_cached_recommendations(user_id)

        if product_ids is None:
            product_ids = get_cold_start_recommendations()
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except psycopg2.Error as error:
        raise HTTPException(status_code=503, detail="Database connection failed") from error

    return {
        "status": "success",
        "user_id": user_id,
        "recommended_product_ids": product_ids,
    }


@train_router.post("/train")
def train_recommendations() -> dict[str, Any]:
    try:
        result = train_model()
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except psycopg2.Error as error:
        raise HTTPException(status_code=503, detail="Database connection failed") from error

    return {
        "status": "success",
        **result,
    }
