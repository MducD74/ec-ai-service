import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import psycopg2
from psycopg2.extras import Json, RealDictCursor


DEFAULT_BACKEND_URL = "http://localhost:3000"


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

    return [int(product_id) for product_id in recommended_product_ids]


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
