import json
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import ACTION_SCORES, BRAND_AFFINITY_BONUS_WEIGHT


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

    scored = interactions.copy()
    scored["score"] = scored["action_type"].map(ACTION_SCORES).fillna(0)
    scored = scored[scored["score"] > 0]

    if scored.empty:
        return pd.DataFrame()

    return scored.pivot_table(
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

    return {product_id: score / max_score for product_id, score in scores.items()}


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

    scored = interactions.copy()
    scored["score"] = scored["action_type"].map(ACTION_SCORES).fillna(0)
    scored = scored[scored["score"] > 0]

    if scored.empty:
        return {}

    popularity = scored.groupby("product_id")["score"].sum().sort_values(ascending=False)

    return {int(product_id): float(score) for product_id, score in popularity.items()}


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

            recommendation_scores[int(similar_product_id)] = (
                recommendation_scores.get(int(similar_product_id), 0.0)
                + float(similarity_score * interaction_score)
            )

    return dict(
        sorted(recommendation_scores.items(), key=lambda item: item[1], reverse=True)[:limit]
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
    product_index_by_id = {int(product["id"]): index for index, product in products.iterrows()}
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
        for product_id, _score in user_scores[user_scores > 0]
        .sort_values(ascending=False)
        .head(limit)
        .items()
    ]
