from typing import Any

import pandas as pd
from logger import ai_service_log

from config import fetch_ai_weights
from db import (
    fetch_products,
    fetch_user_interactions,
    replace_recommendation_cache,
)
from scoring import (
    apply_brand_affinity_bonus,
    build_user_item_matrix,
    get_content_based_scores,
    get_item_based_cf_scores,
    get_popular_product_scores,
    get_top_user_seed_product_ids,
    get_user_favorite_brand,
    normalize_scores,
)


def calculate_hybrid_recommendations(
    user_id: int,
    products: pd.DataFrame,
    interactions: pd.DataFrame,
    user_item_matrix: pd.DataFrame,
    limit: int = 5,
) -> list[int]:
    collaborative_weight, content_weight, brand_weight = fetch_ai_weights()
    ai_service_log.info(
        f"  [user_id={user_id}] weights: collaborative={collaborative_weight}, "
        f"content={content_weight}, brand={brand_weight}"
    )

    product_ids = {int(product_id) for product_id in products["id"].tolist()}
    weighted_scores: dict[int, float] = {}

    if not user_item_matrix.empty and user_id in user_item_matrix.index:
        user_interacted_product_ids = {
            int(product_id)
            for product_id, score in user_item_matrix.loc[user_id].items()
            if score > 0
        }
        ai_service_log.info(
            f"  [user_id={user_id}] interacted with {len(user_interacted_product_ids)} products: "
            f"{sorted(user_interacted_product_ids)}"
        )

        cf_scores = normalize_scores(get_item_based_cf_scores(user_id, user_item_matrix))
        ai_service_log.info(
            f"  [user_id={user_id}] CF scores ({len(cf_scores)} candidates): "
            f"{dict(list(sorted(cf_scores.items(), key=lambda x: x[1], reverse=True))[:5])}"
        )

        seed_product_ids = get_top_user_seed_product_ids(user_id, user_item_matrix)
        ai_service_log.info(
            f"  [user_id={user_id}] CBF seeds (top interacted): {seed_product_ids}"
        )

        cbf_scores = normalize_scores(
            get_content_based_scores(
                products,
                seed_product_ids,
                excluded_product_ids=user_interacted_product_ids,
            )
        )
        ai_service_log.info(
            f"  [user_id={user_id}] CBF scores ({len(cbf_scores)} candidates): "
            f"{dict(list(sorted(cbf_scores.items(), key=lambda x: x[1], reverse=True))[:5])}"
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
        ai_service_log.info(
            f"  [user_id={user_id}] no interaction history, falling back to popularity + CBF"
        )

        popular_scores = normalize_scores(get_popular_product_scores(interactions))
        seed_product_ids = list(popular_scores.keys())[:3]
        ai_service_log.info(
            f"  [user_id={user_id}] popularity seeds: {seed_product_ids}"
        )

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
        fallback = [int(product_id) for product_id in products["id"].head(limit).tolist()]
        ai_service_log.info(
            f"  [user_id={user_id}] no weighted scores, fallback to first {limit} products: {fallback}"
        )
        return fallback

    favorite_brand = get_user_favorite_brand(user_id, interactions, products)
    ai_service_log.info(
        f"  [user_id={user_id}] favorite brand: {favorite_brand!r}"
    )

    weighted_scores = apply_brand_affinity_bonus(
        weighted_scores,
        products,
        favorite_brand,
        bonus_weight=brand_weight,
    )

    top_scores = dict(
        list(sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True))[:limit]
    )
    ai_service_log.info(
        f"  [user_id={user_id}] final top scores (before trim): {top_scores}"
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
    ai_service_log.info(f"Fetched {len(interactions)} user interactions for cold start recommendations")
    user_item_matrix = build_user_item_matrix(interactions)

    return calculate_hybrid_recommendations(
        user_id=-1,
        products=products,
        interactions=interactions,
        user_item_matrix=user_item_matrix,
        limit=limit,
    )


def train_recommendation_cache(limit: int = 5) -> dict[str, Any]:
    ai_service_log.info("Starting recommendation cache training...")

    products = fetch_products()

    if products.empty:
        raise ValueError("Product database is empty")

    ai_service_log.info(f"Loaded {len(products)} products from database")

    interactions = fetch_user_interactions()

    if interactions.empty:
        ai_service_log.info("No user interactions found, clearing recommendation cache")
        replace_recommendation_cache({})
        return {
            "trained_users": 0,
            "message": "No user interactions found",
        }

    ai_service_log.info(
        f"Loaded {len(interactions)} user interactions "
        f"({interactions['user_id'].nunique()} unique users, "
        f"{interactions['product_id'].nunique()} unique products)"
    )

    user_item_matrix = build_user_item_matrix(interactions)
    ai_service_log.info(
        f"Built user-item matrix: {user_item_matrix.shape[0]} users x {user_item_matrix.shape[1]} products"
    )

    user_ids = [int(user_id) for user_id in sorted(interactions["user_id"].dropna().unique())]
    ai_service_log.info(f"Computing hybrid recommendations for {len(user_ids)} users...")

    user_recommendations: dict[int, list[int]] = {}

    for i, user_id in enumerate(user_ids, start=1):
        ai_service_log.info(f"[{i}/{len(user_ids)}] Processing user_id={user_id}...")
        recs = calculate_hybrid_recommendations(
            user_id=user_id,
            products=products,
            interactions=interactions,
            user_item_matrix=user_item_matrix,
            limit=limit,
        )
        user_recommendations[user_id] = recs
        ai_service_log.info(
            f"[{i}/{len(user_ids)}] user_id={user_id} -> recommended_product_ids={recs}"
        )

    replace_recommendation_cache(user_recommendations)
    ai_service_log.info(
        f"Recommendation cache updated successfully for {len(user_recommendations)} users"
    )

    return {
        "trained_users": len(user_recommendations),
        "message": "Recommendation cache refreshed",
    }


def train_model(limit: int = 5) -> dict[str, Any]:
    return train_recommendation_cache(limit=limit)