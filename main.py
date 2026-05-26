from typing import Any

import pandas as pd
from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity
from recommendation import router as recommendation_router
from recommendation import train_router

app = FastAPI(title="AI Recommendation Service")
app.include_router(recommendation_router)
app.include_router(train_router)


sample_interactions = pd.DataFrame(
    [
        {"user_id": "u1", "category_id": 101, "score": 5},
        {"user_id": "u1", "category_id": 102, "score": 3},
        {"user_id": "u2", "category_id": 101, "score": 4},
        {"user_id": "u2", "category_id": 103, "score": 5},
        {"user_id": "u2", "category_id": 104, "score": 2},
        {"user_id": "u3", "category_id": 102, "score": 4},
        {"user_id": "u3", "category_id": 103, "score": 4},
        {"user_id": "u3", "category_id": 105, "score": 5},
        {"user_id": "u4", "category_id": 101, "score": 2},
        {"user_id": "u4", "category_id": 104, "score": 5},
        {"user_id": "u4", "category_id": 106, "score": 4},
    ]
)


def build_user_item_matrix(interactions: pd.DataFrame) -> pd.DataFrame:
    return interactions.pivot_table(
        index="user_id",
        columns="category_id",
        values="score",
        aggfunc="sum",
        fill_value=0,
    )


def get_recommendations(user_id: str, interactions: pd.DataFrame) -> list[int]:
    user_item_matrix = build_user_item_matrix(interactions)

    if user_id not in user_item_matrix.index:
        return []

    similarity_matrix = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=user_item_matrix.index,
        columns=user_item_matrix.index,
    )

    similar_users = similarity_df[user_id].drop(index=user_id).sort_values(ascending=False)
    current_user_scores = user_item_matrix.loc[user_id]
    interacted_categories = set(current_user_scores[current_user_scores > 0].index)

    recommendation_scores: dict[int, float] = {}

    for similar_user_id, similarity_score in similar_users.items():
        if similarity_score <= 0:
            continue

        similar_user_scores = user_item_matrix.loc[similar_user_id]
        candidate_categories = similar_user_scores[similar_user_scores > 0]

        for category_id, score in candidate_categories.items():
            if category_id in interacted_categories:
                continue

            recommendation_scores[int(category_id)] = recommendation_scores.get(
                int(category_id),
                0.0,
            ) + float(similarity_score * score)

    return [
        category_id
        for category_id, _score in sorted(
            recommendation_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


@app.get("/recommend/{user_id}")
def recommend(user_id: str) -> dict[str, Any]:
    recommended_product_ids = get_recommendations(user_id, sample_interactions)

    return {
        "status": "success",
        "user_id": user_id,
        "recommended_product_ids": recommended_product_ids,
    }
