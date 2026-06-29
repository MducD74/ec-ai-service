from sklearn.metrics.pairwise import cosine_similarity

from db import fetch_products
from scoring import build_product_features


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
