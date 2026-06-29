from typing import Any

import psycopg2
from fastapi import APIRouter, HTTPException

from db import fetch_cached_recommendations
from engine import train_model
from similarity import get_similar_product_ids
from vouchers import rank_context_aware_vouchers

router = APIRouter(prefix="/recommend", tags=["recommendations"])
train_router = APIRouter(tags=["training"])


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
            product_ids = []
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
