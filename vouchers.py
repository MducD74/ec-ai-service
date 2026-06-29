from typing import Any

from db import fetch_active_vouchers, fetch_product_price


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
        key=lambda v: (v["discount_amount"], v["is_upsell"]),
        reverse=True,
    )[:limit]
