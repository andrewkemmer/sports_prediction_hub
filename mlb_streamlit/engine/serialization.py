"""Pure-Python serialization helpers for the Streamlit dashboard UI layer.

Zero reliance on pandas or scikit-learn — protects the Streamlit Cloud
deployment from heavy C-extension dependencies.
"""

import json


def safe_serialize_market_weights(optimized_weights: dict) -> str:
    """
    Serializes weights using pure Python primitives.
    Guarantees zero reliance on pandas or scikit-learn to protect Streamlit Cloud.
    """
    # Enforce standard string keys to match the active production models
    required_models = [
        "Logistic Regression (L2, λ=0.1)",
        "Neural Network (MLP)",
        "Random Forest",
        "XGBoost",
        "LightGBM",
    ]

    # Map the market types exactly as defined in engine/markets.py
    required_markets = ["MONEYLINE", "TOTAL", "RUN_LINE"]

    clean_payload = {}

    for market in required_markets:
        clean_payload[market] = {}
        # Safely extract the market data if it exists
        market_data = optimized_weights.get(market, {})

        for model in required_models:
            # Fallback to 0.0 if a model didn't receive weight in that market
            raw_weight = market_data.get(model, 0.0)
            clean_payload[market][model] = round(float(raw_weight), 4)

    return json.dumps(
        {
            "status": "aligned",
            "ensemble_architecture": "Walk-forward selected stacking ensemble",
            "market_weights": clean_payload,
        },
        indent=2,
    )
