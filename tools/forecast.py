"""forecast_demand: weekly p10/p50/p90 from the LightGBM quantile models.

Models are loaded once at startup; features are built at call time from a
live SQL query of the SKU's weekly sales.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import ro_engine
from forecasting.features import FEATURES, MAX_HORIZON, MIN_HISTORY_WEEKS, dense_panel, inference_frame, load_weekly


class ForecastError(ValueError):
    pass


@lru_cache
def load_models() -> tuple[dict, dict]:
    import lightgbm as lgb

    d = Path(get_settings().forecast_artifacts_dir)
    meta_path = d / "metadata.json"
    if not meta_path.exists():
        raise ForecastError("forecast models not trained; run `python -m forecasting.train`")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    models = {q: lgb.Booster(model_file=str(d / f"lgbm_q{int(q * 100):02d}.txt")) for q in meta["quantiles"]}
    meta["training_skus"] = set(meta["training_skus"])
    return models, meta


def forecast_demand(stock_code: str, weeks: int = 4) -> dict:
    stock_code = stock_code.strip().upper()
    if not 1 <= weeks <= MAX_HORIZON:
        raise ForecastError(f"weeks must be between 1 and {MAX_HORIZON}")
    engine = ro_engine()
    with engine.connect() as conn:
        exists = conn.execute(text("SELECT 1 FROM products WHERE stock_code = :c"), {"c": stock_code}).scalar()
    if not exists:
        raise ForecastError(f"unknown stock_code '{stock_code}'")

    models, meta = load_models()
    last_week = pd.Timestamp(meta["last_complete_week"])
    panel = dense_panel(load_weekly(engine, [stock_code]), last_week)
    series = panel.get(stock_code)
    if series is None or len(series) < MIN_HISTORY_WEEKS:
        n = 0 if series is None else len(series)
        raise ForecastError(
            f"insufficient history: {stock_code} has {n} weeks of sales data, the model needs at least "
            f"{MIN_HISTORY_WEEKS}. Do not estimate a forecast for this SKU.")

    X = inference_frame(stock_code, series, weeks)
    q = np.column_stack([models[a].predict(X[FEATURES]) for a in sorted(models)])
    q = np.clip(np.sort(q, axis=1), 0, None) * X["scale"].to_numpy()[:, None]
    q = np.round(q, 1)
    out_weeks = [
        {"week_start": f"{w:%Y-%m-%d}", "p10": float(a), "p50": float(b), "p90": float(c)}
        for w, (a, b, c) in zip(X["target_week"], q)
    ]
    return {
        "stock_code": stock_code,
        "unit": "units per week (gross sales, excluding cancellations)",
        "forecast_origin_week": f"{last_week:%Y-%m-%d}",
        "interval": "80% prediction interval (p10-p90)",
        "weeks": out_weeks,
        "total_p50": round(float(q[:, 1].sum()), 1),
        "recent_weekly_avg_8w": round(float(series["qty"].iloc[-8:].mean()), 1),
        "in_training_set": stock_code in meta["training_skus"],
        "note": "Summing weekly p10/p90 does not give an interval for the total.",
    }
