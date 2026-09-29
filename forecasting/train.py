"""Train the three quantile models (q=0.1, 0.5, 0.9) on the top SKUs and save them.

    python -m forecasting.train
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import rw_engine
from forecasting.features import FEATURES, MAX_HORIZON, dense_panel, last_complete_week, load_weekly, training_frame

QUANTILES = (0.1, 0.5, 0.9)
TOP_N_SKUS = 500
PARAMS = {
    "objective": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 42,
}
NUM_ROUNDS = 400


def top_skus(n: int = TOP_N_SKUS) -> list[str]:
    with rw_engine().connect() as conn:
        return conn.execute(text(
            "SELECT stock_code FROM transactions WHERE NOT is_cancellation "
            "GROUP BY stock_code ORDER BY SUM(quantity) DESC LIMIT :n"), {"n": n}).scalars().all()


def fit_quantiles(train: pd.DataFrame) -> dict[float, lgb.Booster]:
    models = {}
    ds = lgb.Dataset(train[FEATURES], label=train["y_scaled"], free_raw_data=False)
    for q in QUANTILES:
        models[q] = lgb.train({**PARAMS, "alpha": q}, ds, num_boost_round=NUM_ROUNDS)
    return models


def main() -> None:
    engine = rw_engine()
    last_week = last_complete_week(engine)
    skus = top_skus()
    panel = dense_panel(load_weekly(engine, skus), last_week)
    frame = training_frame(panel)
    print(f"training rows: {len(frame):,} across {len(panel)} SKUs, last complete week {last_week:%Y-%m-%d}")
    models = fit_quantiles(frame)

    out = Path(get_settings().forecast_artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    for q, booster in models.items():
        booster.save_model(str(out / f"lgbm_q{int(q * 100):02d}.txt"))
    (out / "metadata.json").write_text(json.dumps({
        "quantiles": list(QUANTILES),
        "features": FEATURES,
        "max_horizon": MAX_HORIZON,
        "last_complete_week": f"{last_week:%Y-%m-%d}",
        "training_skus": skus,
        "training_rows": len(frame),
    }, indent=2), encoding="utf-8")
    print(f"saved models to {out}")


if __name__ == "__main__":
    main()
