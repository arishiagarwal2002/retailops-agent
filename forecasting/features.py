"""Weekly demand series and direct multi-horizon features.

Design: one *global* LightGBM model across SKUs, trained in the "direct"
style - each row is (SKU, forecast origin t, horizon h) with features known
at t and target y[t+h]. That gives an honest interval per horizon without
feeding predictions back in recursively.

Targets and level features are divided by a per-SKU scale (mean weekly
demand up to the origin) so one model can serve SKUs that sell 5/week and
500/week. Predictions are multiplied back.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

MIN_HISTORY_WEEKS = 20
MAX_HORIZON = 12
LAGS = (1, 2, 4, 8)  # relative to the first forecast week, i.e. y[t], y[t-1], y[t-3], y[t-7]

FEATURES = [
    "h", "lag_1", "lag_2", "lag_4", "lag_8", "seasonal_52", "seasonal_52_avg3",
    "roll_mean_4", "roll_mean_8", "roll_mean_12", "roll_std_4", "roll_std_12",
    "week_of_year", "month", "weeks_since_first_sale", "price", "uk_share_8", "log_scale",
]

WEEKLY_SQL = """
SELECT stock_code,
       date_trunc('week', invoice_date)::date AS week,
       SUM(quantity)                          AS qty,
       SUM(quantity * unit_price) / NULLIF(SUM(quantity), 0) AS price,
       SUM(CASE WHEN country = 'United Kingdom' THEN quantity ELSE 0 END) AS uk_qty
FROM transactions
WHERE NOT is_cancellation {filter}
GROUP BY 1, 2
"""


def last_complete_week(engine: Engine) -> pd.Timestamp:
    """The final week in the data is partial (data ends Friday 2011-12-09), so drop it."""
    with engine.connect() as conn:
        end = conn.execute(text("SELECT max(invoice_date) FROM transactions")).scalar()
    end = pd.Timestamp(end)
    week_start = (end - pd.Timedelta(days=end.weekday())).normalize()
    return week_start - pd.Timedelta(weeks=1)


def load_weekly(engine: Engine, stock_codes: list[str] | None = None) -> pd.DataFrame:
    params: dict = {}
    flt = ""
    if stock_codes is not None:
        flt = "AND stock_code = ANY(:codes)"
        params["codes"] = list(stock_codes)
    with engine.connect() as conn:
        df = pd.read_sql(text(WEEKLY_SQL.format(filter=flt)), conn, params=params)
    df["week"] = pd.to_datetime(df["week"])
    return df


def dense_panel(weekly: pd.DataFrame, last_week: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Per-SKU frame indexed by every week from first sale to last_week, zeros filled."""
    out: dict[str, pd.DataFrame] = {}
    for code, g in weekly[weekly["week"] <= last_week].groupby("stock_code"):
        idx = pd.date_range(g["week"].min(), last_week, freq="W-MON")
        s = g.set_index("week").reindex(idx)
        s["qty"] = s["qty"].fillna(0).clip(lower=0)
        s["uk_qty"] = s["uk_qty"].fillna(0).clip(lower=0)
        s["price"] = s["price"].ffill().bfill()
        out[code] = s[["qty", "price", "uk_qty"]]
    return out


def _rows_for_series(code: str, s: pd.DataFrame, origins: np.ndarray, horizons: range,
                     with_target: bool) -> list[dict]:
    y = s["qty"].to_numpy(dtype=float)
    price = s["price"].to_numpy(dtype=float)
    uk = s["uk_qty"].to_numpy(dtype=float)
    weeks = s.index
    n = len(y)
    csum = np.cumsum(y)
    rows = []
    for t in origins:
        if t + 1 < MIN_HISTORY_WEEKS:
            continue
        scale = max(csum[t] / (t + 1), 1.0)
        base = {
            "stock_code": code,
            "origin": weeks[t],
            "lag_1": y[t] / scale,
            "lag_2": y[t - 1] / scale if t >= 1 else np.nan,
            "lag_4": y[t - 3] / scale if t >= 3 else np.nan,
            "lag_8": y[t - 7] / scale if t >= 7 else np.nan,
            "roll_mean_4": y[max(0, t - 3): t + 1].mean() / scale,
            "roll_mean_8": y[max(0, t - 7): t + 1].mean() / scale,
            "roll_mean_12": y[max(0, t - 11): t + 1].mean() / scale,
            "roll_std_4": y[max(0, t - 3): t + 1].std() / scale,
            "roll_std_12": y[max(0, t - 11): t + 1].std() / scale,
            "weeks_since_first_sale": t + 1,
            "price": np.nanmean(price[max(0, t - 3): t + 1]),
            "uk_share_8": (uk[max(0, t - 7): t + 1].sum() / max(y[max(0, t - 7): t + 1].sum(), 1.0)),
            "log_scale": np.log1p(scale),
            "scale": scale,
        }
        for h in horizons:
            target_idx = t + h
            if with_target and target_idx >= n:
                break
            target_week = weeks[t] + pd.Timedelta(weeks=h)
            j = target_idx - 52
            row = dict(base)
            row["h"] = h
            row["target_week"] = target_week
            row["week_of_year"] = int(target_week.isocalendar().week)
            row["month"] = target_week.month
            row["seasonal_52"] = y[j] / scale if 0 <= j < n else np.nan
            window = [y[k] for k in (j - 1, j, j + 1) if 0 <= k < n]
            row["seasonal_52_avg3"] = (np.mean(window) / scale) if window else np.nan
            row["seasonal_naive_raw"] = y[j] if 0 <= j < n else np.nan
            row["last_value_raw"] = y[t]
            if with_target:
                row["y"] = y[target_idx]
                row["y_scaled"] = y[target_idx] / scale
            rows.append(row)
    return rows


def training_frame(panel: dict[str, pd.DataFrame], horizons: range = range(1, MAX_HORIZON + 1)) -> pd.DataFrame:
    rows: list[dict] = []
    for code, s in panel.items():
        origins = np.arange(len(s) - 1)
        rows.extend(_rows_for_series(code, s, origins, horizons, with_target=True))
    return pd.DataFrame(rows)


def inference_frame(code: str, s: pd.DataFrame, weeks: int) -> pd.DataFrame:
    """Features at the last observed week for horizons 1..weeks."""
    return pd.DataFrame(_rows_for_series(code, s, np.array([len(s) - 1]), range(1, weeks + 1),
                                         with_target=False))
