from __future__ import annotations

from typing import Any

import pandas as pd


def evaluate_ground_truth(decisions: pd.DataFrame, daily: pd.DataFrame, horizons: tuple[int, ...] = (5, 10, 20)) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    prices = daily[["date", "close"]].dropna().sort_values("date").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for _, decision in decisions.iterrows():
        decision_date = pd.Timestamp(decision["date"])
        matched = prices.index[prices["date"] == decision_date]
        if len(matched) == 0:
            continue
        idx = int(matched[0])
        current = float(prices.loc[idx, "close"])
        row = decision.to_dict()
        row["gt_status"] = "evaluated"
        for horizon in horizons:
            future_idx = idx + horizon
            key = f"return_{horizon}d"
            if future_idx >= len(prices):
                row[key] = None
                row["gt_status"] = "insufficient_future_data"
                continue
            row[key] = round((float(prices.loc[future_idx, "close"]) / current - 1) * 100, 2)
        row["gt_pass"] = classify_pass(row)
        rows.append(row)
    return pd.DataFrame(rows)


def classify_pass(row: dict[str, Any]) -> bool | None:
    rating = row.get("rating")
    r5 = row.get("return_5d")
    r20 = row.get("return_20d")
    if r5 is None and r20 is None:
        return None
    r5 = float(r5 or 0)
    r20 = float(r20 or 0)
    if rating == "继续深挖":
        return r5 > 3 or r20 > 8
    if rating == "放入观察":
        return -3 <= r5 <= 3
    if rating == "暂时剔除":
        return r5 < -3 or r20 < -8
    if rating == "信息不足":
        return None
    return None

