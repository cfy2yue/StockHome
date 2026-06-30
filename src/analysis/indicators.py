from __future__ import annotations

import pandas as pd


def to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows or [])
    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
        df = df.sort_values("日期")
    for col in ["收盘", "开盘", "最高", "最低", "成交量"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "收盘" not in out:
        return out
    for n in [5, 20, 60, 120]:
        out[f"MA{n}"] = out["收盘"].rolling(n).mean()
    return out


def atr(df: pd.DataFrame, n: int = 20) -> float | None:
    if not {"最高", "最低", "收盘"}.issubset(df.columns) or len(df) < n + 1:
        return None
    prev_close = df["收盘"].shift(1)
    tr = pd.concat([(df["最高"] - df["最低"]).abs(), (df["最高"] - prev_close).abs(), (df["最低"] - prev_close).abs()], axis=1).max(axis=1)
    value = tr.rolling(n).mean().iloc[-1]
    return None if pd.isna(value) else float(value)
