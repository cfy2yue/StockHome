from __future__ import annotations

import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    for window in [5, 20, 60, 120]:
        out[f"ma{window}"] = close.rolling(window).mean()
        out[f"return_{window}d"] = close.pct_change(window) * 100
    out["ma200"] = close.rolling(200).mean()
    out["ma200_slope20"] = out["ma200"] - out["ma200"].shift(20)
    out["volume_ma20"] = out["volume"].rolling(20).mean()
    out["volume_ratio20"] = out["volume"] / out["volume_ma20"]
    out["rsi14"] = rsi(close, 14)
    macd_df = macd(close)
    out = pd.concat([out, macd_df], axis=1)
    out["atr20"] = atr(out, 20)
    out["bb_mid20"] = close.rolling(20).mean()
    out["bb_std20"] = close.rolling(20).std()
    out["bb_upper20"] = out["bb_mid20"] + 2 * out["bb_std20"]
    out["bb_lower20"] = out["bb_mid20"] - 2 * out["bb_std20"]
    out["drawdown60"] = (close / close.rolling(60).max() - 1) * 100
    return out


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series) -> pd.DataFrame:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = (dif - dea) * 2
    return pd.DataFrame({"macd_dif": dif, "macd_dea": dea, "macd_hist": hist})


def atr(df: pd.DataFrame, window: int = 20) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()
