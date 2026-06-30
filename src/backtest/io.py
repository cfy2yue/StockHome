from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REQUIRED_DAILY_COLUMNS = {"date", "open", "high", "low", "close", "volume", "amount"}


def read_yaml(path: str | Path) -> Any:
    text = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def write_yaml(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def load_universe(path: str | Path) -> dict:
    data = read_yaml(path)
    train = data.get("train") or []
    test = data.get("test") or []
    if not isinstance(train, list) or not isinstance(test, list):
        raise ValueError("universe must contain list fields: train, test")
    return {"train": train, "test": test}


def load_daily_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename_map = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    missing = REQUIRED_DAILY_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"daily csv missing required columns: {sorted(missing)}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return df


def load_financial_json(path: str | Path) -> list[dict]:
    target = Path(path)
    if not target.exists():
        return []
    data = read_yaml(target)
    if isinstance(data, dict) and "records" in data:
        data = data["records"]
    if not isinstance(data, list):
        return []
    records = []
    for item in data:
        if not isinstance(item, dict):
            continue
        record = dict(item)
        if "publish_date" in record:
            record["publish_date"] = pd.to_datetime(record["publish_date"], errors="coerce")
        records.append(record)
    return records


def load_news_json(path: str | Path) -> list[dict]:
    target = Path(path)
    if not target.exists():
        return []
    data = read_yaml(target)
    if isinstance(data, dict) and "events" in data:
        data = data["events"]
    return data if isinstance(data, list) else []


def load_weights(path: str | Path) -> dict[str, float]:
    data = read_yaml(path)
    return {str(k): float(v) for k, v in data.items()}


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
