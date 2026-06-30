from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Any

import pandas as pd


IMPORTANT_KEYWORDS = {
    "业绩": 7,
    "预增": 7,
    "预减": 7,
    "订单": 6,
    "中标": 6,
    "产能": 6,
    "扩产": 6,
    "减持": 8,
    "增持": 6,
    "诉讼": 8,
    "监管": 8,
    "处罚": 8,
    "融资": 6,
    "定增": 6,
    "价格": 5,
    "政策": 7,
    "停产": 8,
}


@dataclass(frozen=True)
class NewsEvent:
    title: str
    available_at: pd.Timestamp
    source_type: str
    entity_scope: str
    event_type: str
    direction_hint: str
    materiality_score: int
    evidence_level: str
    dedupe_key: str


def filter_news_for_decision(events: list[dict[str, Any]], decision_date: str | pd.Timestamp) -> list[NewsEvent]:
    cutoff = pd.Timestamp(decision_date).replace(hour=15, minute=0, second=0)
    filtered = []
    seen = set()
    for raw in events:
        event = normalize_news_event(raw)
        if event is None:
            continue
        if event.available_at > cutoff:
            continue
        if event.dedupe_key in seen:
            continue
        seen.add(event.dedupe_key)
        filtered.append(event)
    return sorted(filtered, key=lambda x: (x.materiality_score, x.available_at), reverse=True)


def normalize_news_event(raw: dict[str, Any]) -> NewsEvent | None:
    title = str(raw.get("标题") or raw.get("新闻标题") or raw.get("公告标题") or raw.get("title") or raw.get("事件标题") or "").strip()
    if not title:
        return None
    dt = _parse_available_time(raw)
    source = str(raw.get("来源") or raw.get("source") or raw.get("source_type") or "公开聚合")
    source_type = "官方公告" if "公告" in source or "交易所" in source or "巨潮" in source else "公开新闻"
    event_type, score = classify_event(title)
    return NewsEvent(
        title=title,
        available_at=dt,
        source_type=source_type,
        entity_scope=_entity_scope(title),
        event_type=event_type,
        direction_hint=_direction(title),
        materiality_score=score,
        evidence_level="official_disclosure" if source_type == "官方公告" else "public_aggregator",
        dedupe_key=_dedupe_key(title, dt),
    )


def classify_event(title: str) -> tuple[str, int]:
    matched = [(key, score) for key, score in IMPORTANT_KEYWORDS.items() if key in title]
    if not matched:
        return "一般新闻", 2
    key, score = max(matched, key=lambda item: item[1])
    return key, score


def _parse_available_time(raw: dict[str, Any]) -> pd.Timestamp:
    value = raw.get("发布时间") or raw.get("时间") or raw.get("公告日期") or raw.get("date") or raw.get("datetime")
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.Timestamp("1900-01-01")
    if isinstance(value, str) and len(value.strip()) <= 10:
        # 只有日期没有时间，保守认为下一个交易日才可用；这里用自然日 +1 近似，避免同日偷看。
        return pd.Timestamp(ts.date()) + pd.Timedelta(days=1)
    return pd.Timestamp(ts)


def _entity_scope(title: str) -> str:
    if any(word in title for word in ["行业", "政策", "价格"]):
        return "行业/政策"
    if any(word in title for word in ["订单", "业绩", "减持", "增持", "诉讼", "处罚"]):
        return "公司"
    return "不确定"


def _direction(title: str) -> str:
    if any(word in title for word in ["预增", "订单", "中标", "合同", "增持", "突破"]):
        return "正面"
    if any(word in title for word in ["预减", "减持", "诉讼", "处罚", "停产"]):
        return "负面"
    return "不确定"


def _dedupe_key(title: str, ts: pd.Timestamp) -> str:
    compact = "".join(title.split())[:40]
    return f"{ts.date().isoformat()}:{compact}"
