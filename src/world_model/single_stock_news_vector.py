from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


ENTITY_SCOPES = ("self", "peer", "sector", "upstream", "downstream", "macro")
DIRECTIONS = ("positive", "negative", "mixed", "unknown")
EVIDENCE_LEVELS = ("official_disclosure", "public_aggregator", "media_report", "model_inferred", "manual_review_required")
EVENT_TYPES = (
    "earnings",
    "guidance",
    "regulation",
    "litigation",
    "shareholder",
    "contract",
    "capacity",
    "price",
    "supply_demand",
    "safety",
    "policy",
    "sentiment",
)

SINGLE_STOCK_NEWS_VECTOR_DIMENSIONS = [
    "single_news_count_30d",
    "single_news_official_count_30d",
    "single_news_public_count_30d",
    "single_news_media_count_30d",
    "single_news_model_inferred_count_30d",
    "single_news_manual_review_count_30d",
    *[f"single_news_{scope}_count_30d" for scope in ENTITY_SCOPES],
    *[f"single_news_{scope}_materiality_30d" for scope in ENTITY_SCOPES],
    *[f"single_news_{direction}_count_30d" for direction in DIRECTIONS],
    *[f"single_news_{direction}_materiality_30d" for direction in DIRECTIONS],
    *[f"single_news_{event_type}_score_30d" for event_type in EVENT_TYPES],
    "single_news_conflict_count_30d",
    "single_news_conflict_intensity_30d",
    "single_news_recency_1d_count",
    "single_news_recency_3d_count",
    "single_news_recency_7d_count",
    "single_news_recency_30d_count",
    "single_news_evidence_quality_score_30d",
    "single_news_action_watch_count_30d",
    "single_news_action_verify_count_30d",
    "single_news_top_materiality_30d",
    "single_news_time_safe_count_30d",
    "single_news_time_approx_count_30d",
]


@dataclass(frozen=True)
class SingleStockNewsEvent:
    event_title: str
    available_at: pd.Timestamp
    entity_scope: str
    event_type: str
    direction_hint: str
    materiality_score: float
    evidence_level: str
    conflict_intensity: float
    actionability: str
    time_safety: str


def build_single_stock_news_query_plan(
    symbol: str,
    name: str,
    *,
    peers: list[str] | None = None,
    sector_keywords: list[str] | None = None,
    as_of_time: str | pd.Timestamp | None = None,
    max_results: int = 8,
) -> list[dict[str, Any]]:
    as_of = pd.Timestamp(as_of_time).isoformat() if as_of_time is not None else None
    peer_text = " ".join((peers or [])[:5])
    sector_text = " ".join((sector_keywords or [])[:5])
    queries = [
        ("self", f"{symbol} {name} 公告 业绩 诉讼 减持"),
        ("self", f"{symbol} {name} 订单 合同 产能 项目"),
        ("peer", f"{name} 同行业 同类公司 {peer_text} 业绩 风险"),
        ("sector", f"{name} 所属行业 {sector_text} 政策 价格 供需"),
        ("upstream", f"{name} 上游 原材料 价格 供应"),
        ("downstream", f"{name} 下游 客户 需求 库存"),
        ("macro", f"{name} A股 市场风格 流动性 风险偏好"),
    ]
    return [
        {
            "entity_scope": scope,
            "query": " ".join(query.split()),
            "as_of_time": as_of,
            "max_results": max(1, min(int(max_results), 20)),
            "policy": "filter available_at <= as_of_time before vectorization",
        }
        for scope, query in queries
    ]


def vectorize_single_stock_news(
    events: list[dict[str, Any]],
    as_of_time: str | pd.Timestamp,
    *,
    lookback_days: int = 30,
) -> dict[str, float]:
    cutoff = pd.Timestamp(as_of_time)
    start = cutoff - pd.Timedelta(days=lookback_days)
    normalized = [normalize_single_stock_news_event(event) for event in events]
    usable = [event for event in normalized if event is not None and start <= event.available_at <= cutoff]
    return _vectorize_events(_dedupe(usable), cutoff)


def normalize_single_stock_news_event(raw: dict[str, Any]) -> SingleStockNewsEvent | None:
    title = str(raw.get("event_title") or raw.get("title") or raw.get("公告标题") or raw.get("新闻标题") or "").strip()
    if not title:
        return None
    available_at = _parse_available_at(raw.get("available_at") or raw.get("datetime") or raw.get("date") or raw.get("发布时间"))
    entity_scope = _choice(raw.get("entity_scope"), ENTITY_SCOPES, "self")
    event_type = _choice(raw.get("event_type"), EVENT_TYPES, _infer_event_type(title))
    direction = _choice(raw.get("direction_hint"), DIRECTIONS, _infer_direction(title))
    evidence = _choice(raw.get("evidence_level"), EVIDENCE_LEVELS, "public_aggregator")
    materiality = _clip(raw.get("materiality_score"), 0, 10, _default_materiality(event_type, evidence))
    conflict = _clip(raw.get("conflict_intensity") or raw.get("conflict_score"), 0, 10, 0)
    actionability = _choice(raw.get("actionability"), ("research_only", "watch", "verify", "ignore"), "research_only")
    time_safety = _choice(raw.get("time_safety"), ("strict", "approximate"), "strict")
    return SingleStockNewsEvent(title, available_at, entity_scope, event_type, direction, materiality, evidence, conflict, actionability, time_safety)


def _vectorize_events(events: list[SingleStockNewsEvent], cutoff: pd.Timestamp) -> dict[str, float]:
    vector = {dimension: 0.0 for dimension in SINGLE_STOCK_NEWS_VECTOR_DIMENSIONS}
    if not events:
        return vector

    vector["single_news_count_30d"] = float(len(events))
    vector["single_news_official_count_30d"] = _count(events, evidence_level="official_disclosure")
    vector["single_news_public_count_30d"] = _count(events, evidence_level="public_aggregator")
    vector["single_news_media_count_30d"] = _count(events, evidence_level="media_report")
    vector["single_news_model_inferred_count_30d"] = _count(events, evidence_level="model_inferred")
    vector["single_news_manual_review_count_30d"] = _count(events, evidence_level="manual_review_required")

    for scope in ENTITY_SCOPES:
        scoped = [event for event in events if event.entity_scope == scope]
        vector[f"single_news_{scope}_count_30d"] = float(len(scoped))
        vector[f"single_news_{scope}_materiality_30d"] = round(sum(event.materiality_score for event in scoped), 4)

    for direction in DIRECTIONS:
        directed = [event for event in events if event.direction_hint == direction]
        vector[f"single_news_{direction}_count_30d"] = float(len(directed))
        vector[f"single_news_{direction}_materiality_30d"] = round(sum(event.materiality_score for event in directed), 4)

    for event_type in EVENT_TYPES:
        typed = [event for event in events if event.event_type == event_type]
        vector[f"single_news_{event_type}_score_30d"] = round(sum(_signed_materiality(event) for event in typed), 4)

    vector["single_news_conflict_count_30d"] = float(sum(event.conflict_intensity > 0 for event in events))
    vector["single_news_conflict_intensity_30d"] = round(sum(event.conflict_intensity for event in events), 4)
    vector["single_news_recency_1d_count"] = _recency_count(events, cutoff, 1)
    vector["single_news_recency_3d_count"] = _recency_count(events, cutoff, 3)
    vector["single_news_recency_7d_count"] = _recency_count(events, cutoff, 7)
    vector["single_news_recency_30d_count"] = _recency_count(events, cutoff, 30)
    vector["single_news_evidence_quality_score_30d"] = round(_evidence_quality(events), 4)
    vector["single_news_action_watch_count_30d"] = _count(events, actionability="watch")
    vector["single_news_action_verify_count_30d"] = _count(events, actionability="verify")
    vector["single_news_top_materiality_30d"] = round(max(event.materiality_score for event in events), 4)
    vector["single_news_time_safe_count_30d"] = _count(events, time_safety="strict")
    vector["single_news_time_approx_count_30d"] = _count(events, time_safety="approximate")
    return vector


def _parse_available_at(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.Timestamp("1900-01-01")
    if isinstance(value, str) and len(value.strip()) <= 10:
        return pd.Timestamp(ts.date()) + pd.Timedelta(days=1)
    return pd.Timestamp(ts)


def _choice(value: Any, allowed: tuple[str, ...], default: str) -> str:
    text = str(value or "").strip()
    return text if text in allowed else default


def _clip(value: Any, low: float, high: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _infer_event_type(title: str) -> str:
    rules = [
        ("earnings", ("业绩", "预增", "预减", "利润", "营收")),
        ("guidance", ("指引", "预告")),
        ("regulation", ("监管", "处罚", "问询")),
        ("litigation", ("诉讼", "仲裁", "立案")),
        ("shareholder", ("减持", "增持", "回购", "质押")),
        ("contract", ("订单", "中标", "合同")),
        ("capacity", ("产能", "扩产", "投产", "停产")),
        ("price", ("涨价", "降价", "价格")),
        ("supply_demand", ("供需", "库存", "开工率")),
        ("safety", ("事故", "安全", "环保")),
        ("policy", ("政策", "补贴", "关税")),
    ]
    return next((event_type for event_type, words in rules if any(word in title for word in words)), "sentiment")


def _infer_direction(title: str) -> str:
    if any(word in title for word in ("预增", "中标", "合同", "增持", "回购", "涨价", "突破")):
        return "positive"
    if any(word in title for word in ("预减", "减持", "诉讼", "处罚", "停产", "事故", "问询")):
        return "negative"
    return "unknown"


def _default_materiality(event_type: str, evidence_level: str) -> float:
    base = {
        "regulation": 8,
        "litigation": 8,
        "safety": 8,
        "earnings": 7,
        "guidance": 7,
        "shareholder": 7,
        "contract": 6,
        "capacity": 6,
        "policy": 6,
        "price": 5,
        "supply_demand": 5,
        "sentiment": 2,
    }.get(event_type, 2)
    return min(10, base + (1 if evidence_level == "official_disclosure" else 0))


def _signed_materiality(event: SingleStockNewsEvent) -> float:
    if event.direction_hint == "negative":
        return -event.materiality_score
    if event.direction_hint == "mixed":
        return 0.0
    return event.materiality_score


def _count(events: list[SingleStockNewsEvent], **attrs: str) -> float:
    return float(sum(all(getattr(event, key) == value for key, value in attrs.items()) for event in events))


def _recency_count(events: list[SingleStockNewsEvent], cutoff: pd.Timestamp, days: int) -> float:
    start = cutoff - pd.Timedelta(days=days)
    return float(sum(start <= event.available_at <= cutoff for event in events))


def _evidence_quality(events: list[SingleStockNewsEvent]) -> float:
    weights = {
        "official_disclosure": 1.0,
        "public_aggregator": 0.7,
        "media_report": 0.6,
        "model_inferred": 0.35,
        "manual_review_required": 0.2,
    }
    return sum(weights.get(event.evidence_level, 0.2) for event in events) / len(events) * 10


def _dedupe(events: list[SingleStockNewsEvent]) -> list[SingleStockNewsEvent]:
    seen: set[tuple[str, str]] = set()
    output = []
    for event in sorted(events, key=lambda item: (item.materiality_score, item.available_at), reverse=True):
        key = (event.available_at.date().isoformat(), "".join(event.event_title.split())[:40])
        if key in seen:
            continue
        seen.add(key)
        output.append(event)
    return output
