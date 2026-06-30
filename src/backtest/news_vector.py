from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from .news_filter import NewsEvent, normalize_news_event


NEWS_VECTOR_DIMENSIONS = [
    "news_count_30d",
    "news_official_count_30d",
    "news_public_count_30d",
    "news_company_count_30d",
    "news_industry_policy_count_30d",
    "news_positive_count_30d",
    "news_negative_count_30d",
    "news_uncertain_count_30d",
    "news_materiality_sum_30d",
    "news_materiality_max_30d",
    "news_positive_materiality_30d",
    "news_negative_materiality_30d",
    "news_official_materiality_30d",
    "news_public_materiality_30d",
    "news_net_materiality_30d",
    "news_recency_weighted_materiality_30d",
    "news_earnings_score_30d",
    "news_order_score_30d",
    "news_capacity_score_30d",
    "news_holding_change_score_30d",
    "news_legal_regulatory_score_30d",
    "news_financing_score_30d",
    "news_price_policy_score_30d",
    "news_tech_product_score_30d",
    "news_macro_market_score_30d",
    "news_supply_chain_score_30d",
    "news_risk_event_score_30d",
    "news_opportunity_event_score_30d",
    "news_evidence_quality_score_30d",
    "news_conflict_intensity_30d",
    "news_event_type_diversity_30d",
    "news_top_event_materiality_30d",
]


EVENT_BUCKETS = {
    "earnings": ["业绩", "预增", "预减", "利润", "营收", "年报", "季报"],
    "order": ["订单", "中标", "合同", "客户"],
    "capacity": ["产能", "扩产", "投产", "停产"],
    "holding_change": ["减持", "增持", "回购"],
    "legal_regulatory": ["诉讼", "监管", "处罚", "立案", "问询"],
    "financing": ["融资", "定增", "可转债", "质押"],
    "price_policy": ["价格", "涨价", "降价", "政策", "补贴", "关税"],
    "tech_product": ["技术", "产品", "突破", "研发", "专利"],
    "macro_market": ["市场", "指数", "宏观", "利率", "汇率"],
    "supply_chain": ["供应", "原料", "上游", "下游", "库存"],
}


def vectorize_news_events(
    events: list[dict[str, Any]],
    decision_date: str | pd.Timestamp,
    lookback_days: int = 30,
) -> dict[str, float]:
    return NewsVectorizer(events, lookback_days=lookback_days).vectorize(decision_date)


class NewsVectorizer:
    def __init__(self, events: list[dict[str, Any]], lookback_days: int = 30) -> None:
        self.lookback_days = lookback_days
        normalized = [normalize_news_event(event) for event in events]
        self.events = sorted([event for event in normalized if event is not None], key=lambda event: event.available_at)

    def vectorize(self, decision_date: str | pd.Timestamp) -> dict[str, float]:
        cutoff = pd.Timestamp(decision_date).replace(hour=15, minute=0, second=0)
        start = cutoff - pd.Timedelta(days=self.lookback_days)
        usable = [event for event in self.events if start <= event.available_at <= cutoff]
        return vectorize_normalized_news(_dedupe_normalized(usable), cutoff)


def _legacy_vectorize_news_events(
    events: list[dict[str, Any]],
    decision_date: str | pd.Timestamp,
    lookback_days: int = 30,
) -> dict[str, float]:
    cutoff = pd.Timestamp(decision_date).replace(hour=15, minute=0, second=0)
    start = cutoff - pd.Timedelta(days=lookback_days)
    usable = [event for event in NewsVectorizer(events, lookback_days=lookback_days).events if start <= event.available_at <= cutoff]
    return vectorize_normalized_news(usable, cutoff)


def _dedupe_normalized(events: list[NewsEvent]) -> list[NewsEvent]:
    seen = set()
    out = []
    for event in events:
        if event.dedupe_key in seen:
            continue
        seen.add(event.dedupe_key)
        out.append(event)
    return sorted(out, key=lambda event: (event.materiality_score, event.available_at), reverse=True)


def vectorize_normalized_news(events: list[NewsEvent], cutoff: pd.Timestamp) -> dict[str, float]:
    vector = {dimension: 0.0 for dimension in NEWS_VECTOR_DIMENSIONS}
    if not events:
        return vector

    vector["news_count_30d"] = float(len(events))
    vector["news_official_count_30d"] = float(sum(event.source_type == "官方公告" for event in events))
    vector["news_public_count_30d"] = float(sum(event.source_type == "公开新闻" for event in events))
    vector["news_company_count_30d"] = float(sum(event.entity_scope == "公司" for event in events))
    vector["news_industry_policy_count_30d"] = float(sum(event.entity_scope == "行业/政策" for event in events))
    vector["news_positive_count_30d"] = float(sum(event.direction_hint == "正面" for event in events))
    vector["news_negative_count_30d"] = float(sum(event.direction_hint == "负面" for event in events))
    vector["news_uncertain_count_30d"] = float(sum(event.direction_hint == "不确定" for event in events))

    materialities = [float(event.materiality_score) for event in events]
    vector["news_materiality_sum_30d"] = round(sum(materialities), 4)
    vector["news_materiality_max_30d"] = round(max(materialities), 4)
    vector["news_top_event_materiality_30d"] = round(max(materialities), 4)
    vector["news_positive_materiality_30d"] = round(_direction_sum(events, "正面"), 4)
    vector["news_negative_materiality_30d"] = round(_direction_sum(events, "负面"), 4)
    vector["news_official_materiality_30d"] = round(sum(event.materiality_score for event in events if event.source_type == "官方公告"), 4)
    vector["news_public_materiality_30d"] = round(sum(event.materiality_score for event in events if event.source_type == "公开新闻"), 4)
    vector["news_net_materiality_30d"] = round(vector["news_positive_materiality_30d"] - vector["news_negative_materiality_30d"], 4)
    vector["news_recency_weighted_materiality_30d"] = round(_recency_weighted_sum(events, cutoff), 4)

    for bucket, keywords in EVENT_BUCKETS.items():
        dimension = f"news_{bucket}_score_30d"
        vector[dimension] = round(_bucket_score(events, keywords), 4)

    vector["news_risk_event_score_30d"] = round(
        vector["news_negative_materiality_30d"] + vector["news_legal_regulatory_score_30d"] + max(0.0, -vector["news_capacity_score_30d"]),
        4,
    )
    vector["news_opportunity_event_score_30d"] = round(
        vector["news_positive_materiality_30d"] + vector["news_order_score_30d"] + vector["news_tech_product_score_30d"],
        4,
    )
    vector["news_evidence_quality_score_30d"] = round(_evidence_quality(events), 4)
    vector["news_conflict_intensity_30d"] = round(min(vector["news_positive_materiality_30d"], vector["news_negative_materiality_30d"]), 4)
    vector["news_event_type_diversity_30d"] = float(len(Counter(event.event_type for event in events)))
    return vector


def _direction_sum(events: list[NewsEvent], direction: str) -> float:
    return float(sum(event.materiality_score for event in events if event.direction_hint == direction))


def _recency_weighted_sum(events: list[NewsEvent], cutoff: pd.Timestamp) -> float:
    total = 0.0
    for event in events:
        days = max(0, (cutoff - event.available_at).days)
        weight = 1 / (1 + days / 7)
        direction = -1 if event.direction_hint == "负面" else 1
        total += event.materiality_score * weight * direction
    return total


def _bucket_score(events: list[NewsEvent], keywords: list[str]) -> float:
    score = 0.0
    for event in events:
        if any(keyword in event.title or keyword in event.event_type for keyword in keywords):
            direction = -1 if event.direction_hint == "负面" else 1
            score += event.materiality_score * direction
    return score


def _evidence_quality(events: list[NewsEvent]) -> float:
    if not events:
        return 0.0
    score = 0.0
    for event in events:
        score += 1.0 if event.evidence_level == "official_disclosure" else 0.5
    return score / len(events) * 10
