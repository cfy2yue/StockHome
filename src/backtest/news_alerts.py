from __future__ import annotations

from typing import Any

import pandas as pd


def add_news_alert_features(decisions: pd.DataFrame) -> pd.DataFrame:
    """Add time-safe, cross-sectional news alert features to decision rows."""
    if decisions.empty:
        return decisions.copy()

    out = decisions.copy()
    news_count = _num(out, "news_count_30d")
    attention_rank = news_count.groupby(out["date"]).rank(pct=True, method="average").fillna(0)
    attention_spike = ((attention_rank - 0.80).clip(lower=0) * 20).where(news_count >= 5, 0)
    negative_recency = (-_num(out, "news_recency_weighted_materiality_30d")).clip(lower=0)
    peer_news_count = _num(out, "peer_group_news_count_avg")
    peer_news_risk = _num(out, "peer_group_news_risk_avg")
    peer_news_opportunity = _num(out, "peer_group_news_opportunity_avg")
    peer_attention_gap = news_count - peer_news_count
    peer_silent_active = ((news_count == 0) & (peer_news_count >= 1)).astype(float)
    relative_attention = peer_attention_gap.clip(lower=0, upper=10)
    policy_background = (
        _num(out, "news_industry_policy_count_30d") * 0.50
        + _num(out, "news_price_policy_score_30d").abs() * 0.30
        + _num(out, "news_macro_market_score_30d").abs() * 0.20
    )
    self_intensity = (
        _num(out, "news_company_count_30d") * 0.50
        + _num(out, "news_top_event_materiality_30d") * 0.30
        + _num(out, "news_evidence_quality_score_30d") * 0.20
    )

    out["news_attention_rank_30d"] = attention_rank.round(4)
    out["news_attention_spike_score_30d"] = attention_spike.round(4)
    out["news_negative_recency_score_30d"] = negative_recency.round(4)
    out["news_self_intensity_score_30d"] = self_intensity.round(4)
    out["news_policy_background_score_30d"] = policy_background.round(4)
    out["news_peer_attention_gap_30d"] = peer_attention_gap.round(4)
    out["news_relative_attention_score_30d"] = relative_attention.round(4)
    out["news_peer_silent_while_group_active_30d"] = peer_silent_active
    out["news_sector_background_score_30d"] = (peer_news_opportunity - peer_news_risk).round(4)
    out["news_region_background_score_30d"] = _region_background(out).round(4)

    risk = _num(out, "news_risk_event_score_30d")
    conflict = _num(out, "news_conflict_intensity_30d")
    opportunity = _num(out, "news_opportunity_event_score_30d")
    evidence_quality = _num(out, "news_evidence_quality_score_30d")

    warning = risk * 0.45 + conflict * 0.25 + attention_spike * 0.08 + peer_news_risk * 0.10 + negative_recency * 0.15 + peer_silent_active * 0.15
    opportunity_alert = (
        opportunity * 0.28
        + attention_spike * 0.30
        + relative_attention * 0.12
        + policy_background * 0.05
        + evidence_quality * 0.05
        - warning * 0.25
    )

    out["news_warning_score_30d"] = warning.clip(lower=0).round(4)
    out["news_opportunity_alert_score_30d"] = opportunity_alert.clip(lower=0).round(4)
    out["news_alert_label"] = out.apply(_label, axis=1)
    out["news_alert_reason"] = out.apply(_reason, axis=1)
    return out


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def _label(row: pd.Series) -> str:
    if float(row.get("news_warning_score_30d") or 0) >= 1.0:
        return "风险预警"
    if float(row.get("news_opportunity_alert_score_30d") or 0) > 0:
        return "机会提醒"
    if float(row.get("news_attention_spike_score_30d") or 0) > 0:
        return "关注度异常"
    return "无明显新闻提醒"


def _reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if _value(row, "news_risk_event_score_30d") > 0:
        reasons.append("标题含减持/处罚/诉讼/停产等风险关键词")
    if _value(row, "news_conflict_intensity_30d") > 0:
        reasons.append("正负新闻同时出现，信息冲突升高")
    if _value(row, "news_negative_recency_score_30d") > 0:
        reasons.append("近期新闻净重大性偏负")
    if _value(row, "peer_group_news_risk_avg") > 0:
        reasons.append("同组股票存在新闻风险溢出")
    if _value(row, "news_attention_spike_score_30d") > 0:
        reasons.append("30日新闻量处于当日候选池前20%")
    if _value(row, "news_relative_attention_score_30d") > 0:
        reasons.append("自身新闻曝光高于同组平均")
    if _value(row, "news_peer_silent_while_group_active_30d") > 0:
        reasons.append("同组有新闻但自身缺少同步信息")
    if _value(row, "news_policy_background_score_30d") > 0:
        reasons.append("政策/宏观/价格背景被新闻提及")
    if _value(row, "news_opportunity_event_score_30d") > 0:
        reasons.append("标题含订单/中标/业绩预增/产品技术等机会关键词")
    return "；".join(reasons) if reasons else "无明显新闻触发"


def _value(row: pd.Series, key: str) -> float:
    value: Any = row.get(key, 0)
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _region_background(df: pd.DataFrame) -> pd.Series:
    if "region_group" not in df:
        return pd.Series(0.0, index=df.index)
    risk = _num(df, "news_risk_event_score_30d")
    opportunity = _num(df, "news_opportunity_event_score_30d")
    net = opportunity - risk
    group_cols = [df["date"], df["region_group"]]
    group_sum = net.groupby(group_cols).transform("sum")
    group_count = df.groupby(["date", "region_group"])["date"].transform("count")
    denom = (group_count - 1).replace(0, pd.NA)
    return ((group_sum - net) / denom).fillna(0.0)
