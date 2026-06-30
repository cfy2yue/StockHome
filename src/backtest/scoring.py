from __future__ import annotations

from typing import Any

import pandas as pd

from .book_rules import RuleTrigger


DEFAULT_WEIGHTS = {
    "trend_structure": 0.20,
    "fundamental_quality": 0.15,
    "financial_safety": 0.15,
    "valuation_pressure": 0.10,
    "market_regime": 0.10,
    "book_strategy_match": 0.15,
    "counterevidence_risk": 0.10,
    "data_completeness": 0.05,
}

RATINGS = ("继续深挖", "放入观察", "暂时剔除", "信息不足")


def score_decision(
    latest: pd.Series,
    financial_records: list[dict[str, Any]],
    triggers: list[RuleTrigger],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    weights = normalize_weights(weights or DEFAULT_WEIGHTS)
    scores = {
        "trend_structure": _trend_score(latest),
        "fundamental_quality": _financial_score(financial_records),
        "financial_safety": _safety_score(financial_records),
        "valuation_pressure": 5.0,
        "market_regime": 5.0,
        "book_strategy_match": _book_score(triggers),
        "counterevidence_risk": _counter_score(latest, triggers),
        "data_completeness": _completeness_score(latest, financial_records),
    }
    total = round(sum(scores[k] * weights.get(k, 0) for k in scores), 2)
    notes: list[str] = []
    if any(t.effect == "cap_score_4" for t in triggers):
        total = min(total, 4.0)
        notes.append("触发风险上限规则，总分最高 4 分")
    if sum(1 for t in triggers if t.effect == "plus_1") >= 3:
        total = min(total, 8.0)
        notes.append("多重加分信号触发，总分最高 8 分")
    rating = rating_from_score(total, scores["data_completeness"])
    return {"scores": scores, "total_score": total, "rating": rating, "notes": notes}


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    merged = {**DEFAULT_WEIGHTS, **(weights or {})}
    total = sum(max(0, float(v)) for v in merged.values())
    if total <= 0:
        return DEFAULT_WEIGHTS.copy()
    return {k: max(0, float(v)) / total for k, v in merged.items()}


def rating_from_score(total: float, completeness: float = 10) -> str:
    if completeness <= 2:
        return "信息不足"
    if total >= 8:
        return "继续深挖"
    if total >= 5:
        return "放入观察"
    if total >= 2:
        return "暂时剔除"
    return "信息不足"


def _trend_score(latest: pd.Series) -> float:
    score = 5.0
    close = _value(latest.get("close"))
    if close > _value(latest.get("ma20")) > _value(latest.get("ma60")):
        score += 2
    if close > _value(latest.get("ma120")):
        score += 1
    if _value(latest.get("return_20d")) > 8:
        score += 1
    if _value(latest.get("rsi14")) > 80:
        score -= 1.5
    if close < _value(latest.get("ma60")):
        score -= 2
    return _clip(score)


def _financial_score(records: list[dict[str, Any]]) -> float:
    latest = _latest_financial(records)
    if not latest:
        return 4
    score = 5.0
    pni = _value(latest.get("yoypni") or latest.get("net_profit_yoy"))
    roe = _value(latest.get("roe"))
    if pni > 0:
        score += 1.5
    if pni < -20:
        score -= 1.5
    if pni < -40:
        score -= 1.5
    if roe > 10:
        score += 1
    if roe < 5:
        score -= 1
    return _clip(score)


def _safety_score(records: list[dict[str, Any]]) -> float:
    latest = _latest_financial(records)
    if not latest:
        return 4
    score = 6.0
    debt = _value(latest.get("debt_to_assets") or latest.get("asset_liability_ratio"))
    current = _value(latest.get("current_ratio"))
    if debt > 70:
        score -= 2
    if current and current < 1:
        score -= 1.5
    return _clip(score)


def _book_score(triggers: list[RuleTrigger]) -> float:
    score = 5.0
    for trigger in triggers:
        if trigger.effect == "plus_1":
            score += 1
        elif trigger.effect in {"minus_1", "counter_risk"}:
            score -= 1
        elif trigger.effect == "minus_2":
            score -= 2
        elif trigger.effect == "watch":
            score -= 0.2
    return _clip(score)


def _counter_score(latest: pd.Series, triggers: list[RuleTrigger]) -> float:
    score = 8.0
    if _value(latest.get("macd_hist")) < 0:
        score -= 1
    if _value(latest.get("rsi14")) > 80:
        score -= 1
    score -= sum(1 for t in triggers if t.effect in {"minus_1", "minus_2", "counter_risk", "cap_score_4"})
    return _clip(score)


def _completeness_score(latest: pd.Series, records: list[dict[str, Any]]) -> float:
    required = ["ma20", "ma60", "rsi14", "atr20", "macd_hist"]
    present = sum(1 for col in required if pd.notna(latest.get(col)))
    score = present / len(required) * 8
    if records:
        score += 2
    return _clip(score)


def _latest_financial(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    return records[-1] if records else None


def _value(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clip(value: float) -> float:
    return round(max(0.0, min(10.0, value)), 2)

