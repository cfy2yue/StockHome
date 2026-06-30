from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RuleTrigger:
    strategy_id: str
    name: str
    effect: str
    formula: str
    source_book: str
    source_chapter: str
    page_range: str
    extraction_method: str
    source_confidence: str


def evaluate_book_rules(history: pd.DataFrame, stock: dict[str, Any]) -> list[RuleTrigger]:
    if len(history) < 20:
        return []
    latest = history.iloc[-1]
    triggers: list[RuleTrigger] = []
    sector_group = str(stock.get("sector_group", ""))
    is_star = str(stock.get("board", "")).lower() == "star" or str(stock.get("code", "")).startswith("688")

    low20 = history["low"].tail(20).min()
    if _number(latest.get("close")) < low20 * (0.92 if is_star else 0.95):
        triggers.append(
            RuleTrigger(
                strategy_id="PPS-M-002",
                name="鳄鱼原则：错误扩大时优先控制风险",
                effect="cap_score_4",
                formula="close < rolling_low_20 * (0.92 if star else 0.95)",
                source_book="专业投机原理",
                source_chapter="第2章",
                page_range="OCR_PAGE 0032-0037，书内页码线索25-30，页码需人工复核",
                extraction_method="full_ocr_txt_deep_dive",
                source_confidence="high",
            )
        )

    if len(history) >= 60:
        drawdown60 = (history["close"].iloc[-1] / history["close"].tail(60).max() - 1) * 100
        threshold = -30 if is_star else -25
        if drawdown60 <= threshold:
            triggers.append(
                RuleTrigger(
                    strategy_id="PPS-M-003",
                    name="保存资本优先：大回撤降低研究优先级",
                    effect="minus_2",
                    formula="drawdown_60d <= (-30 if star else -25)",
                    source_book="专业投机原理",
                    source_chapter="第3章、第18章",
                    page_range="OCR_PAGE 0038-0044、0275-0281，书内页码线索31-37、268-274，页码需人工复核",
                    extraction_method="full_ocr_txt_deep_dive",
                    source_confidence="high",
                )
            )

    if _truthy(latest.get("ma200")):
        if latest["close"] > latest["ma200"] and _number(latest.get("ma200_slope20")) > 0:
            triggers.append(
                RuleTrigger(
                    strategy_id="PPS-Q-017",
                    name="200日均线辅助趋势过滤",
                    effect="plus_1",
                    formula="close > ma200 and ma200_slope20 > 0",
                    source_book="专业投机原理",
                    source_chapter="第8章、第27章",
                    page_range="OCR_PAGE 0098-0118、0364-0385，书内页码线索91-111、357-378，策略ID=PPS-Q-017，提取方式=系统通读OCR；页码需人工复核。",
                    extraction_method="full_ocr_txt_deep_dive",
                    source_confidence="high",
                )
            )
        elif latest["close"] < latest["ma200"] and _number(latest.get("ma200_slope20")) < 0:
            triggers.append(
                RuleTrigger(
                    strategy_id="PPS-Q-017",
                    name="200日均线下方且均线走弱",
                    effect="counter_risk",
                    formula="close < ma200 and ma200_slope20 < 0",
                    source_book="专业投机原理",
                    source_chapter="第8章、第27章",
                    page_range="OCR_PAGE 0098-0118、0364-0385，书内页码线索91-111、357-378，策略ID=PPS-Q-017，提取方式=系统通读OCR；页码需人工复核。",
                    extraction_method="full_ocr_txt_deep_dive",
                    source_confidence="high",
                )
            )

    if _truthy(latest.get("ma20")) and _truthy(latest.get("ma60")):
        trend_ok = latest["close"] > latest["ma20"] and latest["close"] > latest["ma60"] and latest["ma20"] > latest["ma60"]
        if trend_ok:
            triggers.append(
                RuleTrigger(
                    strategy_id="DOW-B-017",
                    name="个股趋势不能长期脱离市场与行业环境",
                    effect="plus_1",
                    formula="close > ma20 and close > ma60 and ma20 > ma60",
                    source_book="道氏理论",
                    source_chapter="第17章、第19章",
                    page_range="OCR_PAGE 185-186；第19章/OCR_PAGE 209；提取方式：OCR正文通读；页码需人工复核",
                    extraction_method="full_ocr_txt_deep_dive",
                    source_confidence="high",
                )
            )

    if len(history) >= 22:
        prev_high20 = history["high"].iloc[-21:-1].max()
        prev_low20 = history["low"].iloc[-21:-1].min()
        if latest["high"] > prev_high20 and latest["close"] < prev_high20:
            triggers.append(
                RuleTrigger(
                    strategy_id="PPS-Q-009",
                    name="2B 法则：创新高后不能延续并跌回前高下方",
                    effect="minus_1",
                    formula="high > previous_20d_high and close < previous_20d_high",
                    source_book="专业投机原理",
                    source_chapter="第7章、第27章",
                    page_range="OCR_PAGE 0090-0093、0364-0385，书内页码线索83-86、357-378，策略ID=PPS-Q-009，提取方式=系统通读OCR；页码需人工复核。",
                    extraction_method="full_ocr_txt_deep_dive",
                    source_confidence="high",
                )
            )
        if latest["low"] < prev_low20 and latest["close"] > prev_low20:
            triggers.append(
                RuleTrigger(
                    strategy_id="PPS-Q-009",
                    name="2B 法则：创新低后不能延续并收回前低上方",
                    effect="plus_1",
                    formula="low < previous_20d_low and close > previous_20d_low",
                    source_book="专业投机原理",
                    source_chapter="第7章、第27章",
                    page_range="OCR_PAGE 0090-0093、0364-0385，书内页码线索83-86、357-378，策略ID=PPS-Q-009，提取方式=系统通读OCR；页码需人工复核。",
                    extraction_method="full_ocr_txt_deep_dive",
                    source_confidence="high",
                )
            )

    upper_shadow = latest["high"] - max(latest["open"], latest["close"])
    real_body = abs(latest["close"] - latest["open"])
    volume_ratio = _number(latest.get("volume_ratio20"))
    if real_body > 0 and upper_shadow > real_body * 2 and volume_ratio > 1.5:
        triggers.append(
            RuleTrigger(
                strategy_id="CANDLE_MACRO_002",
                name="反转形态更应视为趋势变化警报",
                effect="minus_1",
                formula="upper_shadow > real_body * 2 and volume_ratio20 > 1.5",
                source_book="日本蜡烛图技术",
                source_chapter="第四章反转形态",
                page_range="OCR_PAGE_0048-0050；书内页码约28-30；页码需人工复核",
                extraction_method="full_ocr_txt",
                source_confidence="high",
            )
        )

    if len(history) >= 5 and _consecutive_down(history, 4) and _number(latest.get("close")) < _number(latest.get("ma20")):
        triggers.append(
            RuleTrigger(
                strategy_id="PPS-Q-023",
                name="四日法则：连续同向后关注反转风险",
                effect="counter_risk",
                formula="consecutive_down_days >= 4 and close < ma20",
                source_book="专业投机原理",
                source_chapter="第27章",
                page_range="OCR_PAGE 0364-0385，书内页码线索357-378，页码需人工复核",
                extraction_method="full_ocr_txt_deep_dive",
                source_confidence="medium",
            )
        )

    if len(history) >= 60:
        recent_high = history["high"].tail(60).max()
        recent_low = history["low"].tail(60).min()
        if recent_high > recent_low:
            retrace = (recent_high - latest["close"]) / (recent_high - recent_low)
            if 0.33 <= retrace <= 0.66:
                triggers.append(
                    RuleTrigger(
                        strategy_id="DOW-B-004",
                        name="次级反应：回撤处于 33%-66% 观察区",
                        effect="watch",
                        formula="0.33 <= (recent_60d_high - close) / (recent_60d_high - recent_60d_low) <= 0.66",
                        source_book="道氏理论",
                        source_chapter="第2章、第10章、第13章",
                        page_range="OCR_PAGE 20-21；第10章/OCR_PAGE 91-104；第13章/OCR_PAGE 147；提取方式：OCR正文通读；页码需人工复核",
                        extraction_method="full_ocr_txt_deep_dive",
                        source_confidence="high",
                    )
                )

    if sector_group == "nonferrous_materials" and len(history) >= 20:
        return20 = _number(latest.get("return_20d"))
        if return20 > 20 and _number(latest.get("rsi14")) > 75:
            triggers.append(
                RuleTrigger(
                    strategy_id="PPS-M-003",
                    name="保存资本优先：周期股急涨后提高反证权重",
                    effect="counter_risk",
                    formula="sector_group == nonferrous_materials and return_20d > 20 and rsi14 > 75",
                    source_book="专业投机原理",
                    source_chapter="第3章、第18章",
                    page_range="OCR_PAGE 0038-0044、0275-0281，书内页码线索31-37、268-274，页码需人工复核",
                    extraction_method="full_ocr_txt_deep_dive",
                    source_confidence="high",
                )
            )
    return triggers


def _number(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _truthy(value: Any) -> bool:
    return not math.isnan(_number(value))


def _consecutive_down(history: pd.DataFrame, days: int) -> bool:
    tail = history.tail(days)
    return len(tail) == days and bool((tail["close"].diff().dropna() < 0).all()) and tail["close"].iloc[-1] < tail["close"].iloc[0]
