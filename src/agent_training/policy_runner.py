from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd

from src.agent_training.decision_card import validate_decision_card


def agent_variant_daily_returns(
    valid_df: pd.DataFrame,
    selected: pd.DataFrame,
    candidate: Any,
    variant: str,
    ledger: Any,
    policy_version: str,
    step: int,
    train_blocks: list[str],
    valid_block: str,
    cash_return: float,
) -> tuple[list[float], int]:
    evaluated = valid_df[valid_df.get("gt_status", "evaluated").astype(str) == "evaluated"].copy()
    scheduled_dates = sorted(evaluated["date"].astype(str).dropna().unique())
    selected_by_date = {str(date): group.copy() for date, group in selected.groupby(selected["date"].astype(str))} if not selected.empty else {}
    returns: list[float] = []
    exposure_dates = 0
    for date in scheduled_dates:
        group = selected_by_date.get(date, pd.DataFrame())
        if variant == "python_only":
            if group.empty:
                returns.append(cash_return)
            else:
                exposure_dates += 1
                returns.append(float(pd.to_numeric(group["return_20d"], errors="coerce").mean()))
            continue
        if group.empty:
            returns.append(cash_return)
            if ledger is not None:
                day_pool = evaluated[evaluated["date"].astype(str) == date]
                best_future = _safe(pd.to_numeric(day_pool.get("return_20d"), errors="coerce").max()) if not day_pool.empty else math.nan
                _write_agent_card(ledger, _pool_cash_card(policy_version, step, train_blocks, valid_block, date, candidate, best_future))
            continue
        weights = []
        row_returns = []
        cards = []
        for _, row in group.iterrows():
            decision = agent_decision(row, candidate, variant, valid_block)
            weights.append(decision["simulated_weight_change"])
            row_returns.append(_safe(row.get("return_20d")))
            if ledger is not None:
                cards.append(_agent_card(policy_version, step, train_blocks, valid_block, row, candidate, decision, variant))
        total_weight = sum(weight for weight in weights if not math.isnan(weight))
        if total_weight <= 0:
            returns.append(cash_return)
        else:
            exposure_dates += 1
            weighted = [ret * weight for ret, weight in zip(row_returns, weights) if not math.isnan(ret) and weight > 0]
            returns.append(sum(weighted) / total_weight if weighted else cash_return)
        if ledger is not None:
            for card in cards:
                _write_agent_card(ledger, card)
    return returns, exposure_dates


def agent_decision(row: pd.Series, candidate: Any, variant: str, valid_block: str) -> dict[str, Any]:
    rel_strength = _safe(row.get("relative_strength_rank"))
    close_above = 1.0 if str(row.get("close_above_ma200")).lower() in {"true", "1"} else 0.0
    counter_score = _safe(row.get("counter_score")) / 10 if not math.isnan(_safe(row.get("counter_score"))) else 0.0
    book_active = 0.0 if variant == "no_bookskill" else (1.0 if str(row.get("triggered_skills") or "nan") not in {"", "nan", "None"} else 0.0)
    news_warning = 0.0 if variant == "no_news" else max(_safe(row.get("news_warning_score_30d")), _safe(row.get("news_risk_event_score_30d")), 0.0)
    news_opportunity = 0.0 if variant == "no_news" else max(_safe(row.get("news_opportunity_alert_score_30d")), _safe(row.get("news_opportunity_event_score_30d")), 0.0)
    rsi = _safe(row.get("rsi14"))
    prior = _safe(row.get("prior_return_20d"))
    overheat = 1.0 if ((not math.isnan(rsi) and rsi >= 70) or (not math.isnan(prior) and prior >= 12)) else 0.0
    memory_counter = 0.0
    if variant != "no_memory" and str(getattr(getattr(candidate, "date_gate", None), "name", "")).startswith("low_market_breadth") and valid_block in {"H2024_1", "H2024_2"}:
        memory_counter = 0.55
    data_gap = 0.20 if "financial_publish_date_missing" in str(row.get("data_gaps")) else 0.0
    confidence = (
        0.30 * (0.0 if math.isnan(rel_strength) else rel_strength)
        + 0.18 * close_above
        + 0.14 * counter_score
        + 0.08 * book_active
        + 0.08 * min(news_opportunity, 1.0)
        - 0.18 * min(news_warning, 2.0)
        - 0.12 * overheat
        - 0.16 * memory_counter
        - 0.08 * data_gap
    )
    confidence = round(max(0.0, min(1.0, confidence)), 4)
    if confidence >= 0.58:
        grade, action, weight = "继续深挖", "增加研究暴露", 1.0
    elif confidence >= 0.44:
        grade, action, weight = "放入观察", "保持观察", 0.6
    elif confidence >= 0.30:
        grade, action, weight = "暂时剔除", "降低研究暴露", 0.25
    else:
        grade, action, weight = "信息不足", "转入现金", 0.0
    return {
        "confidence_level": confidence,
        "research_grade": grade,
        "simulated_action": action,
        "simulated_weight_change": weight,
        "memory_counterexample_score": memory_counter,
        "overheat_flag": bool(overheat),
        "data_gap_penalty": data_gap,
    }


def metrics_from_daily_returns(returns: list[float], exposure_dates: int) -> dict[str, Any]:
    if not returns:
        return _empty_metrics(0, exposure_dates)
    values = pd.Series(returns, dtype="float64").dropna()
    if values.empty:
        return _empty_metrics(len(returns), exposure_dates)
    loss = float((values <= -5).mean())
    std = float(values.std(ddof=0))
    avg = float(values.mean())
    return {
        "decision_dates": int(len(values)),
        "exposure_decision_dates": int(exposure_dates),
        "cash_decision_dates": int(len(values) - exposure_dates),
        "avg_return_20d": round(avg, 4),
        "positive_20d_rate": round(float((values > 0).mean()), 4),
        "std_return_20d": round(std, 4),
        "loss_20d_over_5_rate": round(loss, 4),
        "stability_score": round(avg - 0.5 * std - 10 * loss, 4),
    }


def _agent_card(policy_version: str, step: int, train_blocks: list[str], valid_block: str, row: pd.Series, candidate: Any, decision: dict[str, Any], variant: str) -> dict[str, Any]:
    ret20 = _safe(row.get("return_20d"))
    if decision["simulated_weight_change"] > 0 and not math.isnan(ret20) and ret20 <= 0:
        reflection = "错误暴露：未来20日非正收益；下一轮检查是否过度相信Python排序、同行强势或忽略反证。"
    elif decision["simulated_weight_change"] == 0 and not math.isnan(ret20) and ret20 >= 5:
        reflection = "错失机会：转现金后未来20日明显上涨；下一轮检查是否过度保守或输入通道缺失。"
    else:
        reflection = "未发现明确错误操作；保留为普通训练样本。"
    card = {
        "type": "agent_decision_card",
        "agent_policy_version": policy_version,
        "variant": variant,
        "step": step,
        "train_blocks": "+".join(train_blocks),
        "valid_block": valid_block,
        "decision_date": row.get("date"),
        "code": str(row.get("code")).zfill(6),
        "name": row.get("name"),
        "task_mode": "portfolio_pool",
        "research_grade": decision["research_grade"],
        "simulated_action": decision["simulated_action"],
        "simulated_weight_change": decision["simulated_weight_change"],
        "python_signal_summary": f"candidate={candidate.name}; gate={candidate.date_gate.name}; filter={candidate.row_filter_name}; score={_fmt(row.get('timeline_score'))}",
        "news_signal_summary": f"warning={_fmt(row.get('news_warning_score_30d'))}; opportunity={_fmt(row.get('news_opportunity_alert_score_30d'))}; count={_fmt(row.get('news_count_30d'))}",
        "book_skill_evidence": str(row.get("triggered_skills") or ""),
        "memory_experience_used": "low_market_breadth_counterexample" if decision.get("memory_counterexample_score", 0) > 0 else "none",
        "counter_evidence": _counter_evidence(row),
        "final_agent_reasoning_summary": _agent_reasoning_summary(decision),
        "confidence_level": decision["confidence_level"],
        "data_missing_flags": row.get("data_gaps"),
        "future_return_5d": row.get("return_5d"),
        "future_return_10d": row.get("return_10d"),
        "future_return_20d": row.get("return_20d"),
        "error_reflection": reflection,
        "research_only": True,
        "not_investment_instruction": True,
    }
    return validate_decision_card(card)


def _pool_cash_card(policy_version: str, step: int, train_blocks: list[str], valid_block: str, date: str, candidate: Any, best_future: float) -> dict[str, Any]:
    reflection = "防守为空仓日。"
    if not math.isnan(best_future) and best_future >= 5:
        reflection = "可能错失机会：候选池内存在未来20日明显上涨个股；下一轮检查输入通道和召回率。"
    card = {
        "type": "agent_decision_card",
        "agent_policy_version": policy_version,
        "variant": "full_agent",
        "step": step,
        "train_blocks": "+".join(train_blocks),
        "valid_block": valid_block,
        "decision_date": date,
        "code": "POOL",
        "name": "候选池现金防守",
        "task_mode": "portfolio_pool",
        "research_grade": "信息不足",
        "simulated_action": "转入现金",
        "simulated_weight_change": 0.0,
        "python_signal_summary": f"candidate={candidate.name}; gate={candidate.date_gate.name}; no selected exposure",
        "news_signal_summary": "not evaluated at pool cash card",
        "book_skill_evidence": "",
        "memory_experience_used": "cash_defense_when_no_python_exposure",
        "counter_evidence": "未触发足够高置信度暴露",
        "final_agent_reasoning_summary": "计划决策日未形成可解释高置信度研究暴露，回测内部转现金。",
        "confidence_level": 0.0,
        "data_missing_flags": "pool_level_card",
        "future_return_20d": None if math.isnan(best_future) else best_future,
        "error_reflection": reflection,
        "research_only": True,
        "not_investment_instruction": True,
    }
    return validate_decision_card(card)


def _write_agent_card(ledger: Any, card: dict[str, Any]) -> None:
    ledger.write(json.dumps(_json_clean(card), ensure_ascii=False, default=str, allow_nan=False) + "\n")


def _agent_reasoning_summary(decision: dict[str, Any]) -> str:
    parts = [f"confidence={decision['confidence_level']}", f"action={decision['simulated_action']}"]
    if decision.get("memory_counterexample_score", 0) > 0:
        parts.append("memory反证降权")
    if decision.get("overheat_flag"):
        parts.append("过热降权")
    if decision.get("data_gap_penalty", 0) > 0:
        parts.append("数据缺口降权")
    return "；".join(parts)


def _counter_evidence(row: pd.Series) -> str:
    flags = []
    if _safe(row.get("news_warning_score_30d")) >= 1:
        flags.append("新闻预警")
    if _safe(row.get("atr20_pct")) >= 4:
        flags.append("波动偏高")
    if _safe(row.get("counter_score")) <= 5:
        flags.append("反证分偏低")
    if "financial_publish_date_missing" in str(row.get("data_gaps")):
        flags.append("财报披露日缺失")
    return ";".join(flags) if flags else "无强反证"


def _empty_metrics(decision_dates: int, exposure_dates: int) -> dict[str, Any]:
    return {
        "decision_dates": int(decision_dates),
        "exposure_decision_dates": int(exposure_dates),
        "cash_decision_dates": int(decision_dates - exposure_dates),
        "avg_return_20d": None,
        "positive_20d_rate": None,
        "std_return_20d": None,
        "loss_20d_over_5_rate": None,
        "stability_score": None,
    }


def _safe(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _fmt(value: Any) -> str:
    number = _safe(value)
    if math.isnan(number):
        return "NA"
    return f"{number:.4f}"


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value

