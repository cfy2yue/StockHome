from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .book_rules import evaluate_book_rules
from .calendar import decision_dates
from .ground_truth import evaluate_ground_truth
from .indicators import add_indicators
from .io import load_daily_csv, load_financial_json, load_news_json
from .news_alerts import add_news_alert_features
from .news_vector import NewsVectorizer
from .scoring import score_decision


def run_backtest(
    stocks: list[dict[str, Any]],
    data_dir: str | Path,
    weights: dict[str, float],
    output_dir: str | Path,
    run_label: str,
    keep_details: bool = False,
) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    all_decisions: list[pd.DataFrame] = []
    all_gt: list[pd.DataFrame] = []
    daily_by_code: dict[str, pd.DataFrame] = {}
    for stock in stocks:
        code = str(stock["code"])
        stock_dir = data_dir / code
        daily = add_indicators(load_daily_csv(stock_dir / "daily.csv"))
        daily_by_code[code] = daily
        financial = load_financial_json(stock_dir / "financial.json")
        news_events = load_news_json(stock_dir / "news.json")
        decisions = _decisions_for_stock(stock, daily, financial, NewsVectorizer(news_events), weights)
        if decisions.empty:
            continue
        all_decisions.append(decisions)
    decisions_summary = _apply_context_features(pd.concat(all_decisions, ignore_index=True)) if all_decisions else pd.DataFrame()
    if not decisions_summary.empty:
        for code, daily in daily_by_code.items():
            decisions = decisions_summary[decisions_summary["code"].astype(str) == code].copy()
            if decisions.empty:
                continue
            gt = evaluate_ground_truth(decisions, daily)
            if keep_details:
                decision_path = output_dir / run_label / "decisions" / f"{code}_decisions.csv"
                gt_path = output_dir / run_label / "ground_truth" / f"{code}_ground_truth.csv"
                decision_path.parent.mkdir(parents=True, exist_ok=True)
                gt_path.parent.mkdir(parents=True, exist_ok=True)
                decisions.to_csv(decision_path, index=False, encoding="utf-8-sig")
                gt.to_csv(gt_path, index=False, encoding="utf-8-sig")
            all_gt.append(gt)
    gt_summary = pd.concat(all_gt, ignore_index=True) if all_gt else pd.DataFrame()
    summary_dir = output_dir / run_label
    summary_dir.mkdir(parents=True, exist_ok=True)
    decisions_summary.to_csv(summary_dir / "decisions_summary.csv", index=False, encoding="utf-8-sig")
    gt_summary.to_csv(summary_dir / "ground_truth.csv", index=False, encoding="utf-8-sig")
    return {"decisions": decisions_summary, "ground_truth": gt_summary}


def _decisions_for_stock(
    stock: dict[str, Any],
    daily: pd.DataFrame,
    financial: list[dict[str, Any]],
    news_vectorizer: NewsVectorizer,
    weights: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dates = decision_dates(daily)
    for date in dates:
        history = daily[daily["date"] <= date].copy()
        if len(history) < 60:
            continue
        available_financial = [
            record for record in financial if pd.notna(record.get("publish_date")) and record.get("publish_date") <= date
        ]
        latest = history.iloc[-1]
        triggers = evaluate_book_rules(history, stock)
        scored = score_decision(latest, available_financial, triggers, weights)
        scores = scored["scores"]
        news_vector = news_vectorizer.vectorize(date)
        rows.append(
            {
                "date": date.date().isoformat(),
                "code": stock.get("code"),
                "name": stock.get("name", ""),
                "set": stock.get("set", ""),
                "sector_group": stock.get("sector_group", ""),
                "rating": scored["rating"],
                "total_score": scored["total_score"],
                "trend_score": scores["trend_structure"],
                "financial_score": scores["fundamental_quality"],
                "safety_score": scores["financial_safety"],
                "valuation_score": scores["valuation_pressure"],
                "market_score": scores["market_regime"],
                "book_score": scores["book_strategy_match"],
                "counter_score": scores["counterevidence_risk"],
                "completeness_score": scores["data_completeness"],
                "triggered_skills": ";".join(t.strategy_id for t in triggers),
                "triggered_formulas": ";".join(t.formula for t in triggers),
                "conflict_flags": _conflicts(latest, triggers),
                "data_gaps": "" if available_financial else "financial_publish_date_missing_or_unavailable",
                "notes": ";".join(scored["notes"]),
                "prior_return_20d": latest.get("return_20d"),
                "close_above_ma200": bool(pd.notna(latest.get("ma200")) and latest.get("close") > latest.get("ma200")),
                "rsi14": latest.get("rsi14"),
                "macd_hist": latest.get("macd_hist"),
                "volume_ratio20": latest.get("volume_ratio20"),
                "drawdown60": latest.get("drawdown60"),
                "ma200_slope20": latest.get("ma200_slope20"),
                "atr20_pct": latest.get("atr20") / latest.get("close") * 100 if latest.get("close") else None,
                **news_vector,
            }
        )
    return pd.DataFrame(rows)


def _conflicts(latest: pd.Series, triggers: list) -> str:
    flags = []
    rsi = latest.get("rsi14")
    rsi_value = 0.0 if pd.isna(rsi) else float(rsi)
    if any(t.effect == "plus_1" for t in triggers) and rsi_value > 80:
        flags.append("plus_signal_with_rsi_overbought")
    return ";".join(flags)


def _apply_context_features(decisions: pd.DataFrame) -> pd.DataFrame:
    return add_news_alert_features(_apply_peer_context(_apply_relative_strength(decisions)))


def _apply_relative_strength(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty or "prior_return_20d" not in decisions:
        return decisions
    out = decisions.copy()
    out["relative_strength_rank"] = out.groupby("date")["prior_return_20d"].rank(pct=True, method="average")
    mask = out["relative_strength_rank"] >= 0.67
    source = "PPS-Q-019"
    formula = "relative_strength_20d_rank >= 0.67 within same decision_date universe"
    out.loc[mask, "triggered_skills"] = out.loc[mask, "triggered_skills"].apply(lambda v: _append_token(v, source))
    out.loc[mask, "triggered_formulas"] = out.loc[mask, "triggered_formulas"].apply(lambda v: _append_token(v, formula))
    return out


def _apply_peer_context(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty or "sector_group" not in decisions:
        return decisions
    out = decisions.copy()
    group_cols = ["date", "sector_group"]
    out["peer_group_size"] = out.groupby(group_cols)["code"].transform("count")
    numeric_cols = {
        "prior_return_20d": "peer_group_avg_return_20d",
        "news_risk_event_score_30d": "peer_group_news_risk_avg",
        "news_opportunity_event_score_30d": "peer_group_news_opportunity_avg",
        "news_count_30d": "peer_group_news_count_avg",
    }
    for source, target in numeric_cols.items():
        if source not in out:
            continue
        values = pd.to_numeric(out[source], errors="coerce")
        group_sum = values.groupby([out["date"], out["sector_group"]]).transform("sum")
        denom = (out["peer_group_size"] - 1).replace(0, pd.NA)
        out[target] = (group_sum - values) / denom
    if "prior_return_20d" in out and "peer_group_avg_return_20d" in out:
        out["peer_relative_to_group_20d"] = pd.to_numeric(out["prior_return_20d"], errors="coerce") - out["peer_group_avg_return_20d"]
        positive = (pd.to_numeric(out["prior_return_20d"], errors="coerce") > 0).astype(float)
        positive_sum = positive.groupby([out["date"], out["sector_group"]]).transform("sum")
        denom = (out["peer_group_size"] - 1).replace(0, pd.NA)
        out["peer_group_positive_breadth_20d"] = (positive_sum - positive) / denom
    if "close_above_ma200" in out:
        above = out["close_above_ma200"].astype(str).str.lower().isin(["true", "1"]).astype(float)
        above_sum = above.groupby([out["date"], out["sector_group"]]).transform("sum")
        denom = (out["peer_group_size"] - 1).replace(0, pd.NA)
        out["peer_group_above_ma200_rate"] = (above_sum - above) / denom
    peer_cols = [col for col in out.columns if col.startswith("peer_")]
    out[peer_cols] = out[peer_cols].fillna(0)
    return out


def _append_token(value, token: str) -> str:
    parts = [part for part in str(value or "").split(";") if part and part != "nan"]
    if token not in parts:
        parts.append(token)
    return ";".join(parts)
