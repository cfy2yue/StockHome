"""Backtest the user-facing stock agent capabilities.

This is intentionally framed around how the product is used:

1. P0 single-stock watch: a user already has a stock in mind. The system gives
   buy / add / hold / reduce / sell / wait style target-position decisions.
2. P1 candidate selection: a user gives a 200-stock candidate pool. The system
   first selects a small diversified shortlist by industry, then applies the
   same P0 single-stock decision logic to each selected stock.

Future returns are used only for offline evaluation. Decision rules use
decision-time features from the local joined cache.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.dual_mode_round import (  # noqa: E402
    BANK_ANNUAL_RATE,
    DEFAULT_JOINED_GT_CACHE_PATH,
    DEFAULT_PORTFOLIO_PRESET,
    TIME_BLOCKS,
    _portfolio_ranker_details,
)


REPORT_DIR = ROOT / "reports" / "date_generalization"
OUTPUT_PREFIX = "user_capability_backtest_v1"
DEFAULT_EXCLUDE_GLOBS = [
    "reports/date_generalization/*sample_plan*.csv",
    "reports/date_generalization/*evidence_pack.jsonl",
    "reports/date_generalization/*decision_ledger.jsonl",
]
BLOCK_GROUPS = {
    "Y2023H2": ["H2023_2"],
    "Y2024": ["H2024_1", "H2024_2"],
    "Y2025": ["H2025_1", "H2025_2"],
    "H2026": ["H2026_1"],
}
DEFAULT_FREQUENCIES = ["weekly_friday", "every_2_weeks", "twice_weekly"]
FUTURE_COLUMNS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "gt_status",
    "gt_pass",
    "pool_excess_20d",
    "rule_outcome_label",
    "target",
    "label",
    "outcome",
}
CODE_KEYS = {"code", "stock_code", "ts_code"}
TS_CODE_RE = re.compile(r"(?P<code>\d{6})(?:\.(?:SH|SZ|BJ))?$", re.IGNORECASE)


@dataclass(frozen=True)
class BacktestConfig:
    output_prefix: str
    single_stock_count: int
    candidate_pool_size: int
    panels: int
    holdout_seed: str
    frequencies: list[str]
    latest_only: bool = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run user-facing P0/P1 capability backtest.")
    parser.add_argument("--joined-cache", type=Path, default=DEFAULT_JOINED_GT_CACHE_PATH)
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--single-stock-count", type=int, default=100)
    parser.add_argument("--candidate-pool-size", type=int, default=200)
    parser.add_argument("--panels", type=int, default=3)
    parser.add_argument("--holdout-seed", default="user-capability-holdout-v1")
    parser.add_argument("--frequencies", default=",".join(DEFAULT_FREQUENCIES))
    parser.add_argument("--exclude-glob", action="append", default=list(DEFAULT_EXCLUDE_GLOBS))
    parser.add_argument("--latest-only", action="store_true", help="Only run H2026; useful for quick smoke.")
    args = parser.parse_args()

    cfg = BacktestConfig(
        output_prefix=safe_prefix(args.output_prefix),
        single_stock_count=args.single_stock_count,
        candidate_pool_size=args.candidate_pool_size,
        panels=args.panels,
        holdout_seed=args.holdout_seed,
        frequencies=parse_csv(args.frequencies) or DEFAULT_FREQUENCIES,
        latest_only=bool(args.latest_only),
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_feature_frame(args.joined_cache)
    excluded_codes, exclusion_summary = load_excluded_codes(args.exclude_glob, output_prefix=cfg.output_prefix)
    eligible_codes = sorted(set(frame["code"].dropna().astype(str)) - excluded_codes)
    if len(eligible_codes) < max(cfg.single_stock_count, cfg.candidate_pool_size):
        # Strictly unseen-by-prior-DS codes can be too few after many experiments.
        # Fall back to hash-stable codes while keeping an audit column honest.
        eligible_codes = sorted(frame["code"].dropna().astype(str).unique())
        holdout_definition = "hash_stable_codes_prior_ds_exclusion_insufficient"
    else:
        holdout_definition = "exclude_codes_seen_in_prior_sample_or_ds_artifacts"

    panel_specs = build_holdout_panels(
        eligible_codes,
        panels=cfg.panels,
        single_stock_count=cfg.single_stock_count,
        candidate_pool_size=cfg.candidate_pool_size,
        seed=cfg.holdout_seed,
    )
    scored = add_policy_scores(frame)
    blocks_to_run = {"H2026": ["H2026_1"]} if cfg.latest_only else BLOCK_GROUPS

    single_detail, single_summary = run_single_stock_watch_backtest(scored, panel_specs, cfg, blocks_to_run)
    candidate_detail, candidate_selection, candidate_summary = run_candidate_then_watch_backtest(scored, panel_specs, cfg, blocks_to_run)
    baselines = build_baseline_summary(scored, panel_specs, cfg, blocks_to_run)

    paths = write_outputs(
        cfg,
        single_detail,
        single_summary,
        candidate_detail,
        candidate_selection,
        candidate_summary,
        baselines,
        exclusion_summary=exclusion_summary,
        holdout_definition=holdout_definition,
    )
    print("A股研究Agent")
    print(f"single_decisions={len(single_detail)} candidate_decisions={len(candidate_detail)}")
    print(f"report={paths['report']}")


def load_feature_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    frame.columns = [col.lstrip("\ufeff") for col in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["valid_block"] = frame["date"].map(block_for_date)
    if "return_20d" in frame:
        frame["return_20d"] = pd.to_numeric(frame["return_20d"], errors="coerce")
    if "gt_status" in frame:
        frame = frame[frame["gt_status"].astype(str).eq("evaluated")].copy()
    frame = frame[frame["return_20d"].notna()].copy()
    return frame.reset_index(drop=True)


def add_policy_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ranker = _portfolio_ranker_details(
        out,
        preset=DEFAULT_PORTFOLIO_PRESET,
        valid_block="user_capability_backtest",
        decision_frequency="user_capability_multi_frequency",
    )
    out["rev_chip_score"] = pd.to_numeric(ranker["score"], errors="coerce")
    out["rev_chip_score_quantile"] = pd.to_numeric(ranker["score_quantile"], errors="coerce")
    out["agent_policy_score"] = agent_policy_score(out)
    out["candidate_selector_score"] = candidate_selector_score(out)
    out["industry_for_selection"] = out.get("tushare_industry", pd.Series("UNKNOWN", index=out.index)).fillna("UNKNOWN").astype(str)
    return out


def agent_policy_score(frame: pd.DataFrame) -> pd.Series:
    rev = _num(frame, "rev_chip_score_quantile").fillna(0.5)
    peer = _num(frame, "tushare_industry_positive_breadth_20d").fillna(_num(frame, "peer_group_positive_breadth_20d")).fillna(0.5)
    peer_rel = _num(frame, "tushare_industry_relative_return_20d").fillna(_num(frame, "peer_relative_to_group_20d")).fillna(0.0)
    news_opp = _num(frame, "news_opportunity_score").fillna(_num(frame, "news_opportunity_event_score_30d")).fillna(0.0)
    news_warn = _num(frame, "news_warning_score").fillna(_num(frame, "news_warning_score_30d")).fillna(0.0)
    fin_risk = _num(frame, "financial_quality_risk_score").fillna(0.0)
    fin_surprise = _num(frame, "financial_surprise_score").fillna(0.0)
    missing_news = _num(frame, "news_missing_rate").fillna(1.0)
    volatility = _num(frame, "kline_volatility_ratio_20_60").fillna(_num(frame, "kline_volatility_ratio_3_20")).fillna(1.0)
    score = (
        1.10 * rev
        + 0.12 * (peer - 0.5)
        + 0.05 * np.tanh(peer_rel / 5.0)
        + 0.08 * news_opp
        + 0.04 * fin_surprise.clip(lower=-1.0, upper=1.0)
        - 0.22 * news_warn
        - 0.16 * fin_risk
        - 0.06 * missing_news
        - 0.04 * volatility.clip(lower=0.0, upper=3.0)
    )
    return pd.to_numeric(score, errors="coerce").fillna(0.0)


def candidate_selector_score(frame: pd.DataFrame) -> pd.Series:
    # Candidate selection is broader than the final position decision: it keeps
    # industry/area context and positive non-price confirmation as tie breakers.
    policy = _num(frame, "agent_policy_score").fillna(0.0)
    industry_rel = _num(frame, "tushare_industry_relative_return_20d").fillna(0.0)
    area_rel = _num(frame, "tushare_area_relative_return_20d").fillna(0.0)
    news_quality = _num(frame, "news_evidence_quality").fillna(0.0)
    official = _num(frame, "official_confirmation_score").fillna(0.0)
    book = _num(frame, "book_score").fillna(0.0)
    return policy + 0.08 * np.tanh(industry_rel / 5.0) + 0.04 * np.tanh(area_rel / 5.0) + 0.03 * news_quality + 0.03 * official + 0.02 * book


def operation_decision(row: pd.Series | dict[str, Any], *, previous_position: float = 0.0) -> dict[str, Any]:
    score_q = _safe(row.get("rev_chip_score_quantile"), 0.5)
    policy = _safe(row.get("agent_policy_score"), 0.0)
    news_warning = max(_safe(row.get("news_warning_score"), 0.0), _safe(row.get("news_warning_score_30d"), 0.0))
    news_opp = max(_safe(row.get("news_opportunity_score"), 0.0), _safe(row.get("news_opportunity_event_score_30d"), 0.0))
    fin_risk = _safe(row.get("financial_quality_risk_score"), 0.0)
    fin_event_count = _safe(row.get("financial_report_event_count"), 0.0)
    fin_surprise = _safe(row.get("financial_surprise_score"), 0.0)
    peer_breadth = _safe(row.get("tushare_industry_positive_breadth_20d"), _safe(row.get("peer_group_positive_breadth_20d"), 0.5))
    peer_rel = _safe(row.get("tushare_industry_relative_return_20d"), _safe(row.get("peer_relative_to_group_20d"), 0.0))
    prior20 = _safe(row.get("kline_return_20d"), _safe(row.get("prior_return_20d"), 0.0))
    prior60 = _safe(row.get("kline_return_60d"), 0.0)
    rsi = _safe(row.get("kline_rsi14"), _safe(row.get("rsi14"), 50.0))
    overhang = _safe(row.get("upper_overhang"), 0.0)
    lower_support = _safe(row.get("lower_support"), 0.0)
    missing_news = _safe(row.get("news_missing_rate"), 1.0)
    financial_status = str(row.get("financial_report_join_status") or "")

    hard_risks: list[str] = []
    soft_supports: list[str] = []
    if news_warning >= 0.67:
        hard_risks.append("news_warning>=0.67")
    if fin_risk >= 0.70 or (fin_event_count >= 1 and fin_surprise <= -0.35):
        hard_risks.append("financial_high_risk_or_negative_surprise")
    if peer_rel <= -4.0 and peer_breadth <= 0.40:
        hard_risks.append("industry_peer_weak")
    if (prior20 >= 25 and rsi >= 75 and overhang >= 0.25) or (prior60 >= 45 and rsi >= 78):
        hard_risks.append("overheat_chase_risk")
    if prior20 <= -18 and peer_rel <= -3.0:
        hard_risks.append("falling_with_peer_weakness")

    if news_opp >= 0.33 and news_warning < 0.5:
        soft_supports.append("news_opportunity_without_high_warning")
    if fin_event_count >= 1 and fin_risk <= 0.4 and fin_surprise >= 0:
        soft_supports.append("financial_event_low_risk")
    if peer_breadth >= 0.58 and peer_rel >= 0:
        soft_supports.append("peer_context_supportive")
    if lower_support >= 0.20:
        soft_supports.append("chip_lower_support")

    target = 0.0
    reason = "score_or_confirmation_insufficient"
    if len(hard_risks) >= 2 or (hard_risks and score_q < 0.88):
        target = 0.0
        reason = "hard_risk_blocks_buy_or_requires_exit"
    elif score_q >= 0.90 and policy >= 0.62 and len(soft_supports) >= 1:
        target = 0.80
        reason = "high_score_with_cross_channel_support"
    elif score_q >= 0.82 and policy >= 0.54:
        target = 0.60
        reason = "score_strong_no_hard_risk"
    elif score_q >= 0.70 and policy >= 0.45 and not hard_risks:
        target = 0.35
        reason = "moderate_score_watch_position"
    elif previous_position > 0 and score_q >= 0.55 and not hard_risks:
        target = min(previous_position, 0.30)
        reason = "existing_position_hold_small"

    if missing_news >= 0.95 and financial_status in {"", "no_event_in_window"} and target > 0.60:
        target = 0.60
        reason += ";cap_for_news_financial_sparse"
    target = float(max(0.0, min(1.0, target)))
    action = action_from_position_change(previous_position, target)
    return {
        "operation_action": action,
        "target_position": round(target, 4),
        "previous_position": round(float(previous_position), 4),
        "position_change": round(target - float(previous_position), 4),
        "operation_reason_code": reason,
        "hard_risk_count": len(hard_risks),
        "hard_risk_reasons": ";".join(hard_risks) if hard_risks else "none",
        "support_count": len(soft_supports),
        "support_reasons": ";".join(soft_supports) if soft_supports else "none",
        "rev_chip_score_quantile": round(score_q, 6),
        "agent_policy_score": round(policy, 6),
    }


def action_from_position_change(previous: float, target: float) -> str:
    if target <= 0.0:
        return "卖出/不买" if previous > 0 else "等待不买"
    if previous <= 0.0:
        return "买入" if target >= 0.35 else "小仓试探"
    delta = target - previous
    if delta >= 0.15:
        return "加仓"
    if delta <= -0.15:
        return "减仓"
    return "持有"


def run_single_stock_watch_backtest(
    frame: pd.DataFrame,
    panel_specs: list[dict[str, Any]],
    cfg: BacktestConfig,
    block_groups: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, Any]] = []
    for panel in panel_specs:
        codes = set(panel["single_codes"])
        panel_frame = frame[frame["code"].isin(codes)].copy()
        for period_name, blocks in block_groups.items():
            period_frame = panel_frame[panel_frame["valid_block"].isin(blocks)].copy()
            if period_frame.empty:
                continue
            for frequency in cfg.frequencies:
                scheduled = apply_frequency(period_frame, frequency)
                if scheduled.empty:
                    continue
                for code, stock_rows in scheduled.groupby("code", sort=True):
                    previous_position = 0.0
                    for _, row in stock_rows.sort_values("date").iterrows():
                        decision = operation_decision(row, previous_position=previous_position)
                        detail_rows.append(
                            {
                                "task_mode": "single_stock_watch",
                                "panel_id": panel["panel_id"],
                                "period": period_name,
                                "decision_frequency": frequency,
                                "date": row["date"],
                                "code": code,
                                "name": row.get("name", ""),
                                "return_20d": row["return_20d"],
                                **decision,
                                "baseline_hold_return_20d": row["return_20d"],
                                "cash_return_20d": bank_return_20d(),
                                "strategy_return_20d": decision["target_position"] * row["return_20d"] + (1 - decision["target_position"]) * bank_return_20d(),
                                "capital_100k_after_20d": 100_000 * (1 + (decision["target_position"] * row["return_20d"] + (1 - decision["target_position"]) * bank_return_20d()) / 100.0),
                                "holdout_definition": panel["holdout_definition"],
                                "research_only": True,
                                "not_auto_trade": True,
                            }
                        )
                        previous_position = decision["target_position"]
    detail = pd.DataFrame(detail_rows)
    return detail, summarize_operation_detail(detail, group_cols=["task_mode", "panel_id", "period", "decision_frequency"])


def run_candidate_then_watch_backtest(
    frame: pd.DataFrame,
    panel_specs: list[dict[str, Any]],
    cfg: BacktestConfig,
    block_groups: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for panel in panel_specs:
        codes = set(panel["candidate_codes"])
        panel_frame = frame[frame["code"].isin(codes)].copy()
        for period_name, blocks in block_groups.items():
            period_frame = panel_frame[panel_frame["valid_block"].isin(blocks)].copy()
            if period_frame.empty:
                continue
            for frequency in cfg.frequencies:
                scheduled = apply_frequency(period_frame, frequency)
                if scheduled.empty:
                    continue
                previous_positions: dict[str, float] = {}
                for date, day in scheduled.groupby("date", sort=True):
                    day = day.copy()
                    selected = select_candidate_shortlist(day, pool_size=cfg.candidate_pool_size)
                    selection_rows.append(selection_metric_row(day, selected, panel, period_name, frequency, date))
                    for _, row in selected.iterrows():
                        code = str(row["code"]).zfill(6)
                        previous = previous_positions.get(code, 0.0)
                        decision = operation_decision(row, previous_position=previous)
                        detail_rows.append(
                            {
                                "task_mode": "candidate_select_then_single_watch",
                                "panel_id": panel["panel_id"],
                                "period": period_name,
                                "decision_frequency": frequency,
                                "date": date,
                                "code": code,
                                "name": row.get("name", ""),
                                "tushare_industry": row.get("tushare_industry", "UNKNOWN"),
                                "selection_rank": int(row.get("selection_rank", 0)),
                                "return_20d": row["return_20d"],
                                **decision,
                                "baseline_pool_mean_return_20d": pd.to_numeric(day["return_20d"], errors="coerce").mean(),
                                "cash_return_20d": bank_return_20d(),
                                "strategy_return_20d": decision["target_position"] * row["return_20d"] + (1 - decision["target_position"]) * bank_return_20d(),
                                "capital_100k_after_20d": 100_000 * (1 + (decision["target_position"] * row["return_20d"] + (1 - decision["target_position"]) * bank_return_20d()) / 100.0),
                                "holdout_definition": panel["holdout_definition"],
                                "research_only": True,
                                "not_auto_trade": True,
                            }
                        )
                        previous_positions[code] = decision["target_position"]
    detail = pd.DataFrame(detail_rows)
    selection = pd.DataFrame(selection_rows)
    summary = summarize_operation_detail(
        detail,
        group_cols=["task_mode", "panel_id", "period", "decision_frequency"],
        extra_baseline_col="baseline_pool_mean_return_20d",
    )
    return detail, selection, summary


def select_candidate_shortlist(day: pd.DataFrame, *, pool_size: int) -> pd.DataFrame:
    if day.empty:
        return day.copy()
    data = day.copy()
    data = data.sort_values(["candidate_selector_score", "code"], ascending=[False, True])
    per_industry_cap = 2
    target_n = max(3, min(12, int(round(pool_size * 0.06))))
    industry_first = (
        data.groupby("industry_for_selection", group_keys=False)
        .head(per_industry_cap)
        .sort_values(["candidate_selector_score", "code"], ascending=[False, True])
        .head(target_n)
        .copy()
    )
    if len(industry_first) < target_n:
        rest = data[~data["code"].isin(industry_first["code"])].head(target_n - len(industry_first))
        industry_first = pd.concat([industry_first, rest], ignore_index=False)
    out = industry_first.head(target_n).copy()
    out["selection_rank"] = range(1, len(out) + 1)
    return out


def summarize_operation_detail(detail: pd.DataFrame, *, group_cols: list[str], extra_baseline_col: str | None = None) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in detail.groupby(group_cols, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        strategy = pd.to_numeric(group["strategy_return_20d"], errors="coerce")
        hold = pd.to_numeric(group["baseline_hold_return_20d"] if "baseline_hold_return_20d" in group else group["return_20d"], errors="coerce")
        active = pd.to_numeric(group["target_position"], errors="coerce").fillna(0.0) >= 0.35
        active_strategy = strategy[active]
        active_hold = hold[active]
        row = {col: value for col, value in zip(group_cols, key_values)}
        row.update(
            {
                "decision_count": int(len(group)),
                "unique_stocks": int(group["code"].nunique()),
                "decision_dates": int(group["date"].nunique()),
                "avg_target_position": round(float(pd.to_numeric(group["target_position"], errors="coerce").mean()), 6),
                "active_decision_rate": round(float(active.mean()), 6),
                "active_decision_count": int(active.sum()),
                "active_strategy_positive_20d_rate": round(float((active_strategy > 0).mean()), 6) if len(active_strategy) else np.nan,
                "active_strategy_avg_return_20d": round(float(active_strategy.mean()), 6) if len(active_strategy) else np.nan,
                "active_hold_positive_20d_rate": round(float((active_hold > 0).mean()), 6) if len(active_hold) else np.nan,
                "active_hold_avg_return_20d": round(float(active_hold.mean()), 6) if len(active_hold) else np.nan,
                "active_excess_avg_return_vs_hold": round(float(active_strategy.mean() - active_hold.mean()), 6) if len(active_strategy) and len(active_hold) else np.nan,
                "buy_or_add_rate": round(float(group["operation_action"].isin(["买入", "加仓", "小仓试探"]).mean()), 6),
                "strategy_positive_20d_rate": round(float((strategy > 0).mean()), 6),
                "strategy_avg_return_20d": round(float(strategy.mean()), 6),
                "strategy_std_return_20d": round(float(strategy.std(ddof=0)), 6),
                "strategy_loss_gt5_rate": round(float((strategy <= -5).mean()), 6),
                "hold_positive_20d_rate": round(float((hold > 0).mean()), 6),
                "hold_avg_return_20d": round(float(hold.mean()), 6),
                "excess_avg_return_vs_hold": round(float(strategy.mean() - hold.mean()), 6),
                "capital_100k_mean_after_20d": round(float(pd.to_numeric(group["capital_100k_after_20d"], errors="coerce").mean()), 2),
                "research_only": True,
                "not_auto_trade": True,
            }
        )
        if extra_baseline_col and extra_baseline_col in group:
            baseline = pd.to_numeric(group[extra_baseline_col], errors="coerce")
            row["excess_avg_return_vs_candidate_pool"] = round(float(strategy.mean() - baseline.mean()), 6)
        rows.append(row)
    return pd.DataFrame(rows)


def selection_metric_row(day: pd.DataFrame, selected: pd.DataFrame, panel: dict[str, Any], period: str, frequency: str, date: str) -> dict[str, Any]:
    pool_ret = pd.to_numeric(day["return_20d"], errors="coerce")
    selected_ret = pd.to_numeric(selected["return_20d"], errors="coerce") if not selected.empty else pd.Series(dtype=float)
    return {
        "task_mode": "candidate_selection",
        "panel_id": panel["panel_id"],
        "period": period,
        "decision_frequency": frequency,
        "date": date,
        "candidate_pool_rows": int(len(day)),
        "selected_count": int(len(selected)),
        "selected_industries": int(selected["industry_for_selection"].nunique()) if not selected.empty else 0,
        "pool_positive_20d_rate": round(float((pool_ret > 0).mean()), 6) if len(pool_ret) else np.nan,
        "pool_avg_return_20d": round(float(pool_ret.mean()), 6) if len(pool_ret) else np.nan,
        "selected_positive_20d_rate": round(float((selected_ret > 0).mean()), 6) if len(selected_ret) else np.nan,
        "selected_avg_return_20d": round(float(selected_ret.mean()), 6) if len(selected_ret) else np.nan,
        "selected_excess_vs_pool": round(float(selected_ret.mean() - pool_ret.mean()), 6) if len(selected_ret) and len(pool_ret) else np.nan,
        "research_only": True,
        "not_auto_trade": True,
    }


def build_baseline_summary(
    frame: pd.DataFrame,
    panel_specs: list[dict[str, Any]],
    cfg: BacktestConfig,
    block_groups: dict[str, list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for panel in panel_specs:
        for universe_name, codes in [("single_100_holdout", panel["single_codes"]), ("candidate_200_holdout", panel["candidate_codes"])]:
            scoped = frame[frame["code"].isin(codes)].copy()
            for period_name, blocks in block_groups.items():
                period = scoped[scoped["valid_block"].isin(blocks)].copy()
                if period.empty:
                    continue
                for frequency in cfg.frequencies:
                    scheduled = apply_frequency(period, frequency)
                    returns = pd.to_numeric(scheduled["return_20d"], errors="coerce").dropna()
                    rows.append(
                        {
                            "baseline": "long_hold_each_decision_20d",
                            "panel_id": panel["panel_id"],
                            "universe": universe_name,
                            "period": period_name,
                            "decision_frequency": frequency,
                            "decision_count": int(len(returns)),
                            "unique_stocks": int(scheduled["code"].nunique()) if not scheduled.empty else 0,
                            "positive_20d_rate": round(float((returns > 0).mean()), 6) if len(returns) else np.nan,
                            "avg_return_20d": round(float(returns.mean()), 6) if len(returns) else np.nan,
                            "std_return_20d": round(float(returns.std(ddof=0)), 6) if len(returns) else np.nan,
                            "loss_gt5_rate": round(float((returns <= -5).mean()), 6) if len(returns) else np.nan,
                            "research_only": True,
                            "not_auto_trade": True,
                        }
                    )
                    rows.append(
                        {
                            "baseline": "cash_3pct_annualized",
                            "panel_id": panel["panel_id"],
                            "universe": universe_name,
                            "period": period_name,
                            "decision_frequency": frequency,
                            "decision_count": int(len(returns)),
                            "unique_stocks": int(scheduled["code"].nunique()) if not scheduled.empty else 0,
                            "positive_20d_rate": 1.0 if len(returns) else np.nan,
                            "avg_return_20d": round(bank_return_20d(), 6) if len(returns) else np.nan,
                            "std_return_20d": 0.0 if len(returns) else np.nan,
                            "loss_gt5_rate": 0.0 if len(returns) else np.nan,
                            "research_only": True,
                            "not_auto_trade": True,
                        }
                    )
    return pd.DataFrame(rows)


def build_holdout_panels(
    codes: list[str],
    *,
    panels: int,
    single_stock_count: int,
    candidate_pool_size: int,
    seed: str,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for idx in range(max(1, panels)):
        ordered = stable_order(codes, f"{seed}:panel:{idx}")
        need = max(single_stock_count, candidate_pool_size)
        chosen = ordered[: min(need, len(ordered))]
        specs.append(
            {
                "panel_id": f"holdout_panel_{idx + 1:02d}",
                "single_codes": chosen[: min(single_stock_count, len(chosen))],
                "candidate_codes": chosen[: min(candidate_pool_size, len(chosen))],
                "holdout_definition": "code_holdout_stable_hash_not_selected_by_future_return",
            }
        )
    return specs


def load_excluded_codes(globs: Iterable[str], *, output_prefix: str) -> tuple[set[str], pd.DataFrame]:
    codes: set[str] = set()
    rows: list[dict[str, Any]] = []
    for pattern in globs:
        for path in sorted(ROOT.glob(pattern)):
            if path.name.startswith(output_prefix):
                continue
            found = codes_from_artifact(path)
            if found:
                codes.update(found)
                rows.append({"artifact": str(path.relative_to(ROOT)), "codes": len(found)})
    return codes, pd.DataFrame(rows)


def codes_from_artifact(path: Path) -> set[str]:
    try:
        if path.suffix == ".jsonl":
            found: set[str] = set()
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        found.update(_codes_from_json_obj(json.loads(line)))
                    except json.JSONDecodeError:
                        continue
            return found
        header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
        header = [col.lstrip("\ufeff") for col in header]
        if "code" not in header:
            return set()
        frame = pd.read_csv(path, dtype={"code": str}, usecols=["code"], low_memory=False, encoding="utf-8-sig")
        return set(frame["code"].dropna().astype(str).str.zfill(6))
    except Exception:
        return set()


def _codes_from_json_obj(obj: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key) in CODE_KEYS:
                code = _normalize_code_value(value)
                if code:
                    found.add(code)
            if isinstance(value, (dict, list)):
                found.update(_codes_from_json_obj(value))
    elif isinstance(obj, list):
        for item in obj:
            found.update(_codes_from_json_obj(item))
    return found


def _normalize_code_value(value: Any) -> str:
    text = str(value or "").strip()
    match = TS_CODE_RE.match(text)
    return match.group("code") if match else ""


def apply_frequency(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frame.empty or frequency in {"", "all_dates"}:
        return frame.sort_values(["date", "code"]).copy()
    out = frame.copy()
    dates = pd.to_datetime(out["date"], errors="coerce")
    if frequency == "weekly_friday":
        selected = out[dates.dt.weekday.eq(4)].copy()
    elif frequency == "weekly_tuesday":
        selected = out[dates.dt.weekday.eq(1)].copy()
    elif frequency == "twice_weekly":
        selected = out[dates.dt.weekday.isin([1, 4])].copy()
    elif frequency == "every_2_weeks":
        selected = out[dates.dt.isocalendar().week.astype(int).mod(2).eq(0)].copy()
    elif frequency == "monthly_last":
        out["_ym"] = dates.dt.to_period("M").astype(str)
        last_dates = out.groupby("_ym")["date"].max()
        selected = out[out["date"].isin(set(last_dates))].drop(columns=["_ym"], errors="ignore").copy()
    else:
        raise ValueError(f"unknown frequency: {frequency}")
    return selected.sort_values(["date", "code"]).reset_index(drop=True)


def write_outputs(
    cfg: BacktestConfig,
    single_detail: pd.DataFrame,
    single_summary: pd.DataFrame,
    candidate_detail: pd.DataFrame,
    candidate_selection: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    baselines: pd.DataFrame,
    *,
    exclusion_summary: pd.DataFrame,
    holdout_definition: str,
) -> dict[str, Path]:
    prefix = cfg.output_prefix
    paths = {
        "single_detail": REPORT_DIR / f"{prefix}_single_stock_detail.csv",
        "single_summary": REPORT_DIR / f"{prefix}_single_stock_summary.csv",
        "candidate_detail": REPORT_DIR / f"{prefix}_candidate_then_watch_detail.csv",
        "candidate_selection": REPORT_DIR / f"{prefix}_candidate_selection_detail.csv",
        "candidate_summary": REPORT_DIR / f"{prefix}_candidate_then_watch_summary.csv",
        "baselines": REPORT_DIR / f"{prefix}_baselines.csv",
        "exclusions": REPORT_DIR / f"{prefix}_exclusion_summary.csv",
        "report": REPORT_DIR / f"{prefix}.md",
    }
    single_detail.to_csv(paths["single_detail"], index=False, encoding="utf-8-sig")
    single_summary.to_csv(paths["single_summary"], index=False, encoding="utf-8-sig")
    candidate_detail.to_csv(paths["candidate_detail"], index=False, encoding="utf-8-sig")
    candidate_selection.to_csv(paths["candidate_selection"], index=False, encoding="utf-8-sig")
    candidate_summary.to_csv(paths["candidate_summary"], index=False, encoding="utf-8-sig")
    baselines.to_csv(paths["baselines"], index=False, encoding="utf-8-sig")
    exclusion_summary.to_csv(paths["exclusions"], index=False, encoding="utf-8-sig")
    paths["report"].write_text(
        render_report(
            cfg,
            single_summary,
            candidate_summary,
            candidate_selection,
            baselines,
            paths=paths,
            holdout_definition=holdout_definition,
        ),
        encoding="utf-8",
    )
    return paths


def render_report(
    cfg: BacktestConfig,
    single_summary: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    candidate_selection: pd.DataFrame,
    baselines: pd.DataFrame,
    *,
    paths: dict[str, Path],
    holdout_definition: str,
) -> str:
    single_overall = aggregate_user_summary(single_summary, ["task_mode", "period", "decision_frequency"])
    candidate_overall = aggregate_user_summary(candidate_summary, ["task_mode", "period", "decision_frequency"])
    selection_overall = aggregate_selection(candidate_selection)
    lines = [
        "# User Capability Backtest v1",
        "",
        "本报告按用户真实使用路径测试：先给明确操作建议，再把同一套逻辑放回历史数据评估。系统不自动交易、不接券商接口，不承诺收益。",
        "",
        "## What Was Tested",
        "",
        f"- P0 单支盯盘：每个 panel 稳定抽取 `{cfg.single_stock_count}` 支 holdout 股票；每个决策点按同一套规则输出买入/加仓/持有/减仓/卖出/等待和目标仓位；每支股票按 10 万初始资金折算 20 日结果。",
        f"- P1 候选选择再盯盘：每个 panel 稳定抽取 `{cfg.candidate_pool_size}` 支 holdout 候选；每个决策日先按行业分散选出最多 12 支，再进入同一套 P0 单股操作逻辑。",
        f"- panels: `{cfg.panels}`；frequencies: `{','.join(cfg.frequencies)}`；holdout_definition: `{holdout_definition}`。",
        "- 未来 20 日收益只用于离线评估，未进入操作规则。",
        "",
        "## P0 Single-Stock Watch Summary",
        "",
        markdown_table(single_overall),
        "",
        "## P1 Candidate Selection Then Watch Summary",
        "",
        markdown_table(candidate_overall),
        "",
        "## Candidate Selection Quality Before Watch Logic",
        "",
        markdown_table(selection_overall),
        "",
        "## Baselines",
        "",
        markdown_table(aggregate_baselines(baselines)),
        "",
        "## Interpretation",
        "",
        "- `strategy_positive_20d_rate` 是用户按系统目标仓位操作后的 20 日收益为正比例；`hold_positive_20d_rate` 是同样股票直接持有的基线。",
        "- `active_strategy_positive_20d_rate` 只统计目标仓位 >=35% 的有效介入决策，更接近用户真正执行买入/加仓/持有后的胜率。",
        "- `excess_avg_return_vs_hold` > 0 表示操作层相对直接持有提高了均值；若同时 `active_decision_rate` 很低，说明主要是防守能力，不应夸大为高进攻 alpha。",
        "- 候选模式要同时看 `selected_excess_vs_pool` 和后续 watch 后的 `strategy_avg_return_20d`，因为用户真实路径是先筛选，再决定是否买/加/减/卖。",
        "- 当前脚本是本地确定性执行层的大样本验收；下一步可从本报告的关键日期/行业样本中抽样调用 DS Flash/Pro 做解释审计和阈值复核。",
        "",
        "## Artifacts",
        "",
        *(f"- `{path}`" for path in paths.values()),
        "",
    ]
    return "\n".join(lines)


def aggregate_user_summary(summary: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    out = (
        summary.groupby(group_cols, dropna=False)
        .agg(
            panels=("panel_id", "nunique"),
            decision_count_mean=("decision_count", "mean"),
            decision_count_std=("decision_count", "std"),
            unique_stocks_mean=("unique_stocks", "mean"),
            avg_target_position=("avg_target_position", "mean"),
            avg_target_position_std=("avg_target_position", "std"),
            active_decision_rate=("active_decision_rate", "mean"),
            active_decision_rate_std=("active_decision_rate", "std"),
            active_decision_count_mean=("active_decision_count", "mean"),
            active_strategy_positive_20d_rate=("active_strategy_positive_20d_rate", "mean"),
            active_strategy_positive_20d_rate_std=("active_strategy_positive_20d_rate", "std"),
            active_strategy_avg_return_20d=("active_strategy_avg_return_20d", "mean"),
            active_strategy_avg_return_20d_std=("active_strategy_avg_return_20d", "std"),
            active_hold_positive_20d_rate=("active_hold_positive_20d_rate", "mean"),
            active_hold_avg_return_20d=("active_hold_avg_return_20d", "mean"),
            active_excess_avg_return_vs_hold=("active_excess_avg_return_vs_hold", "mean"),
            active_excess_avg_return_vs_hold_std=("active_excess_avg_return_vs_hold", "std"),
            strategy_positive_20d_rate=("strategy_positive_20d_rate", "mean"),
            strategy_positive_20d_rate_std=("strategy_positive_20d_rate", "std"),
            strategy_avg_return_20d=("strategy_avg_return_20d", "mean"),
            strategy_avg_return_20d_std=("strategy_avg_return_20d", "std"),
            strategy_std_return_20d=("strategy_std_return_20d", "mean"),
            strategy_loss_gt5_rate=("strategy_loss_gt5_rate", "mean"),
            strategy_loss_gt5_rate_std=("strategy_loss_gt5_rate", "std"),
            hold_positive_20d_rate=("hold_positive_20d_rate", "mean"),
            hold_positive_20d_rate_std=("hold_positive_20d_rate", "std"),
            hold_avg_return_20d=("hold_avg_return_20d", "mean"),
            hold_avg_return_20d_std=("hold_avg_return_20d", "std"),
            excess_avg_return_vs_hold=("excess_avg_return_vs_hold", "mean"),
            excess_avg_return_vs_hold_std=("excess_avg_return_vs_hold", "std"),
            capital_100k_mean_after_20d=("capital_100k_mean_after_20d", "mean"),
            capital_100k_mean_after_20d_std=("capital_100k_mean_after_20d", "std"),
        )
        .reset_index()
    )
    return out.fillna(0.0).round(6)


def aggregate_selection(selection: pd.DataFrame) -> pd.DataFrame:
    if selection.empty:
        return pd.DataFrame()
    out = (
        selection.groupby(["period", "decision_frequency"], dropna=False)
        .agg(
            panels=("panel_id", "nunique"),
            decision_dates=("date", "nunique"),
            selected_count_mean=("selected_count", "mean"),
            selected_count_std=("selected_count", "std"),
            selected_industries_mean=("selected_industries", "mean"),
            pool_positive_20d_rate=("pool_positive_20d_rate", "mean"),
            selected_positive_20d_rate=("selected_positive_20d_rate", "mean"),
            selected_positive_20d_rate_std=("selected_positive_20d_rate", "std"),
            pool_avg_return_20d=("pool_avg_return_20d", "mean"),
            selected_avg_return_20d=("selected_avg_return_20d", "mean"),
            selected_avg_return_20d_std=("selected_avg_return_20d", "std"),
            selected_excess_vs_pool=("selected_excess_vs_pool", "mean"),
            selected_excess_vs_pool_std=("selected_excess_vs_pool", "std"),
        )
        .reset_index()
    )
    return out.fillna(0.0).round(6)


def aggregate_baselines(baselines: pd.DataFrame) -> pd.DataFrame:
    if baselines.empty:
        return pd.DataFrame()
    out = (
        baselines.groupby(["baseline", "universe", "period", "decision_frequency"], dropna=False)
        .agg(
            panels=("panel_id", "nunique"),
            decision_count_mean=("decision_count", "mean"),
            decision_count_std=("decision_count", "std"),
            positive_20d_rate=("positive_20d_rate", "mean"),
            positive_20d_rate_std=("positive_20d_rate", "std"),
            avg_return_20d=("avg_return_20d", "mean"),
            avg_return_20d_std=("avg_return_20d", "std"),
            std_return_20d=("std_return_20d", "mean"),
            loss_gt5_rate=("loss_gt5_rate", "mean"),
            loss_gt5_rate_std=("loss_gt5_rate", "std"),
        )
        .reset_index()
    )
    return out.fillna(0.0).round(6)


def block_for_date(value: Any) -> str | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    for block, (start, end) in TIME_BLOCKS.items():
        if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
            return block
    return None


def bank_return_20d() -> float:
    return ((1 + BANK_ANNUAL_RATE) ** (20 / 252) - 1) * 100


def stable_order(items: Iterable[str], seed: str) -> list[str]:
    return sorted(set(items), key=lambda item: hashlib.sha256(f"{seed}:{item}".encode("utf-8")).hexdigest())


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def safe_prefix(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value)).strip("_")
    return safe or OUTPUT_PREFIX


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _safe(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(number) else number


def markdown_table(frame: pd.DataFrame, *, max_rows: int = 80) -> str:
    if frame.empty:
        return "_empty_"
    try:
        return frame.head(max_rows).to_markdown(index=False)
    except Exception:
        return frame.head(max_rows).to_csv(index=False)


if __name__ == "__main__":
    main()
