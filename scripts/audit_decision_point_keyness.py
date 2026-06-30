"""Audit key decision point sampling for Agent training.

The sampler is meant to decide where Agent reasoning is worth spending tokens:
ordinary scheduled dates are cheap controls, while key dates are high-conflict
or high-information dates where different decisions can matter more.

Future returns are used only in offline audit columns and reports. The
rule_outcomes written for Agent consumption are sanitized and contain no future
result fields.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.dual_mode_round import (  # noqa: E402
    DEFAULT_JOINED_GT_CACHE_PATH,
    DEFAULT_PORTFOLIO_PRESET,
    TIME_BLOCKS,
    _portfolio_ranker_details,
)
from src.agent_training.quant_tool_context import sanitize_quant_tool_outcome  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
OUTPUT_PREFIX = "decision_point_keyness_v1"
HIGH_RANKER_QUANTILE = 0.80
HORIZONS = [5, 10, 20]

EX_ANTE_COLUMNS = [
    "date",
    "code",
    "name",
    "gt_status",
    "return_5d",
    "return_10d",
    "return_20d",
    "kline_return_3d",
    "kline_return_5d",
    "kline_return_10d",
    "kline_return_20d",
    "kline_return_60d",
    "kline_return_120d",
    "kline_return_240d",
    "kline_drawdown_20d",
    "kline_drawdown_60d",
    "kline_drawdown_120d",
    "kline_range_position_20d",
    "kline_range_position_60d",
    "kline_range_position_120d",
    "kline_efficiency_ratio_20d",
    "kline_direction_reversal_rate_20d",
    "kline_oscillation_cross_count_20d",
    "kline_volatility_20d",
    "kline_volatility_60d",
    "kline_atr20_pct",
    "peer_relative_to_group_20d",
    "peer_group_positive_breadth_20d",
    "peer_group_news_risk_avg",
    "peer_group_news_opportunity_avg",
    "corr_peer_avg_return_20d",
    "corr_peer_relative_return_20d",
    "corr_peer_positive_breadth_20d",
    "tushare_industry_relative_return_20d",
    "tushare_industry_positive_breadth_20d",
    "tushare_area_relative_return_20d",
    "tushare_area_positive_breadth_20d",
    "news_missing_rate",
    "news_count_30d",
    "news_warning_score",
    "news_opportunity_score",
    "news_conflict_intensity_30d",
    "news_evidence_quality_score_30d",
    "financial_report_missing_rate",
    "financial_report_event_count",
    "financial_quality_risk_score",
    "financial_surprise_score",
    "financial_disclosure_quality_score",
    "lower_support",
    "chip_concentration",
    "cost_band_width",
    "upper_overhang",
    "winner_rate_pct",
    "neg_winner_rate",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit key decision point sampler.")
    parser.add_argument("--joined-cache", default=str(DEFAULT_JOINED_GT_CACHE_PATH))
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--high-ranker-quantile", type=float, default=HIGH_RANKER_QUANTILE)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_frame(Path(args.joined_cache), high_ranker_quantile=args.high_ranker_quantile)
    daily = build_daily_keyness(frame, high_ranker_quantile=args.high_ranker_quantile)
    aggregate = aggregate_keyness(daily)
    frequency = audit_frequency_overlays(daily)
    rule_outcomes = build_rule_outcomes(aggregate, frequency)

    daily_path = REPORT_DIR / f"{args.output_prefix}_daily.csv"
    aggregate_path = REPORT_DIR / f"{args.output_prefix}_aggregate.csv"
    frequency_path = REPORT_DIR / f"{args.output_prefix}_frequency_overlay.csv"
    outcomes_path = REPORT_DIR / f"{args.output_prefix}_rule_outcomes.jsonl"
    report_path = REPORT_DIR / f"{args.output_prefix}.md"

    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    frequency.to_csv(frequency_path, index=False, encoding="utf-8-sig")
    write_rule_outcomes(outcomes_path, rule_outcomes)
    write_report(report_path, daily, aggregate, frequency, args.high_ranker_quantile)

    print("A股研究Agent")
    print(f"rows={len(frame)}")
    print(f"daily_rows={len(daily)}")
    print(f"aggregate_rows={len(aggregate)}")
    print(f"report={report_path}")
    print(f"rule_outcomes={outcomes_path}")


def load_frame(path: Path, *, high_ranker_quantile: float) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0)
    usecols = [col for col in EX_ANTE_COLUMNS if col in header.columns]
    frame = pd.read_csv(path, usecols=usecols, dtype={"code": str}, low_memory=False)
    frame.columns = [col.lstrip("\ufeff") for col in frame.columns]
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    if "gt_status" in frame:
        frame = frame[frame["gt_status"].astype(str).eq("evaluated")].copy()
    for horizon in HORIZONS:
        frame[f"return_{horizon}d"] = pd.to_numeric(frame.get(f"return_{horizon}d"), errors="coerce")
    frame = frame.dropna(subset=["date", "code", "return_20d"]).copy()
    frame["time_block"] = frame["date"].map(block_for_date)
    frame = frame[frame["time_block"].isin(TIME_BLOCKS)].copy()

    ranker = _portfolio_ranker_details(
        frame,
        preset=DEFAULT_PORTFOLIO_PRESET,
        valid_block="all_walkforward",
        decision_frequency="all_dates",
    )
    frame["rev_chip_score"] = pd.to_numeric(ranker["score"], errors="coerce")
    frame["rev_chip_score_quantile"] = pd.to_numeric(ranker["score_quantile"], errors="coerce")
    frame["portfolio_candidate_pool"] = frame["rev_chip_score_quantile"] >= high_ranker_quantile
    return frame.reset_index(drop=True)


def block_for_date(value: Any) -> str | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    for block, (start, end) in TIME_BLOCKS.items():
        if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
            return block
    return None


def build_daily_keyness(frame: pd.DataFrame, *, high_ranker_quantile: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task_mode in ["portfolio_pool", "single_stock"]:
        work = frame[frame["portfolio_candidate_pool"]].copy() if task_mode == "portfolio_pool" else frame.copy()
        if work.empty:
            continue
        for date, group in work.groupby("date", sort=True):
            rows.append(daily_row(group, task_mode=task_mode, date=str(date), high_ranker_quantile=high_ranker_quantile))
    daily = pd.DataFrame(rows)
    if daily.empty:
        return daily
    daily = assign_key_scores(daily)
    return daily.sort_values(["task_mode", "date"]).reset_index(drop=True)


def daily_row(group: pd.DataFrame, *, task_mode: str, date: str, high_ranker_quantile: float) -> dict[str, Any]:
    score = numeric(group, "rev_chip_score")
    score_rank = score.rank(pct=True, method="average")
    row: dict[str, Any] = {
        "date": date,
        "time_block": str(group["time_block"].iloc[0]),
        "task_mode": task_mode,
        "candidate_rows": int(len(group)),
        "unique_stocks": int(group["code"].nunique()),
        "high_ranker_quantile": high_ranker_quantile if task_mode == "portfolio_pool" else np.nan,
        "score_std": safe_std(score),
        "score_iqr": safe_quantile(score, 0.75) - safe_quantile(score, 0.25),
        "score_top_median_gap": safe_quantile(score, 0.90) - safe_quantile(score, 0.50),
        "score_bottom_median_gap": safe_quantile(score, 0.50) - safe_quantile(score, 0.10),
        "score_rank_dispersion": safe_std(score_rank),
        "multiscale_return_tension": mean_abs_diff(group, "kline_return_3d", "kline_return_60d"),
        "mid_long_return_tension": mean_abs_diff(group, "kline_return_20d", "kline_return_120d"),
        "range_position_tension": mean_abs_diff(group, "kline_range_position_20d", "kline_range_position_120d"),
        "reversal_activity": mean_of(group, ["kline_direction_reversal_rate_20d", "kline_oscillation_cross_count_20d"]),
        "volatility_pressure": mean_of(group, ["kline_volatility_20d", "kline_atr20_pct"]),
        "peer_breadth": mean_of(group, ["peer_group_positive_breadth_20d", "corr_peer_positive_breadth_20d", "tushare_industry_positive_breadth_20d"]),
        "peer_relative_strength": mean_of(group, ["peer_relative_to_group_20d", "corr_peer_relative_return_20d", "tushare_industry_relative_return_20d"]),
        "news_coverage_rate": float((numeric(group, "news_missing_rate").fillna(1.0) < 0.75).mean()),
        "news_conflict_pressure": mean_of(group, ["news_warning_score", "news_conflict_intensity_30d", "peer_group_news_risk_avg"]),
        "news_positive_context": mean_of(group, ["news_opportunity_score", "peer_group_news_opportunity_avg", "news_evidence_quality_score_30d"]),
        "financial_event_rate": float((numeric(group, "financial_report_event_count").fillna(0.0) > 0).mean()),
        "financial_risk_pressure": mean_of(group, ["financial_quality_risk_score", "financial_report_missing_rate"]),
        "financial_positive_context": mean_of(group, ["financial_surprise_score", "financial_disclosure_quality_score"]),
        "chip_support": mean_of(group, ["lower_support", "chip_concentration"]),
        "chip_overhang_pressure": mean_of(group, ["upper_overhang", "cost_band_width", "neg_winner_rate"]),
    }
    row["channel_conflict_pressure"] = nanmean(
        [
            row["volatility_pressure"],
            row["news_conflict_pressure"],
            row["financial_risk_pressure"],
            row["chip_overhang_pressure"],
            -row["peer_breadth"] if not pd.isna(row["peer_breadth"]) else np.nan,
        ]
    )
    row["channel_positive_context"] = nanmean(
        [
            row["peer_breadth"],
            row["peer_relative_strength"],
            row["news_positive_context"],
            row["financial_positive_context"],
            row["chip_support"],
        ]
    )
    for horizon in HORIZONS:
        returns = numeric(group, f"return_{horizon}d")
        row[f"future_return_std_{horizon}d"] = safe_std(returns)
        row[f"future_return_iqr_{horizon}d"] = safe_quantile(returns, 0.75) - safe_quantile(returns, 0.25)
        row[f"pool_avg_return_{horizon}d"] = safe_mean(returns)
        top = group.loc[score.sort_values(ascending=False).head(max(1, math.ceil(len(group) * 0.05))).index]
        top_returns = numeric(top, f"return_{horizon}d")
        row[f"rev_chip_top5_avg_return_{horizon}d"] = safe_mean(top_returns)
        row[f"rev_chip_top5_pool_excess_{horizon}d"] = row[f"rev_chip_top5_avg_return_{horizon}d"] - row[f"pool_avg_return_{horizon}d"]
        row[f"rev_chip_top5_positive_rate_{horizon}d"] = safe_positive_rate(top_returns)
    return row


def assign_key_scores(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    components = [
        "score_std",
        "score_iqr",
        "score_top_median_gap",
        "score_rank_dispersion",
        "multiscale_return_tension",
        "mid_long_return_tension",
        "range_position_tension",
        "reversal_activity",
        "volatility_pressure",
        "channel_conflict_pressure",
        "channel_positive_context",
    ]
    for task_mode, idx in out.groupby("task_mode").groups.items():
        comp_z = []
        for col in components:
            values = pd.to_numeric(out.loc[idx, col], errors="coerce")
            z = zscore(values)
            out.loc[idx, f"{col}_z"] = z
            comp_z.append(z)
        raw = pd.concat(comp_z, axis=1).mean(axis=1)
        out.loc[idx, "key_score"] = raw
        out.loc[idx, "key_score_pct"] = raw.rank(pct=True, method="average")
    out["sampler_bucket"] = np.select(
        [
            out["key_score_pct"].ge(0.90),
            out["key_score_pct"].ge(0.80),
            out["key_score_pct"].ge(0.60),
        ],
        ["key_top10", "key_top20", "watch_mid40"],
        default="ordinary_bottom60",
    )
    return out


def aggregate_keyness(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if daily.empty:
        return pd.DataFrame()
    for task_mode, task_group in daily.groupby("task_mode", sort=True):
        thresholds = {
            horizon: {
                "dispersion_q70": safe_quantile(task_group[f"future_return_std_{horizon}d"], 0.70),
                "impact_q70": safe_quantile(task_group[f"rev_chip_top5_pool_excess_{horizon}d"].abs(), 0.70),
            }
            for horizon in HORIZONS
        }
        for horizon in HORIZONS:
            high_impact = (
                pd.to_numeric(task_group[f"future_return_std_{horizon}d"], errors="coerce").ge(thresholds[horizon]["dispersion_q70"])
                | pd.to_numeric(task_group[f"rev_chip_top5_pool_excess_{horizon}d"], errors="coerce").abs().ge(thresholds[horizon]["impact_q70"])
            )
            all_high_count = int(high_impact.sum())
            for bucket, group in task_group.groupby("sampler_bucket", sort=True):
                idx = group.index
                rows.append(
                    {
                        "task_mode": task_mode,
                        "horizon": f"{horizon}d",
                        "sampler_bucket": bucket,
                        "dates": int(len(group)),
                        "date_share": round(float(len(group) / len(task_group)), 6),
                        "high_impact_dates": int(high_impact.loc[idx].sum()),
                        "high_impact_capture_rate": round(float(high_impact.loc[idx].sum() / max(1, all_high_count)), 6),
                        "high_impact_precision": round(float(high_impact.loc[idx].mean()), 6),
                        "avg_future_return_std": mean(group, f"future_return_std_{horizon}d"),
                        "avg_abs_top5_pool_excess": mean_abs(group, f"rev_chip_top5_pool_excess_{horizon}d"),
                        "avg_top5_positive_rate": mean(group, f"rev_chip_top5_positive_rate_{horizon}d"),
                        "avg_key_score_pct": mean(group, "key_score_pct"),
                        "research_only": True,
                        "not_investment_instruction": True,
                    }
                )
            top20 = task_group[task_group["key_score_pct"].ge(0.80)]
            idx = top20.index
            rows.append(
                {
                    "task_mode": task_mode,
                    "horizon": f"{horizon}d",
                    "sampler_bucket": "key_top20_combined",
                    "dates": int(len(top20)),
                    "date_share": round(float(len(top20) / len(task_group)), 6),
                    "high_impact_dates": int(high_impact.loc[idx].sum()),
                    "high_impact_capture_rate": round(float(high_impact.loc[idx].sum() / max(1, all_high_count)), 6),
                    "high_impact_precision": round(float(high_impact.loc[idx].mean()), 6) if len(top20) else np.nan,
                    "avg_future_return_std": mean(top20, f"future_return_std_{horizon}d"),
                    "avg_abs_top5_pool_excess": mean_abs(top20, f"rev_chip_top5_pool_excess_{horizon}d"),
                    "avg_top5_positive_rate": mean(top20, f"rev_chip_top5_positive_rate_{horizon}d"),
                    "avg_key_score_pct": mean(top20, "key_score_pct"),
                    "research_only": True,
                    "not_investment_instruction": True,
                }
            )
    return pd.DataFrame(rows)


def audit_frequency_overlays(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if daily.empty:
        return pd.DataFrame()
    date_ts = pd.to_datetime(daily["date"], errors="coerce")
    masks = {
        "all_dates": pd.Series(True, index=daily.index),
        "weekly_tuesday": date_ts.dt.weekday.eq(1),
        "weekly_friday": date_ts.dt.weekday.eq(4),
        "every_2_weeks": date_ts.dt.isocalendar().week.astype(int).mod(2).eq(0),
        "key_top20_all_dates": daily["key_score_pct"].ge(0.80),
        "weekly_tuesday_or_key_top20": date_ts.dt.weekday.eq(1) | daily["key_score_pct"].ge(0.80),
        "weekly_friday_or_key_top20": date_ts.dt.weekday.eq(4) | daily["key_score_pct"].ge(0.80),
        "every_2_weeks_or_key_top20": date_ts.dt.isocalendar().week.astype(int).mod(2).eq(0) | daily["key_score_pct"].ge(0.80),
    }
    for task_mode, task_group in daily.groupby("task_mode", sort=True):
        base_idx = task_group.index
        for horizon in HORIZONS:
            dispersion_q70 = safe_quantile(task_group[f"future_return_std_{horizon}d"], 0.70)
            impact_q70 = safe_quantile(task_group[f"rev_chip_top5_pool_excess_{horizon}d"].abs(), 0.70)
            high_impact = (
                pd.to_numeric(task_group[f"future_return_std_{horizon}d"], errors="coerce").ge(dispersion_q70)
                | pd.to_numeric(task_group[f"rev_chip_top5_pool_excess_{horizon}d"], errors="coerce").abs().ge(impact_q70)
            )
            total_high = int(high_impact.sum())
            for name, mask in masks.items():
                selected_idx = base_idx[mask.loc[base_idx].fillna(False)]
                selected = task_group.loc[selected_idx]
                rows.append(
                    {
                        "task_mode": task_mode,
                        "horizon": f"{horizon}d",
                        "decision_schedule": name,
                        "dates": int(len(selected)),
                        "date_share": round(float(len(selected) / max(1, len(task_group))), 6),
                        "high_impact_capture_rate": round(float(high_impact.loc[selected_idx].sum() / max(1, total_high)), 6),
                        "high_impact_precision": round(float(high_impact.loc[selected_idx].mean()), 6) if len(selected) else np.nan,
                        "avg_abs_top5_pool_excess": mean_abs(selected, f"rev_chip_top5_pool_excess_{horizon}d"),
                        "avg_future_return_std": mean(selected, f"future_return_std_{horizon}d"),
                        "avg_key_score_pct": mean(selected, "key_score_pct"),
                        "research_only": True,
                        "not_investment_instruction": True,
                    }
                )
    return pd.DataFrame(rows)


def build_rule_outcomes(aggregate: pd.DataFrame, frequency: pd.DataFrame) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    if aggregate.empty:
        return outcomes
    for task_mode in ["portfolio_pool", "single_stock"]:
        task_agg = aggregate[
            (aggregate["task_mode"].eq(task_mode))
            & (aggregate["horizon"].eq("20d"))
            & (aggregate["sampler_bucket"].eq("key_top20_combined"))
        ]
        task_freq = frequency[
            (frequency["task_mode"].eq(task_mode))
            & (frequency["horizon"].eq("20d"))
            & (frequency["decision_schedule"].eq("weekly_tuesday_or_key_top20"))
        ]
        if task_agg.empty:
            continue
        row = task_agg.iloc[0]
        freq_row = task_freq.iloc[0] if not task_freq.empty else None
        capture = float(row.get("high_impact_capture_rate", 0.0) or 0.0)
        precision = float(row.get("high_impact_precision", 0.0) or 0.0)
        date_share = float(row.get("date_share", 0.0) or 0.0)
        status = "observe_training_sampler_candidate"
        if capture >= 0.25 and precision >= 0.50 and date_share <= 0.25:
            status = "accepted_training_sampler_candidate"
        outcome = sanitize_quant_tool_outcome(
            {
                "tool_id": "decision_keypoint_sampler_v1",
                "tool_version": "2026-06-28",
                "task_mode": task_mode,
                "policy_profile": "training_sampler_not_alpha",
                "policy_status": status,
                "decision_frequency": "scheduled_dates_plus_key_top20",
                "feature_group": "multiscale_kline_peer_chip_news_financial_conflict",
                "selection_mode": "key_score_top20_with_ordinary_controls",
                "cap_pct": 0.20,
                "tool_grade": "observe",
                "score": round(capture, 6),
                "confidence": round(precision, 6),
                "risk_tier": "token_allocation_tool_only",
                "action_hint": "use_for_training_sample_mix_only",
                "usable_in_agent_default": False,
                "top_features": [
                    "score_dispersion",
                    "multiscale_return_tension",
                    "channel_conflict_pressure",
                    "chip_support_overhang",
                    "peer_breadth",
                ],
                "required_confirmation": [
                    "leakage_audit_pass",
                    "ordinary_control_dates_kept",
                    "task_mode_separate_metrics",
                    "DS_round_reports_bad_exposure_and_missed_positive",
                ],
                "counter_evidence": [
                    "key_score_is_sampling_priority_not_return_prediction",
                    "do_not_change_user_grade_from_this_tool_alone",
                ],
                "promotion_status": status,
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
        outcome["source_ref_ids"] = [
            "reports/date_generalization/decision_point_keyness_v1.md",
            "reports/date_generalization/decision_point_keyness_v1_aggregate.csv",
        ]
        if freq_row is not None:
            outcome["missing_flags"] = [f"weekly_tuesday_or_key_top20_date_share={freq_row.get('date_share')}"]
        outcomes.append(outcome)
    return outcomes


def write_rule_outcomes(path: Path, outcomes: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for outcome in outcomes:
            handle.write(json.dumps(outcome, ensure_ascii=False, sort_keys=True) + "\n")


def write_report(path: Path, daily: pd.DataFrame, aggregate: pd.DataFrame, frequency: pd.DataFrame, high_ranker_quantile: float) -> None:
    lines = [
        "# Decision Point Keyness Audit v1",
        "",
        "本报告只用于 A 股研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Purpose",
        "",
        "本轮目标是把不同决策频率训练拆成两层：基础日程负责稳定覆盖，关键决策点负责提高 Agent 反思训练的信息密度。脚本不调用 DeepSeek，不读取 API key。未来收益只用于离线验证关键点筛选器是否抓住高影响日期，不进入 Agent evidence 或 rule_outcomes。",
        "",
        "## Sampler",
        "",
        f"- 组合候选池：`rev_plus_chip_core score_quantile >= {high_ranker_quantile:.2f}`。",
        "- 单支候选池：全 evaluated stock-date。",
        "- key_score 输入：ranker 分数分散度、多尺度 K 线张力、区间位置张力、反转活跃度、波动压力、新闻/财报/筹码/同行冲突和正向上下文。",
        "- key_score 不使用 `return_5d/10d/20d`；这些字段只用于本报告的离线高影响验证。",
        "",
        "## Bucket Audit",
        "",
        markdown_table(
            aggregate[
                aggregate["sampler_bucket"].isin(["key_top20_combined", "ordinary_bottom60"])
                & aggregate["horizon"].isin(["5d", "10d", "20d"])
            ],
            [
                "task_mode",
                "horizon",
                "sampler_bucket",
                "dates",
                "date_share",
                "high_impact_capture_rate",
                "high_impact_precision",
                "avg_abs_top5_pool_excess",
                "avg_future_return_std",
            ],
            max_rows=24,
        ),
        "",
        "## Frequency Overlay",
        "",
        markdown_table(
            frequency[
                frequency["decision_schedule"].isin(
                    [
                        "weekly_tuesday",
                        "weekly_friday",
                        "every_2_weeks",
                        "key_top20_all_dates",
                        "weekly_tuesday_or_key_top20",
                        "every_2_weeks_or_key_top20",
                    ]
                )
                & frequency["horizon"].eq("20d")
            ],
            [
                "task_mode",
                "decision_schedule",
                "dates",
                "date_share",
                "high_impact_capture_rate",
                "high_impact_precision",
                "avg_abs_top5_pool_excess",
            ],
            max_rows=40,
        ),
        "",
        "## Decision",
        "",
        "- `decision_keypoint_sampler_v1` 只能作为手工 key_score baseline / 诊断工具，不是收益预测器，也不得改变用户分级。",
        "- 本轮手工 key_score 对组合 20 日高影响日期捕捉偏弱，不能直接采用“60% key_top20 + 40% ordinary controls”。",
        "- 快速/日频训练不应全量跑所有日期；应先用 walk-forward ML 关键点采样器验证，再保留普通点作为反过拟合对照。",
        "- 如果后续 DS round 中关键点采样只提高波动、不降低 bad exposure 或 missed-positive，则降级为纯诊断工具。",
        "",
        "## Artifacts",
        "",
        "- `reports/date_generalization/decision_point_keyness_v1_daily.csv`",
        "- `reports/date_generalization/decision_point_keyness_v1_aggregate.csv`",
        "- `reports/date_generalization/decision_point_keyness_v1_frequency_overlay.csv`",
        "- `reports/date_generalization/decision_point_keyness_v1_rule_outcomes.jsonl`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def numeric(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[col], errors="coerce")


def safe_mean(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce")
    return round(float(vals.mean()), 6) if vals.notna().any() else np.nan


def safe_std(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce")
    return round(float(vals.std()), 6) if vals.notna().sum() > 1 else 0.0


def safe_quantile(values: Any, q: float) -> float:
    vals = pd.to_numeric(values, errors="coerce") if not isinstance(values, pd.Series) else pd.to_numeric(values, errors="coerce")
    return float(vals.quantile(q)) if vals.notna().any() else np.nan


def safe_positive_rate(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    return round(float((vals > 0).mean()), 6) if len(vals) else np.nan


def mean(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame:
        return np.nan
    return safe_mean(frame[col])


def mean_abs(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame:
        return np.nan
    vals = pd.to_numeric(frame[col], errors="coerce").abs()
    return safe_mean(vals)


def mean_of(frame: pd.DataFrame, cols: list[str]) -> float:
    values = [numeric(frame, col) for col in cols if col in frame]
    if not values:
        return np.nan
    merged = pd.concat(values, axis=1)
    return safe_mean(merged.mean(axis=1))


def mean_abs_diff(frame: pd.DataFrame, left: str, right: str) -> float:
    if left not in frame or right not in frame:
        return np.nan
    return safe_mean((numeric(frame, left) - numeric(frame, right)).abs())


def nanmean(values: list[float]) -> float:
    vals = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    return round(float(vals.mean()), 6) if len(vals) else np.nan


def zscore(values: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    std = float(vals.std())
    if std <= 0 or math.isnan(std):
        return pd.Series(0.0, index=values.index)
    return ((vals - float(vals.mean())) / std).fillna(0.0)


def markdown_table(frame: pd.DataFrame, cols: list[str], *, max_rows: int) -> str:
    if frame.empty:
        return "_No rows._"
    show = frame[[col for col in cols if col in frame]].head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_numeric_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    header = "| " + " | ".join(show.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(show.columns)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in show.astype(str).to_numpy()]
    return "\n".join([header, sep, *body])


if __name__ == "__main__":
    main()
