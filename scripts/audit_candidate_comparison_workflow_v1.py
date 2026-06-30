"""P1 candidate-comparison workflow audit.

This audit evaluates the user-facing task "I have 2-20 candidate stocks;
which ones deserve more research attention?" It is deliberately different
from the portfolio TopK audit:

- candidate pools are sampled by stable hashes, not by future returns or by
  the evaluated score;
- same-sector and cross-sector tasks are evaluated separately;
- labels/forward returns are used only in offline metrics and never written
  to the sample plan for agent inference.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_full_channel_ablation_round import GT_SOURCES  # noqa: E402
from src.agent_training.dual_mode_round import (  # noqa: E402
    DEFAULT_CHIP_CORE_FEATURES_PATH,
    DEFAULT_CORR_PEER_FEATURES_PATH,
    DEFAULT_KLINE_FEATURES_PATH,
    DEFAULT_TUSHARE_PEER_FEATURES_PATH,
    TIME_BLOCKS,
    load_ground_truth,
)

REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_PREFIX = "candidate_comparison_workflow_v1"

FUTURE_RESULT_COLUMNS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "fwd_ret_20d",
    "pool_excess_20d",
    "rank_pct_in_date",
    "rank_pct_in_industry_date",
    "gt_status",
    "gt_pass",
    "rule_outcome_label",
}

P1_DEFAULT_SCORE = "p1_default_selector_v1"

SCORE_COLUMNS = [
    P1_DEFAULT_SCORE,
    "candidate_context_blend_v1",
    "rev_chip_core",
    "original_total_score",
    "single_watch_proxy",
    "rank_avg_rev_watch",
    "cross_low_risk_anchor_v1",
    "cross_peer_confirmed_anchor_v1",
    "cross_news_financial_confirmed_v1",
    "cross_balanced_quality_anchor_v1",
    "cross_h2026_defensive_anchor_v1",
    "ml_ridge_walkforward_v1",
    "ml_hgbr_walkforward_v1",
]

SAFE_AGENT_FEATURES = [
    "kline_return_5d",
    "kline_return_20d",
    "kline_return_60d",
    "kline_drawdown_20d",
    "kline_drawdown_60d",
    "kline_volatility_ratio_3_20",
    "kline_rsi14",
    "kline_mean_reversion_z20",
    "corr_peer_avg_return_20d",
    "corr_peer_relative_return_20d",
    "corr_peer_positive_breadth_20d",
    "tushare_industry_relative_return_20d",
    "tushare_industry_positive_breadth_20d",
    "tushare_industry_news_attention_gap",
    "tushare_area_relative_return_20d",
    "lower_support",
    "chip_concentration",
    "cost_band_width",
    "upper_overhang",
    "winner_rate_pct",
    "news_warning_score",
    "news_opportunity_score",
    "news_missing_rate",
    "policy_background_score",
    "official_confirmation_score",
    "announcement_materiality_score",
    "financial_report_join_status",
    "financial_report_event_count",
    "financial_quality_risk_score",
    "financial_surprise_score",
    "financial_disclosure_quality_score",
    "financial_report_missing_rate",
    "financial_report_event_types",
    "triggered_skills",
    "data_gaps",
]


@dataclass(frozen=True)
class AuditConfig:
    candidate_size: int = 8
    repeats: int = 3
    industries_per_date: int = 3
    decision_frequency: str = "every_2_weeks"
    min_industry_size: int = 12
    max_dates_per_block: int = 0
    output_prefix: str = DEFAULT_PREFIX


def stable_hash_int(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def assign_time_block(frame: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(frame["date"], errors="coerce")
    out = pd.Series("OUT_OF_SCOPE", index=frame.index, dtype="object")
    for block, (start, end) in TIME_BLOCKS.items():
        mask = dates.between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")
        out.loc[mask] = block
    return out


def apply_decision_frequency(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frame.empty or frequency in {"", "all_dates"}:
        return frame.copy()
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if frequency == "weekly_friday":
        return frame[dates.dt.weekday.eq(4)].copy()
    if frequency == "weekly_tuesday":
        return frame[dates.dt.weekday.eq(1)].copy()
    if frequency == "every_2_weeks":
        return frame[dates.dt.isocalendar().week.astype(int).mod(2).eq(0)].copy()
    raise ValueError(f"unknown decision_frequency: {frequency}")


def cross_section_z(frame: pd.DataFrame, field: str) -> pd.Series:
    values = pd.to_numeric(frame.get(field, pd.Series(np.nan, index=frame.index)), errors="coerce")

    def _z(group: pd.Series) -> pd.Series:
        std = float(group.std())
        if math.isnan(std) or std <= 0 or len(group) < 5:
            return pd.Series(0.0, index=group.index)
        return (group - float(group.mean())) / std

    return values.groupby(frame["date"].astype(str), sort=False).transform(_z).fillna(0.0)


def rev_chip_core_score(frame: pd.DataFrame) -> pd.Series:
    """Fast implementation of the production rev+chip_core formula."""
    reversal_fields = [c for c in ["kline_return_20d", "kline_return_60d", "corr_peer_avg_return_20d"] if c in frame]
    if reversal_fields:
        reversal_raw = -sum(cross_section_z(frame, c) for c in reversal_fields) / len(reversal_fields)
        reversal_component = _z_series_by_date(frame, reversal_raw)
        parts = [reversal_component]
    else:
        parts = []
    for field in ["lower_support", "chip_concentration", "cost_band_width", "upper_overhang", "winner_rate_pct", "neg_winner_rate"]:
        if field in frame:
            parts.append(cross_section_z(frame, field))
    if not parts:
        return pd.Series(0.0, index=frame.index)
    return sum(parts) / len(parts)


def _z_series_by_date(frame: pd.DataFrame, values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")

    def _z(group: pd.Series) -> pd.Series:
        std = float(group.std())
        if math.isnan(std) or std <= 0 or len(group) < 5:
            return pd.Series(0.0, index=group.index)
        return (group - float(group.mean())) / std

    return values.groupby(frame["date"].astype(str), sort=False).transform(_z).fillna(0.0)


def add_candidate_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["rev_chip_core"] = rev_chip_core_score(out)
    out["rev_chip_core_quantile"] = out.groupby("date")["rev_chip_core"].rank(method="average", pct=True)

    out["original_total_score"] = cross_section_z(out, "total_score") if "total_score" in out else 0.0

    # A conservative single-stock proxy: opportunity ranker minus obvious risk/gap pressure.
    risk_pressure = pd.Series(0.0, index=out.index)
    for field, weight in [
        ("news_warning_score", 0.20),
        ("financial_quality_risk_score", 0.18),
        ("news_missing_rate", 0.08),
        ("financial_report_missing_rate", 0.08),
        ("kline_volatility_ratio_3_20", 0.05),
    ]:
        if field in out:
            risk_pressure += cross_section_z(out, field).clip(lower=0.0) * weight
    out["single_watch_proxy"] = out["rev_chip_core"] - risk_pressure

    # P1 context blend: ranker is still the backbone; orthogonal channels only
    # adjust it softly so this remains robust across sparse news/financial data.
    blend = 0.62 * out["rev_chip_core"]
    for field, weight in [
        ("tushare_industry_positive_breadth_20d", 0.09),
        ("tushare_industry_relative_return_20d", 0.06),
        ("tushare_area_relative_return_20d", 0.03),
        ("news_opportunity_score", 0.05),
        ("policy_background_score", 0.03),
        ("official_confirmation_score", 0.03),
        ("announcement_materiality_score", 0.03),
        ("financial_surprise_score", 0.04),
        ("financial_disclosure_quality_score", 0.03),
    ]:
        if field in out:
            blend += weight * cross_section_z(out, field)
    for field, weight in [
        ("news_warning_score", 0.08),
        ("financial_quality_risk_score", 0.07),
        ("news_missing_rate", 0.03),
        ("financial_report_missing_rate", 0.03),
    ]:
        if field in out:
            blend -= weight * cross_section_z(out, field).clip(lower=0.0)
    out["candidate_context_blend_v1"] = blend
    return out


def load_candidate_frame() -> pd.DataFrame:
    frame = load_ground_truth(
        GT_SOURCES,
        kline_features_path=DEFAULT_KLINE_FEATURES_PATH,
        corr_peer_features_path=DEFAULT_CORR_PEER_FEATURES_PATH,
        tushare_peer_features_path=DEFAULT_TUSHARE_PEER_FEATURES_PATH,
        chip_core_features_path=DEFAULT_CHIP_CORE_FEATURES_PATH,
    )
    frame = frame.copy()
    frame["code"] = frame["code"].astype(str).str.extract(r"(\d{6})", expand=False).fillna(frame["code"].astype(str)).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame["return_20d"] = pd.to_numeric(frame["return_20d"], errors="coerce")
    frame["time_block"] = assign_time_block(frame)
    frame = frame[frame["time_block"].ne("OUT_OF_SCOPE") & frame["return_20d"].notna()].copy()
    frame["tushare_industry"] = frame.get("tushare_industry", "UNKNOWN").fillna("UNKNOWN").astype(str)
    frame.loc[frame["tushare_industry"].isin(["", "nan", "None", "NA"]), "tushare_industry"] = "UNKNOWN"
    frame["tushare_area"] = frame.get("tushare_area", "UNKNOWN").fillna("UNKNOWN").astype(str)
    return add_candidate_scores(frame)


def choose_block_dates(block_frame: pd.DataFrame, cfg: AuditConfig) -> list[str]:
    dates = sorted(block_frame["date"].dropna().astype(str).unique())
    if cfg.max_dates_per_block <= 0 or len(dates) <= cfg.max_dates_per_block:
        return dates
    # Stable spacing, not score/return based.
    positions = np.linspace(0, len(dates) - 1, cfg.max_dates_per_block).round().astype(int)
    return [dates[int(pos)] for pos in sorted(set(positions))]


def build_candidate_groups(frame: pd.DataFrame, cfg: AuditConfig) -> pd.DataFrame:
    work = apply_decision_frequency(frame, cfg.decision_frequency)
    rows: list[dict[str, Any]] = []
    group_id = 0
    for block in TIME_BLOCKS:
        block_frame = work[work["time_block"].eq(block)].copy()
        for date in choose_block_dates(block_frame, cfg):
            day = block_frame[block_frame["date"].eq(date)].copy()
            if day.empty:
                continue
            for seed in range(cfg.repeats):
                same_rows = _same_sector_groups(day, cfg, seed=seed)
                cross_rows = _cross_sector_group(day, cfg, seed=seed)
                for scenario, selected in [*same_rows, *cross_rows]:
                    if selected.empty:
                        continue
                    group_id += 1
                    candidate_codes = ";".join(selected["code"].astype(str).tolist())
                    candidate_names = ";".join(selected.get("name", selected["code"]).astype(str).tolist())
                    for _, row in selected.iterrows():
                        rows.append(
                            {
                                "comparison_group_id": f"CMP{group_id:06d}",
                                "comparison_scenario": scenario,
                                "repeat_seed": seed,
                                "time_block": block,
                                "date": date,
                                "candidate_count": int(len(selected)),
                                "candidate_codes": candidate_codes,
                                "candidate_names": candidate_names,
                                "industry_context": _industry_context(selected, scenario),
                                "code": row["code"],
                                "name": row.get("name", row["code"]),
                                "tushare_industry": row.get("tushare_industry", "UNKNOWN"),
                                "tushare_area": row.get("tushare_area", "UNKNOWN"),
                                "return_20d": row["return_20d"],
                                **{col: row.get(col, np.nan) for col in SCORE_COLUMNS},
                                **{col: row.get(col, np.nan) for col in SAFE_AGENT_FEATURES if col in selected.columns},
                            }
                        )
    return pd.DataFrame(rows)


def _same_sector_groups(day: pd.DataFrame, cfg: AuditConfig, *, seed: int) -> list[tuple[str, pd.DataFrame]]:
    valid = day[day["tushare_industry"].ne("UNKNOWN")].copy()
    counts = valid.groupby("tushare_industry").size().rename("n").reset_index()
    counts = counts[counts["n"].ge(max(cfg.min_industry_size, cfg.candidate_size))]
    if counts.empty:
        return []
    counts["_order"] = counts["tushare_industry"].map(lambda x: stable_hash_int(day["date"].iloc[0], x, seed, "same_sector_industry"))
    counts = counts.sort_values(["n", "_order"], ascending=[False, True]).head(cfg.industries_per_date)
    groups: list[tuple[str, pd.DataFrame]] = []
    for industry in counts["tushare_industry"].tolist():
        pool = valid[valid["tushare_industry"].eq(industry)].copy()
        selected = stable_sample(pool, cfg.candidate_size, seed=seed, salt=f"same_sector:{industry}")
        if len(selected) >= min(cfg.candidate_size, 3):
            groups.append(("same_sector", selected))
    return groups


def _cross_sector_group(day: pd.DataFrame, cfg: AuditConfig, *, seed: int) -> list[tuple[str, pd.DataFrame]]:
    valid = day[day["tushare_industry"].ne("UNKNOWN")].copy()
    if valid.empty:
        return []
    counts = valid.groupby("tushare_industry").size().rename("n").reset_index()
    counts = counts[counts["n"].ge(3)]
    if len(counts) < min(cfg.candidate_size, 3):
        return []
    counts["_order"] = counts["tushare_industry"].map(lambda x: stable_hash_int(day["date"].iloc[0], x, seed, "cross_sector_industry"))
    industries = counts.sort_values(["_order", "tushare_industry"]).head(cfg.candidate_size)["tushare_industry"].tolist()
    selected_parts = []
    for industry in industries:
        pool = valid[valid["tushare_industry"].eq(industry)].copy()
        selected_parts.append(stable_sample(pool, 1, seed=seed, salt=f"cross_sector:{industry}"))
    selected = pd.concat(selected_parts, ignore_index=False) if selected_parts else pd.DataFrame()
    return [("cross_sector", selected)] if len(selected) >= min(cfg.candidate_size, 3) else []


def stable_sample(frame: pd.DataFrame, n: int, *, seed: int, salt: str) -> pd.DataFrame:
    out = frame.copy()
    out["_sample_order"] = [
        stable_hash_int(salt, seed, row.date, row.code, row.tushare_industry)
        for row in out[["date", "code", "tushare_industry"]].itertuples(index=False)
    ]
    return out.sort_values(["_sample_order", "code"]).head(n).drop(columns=["_sample_order"])


def _industry_context(group: pd.DataFrame, scenario: str) -> str:
    industries = group["tushare_industry"].fillna("UNKNOWN").astype(str).value_counts()
    if scenario == "same_sector":
        return industries.index[0] if len(industries) else "UNKNOWN"
    return ";".join(industries.index[: min(5, len(industries))].tolist())


def evaluate_groups(candidate_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, Any]] = []
    if candidate_rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    candidate_rows = ensure_task_default_score(candidate_rows)
    for group_id, group in candidate_rows.groupby("comparison_group_id", sort=True):
        group = group.copy()
        returns = pd.to_numeric(group["return_20d"], errors="coerce")
        if returns.notna().sum() < 3:
            continue
        base_row = _evaluate_equal_baseline(group_id, group, returns)
        detail_rows.append(base_row)
        for score_name in SCORE_COLUMNS:
            detail_rows.append(_evaluate_scored_group(group_id, group, returns, score_name))
    detail = pd.DataFrame(detail_rows)
    aggregate = aggregate_metrics(detail)
    return detail, aggregate


def ensure_task_default_score(
    candidate_rows: pd.DataFrame,
    *,
    same_sector_score: str = "rev_chip_core",
    cross_sector_score: str = "rank_avg_rev_watch",
) -> pd.DataFrame:
    out = candidate_rows.copy()
    if "rank_avg_rev_watch" not in out.columns or not pd.to_numeric(out.get("rank_avg_rev_watch"), errors="coerce").notna().any():
        out["rank_avg_rev_watch"] = _group_z(out, "rev_chip_core") + _group_z(out, "single_watch_proxy")

    fallback = pd.Series(-999.0, index=out.index)
    watch_score = pd.to_numeric(out.get("single_watch_proxy", fallback), errors="coerce").fillna(-999.0)
    same_score = pd.to_numeric(out.get(same_sector_score, fallback), errors="coerce").fillna(watch_score)
    cross_score = pd.to_numeric(out.get(cross_sector_score, fallback), errors="coerce").fillna(watch_score)
    out[P1_DEFAULT_SCORE] = np.where(out["comparison_scenario"].astype(str).eq("cross_sector"), cross_score, same_score)
    return out


def _group_z(frame: pd.DataFrame, field: str) -> pd.Series:
    values = pd.to_numeric(frame.get(field, pd.Series(np.nan, index=frame.index)), errors="coerce")

    def _z(group: pd.Series) -> pd.Series:
        std = float(group.std())
        if math.isnan(std) or std <= 0 or len(group) < 2:
            return pd.Series(0.0, index=group.index)
        return (group - float(group.mean())) / std

    return values.groupby(frame["comparison_group_id"].astype(str), sort=False).transform(_z).fillna(0.0)


def _evaluate_equal_baseline(group_id: str, group: pd.DataFrame, returns: pd.Series) -> dict[str, Any]:
    group_mean = float(returns.mean())
    top2_mean = group_mean
    return {
        "comparison_group_id": group_id,
        "comparison_scenario": group["comparison_scenario"].iloc[0],
        "repeat_seed": int(group["repeat_seed"].iloc[0]),
        "time_block": group["time_block"].iloc[0],
        "date": group["date"].iloc[0],
        "score_name": "equal_or_random_baseline",
        "candidate_count": int(len(group)),
        "rank_ic": np.nan,
        "top1_code": "random_expected",
        "top1_name": "random_expected",
        "top1_return_20d": group_mean,
        "top2_mean_return_20d": top2_mean,
        "group_mean_return_20d": group_mean,
        "group_positive_rate": float((returns > 0).mean()),
        "top1_excess_20d": 0.0,
        "top2_excess_20d": 0.0,
        "top1_positive": group_mean > 0,
        "top2_positive_rate": float((returns > 0).mean()),
        "top1_is_best": False,
        "top1_is_worst": False,
        "top1_regret_vs_best": float(returns.max() - group_mean),
        "bottom2_loss_gt5_rate": float((returns <= -5).mean()),
        "research_only": True,
        "not_investment_instruction": True,
    }


def _evaluate_scored_group(group_id: str, group: pd.DataFrame, returns: pd.Series, score_name: str) -> dict[str, Any]:
    scores = pd.to_numeric(group.get(score_name), errors="coerce")
    eval_frame = group.assign(_score=scores, _ret=returns).dropna(subset=["_score", "_ret"])
    if eval_frame.empty:
        rank_ic = np.nan
        selected = group.iloc[[0]].copy()
        top2 = group.head(2).copy()
    else:
        rank_ic = eval_frame["_score"].rank().corr(eval_frame["_ret"].rank(), method="pearson") if eval_frame["_score"].nunique() > 1 else np.nan
        selected = eval_frame.sort_values(["_score", "code"], ascending=[False, True]).head(1)
        top2 = eval_frame.sort_values(["_score", "code"], ascending=[False, True]).head(min(2, len(eval_frame)))
    top1 = selected.iloc[0]
    top1_ret = float(top1["_ret"] if "_ret" in top1 else top1["return_20d"])
    top2_returns = pd.to_numeric(top2["_ret"] if "_ret" in top2 else top2["return_20d"], errors="coerce")
    bottom2 = eval_frame.sort_values(["_score", "code"], ascending=[True, True]).head(min(2, len(eval_frame))) if not eval_frame.empty else group.head(0)
    bottom2_returns = pd.to_numeric(bottom2["_ret"] if "_ret" in bottom2 else pd.Series(dtype=float), errors="coerce")
    group_mean = float(returns.mean())
    return {
        "comparison_group_id": group_id,
        "comparison_scenario": group["comparison_scenario"].iloc[0],
        "repeat_seed": int(group["repeat_seed"].iloc[0]),
        "time_block": group["time_block"].iloc[0],
        "date": group["date"].iloc[0],
        "score_name": score_name,
        "candidate_count": int(len(group)),
        "rank_ic": float(rank_ic) if not pd.isna(rank_ic) else np.nan,
        "top1_code": str(top1["code"]),
        "top1_name": str(top1.get("name", top1["code"])),
        "top1_return_20d": top1_ret,
        "top2_mean_return_20d": float(top2_returns.mean()) if top2_returns.notna().any() else np.nan,
        "group_mean_return_20d": group_mean,
        "group_positive_rate": float((returns > 0).mean()),
        "top1_excess_20d": top1_ret - group_mean,
        "top2_excess_20d": float(top2_returns.mean() - group_mean) if top2_returns.notna().any() else np.nan,
        "top1_positive": top1_ret > 0,
        "top2_positive_rate": float((top2_returns > 0).mean()) if top2_returns.notna().any() else np.nan,
        "top1_is_best": bool(top1_ret >= returns.max()),
        "top1_is_worst": bool(top1_ret <= returns.min()),
        "top1_regret_vs_best": float(returns.max() - top1_ret),
        "bottom2_loss_gt5_rate": float((bottom2_returns <= -5).mean()) if bottom2_returns.notna().any() else np.nan,
        "research_only": True,
        "not_investment_instruction": True,
    }


def aggregate_metrics(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["comparison_scenario", "score_name", "time_block"]
    for key_values, group in detail.groupby(keys, sort=True):
        rows.append(_aggregate_row(key_values, group))
    for key_values, group in detail.groupby(["comparison_scenario", "score_name"], sort=True):
        rows.append(_aggregate_row((key_values[0], key_values[1], "ALL"), group))
    return pd.DataFrame(rows).sort_values(["comparison_scenario", "time_block", "score_name"]).reset_index(drop=True)


def _aggregate_row(key_values: tuple[Any, ...], group: pd.DataFrame) -> dict[str, Any]:
    scenario, score_name, block = key_values
    rank_ic = pd.to_numeric(group["rank_ic"], errors="coerce")
    return {
        "comparison_scenario": scenario,
        "score_name": score_name,
        "time_block": block,
        "n_groups": int(group["comparison_group_id"].nunique()),
        "avg_candidate_count": round(float(pd.to_numeric(group["candidate_count"], errors="coerce").mean()), 2),
        "mean_rank_ic": round(float(rank_ic.mean()), 6) if rank_ic.notna().any() else np.nan,
        "rank_ic_positive_rate": round(float((rank_ic.dropna() > 0).mean()), 6) if rank_ic.notna().any() else np.nan,
        "top1_excess_mean": round(float(pd.to_numeric(group["top1_excess_20d"], errors="coerce").mean()), 6),
        "top2_excess_mean": round(float(pd.to_numeric(group["top2_excess_20d"], errors="coerce").mean()), 6),
        "top1_positive_rate": round(float(group["top1_positive"].astype(bool).mean()), 6),
        "top2_positive_rate": round(float(pd.to_numeric(group["top2_positive_rate"], errors="coerce").mean()), 6),
        "top1_best_rate": round(float(group["top1_is_best"].astype(bool).mean()), 6),
        "top1_worst_rate": round(float(group["top1_is_worst"].astype(bool).mean()), 6),
        "regret_mean": round(float(pd.to_numeric(group["top1_regret_vs_best"], errors="coerce").mean()), 6),
        "bottom2_loss_gt5_rate": round(float(pd.to_numeric(group["bottom2_loss_gt5_rate"], errors="coerce").mean()), 6),
        "research_only": True,
        "not_investment_instruction": True,
    }


def build_agent_sample_plan(candidate_rows: pd.DataFrame, *, max_groups: int = 36) -> pd.DataFrame:
    """Return agent-facing comparison cases without any future result columns."""
    if candidate_rows.empty:
        return pd.DataFrame()
    group_meta = (
        candidate_rows[
            [
                "comparison_group_id",
                "comparison_scenario",
                "repeat_seed",
                "time_block",
                "date",
                "candidate_count",
                "candidate_codes",
                "candidate_names",
                "industry_context",
            ]
        ]
        .drop_duplicates("comparison_group_id")
        .sort_values(["time_block", "comparison_scenario", "repeat_seed", "date", "comparison_group_id"])
    )
    selected = group_meta.groupby(["time_block", "comparison_scenario"], group_keys=False).head(max(1, max_groups // 14))
    if len(selected) < max_groups:
        rest = group_meta[~group_meta["comparison_group_id"].isin(selected["comparison_group_id"])]
        selected = pd.concat([selected, rest.head(max_groups - len(selected))], ignore_index=True)
    plan = selected.head(max_groups).copy()
    plan["task_mode"] = "candidate_comparison"
    plan["user_question_template"] = plan["comparison_scenario"].map(
        {
            "same_sector": "我在同一领域看了这些候选，只想重点研究1-2支，请给研究优先级。",
            "cross_sector": "我看中了几个不同领域候选，只想重点研究1-2支，请给研究优先级。",
        }
    )
    plan["research_only"] = True
    plan["not_investment_instruction"] = True
    forbidden = sorted(FUTURE_RESULT_COLUMNS & set(plan.columns))
    if forbidden:
        raise ValueError(f"sample plan leaked future/result fields: {forbidden}")
    return plan


def write_findings(aggregate: pd.DataFrame, sample_plan: pd.DataFrame, cfg: AuditConfig, path: Path) -> None:
    lines = [
        "# P1 候选对比工作流 v1 审计",
        "",
        "本报告只做 A 股研究辅助评估，不构成投资建议，不接券商、不自动交易。",
        "",
        "## 设置",
        "",
        f"- candidate_size: `{cfg.candidate_size}`",
        f"- repeats: `{cfg.repeats}`（稳定 hash 抽样，不看未来收益，不按待测 score 选样本）",
        f"- decision_frequency: `{cfg.decision_frequency}`",
        f"- industries_per_date: `{cfg.industries_per_date}`",
        f"- agent sample plan rows: `{len(sample_plan)}`（不含未来收益/GT 字段）",
        "",
        "## ALL 汇总",
        "",
    ]
    all_rows = aggregate[aggregate["time_block"].eq("ALL")] if not aggregate.empty else pd.DataFrame()
    if all_rows.empty:
        lines.append("无可用结果。")
    else:
        cols = [
            "comparison_scenario",
            "score_name",
            "n_groups",
            "mean_rank_ic",
            "rank_ic_positive_rate",
            "top1_excess_mean",
            "top2_excess_mean",
            "top1_positive_rate",
            "top1_worst_rate",
            "regret_mean",
        ]
        lines.append(all_rows[cols].to_markdown(index=False))
    lines.extend(
        [
            "",
            "## 使用判定",
            "",
            "- `candidate_context_blend_v1` 若跨同领域/跨领域、跨时间块均优于 `rev_chip_core`，才可进入 Flash smoke。",
            "- 若只在 H2026 或单一场景好看，标记为观察，不升默认，避免日期过拟合。",
            "- 等权/随机基线用灰色参考口径理解：它代表用户随便挑一个候选或平均看全部候选的期望表现。",
            "- 下一步若调用 DeepSeek，必须使用 sample plan，并重新做 no_news/no_financial/no_peer/no_bookskill/no_quant 消融。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit P1 candidate-comparison workflow.")
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--candidate-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--industries-per-date", type=int, default=3)
    parser.add_argument("--decision-frequency", default="every_2_weeks", choices=["all_dates", "weekly_friday", "weekly_tuesday", "every_2_weeks"])
    parser.add_argument("--min-industry-size", type=int, default=12)
    parser.add_argument("--max-dates-per-block", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AuditConfig(
        candidate_size=args.candidate_size,
        repeats=args.repeats,
        industries_per_date=args.industries_per_date,
        decision_frequency=args.decision_frequency,
        min_industry_size=args.min_industry_size,
        max_dates_per_block=args.max_dates_per_block,
        output_prefix=args.output_prefix,
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_candidate_frame()
    candidate_rows = build_candidate_groups(frame, cfg)
    candidate_rows = ensure_task_default_score(candidate_rows)
    detail, aggregate = evaluate_groups(candidate_rows)
    sample_plan = build_agent_sample_plan(candidate_rows)

    candidate_rows.drop(columns=["return_20d"], errors="ignore").to_csv(
        REPORT_DIR / f"{cfg.output_prefix}_candidate_rows_no_gt.csv", index=False, encoding="utf-8-sig"
    )
    detail.to_csv(REPORT_DIR / f"{cfg.output_prefix}_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(REPORT_DIR / f"{cfg.output_prefix}_aggregate.csv", index=False, encoding="utf-8-sig")
    sample_plan.to_csv(REPORT_DIR / f"{cfg.output_prefix}_sample_plan.csv", index=False, encoding="utf-8-sig")
    write_findings(aggregate, sample_plan, cfg, REPORT_DIR / f"{cfg.output_prefix}_findings.md")
    print(f"candidate_groups={candidate_rows['comparison_group_id'].nunique() if not candidate_rows.empty else 0}")
    print(f"detail_rows={len(detail)} aggregate_rows={len(aggregate)} sample_plan_rows={len(sample_plan)}")
    if not aggregate.empty:
        print(aggregate[aggregate["time_block"].eq("ALL")][["comparison_scenario", "score_name", "mean_rank_ic", "top1_excess_mean", "top1_positive_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
