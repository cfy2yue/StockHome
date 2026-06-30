"""Regime-aware K-line/peer/chip scorer experiment.

This is a local, no-DS experiment. Realized returns are used only for offline
walk-forward evaluation. Rule outcomes written for Agent use are sanitized and
contain no future/result fields.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.dual_mode_round import (  # noqa: E402
    DEFAULT_JOINED_GT_CACHE_PATH,
    DEFAULT_PORTFOLIO_PRESET,
    TIME_BLOCKS,
    _portfolio_ranker_details,
)
from src.agent_training.quant_tool_context import FUTURE_RESULT_FIELDS, sanitize_quant_tool_outcome  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
OUTPUT_PREFIX = "kline_peer_chip_regime_scorer_v1"
ROUND_TRIP_COST_PCT = 1.5
HIGH_RANKER_QUANTILE = 0.80
TOP_PCTS = [0.05, 0.10, 0.20]
MIN_TRAIN_ROWS = 1000
MIN_VALID_ROWS = 200
BLOCK_ORDER = list(TIME_BLOCKS.keys())
VALID_BLOCKS = BLOCK_ORDER[1:]


CORE_FEATURES = [
    "kline_return_3d",
    "kline_return_5d",
    "kline_return_10d",
    "kline_return_20d",
    "kline_return_60d",
    "kline_return_120d",
    "kline_drawdown_20d",
    "kline_drawdown_60d",
    "kline_range_position_20d",
    "kline_range_position_60d",
    "kline_efficiency_ratio_20d",
    "kline_direction_reversal_rate_20d",
    "kline_oscillation_cross_count_20d",
    "kline_mean_reversion_z20",
    "kline_rsi14",
    "kline_atr20_pct",
    "kline_volatility_ratio_20_60",
    "kline_volatility_ratio_20_120",
    "kline_ma_gap_20_60",
    "kline_ma_gap_close_200",
    "corr_peer_avg_return_20d",
    "corr_peer_relative_return_20d",
    "corr_peer_positive_breadth_20d",
    "corr_peer_avg_corr",
    "tushare_industry_relative_return_20d",
    "tushare_industry_positive_breadth_20d",
    "tushare_industry_above_ma200_rate",
    "tushare_area_relative_return_20d",
    "tushare_area_positive_breadth_20d",
    "lower_support",
    "chip_concentration",
    "cost_band_width",
    "upper_overhang",
    "winner_rate_pct",
    "neg_winner_rate",
]

NEGATIVE_ALIGNED = {
    "kline_return_3d",
    "kline_return_5d",
    "kline_return_10d",
    "kline_return_20d",
    "kline_return_60d",
    "kline_drawdown_20d",
    "kline_drawdown_60d",
    "kline_range_position_20d",
    "kline_mean_reversion_z20",
    "corr_peer_avg_return_20d",
    "corr_peer_positive_breadth_20d",
    "tushare_industry_relative_return_20d",
    "tushare_area_relative_return_20d",
    "upper_overhang",
    "cost_band_width",
}

REGIME_BASE_FIELDS = [
    "market_kline_return_20d_mean",
    "market_kline_return_60d_mean",
    "market_kline_atr20_mean",
    "market_industry_breadth_mean",
    "market_lower_support_mean",
    "market_upper_overhang_mean",
    "market_news_coverage_rate",
]


@dataclass
class Fitted:
    variant: str
    task_mode: str
    features: list[str]
    scaler: StandardScaler
    model: LogisticRegression


def main() -> None:
    parser = argparse.ArgumentParser(description="Run regime-aware kline/peer/chip scorer experiment.")
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--joined-cache", default=str(DEFAULT_JOINED_GT_CACHE_PATH))
    parser.add_argument("--high-ranker-quantile", type=float, default=HIGH_RANKER_QUANTILE)
    parser.add_argument("--top-pcts", default=",".join(str(x) for x in TOP_PCTS))
    parser.add_argument("--decision-frequency", choices=["all_dates", "every_2_weeks", "weekly_friday", "weekly_tuesday"], default="all_dates")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    top_pcts = [float(item) for item in str(args.top_pcts).split(",") if item.strip()]
    frame = load_frame(Path(args.joined_cache), high_ranker_quantile=args.high_ranker_quantile)
    frame = add_market_regime_features(frame)
    scored, metrics = run_walkforward(
        frame,
        top_pcts=top_pcts,
        high_ranker_quantile=args.high_ranker_quantile,
        decision_frequency=args.decision_frequency,
    )
    aggregate = aggregate_metrics(metrics)
    coverage = feature_coverage(frame)
    paths = write_outputs(
        prefix=args.output_prefix,
        scored=scored,
        metrics=metrics,
        aggregate=aggregate,
        coverage=coverage,
        high_ranker_quantile=args.high_ranker_quantile,
        top_pcts=top_pcts,
        decision_frequency=args.decision_frequency,
    )
    outcomes_path = REPORT_DIR / f"{args.output_prefix}_rule_outcomes.jsonl"
    write_rule_outcomes(outcomes_path, build_rule_outcomes(aggregate))

    print("A股研究Agent")
    print(f"rows={len(frame)}")
    print(f"scored_rows={len(scored)}")
    print(f"metrics={len(metrics)}")
    print(f"report={paths['report']}")
    print(f"rule_outcomes={outcomes_path}")


def load_frame(path: Path, *, high_ranker_quantile: float) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    frame.columns = [col.lstrip("\ufeff") for col in frame.columns]
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    if "gt_status" in frame and frame["gt_status"].notna().any():
        frame = frame[frame["gt_status"].astype(str).eq("evaluated")].copy()
    frame["return_20d"] = pd.to_numeric(frame["return_20d"], errors="coerce")
    frame = frame.dropna(subset=["date", "code", "return_20d"]).copy()
    frame["time_block"] = frame["date"].map(block_for_date)
    frame = frame[frame["time_block"].isin(BLOCK_ORDER)].copy()

    ranker = _portfolio_ranker_details(
        frame,
        preset=DEFAULT_PORTFOLIO_PRESET,
        valid_block="all_walkforward",
        decision_frequency="every_2_weeks",
    )
    frame["rev_chip_score"] = ranker["score"]
    frame["rev_chip_score_quantile"] = ranker["score_quantile"]
    frame["portfolio_candidate_pool"] = (
        pd.to_numeric(frame["rev_chip_score_quantile"], errors="coerce") >= high_ranker_quantile
    )
    return frame.reset_index(drop=True)


def block_for_date(value: Any) -> str | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    for block, (start, end) in TIME_BLOCKS.items():
        if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
            return block
    return None


def add_market_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in CORE_FEATURES:
        if col not in out:
            out[col] = np.nan
    out["news_coverage_flag"] = (pd.to_numeric(out.get("news_missing_rate", 1.0), errors="coerce").fillna(1.0) < 0.75).astype(float)
    grouped = out.groupby(out["date"].astype(str), sort=False)
    market = pd.DataFrame(
        {
            "market_kline_return_20d_mean": grouped["kline_return_20d"].mean(),
            "market_kline_return_60d_mean": grouped["kline_return_60d"].mean(),
            "market_kline_atr20_mean": grouped["kline_atr20_pct"].mean(),
            "market_industry_breadth_mean": grouped["tushare_industry_positive_breadth_20d"].mean(),
            "market_lower_support_mean": grouped["lower_support"].mean(),
            "market_upper_overhang_mean": grouped["upper_overhang"].mean(),
            "market_news_coverage_rate": grouped["news_coverage_flag"].mean(),
        }
    ).reset_index(names="date")
    out = out.merge(market, on="date", how="left")
    return out


def run_walkforward(frame: pd.DataFrame, *, top_pcts: list[float], high_ranker_quantile: float, decision_frequency: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored_parts = []
    metric_rows = []
    for valid_block in VALID_BLOCKS:
        train_blocks = BLOCK_ORDER[: BLOCK_ORDER.index(valid_block)]
        train_all = apply_frequency(frame[frame["time_block"].isin(train_blocks)].copy(), decision_frequency)
        valid_all = apply_frequency(frame[frame["time_block"].eq(valid_block)].copy(), decision_frequency)
        if len(train_all) < MIN_TRAIN_ROWS or len(valid_all) < MIN_VALID_ROWS:
            continue
        regime_spec = fit_regime_spec(train_all)
        train_all = apply_regime_spec(train_all, regime_spec)
        valid_all = apply_regime_spec(valid_all, regime_spec)

        for task_mode in ["portfolio_pool", "single_stock"]:
            train = task_frame(train_all, task_mode)
            valid = task_frame(valid_all, task_mode)
            if len(train) < MIN_TRAIN_ROWS or len(valid) < MIN_VALID_ROWS:
                continue
            models = fit_models(train, task_mode=task_mode)
            valid = valid.copy()
            valid["baseline_rev_chip_score"] = pd.to_numeric(valid["rev_chip_score"], errors="coerce").fillna(0.0)
            valid["manual_regime_reversal_score"] = manual_score(valid)
            for model in models:
                valid[model.variant] = score_model(model, valid)
            valid["task_mode"] = task_mode
            valid["valid_block"] = valid_block
            valid["train_blocks"] = "+".join(train_blocks)
            scored_parts.append(valid)

            variants = ["baseline_rev_chip_score", "manual_regime_reversal_score", *[m.variant for m in models]]
            for top_pct in top_pcts:
                for variant in variants:
                    metric_rows.append(
                        evaluate(
                            valid,
                            variant,
                            task_mode=task_mode,
                            valid_block=valid_block,
                            train_blocks=train_blocks,
                            top_pct=top_pct,
                            high_ranker_quantile=high_ranker_quantile,
                            decision_frequency=decision_frequency,
                        )
                    )
    scored = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    return scored, metrics


def apply_frequency(frame: pd.DataFrame, decision_frequency: str) -> pd.DataFrame:
    if frame.empty or decision_frequency == "all_dates":
        return frame.copy()
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if decision_frequency == "weekly_friday":
        return frame[dates.dt.weekday.eq(4)].copy()
    if decision_frequency == "weekly_tuesday":
        return frame[dates.dt.weekday.eq(1)].copy()
    if decision_frequency == "every_2_weeks":
        return frame[dates.dt.isocalendar().week.astype(int).mod(2).eq(0)].copy()
    raise ValueError(decision_frequency)


def task_frame(frame: pd.DataFrame, task_mode: str) -> pd.DataFrame:
    if task_mode == "portfolio_pool":
        return frame[frame["portfolio_candidate_pool"]].copy()
    if task_mode == "single_stock":
        return frame.copy()
    raise ValueError(task_mode)


def fit_regime_spec(train: pd.DataFrame) -> dict[str, float]:
    spec = {}
    for col in REGIME_BASE_FIELDS:
        vals = pd.to_numeric(train[col], errors="coerce")
        spec[f"{col}_q35"] = float(vals.quantile(0.35)) if vals.notna().any() else 0.0
        spec[f"{col}_q50"] = float(vals.quantile(0.50)) if vals.notna().any() else 0.0
        spec[f"{col}_q65"] = float(vals.quantile(0.65)) if vals.notna().any() else 0.0
    return spec


def apply_regime_spec(frame: pd.DataFrame, spec: dict[str, float]) -> pd.DataFrame:
    out = frame.copy()
    weak_market = (
        pd.to_numeric(out["market_kline_return_60d_mean"], errors="coerce").fillna(0.0)
        <= spec.get("market_kline_return_60d_mean_q35", 0.0)
    )
    high_vol = (
        pd.to_numeric(out["market_kline_atr20_mean"], errors="coerce").fillna(0.0)
        >= spec.get("market_kline_atr20_mean_q65", 0.0)
    )
    support_ok = (
        pd.to_numeric(out["market_lower_support_mean"], errors="coerce").fillna(0.0)
        >= spec.get("market_lower_support_mean_q50", 0.0)
    )
    breadth_ok = (
        pd.to_numeric(out["market_industry_breadth_mean"], errors="coerce").fillna(0.0)
        >= spec.get("market_industry_breadth_mean_q50", 0.0)
    )
    out["regime_weak_market"] = weak_market.astype(float)
    out["regime_high_vol"] = high_vol.astype(float)
    out["regime_repair_setup"] = (weak_market & support_ok & breadth_ok & ~high_vol).astype(float)
    out["regime_low_signal"] = (weak_market & high_vol).astype(float)
    return out


def fit_models(train: pd.DataFrame, *, task_mode: str) -> list[Fitted]:
    target = make_target(train, task_mode=task_mode)
    models = []
    for variant, use_regime in [
        ("logistic_kline_peer_chip", False),
        ("logistic_kline_peer_chip_regime", True),
    ]:
        features = build_feature_names(train, use_regime=use_regime)
        x = matrix(train, features)
        if x.shape[0] < MIN_TRAIN_ROWS or target.loc[x.index].nunique() < 2:
            continue
        scaler = StandardScaler()
        xs = scaler.fit_transform(x)
        model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
        model.fit(xs, target.loc[x.index].astype(int))
        models.append(Fitted(variant, task_mode, list(x.columns), scaler, model))
    return models


def make_target(frame: pd.DataFrame, *, task_mode: str) -> pd.Series:
    returns = pd.to_numeric(frame["return_20d"], errors="coerce")
    if task_mode == "portfolio_pool":
        excess = returns - returns.groupby(frame["date"].astype(str)).transform("mean")
        rank = excess.groupby(frame["date"].astype(str)).rank(pct=True, method="average")
        return ((rank >= 0.70) & (returns > 0)).astype(int)
    base = returns.groupby(frame["date"].astype(str)).transform("mean")
    rank = returns.groupby(frame["date"].astype(str)).rank(pct=True, method="average")
    return ((rank >= 0.70) & (returns > base) & (returns > 0)).astype(int)


def build_feature_names(frame: pd.DataFrame, *, use_regime: bool) -> list[str]:
    base = [col for col in CORE_FEATURES if col in frame]
    if not use_regime:
        return base
    extras = [
        "regime_weak_market",
        "regime_high_vol",
        "regime_repair_setup",
        "regime_low_signal",
        *REGIME_BASE_FIELDS,
    ]
    names = [*base, *[col for col in extras if col in frame]]
    for col in base:
        for regime in ["regime_weak_market", "regime_repair_setup", "regime_low_signal"]:
            names.append(f"{col}__x__{regime}")
    return names


def matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    for feature in features:
        if "__x__" in feature:
            left, right = feature.split("__x__", 1)
            values = numeric(frame, left) * numeric(frame, right)
        else:
            values = numeric(frame, feature)
        if feature.split("__x__", 1)[0] in NEGATIVE_ALIGNED:
            values = -values
        med = values.median()
        data[feature] = values.fillna(0.0 if pd.isna(med) else med)
    x = pd.DataFrame(data, index=frame.index)
    nunique = x.nunique(dropna=True)
    return x[[col for col in x.columns if nunique[col] >= 2]]


def numeric(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame.get(col, pd.Series(0.0, index=frame.index)), errors="coerce")


def score_model(model: Fitted, frame: pd.DataFrame) -> pd.Series:
    x = matrix(frame, model.features).reindex(columns=model.features, fill_value=0.0)
    if x.empty:
        return pd.Series(0.0, index=frame.index)
    probs = model.model.predict_proba(model.scaler.transform(x))[:, 1]
    out = pd.Series(0.0, index=frame.index)
    out.loc[x.index] = probs
    return out


def manual_score(frame: pd.DataFrame) -> pd.Series:
    return (
        -0.22 * z(frame, "kline_return_20d")
        -0.20 * z(frame, "kline_return_60d")
        -0.16 * z(frame, "corr_peer_avg_return_20d")
        +0.14 * z(frame, "lower_support")
        -0.12 * z(frame, "upper_overhang")
        +0.08 * z(frame, "kline_direction_reversal_rate_20d")
        +0.08 * z(frame, "kline_oscillation_cross_count_20d")
        +0.10 * numeric(frame, "regime_repair_setup")
        -0.10 * numeric(frame, "regime_low_signal")
    )


def z(frame: pd.DataFrame, col: str) -> pd.Series:
    values = numeric(frame, col).fillna(0.0)

    def _z(group: pd.Series) -> pd.Series:
        std = float(group.std())
        if std <= 0 or math.isnan(std) or len(group) < 5:
            return pd.Series(0.0, index=group.index)
        return (group - float(group.mean())) / std

    return values.groupby(frame["date"].astype(str), sort=False).transform(_z)


def evaluate(
    valid: pd.DataFrame,
    variant: str,
    *,
    task_mode: str,
    valid_block: str,
    train_blocks: list[str],
    top_pct: float,
    high_ranker_quantile: float,
    decision_frequency: str,
) -> dict[str, Any]:
    score = pd.to_numeric(valid[variant], errors="coerce").fillna(0.0)
    selected = select_top(valid, score, top_pct=top_pct)
    returns = pd.to_numeric(valid["return_20d"], errors="coerce")
    selected_returns = pd.to_numeric(selected["return_20d"], errors="coerce") if not selected.empty else pd.Series(dtype=float)
    base_by_date = returns.groupby(valid["date"].astype(str)).transform("mean")
    selected_base = base_by_date.loc[selected.index] if not selected.empty else pd.Series(dtype=float)
    selected_excess = selected_returns - selected_base
    daily_ic = []
    for _, idx in valid.groupby(valid["date"].astype(str), sort=True).groups.items():
        if len(idx) < 5:
            continue
        corr = pd.Series(score.loc[idx]).corr(pd.Series(returns.loc[idx]), method="spearman")
        if not pd.isna(corr):
            daily_ic.append(float(corr))
    concentration = selected["code"].astype(str).value_counts(normalize=True).max() if not selected.empty else np.nan
    return {
        "task_mode": task_mode,
        "variant": variant,
        "top_pct": top_pct,
        "decision_frequency": decision_frequency,
        "train_blocks": "+".join(train_blocks),
        "valid_block": valid_block,
        "high_ranker_quantile": high_ranker_quantile if task_mode == "portfolio_pool" else np.nan,
        "candidate_rows": int(len(valid)),
        "selected_rows": int(len(selected)),
        "coverage_dates": int(selected["date"].nunique()) if not selected.empty else 0,
        "unique_stocks": int(selected["code"].nunique()) if not selected.empty else 0,
        "top_stock_concentration": round(float(concentration), 6) if not pd.isna(concentration) else np.nan,
        "rank_ic": round(float(np.mean(daily_ic)), 6) if daily_ic else np.nan,
        "ic_positive_rate": round(float(np.mean([v > 0 for v in daily_ic])), 6) if daily_ic else np.nan,
        "avg_return_20d": round(float(selected_returns.mean()), 6) if not selected_returns.empty else np.nan,
        "base_avg_return_20d": round(float(selected_base.mean()), 6) if not selected_base.empty else np.nan,
        "positive_20d_rate": round(float((selected_returns > 0).mean()), 6) if not selected_returns.empty else np.nan,
        "base_positive_20d_rate": round(float((selected_base > 0).mean()), 6) if not selected_base.empty else np.nan,
        "pool_excess_20d": round(float(selected_excess.mean()), 6) if not selected_excess.empty else np.nan,
        "net_pool_excess_after_cost": round(float(selected_excess.mean() - ROUND_TRIP_COST_PCT), 6) if not selected_excess.empty else np.nan,
        "std_return_20d": round(float(selected_returns.std()), 6) if len(selected_returns) > 1 else np.nan,
        "active_exposure": round(float(len(selected) / max(1, len(valid))), 6),
        "research_only": True,
        "not_investment_instruction": True,
    }


def select_top(frame: pd.DataFrame, score: pd.Series, *, top_pct: float) -> pd.DataFrame:
    data = frame.copy()
    data["_score"] = score
    rows = []
    for _, group in data.groupby(data["date"].astype(str), sort=True):
        k = max(1, int(math.ceil(len(group) * top_pct)))
        rows.append(group.sort_values(["_score", "code"], ascending=[False, True]).head(k))
    if not rows:
        return data.iloc[0:0].drop(columns=["_score"], errors="ignore")
    return pd.concat(rows, ignore_index=False).drop(columns=["_score"])


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    keys = ["task_mode", "variant", "top_pct", "decision_frequency"]
    for values, group in metrics.groupby(keys, sort=True):
        h = group[group["valid_block"].eq("H2026_1")]
        prior = group[~group["valid_block"].eq("H2026_1")]
        row = {key: value for key, value in zip(keys, values)}
        row.update(
            {
                "blocks": int(group["valid_block"].nunique()),
                "prior_blocks": int(prior["valid_block"].nunique()),
                "prior_mean_rank_ic": mean(prior, "rank_ic"),
                "h2026_rank_ic": mean(h, "rank_ic"),
                "prior_pool_excess_20d": mean(prior, "pool_excess_20d"),
                "h2026_pool_excess_20d": mean(h, "pool_excess_20d"),
                "prior_net_pool_excess_after_cost": mean(prior, "net_pool_excess_after_cost"),
                "h2026_net_pool_excess_after_cost": mean(h, "net_pool_excess_after_cost"),
                "prior_positive_20d_rate": mean(prior, "positive_20d_rate"),
                "h2026_positive_20d_rate": mean(h, "positive_20d_rate"),
                "max_top_stock_concentration": round(float(group["top_stock_concentration"].max()), 6),
                "promotion_status": promotion_status(prior, h),
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["task_mode", "promotion_status", "prior_mean_rank_ic", "h2026_rank_ic"], ascending=[True, True, False, False])


def mean(frame: pd.DataFrame, col: str) -> float:
    if frame.empty:
        return np.nan
    return round(float(pd.to_numeric(frame[col], errors="coerce").mean()), 6)


def promotion_status(prior: pd.DataFrame, h2026: pd.DataFrame) -> str:
    if prior.empty or h2026.empty:
        return "observe_insufficient_blocks"
    prior_ic = float(prior["rank_ic"].mean())
    h_ic = float(h2026["rank_ic"].mean())
    prior_net = float(prior["net_pool_excess_after_cost"].mean())
    h_net = float(h2026["net_pool_excess_after_cost"].mean())
    concentration = float(pd.concat([prior, h2026])["top_stock_concentration"].max())
    if prior_ic >= 0.03 and h_ic >= 0.03 and prior_net > 0 and h_net > 0 and concentration <= 0.25:
        return "accepted_candidate"
    if h_ic >= 0.03 and h_net > 0 and concentration <= 0.35:
        return "observe_latest_positive_prior_weak"
    if prior_ic >= 0.03 and concentration <= 0.35:
        return "observe_prior_positive_latest_weak"
    return "rejected_or_diagnostic_only"


def feature_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [*CORE_FEATURES, *REGIME_BASE_FIELDS, "regime_weak_market", "regime_repair_setup", "regime_low_signal"]:
        if col not in frame:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce")
        rows.append(
            {
                "feature": col,
                "non_null_rate": round(float(vals.notna().mean()), 6),
                "non_zero_rate": round(float(vals.fillna(0).ne(0).mean()), 6),
                "unique_values": int(vals.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    *,
    prefix: str,
    scored: pd.DataFrame,
    metrics: pd.DataFrame,
    aggregate: pd.DataFrame,
    coverage: pd.DataFrame,
    high_ranker_quantile: float,
    top_pcts: list[float],
    decision_frequency: str,
) -> dict[str, Path]:
    detail_path = REPORT_DIR / f"{prefix}_scored_detail.csv"
    metrics_path = REPORT_DIR / f"{prefix}_step_metrics.csv"
    aggregate_path = REPORT_DIR / f"{prefix}_aggregate.csv"
    coverage_path = REPORT_DIR / f"{prefix}_feature_coverage.csv"
    report_path = REPORT_DIR / f"{prefix}.md"
    detail_cols = [
        "date",
        "code",
        "name",
        "time_block",
        "task_mode",
        "valid_block",
        "return_20d",
        "rev_chip_score_quantile",
        "baseline_rev_chip_score",
        "manual_regime_reversal_score",
        "logistic_kline_peer_chip",
        "logistic_kline_peer_chip_regime",
        "regime_weak_market",
        "regime_repair_setup",
        "regime_low_signal",
    ]
    scored[[col for col in detail_cols if col in scored]].to_csv(detail_path, index=False, encoding="utf-8-sig")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    write_report(
        report_path,
        aggregate,
        metrics,
        coverage,
        high_ranker_quantile=high_ranker_quantile,
        top_pcts=top_pcts,
        decision_frequency=decision_frequency,
    )
    return {"detail": detail_path, "metrics": metrics_path, "aggregate": aggregate_path, "coverage": coverage_path, "report": report_path}


def write_report(path: Path, aggregate: pd.DataFrame, metrics: pd.DataFrame, coverage: pd.DataFrame, *, high_ranker_quantile: float, top_pcts: list[float], decision_frequency: str) -> None:
    accepted = aggregate[aggregate["promotion_status"].eq("accepted_candidate")] if not aggregate.empty else pd.DataFrame()
    latest_positive = aggregate[aggregate["promotion_status"].str.contains("latest_positive", na=False)] if not aggregate.empty else pd.DataFrame()
    lines = [
        "# Kline Peer Chip Regime Scorer v1",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Setup",
        "",
        f"- portfolio candidate pool: `rev_plus_chip_core score_quantile >= {high_ranker_quantile:.2f}`",
        f"- top_pcts: `{','.join(f'{x:.2f}' for x in top_pcts)}`",
        f"- decision_frequency: `{decision_frequency}`",
        "- split: half-year walk-forward; each valid block only trains on prior blocks",
        "- labels: offline-only 20d outcomes; labels are not written to evidence pack or Agent prompt",
        "- tasks: `portfolio_pool` and `single_stock`",
        "",
        "## Aggregate",
        "",
        table(aggregate),
        "",
        "## Accepted Candidates",
        "",
        table(accepted),
        "",
        "## Latest-Positive But Prior-Weak",
        "",
        table(latest_positive),
        "",
        "## Step Metrics",
        "",
        table(metrics),
        "",
        "## Feature Coverage",
        "",
        table(coverage.sort_values("non_null_rate").head(20) if not coverage.empty else coverage),
        "",
        "## Interpretation",
        "",
        "- `accepted_candidate` 才可进入下一轮 DS ablation；`observe_*` 只能作为观察型工具。",
        "- 如果 regime 版本仅改善 H2026 但 prior blocks 不稳，说明它可能是在拟合最新块，不可直接升默认。",
        "- 单支模式与组合模式分开判断；单支 top 分位只代表机会侧复核，不能单独触发买入/加仓。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_无数据_"
    return frame.to_markdown(index=False)


def build_rule_outcomes(aggregate: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    if aggregate.empty:
        return rows
    for _, row in aggregate.iterrows():
        variant = str(row["variant"])
        if variant == "baseline_rev_chip_score":
            continue
        status = str(row["promotion_status"])
        usable = status == "accepted_candidate"
        item = {
            "tool_id": (
                f"kline_peer_chip_regime_scorer:{row['task_mode']}:{variant}:"
                f"top{int(float(row['top_pct']) * 100)}:{row.get('decision_frequency', 'all_dates')}"
            ),
            "tool_version": "v1",
            "task_mode": row["task_mode"],
            "policy_profile": "kline_peer_chip_regime_walkforward_v1",
            "decision_frequency": row.get("decision_frequency", "all_dates"),
            "feature_group": "kline_peer_chip_regime",
            "selection_mode": "walk_forward_top_pct_rerank",
            "score": {"accepted_candidate": 0.8, "observe_latest_positive_prior_weak": 0.45, "observe_prior_positive_latest_weak": 0.35}.get(status, 0.1),
            "score_quantile": None,
            "confidence": {"accepted_candidate": 0.75, "observe_latest_positive_prior_weak": 0.45, "observe_prior_positive_latest_weak": 0.35}.get(status, 0.2),
            "action_hint": "continue_research" if usable else "observe",
            "usable_in_agent_default": usable,
            "top_features": ["kline_return_20d", "kline_return_60d", "corr_peer_avg_return_20d", "lower_support", "regime_repair_setup"],
            "missing_flags": [],
            "counter_evidence": status_counter(status),
            "source_ref_ids": ["kline_peer_chip_regime_scorer_v1"],
            "train_valid_test_blocks": "walk_forward_H2023_2_to_H2026_1",
            "promotion_status": status,
            "research_only": True,
            "not_investment_instruction": True,
        }
        rows.append(sanitize_quant_tool_outcome(item))
    return rows


def status_counter(status: str) -> list[str]:
    if status == "accepted_candidate":
        return ["research_only", "requires_agent_counter_evidence_review", "no_investment_instruction"]
    if status == "observe_latest_positive_prior_weak":
        return ["latest_block_positive_but_prior_gate_failed", "observe_only", "requires_more_scaling"]
    if status == "observe_prior_positive_latest_weak":
        return ["prior_blocks_positive_but_latest_gate_failed", "observe_only", "requires_regime_guard"]
    return ["walk_forward_gate_failed", "diagnostic_only", "do_not_use_for_default_promotion"]


def write_rule_outcomes(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            leaked = future_keys(row)
            if leaked:
                raise ValueError(f"future fields leaked into rule outcome: {sorted(leaked)}")
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def future_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        leaked = {str(key) for key in value if str(key) in FUTURE_RESULT_FIELDS}
        for child in value.values():
            leaked.update(future_keys(child))
        return leaked
    if isinstance(value, list):
        leaked: set[str] = set()
        for child in value:
            leaked.update(future_keys(child))
        return leaked
    return set()


if __name__ == "__main__":
    main()
