"""Train/evaluate a lightweight positive-evidence scorer without DS calls.

The scorer is an offline research tool. Labels and realized returns are used
only for walk-forward evaluation and reports; the optional quant-tool outcomes
written for Agent consumption contain no future/result fields.
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
from sklearn.ensemble import HistGradientBoostingClassifier
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
OUTPUT_PREFIX = "positive_evidence_scorer_v1"
DEFAULT_RULE_OUTCOMES = ""
ROUND_TRIP_COST_PCT = 1.5
MIN_TRAIN_ROWS = 1000
MIN_VALID_ROWS = 200
HIGH_RANKER_QUANTILE = 0.80
TOP_PCT = 0.10

BLOCK_ORDER = list(TIME_BLOCKS.keys())
VALID_BLOCKS = BLOCK_ORDER[1:]


FEATURE_GROUPS: dict[str, list[str]] = {
    "news_quality": [
        "news_count_30d",
        "news_missing_rate",
        "news_official_count_30d",
        "news_company_count_30d",
        "news_industry_policy_count_30d",
        "news_materiality_max_30d",
        "news_net_materiality_30d",
        "news_opportunity_score",
        "news_warning_score",
        "policy_background_score",
        "official_confirmation_score",
        "announcement_materiality_score",
        "news_timestamp_quality",
        "news_evidence_quality",
        "news_conflict_intensity_30d",
    ],
    "financial_event": [
        "financial_report_missing_rate",
        "financial_report_event_count",
        "financial_report_materiality_score",
        "financial_quality_risk_score",
        "financial_surprise_score",
        "financial_disclosure_quality_score",
        "financial_report_window_days",
    ],
    "peer_relative": [
        "corr_peer_avg_return_20d",
        "corr_peer_relative_return_20d",
        "corr_peer_positive_breadth_20d",
        "corr_peer_avg_corr",
        "tushare_industry_relative_return_20d",
        "tushare_industry_positive_breadth_20d",
        "tushare_industry_above_ma200_rate",
        "tushare_industry_news_attention_gap",
        "tushare_area_relative_return_20d",
        "tushare_area_positive_breadth_20d",
        "tushare_area_above_ma200_rate",
        "tushare_area_news_attention_gap",
    ],
    "chip_core": [
        "lower_support",
        "chip_concentration",
        "cost_band_width",
        "upper_overhang",
        "winner_rate_pct",
        "neg_winner_rate",
    ],
    "kline_multiscale": [
        "kline_return_3d",
        "kline_return_5d",
        "kline_return_10d",
        "kline_return_20d",
        "kline_return_60d",
        "kline_return_120d",
        "kline_return_240d",
        "kline_drawdown_20d",
        "kline_drawdown_60d",
        "kline_range_position_20d",
        "kline_range_position_60d",
        "kline_efficiency_ratio_20d",
        "kline_efficiency_ratio_60d",
        "kline_direction_reversal_rate_20d",
        "kline_direction_reversal_rate_60d",
        "kline_oscillation_cross_count_20d",
        "kline_oscillation_cross_count_60d",
        "kline_mean_reversion_z20",
        "kline_rsi14",
        "kline_atr20_pct",
        "kline_volatility_ratio_20_60",
        "kline_volatility_ratio_20_120",
        "kline_ma_gap_20_60",
        "kline_ma_gap_close_200",
    ],
    "legacy_book_python": [
        "book_score",
        "counter_score",
        "completeness_score",
        "relative_strength_rank",
        "close_above_ma200",
        "rsi14",
        "atr20_pct",
        "volume_ratio20",
    ],
}

MODEL_FEATURE_SETS: dict[str, list[str]] = {
    "positive_core": ["news_quality", "financial_event", "peer_relative", "chip_core", "legacy_book_python"],
    "kline_peer_only": ["kline_multiscale", "peer_relative", "chip_core"],
    "all_channels": list(FEATURE_GROUPS.keys()),
}

NEGATIVE_ALIGNED_FEATURES = {
    "news_missing_rate",
    "news_warning_score",
    "news_conflict_intensity_30d",
    "financial_report_missing_rate",
    "financial_quality_risk_score",
    "financial_report_window_days",
    "upper_overhang",
    "cost_band_width",
    "kline_return_20d",
    "kline_return_60d",
    "kline_drawdown_20d",
    "kline_drawdown_60d",
    "kline_mean_reversion_z20",
    "relative_strength_rank",
    "counter_score",
    "atr20_pct",
}


@dataclass
class FittedModel:
    variant: str
    feature_set: str
    features: list[str]
    scaler: StandardScaler | None = None
    model: Any | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run positive-evidence scorer walk-forward experiment.")
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--joined-cache", default=str(DEFAULT_JOINED_GT_CACHE_PATH))
    parser.add_argument("--rule-outcomes-output", default=DEFAULT_RULE_OUTCOMES)
    parser.add_argument("--high-ranker-quantile", type=float, default=HIGH_RANKER_QUANTILE)
    parser.add_argument("--top-pct", type=float, default=TOP_PCT)
    parser.add_argument("--model-family", choices=["manual_only", "logistic_only", "all"], default="all")
    parser.add_argument("--append-rule-outcomes", action="store_true")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_frame(Path(args.joined_cache), high_ranker_quantile=args.high_ranker_quantile)
    enriched = add_positive_evidence_features(frame)
    coverage = feature_coverage(enriched)
    scored, step_metrics = run_walkforward(enriched, top_pct=args.top_pct, model_family=args.model_family)
    aggregate = aggregate_metrics(step_metrics)

    paths = write_outputs(
        prefix=args.output_prefix,
        scored=scored,
        step_metrics=step_metrics,
        aggregate=aggregate,
        coverage=coverage,
        high_ranker_quantile=args.high_ranker_quantile,
        top_pct=args.top_pct,
        model_family=args.model_family,
    )
    rule_outcomes_path = Path(args.rule_outcomes_output) if args.rule_outcomes_output else REPORT_DIR / f"{args.output_prefix}_rule_outcomes.jsonl"
    rule_outcomes = build_rule_outcomes(aggregate)
    write_rule_outcomes(rule_outcomes_path, rule_outcomes, append=args.append_rule_outcomes)

    print("A股研究Agent")
    print(f"candidate_rows={len(enriched)}")
    print(f"scored_rows={len(scored)}")
    print(f"step_metrics={len(step_metrics)}")
    print(f"report={paths['report']}")
    print(f"rule_outcomes={rule_outcomes_path}")


def load_frame(path: Path, *, high_ranker_quantile: float) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing joined cache: {path}")
    frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    frame.columns = [c.lstrip("\ufeff") for c in frame.columns]
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    if "gt_status" in frame and frame["gt_status"].notna().any():
        frame = frame[frame["gt_status"].astype(str).eq("evaluated")].copy()
    frame["return_20d"] = pd.to_numeric(frame["return_20d"], errors="coerce")
    frame = frame.dropna(subset=["date", "code", "return_20d"]).copy()
    ranker = _portfolio_ranker_details(
        frame,
        preset=DEFAULT_PORTFOLIO_PRESET,
        valid_block="all_walkforward",
        decision_frequency="every_2_weeks",
    )
    frame["rev_chip_score"] = ranker["score"]
    frame["rev_chip_score_quantile"] = ranker["score_quantile"]
    frame = frame[pd.to_numeric(frame["rev_chip_score_quantile"], errors="coerce") >= high_ranker_quantile].copy()
    frame["time_block"] = frame["date"].map(block_for_date)
    return frame[frame["time_block"].isin(BLOCK_ORDER)].reset_index(drop=True)


def block_for_date(value: Any) -> str | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    for block, (start, end) in TIME_BLOCKS.items():
        if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
            return block
    return None


def add_positive_evidence_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for group, cols in FEATURE_GROUPS.items():
        for col in cols:
            if col not in out:
                out[col] = np.nan
            out[f"{col}__filled"] = pd.to_numeric(out[col], errors="coerce")
            median = out[f"{col}__filled"].median()
            out[f"{col}__filled"] = out[f"{col}__filled"].fillna(0.0 if pd.isna(median) else median)

    out["book_skill_present"] = out.get("triggered_skills", pd.Series("", index=out.index)).fillna("").astype(str).str.len().gt(0).astype(float)
    out["financial_event_matched"] = out.get("financial_report_join_status", pd.Series("", index=out.index)).fillna("").astype(str).eq("event_window_matched").astype(float)
    out["news_available"] = (pd.to_numeric(out.get("news_missing_rate", 1.0), errors="coerce").fillna(1.0) < 0.75).astype(float)

    out["manual_positive_evidence_score"] = (
        0.18 * z(out, "news_opportunity_score")
        - 0.18 * z(out, "news_warning_score")
        + 0.15 * z(out, "news_evidence_quality")
        + 0.12 * z(out, "official_confirmation_score")
        + 0.12 * z(out, "announcement_materiality_score")
        + 0.16 * z(out, "financial_report_event_count")
        - 0.18 * z(out, "financial_quality_risk_score")
        + 0.14 * z(out, "financial_surprise_score")
        + 0.16 * z(out, "tushare_industry_positive_breadth_20d")
        + 0.12 * z(out, "tushare_industry_relative_return_20d")
        + 0.16 * z(out, "lower_support")
        - 0.12 * z(out, "upper_overhang")
        + 0.08 * out["book_skill_present"]
    )
    out["manual_kline_peer_score"] = (
        -0.25 * z(out, "kline_return_20d")
        -0.22 * z(out, "kline_return_60d")
        -0.18 * z(out, "corr_peer_avg_return_20d")
        +0.16 * z(out, "lower_support")
        -0.12 * z(out, "upper_overhang")
        +0.10 * z(out, "kline_direction_reversal_rate_20d")
        +0.08 * z(out, "kline_oscillation_cross_count_20d")
    )
    out["manual_all_channel_score"] = 0.55 * out["manual_kline_peer_score"] + 0.45 * out["manual_positive_evidence_score"]
    return out


def z(frame: pd.DataFrame, col: str) -> pd.Series:
    values = pd.to_numeric(frame.get(col, pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)

    def _z(group: pd.Series) -> pd.Series:
        std = float(group.std())
        if std <= 0 or math.isnan(std) or len(group) < 5:
            return pd.Series(0.0, index=group.index)
        return (group - float(group.mean())) / std

    return values.groupby(frame["date"].astype(str), sort=False).transform(_z)


def feature_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, cols in FEATURE_GROUPS.items():
        available = [col for col in cols if col in frame]
        for col in available:
            series = pd.to_numeric(frame[col], errors="coerce")
            rows.append(
                {
                    "feature_group": group,
                    "feature": col,
                    "non_null_rate": round(float(series.notna().mean()), 4),
                    "non_zero_rate": round(float(series.fillna(0).ne(0).mean()), 4),
                    "unique_values": int(series.nunique(dropna=True)),
                    "research_only": True,
                    "not_investment_instruction": True,
                }
            )
    return pd.DataFrame(rows)


def run_walkforward(frame: pd.DataFrame, *, top_pct: float, model_family: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored_parts = []
    metric_rows = []
    for valid_block in VALID_BLOCKS:
        train_blocks = BLOCK_ORDER[: BLOCK_ORDER.index(valid_block)]
        train = frame[frame["time_block"].isin(train_blocks)].copy()
        valid = frame[frame["time_block"].eq(valid_block)].copy()
        if len(train) < MIN_TRAIN_ROWS or len(valid) < MIN_VALID_ROWS:
            continue
        models = fit_models(train, model_family=model_family)
        valid_scored = valid.copy()
        valid_scored["baseline_rev_chip_score"] = pd.to_numeric(valid_scored["rev_chip_score"], errors="coerce").fillna(0.0)
        valid_scored["manual_positive_evidence"] = valid_scored["manual_positive_evidence_score"]
        valid_scored["manual_kline_peer"] = valid_scored["manual_kline_peer_score"]
        valid_scored["manual_all_channel"] = valid_scored["manual_all_channel_score"]
        for model in models:
            valid_scored[f"{model.variant}_{model.feature_set}"] = score_model(model, valid_scored)
        valid_scored["train_blocks"] = "+".join(train_blocks)
        valid_scored["valid_block"] = valid_block
        scored_parts.append(valid_scored)

        variants = [
            "baseline_rev_chip_score",
            "manual_positive_evidence",
            "manual_kline_peer",
            "manual_all_channel",
            *[f"{model.variant}_{model.feature_set}" for model in models],
        ]
        for variant in variants:
            metric_rows.append(evaluate_variant(valid_scored, variant, train_blocks=train_blocks, valid_block=valid_block, top_pct=top_pct))
    scored = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    return scored, metrics


def fit_models(train: pd.DataFrame, *, model_family: str) -> list[FittedModel]:
    if model_family == "manual_only":
        return []
    y = make_target(train)
    models: list[FittedModel] = []
    for feature_set, groups in MODEL_FEATURE_SETS.items():
        features = flatten_features(groups, train)
        if len(features) < 5 or y.nunique(dropna=True) < 2:
            continue
        x = build_matrix(train, features)
        if x.shape[0] < MIN_TRAIN_ROWS:
            continue
        fitted_features = list(x.columns)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        logistic = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
        logistic.fit(x_scaled, y.loc[x.index].astype(int))
        models.append(FittedModel("logistic", feature_set, fitted_features, scaler, logistic))

        if model_family == "all":
            gbdt = HistGradientBoostingClassifier(max_iter=80, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.2, random_state=42)
            gbdt.fit(x, y.loc[x.index].astype(int))
            models.append(FittedModel("gbdt", feature_set, fitted_features, None, gbdt))
    return models


def make_target(frame: pd.DataFrame) -> pd.Series:
    returns = pd.to_numeric(frame["return_20d"], errors="coerce")
    pool_excess = returns - returns.groupby(frame["date"].astype(str)).transform("mean")
    rank = pool_excess.groupby(frame["date"].astype(str)).rank(pct=True, method="average")
    return ((rank >= 0.70) & (returns > 0)).astype(int)


def flatten_features(groups: list[str], frame: pd.DataFrame) -> list[str]:
    features: list[str] = []
    for group in groups:
        features.extend([col for col in FEATURE_GROUPS[group] if col in frame])
    extras = ["book_skill_present", "financial_event_matched", "news_available"]
    features.extend([col for col in extras if col in frame])
    return sorted(set(features))


def build_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    data = {}
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        if feature in NEGATIVE_ALIGNED_FEATURES:
            values = -values
        median = values.median()
        data[feature] = values.fillna(0.0 if pd.isna(median) else median)
    x = pd.DataFrame(data, index=frame.index)
    keep = x.notna().all(axis=1)
    x = x[keep]
    nunique = x.nunique(dropna=True)
    cols = [col for col in x.columns if nunique[col] >= 2]
    return x[cols]


def score_model(model: FittedModel, frame: pd.DataFrame) -> pd.Series:
    x = build_matrix(frame, model.features)
    if x.empty:
        return pd.Series(0.0, index=frame.index)
    x = x.reindex(columns=model.features, fill_value=0.0)
    if model.scaler is not None:
        probs = model.model.predict_proba(model.scaler.transform(x))[:, 1]
    else:
        probs = model.model.predict_proba(x)[:, 1]
    out = pd.Series(0.0, index=frame.index)
    out.loc[x.index] = probs
    return out


def evaluate_variant(valid: pd.DataFrame, variant: str, *, train_blocks: list[str], valid_block: str, top_pct: float) -> dict[str, Any]:
    score = pd.to_numeric(valid[variant], errors="coerce").fillna(0.0)
    returns = pd.to_numeric(valid["return_20d"], errors="coerce")
    base_mean = valid.groupby("date")["return_20d"].transform(lambda s: pd.to_numeric(s, errors="coerce").mean())
    selected = select_top_by_date(valid, score, top_pct=top_pct)
    selected_returns = returns.loc[selected.index]
    selected_excess = selected_returns - base_mean.loc[selected.index]
    dates = valid["date"].astype(str)
    daily_ic = []
    for _, idx in valid.groupby(dates, sort=True).groups.items():
        if len(idx) < 5:
            continue
        corr = pd.Series(score.loc[idx]).corr(pd.Series(returns.loc[idx]), method="spearman")
        if not pd.isna(corr):
            daily_ic.append(float(corr))
    concentration = selected["code"].astype(str).value_counts(normalize=True).max() if not selected.empty else np.nan
    return {
        "variant": variant,
        "train_blocks": "+".join(train_blocks),
        "valid_block": valid_block,
        "candidate_rows": int(len(valid)),
        "selected_rows": int(len(selected)),
        "coverage_dates": int(selected["date"].nunique()) if not selected.empty else 0,
        "unique_stocks": int(selected["code"].nunique()) if not selected.empty else 0,
        "top_stock_concentration": round(float(concentration), 4) if not pd.isna(concentration) else np.nan,
        "rank_ic": round(float(np.mean(daily_ic)), 6) if daily_ic else np.nan,
        "ic_positive_rate": round(float(np.mean([v > 0 for v in daily_ic])), 6) if daily_ic else np.nan,
        "avg_return_20d": round(float(selected_returns.mean()), 6) if not selected_returns.empty else np.nan,
        "positive_20d_rate": round(float((selected_returns > 0).mean()), 6) if not selected_returns.empty else np.nan,
        "pool_excess_20d": round(float(selected_excess.mean()), 6) if not selected_excess.empty else np.nan,
        "net_pool_excess_after_cost": round(float(selected_excess.mean() - ROUND_TRIP_COST_PCT), 6) if not selected_excess.empty else np.nan,
        "std_return_20d": round(float(selected_returns.std()), 6) if len(selected_returns) > 1 else np.nan,
        "active_exposure": round(float(len(selected) / max(1, len(valid))), 6),
        "research_only": True,
        "not_investment_instruction": True,
    }


def select_top_by_date(frame: pd.DataFrame, score: pd.Series, *, top_pct: float) -> pd.DataFrame:
    selected = []
    data = frame.copy()
    data["_score"] = score
    for _, group in data.groupby(data["date"].astype(str), sort=True):
        if group.empty:
            continue
        k = max(1, int(math.ceil(len(group) * top_pct)))
        selected.append(group.sort_values(["_score", "code"], ascending=[False, True]).head(k))
    if not selected:
        return data.iloc[0:0].copy()
    return pd.concat(selected, ignore_index=False).drop(columns=["_score"])


def aggregate_metrics(step_metrics: pd.DataFrame) -> pd.DataFrame:
    if step_metrics.empty:
        return pd.DataFrame()
    rows = []
    for variant, group in step_metrics.groupby("variant", sort=True):
        h2026 = group[group["valid_block"].eq("H2026_1")]
        prior = group[~group["valid_block"].eq("H2026_1")]
        rows.append(
            {
                "variant": variant,
                "blocks": int(group["valid_block"].nunique()),
                "prior_blocks": int(prior["valid_block"].nunique()),
                "mean_rank_ic": round(float(group["rank_ic"].mean()), 6),
                "prior_mean_rank_ic": round(float(prior["rank_ic"].mean()), 6) if not prior.empty else np.nan,
                "h2026_rank_ic": round(float(h2026["rank_ic"].mean()), 6) if not h2026.empty else np.nan,
                "mean_pool_excess_20d": round(float(group["pool_excess_20d"].mean()), 6),
                "prior_pool_excess_20d": round(float(prior["pool_excess_20d"].mean()), 6) if not prior.empty else np.nan,
                "h2026_pool_excess_20d": round(float(h2026["pool_excess_20d"].mean()), 6) if not h2026.empty else np.nan,
                "mean_net_pool_excess_after_cost": round(float(group["net_pool_excess_after_cost"].mean()), 6),
                "h2026_net_pool_excess_after_cost": round(float(h2026["net_pool_excess_after_cost"].mean()), 6) if not h2026.empty else np.nan,
                "mean_positive_20d_rate": round(float(group["positive_20d_rate"].mean()), 6),
                "h2026_positive_20d_rate": round(float(h2026["positive_20d_rate"].mean()), 6) if not h2026.empty else np.nan,
                "mean_active_exposure": round(float(group["active_exposure"].mean()), 6),
                "max_top_stock_concentration": round(float(group["top_stock_concentration"].max()), 6),
                "promotion_status": promotion_status(group),
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return pd.DataFrame(rows).sort_values(["promotion_status", "prior_mean_rank_ic", "h2026_rank_ic"], ascending=[True, False, False])


def promotion_status(group: pd.DataFrame) -> str:
    h2026 = group[group["valid_block"].eq("H2026_1")]
    prior = group[~group["valid_block"].eq("H2026_1")]
    if prior.empty or h2026.empty:
        return "observe_insufficient_blocks"
    prior_ic = float(prior["rank_ic"].mean())
    h_ic = float(h2026["rank_ic"].mean())
    prior_net = float(prior["net_pool_excess_after_cost"].mean())
    h_net = float(h2026["net_pool_excess_after_cost"].mean())
    concentration = float(group["top_stock_concentration"].max())
    if prior_ic >= 0.03 and h_ic >= 0.03 and prior_net > 0 and h_net > 0 and concentration <= 0.25:
        return "accepted_candidate"
    if prior_ic >= 0.03 and concentration <= 0.35:
        return "observe_prior_positive_latest_weak"
    return "rejected_or_diagnostic_only"


def write_outputs(
    *,
    prefix: str,
    scored: pd.DataFrame,
    step_metrics: pd.DataFrame,
    aggregate: pd.DataFrame,
    coverage: pd.DataFrame,
    high_ranker_quantile: float,
    top_pct: float,
    model_family: str,
) -> dict[str, Path]:
    detail_cols = [
        "date",
        "code",
        "name",
        "time_block",
        "valid_block",
        "train_blocks",
        "return_20d",
        "rev_chip_score_quantile",
        "manual_positive_evidence_score",
        "manual_kline_peer_score",
        "manual_all_channel_score",
    ]
    score_cols = [col for col in scored.columns if col.startswith("logistic_") or col.startswith("gbdt_") or col.startswith("baseline_")]
    detail_path = REPORT_DIR / f"{prefix}_scored_detail.csv"
    metrics_path = REPORT_DIR / f"{prefix}_step_metrics.csv"
    aggregate_path = REPORT_DIR / f"{prefix}_aggregate.csv"
    coverage_path = REPORT_DIR / f"{prefix}_feature_coverage.csv"
    report_path = REPORT_DIR / f"{prefix}.md"
    scored[[col for col in [*detail_cols, *score_cols] if col in scored]].to_csv(detail_path, index=False, encoding="utf-8-sig")
    step_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    write_report(report_path, aggregate, step_metrics, coverage, high_ranker_quantile=high_ranker_quantile, top_pct=top_pct, model_family=model_family)
    return {"detail": detail_path, "metrics": metrics_path, "aggregate": aggregate_path, "coverage": coverage_path, "report": report_path}


def write_report(path: Path, aggregate: pd.DataFrame, step_metrics: pd.DataFrame, coverage: pd.DataFrame, *, high_ranker_quantile: float, top_pct: float, model_family: str) -> None:
    top = aggregate.head(12) if not aggregate.empty else pd.DataFrame()
    weak = aggregate[aggregate["promotion_status"].ne("accepted_candidate")] if not aggregate.empty else pd.DataFrame()
    low_coverage = coverage.sort_values("non_null_rate").head(12) if not coverage.empty else pd.DataFrame()
    lines = [
        "# Positive Evidence Scorer Experiment v1",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Setup",
        "",
        f"- base candidate pool: `rev_plus_chip_core score_quantile >= {high_ranker_quantile:.2f}`",
        f"- selection per date: top `{top_pct:.2%}` inside the high-ranker candidate pool",
        f"- model_family: `{model_family}`",
        "- split: walk-forward by half-year block; each valid block only trains on prior blocks",
        "- labels: offline-only 20d outcomes; labels are not written to evidence pack or prompt",
        "- compared variants: rev+chip baseline, manual positive-evidence score, kline/peer score, logistic/GBDT over channel groups",
        "",
        "## Aggregate",
        "",
        _table(top),
        "",
        "## Step Metrics",
        "",
        _table(step_metrics),
        "",
        "## Weak / Not Promoted",
        "",
        _table(weak.head(12)),
        "",
        "## Lowest Feature Coverage",
        "",
        _table(low_coverage),
        "",
        "## Interpretation",
        "",
        "- 若某个 scorer 只在 prior blocks 好、H2026 弱，只能作为 observe，不得升默认。",
        "- 若 kline/peer-only 不弱于 all-channels，说明当前新闻/财报/BookSkill 正向信息还未形成稳定增量。",
        "- 若 positive-evidence scorer 稳定优于 baseline，下一步才接入 evidence pack 作为 `quant_tool_summaries`，再跑小 DS ablation。",
        "- 所有收益、RankIC 和 pool excess 只出现在离线报告中，不进入 Agent prompt。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_无数据_"
    return frame.to_markdown(index=False)


def build_rule_outcomes(aggregate: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    if aggregate.empty:
        return rows
    for _, row in aggregate.iterrows():
        variant = str(row["variant"])
        status = str(row["promotion_status"])
        usable = status == "accepted_candidate"
        if variant == "baseline_rev_chip_score":
            continue
        outcome = {
            "tool_id": f"positive_evidence_scorer:{variant}",
            "tool_version": "v1",
            "task_mode": "portfolio_pool",
            "policy_profile": "positive_evidence_walkforward_v1",
            "decision_frequency": "every_2_weeks",
            "feature_group": _variant_feature_group(variant),
            "selection_mode": "walk_forward_positive_evidence_rerank",
            "score": _status_score(status),
            "score_quantile": None,
            "confidence": _status_confidence(status),
            "action_hint": "observe" if not usable else "continue_research",
            "usable_in_agent_default": usable,
            "top_features": _variant_top_features(variant),
            "missing_flags": [],
            "counter_evidence": _status_counter_evidence(status),
            "source_ref_ids": ["positive_evidence_scorer_experiment_v1"],
            "train_valid_test_blocks": "walk_forward_H2023_2_to_H2026_1",
            "promotion_status": status,
            "research_only": True,
            "not_investment_instruction": True,
        }
        rows.append(sanitize_quant_tool_outcome(outcome))
    return rows


def write_rule_outcomes(path: Path, rows: list[dict[str, Any]], *, append: bool) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            leaked = set(row).intersection(FUTURE_RESULT_FIELDS)
            if leaked:
                raise ValueError(f"future field leaked into rule outcome: {sorted(leaked)}")
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _variant_feature_group(variant: str) -> str:
    if "kline" in variant:
        return "kline_peer_chip"
    if "positive" in variant:
        return "news_financial_peer_chip_bookskill"
    if "all_channel" in variant or "all_channels" in variant:
        return "all_channels"
    return "positive_evidence"


def _variant_top_features(variant: str) -> list[str]:
    if "kline" in variant:
        return ["kline_return_20d", "kline_return_60d", "corr_peer_avg_return_20d", "lower_support"]
    if "positive" in variant:
        return ["news_evidence_quality", "financial_report_event_count", "tushare_industry_positive_breadth_20d", "lower_support"]
    return ["news_evidence_quality", "kline_return_20d", "tushare_industry_relative_return_20d", "lower_support"]


def _status_score(status: str) -> float:
    return {"accepted_candidate": 0.8, "observe_prior_positive_latest_weak": 0.45}.get(status, 0.1)


def _status_confidence(status: str) -> float:
    return {"accepted_candidate": 0.75, "observe_prior_positive_latest_weak": 0.45}.get(status, 0.2)


def _status_counter_evidence(status: str) -> list[str]:
    if status == "accepted_candidate":
        return ["research_only", "requires_current_channel_confirmation", "no_investment_instruction"]
    if status == "observe_prior_positive_latest_weak":
        return ["latest_block_weak_or_cost_gate_failed", "observe_only", "requires_ablation_before_default"]
    return ["walk_forward_gate_failed", "diagnostic_only", "do_not_use_for_default_promotion"]


if __name__ == "__main__":
    main()
