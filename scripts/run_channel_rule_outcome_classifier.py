"""Train a local channel rule-outcome classifier.

This script is an offline research tool. Future 20-day outcomes are used only
to create walk-forward labels and evaluation reports. Agent-facing rule
outcomes are sanitized and do not contain future/result fields.
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

from src.agent_training.dual_mode_round import DEFAULT_JOINED_GT_CACHE_PATH, TIME_BLOCKS  # noqa: E402
from src.agent_training.quant_tool_context import FUTURE_RESULT_FIELDS, sanitize_quant_tool_outcome  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
OUTPUT_PREFIX = "channel_rule_outcome_classifier_v1"
LABELS = ["hard_counter", "soft_gap", "neutral", "positive_support"]
VALID_BLOCKS = list(TIME_BLOCKS.keys())[1:]
BLOCK_ORDER = list(TIME_BLOCKS.keys())
MIN_TRAIN_ROWS = 1000
MIN_VALID_ROWS = 200
DEFAULT_TOP_PCT = 0.15


FEATURE_GROUPS: dict[str, list[str]] = {
    "news_semantic": [
        "news_missing_rate",
        "news_count_30d",
        "news_warning_score",
        "news_opportunity_score",
        "policy_background_score",
        "official_confirmation_score",
        "announcement_materiality_score",
        "news_timestamp_quality",
        "news_evidence_quality",
        "news_conflict_intensity_30d",
        "tushare_industry_news_warning_avg",
        "tushare_industry_news_opportunity_avg",
        "tushare_industry_news_attention_gap",
        "tushare_area_news_warning_avg",
        "tushare_area_news_opportunity_avg",
        "tushare_area_news_attention_gap",
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
        "peer_relative_to_group_20d",
        "peer_group_positive_breadth_20d",
        "peer_group_above_ma200_rate",
        "corr_peer_avg_return_20d",
        "corr_peer_relative_return_20d",
        "corr_peer_positive_breadth_20d",
        "tushare_industry_relative_return_20d",
        "tushare_industry_positive_breadth_20d",
        "tushare_industry_above_ma200_rate",
        "tushare_area_relative_return_20d",
        "tushare_area_positive_breadth_20d",
        "tushare_area_above_ma200_rate",
    ],
    "chip_kline": [
        "lower_support",
        "chip_concentration",
        "cost_band_width",
        "upper_overhang",
        "winner_rate_pct",
        "neg_winner_rate",
        "kline_return_3d",
        "kline_return_5d",
        "kline_return_10d",
        "kline_return_20d",
        "kline_return_60d",
        "kline_drawdown_20d",
        "kline_drawdown_60d",
        "kline_range_position_20d",
        "kline_efficiency_ratio_20d",
        "kline_direction_reversal_rate_20d",
        "kline_oscillation_cross_count_20d",
        "kline_mean_reversion_z20",
        "kline_rsi14",
        "kline_atr20_pct",
        "kline_ma_gap_close_200",
    ],
    "book_python": [
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

NEGATIVE_ALIGNED_FEATURES = {
    "news_missing_rate",
    "news_warning_score",
    "news_conflict_intensity_30d",
    "financial_report_missing_rate",
    "financial_quality_risk_score",
    "financial_report_window_days",
    "upper_overhang",
    "cost_band_width",
    "neg_winner_rate",
    "counter_score",
    "relative_strength_rank",
    "kline_return_3d",
    "kline_return_5d",
    "kline_return_10d",
    "kline_return_20d",
    "kline_return_60d",
    "kline_drawdown_20d",
    "kline_drawdown_60d",
    "kline_mean_reversion_z20",
    "kline_atr20_pct",
}


@dataclass
class FittedOutcomeModel:
    variant: str
    features: list[str]
    scaler: StandardScaler | None
    model: Any
    classes: list[str]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run channel rule-outcome classifier walk-forward experiment.")
    parser.add_argument("--joined-cache", default=str(DEFAULT_JOINED_GT_CACHE_PATH))
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--model-family", choices=["manual_only", "logistic_only", "all"], default="all")
    parser.add_argument("--top-pct", type=float, default=DEFAULT_TOP_PCT)
    parser.add_argument("--rule-outcomes-output", default="")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_frame(Path(args.joined_cache))
    enriched = add_engineered_features(frame)
    coverage = feature_coverage(enriched)
    scored, step_metrics = run_walkforward(enriched, model_family=args.model_family, top_pct=args.top_pct)
    aggregate = aggregate_metrics(step_metrics)
    paths = write_outputs(
        prefix=args.output_prefix,
        scored=scored,
        step_metrics=step_metrics,
        aggregate=aggregate,
        coverage=coverage,
        top_pct=args.top_pct,
        model_family=args.model_family,
    )
    outcomes = build_rule_outcomes(aggregate)
    outcomes_path = Path(args.rule_outcomes_output) if args.rule_outcomes_output else REPORT_DIR / f"{args.output_prefix}_rule_outcomes.jsonl"
    write_rule_outcomes(outcomes_path, outcomes)

    print("A股研究Agent")
    print(f"rows={len(enriched)}")
    print(f"scored_rows={len(scored)}")
    print(f"step_metrics={len(step_metrics)}")
    print(f"report={paths['report']}")
    print(f"rule_outcomes={outcomes_path}")


def load_frame(path: Path) -> pd.DataFrame:
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
    frame["pool_mean_return_20d"] = frame.groupby(frame["date"].astype(str))["return_20d"].transform("mean")
    frame["pool_excess_20d"] = frame["return_20d"] - frame["pool_mean_return_20d"]
    frame["pool_return_rank_pct"] = frame.groupby(frame["date"].astype(str))["pool_excess_20d"].rank(pct=True, method="average")
    frame["rule_outcome_label"] = make_rule_outcome_label(frame)
    return frame.reset_index(drop=True)


def block_for_date(value: Any) -> str | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    for block, (start, end) in TIME_BLOCKS.items():
        if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
            return block
    return None


def make_rule_outcome_label(frame: pd.DataFrame) -> pd.Series:
    returns = pd.to_numeric(frame["return_20d"], errors="coerce")
    excess = pd.to_numeric(frame.get("pool_excess_20d", 0.0), errors="coerce").fillna(0.0)
    rank = pd.to_numeric(frame.get("pool_return_rank_pct", 0.5), errors="coerce").fillna(0.5)
    soft_gap_input = soft_gap_input_mask(frame)
    label = pd.Series("neutral", index=frame.index, dtype="object")
    label.loc[(returns <= -5.0) | (excess <= -5.0)] = "hard_counter"
    label.loc[(returns > 0.0) & (excess >= 2.0) & (rank >= 0.70)] = "positive_support"
    label.loc[soft_gap_input & returns.ge(-2.0) & excess.ge(-1.5) & label.eq("neutral")] = "soft_gap"
    return label


def soft_gap_input_mask(frame: pd.DataFrame) -> pd.Series:
    news_missing = pd.to_numeric(frame.get("news_missing_rate", 1.0), errors="coerce").fillna(1.0) >= 0.75
    fin_missing = pd.to_numeric(frame.get("financial_report_missing_rate", 1.0), errors="coerce").fillna(1.0) >= 0.75
    book_missing = frame.get("triggered_skills", pd.Series("", index=frame.index)).fillna("").astype(str).str.len().eq(0)
    weak_source = pd.to_numeric(frame.get("news_evidence_quality", 0.0), errors="coerce").fillna(0.0) < 0.35
    return news_missing | fin_missing | book_missing | weak_source


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for cols in FEATURE_GROUPS.values():
        for col in cols:
            if col not in out:
                out[col] = np.nan
    out["bookskill_available_flag"] = out.get("triggered_skills", pd.Series("", index=out.index)).fillna("").astype(str).str.len().gt(0).astype(float)
    out["news_soft_gap_flag"] = (pd.to_numeric(out.get("news_missing_rate", 1.0), errors="coerce").fillna(1.0) >= 0.75).astype(float)
    out["financial_soft_gap_flag"] = (pd.to_numeric(out.get("financial_report_missing_rate", 1.0), errors="coerce").fillna(1.0) >= 0.75).astype(float)
    out["peer_weak_flag"] = (pd.to_numeric(out.get("tushare_industry_positive_breadth_20d", 0.5), errors="coerce").fillna(0.5) < 0.4).astype(float)
    out["chip_overhang_flag"] = (pd.to_numeric(out.get("upper_overhang", 0.0), errors="coerce").fillna(0.0) > 0.6).astype(float)
    out["manual_hard_counter_score"] = sigmoid(
        0.85 * z(out, "news_warning_score")
        + 0.70 * z(out, "financial_quality_risk_score")
        + 0.55 * z(out, "upper_overhang")
        + 0.40 * out["peer_weak_flag"]
        + 0.25 * out["news_soft_gap_flag"]
        - 0.35 * z(out, "lower_support")
        - 0.25 * z(out, "official_confirmation_score")
    )
    out["manual_positive_support_score"] = sigmoid(
        -0.45 * z(out, "kline_return_20d")
        -0.35 * z(out, "kline_return_60d")
        + 0.45 * z(out, "lower_support")
        + 0.40 * z(out, "tushare_industry_positive_breadth_20d")
        + 0.30 * z(out, "financial_surprise_score")
        + 0.25 * z(out, "official_confirmation_score")
        - 0.35 * z(out, "upper_overhang")
    )
    out["manual_soft_gap_score"] = (
        (out["news_soft_gap_flag"] + out["financial_soft_gap_flag"] + (1.0 - out["bookskill_available_flag"])) / 3.0
    ) * (1.0 - out["manual_hard_counter_score"])
    return out


def sigmoid(value: pd.Series) -> pd.Series:
    clipped = pd.to_numeric(value, errors="coerce").fillna(0.0).clip(-8, 8)
    return 1.0 / (1.0 + np.exp(-clipped))


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
        for col in cols:
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


def all_model_features(frame: pd.DataFrame) -> list[str]:
    features = sorted({col for cols in FEATURE_GROUPS.values() for col in cols if col in frame})
    features.extend(
        [
            "bookskill_available_flag",
            "news_soft_gap_flag",
            "financial_soft_gap_flag",
            "peer_weak_flag",
            "chip_overhang_flag",
            "manual_hard_counter_score",
            "manual_positive_support_score",
            "manual_soft_gap_score",
        ]
    )
    return [col for col in dict.fromkeys(features) if col in frame]


def run_walkforward(frame: pd.DataFrame, *, model_family: str, top_pct: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored_parts = []
    metric_rows = []
    for valid_block in VALID_BLOCKS:
        train_blocks = BLOCK_ORDER[: BLOCK_ORDER.index(valid_block)]
        train = frame[frame["time_block"].isin(train_blocks)].copy()
        valid = frame[frame["time_block"].eq(valid_block)].copy()
        if len(train) < MIN_TRAIN_ROWS or len(valid) < MIN_VALID_ROWS:
            continue
        models = fit_models(train, model_family=model_family)
        scored = valid.copy()
        add_manual_probabilities(scored)
        for model in models:
            add_model_probabilities(scored, model)
        scored["train_blocks"] = "+".join(train_blocks)
        scored["valid_block"] = valid_block
        scored_parts.append(scored)
        variants = ["manual", *[model.variant for model in models]]
        for variant in variants:
            for action_class in ["hard_counter", "soft_gap", "positive_support"]:
                metric_rows.append(evaluate_action(scored, variant, action_class, top_pct=top_pct, train_blocks=train_blocks, valid_block=valid_block))
    return (
        pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame(),
        pd.DataFrame(metric_rows),
    )


def fit_models(train: pd.DataFrame, *, model_family: str) -> list[FittedOutcomeModel]:
    if model_family == "manual_only":
        return []
    y = train["rule_outcome_label"].astype(str)
    if y.nunique() < 2:
        return []
    features = all_model_features(train)
    x = build_matrix(train, features)
    y = y.loc[x.index]
    if len(x) < MIN_TRAIN_ROWS:
        return []
    fitted: list[FittedOutcomeModel] = []
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    logistic = LogisticRegression(max_iter=600, class_weight="balanced", random_state=42)
    logistic.fit(x_scaled, y)
    fitted.append(FittedOutcomeModel("logistic_channel_outcome", list(x.columns), scaler, logistic, [str(c) for c in logistic.classes_]))
    if model_family == "all":
        gbdt = HistGradientBoostingClassifier(max_iter=80, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.2, random_state=42)
        gbdt.fit(x, y)
        fitted.append(FittedOutcomeModel("gbdt_channel_outcome", list(x.columns), None, gbdt, [str(c) for c in gbdt.classes_]))
    return fitted


def build_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    data = {}
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        if feature in NEGATIVE_ALIGNED_FEATURES:
            values = -values
        median = values.median()
        data[feature] = values.fillna(0.0 if pd.isna(median) else median)
    x = pd.DataFrame(data, index=frame.index)
    nunique = x.nunique(dropna=True)
    cols = [col for col in x.columns if nunique[col] >= 2]
    return x[cols]


def add_manual_probabilities(frame: pd.DataFrame) -> None:
    hard = pd.to_numeric(frame["manual_hard_counter_score"], errors="coerce").fillna(0.0)
    positive = pd.to_numeric(frame["manual_positive_support_score"], errors="coerce").fillna(0.0)
    soft = pd.to_numeric(frame["manual_soft_gap_score"], errors="coerce").fillna(0.0)
    neutral = (1.0 - pd.concat([hard, positive, soft], axis=1).max(axis=1)).clip(lower=0.05)
    total = hard + positive + soft + neutral
    frame["manual__prob_hard_counter"] = hard / total
    frame["manual__prob_positive_support"] = positive / total
    frame["manual__prob_soft_gap"] = soft / total
    frame["manual__prob_neutral"] = neutral / total


def add_model_probabilities(frame: pd.DataFrame, model: FittedOutcomeModel) -> None:
    x = build_matrix(frame, model.features).reindex(columns=model.features, fill_value=0.0)
    if model.scaler is not None:
        probs = model.model.predict_proba(model.scaler.transform(x))
    else:
        probs = model.model.predict_proba(x)
    for idx, class_name in enumerate(model.classes):
        frame[f"{model.variant}__prob_{class_name}"] = 0.0
        frame.loc[x.index, f"{model.variant}__prob_{class_name}"] = probs[:, idx]
    for class_name in LABELS:
        col = f"{model.variant}__prob_{class_name}"
        if col not in frame:
            frame[col] = 0.0


def evaluate_action(
    valid: pd.DataFrame,
    variant: str,
    action_class: str,
    *,
    top_pct: float,
    train_blocks: list[str],
    valid_block: str,
) -> dict[str, Any]:
    prob_col = f"{variant}__prob_{action_class}"
    score = pd.to_numeric(valid.get(prob_col, 0.0), errors="coerce").fillna(0.0)
    selected = select_top_by_date(valid, score, top_pct=top_pct)
    returns = pd.to_numeric(valid["return_20d"], errors="coerce")
    selected_returns = pd.to_numeric(selected["return_20d"], errors="coerce") if not selected.empty else pd.Series(dtype=float)
    selected_excess = pd.to_numeric(selected.get("pool_excess_20d", pd.Series(dtype=float)), errors="coerce")
    baseline_positive = float((returns > 0).mean()) if not returns.empty else np.nan
    baseline_loss = float((returns <= -5.0).mean()) if not returns.empty else np.nan
    baseline_avg = float(returns.mean()) if not returns.empty else np.nan
    concentration = selected["code"].astype(str).value_counts(normalize=True).max() if not selected.empty else np.nan
    return {
        "variant": variant,
        "action_class": action_class,
        "train_blocks": "+".join(train_blocks),
        "valid_block": valid_block,
        "candidate_rows": int(len(valid)),
        "selected_rows": int(len(selected)),
        "coverage_dates": int(selected["date"].nunique()) if not selected.empty else 0,
        "unique_stocks": int(selected["code"].nunique()) if not selected.empty else 0,
        "top_stock_concentration": round(float(concentration), 4) if not pd.isna(concentration) else np.nan,
        "baseline_positive_20d_rate": round(baseline_positive, 6) if not pd.isna(baseline_positive) else np.nan,
        "selected_positive_20d_rate": round(float((selected_returns > 0).mean()), 6) if not selected_returns.empty else np.nan,
        "positive_rate_lift": round(float((selected_returns > 0).mean() - baseline_positive), 6) if not selected_returns.empty else np.nan,
        "baseline_loss_gt5_rate": round(baseline_loss, 6) if not pd.isna(baseline_loss) else np.nan,
        "selected_loss_gt5_rate": round(float((selected_returns <= -5.0).mean()), 6) if not selected_returns.empty else np.nan,
        "loss_gt5_lift": round(float((selected_returns <= -5.0).mean() - baseline_loss), 6) if not selected_returns.empty else np.nan,
        "baseline_avg_return_20d": round(baseline_avg, 6) if not pd.isna(baseline_avg) else np.nan,
        "selected_avg_return_20d": round(float(selected_returns.mean()), 6) if not selected_returns.empty else np.nan,
        "selected_pool_excess_20d": round(float(selected_excess.mean()), 6) if not selected_excess.empty else np.nan,
        "mean_predicted_probability": round(float(score.loc[selected.index].mean()), 6) if not selected.empty else np.nan,
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
        return data.iloc[0:0].drop(columns=["_score"])
    return pd.concat(selected, ignore_index=False).drop(columns=["_score"])


def aggregate_metrics(step_metrics: pd.DataFrame) -> pd.DataFrame:
    if step_metrics.empty:
        return pd.DataFrame()
    rows = []
    for (variant, action_class), group in step_metrics.groupby(["variant", "action_class"], sort=True):
        prior = group[~group["valid_block"].eq("H2026_1")]
        h2026 = group[group["valid_block"].eq("H2026_1")]
        rows.append(
            {
                "variant": variant,
                "action_class": action_class,
                "blocks": int(group["valid_block"].nunique()),
                "prior_blocks": int(prior["valid_block"].nunique()),
                "mean_positive_rate_lift": _mean(group, "positive_rate_lift"),
                "prior_positive_rate_lift": _mean(prior, "positive_rate_lift"),
                "h2026_positive_rate_lift": _mean(h2026, "positive_rate_lift"),
                "mean_loss_gt5_lift": _mean(group, "loss_gt5_lift"),
                "prior_loss_gt5_lift": _mean(prior, "loss_gt5_lift"),
                "h2026_loss_gt5_lift": _mean(h2026, "loss_gt5_lift"),
                "mean_selected_pool_excess_20d": _mean(group, "selected_pool_excess_20d"),
                "prior_selected_pool_excess_20d": _mean(prior, "selected_pool_excess_20d"),
                "h2026_selected_pool_excess_20d": _mean(h2026, "selected_pool_excess_20d"),
                "mean_selected_avg_return_20d": _mean(group, "selected_avg_return_20d"),
                "h2026_selected_avg_return_20d": _mean(h2026, "selected_avg_return_20d"),
                "mean_selected_rows": _mean(group, "selected_rows"),
                "max_top_stock_concentration": _max(group, "top_stock_concentration"),
                "promotion_status": promotion_status(group, action_class),
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return pd.DataFrame(rows).sort_values(["promotion_status", "variant", "action_class"]).reset_index(drop=True)


def _mean(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame:
        return np.nan
    return round(float(pd.to_numeric(frame[col], errors="coerce").mean()), 6)


def _max(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame:
        return np.nan
    return round(float(pd.to_numeric(frame[col], errors="coerce").max()), 6)


def promotion_status(group: pd.DataFrame, action_class: str) -> str:
    prior = group[~group["valid_block"].eq("H2026_1")]
    h2026 = group[group["valid_block"].eq("H2026_1")]
    if prior.empty or h2026.empty:
        return "observe_insufficient_blocks"
    concentration = _max(group, "top_stock_concentration")
    prior_pool = _mean(prior, "selected_pool_excess_20d")
    h_pool = _mean(h2026, "selected_pool_excess_20d")
    if action_class == "positive_support":
        prior_lift = _mean(prior, "positive_rate_lift")
        h_lift = _mean(h2026, "positive_rate_lift")
        if prior_lift >= 0.03 and h_lift >= 0.03 and prior_pool > 0 and h_pool > 0 and concentration <= 0.25:
            return "accepted_positive_support_candidate"
        if prior_lift >= 0.03 and prior_pool > 0:
            return "observe_prior_positive_latest_weak"
    if action_class == "hard_counter":
        prior_loss = _mean(prior, "loss_gt5_lift")
        h_loss = _mean(h2026, "loss_gt5_lift")
        if prior_loss >= 0.05 and h_loss >= 0.03 and prior_pool < 0 and h_pool < 0 and concentration <= 0.35:
            return "accepted_guard_candidate"
        if prior_loss >= 0.05 and prior_pool < 0:
            return "observe_guard_prior_positive_latest_weak"
    if action_class == "soft_gap":
        prior_avg = _mean(prior, "selected_pool_excess_20d")
        h_avg = _mean(h2026, "selected_pool_excess_20d")
        if prior_avg >= -0.5 and h_avg >= -0.5:
            return "accepted_soft_gap_hygiene_candidate"
        if prior_avg >= -0.5:
            return "observe_soft_gap_prior_ok_latest_weak"
    return "rejected_or_diagnostic_only"


def write_outputs(
    *,
    prefix: str,
    scored: pd.DataFrame,
    step_metrics: pd.DataFrame,
    aggregate: pd.DataFrame,
    coverage: pd.DataFrame,
    top_pct: float,
    model_family: str,
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
        "valid_block",
        "train_blocks",
        "rule_outcome_label",
        "return_20d",
        "pool_excess_20d",
        "manual__prob_hard_counter",
        "manual__prob_soft_gap",
        "manual__prob_positive_support",
    ]
    prob_cols = [col for col in scored.columns if "__prob_" in col and col not in detail_cols]
    scored[[col for col in [*detail_cols, *prob_cols] if col in scored]].to_csv(detail_path, index=False, encoding="utf-8-sig")
    step_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    write_report(report_path, aggregate, step_metrics, coverage, top_pct=top_pct, model_family=model_family)
    return {"detail": detail_path, "metrics": metrics_path, "aggregate": aggregate_path, "coverage": coverage_path, "report": report_path}


def write_report(path: Path, aggregate: pd.DataFrame, step_metrics: pd.DataFrame, coverage: pd.DataFrame, *, top_pct: float, model_family: str) -> None:
    lines = [
        "# Channel Rule Outcome Classifier v1",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Setup",
        "",
        f"- model_family: `{model_family}`",
        f"- per-date review slice: top `{top_pct:.1%}` by predicted action probability",
        "- split: walk-forward by half-year block; each valid block trains only on prior blocks",
        "- labels: offline-only `hard_counter / soft_gap / neutral / positive_support` from future 20d outcomes",
        "- Agent-facing rule outcomes are sanitized and contain no future/result fields",
        "",
        "## Aggregate",
        "",
        _table(aggregate),
        "",
        "## Step Metrics",
        "",
        _table(step_metrics),
        "",
        "## Lowest Feature Coverage",
        "",
        _table(coverage.sort_values("non_null_rate").head(16) if not coverage.empty else coverage),
        "",
        "## Interpretation",
        "",
        "- `hard_counter` 有效时，应作为降权/反证工具候选，而不是正向选择工具。",
        "- `soft_gap` 有效时，只证明缺失不应被机械当成方向性负面；不构成正向 alpha。",
        "- `positive_support` 只有在 prior blocks 和 H2026/OOT 都稳定提升时，才允许进入 DS ablation。",
        "- 若 v1 不过线，下一轮优先改善标签、输入覆盖和单支/组合分任务，而不是 prompt-forcing Agent。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_无数据_"
    return frame.to_markdown(index=False)


def build_rule_outcomes(aggregate: pd.DataFrame) -> list[dict[str, Any]]:
    if aggregate.empty:
        return []
    rows = []
    for _, row in aggregate.iterrows():
        status = str(row.get("promotion_status", ""))
        if status == "rejected_or_diagnostic_only":
            continue
        action_class = str(row.get("action_class", ""))
        action_hint = {
            "hard_counter": "use_as_counterevidence_guard_observe",
            "soft_gap": "treat_missing_as_confidence_discount_observe",
            "positive_support": "use_as_positive_support_observe_only",
        }.get(action_class, "observe_only")
        item = {
            "tool_id": f"channel_rule_outcome_classifier_v1_{row.get('variant')}_{action_class}",
            "tool_version": "channel_rule_outcome_classifier_v1",
            "task_mode": "portfolio_pool",
            "feature_group": "news_financial_peer_chip_kline_bookskill",
            "selection_mode": action_class,
            "score": 1.0 if status.startswith("accepted") else 0.5,
            "confidence": 0.65 if status.startswith("accepted") else 0.4,
            "action_hint": action_hint,
            "usable_in_agent_default": False,
            "top_features": [
                "news_missing_rate",
                "financial_report_missing_rate",
                "tushare_industry_positive_breadth_20d",
                "upper_overhang",
                "lower_support",
                "bookskill_available_flag",
            ],
            "missing_flags": [],
            "counter_evidence": [
                "v1_observe_only",
                "requires_ds_ablation_before_default",
                "future_labels_not_exposed_to_agent",
            ],
            "source_ref_ids": ["reports/date_generalization/channel_rule_outcome_classifier_v1.md"],
            "train_valid_test_blocks": "walk_forward_prior_blocks; H2026/OOT reported offline only",
            "promotion_status": status,
            "research_only": True,
            "not_investment_instruction": True,
        }
        rows.append(sanitize_quant_tool_outcome(item))
    return rows


def write_rule_outcomes(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            leaked = future_keys(row)
            if leaked:
                raise ValueError(f"future fields leaked to rule outcome: {sorted(leaked)}")
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
