"""Audit a lightweight ML confirmer for the P0 small-entry branch.

This experiment is local-only and no-DeepSeek. It starts from the strongest
current P0 user-facing branch, `branch_stack_v1.small_buy_hold`, and asks a
bounded question: can a simple, pre-registered ML confirmer separate better
small-entry candidates without turning the branch into a brittle hard filter?

Future returns are used only for offline evaluation. Agent preview rows are
strictly field-whitelisted and contain no GT/future/result fields.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_p0_decision_stack_v1 import (  # noqa: E402
    FUTURE_OR_RESULT_FIELDS,
    FINAL_OOT,
    MIN_TARGET_ROWS,
    MIN_TRAIN_ROWS,
    MIN_VALID_ROWS,
    TARGET_BLOCKS,
    apply_frequency,
    apply_policy,
    build_scored_target,
    load_stack_frame,
    safe_float,
    safe_prefix,
    stable_hash_int,
)
from scripts.audit_p0_operation_policy_v1 import with_operation_actions  # noqa: E402
from scripts.audit_single_stock_review_quality import _rolling_split  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_PREFIX = "p0_small_entry_ml_confirmer_v1"
DEFAULT_FREQUENCIES = "weekly_friday,every_2_weeks,weekly_tuesday"
DEFAULT_KLINE_GROUP = "kline_peer_chip_news_risk"
MAX_HGB_TRAIN_ROWS = 60000
MIN_CONFIRMER_TRAIN_ROWS = 80
MIN_CONFIRMER_VALID_ROWS = 30
MIN_CONFIRMER_TARGET_ROWS = 30
MIN_PROMOTION_PRIOR_BLOCKS = 2
MIN_PROMOTION_PRIOR_SELECTED_ROWS_MEAN = 30
PANEL_SIZE = 100
PANEL_SEEDS = 12
CONFIRM_QUANTILES = (0.50, 0.60, 0.70)

STACK_FEATURES = [
    "target_position",
    "opp_score",
    "opp_threshold",
    "opp_margin",
    "opp_quantile_in_date",
    "kline_opp_score",
    "kline_opp_threshold",
    "kline_opp_margin",
    "kline_risk_score",
    "kline_risk_threshold",
    "kline_risk_margin",
    "risk_score",
    "review_priority_score",
    "channel_hard_counter_prob",
    "channel_soft_gap_prob",
    "channel_positive_support_prob",
    "channel_score_coverage",
]

NEWS_FIN_FEATURES = [
    "event_count",
    "self_news_intensity",
    "news_warning_score",
    "news_opportunity_score",
    "news_missing_rate",
    "news_evidence_quality",
    "policy_background_score",
    "official_confirmation_score",
    "announcement_materiality_score",
    "news_timestamp_quality",
    "financial_report_missing_rate",
    "financial_quality_risk_score",
    "financial_surprise_score",
    "financial_disclosure_quality_score",
    "financial_report_materiality_score",
]

PEER_CHIP_KLINE_FEATURES = [
    "corr_peer_relative_return_20d",
    "corr_peer_avg_return_20d",
    "corr_peer_positive_breadth_20d",
    "tushare_industry_relative_return_20d",
    "tushare_industry_positive_breadth_20d",
    "tushare_area_relative_return_20d",
    "tushare_area_positive_breadth_20d",
    "lower_support",
    "upper_overhang",
    "winner_rate_pct",
    "cost_band_width",
    "chip_concentration",
    "kline_rsi14",
    "kline_return_5d",
    "kline_return_20d",
    "kline_return_60d",
    "kline_drawdown_20d",
    "kline_drawdown_60d",
    "kline_range_position_20d",
    "kline_range_position_60d",
    "kline_volatility_ratio_20_60",
    "kline_efficiency_ratio_20d",
]


@dataclass(frozen=True)
class FeatureSet:
    feature_set: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    penalty: str
    c_value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit lightweight ML confirmation on P0 small-entry branch.")
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--frequencies", default=DEFAULT_FREQUENCIES)
    parser.add_argument("--kline-feature-group", default=DEFAULT_KLINE_GROUP)
    parser.add_argument("--max-hgb-train-rows", type=int, default=MAX_HGB_TRAIN_ROWS)
    parser.add_argument("--panel-size", type=int, default=PANEL_SIZE)
    parser.add_argument("--panel-seeds", type=int, default=PANEL_SEEDS)
    parser.add_argument("--preview-max-rows", type=int, default=800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("ignore", message="Skipping features without any observed values.*", module="sklearn")
    warnings.filterwarnings("ignore", message="Inconsistent values: penalty=.*", module="sklearn")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frequencies = [item.strip() for item in args.frequencies.split(",") if item.strip()]
    frame, feature_groups, notes = load_stack_frame()

    all_small_blocks: list[pd.DataFrame] = []
    hygiene_rows: list[dict[str, Any]] = []
    for frequency in frequencies:
        freq_frame = apply_frequency(frame, frequency)
        for target_block in TARGET_BLOCKS:
            train, validation, target = _rolling_split(freq_frame, target_block)
            if len(train) < MIN_TRAIN_ROWS or len(validation) < MIN_VALID_ROWS or len(target) < MIN_TARGET_ROWS:
                hygiene_rows.append(
                    {
                        "frequency": frequency,
                        "target_block": target_block,
                        "stage": "base_stack",
                        "status": "skip_insufficient_stack_rows",
                        "train_rows": len(train),
                        "validation_rows": len(validation),
                        "target_rows": len(target),
                    }
                )
                continue
            scored = build_scored_target(
                train,
                validation,
                target,
                feature_groups,
                kline_feature_group=args.kline_feature_group,
                max_hgb_train_rows=args.max_hgb_train_rows,
            )
            if scored.empty:
                hygiene_rows.append(
                    {
                        "frequency": frequency,
                        "target_block": target_block,
                        "stage": "base_stack",
                        "status": "skip_stack_model_unavailable",
                        "train_rows": len(train),
                        "validation_rows": len(validation),
                        "target_rows": len(target),
                    }
                )
                continue
            small = build_small_entry_frame(scored, target, frequency, target_block)
            all_small_blocks.append(small)

    small_frame = pd.concat(all_small_blocks, ignore_index=True) if all_small_blocks else pd.DataFrame()
    metric_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []

    if not small_frame.empty:
        for frequency in frequencies:
            freq_small = small_frame[small_frame["frequency"].eq(frequency)].copy()
            metric_rows.extend(evaluate_baselines(freq_small))
            for target_block in sorted(freq_small["target_block"].unique(), key=block_index):
                train, validation, split_context = prior_tail_train_validation(freq_small, target_block)
                target = freq_small[freq_small["target_block"].eq(target_block)].copy()
                if (
                    len(train) < MIN_CONFIRMER_TRAIN_ROWS
                    or len(validation) < MIN_CONFIRMER_VALID_ROWS
                    or len(target) < MIN_CONFIRMER_TARGET_ROWS
                ):
                    hygiene_rows.append(
                        {
                            "frequency": frequency,
                            "target_block": target_block,
                            "stage": "ml_confirmer",
                            "status": "skip_insufficient_small_entry_rows",
                            "split_context": split_context,
                            "train_rows": len(train),
                            "validation_rows": len(validation),
                            "target_rows": len(target),
                        }
                    )
                    continue
                for feature_set in feature_sets_for(freq_small):
                    for model_spec in model_specs():
                        if len(feature_set.columns) < 3:
                            continue
                        fitted = fit_logistic(train, feature_set.columns, model_spec)
                        if fitted is None:
                            hygiene_rows.append(
                                {
                                    "frequency": frequency,
                                    "target_block": target_block,
                                    "stage": "ml_confirmer",
                                    "status": "skip_single_class_or_fit_failed",
                                    "feature_set": feature_set.feature_set,
                                    "model_name": model_spec.model_name,
                                    "train_rows": len(train),
                                    "validation_rows": len(validation),
                                    "target_rows": len(target),
                                }
                            )
                            continue
                        validation_scores = predict_score(fitted, validation, feature_set.columns)
                        target_scores = predict_score(fitted, target, feature_set.columns)
                        validation = validation.assign(_confirmer_score=validation_scores)
                        target_scored = target.assign(_confirmer_score=target_scores)
                        feature_rows.extend(feature_importance_rows(fitted, feature_set, model_spec, frequency, target_block))
                        for quantile in CONFIRM_QUANTILES:
                            threshold = validation_threshold(validation["_confirmer_score"], quantile)
                            variant = variant_id(feature_set.feature_set, model_spec.model_name, quantile)
                            selected = target_scored[target_scored["_confirmer_score"] >= threshold].copy()
                            metric_rows.append(
                                evaluate_variant(
                                    base=target,
                                    selected=selected,
                                    frequency=frequency,
                                    target_block=target_block,
                                    variant=variant,
                                    feature_set=feature_set.feature_set,
                                    model_name=model_spec.model_name,
                                    confirm_quantile=quantile,
                                    score_threshold=threshold,
                                    train_blocks=sorted(train["target_block"].astype(str).unique()),
                                    validation_block=split_context,
                                    selected_feature_count=len(feature_set.columns),
                                )
                            )
                            if target_block == FINAL_OOT:
                                panel_rows.extend(
                                    panel_stability(
                                        target_scored,
                                        frequency=frequency,
                                        variant=variant,
                                        threshold=threshold,
                                        panel_size=args.panel_size,
                                        panel_seeds=args.panel_seeds,
                                    )
                                )
                                if frequency == "weekly_friday" and quantile in {0.50, 0.60}:
                                    preview_rows.extend(
                                        build_preview_rows(
                                            target_scored,
                                            frequency=frequency,
                                            variant=variant,
                                            threshold=threshold,
                                            feature_set=feature_set.feature_set,
                                            max_rows=args.preview_max_rows // 4,
                                        )
                                    )

    metrics = pd.DataFrame(metric_rows)
    panels = pd.DataFrame(panel_rows)
    preview = pd.DataFrame(preview_rows)
    features = pd.DataFrame(feature_rows)
    hygiene = pd.DataFrame(hygiene_rows)
    summary = summarize(metrics)
    panel_summary = summarize_panels(panels)
    prefix = safe_prefix(args.output_prefix)
    paths = {
        "metrics": REPORT_DIR / f"{prefix}_metrics.csv",
        "summary": REPORT_DIR / f"{prefix}_summary.csv",
        "panel_detail": REPORT_DIR / f"{prefix}_h2026_panel_detail.csv",
        "panel_summary": REPORT_DIR / f"{prefix}_h2026_panel_summary.csv",
        "feature_importance": REPORT_DIR / f"{prefix}_feature_importance.csv",
        "preview": REPORT_DIR / f"{prefix}_agent_preview_no_gt.jsonl",
        "hygiene": REPORT_DIR / f"{prefix}_hygiene.csv",
        "report": REPORT_DIR / f"{prefix}.md",
    }
    metrics.to_csv(paths["metrics"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    panels.to_csv(paths["panel_detail"], index=False, encoding="utf-8-sig")
    panel_summary.to_csv(paths["panel_summary"], index=False, encoding="utf-8-sig")
    features.to_csv(paths["feature_importance"], index=False, encoding="utf-8-sig")
    hygiene.to_csv(paths["hygiene"], index=False, encoding="utf-8-sig")
    write_jsonl(paths["preview"], preview)
    paths["report"].write_text(render_report(args, notes, small_frame, summary, metrics, panel_summary, hygiene, paths), encoding="utf-8")
    print("A股研究Agent")
    print(f"small_rows={len(small_frame)} metrics={len(metrics)} summary={len(summary)} preview={len(preview)}")
    print(f"report={paths['report']}")


def build_small_entry_frame(scored: pd.DataFrame, target: pd.DataFrame, frequency: str, target_block: str) -> pd.DataFrame:
    branch = with_operation_actions(apply_policy(scored, "branch_stack_v1"))
    small = branch[branch["operation_action"].astype(str).eq("small_buy_hold")].copy()
    if small.empty:
        return small
    small["frequency"] = frequency
    small["target_block"] = target_block
    small["code"] = small["code"].astype(str).str.zfill(6)
    enriched = add_stack_derived_features(small)
    feature_cols = safe_feature_columns(target)
    raw = target[["date", "code", *feature_cols]].copy()
    raw["code"] = raw["code"].astype(str).str.zfill(6)
    out = enriched.merge(raw.drop_duplicates(["date", "code"]), on=["date", "code"], how="left")
    out["positive_20d"] = pd.to_numeric(out["return_20d"], errors="coerce").gt(0).astype(int)
    return out


def add_stack_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["opp_margin"] = num(out, "opp_score") - num(out, "opp_threshold")
    out["kline_opp_margin"] = num(out, "kline_opp_score") - num(out, "kline_opp_threshold")
    out["kline_risk_margin"] = num(out, "kline_risk_threshold") - num(out, "kline_risk_score")
    for col in ["risk_review_queue", "risk_queue_high_hard_counter", "kline_hard_risk", "opp_active", "kline_active"]:
        if col in out:
            out[col] = out[col].astype(int)
    return out


def safe_feature_columns(frame: pd.DataFrame) -> list[str]:
    wanted = [*NEWS_FIN_FEATURES, *PEER_CHIP_KLINE_FEATURES]
    return [col for col in dict.fromkeys(wanted) if col in frame.columns and not forbidden_field(col)]


def feature_sets_for(frame: pd.DataFrame) -> list[FeatureSet]:
    available = set(frame.columns)
    stack = tuple(col for col in STACK_FEATURES if col in available and not forbidden_field(col))
    news_fin = tuple(col for col in NEWS_FIN_FEATURES if col in available and not forbidden_field(col))
    peer_chip = tuple(col for col in PEER_CHIP_KLINE_FEATURES if col in available and not forbidden_field(col))
    return [
        FeatureSet("stack_margins_only", stack),
        FeatureSet("stack_plus_news_fin", tuple(dict.fromkeys([*stack, *news_fin]))),
        FeatureSet("stack_plus_peer_chip_kline", tuple(dict.fromkeys([*stack, *peer_chip]))),
        FeatureSet("stack_plus_registered_all", tuple(dict.fromkeys([*stack, *news_fin, *peer_chip]))),
    ]


def model_specs() -> list[ModelSpec]:
    return [
        ModelSpec("logistic_l2_c050", penalty="l2", c_value=0.50),
        ModelSpec("logistic_l1_c005", penalty="l1", c_value=0.05),
    ]


def fit_logistic(train: pd.DataFrame, features: tuple[str, ...], spec: ModelSpec) -> Pipeline | None:
    y = pd.to_numeric(train["positive_20d"], errors="coerce").fillna(0).astype(int)
    if y.nunique() < 2:
        return None
    solver = "liblinear" if spec.penalty == "l1" else "lbfgs"
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=spec.c_value,
                    penalty=spec.penalty,
                    solver=solver,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=17,
                ),
            ),
        ]
    )
    try:
        model.fit(train.loc[:, features], y)
    except Exception:
        return None
    return model


def predict_score(model: Pipeline, frame: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    return model.predict_proba(frame.loc[:, features])[:, 1]


def validation_threshold(scores: pd.Series | np.ndarray, quantile: float) -> float:
    values = pd.to_numeric(pd.Series(scores), errors="coerce").dropna()
    if values.empty:
        return float("inf")
    return float(values.quantile(float(quantile)))


def evaluate_baselines(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for keys, group in frame.groupby(["frequency", "target_block"], sort=True):
        rows.append(
            evaluate_variant(
                base=group,
                selected=group,
                frequency=str(keys[0]),
                target_block=str(keys[1]),
                variant="small_entry_all",
                feature_set="branch_reference",
                model_name="none",
                confirm_quantile=0.0,
                score_threshold=np.nan,
                train_blocks=[],
                validation_block="none",
                selected_feature_count=0,
            )
        )
    return rows


def evaluate_variant(
    *,
    base: pd.DataFrame,
    selected: pd.DataFrame,
    frequency: str,
    target_block: str,
    variant: str,
    feature_set: str,
    model_name: str,
    confirm_quantile: float,
    score_threshold: float,
    train_blocks: list[str],
    validation_block: str,
    selected_feature_count: int,
) -> dict[str, Any]:
    base_ret = pd.to_numeric(base["return_20d"], errors="coerce").dropna()
    ret = pd.to_numeric(selected["return_20d"], errors="coerce").dropna()
    excluded = base[~base.index.isin(selected.index)].copy()
    excluded_ret = pd.to_numeric(excluded["return_20d"], errors="coerce").dropna()
    return {
        "frequency": frequency,
        "target_block": target_block,
        "variant": variant,
        "feature_set": feature_set,
        "model_name": model_name,
        "confirm_quantile": round(float(confirm_quantile), 4),
        "score_threshold": round(float(score_threshold), 8) if not pd.isna(score_threshold) else np.nan,
        "train_blocks": ";".join(train_blocks),
        "validation_block": validation_block,
        "selected_feature_count": int(selected_feature_count),
        "base_rows": int(len(base)),
        "selected_rows": int(len(selected)),
        "selected_rate": round(float(len(selected) / max(1, len(base))), 6),
        "base_pos20": positive_rate(base_ret),
        "base_avg20": mean_value(base_ret),
        "base_loss_gt5": rate_le(base_ret, -5),
        "base_gain_gt5": rate_ge(base_ret, 5),
        "selected_pos20": positive_rate(ret),
        "selected_avg20": mean_value(ret),
        "selected_loss_gt5": rate_le(ret, -5),
        "selected_gain_gt5": rate_ge(ret, 5),
        "delta_pos20_vs_base": delta(positive_rate(ret), positive_rate(base_ret)),
        "delta_avg20_vs_base": delta(mean_value(ret), mean_value(base_ret)),
        "delta_loss_gt5_vs_base": delta(rate_le(ret, -5), rate_le(base_ret, -5)),
        "excluded_rows": int(len(excluded)),
        "excluded_positive_rate": positive_rate(excluded_ret),
        "missed_positive_rows": int((excluded_ret > 0).sum()) if len(excluded_ret) else 0,
        "missed_large_gain_rows": int((excluded_ret >= 5).sum()) if len(excluded_ret) else 0,
    }


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in metrics.groupby(["frequency", "variant"], sort=True):
        h = group[group["target_block"].eq(FINAL_OOT)]
        prior = group[~group["target_block"].eq(FINAL_OOT)]
        hrow = h.iloc[0] if not h.empty else pd.Series(dtype=object)
        row = {
            "frequency": keys[0],
            "variant": keys[1],
            "feature_set": first(group, "feature_set"),
            "model_name": first(group, "model_name"),
            "confirm_quantile": get_val(group.iloc[0], "confirm_quantile"),
            "prior_blocks": int(prior["target_block"].nunique()),
            "prior_selected_rows_mean": mean_col(prior, "selected_rows"),
            "prior_selected_rate_mean": mean_col(prior, "selected_rate"),
            "prior_delta_pos_mean": mean_col(prior, "delta_pos20_vs_base"),
            "prior_delta_avg_mean": mean_col(prior, "delta_avg20_vs_base"),
            "prior_delta_pos_hit": hit_rate(prior, "delta_pos20_vs_base", 0),
            "prior_delta_avg_hit": hit_rate(prior, "delta_avg20_vs_base", 0),
            "h2026_base_rows": get_val(hrow, "base_rows"),
            "h2026_selected_rows": get_val(hrow, "selected_rows"),
            "h2026_selected_rate": get_val(hrow, "selected_rate"),
            "h2026_base_pos20": get_val(hrow, "base_pos20"),
            "h2026_selected_pos20": get_val(hrow, "selected_pos20"),
            "h2026_selected_avg20": get_val(hrow, "selected_avg20"),
            "h2026_selected_loss_gt5": get_val(hrow, "selected_loss_gt5"),
            "h2026_delta_pos": get_val(hrow, "delta_pos20_vs_base"),
            "h2026_delta_avg": get_val(hrow, "delta_avg20_vs_base"),
            "h2026_delta_loss": get_val(hrow, "delta_loss_gt5_vs_base"),
            "h2026_missed_positive_rows": get_val(hrow, "missed_positive_rows"),
            "h2026_missed_large_gain_rows": get_val(hrow, "missed_large_gain_rows"),
        }
        row["promotion_status"] = promotion_status(row)
        row["rank_score"] = rank_score(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["promotion_status", "rank_score"], ascending=[True, False])


def promotion_status(row: dict[str, Any]) -> str:
    if row.get("variant") == "small_entry_all":
        return "branch_reference"
    h_rows = safe_float(row.get("h2026_selected_rows"))
    h_rate = safe_float(row.get("h2026_selected_rate"))
    h_pos = safe_float(row.get("h2026_selected_pos20"))
    h_avg = safe_float(row.get("h2026_selected_avg20"))
    h_loss = safe_float(row.get("h2026_selected_loss_gt5"))
    d_pos = safe_float(row.get("h2026_delta_pos"))
    d_avg = safe_float(row.get("h2026_delta_avg"))
    d_loss = safe_float(row.get("h2026_delta_loss"))
    prior_blocks = safe_float(row.get("prior_blocks"))
    prior_selected_rows = safe_float(row.get("prior_selected_rows_mean"))
    prior_pos_hit = safe_float(row.get("prior_delta_pos_hit"))
    prior_avg_hit = safe_float(row.get("prior_delta_avg_hit"))
    if (
        h_rows >= 80
        and h_rate >= 0.40
        and h_pos >= 0.65
        and d_pos >= 0.02
        and d_avg >= 0
        and d_loss <= 0
        and prior_blocks >= MIN_PROMOTION_PRIOR_BLOCKS
        and prior_selected_rows >= MIN_PROMOTION_PRIOR_SELECTED_ROWS_MEAN
        and prior_pos_hit >= 0.67
        and prior_avg_hit >= 0.67
    ):
        return "green_candidate_for_ds_confirmation"
    if (
        h_rows >= 60
        and h_rate >= 0.35
        and h_pos >= 0.62
        and d_pos >= 0.005
        and d_avg >= 0
        and h_loss <= 0.20
        and prior_blocks >= MIN_PROMOTION_PRIOR_BLOCKS
        and prior_selected_rows >= MIN_PROMOTION_PRIOR_SELECTED_ROWS_MEAN
        and prior_pos_hit >= 0.50
    ):
        return "yellow_candidate_needs_fresh_panel"
    if h_rows >= 30 and h_avg > 0 and h_pos >= 0.58:
        return "observe_diagnostic_only"
    return "reject_or_reference_only"


def rank_score(row: dict[str, Any]) -> float:
    return (
        25 * safe_float(row.get("h2026_selected_pos20"))
        + safe_float(row.get("h2026_selected_avg20"))
        + 100 * safe_float(row.get("h2026_delta_pos"))
        + 0.5 * safe_float(row.get("h2026_delta_avg"))
        - 8 * max(0.0, safe_float(row.get("h2026_selected_loss_gt5")))
        + 2 * safe_float(row.get("prior_delta_pos_hit"))
    )


def panel_stability(
    target_scored: pd.DataFrame,
    *,
    frequency: str,
    variant: str,
    threshold: float,
    panel_size: int,
    panel_seeds: int,
) -> list[dict[str, Any]]:
    rows = []
    codes = sorted(target_scored["code"].astype(str).str.zfill(6).unique())
    for seed in range(max(1, panel_seeds)):
        ordered = sorted(codes, key=lambda code: stable_hash_int("p0_small_ml_panel", seed, frequency, variant, code))
        selected_codes = set(ordered[: min(panel_size, len(ordered))])
        panel = target_scored[target_scored["code"].astype(str).str.zfill(6).isin(selected_codes)].copy()
        selected = panel[panel["_confirmer_score"] >= threshold].copy()
        row = evaluate_variant(
            base=panel,
            selected=selected,
            frequency=frequency,
            target_block=FINAL_OOT,
            variant=variant,
            feature_set="panel_eval",
            model_name="panel_eval",
            confirm_quantile=0.0,
            score_threshold=threshold,
            train_blocks=[],
            validation_block="H2025_2",
            selected_feature_count=0,
        )
        row["panel_seed"] = seed
        row["panel_size_codes"] = len(selected_codes)
        rows.append(row)
    return rows


def summarize_panels(panels: pd.DataFrame) -> pd.DataFrame:
    if panels.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in panels.groupby(["frequency", "variant"], sort=True):
        rows.append(
            {
                "frequency": keys[0],
                "variant": keys[1],
                "panels": int(group["panel_seed"].nunique()),
                "selected_rows_mean": round(float(pd.to_numeric(group["selected_rows"], errors="coerce").mean()), 3),
                "selected_rate_mean±std": fmt_mean_std(group, "selected_rate"),
                "selected_pos20_mean±std": fmt_mean_std(group, "selected_pos20"),
                "selected_avg20_mean±std": fmt_mean_std(group, "selected_avg20"),
                "selected_loss_gt5_mean±std": fmt_mean_std(group, "selected_loss_gt5"),
                "delta_pos_mean±std": fmt_mean_std(group, "delta_pos20_vs_base"),
                "delta_avg_mean±std": fmt_mean_std(group, "delta_avg20_vs_base"),
            }
        )
    return pd.DataFrame(rows)


def build_preview_rows(
    frame: pd.DataFrame,
    *,
    frequency: str,
    variant: str,
    threshold: float,
    feature_set: str,
    max_rows: int,
) -> list[dict[str, Any]]:
    ordered = frame.sort_values("_confirmer_score", ascending=False).head(max_rows)
    rows: list[dict[str, Any]] = []
    for _, row in ordered.iterrows():
        confirmed = safe_float(row.get("_confirmer_score")) >= threshold
        rows.append(
            {
                "date": str(row.get("date")),
                "code": str(row.get("code")).zfill(6),
                "name": str(row.get("name", "")),
                "time_block": str(row.get("time_block")),
                "tool_id": "p0_small_entry_ml_confirmer_v1",
                "frequency": frequency,
                "base_branch": "branch_stack_v1.small_buy_hold",
                "variant": variant,
                "feature_set": feature_set,
                "operation_action_cn": "小仓试探/持有" if confirmed else "等待/补证据",
                "position_cap_hint": 0.10 if confirmed else 0.0,
                "ml_confirmer_score": preview_num(row.get("_confirmer_score")),
                "ml_confirmer_threshold": preview_num(threshold),
                "opp_margin": preview_num(row.get("opp_margin")),
                "opp_quantile_in_date": preview_num(row.get("opp_quantile_in_date")),
                "kline_opp_margin": preview_num(row.get("kline_opp_margin")),
                "kline_risk_margin": preview_num(row.get("kline_risk_margin")),
                "news_warning_score": preview_num(row.get("news_warning_score")),
                "news_missing_rate": preview_num(row.get("news_missing_rate")),
                "financial_quality_risk_score": preview_num(row.get("financial_quality_risk_score")),
                "financial_report_missing_rate": preview_num(row.get("financial_report_missing_rate")),
                "corr_peer_relative_return_20d": preview_num(row.get("corr_peer_relative_return_20d")),
                "lower_support": preview_num(row.get("lower_support")),
                "upper_overhang": preview_num(row.get("upper_overhang")),
                "agent_instruction": "treat as small-entry confirmation context only; require semantic news/financial/BookSkill/RAG review before raising exposure",
                "auto_trade": False,
            }
        )
    return rows


def render_report(
    args: argparse.Namespace,
    notes: list[str],
    small_frame: pd.DataFrame,
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    panel_summary: pd.DataFrame,
    hygiene: pd.DataFrame,
    paths: dict[str, Path],
) -> str:
    h2026 = metrics[metrics["target_block"].eq(FINAL_OOT)].copy()
    lines = [
        "# P0 Small-Entry ML Confirmer v1",
        "",
        "本报告审计一个轻量、预注册、本地运行的 ML confirmer。它只作用于 `branch_stack_v1.small_buy_hold` 小仓分叉，目标是确认/降权，不是新的买入引擎。实验不调用 DeepSeek、不读取 key。",
        "",
        "## Setup",
        "",
        f"- frequencies: `{args.frequencies}`",
        f"- kline_feature_group: `{args.kline_feature_group}`",
        "- base branch: `branch_stack_v1.small_buy_hold` from P0 operation policy v1.",
        "- models: logistic L2 C=0.50 and logistic L1 C=0.05 only; no GBDT/HGB.",
        "- confirmation thresholds are pre-registered validation quantiles: 0.50, 0.60, 0.70.",
        "- confirmer split: because small-entry rows are sparse in some half-year blocks, validation uses the latest 25% tail of prior small-entry rows; target block remains untouched.",
        "- H2026_1 is final OOT and is not used for feature/model/threshold selection.",
        "- Agent preview contains no future return, GT, label, or result fields. Missing news/financial values are JSON null.",
        "",
        "## Coverage Notes",
        "",
    ]
    lines.extend([f"- {note}" for note in notes[-10:]])
    if not small_frame.empty:
        lines.append(f"- small_entry_rows_total={len(small_frame)}")
        lines.append(f"- small_entry_blocks={small_frame['target_block'].nunique()}")
        lines.append(f"- small_entry_frequencies={small_frame['frequency'].nunique()}")
    lines.extend(
        [
            "",
            "## Main Summary",
            "",
            markdown_table(summary.head(48)),
            "",
            "## H2026 Detail",
            "",
            markdown_table(
                h2026[
                    [
                        "frequency",
                        "variant",
                        "feature_set",
                        "selected_feature_count",
                        "base_rows",
                        "selected_rows",
                        "selected_rate",
                        "base_pos20",
                        "selected_pos20",
                        "selected_avg20",
                        "selected_loss_gt5",
                        "delta_pos20_vs_base",
                        "delta_avg20_vs_base",
                        "missed_positive_rows",
                        "missed_large_gain_rows",
                    ]
                ].sort_values(["selected_pos20", "selected_avg20"], ascending=[False, False])
                if not h2026.empty
                else pd.DataFrame()
            ),
            "",
            "## H2026 Panel Stability",
            "",
            markdown_table(panel_summary),
            "",
            "## Hygiene",
            "",
            markdown_table(hygiene) if not hygiene.empty else "_empty_",
            "",
            "## Decision Rules",
            "",
            f"- `green_candidate_for_ds_confirmation` requires stable H2026 lift, no higher large-loss rate, enough rows, at least {MIN_PROMOTION_PRIOR_BLOCKS} prior blocks, and prior selected rows mean >= {MIN_PROMOTION_PRIOR_SELECTED_ROWS_MEAN}.",
            f"- `yellow_candidate_needs_fresh_panel` also requires at least {MIN_PROMOTION_PRIOR_BLOCKS} prior blocks and prior selected rows mean >= {MIN_PROMOTION_PRIOR_SELECTED_ROWS_MEAN}; sparse prior samples stay diagnostic only.",
            "- If ML variants fail to beat `small_entry_all`, close this branch as a hard filter and use the scores only as Agent context.",
            "- Missing financial/news values are data gaps, never low-risk evidence.",
            "",
            "## Artifacts",
            "",
        ]
    )
    lines.extend([f"- `{path}`" for path in paths.values()])
    return "\n".join(lines) + "\n"


def prior_tail_train_validation(
    frame: pd.DataFrame,
    target_block: str,
    validation_fraction: float = 0.25,
    min_validation_rows: int = MIN_CONFIRMER_VALID_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Use only prior small-entry rows and reserve the latest tail for validation.

    Half-year validation blocks are too sparse for this narrow branch: some
    blocks have fewer than 10 small-entry rows. A prior-tail split keeps the
    split time-safe while avoiding H2026-driven threshold selection.
    """
    blocks = sorted(frame["target_block"].dropna().unique(), key=block_index)
    if target_block not in blocks:
        return pd.DataFrame(), pd.DataFrame(), "missing_target_block"
    prior_blocks = blocks[: blocks.index(target_block)]
    prior = frame[frame["target_block"].isin(prior_blocks)].copy()
    if prior.empty:
        return pd.DataFrame(), pd.DataFrame(), "no_prior_rows"
    prior = prior.sort_values(["date", "code"]).reset_index(drop=True)
    validation_rows = max(min_validation_rows, int(math.ceil(len(prior) * validation_fraction)))
    validation_rows = min(validation_rows, max(0, len(prior) - 1))
    if validation_rows <= 0:
        return prior.iloc[0:0].copy(), prior.iloc[0:0].copy(), "insufficient_prior_for_tail"
    train = prior.iloc[: len(prior) - validation_rows].copy()
    validation = prior.iloc[len(prior) - validation_rows :].copy()
    train_context = f"{train['target_block'].iloc[0]}..{train['target_block'].iloc[-1]}" if not train.empty else "empty"
    val_context = f"{validation['target_block'].iloc[0]}..{validation['target_block'].iloc[-1]}" if not validation.empty else "empty"
    return train, validation, f"prior_tail_25pct_train={train_context}_validation={val_context}"


def block_index(block: str) -> int:
    try:
        return TARGET_BLOCKS.index(str(block))
    except ValueError:
        return 999


def variant_id(feature_set: str, model_name: str, quantile: float) -> str:
    return f"{feature_set}__{model_name}__top{int(round((1 - quantile) * 100)):02d}"


def feature_importance_rows(
    model: Pipeline,
    feature_set: FeatureSet,
    spec: ModelSpec,
    frequency: str,
    target_block: str,
) -> list[dict[str, Any]]:
    clf = model.named_steps["clf"]
    coef = getattr(clf, "coef_", None)
    if coef is None or len(coef) == 0:
        return []
    raw_coef = coef[0][: len(feature_set.columns)]
    rows = []
    for feature, value in zip(feature_set.columns, raw_coef):
        rows.append(
            {
                "frequency": frequency,
                "target_block": target_block,
                "feature_set": feature_set.feature_set,
                "model_name": spec.model_name,
                "feature": feature,
                "coef": round(float(value), 8),
                "abs_coef": round(abs(float(value)), 8),
                "direction": "positive" if value > 0 else "negative" if value < 0 else "zero",
            }
        )
    return rows


def write_jsonl(path: Path, frame: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in frame.to_dict("records"):
            assert_no_future_fields(record)
            handle.write(json.dumps(json_safe(record), ensure_ascii=False, default=str, allow_nan=False) + "\n")


def assert_no_future_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lower = str(key).lower()
            if key in FUTURE_OR_RESULT_FIELDS or lower.startswith("return_") or "future" in lower or "gt_" in lower:
                raise ValueError(f"future/result field leaked: {key}")
            assert_no_future_fields(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_future_fields(item)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if pd.isna(value):
        return None
    return value


def forbidden_field(field: str) -> bool:
    lower = str(field).lower()
    return (
        field in FUTURE_OR_RESULT_FIELDS
        or lower.startswith("return_")
        or "future" in lower
        or "gt_" in lower
        or "label" in lower
        or "cash_adjusted" in lower
        or "positive_20d" in lower
        or "loss_gt5" in lower
    )


def num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def preview_num(value: Any) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return None
    return round(float(parsed), 6)


def positive_rate(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return round(float((values > 0).mean()), 6) if len(values) else np.nan


def mean_value(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return round(float(values.mean()), 6) if len(values) else np.nan


def rate_le(values: pd.Series, threshold: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return round(float((values <= threshold).mean()), 6) if len(values) else np.nan


def rate_ge(values: pd.Series, threshold: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return round(float((values >= threshold).mean()), 6) if len(values) else np.nan


def delta(left: float, right: float) -> float:
    if pd.isna(left) or pd.isna(right):
        return np.nan
    return round(float(left - right), 6)


def first(frame: pd.DataFrame, column: str) -> Any:
    if frame.empty or column not in frame:
        return ""
    return frame.iloc[0].get(column, "")


def mean_col(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return round(float(values.mean()), 6) if len(values) else np.nan


def hit_rate(frame: pd.DataFrame, column: str, threshold: float) -> float:
    if frame.empty or column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return round(float((values > threshold).mean()), 6) if len(values) else np.nan


def get_val(row: pd.Series, column: str) -> float:
    if row.empty:
        return np.nan
    value = row.get(column)
    try:
        output = float(value)
    except (TypeError, ValueError):
        return np.nan
    return round(output, 6) if not math.isnan(output) else np.nan


def fmt_mean_std(frame: pd.DataFrame, column: str) -> str:
    values = pd.to_numeric(frame.get(column), errors="coerce").dropna()
    if values.empty:
        return "NA"
    return f"{values.mean():.4f}±{values.std(ddof=0):.4f}"


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    return frame.to_markdown(index=False)


if __name__ == "__main__":
    main()
