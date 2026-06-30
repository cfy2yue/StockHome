"""Audit a transfer-trained confirmer for the P0 small-entry branch.

This no-DeepSeek experiment addresses a weakness in
`p0_small_entry_ml_confirmer_v1`: training only on the small-entry branch left
too few prior rows in several blocks. Here, a lightweight logistic confirmer is
trained on a larger historical candidate cohort, then applied only to the
user-facing `branch_stack_v1.small_buy_hold` branch.

Future 20-day returns are used only as offline training/evaluation labels.
Agent preview rows are field-whitelisted and contain no future labels or
returns.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_p0_decision_stack_v1 import (  # noqa: E402
    FINAL_OOT,
    FUTURE_OR_RESULT_FIELDS,
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
from scripts.audit_p0_small_entry_ml_confirmer_v1 import (  # noqa: E402
    FeatureSet,
    NEWS_FIN_FEATURES,
    PEER_CHIP_KLINE_FEATURES,
    STACK_FEATURES,
    delta,
    fit_logistic,
    mean_value,
    model_specs,
    positive_rate,
    predict_score,
    preview_num,
    rate_le,
    validation_threshold,
)
from scripts.audit_single_stock_review_quality import _rolling_split  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_PREFIX = "p0_small_entry_transfer_confirmer_v1"
DEFAULT_FREQUENCIES = "weekly_friday,every_2_weeks,weekly_tuesday"
DEFAULT_KLINE_GROUP = "kline_peer_chip_news_risk"
MAX_HGB_TRAIN_ROWS = 60000
PANEL_SIZE = 100
PANEL_SEEDS = 12
CONFIRM_QUANTILES = (0.50, 0.60, 0.70, 0.80)
MIN_TRANSFER_TRAIN_ROWS = 800
MIN_TRANSFER_VALID_ROWS = 200
MIN_PROMOTION_PRIOR_BLOCKS = 2
MIN_PROMOTION_PRIOR_SELECTED_ROWS_MEAN = 30

BOOK_SKILL_IDS = ("PPS-Q-017", "PPS-M-003", "DOW-B-004", "PPS-Q-019", "PPS-Q-023")


@dataclass(frozen=True)
class CohortSpec:
    cohort: str
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit transfer confirmer for P0 small-entry branch.")
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
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frequencies = [item.strip() for item in args.frequencies.split(",") if item.strip()]
    frame, feature_groups, notes = load_stack_frame()

    scored_blocks: list[pd.DataFrame] = []
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
            enriched = enrich_candidate_frame(scored, target, frequency, target_block)
            scored_blocks.append(enriched)

    candidates = pd.concat(scored_blocks, ignore_index=True) if scored_blocks else pd.DataFrame()
    metric_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []

    if not candidates.empty:
        for frequency in frequencies:
            freq_candidates = candidates[candidates["frequency"].eq(frequency)].copy()
            metric_rows.extend(evaluate_branch_reference(freq_candidates))
            for target_block in sorted(freq_candidates["target_block"].unique(), key=block_index):
                prior = freq_candidates[freq_candidates["target_block"].map(block_index) < block_index(target_block)].copy()
                target_all = freq_candidates[freq_candidates["target_block"].eq(target_block)].copy()
                target_small = target_all[target_all["operation_action"].astype(str).eq("small_buy_hold")].copy()
                if target_small.empty:
                    continue
                train, validation, split_context = prior_tail_train_validation(prior)
                if len(train) < MIN_TRANSFER_TRAIN_ROWS or len(validation) < MIN_TRANSFER_VALID_ROWS:
                    hygiene_rows.append(
                        {
                            "frequency": frequency,
                            "target_block": target_block,
                            "stage": "transfer_confirmer",
                            "status": "skip_insufficient_transfer_rows",
                            "split_context": split_context,
                            "train_rows": len(train),
                            "validation_rows": len(validation),
                            "target_rows": len(target_all),
                            "target_small_rows": len(target_small),
                        }
                    )
                    continue
                for cohort in cohort_specs():
                    train_cohort = apply_train_cohort(train, cohort.cohort)
                    validation_cohort = apply_train_cohort(validation, cohort.cohort)
                    if len(train_cohort) < MIN_TRANSFER_TRAIN_ROWS or len(validation_cohort) < MIN_TRANSFER_VALID_ROWS:
                        hygiene_rows.append(
                            {
                                "frequency": frequency,
                                "target_block": target_block,
                                "stage": "transfer_confirmer",
                                "status": "skip_insufficient_cohort_rows",
                                "cohort": cohort.cohort,
                                "train_rows": len(train_cohort),
                                "validation_rows": len(validation_cohort),
                                "target_small_rows": len(target_small),
                            }
                        )
                        continue
                    for feature_set in feature_sets_for(candidates):
                        if len(feature_set.columns) < 5:
                            continue
                        for model_spec in model_specs():
                            fitted = fit_logistic(train_cohort, feature_set.columns, model_spec)
                            if fitted is None:
                                continue
                            validation_scores = predict_score(fitted, validation_cohort, feature_set.columns)
                            target_scores = predict_score(fitted, target_small, feature_set.columns)
                            validation_scored = validation_cohort.assign(_transfer_score=validation_scores)
                            target_scored = target_small.assign(_transfer_score=target_scores)
                            feature_rows.extend(
                                feature_importance_rows(
                                    fitted,
                                    feature_set,
                                    frequency=frequency,
                                    target_block=target_block,
                                    cohort=cohort.cohort,
                                    model_name=model_spec.model_name,
                                )
                            )
                            for quantile in CONFIRM_QUANTILES:
                                threshold = validation_threshold(validation_scored["_transfer_score"], quantile)
                                selected = target_scored[target_scored["_transfer_score"] >= threshold].copy()
                                variant = variant_id(cohort.cohort, feature_set.feature_set, model_spec.model_name, quantile)
                                metric_rows.append(
                                    evaluate_variant(
                                        base=target_small,
                                        selected=selected,
                                        frequency=frequency,
                                        target_block=target_block,
                                        variant=variant,
                                        cohort=cohort.cohort,
                                        feature_set=feature_set.feature_set,
                                        model_name=model_spec.model_name,
                                        confirm_quantile=quantile,
                                        score_threshold=threshold,
                                        train_blocks=sorted(train_cohort["target_block"].astype(str).unique(), key=block_index),
                                        validation_context=split_context,
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
                                                cohort=cohort.cohort,
                                                max_rows=max(10, args.preview_max_rows // 8),
                                            )
                                        )

    metrics = pd.DataFrame(metric_rows)
    summary = summarize(metrics)
    panels = pd.DataFrame(panel_rows)
    panel_summary = summarize_panels(panels)
    features = pd.DataFrame(feature_rows)
    preview = pd.DataFrame(preview_rows)
    hygiene = pd.DataFrame(hygiene_rows)

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
    paths["report"].write_text(
        render_report(args, notes, candidates, summary, metrics, panel_summary, hygiene, paths),
        encoding="utf-8",
    )
    print("A股研究Agent")
    print(f"candidate_rows={len(candidates)} metrics={len(metrics)} summary={len(summary)} preview={len(preview)}")
    print(f"report={paths['report']}")


def enrich_candidate_frame(scored: pd.DataFrame, target: pd.DataFrame, frequency: str, target_block: str) -> pd.DataFrame:
    branch = with_operation_actions(apply_policy(scored, "branch_stack_v1"))
    branch["frequency"] = frequency
    branch["target_block"] = target_block
    branch["code"] = branch["code"].astype(str).str.zfill(6)
    branch = add_stack_derived_features(branch)

    feature_cols = safe_raw_feature_columns(target)
    raw = target[["date", "code", *feature_cols]].copy()
    raw["code"] = raw["code"].astype(str).str.zfill(6)
    out = branch.merge(raw.drop_duplicates(["date", "code"]), on=["date", "code"], how="left")
    out = add_derived_channel_features(out)
    out["positive_20d"] = pd.to_numeric(out["return_20d"], errors="coerce").gt(0).astype(int)
    return out


def add_stack_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["opp_margin"] = num(out, "opp_score") - num(out, "opp_threshold")
    out["kline_opp_margin"] = num(out, "kline_opp_score") - num(out, "kline_opp_threshold")
    out["kline_risk_margin"] = num(out, "kline_risk_threshold") - num(out, "kline_risk_score")
    out["target_position"] = pd.to_numeric(out.get("target_position"), errors="coerce").fillna(0.0)
    for col in ["risk_review_queue", "risk_queue_high_hard_counter", "kline_hard_risk", "opp_active", "kline_active"]:
        if col in out:
            out[col] = out[col].astype(int)
    return out


def safe_raw_feature_columns(frame: pd.DataFrame) -> list[str]:
    wanted = [
        *NEWS_FIN_FEATURES,
        *PEER_CHIP_KLINE_FEATURES,
        "book_score",
        "triggered_skills",
        "news_net_materiality_30d",
        "news_positive_materiality_30d",
        "news_negative_materiality_30d",
        "news_evidence_quality_score_30d",
        "news_conflict_intensity_30d",
        "news_recency_weighted_materiality_30d",
        "peer_group_news_risk_avg",
        "peer_group_news_opportunity_avg",
        "tushare_industry_news_attention_gap",
        "tushare_area_news_attention_gap",
        "financial_report_event_count",
        "announcement_materiality_score",
        "financial_report_materiality_score",
    ]
    return [col for col in dict.fromkeys(wanted) if col in frame.columns and not forbidden_field(col)]


def add_derived_channel_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["news_opportunity_minus_warning"] = num(out, "news_opportunity_score") - num(out, "news_warning_score")
    out["news_peer_opportunity_gap"] = num(out, "news_opportunity_score") - num(out, "peer_group_news_opportunity_avg")
    out["news_peer_risk_gap"] = num(out, "news_warning_score") - num(out, "peer_group_news_risk_avg")
    out["news_quality_adjusted_opportunity"] = num(out, "news_opportunity_score") * (
        1.0 - num(out, "news_missing_rate").fillna(1.0)
    ) * num(out, "news_evidence_quality").fillna(0.0)
    out["financial_quality_minus_risk"] = num(out, "financial_disclosure_quality_score") - num(
        out, "financial_quality_risk_score"
    )
    out["financial_surprise_minus_risk"] = num(out, "financial_surprise_score") - num(out, "financial_quality_risk_score")
    out["peer_relative_strength_blend"] = num(out, "peer_relative_to_group_20d") + num(
        out, "corr_peer_relative_return_20d"
    )
    out["support_overhang_gap"] = num(out, "lower_support") - num(out, "upper_overhang")
    triggered = out.get("triggered_skills", pd.Series("", index=out.index)).fillna("").astype(str)
    out["bookskill_any_triggered"] = triggered.str.len().gt(0).astype(int)
    for skill_id in BOOK_SKILL_IDS:
        out[f"skill_{skill_id.replace('-', '_').replace('.', '_')}_triggered"] = triggered.str.contains(
            skill_id, regex=False
        ).astype(int)
    return out


def feature_sets_for(frame: pd.DataFrame) -> list[FeatureSet]:
    available = set(frame.columns)
    stack = tuple(col for col in STACK_FEATURES if col in available and not forbidden_field(col))
    news_fin = tuple(col for col in NEWS_FIN_FEATURES if col in available and not forbidden_field(col))
    peer_chip = tuple(col for col in PEER_CHIP_KLINE_FEATURES if col in available and not forbidden_field(col))
    derived = tuple(
        col
        for col in [
            "book_score",
            "bookskill_any_triggered",
            "skill_PPS_Q_017_triggered",
            "skill_PPS_M_003_triggered",
            "skill_DOW_B_004_triggered",
            "skill_PPS_Q_019_triggered",
            "skill_PPS_Q_023_triggered",
            "news_net_materiality_30d",
            "news_positive_materiality_30d",
            "news_negative_materiality_30d",
            "news_conflict_intensity_30d",
            "news_opportunity_minus_warning",
            "news_peer_opportunity_gap",
            "news_peer_risk_gap",
            "news_quality_adjusted_opportunity",
            "financial_quality_minus_risk",
            "financial_surprise_minus_risk",
            "financial_report_event_count",
            "peer_relative_strength_blend",
            "support_overhang_gap",
        ]
        if col in available and not forbidden_field(col)
    )
    return [
        FeatureSet("stack_margins_only", stack),
        FeatureSet("stack_plus_news_fin", tuple(dict.fromkeys([*stack, *news_fin, *derived]))),
        FeatureSet("stack_plus_peer_chip_kline", tuple(dict.fromkeys([*stack, *peer_chip]))),
        FeatureSet("stack_plus_all_channels", tuple(dict.fromkeys([*stack, *news_fin, *peer_chip, *derived]))),
    ]


def cohort_specs() -> list[CohortSpec]:
    return [
        CohortSpec("all_scored_rows", "all scored decision rows from prior blocks"),
        CohortSpec("opportunity_context_rows", "rows with nonzero target position or strong opportunity rank"),
    ]


def apply_train_cohort(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    if cohort == "all_scored_rows":
        return frame.copy()
    if cohort == "opportunity_context_rows":
        pos = pd.to_numeric(frame["target_position"], errors="coerce").fillna(0.0)
        opp_q = pd.to_numeric(frame.get("opp_quantile_in_date"), errors="coerce").fillna(0.0)
        return frame[(pos >= 0.10) | (opp_q >= 0.70)].copy()
    raise ValueError(f"unknown cohort: {cohort}")


def prior_tail_train_validation(
    prior: pd.DataFrame,
    validation_fraction: float = 0.20,
    min_validation_rows: int = MIN_TRANSFER_VALID_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if prior.empty:
        return pd.DataFrame(), pd.DataFrame(), "no_prior_rows"
    ordered = prior.sort_values(["date", "code"]).reset_index(drop=True)
    validation_rows = max(min_validation_rows, int(math.ceil(len(ordered) * validation_fraction)))
    validation_rows = min(validation_rows, max(0, len(ordered) - 1))
    if validation_rows <= 0:
        return ordered.iloc[0:0].copy(), ordered.iloc[0:0].copy(), "insufficient_prior_tail"
    train = ordered.iloc[: len(ordered) - validation_rows].copy()
    validation = ordered.iloc[len(ordered) - validation_rows :].copy()
    train_ctx = f"{train['target_block'].iloc[0]}..{train['target_block'].iloc[-1]}" if not train.empty else "empty"
    val_ctx = (
        f"{validation['target_block'].iloc[0]}..{validation['target_block'].iloc[-1]}"
        if not validation.empty
        else "empty"
    )
    return train, validation, f"prior_tail_20pct_train={train_ctx}_validation={val_ctx}"


def evaluate_branch_reference(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    small = frame[frame["operation_action"].astype(str).eq("small_buy_hold")].copy()
    for (frequency, target_block), group in small.groupby(["frequency", "target_block"], sort=True):
        rows.append(
            evaluate_variant(
                base=group,
                selected=group,
                frequency=str(frequency),
                target_block=str(target_block),
                variant="small_entry_all",
                cohort="branch_reference",
                feature_set="branch_reference",
                model_name="none",
                confirm_quantile=0.0,
                score_threshold=np.nan,
                train_blocks=[],
                validation_context="none",
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
    cohort: str,
    feature_set: str,
    model_name: str,
    confirm_quantile: float,
    score_threshold: float,
    train_blocks: list[str],
    validation_context: str,
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
        "cohort": cohort,
        "feature_set": feature_set,
        "model_name": model_name,
        "confirm_quantile": round(float(confirm_quantile), 4),
        "score_threshold": round(float(score_threshold), 8) if not pd.isna(score_threshold) else np.nan,
        "train_blocks": ";".join(train_blocks),
        "validation_context": validation_context,
        "selected_feature_count": int(selected_feature_count),
        "base_rows": int(len(base)),
        "selected_rows": int(len(selected)),
        "selected_rate": round(float(len(selected) / max(1, len(base))), 6),
        "base_pos20": positive_rate(base_ret),
        "base_avg20": mean_value(base_ret),
        "base_loss_gt5": rate_le(base_ret, -5),
        "selected_pos20": positive_rate(ret),
        "selected_avg20": mean_value(ret),
        "selected_loss_gt5": rate_le(ret, -5),
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
    rows = []
    for (frequency, variant), group in metrics.groupby(["frequency", "variant"], sort=True):
        h = group[group["target_block"].eq(FINAL_OOT)]
        prior = group[~group["target_block"].eq(FINAL_OOT)]
        hrow = h.iloc[0] if not h.empty else pd.Series(dtype=object)
        row = {
            "frequency": frequency,
            "variant": variant,
            "cohort": first(group, "cohort"),
            "feature_set": first(group, "feature_set"),
            "model_name": first(group, "model_name"),
            "confirm_quantile": get_val(group.iloc[0], "confirm_quantile"),
            "prior_blocks": int(prior["target_block"].nunique()),
            "prior_selected_rows_mean": mean_col(prior, "selected_rows"),
            "prior_selected_rate_mean": mean_col(prior, "selected_rate"),
            "prior_delta_pos_mean": mean_col(prior, "delta_pos20_vs_base"),
            "prior_delta_avg_mean": mean_col(prior, "delta_avg20_vs_base"),
            "prior_delta_loss_mean": mean_col(prior, "delta_loss_gt5_vs_base"),
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
    prior_rows = safe_float(row.get("prior_selected_rows_mean"))
    prior_pos_hit = safe_float(row.get("prior_delta_pos_hit"))
    prior_avg_hit = safe_float(row.get("prior_delta_avg_hit"))
    if (
        h_rows >= 80
        and h_rate >= 0.35
        and h_pos >= 0.65
        and d_pos >= 0.02
        and d_avg >= 0
        and d_loss <= 0
        and h_loss <= 0.18
        and prior_blocks >= MIN_PROMOTION_PRIOR_BLOCKS
        and prior_rows >= MIN_PROMOTION_PRIOR_SELECTED_ROWS_MEAN
        and prior_pos_hit >= 0.67
        and prior_avg_hit >= 0.67
    ):
        return "green_candidate_for_ds_confirmation"
    if (
        h_rows >= 60
        and h_rate >= 0.25
        and h_pos >= 0.62
        and d_pos >= 0.005
        and d_avg >= 0
        and h_loss <= 0.20
        and prior_blocks >= MIN_PROMOTION_PRIOR_BLOCKS
        and prior_rows >= MIN_PROMOTION_PRIOR_SELECTED_ROWS_MEAN
        and prior_pos_hit >= 0.50
        and prior_avg_hit >= 0.50
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
        + 2 * safe_float(row.get("prior_delta_avg_hit"))
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
        ordered = sorted(codes, key=lambda code: stable_hash_int("p0_transfer_panel", seed, frequency, variant, code))
        selected_codes = set(ordered[: min(panel_size, len(ordered))])
        panel = target_scored[target_scored["code"].astype(str).str.zfill(6).isin(selected_codes)].copy()
        selected = panel[panel["_transfer_score"] >= threshold].copy()
        row = evaluate_variant(
            base=panel,
            selected=selected,
            frequency=frequency,
            target_block=FINAL_OOT,
            variant=variant,
            cohort="panel_eval",
            feature_set="panel_eval",
            model_name="panel_eval",
            confirm_quantile=0.0,
            score_threshold=threshold,
            train_blocks=[],
            validation_context="H2025_2",
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
    for (frequency, variant), group in panels.groupby(["frequency", "variant"], sort=True):
        rows.append(
            {
                "frequency": frequency,
                "variant": variant,
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
    cohort: str,
    max_rows: int,
) -> list[dict[str, Any]]:
    ordered = frame.sort_values("_transfer_score", ascending=False).head(max_rows)
    rows = []
    for _, row in ordered.iterrows():
        confirmed = safe_float(row.get("_transfer_score")) >= threshold
        rows.append(
            {
                "date": str(row.get("date")),
                "code": str(row.get("code")).zfill(6),
                "name": str(row.get("name", "")),
                "time_block": str(row.get("time_block")),
                "tool_id": "p0_small_entry_transfer_confirmer_v1",
                "frequency": frequency,
                "base_branch": "branch_stack_v1.small_buy_hold",
                "variant": variant,
                "training_cohort": cohort,
                "feature_set": feature_set,
                "operation_action_cn": "小仓试探/持有" if confirmed else "等待/补证据",
                "position_cap_hint": 0.10 if confirmed else 0.0,
                "transfer_score": preview_num(row.get("_transfer_score")),
                "transfer_threshold": preview_num(threshold),
                "opp_margin": preview_num(row.get("opp_margin")),
                "kline_opp_margin": preview_num(row.get("kline_opp_margin")),
                "kline_risk_margin": preview_num(row.get("kline_risk_margin")),
                "news_opportunity_minus_warning": preview_num(row.get("news_opportunity_minus_warning")),
                "news_peer_opportunity_gap": preview_num(row.get("news_peer_opportunity_gap")),
                "financial_quality_minus_risk": preview_num(row.get("financial_quality_minus_risk")),
                "financial_surprise_minus_risk": preview_num(row.get("financial_surprise_minus_risk")),
                "peer_relative_strength_blend": preview_num(row.get("peer_relative_strength_blend")),
                "support_overhang_gap": preview_num(row.get("support_overhang_gap")),
                "book_score": preview_num(row.get("book_score")),
                "pps_q017_triggered": bool(row.get("skill_PPS_Q_017_triggered", 0)),
                "agent_instruction": "use only as transfer-trained small-entry confirmation context; require semantic news/financial/BookSkill review before raising exposure",
                "auto_trade": False,
            }
        )
    return rows


def feature_importance_rows(
    model: Any,
    feature_set: FeatureSet,
    *,
    frequency: str,
    target_block: str,
    cohort: str,
    model_name: str,
) -> list[dict[str, Any]]:
    clf = model.named_steps["clf"]
    coef = getattr(clf, "coef_", None)
    if coef is None or len(coef) == 0:
        return []
    raw_coef = coef[0][: len(feature_set.columns)]
    return [
        {
            "frequency": frequency,
            "target_block": target_block,
            "cohort": cohort,
            "feature_set": feature_set.feature_set,
            "model_name": model_name,
            "feature": feature,
            "coef": round(float(value), 8),
            "abs_coef": round(abs(float(value)), 8),
            "direction": "positive" if value > 0 else "negative" if value < 0 else "zero",
        }
        for feature, value in zip(feature_set.columns, raw_coef)
    ]


def render_report(
    args: argparse.Namespace,
    notes: list[str],
    candidates: pd.DataFrame,
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    panel_summary: pd.DataFrame,
    hygiene: pd.DataFrame,
    paths: dict[str, Path],
) -> str:
    h2026 = metrics[metrics["target_block"].eq(FINAL_OOT)].copy() if not metrics.empty else pd.DataFrame()
    green_rows = summary[summary["promotion_status"].astype(str).str.contains("green", regex=True)].copy()
    yellow_rows = summary[summary["promotion_status"].astype(str).str.contains("yellow", regex=True)].copy()
    green_yellow = pd.concat([green_rows, yellow_rows], ignore_index=True)
    lines = [
        "# P0 Small-Entry Transfer Confirmer v1",
        "",
        "本实验训练一个 transfer confirmer：训练样本来自更大的历史候选池，应用范围只限 `branch_stack_v1.small_buy_hold` 小仓分叉。它是本地 tool 训练，不调用 DeepSeek，不读取 key。",
        "",
        "## Setup",
        "",
        f"- frequencies: `{args.frequencies}`",
        f"- kline_feature_group: `{args.kline_feature_group}`",
        "- training cohorts: `all_scored_rows`, `opportunity_context_rows`.",
        "- models: logistic L2 C=0.50 and logistic L1 C=0.05.",
        "- thresholds: validation score quantiles 0.50/0.60/0.70/0.80; H2026 is final OOT.",
        "- labels/future returns are used only for offline training/evaluation and never enter preview rows.",
        "",
        "## Coverage",
        "",
    ]
    lines.extend([f"- {note}" for note in notes[-8:]])
    if not candidates.empty:
        small = candidates[candidates["operation_action"].astype(str).eq("small_buy_hold")]
        lines.extend(
            [
                f"- candidate_rows_total={len(candidates)}",
                f"- small_entry_rows_total={len(small)}",
                f"- frequencies={candidates['frequency'].nunique()}",
                f"- blocks={candidates['target_block'].nunique()}",
            ]
        )
    lines.extend(
        [
            "",
            "## Main Verdict",
            "",
        ]
    )
    if green_yellow.empty:
        lines.append("- 没有配置达到 green/yellow。若 H2026 局部很亮但 prior hit/coverage 不足，仍只作为 Agent 灰色上下文。")
    else:
        lines.append(
            f"- 出现 {len(green_rows)} 个 green、{len(yellow_rows)} 个 yellow 候选；"
            "当前仅能作为 fresh panel 与 DS Flash/Pro on/off 语义确认的候选，不能直接升为默认。"
        )
    lines.extend(
        [
            "- 这轮直接回答一个训练问题：扩大训练样本是否能解决小仓分叉 prior 支撑不足。结果见下表，升权必须看 prior 与 H2026 同时通过。",
            "",
            "## Summary",
            "",
            markdown_table(summary.head(60)),
            "",
            "## H2026 Detail",
            "",
            markdown_table(
                h2026[
                    [
                        "frequency",
                        "variant",
                        "cohort",
                        "feature_set",
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
            "- green/yellow 必须同时满足：H2026 正收益率/均值/大亏率、prior hit、prior selected rows、selected rate；不能只看单块高收益。",
            "- 如果 news/financial/bookskill 特征只在 H2026 亮、prior 不稳，进入 Agent evidence 时只能写成假设/复核问题，不得写成硬阈值。",
            "- 若该 transfer confirmer 失败，下一步应转向真正的 DS 新闻/公告语义问卷或更完整财报披露源，而不是继续扩大 logistic 网格。",
            "",
            "## Artifacts",
            "",
        ]
    )
    lines.extend([f"- `{path}`" for path in paths.values()])
    return "\n".join(lines) + "\n"


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
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
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


def block_index(block: str) -> int:
    try:
        return TARGET_BLOCKS.index(str(block))
    except ValueError:
        return 999


def variant_id(cohort: str, feature_set: str, model_name: str, quantile: float) -> str:
    return f"{cohort}__{feature_set}__{model_name}__top{int(round((1 - quantile) * 100)):02d}"


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
