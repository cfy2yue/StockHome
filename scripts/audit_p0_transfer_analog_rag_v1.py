"""Audit analog-case retrieval on P0 transfer/channel small-entry candidates.

This no-DeepSeek experiment turns prior realized small-entry cases into a
lightweight case-memory/RAG signal. For each target block, only earlier blocks
are allowed in the analog bank. Future 20-day returns are used for offline
evaluation and for past-case labels only; agent preview rows are whitelisted and
contain no future labels or returns.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_p0_decision_stack_v1 import (  # noqa: E402
    FINAL_OOT,
    MIN_TARGET_ROWS,
    MIN_TRAIN_ROWS,
    MIN_VALID_ROWS,
    TARGET_BLOCKS,
    apply_frequency,
    build_scored_target,
    load_stack_frame,
    safe_float,
    safe_prefix,
    stable_hash_int,
)
from scripts.audit_p0_small_entry_ml_confirmer_v1 import (  # noqa: E402
    fit_logistic,
    model_specs,
    predict_score,
    validation_threshold,
)
from scripts.audit_p0_small_entry_transfer_confirmer_v1 import (  # noqa: E402
    apply_train_cohort,
    block_index,
    enrich_candidate_frame,
    feature_sets_for,
    forbidden_field,
    prior_tail_train_validation,
)
from scripts.audit_p0_transfer_channel_confirm_v1 import (  # noqa: E402
    DEFAULT_SUMMARY,
    JOINED_CACHE,
    TransferConfig,
    add_channel_flags,
    attach_extra_channel_columns,
    load_channel_extra,
    load_transfer_configs,
    write_jsonl,
)
from scripts.audit_single_stock_review_quality import _rolling_split  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_PREFIX = "p0_transfer_analog_rag_v1"
PANEL_SIZE = 100
PANEL_SEEDS = 12
MIN_ANALOG_BANK_ROWS = 30
ANALOG_TOP_KS = (15, 30)

ANALOG_FEATURE_CANDIDATES = (
    "_transfer_score",
    "target_position",
    "opp_score",
    "risk_score",
    "opp_quantile_in_date",
    "channel_support_count",
    "channel_hard_counter_count",
    "channel_soft_gap_count",
    "news_warning_score",
    "news_opportunity_score",
    "news_missing_rate",
    "official_confirmation_score",
    "announcement_materiality_score",
    "financial_report_missing_rate",
    "financial_quality_risk_score",
    "financial_surprise_score",
    "peer_group_positive_breadth_20d",
    "peer_relative_to_group_20d",
    "tushare_industry_positive_breadth_20d",
    "tushare_industry_relative_return_20d",
    "tushare_area_positive_breadth_20d",
    "prior_return_20d",
    "rsi14",
    "drawdown60",
    "close_above_ma200",
    "lower_support",
    "upper_overhang",
    "winner_rate_pct",
)


@dataclass(frozen=True)
class AnalogSpec:
    analog_id: str
    top_k: int
    min_neighbors: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit analog RAG confirmations for transfer small-entry candidates.")
    parser.add_argument("--transfer-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--joined-cache", type=Path, default=JOINED_CACHE)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--kline-feature-group", default="kline_peer_chip_news_risk")
    parser.add_argument("--max-hgb-train-rows", type=int, default=60000)
    parser.add_argument("--panel-size", type=int, default=PANEL_SIZE)
    parser.add_argument("--panel-seeds", type=int, default=PANEL_SEEDS)
    parser.add_argument("--preview-max-rows", type=int, default=500)
    parser.add_argument(
        "--transfer-status-regex",
        default="yellow",
        help="Regex over p0_small_entry_transfer_confirmer_v1 promotion_status; default preserves yellow-only behavior.",
    )
    parser.add_argument(
        "--max-transfer-configs",
        type=int,
        default=0,
        help="Optional cap after rank sorting; 0 means no cap.",
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("ignore", message="Skipping features without any observed values.*", module="sklearn")
    warnings.filterwarnings("ignore", message="Inconsistent values: penalty=.*", module="sklearn")

    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    configs = load_transfer_configs(
        args.transfer_summary,
        status_regex=args.transfer_status_regex,
        max_configs=args.max_transfer_configs if args.max_transfer_configs > 0 else None,
    )
    if not configs:
        raise RuntimeError("no yellow transfer configs found; run p0_small_entry_transfer_confirmer_v1 first")
    frequencies = sorted({cfg.frequency for cfg in configs})

    frame, feature_groups, notes = load_stack_frame()
    channel_extra = load_channel_extra(args.joined_cache)

    candidate_blocks: list[pd.DataFrame] = []
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
            enriched = attach_extra_channel_columns(enriched, channel_extra)
            enriched = add_channel_flags(enriched)
            candidate_blocks.append(enriched)

    candidates = pd.concat(candidate_blocks, ignore_index=True) if candidate_blocks else pd.DataFrame()
    if candidates.empty:
        raise RuntimeError("no candidates produced")

    metric_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    for cfg in configs:
        cfg_candidates = candidates[candidates["frequency"].eq(cfg.frequency)].copy()
        feature_set = next((item for item in feature_sets_for(cfg_candidates) if item.feature_set == cfg.feature_set), None)
        model_spec = next((item for item in model_specs() if item.model_name == cfg.model_name), None)
        if feature_set is None or model_spec is None:
            hygiene_rows.append(
                {
                    "frequency": cfg.frequency,
                    "target_block": "ALL",
                    "stage": "transfer_config",
                    "status": "skip_missing_feature_or_model",
                    "variant": cfg.variant,
                }
            )
            continue
        for target_block in sorted(cfg_candidates["target_block"].dropna().unique(), key=block_index):
            prior = cfg_candidates[cfg_candidates["target_block"].map(block_index) < block_index(target_block)].copy()
            target_all = cfg_candidates[cfg_candidates["target_block"].eq(target_block)].copy()
            target_small = target_all[target_all["operation_action"].astype(str).eq("small_buy_hold")].copy()
            if target_small.empty:
                continue
            train, validation, split_context = prior_tail_train_validation(prior)
            train_cohort = apply_train_cohort(train, cfg.cohort)
            validation_cohort = apply_train_cohort(validation, cfg.cohort)
            if len(train_cohort) < 800 or len(validation_cohort) < 200:
                hygiene_rows.append(
                    {
                        "frequency": cfg.frequency,
                        "target_block": target_block,
                        "stage": "transfer_score",
                        "status": "skip_insufficient_transfer_rows",
                        "variant": cfg.variant,
                        "train_rows": len(train_cohort),
                        "validation_rows": len(validation_cohort),
                        "target_small_rows": len(target_small),
                    }
                )
                continue
            fitted = fit_logistic(train_cohort, feature_set.columns, model_spec)
            if fitted is None:
                continue

            validation_scored = validation_cohort.assign(
                _transfer_score=predict_score(fitted, validation_cohort, feature_set.columns)
            )
            target_scored = target_small.assign(_transfer_score=predict_score(fitted, target_small, feature_set.columns))
            threshold = validation_threshold(validation_scored["_transfer_score"], cfg.confirm_quantile)
            target_scored["_transfer_threshold"] = threshold
            transfer_selected = target_scored[target_scored["_transfer_score"] >= threshold].copy()
            transfer_selected["_transfer_threshold"] = threshold

            prior_small = prior[prior["operation_action"].astype(str).eq("small_buy_hold")].copy()
            if prior_small.empty:
                hygiene_rows.append(
                    {
                        "frequency": cfg.frequency,
                        "target_block": target_block,
                        "stage": "analog_bank",
                        "status": "skip_no_prior_small_branch",
                        "variant": cfg.variant,
                    }
                )
                continue
            prior_small["_transfer_score"] = predict_score(fitted, prior_small, feature_set.columns)
            prior_small["_transfer_threshold"] = threshold
            analog_bank = prior_small[prior_small["_transfer_score"] >= threshold].copy()
            analog_bank["_transfer_threshold"] = threshold
            if len(analog_bank) < MIN_ANALOG_BANK_ROWS:
                hygiene_rows.append(
                    {
                        "frequency": cfg.frequency,
                        "target_block": target_block,
                        "stage": "analog_bank",
                        "status": "skip_insufficient_analog_bank_rows",
                        "variant": cfg.variant,
                        "analog_bank_rows": len(analog_bank),
                    }
                )
                continue

            for spec in analog_specs():
                transfer_with_analog = add_analog_features(transfer_selected, analog_bank, spec=spec)
                for gate_id, selected in apply_analog_gates(transfer_with_analog).items():
                    metric_rows.append(
                        evaluate_gate(
                            branch_base=target_small,
                            transfer_base=transfer_with_analog,
                            selected=selected,
                            cfg=cfg,
                            target_block=target_block,
                            spec=spec,
                            gate_id=gate_id,
                            threshold=threshold,
                            validation_context=split_context,
                        )
                    )
                    if target_block == FINAL_OOT:
                        panel_rows.extend(
                            panel_stability(
                                target_scored,
                                transfer_with_analog=transfer_with_analog,
                                analog_bank=analog_bank,
                                spec=spec,
                                gate_id=gate_id,
                                cfg=cfg,
                                threshold=threshold,
                                panel_size=args.panel_size,
                                panel_seeds=args.panel_seeds,
                            )
                        )
                if target_block == FINAL_OOT:
                    preview_rows.extend(
                        build_preview_rows(transfer_with_analog, cfg=cfg, spec=spec, max_rows=args.preview_max_rows)
                    )

    metrics = pd.DataFrame(metric_rows)
    summary = summarize(metrics)
    panel_detail = pd.DataFrame(panel_rows)
    panel_summary = summarize_panels(panel_detail)
    preview = pd.DataFrame(preview_rows).drop_duplicates(["date", "code", "variant", "analog_id", "gate_id"])
    hygiene = pd.DataFrame(hygiene_rows)

    prefix = safe_prefix(args.output_prefix)
    paths = {
        "metrics": REPORT_DIR / f"{prefix}_metrics.csv",
        "summary": REPORT_DIR / f"{prefix}_summary.csv",
        "panel_detail": REPORT_DIR / f"{prefix}_h2026_panel_detail.csv",
        "panel_summary": REPORT_DIR / f"{prefix}_h2026_panel_summary.csv",
        "preview": REPORT_DIR / f"{prefix}_agent_preview_no_gt.jsonl",
        "hygiene": REPORT_DIR / f"{prefix}_hygiene.csv",
        "report": REPORT_DIR / f"{prefix}.md",
    }
    metrics.to_csv(paths["metrics"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    panel_detail.to_csv(paths["panel_detail"], index=False, encoding="utf-8-sig")
    panel_summary.to_csv(paths["panel_summary"], index=False, encoding="utf-8-sig")
    hygiene.to_csv(paths["hygiene"], index=False, encoding="utf-8-sig")
    write_jsonl(paths["preview"], preview)
    paths["report"].write_text(render_report(notes, configs, summary, metrics, panel_summary, hygiene, paths), encoding="utf-8")

    print("A股研究Agent")
    print(f"configs={len(configs)} candidates={len(candidates)} metrics={len(metrics)} summary={len(summary)}")
    print(f"report={paths['report']}")


def analog_specs() -> list[AnalogSpec]:
    return [
        AnalogSpec("analog_k15_min10", top_k=15, min_neighbors=10),
        AnalogSpec("analog_k30_min20", top_k=30, min_neighbors=20),
    ]


def analog_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in ANALOG_FEATURE_CANDIDATES if col in frame.columns and not forbidden_field(col)]


def add_analog_features(target: pd.DataFrame, bank: pd.DataFrame, *, spec: AnalogSpec) -> pd.DataFrame:
    out = target.copy()
    if out.empty:
        return with_empty_analog_columns(out, spec)
    features = analog_feature_columns(pd.concat([out.head(1), bank.head(1)], ignore_index=True))
    if not features or len(bank) < spec.min_neighbors:
        return with_empty_analog_columns(out, spec)

    bank_features = numeric_matrix(bank, features)
    target_features = numeric_matrix(out, features)
    valid_feature_mask = np.isfinite(bank_features).any(axis=0)
    if not valid_feature_mask.any():
        return with_empty_analog_columns(out, spec)
    bank_features = bank_features[:, valid_feature_mask]
    target_features = target_features[:, valid_feature_mask]
    features = [feature for feature, keep in zip(features, valid_feature_mask) if keep]
    med = np.nanmedian(bank_features, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    bank_features = np.where(np.isfinite(bank_features), bank_features, med)
    target_features = np.where(np.isfinite(target_features), target_features, med)
    scale = np.nanstd(bank_features, axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    bank_z = (bank_features - med) / scale
    target_z = (target_features - med) / scale

    bank_returns = pd.to_numeric(bank.get("return_20d"), errors="coerce").to_numpy(dtype=float)
    bank_codes = bank.get("code", pd.Series("", index=bank.index)).astype(str).str.zfill(6).to_numpy()
    bank_dates = bank.get("date", pd.Series("", index=bank.index)).astype(str).to_numpy()
    top_k = min(spec.top_k, len(bank))

    neighbors_count: list[int] = []
    pos_rates: list[float] = []
    avg_returns: list[float] = []
    loss_rates: list[float] = []
    avg_distance: list[float] = []
    case_ids: list[str] = []
    for row in target_z:
        distances = np.sqrt(np.nanmean((bank_z - row) ** 2, axis=1))
        order = np.argsort(distances)[:top_k]
        rets = bank_returns[order]
        valid = np.isfinite(rets)
        rets = rets[valid]
        valid_order = order[valid]
        neighbors_count.append(int(len(rets)))
        if len(rets) >= spec.min_neighbors:
            pos_rates.append(round(float((rets > 0).mean()), 6))
            avg_returns.append(round(float(rets.mean()), 6))
            loss_rates.append(round(float((rets <= -5).mean()), 6))
            avg_distance.append(round(float(np.nanmean(distances[valid_order])), 6))
            case_ids.append(
                ";".join(
                    f"{bank_dates[idx]}:{bank_codes[idx]}:{distances[idx]:.3f}" for idx in valid_order[: min(3, len(valid_order))]
                )
            )
        else:
            pos_rates.append(np.nan)
            avg_returns.append(np.nan)
            loss_rates.append(np.nan)
            avg_distance.append(np.nan)
            case_ids.append("")

    out["analog_id"] = spec.analog_id
    out["analog_top_k"] = spec.top_k
    out["analog_min_neighbors"] = spec.min_neighbors
    out["analog_feature_count"] = len(features)
    out["analog_neighbor_count"] = neighbors_count
    out["analog_pos_rate"] = pos_rates
    out["analog_avg_return"] = avg_returns
    out["analog_loss_gt5_rate"] = loss_rates
    out["analog_avg_distance"] = avg_distance
    out["analog_top_case_refs"] = case_ids
    return out


def with_empty_analog_columns(frame: pd.DataFrame, spec: AnalogSpec) -> pd.DataFrame:
    out = frame.copy()
    out["analog_id"] = spec.analog_id
    out["analog_top_k"] = spec.top_k
    out["analog_min_neighbors"] = spec.min_neighbors
    out["analog_feature_count"] = 0
    out["analog_neighbor_count"] = 0
    out["analog_pos_rate"] = np.nan
    out["analog_avg_return"] = np.nan
    out["analog_loss_gt5_rate"] = np.nan
    out["analog_avg_distance"] = np.nan
    out["analog_top_case_refs"] = ""
    return out


def numeric_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = []
    for col in columns:
        values.append(pd.to_numeric(frame.get(col), errors="coerce").to_numpy(dtype=float))
    return np.vstack(values).T if values else np.empty((len(frame), 0))


def boolish(
    frame: pd.DataFrame,
    column: str,
    *,
    default: bool = False,
    numeric_zero_is_false: bool = False,
) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index)
    series = frame[column]
    if numeric_zero_is_false:
        return pd.to_numeric(series, errors="coerce").fillna(0).eq(0)
    return series.fillna(default).astype(bool)


def apply_analog_gates(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if frame.empty:
        return {
            "transfer_only": frame,
            "analog_pos_ge060": frame,
            "analog_pos_ge065": frame,
            "analog_pos_ge070": frame,
            "analog_pos_ge065_loss_le020": frame,
            "news_financial_clean_plus_analog065": frame,
            "chip_support_plus_analog065": frame,
            "analog_guard_remove_weak_cases": frame,
        }
    pos = pd.to_numeric(frame.get("analog_pos_rate"), errors="coerce")
    avg = pd.to_numeric(frame.get("analog_avg_return"), errors="coerce")
    loss = pd.to_numeric(frame.get("analog_loss_gt5_rate"), errors="coerce")
    neighbors = pd.to_numeric(frame.get("analog_neighbor_count"), errors="coerce")
    enough = neighbors >= pd.to_numeric(frame.get("analog_min_neighbors"), errors="coerce")
    no_hard = boolish(frame, "channel_hard_counter_count", default=False, numeric_zero_is_false=True)
    news_fin_mask = no_hard & boolish(frame, "news_low_warning") & (
        boolish(frame, "financial_no_recent_event")
        | (~boolish(frame, "financial_high_risk_event") & ~boolish(frame, "financial_missing"))
    )
    chip_clean_mask = no_hard & boolish(frame, "chip_support_visible") & boolish(frame, "kline_not_overheated")
    return {
        "transfer_only": frame.copy(),
        "analog_pos_ge060": frame[enough & (pos >= 0.60)].copy(),
        "analog_pos_ge065": frame[enough & (pos >= 0.65)].copy(),
        "analog_pos_ge070": frame[enough & (pos >= 0.70)].copy(),
        "analog_pos_ge065_loss_le020": frame[enough & (pos >= 0.65) & (loss <= 0.20) & (avg > 0)].copy(),
        "news_financial_clean_plus_analog065": frame[
            news_fin_mask & enough & (pos >= 0.65) & (avg > 0)
        ].copy(),
        "chip_support_plus_analog065": frame[
            chip_clean_mask & enough & (pos >= 0.65) & (loss <= 0.25)
        ].copy(),
        "analog_guard_remove_weak_cases": frame[~(enough & ((pos < 0.50) | ((loss > 0.30) & (avg <= 0))))].copy(),
    }


def evaluate_gate(
    *,
    branch_base: pd.DataFrame,
    transfer_base: pd.DataFrame,
    selected: pd.DataFrame,
    cfg: TransferConfig,
    target_block: str,
    spec: AnalogSpec,
    gate_id: str,
    threshold: float,
    validation_context: str,
) -> dict[str, Any]:
    branch_ret = pd.to_numeric(branch_base.get("return_20d"), errors="coerce").dropna()
    transfer_ret = pd.to_numeric(transfer_base.get("return_20d"), errors="coerce").dropna()
    ret = pd.to_numeric(selected.get("return_20d"), errors="coerce").dropna()
    excluded = transfer_base[~transfer_base.index.isin(selected.index)].copy()
    excluded_ret = pd.to_numeric(excluded.get("return_20d"), errors="coerce").dropna()
    return {
        "frequency": cfg.frequency,
        "target_block": target_block,
        "variant": cfg.variant,
        "cohort": cfg.cohort,
        "feature_set": cfg.feature_set,
        "model_name": cfg.model_name,
        "confirm_quantile": round(float(cfg.confirm_quantile), 4),
        "transfer_threshold": round(float(threshold), 8),
        "analog_id": spec.analog_id,
        "analog_top_k": spec.top_k,
        "analog_min_neighbors": spec.min_neighbors,
        "gate_id": gate_id,
        "validation_context": validation_context,
        "branch_rows": int(len(branch_base)),
        "transfer_rows": int(len(transfer_base)),
        "selected_rows": int(len(selected)),
        "selected_rate_vs_transfer": round(float(len(selected) / max(1, len(transfer_base))), 6),
        "branch_pos20": positive_rate(branch_ret),
        "branch_avg20": mean_value(branch_ret),
        "branch_loss_gt5": rate_le(branch_ret, -5),
        "transfer_pos20": positive_rate(transfer_ret),
        "transfer_avg20": mean_value(transfer_ret),
        "transfer_loss_gt5": rate_le(transfer_ret, -5),
        "selected_pos20": positive_rate(ret),
        "selected_avg20": mean_value(ret),
        "selected_loss_gt5": rate_le(ret, -5),
        "delta_pos_vs_transfer": delta(positive_rate(ret), positive_rate(transfer_ret)),
        "delta_avg_vs_transfer": delta(mean_value(ret), mean_value(transfer_ret)),
        "delta_loss_vs_transfer": delta(rate_le(ret, -5), rate_le(transfer_ret, -5)),
        "delta_pos_vs_branch": delta(positive_rate(ret), positive_rate(branch_ret)),
        "delta_avg_vs_branch": delta(mean_value(ret), mean_value(branch_ret)),
        "missed_transfer_positive_rows": int(pd.to_numeric(excluded_ret, errors="coerce").gt(0).sum()),
        "captured_transfer_loss_gt5_rows": int(pd.to_numeric(ret, errors="coerce").le(-5).sum()),
        "avg_analog_pos_rate": mean_value(selected.get("analog_pos_rate", pd.Series(dtype=float))),
        "avg_analog_avg_return": mean_value(selected.get("analog_avg_return", pd.Series(dtype=float))),
        "avg_analog_loss_gt5_rate": mean_value(selected.get("analog_loss_gt5_rate", pd.Series(dtype=float))),
        "avg_analog_neighbor_count": mean_value(selected.get("analog_neighbor_count", pd.Series(dtype=float))),
    }


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (frequency, variant, analog_id, gate_id), group in metrics.groupby(
        ["frequency", "variant", "analog_id", "gate_id"], sort=True
    ):
        h = group[group["target_block"].eq(FINAL_OOT)]
        prior = group[~group["target_block"].eq(FINAL_OOT)]
        if h.empty:
            continue
        h_row = h.iloc[0]
        prior_rows = pd.to_numeric(prior.get("selected_rows"), errors="coerce") if not prior.empty else pd.Series(dtype=float)
        prior_delta_pos = pd.to_numeric(prior.get("delta_pos_vs_transfer"), errors="coerce") if not prior.empty else pd.Series(dtype=float)
        prior_delta_avg = pd.to_numeric(prior.get("delta_avg_vs_transfer"), errors="coerce") if not prior.empty else pd.Series(dtype=float)
        if not prior.empty:
            prior_eval_mask = (
                pd.to_numeric(prior.get("selected_rows"), errors="coerce").fillna(0) >= 15
            ) & pd.to_numeric(prior.get("delta_pos_vs_transfer"), errors="coerce").notna() & pd.to_numeric(
                prior.get("delta_avg_vs_transfer"), errors="coerce"
            ).notna()
            prior_eval = prior[prior_eval_mask].copy()
        else:
            prior_eval = pd.DataFrame()
        prior_eval_rows = (
            pd.to_numeric(prior_eval.get("selected_rows"), errors="coerce") if not prior_eval.empty else pd.Series(dtype=float)
        )
        prior_eval_delta_pos = (
            pd.to_numeric(prior_eval.get("delta_pos_vs_transfer"), errors="coerce")
            if not prior_eval.empty
            else pd.Series(dtype=float)
        )
        prior_eval_delta_avg = (
            pd.to_numeric(prior_eval.get("delta_avg_vs_transfer"), errors="coerce")
            if not prior_eval.empty
            else pd.Series(dtype=float)
        )
        row = {
            "frequency": frequency,
            "variant": variant,
            "analog_id": analog_id,
            "gate_id": gate_id,
            "prior_blocks": int(prior["target_block"].nunique()) if not prior.empty else 0,
            "prior_evaluable_blocks": int(prior_eval["target_block"].nunique()) if not prior_eval.empty else 0,
            "prior_selected_rows_mean": mean_value(prior_rows),
            "prior_evaluable_selected_rows_mean": mean_value(prior_eval_rows),
            "prior_delta_pos_vs_transfer_mean": mean_value(prior_delta_pos),
            "prior_delta_avg_vs_transfer_mean": mean_value(prior_delta_avg),
            "prior_delta_pos_hit": positive_rate(prior_delta_pos),
            "prior_delta_avg_hit": positive_rate(prior_delta_avg),
            "prior_evaluable_delta_pos_vs_transfer_mean": mean_value(prior_eval_delta_pos),
            "prior_evaluable_delta_avg_vs_transfer_mean": mean_value(prior_eval_delta_avg),
            "prior_evaluable_delta_pos_hit": positive_rate(prior_eval_delta_pos),
            "prior_evaluable_delta_avg_hit": positive_rate(prior_eval_delta_avg),
            "h2026_branch_rows": h_row.get("branch_rows"),
            "h2026_transfer_rows": h_row.get("transfer_rows"),
            "h2026_selected_rows": h_row.get("selected_rows"),
            "h2026_selected_rate_vs_transfer": h_row.get("selected_rate_vs_transfer"),
            "h2026_branch_pos20": h_row.get("branch_pos20"),
            "h2026_transfer_pos20": h_row.get("transfer_pos20"),
            "h2026_selected_pos20": h_row.get("selected_pos20"),
            "h2026_selected_avg20": h_row.get("selected_avg20"),
            "h2026_selected_loss_gt5": h_row.get("selected_loss_gt5"),
            "h2026_delta_pos_vs_transfer": h_row.get("delta_pos_vs_transfer"),
            "h2026_delta_avg_vs_transfer": h_row.get("delta_avg_vs_transfer"),
            "h2026_delta_pos_vs_branch": h_row.get("delta_pos_vs_branch"),
            "h2026_delta_avg_vs_branch": h_row.get("delta_avg_vs_branch"),
            "h2026_missed_transfer_positive_rows": h_row.get("missed_transfer_positive_rows"),
            "h2026_captured_transfer_loss_gt5_rows": h_row.get("captured_transfer_loss_gt5_rows"),
            "h2026_avg_analog_pos_rate": h_row.get("avg_analog_pos_rate"),
            "h2026_avg_analog_avg_return": h_row.get("avg_analog_avg_return"),
            "h2026_avg_analog_loss_gt5_rate": h_row.get("avg_analog_loss_gt5_rate"),
            "h2026_avg_analog_neighbor_count": h_row.get("avg_analog_neighbor_count"),
        }
        row["promotion_status"] = analog_status(row)
        row["rank_score"] = rank_score(row)
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["promotion_status", "rank_score"], ascending=[True, False])
    return out.round(6)


def analog_status(row: dict[str, Any]) -> str:
    if str(row.get("gate_id")) == "transfer_only":
        return "transfer_reference"
    h_rows = safe_float(row.get("h2026_selected_rows"))
    prior_blocks = safe_float(row.get("prior_blocks"))
    prior_evaluable_blocks = safe_float(row.get("prior_evaluable_blocks"))
    prior_rows = safe_float(row.get("prior_selected_rows_mean"))
    prior_eval_rows = safe_float(row.get("prior_evaluable_selected_rows_mean"))
    prior_pos_hit = safe_float(row.get("prior_evaluable_delta_pos_hit"))
    prior_avg_hit = safe_float(row.get("prior_evaluable_delta_avg_hit"))
    h_pos = safe_float(row.get("h2026_selected_pos20"))
    h_avg = safe_float(row.get("h2026_selected_avg20"))
    h_loss = safe_float(row.get("h2026_selected_loss_gt5"))
    h_delta_pos = safe_float(row.get("h2026_delta_pos_vs_transfer"))
    h_delta_avg = safe_float(row.get("h2026_delta_avg_vs_transfer"))

    if h_rows < 15:
        return "reject_too_sparse"
    if h_delta_pos < 0 or h_delta_avg < 0:
        return "reject_or_false_filter_risk"
    if (
        prior_evaluable_blocks < 2
        and h_rows >= 25
        and h_pos >= 0.65
        and h_avg >= 4.0
        and h_delta_pos >= 0
    ):
        return "observe_latest_block_bright_time_support_insufficient"
    if (
        h_rows >= 40
        and prior_blocks >= 3
        and prior_evaluable_blocks >= 2
        and prior_eval_rows >= 20
        and prior_pos_hit >= 0.75
        and prior_avg_hit >= 0.75
        and h_pos >= 0.70
        and h_avg >= 5.0
        and h_loss <= 0.15
        and h_delta_pos >= 0.03
        and h_delta_avg >= 0
    ):
        return "green_candidate_for_ds_confirmation"
    if (
        h_rows >= 25
        and prior_blocks >= 3
        and prior_evaluable_blocks >= 2
        and prior_eval_rows >= 15
        and prior_pos_hit >= 0.50
        and h_pos >= 0.65
        and h_avg >= 4.0
        and h_delta_pos >= 0
    ):
        return "yellow_candidate_needs_fresh_panel"
    return "observe_diagnostic_only"


def panel_stability(
    target_scored: pd.DataFrame,
    *,
    transfer_with_analog: pd.DataFrame,
    analog_bank: pd.DataFrame,
    spec: AnalogSpec,
    gate_id: str,
    cfg: TransferConfig,
    threshold: float,
    panel_size: int,
    panel_seeds: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    codes = sorted(target_scored["code"].astype(str).str.zfill(6).unique())
    selected_index = set(transfer_with_analog.index)
    for seed in range(max(1, panel_seeds)):
        ordered = sorted(codes, key=lambda code: stable_hash_int("p0_transfer_analog_panel", seed, cfg.variant, gate_id, code))
        selected_codes = set(ordered[: min(panel_size, len(ordered))])
        panel = target_scored[target_scored["code"].astype(str).str.zfill(6).isin(selected_codes)].copy()
        panel_transfer = panel[panel.index.isin(selected_index)].copy()
        panel_transfer = add_analog_features(panel_transfer, analog_bank, spec=spec)
        panel_gate = apply_analog_gates(panel_transfer).get(gate_id, panel_transfer.iloc[0:0].copy())
        row = evaluate_gate(
            branch_base=panel,
            transfer_base=panel_transfer,
            selected=panel_gate,
            cfg=cfg,
            target_block=FINAL_OOT,
            spec=spec,
            gate_id=gate_id,
            threshold=threshold,
            validation_context="H2025_2_panel",
        )
        row["panel_seed"] = seed
        row["panel_size_codes"] = len(selected_codes)
        rows.append(row)
    return rows


def summarize_panels(panels: pd.DataFrame) -> pd.DataFrame:
    if panels.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (frequency, variant, analog_id, gate_id), group in panels.groupby(
        ["frequency", "variant", "analog_id", "gate_id"], sort=True
    ):
        rows.append(
            {
                "frequency": frequency,
                "variant": variant,
                "analog_id": analog_id,
                "gate_id": gate_id,
                "panels": int(group["panel_seed"].nunique()),
                "selected_rows_mean": round(float(pd.to_numeric(group["selected_rows"], errors="coerce").mean()), 3),
                "selected_rate_vs_transfer_mean±std": fmt_mean_std(group, "selected_rate_vs_transfer"),
                "selected_pos20_mean±std": fmt_mean_std(group, "selected_pos20"),
                "selected_avg20_mean±std": fmt_mean_std(group, "selected_avg20"),
                "selected_loss_gt5_mean±std": fmt_mean_std(group, "selected_loss_gt5"),
                "delta_pos_vs_transfer_mean±std": fmt_mean_std(group, "delta_pos_vs_transfer"),
                "delta_avg_vs_transfer_mean±std": fmt_mean_std(group, "delta_avg_vs_transfer"),
                "analog_pos_rate_mean±std": fmt_mean_std(group, "avg_analog_pos_rate"),
            }
        )
    return pd.DataFrame(rows)


def build_preview_rows(frame: pd.DataFrame, *, cfg: TransferConfig, spec: AnalogSpec, max_rows: int) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    gated = apply_analog_gates(frame)
    for gate_id, gate_frame in gated.items():
        if gate_id == "transfer_only" or gate_frame.empty:
            continue
        ordered = gate_frame.sort_values(["analog_pos_rate", "_transfer_score"], ascending=[False, False]).head(
            max(1, max_rows // max(1, len(gated) - 1))
        )
        for _, row in ordered.iterrows():
            rows.append(
                {
                    "date": str(row.get("date")),
                    "code": str(row.get("code")).zfill(6),
                    "time_block": str(row.get("target_block")),
                    "tool_id": "p0_transfer_analog_rag_v1",
                    "frequency": cfg.frequency,
                    "base_branch": "branch_stack_v1.small_buy_hold",
                    "variant": cfg.variant,
                    "analog_id": spec.analog_id,
                    "gate_id": gate_id,
                    "operation_action_cn": "小仓试探/持有",
                    "position_cap_hint": 0.10,
                    "transfer_score": preview_num(row.get("_transfer_score")),
                    "transfer_threshold": preview_num(row.get("_transfer_threshold")),
                    "analog_neighbor_count": preview_num(row.get("analog_neighbor_count")),
                    "analog_pos_rate": preview_num(row.get("analog_pos_rate")),
                    "analog_avg_return": preview_num(row.get("analog_avg_return")),
                    "analog_historical_tail_risk_rate": preview_num(row.get("analog_loss_gt5_rate")),
                    "analog_top_case_refs": str(row.get("analog_top_case_refs", "")),
                    "channel_support_count": preview_num(row.get("channel_support_count")),
                    "channel_hard_counter_count": preview_num(row.get("channel_hard_counter_count")),
                    "news_low_warning": bool(row.get("news_low_warning", False)),
                    "financial_no_recent_event": bool(row.get("financial_no_recent_event", False)),
                    "chip_support_visible": bool(row.get("chip_support_visible", False)),
                    "agent_instruction": "retrieve cited prior analog cases, compare current news/financial/chip context, and use this only as evidence support, not as an automatic exposure raise",
                    "auto_trade": False,
                }
            )
    return rows


def render_report(
    notes: list[str],
    configs: list[TransferConfig],
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    panel_summary: pd.DataFrame,
    hygiene: pd.DataFrame,
    paths: dict[str, Path],
) -> str:
    h2026 = metrics[metrics["target_block"].eq(FINAL_OOT)].copy() if not metrics.empty else pd.DataFrame()
    non_reference = summary[~summary["promotion_status"].astype(str).eq("transfer_reference")].copy() if not summary.empty else pd.DataFrame()
    promoted_like = non_reference[non_reference["promotion_status"].astype(str).str.contains("green|yellow", regex=True)]
    lines = [
        "# P0 Transfer Analog RAG v1",
        "",
        "本实验不调用 DeepSeek，不读取 API key/token。它把过去已发生、已可知结果的小仓候选作为相似案例库，检验 analog/RAG 是否能进一步确认 transfer + channel 候选。",
        "",
        "## Setup",
        "",
        f"- transfer_configs: `{len(configs)}` yellow variants from `p0_small_entry_transfer_confirmer_v1_summary.csv`",
        "- analog specs: `analog_k15_min10`, `analog_k30_min20`.",
        "- analog bank for each target block uses only earlier blocks; H2026_1 is final OOT.",
        "- Future returns are used only for offline metrics and past-case labels; preview JSONL excludes future/GT/result/label fields.",
        "",
        "## Coverage Notes",
        "",
    ]
    lines.extend([f"- {note}" for note in notes[-6:]])
    lines.extend(["", "## Main Verdict", ""])
    if promoted_like.empty:
        lines.append("- 没有 analog/RAG gate 从 transfer yellow 进一步晋级；当前相似案例检索不应作为默认硬过滤。")
    else:
        green = int(promoted_like["promotion_status"].astype(str).str.contains("green").sum())
        yellow = int(promoted_like["promotion_status"].astype(str).str.contains("yellow").sum())
        lines.append(f"- 出现 {green} 个 green、{yellow} 个 yellow analog/RAG 候选；仍必须先过 fresh panel 和 DS Flash/Pro on/off。")
    lines.extend(
        [
            "- 如果 analog gate 只在 H2026 高、但 prior hit 不足或 selected rows 太少，只能写成案例检索证据，不得写成自动买入阈值。",
            "",
            "## Summary",
            "",
            markdown_table(summary.head(100)),
            "",
            "## H2026 Detail",
            "",
            markdown_table(h2026.sort_values(["selected_pos20", "selected_avg20"], ascending=[False, False]).head(100)),
            "",
            "## H2026 Panel Stability",
            "",
            markdown_table(panel_summary.head(100)),
            "",
            "## Hygiene",
            "",
            markdown_table(hygiene) if not hygiene.empty else "_empty_",
            "",
            "## Decision Rules",
            "",
            "- analog/RAG 只能作为 Agent 解释、反证和相似案例复核材料；不能绕过新闻/公告/财报/同行/BookSkill 语义复核。",
            "- `analog_pos_rate` 来自历史相似案例，不是当前股票未来概率；必须同时检查样本数、距离、案例日期和当前信息差异。",
            "- 相似案例引用只暴露历史 date/code/distance，不暴露历史未来收益给 Agent preview。",
            "",
            "## Artifacts",
            "",
            *[f"- `{path}`" for path in paths.values()],
            "",
        ]
    )
    return "\n".join(lines)


def rank_score(row: dict[str, Any]) -> float:
    return (
        25 * safe_float(row.get("h2026_selected_pos20"))
        + safe_float(row.get("h2026_selected_avg20"))
        + 80 * safe_float(row.get("h2026_delta_pos_vs_transfer"))
        + 0.4 * safe_float(row.get("h2026_delta_avg_vs_transfer"))
        + 4 * safe_float(row.get("prior_delta_pos_hit"))
        + 3 * safe_float(row.get("prior_delta_avg_hit"))
        - 8 * safe_float(row.get("h2026_selected_loss_gt5"))
        - 0.02 * safe_float(row.get("h2026_missed_transfer_positive_rows"))
    )


def positive_rate(values: pd.Series | Any) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return round(float((values > 0).mean()), 6)


def mean_value(values: pd.Series | Any) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return round(float(values.mean()), 6)


def rate_le(values: pd.Series | Any, threshold: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return round(float((values <= threshold).mean()), 6)


def delta(value: float, base: float) -> float:
    if pd.isna(value) or pd.isna(base):
        return np.nan
    return round(float(value - base), 6)


def preview_num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return round(number, 6)


def fmt_mean_std(frame: pd.DataFrame, column: str) -> str:
    vals = pd.to_numeric(frame.get(column), errors="coerce").dropna()
    if vals.empty:
        return ""
    return f"{float(vals.mean()):.4f}±{float(vals.std() if len(vals) > 1 else 0.0):.4f}"


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    safe = frame.copy()
    cols = [col for col in safe.columns if not forbidden_field(col)]
    safe = safe[cols].fillna("")
    rows = safe.astype(str).values.tolist()
    return "\n".join(
        [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


if __name__ == "__main__":
    main()
