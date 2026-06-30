"""Audit a local P0 single-stock decision stack.

This is a no-DeepSeek, time-safe experiment. It combines the currently accepted
single-stock opportunity scorer, the capped risk review queue logic, and the
new multiscale K-line/peer/chip scorer into deterministic operation policies.

Future 20d returns are used only for offline evaluation. Agent preview rows are
field-whitelisted and contain only decision-time scores, thresholds, and
operation hints.
"""
from __future__ import annotations

import argparse
import hashlib
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

from scripts.audit_p0_multiscale_kline_peer_tool_v1 import (  # noqa: E402
    BANK_20D_RETURN_PCT,
    apply_frequency,
    attach_scores as attach_kline_scores,
    build_feature_map as build_kline_feature_map,
    choose_opportunity_threshold as choose_kline_opportunity_threshold,
    choose_risk_threshold as choose_kline_risk_threshold,
    fit_model as fit_kline_model,
    opportunity_label as kline_opportunity_label,
    risk_label as kline_risk_label,
)
from scripts.audit_single_stock_channel_scorer_v1 import (  # noqa: E402
    CHANNEL_FEATURES,
    attach_channel_features,
    load_safe_channel_scores,
)
from scripts.audit_single_stock_opportunity_scorer_v2 import (  # noqa: E402
    FUTURE_OR_RESULT_FIELDS as OPPORTUNITY_FUTURE_FIELDS,
    load_experiment_frame,
)
from scripts.audit_single_stock_review_quality import (  # noqa: E402
    FINAL_OOT,
    MIN_TARGET_ROWS,
    MIN_TRAIN_ROWS,
    MIN_VALID_ROWS,
    TARGET_BLOCKS,
    block_base_metrics,
    choose_opportunity_threshold,
    fit_risk_model,
    score_risk,
)
from scripts.audit_single_stock_risk_calibration_v2 import (  # noqa: E402
    MAX_DEFAULT_REVIEW_EXPOSURE,
    add_review_priority_score,
    choose_capped_policy,
    select_top_pct_per_date,
)
from scripts.run_lightweight_ml_channel_experiment import (  # noqa: E402
    _rolling_split,
    fit_additive_bin_model,
    score_frame,
)


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_PREFIX = "p0_decision_stack_v1"
DEFAULT_FREQUENCIES = "every_2_weeks,weekly_friday,weekly_tuesday"
DEFAULT_KLINE_GROUP = "kline_peer_chip_news_risk"
MAX_HGB_TRAIN_ROWS = 60000
ACTIVE_POSITION_THRESHOLD = 0.35
PANEL_SIZE = 100
PANEL_SEEDS = 3

FUTURE_OR_RESULT_FIELDS = set(OPPORTUNITY_FUTURE_FIELDS) | {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "positive_20d",
    "loss_gt5",
    "loss_gt5_flag",
    "single_stock_label",
    "single_stock_action",
    "gt_status",
    "gt_pass",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit P0 local decision stack without DS/API calls.")
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--frequencies", default=DEFAULT_FREQUENCIES)
    parser.add_argument("--kline-feature-group", default=DEFAULT_KLINE_GROUP)
    parser.add_argument("--max-hgb-train-rows", type=int, default=MAX_HGB_TRAIN_ROWS)
    parser.add_argument("--panel-size", type=int, default=PANEL_SIZE)
    parser.add_argument("--panel-seeds", type=int, default=PANEL_SEEDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frequencies = [item.strip() for item in args.frequencies.split(",") if item.strip()]
    frame, feature_groups, notes = load_stack_frame()
    metrics_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
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
                        "status": "skip_insufficient_rows",
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
                        "status": "skip_model_unavailable",
                        "train_rows": len(train),
                        "validation_rows": len(validation),
                        "target_rows": len(target),
                    }
                )
                continue
            for policy_name in policy_names():
                policy_frame = apply_policy(scored, policy_name)
                row = evaluate_policy(policy_frame, frequency=frequency, target_block=target_block, policy_name=policy_name)
                metrics_rows.append(row)
                if target_block == FINAL_OOT:
                    panel_rows.extend(panel_stability(policy_frame, row, panel_size=args.panel_size, panel_seeds=args.panel_seeds))
                    if policy_name in {"branch_stack_v1", "opp_kline_confirm_no_raise"}:
                        preview_rows.extend(build_agent_preview_rows(policy_frame, row, max_rows=1000))

    metrics = pd.DataFrame(metrics_rows)
    panels = pd.DataFrame(panel_rows)
    preview = pd.DataFrame(preview_rows)
    hygiene = pd.DataFrame(hygiene_rows)
    summary = summarize(metrics)
    prefix = safe_prefix(args.output_prefix)
    paths = {
        "metrics": REPORT_DIR / f"{prefix}_metrics.csv",
        "summary": REPORT_DIR / f"{prefix}_summary.csv",
        "panels": REPORT_DIR / f"{prefix}_h2026_panel_stability.csv",
        "preview": REPORT_DIR / f"{prefix}_agent_preview.jsonl",
        "hygiene": REPORT_DIR / f"{prefix}_hygiene.csv",
        "report": REPORT_DIR / f"{prefix}.md",
    }
    metrics.to_csv(paths["metrics"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    panels.to_csv(paths["panels"], index=False, encoding="utf-8-sig")
    hygiene.to_csv(paths["hygiene"], index=False, encoding="utf-8-sig")
    write_jsonl(paths["preview"], preview)
    paths["report"].write_text(render_report(args, notes, summary, metrics, panels, hygiene, paths), encoding="utf-8")
    print("A股研究Agent")
    print(f"rows={len(frame)} metrics={len(metrics)} summary={len(summary)} preview={len(preview)}")
    print(f"report={paths['report']}")


def load_stack_frame() -> tuple[pd.DataFrame, dict[str, list[str]], list[str]]:
    frame, feature_groups, notes = load_experiment_frame()
    channel_scores, channel_features = load_safe_channel_scores()
    frame = attach_channel_features(frame, channel_scores)
    frame["return_20d"] = pd.to_numeric(frame["return_20d"], errors="coerce")
    frame["positive_20d"] = frame["return_20d"].gt(0).astype(float)
    frame["loss_gt5_flag"] = frame["return_20d"].le(-5).astype(float)
    notes = [*notes, f"channel_features={len(channel_features)}"]
    feature_groups = dict(feature_groups)
    feature_groups["risk_with_channel"] = sorted(set(feature_groups.get("baseline_existing", []) + CHANNEL_FEATURES))
    return frame.dropna(subset=["return_20d"]).reset_index(drop=True), feature_groups, notes


def build_scored_target(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    target: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    *,
    kline_feature_group: str,
    max_hgb_train_rows: int,
) -> pd.DataFrame:
    opp_features = feature_groups.get("baseline_existing", [])
    if len(opp_features) < 5:
        return pd.DataFrame()
    opp_model = fit_additive_bin_model(train, opp_features, feature_group="p0_stack_opportunity")
    validation_opp = score_frame(validation, opp_model)
    target_opp = score_frame(target, opp_model)
    opp_threshold, opp_validation = choose_opportunity_threshold(validation_opp)

    kline_features = build_kline_feature_map(train).get(kline_feature_group, [])
    if len(kline_features) < 5:
        return pd.DataFrame()
    kline_opp_model = fit_kline_model(
        train,
        kline_features,
        label=kline_opportunity_label(train),
        model_name="hgb",
        feature_group=kline_feature_group,
        max_hgb_train_rows=max_hgb_train_rows,
    )
    kline_risk_model = fit_kline_model(
        train,
        kline_features,
        label=kline_risk_label(train),
        model_name="hgb",
        feature_group=kline_feature_group,
        max_hgb_train_rows=max_hgb_train_rows,
    )
    if kline_opp_model is None or kline_risk_model is None:
        return pd.DataFrame()
    validation_kline = attach_kline_scores(validation, kline_opp_model, kline_risk_model)
    target_kline = attach_kline_scores(target, kline_opp_model, kline_risk_model)
    kline_opp_threshold, kline_opp_validation = choose_kline_opportunity_threshold(validation_kline)
    kline_risk_threshold, kline_risk_validation = choose_kline_risk_threshold(validation_kline)

    risk_features = feature_groups.get("risk_with_channel", [])
    risk_model = fit_risk_model(train, risk_features)
    validation_risk = add_review_priority_score(score_risk(validation, risk_model))
    target_risk = add_review_priority_score(score_risk(target, risk_model))
    risk_pct, risk_validation = choose_capped_policy(validation_risk, "review_priority_score")
    risk_pct = min(float(risk_pct), 0.10, MAX_DEFAULT_REVIEW_EXPOSURE)
    risk_selected = select_top_pct_per_date(target_risk, "review_priority_score", risk_pct)
    risk_keys = set(key_series(risk_selected))

    scored = target[["date", "code", "time_block", "return_20d"] + (["name"] if "name" in target.columns else [])].copy()
    scored["opp_score"] = pd.to_numeric(target_opp["ml_score"], errors="coerce")
    scored["opp_threshold"] = float(opp_threshold)
    scored["opp_quantile_in_date"] = scored["opp_score"].groupby(scored["date"].astype(str)).rank(pct=True, method="average")
    scored["opp_active"] = scored["opp_score"] >= float(opp_threshold)
    scored["opp_strong"] = scored["opp_active"] & (scored["opp_quantile_in_date"] >= 0.75)
    scored["kline_opp_score"] = pd.to_numeric(target_kline["opp_score"], errors="coerce")
    scored["kline_risk_score"] = pd.to_numeric(target_kline["risk_score"], errors="coerce")
    scored["kline_opp_threshold"] = float(kline_opp_threshold)
    scored["kline_risk_threshold"] = float(kline_risk_threshold)
    scored["kline_active"] = (scored["kline_opp_score"] >= float(kline_opp_threshold)) & (
        scored["kline_risk_score"] < float(kline_risk_threshold)
    )
    scored["kline_hard_risk"] = scored["kline_risk_score"] >= float(kline_risk_threshold)
    scored["risk_review_queue"] = pd.Series(key_series(scored), index=scored.index).isin(risk_keys)
    scored["risk_review_cap_pct"] = risk_pct
    scored = scored.merge(
        target_risk[
            [
                "date",
                "code",
                "risk_score",
                "review_priority_score",
                "channel_hard_counter_prob",
                "channel_soft_gap_prob",
                "channel_positive_support_prob",
                "channel_score_coverage",
            ]
        ],
        on=["date", "code"],
        how="left",
    )
    scored["risk_queue_high_hard_counter"] = scored["risk_review_queue"] & (
        pd.to_numeric(scored["channel_hard_counter_prob"], errors="coerce").fillna(0.0) >= 0.95
    )
    scored["confirmation_count"] = scored[["opp_active", "kline_active"]].astype(int).sum(axis=1)
    scored["tool_threshold_context"] = (
        "opp_validation_pos="
        + str(round(float(opp_validation.get("positive_20d_rate", np.nan)), 4))
        + ";kline_validation_active_pos_delta="
        + str(round(float(kline_opp_validation.get("active_pos_delta", np.nan)), 4))
        + ";risk_validation_recall="
        + str(round(float(risk_validation.get("risk_recall", np.nan)), 4))
    )
    return scored


def policy_names() -> list[str]:
    return [
        "hold_all_baseline",
        "bank_all_baseline",
        "opp_only",
        "opp_with_risk_no_raise",
        "kline_only_no_raise",
        "opp_or_kline_no_raise",
        "opp_kline_confirm_no_raise",
        "branch_stack_v1",
    ]


def apply_policy(frame: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    out = frame.copy()
    if policy_name == "hold_all_baseline":
        position = pd.Series(1.0, index=out.index)
    elif policy_name == "bank_all_baseline":
        position = pd.Series(0.0, index=out.index)
    elif policy_name == "opp_only":
        position = np.where(out["opp_active"], 0.60, 0.10)
    elif policy_name == "kline_only_no_raise":
        position = np.where(out["kline_active"], 0.50, 0.10)
    elif policy_name == "opp_or_kline_no_raise":
        position = np.where(out["opp_active"] | out["kline_active"], 0.50, 0.10)
    elif policy_name == "opp_kline_confirm_no_raise":
        position = np.select(
            [out["opp_active"] & out["kline_active"], out["opp_active"] | out["kline_active"]],
            [0.60, 0.20],
            default=0.05,
        )
    elif policy_name == "branch_stack_v1":
        position = np.select(
            [
                out["opp_strong"] & out["kline_active"],
                out["opp_active"] & out["kline_active"],
                out["opp_active"] | out["kline_active"],
            ],
            [0.70, 0.55, 0.20],
            default=0.05,
        )
    elif policy_name == "opp_with_risk_no_raise":
        position = np.where(out["opp_active"], 0.60, 0.10)
    else:
        raise ValueError(f"unknown policy: {policy_name}")
    out["target_position"] = pd.to_numeric(pd.Series(position, index=out.index), errors="coerce").fillna(0.0)
    if policy_name not in {"hold_all_baseline", "bank_all_baseline", "opp_only", "kline_only_no_raise"}:
        out.loc[out["risk_review_queue"], "target_position"] = np.minimum(out.loc[out["risk_review_queue"], "target_position"], 0.05)
        out.loc[out["risk_queue_high_hard_counter"] | out["kline_hard_risk"], "target_position"] = np.minimum(
            out.loc[out["risk_queue_high_hard_counter"] | out["kline_hard_risk"], "target_position"],
            0.0,
        )
    out["policy_name"] = policy_name
    out["operation_hint"] = out.apply(operation_hint, axis=1)
    ret = pd.to_numeric(out["return_20d"], errors="coerce")
    pos = pd.to_numeric(out["target_position"], errors="coerce").fillna(0.0)
    out["cash_adjusted_return_20d"] = pos * ret + (1.0 - pos) * BANK_20D_RETURN_PCT
    return out


def operation_hint(row: pd.Series) -> str:
    pos = safe_float(row.get("target_position"))
    if bool(row.get("risk_queue_high_hard_counter")) or pos <= 0:
        return "avoid_or_reduce"
    if bool(row.get("risk_review_queue")):
        return "hold_small_or_reduce_review"
    if pos >= 0.60:
        return "trial_buy_or_add_if_user_confirms"
    if pos >= ACTIVE_POSITION_THRESHOLD:
        return "small_buy_or_hold"
    if pos > 0:
        return "wait_or_hold_tiny_position"
    return "wait_for_better_evidence"


def evaluate_policy(frame: pd.DataFrame, *, frequency: str, target_block: str, policy_name: str) -> dict[str, Any]:
    ret = pd.to_numeric(frame["return_20d"], errors="coerce")
    cash = pd.to_numeric(frame["cash_adjusted_return_20d"], errors="coerce")
    pos = pd.to_numeric(frame["target_position"], errors="coerce").fillna(0.0)
    active = frame[pos >= ACTIVE_POSITION_THRESHOLD]
    active_ret = pd.to_numeric(active["return_20d"], errors="coerce")
    base = block_base_metrics(frame)
    return {
        "frequency": frequency,
        "target_block": target_block,
        "policy_name": policy_name,
        "candidate_rows": int(len(frame)),
        "base_pos": base["base_pos"],
        "base_avg_return": base["base_mean_ret"],
        "base_loss_gt5": base["base_loss_gt5"],
        "strategy_positive_rate": round(float((cash > 0).mean()), 6) if len(cash) else np.nan,
        "strategy_avg_return": round(float(cash.mean()), 6) if len(cash) else np.nan,
        "strategy_std_return": round(float(cash.std(ddof=0)), 6) if len(cash) else np.nan,
        "excess_vs_hold_avg": round(float(cash.mean() - ret.mean()), 6) if len(cash) and len(ret) else np.nan,
        "avg_target_position": round(float(pos.mean()), 6),
        "active_rows": int(len(active)),
        "active_rate": round(float(len(active) / max(1, len(frame))), 6),
        "active_pos_rate": round(float((active_ret > 0).mean()), 6) if len(active_ret) else np.nan,
        "active_avg_return": round(float(active_ret.mean()), 6) if len(active_ret) else np.nan,
        "active_loss_gt5_rate": round(float((active_ret <= -5).mean()), 6) if len(active_ret) else np.nan,
        "active_delta_pos_vs_base": round(float((active_ret > 0).mean()) - float(base["base_pos"]), 6)
        if len(active_ret)
        else np.nan,
        "active_delta_avg_vs_base": round(float(active_ret.mean()) - float(base["base_mean_ret"]), 6)
        if len(active_ret)
        else np.nan,
        "risk_review_rate": round(float(pd.to_numeric(frame["risk_review_queue"], errors="coerce").mean()), 6),
        "hard_risk_rate": round(float(pd.to_numeric(frame["risk_queue_high_hard_counter"], errors="coerce").mean()), 6),
        "opp_active_rate": round(float(pd.to_numeric(frame["opp_active"], errors="coerce").mean()), 6),
        "kline_active_rate": round(float(pd.to_numeric(frame["kline_active"], errors="coerce").mean()), 6),
    }


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in metrics.groupby(["frequency", "policy_name"], sort=True):
        h = group[group["target_block"].eq(FINAL_OOT)]
        prior = group[~group["target_block"].eq(FINAL_OOT)]
        hrow = h.iloc[0] if not h.empty else pd.Series(dtype=object)
        opp_ref = metrics[
            metrics["frequency"].eq(keys[0])
            & metrics["policy_name"].eq("opp_only")
            & metrics["target_block"].eq(FINAL_OOT)
        ]
        opp_row = opp_ref.iloc[0] if not opp_ref.empty else pd.Series(dtype=object)
        row = {
            "frequency": keys[0],
            "policy_name": keys[1],
            "prior_blocks": int(prior["target_block"].nunique()),
            "prior_strategy_avg_mean": mean(prior, "strategy_avg_return"),
            "prior_active_pos_mean": mean(prior, "active_pos_rate"),
            "prior_active_avg_mean": mean(prior, "active_avg_return"),
            "prior_strategy_avg_hit_rate": hit_rate(prior, "strategy_avg_return", 0),
            "prior_active_avg_delta_hit_rate": hit_rate(prior, "active_delta_avg_vs_base", 0),
            "h2026_strategy_pos": val(hrow, "strategy_positive_rate"),
            "h2026_strategy_avg": val(hrow, "strategy_avg_return"),
            "h2026_excess_vs_hold": val(hrow, "excess_vs_hold_avg"),
            "h2026_avg_position": val(hrow, "avg_target_position"),
            "h2026_active_rate": val(hrow, "active_rate"),
            "h2026_active_pos": val(hrow, "active_pos_rate"),
            "h2026_active_avg": val(hrow, "active_avg_return"),
            "h2026_active_loss": val(hrow, "active_loss_gt5_rate"),
            "h2026_delta_active_pos_vs_opp": val(hrow, "active_pos_rate") - val(opp_row, "active_pos_rate"),
            "h2026_delta_active_avg_vs_opp": val(hrow, "active_avg_return") - val(opp_row, "active_avg_return"),
            "h2026_delta_strategy_avg_vs_opp": val(hrow, "strategy_avg_return") - val(opp_row, "strategy_avg_return"),
        }
        row["promotion_status"] = promotion_status(row)
        row["rank_score"] = rank_score(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["promotion_status", "rank_score"], ascending=[True, False])


def promotion_status(row: dict[str, Any]) -> str:
    if str(row["policy_name"]).endswith("_baseline"):
        return "baseline_gray_reference"
    h_active = safe_float(row.get("h2026_active_pos"))
    h_avg = safe_float(row.get("h2026_active_avg"))
    h_rate = safe_float(row.get("h2026_active_rate"))
    prior_hit = safe_float(row.get("prior_active_avg_delta_hit_rate"))
    delta_opp_avg = safe_float(row.get("h2026_delta_active_avg_vs_opp"))
    delta_opp_strat = safe_float(row.get("h2026_delta_strategy_avg_vs_opp"))
    if h_active >= 0.60 and h_avg > 0 and 0.03 <= h_rate <= 0.35 and prior_hit >= 0.75 and delta_opp_avg >= 0:
        return "green_candidate_for_ds_ablation"
    if h_active >= 0.50 and h_avg > 0 and prior_hit >= 0.50 and (delta_opp_avg > 0 or delta_opp_strat > 0):
        return "yellow_candidate_needs_fresh_panel"
    if h_avg > 0 and delta_opp_avg > 0:
        return "observe_diagnostic_only"
    return "reject_or_reference_only"


def rank_score(row: dict[str, Any]) -> float:
    return (
        20 * safe_float(row.get("h2026_active_pos"))
        + safe_float(row.get("h2026_active_avg"))
        + 2 * safe_float(row.get("prior_active_avg_delta_hit_rate"))
        + safe_float(row.get("h2026_delta_active_avg_vs_opp"))
        - 2 * max(0.0, safe_float(row.get("h2026_active_rate")) - 0.35)
    )


def panel_stability(frame: pd.DataFrame, metrics: dict[str, Any], *, panel_size: int, panel_seeds: int = PANEL_SEEDS) -> list[dict[str, Any]]:
    rows = []
    codes = sorted(frame["code"].astype(str).unique())
    for seed in range(max(1, int(panel_seeds))):
        ordered = sorted(codes, key=lambda code: stable_hash_int("p0_stack_panel", seed, metrics["frequency"], metrics["policy_name"], code))
        selected_codes = set(ordered[: min(panel_size, len(ordered))])
        panel = frame[frame["code"].astype(str).isin(selected_codes)].copy()
        evaluated = evaluate_policy(
            panel,
            frequency=metrics["frequency"],
            target_block=metrics["target_block"],
            policy_name=metrics["policy_name"],
        )
        rows.append(
            {
                "frequency": metrics["frequency"],
                "policy_name": metrics["policy_name"],
                "target_block": metrics["target_block"],
                "panel_seed": seed,
                "panel_size_codes": len(selected_codes),
                "strategy_positive_rate": evaluated["strategy_positive_rate"],
                "strategy_avg_return": evaluated["strategy_avg_return"],
                "active_rate": evaluated["active_rate"],
                "active_pos_rate": evaluated["active_pos_rate"],
                "active_avg_return": evaluated["active_avg_return"],
                "active_loss_gt5_rate": evaluated["active_loss_gt5_rate"],
                "avg_target_position": evaluated["avg_target_position"],
            }
        )
    return rows


def build_agent_preview_rows(frame: pd.DataFrame, metrics: dict[str, Any], max_rows: int = 1000) -> list[dict[str, Any]]:
    sample = frame.sort_values(["target_position", "opp_quantile_in_date", "kline_opp_score"], ascending=[False, False, False]).head(max_rows)
    rows = []
    for _, row in sample.iterrows():
        rows.append(
            {
                "date": row["date"],
                "code": str(row["code"]).zfill(6),
                "name": str(row.get("name", "")),
                "time_block": row["time_block"],
                "tool_id": "p0_decision_stack_v1",
                "frequency": metrics["frequency"],
                "policy_name": metrics["policy_name"],
                "target_position": round(safe_float(row.get("target_position")), 4),
                "operation_hint": str(row.get("operation_hint", "")),
                "opp_score": round(safe_float(row.get("opp_score")), 6),
                "opp_threshold": round(safe_float(row.get("opp_threshold")), 6),
                "opp_quantile_in_date": round(safe_float(row.get("opp_quantile_in_date")), 6),
                "kline_opp_score": round(safe_float(row.get("kline_opp_score")), 6),
                "kline_opp_threshold": round(safe_float(row.get("kline_opp_threshold")), 6),
                "kline_risk_score": round(safe_float(row.get("kline_risk_score")), 6),
                "kline_risk_threshold": round(safe_float(row.get("kline_risk_threshold")), 6),
                "risk_review_queue": bool(row.get("risk_review_queue")),
                "risk_review_cap_pct": round(safe_float(row.get("risk_review_cap_pct")), 6),
                "confirmation_count": int(row.get("confirmation_count", 0) or 0),
                "threshold_context": str(row.get("tool_threshold_context", "")),
                "source_ref_ids": "single_stock_opportunity_scorer_v2;p0_multiscale_kline_peer_tool_v1;single_stock_risk_calibration_v2",
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return rows


def render_report(
    args: argparse.Namespace,
    notes: list[str],
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    panels: pd.DataFrame,
    hygiene: pd.DataFrame,
    paths: dict[str, Path],
) -> str:
    h2026 = metrics[metrics["target_block"].eq(FINAL_OOT)].copy()
    panel_summary = summarize_panels(panels)
    lines = [
        "# P0 Decision Stack v1",
        "",
        "本实验是本地 walk-forward 审计，不调用 DeepSeek。目标是验证 P0 单支盯盘是否能通过机会工具、K线/同行/筹码确认、风险 no-raise 护栏的分支融合，获得比单一工具更稳定的操作路径。",
        "",
        "## Setup",
        "",
        f"- frequencies: `{args.frequencies}`",
        f"- kline_feature_group: `{args.kline_feature_group}`",
        "- split: train = prior blocks before validation, validation = previous block, target = current block; H2026_1 is final OOT.",
        "- labels/future returns are used only for offline evaluation; agent preview contains no return or label fields.",
        "- baseline rows are gray references: hold_all_baseline and bank_all_baseline.",
        "",
        "## Coverage Notes",
        "",
    ]
    lines.extend([f"- {note}" for note in notes[-10:]])
    lines.extend(
        [
            "",
            "## Main Summary",
            "",
            markdown_table(summary.head(24)),
            "",
            "## H2026 Detail",
            "",
            markdown_table(
                h2026[
                    [
                        "frequency",
                        "policy_name",
                        "strategy_positive_rate",
                        "strategy_avg_return",
                        "excess_vs_hold_avg",
                        "avg_target_position",
                        "active_rate",
                        "active_pos_rate",
                        "active_avg_return",
                        "active_loss_gt5_rate",
                        "opp_active_rate",
                        "kline_active_rate",
                        "risk_review_rate",
                    ]
                ].sort_values(["active_pos_rate", "active_avg_return"], ascending=[False, False])
                if not h2026.empty
                else pd.DataFrame()
            ),
            "",
            "## H2026 100-Stock Panel Stability",
            "",
            markdown_table(panel_summary),
            "",
            "## Hygiene",
            "",
            markdown_table(hygiene) if not hygiene.empty else "_empty_",
            "",
            "## Interpretation",
            "",
            "- `green_candidate_for_ds_ablation` 才能进入小规模 Flash/Pro Agent 消融；否则只作为本地证据或下一轮分支假设。",
            "- `yellow_candidate_needs_fresh_panel` 表示融合在 H2026 有改善但尚需更多 fresh panels 或 DS 复核。",
            "- 如果融合策略只提高含现金正收益率而 active_pos 不高，说明它主要是防守/降仓，不代表主动买入 alpha。",
            "- 风险队列在本脚本中只能 no-raise：它可以压低仓位，不能单独提高仓位。",
            "",
            "## Artifacts",
            "",
        ]
    )
    lines.extend([f"- `{path}`" for path in paths.values()])
    return "\n".join(lines) + "\n"


def summarize_panels(panels: pd.DataFrame) -> pd.DataFrame:
    if panels.empty:
        return panels
    rows = []
    for keys, group in panels.groupby(["frequency", "policy_name"], sort=True):
        rows.append(
            {
                "frequency": keys[0],
                "policy_name": keys[1],
                "panels": int(group["panel_seed"].nunique()),
                "strategy_pos_mean±std": fmt_mean_std(group, "strategy_positive_rate"),
                "strategy_avg_mean±std": fmt_mean_std(group, "strategy_avg_return"),
                "active_pos_mean±std": fmt_mean_std(group, "active_pos_rate"),
                "active_avg_mean±std": fmt_mean_std(group, "active_avg_return"),
                "active_loss_mean±std": fmt_mean_std(group, "active_loss_gt5_rate"),
            }
        )
    return pd.DataFrame(rows)


def key_series(frame: pd.DataFrame) -> list[tuple[str, str]]:
    return list(zip(frame["date"].astype(str), frame["code"].astype(str).str.zfill(6)))


def assert_no_future_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FUTURE_OR_RESULT_FIELDS or str(key).startswith("return_") or str(key).endswith("_20d_return"):
                raise ValueError(f"future/result field leaked: {key}")
            assert_no_future_fields(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_future_fields(item)


def write_jsonl(path: Path, frame: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in frame.to_dict("records"):
            assert_no_future_fields(record)
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def safe_prefix(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value).strip("_") or DEFAULT_PREFIX


def stable_hash_int(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(out) else out


def mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    return round(float(pd.to_numeric(frame[column], errors="coerce").mean()), 6)


def hit_rate(frame: pd.DataFrame, column: str, threshold: float) -> float:
    if frame.empty or column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return round(float((values > threshold).mean()), 6) if len(values) else 0.0


def val(row: pd.Series, column: str) -> float:
    if row.empty:
        return np.nan
    value = safe_float(row.get(column))
    return round(value, 6)


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
