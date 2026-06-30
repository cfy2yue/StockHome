"""Audit P0 single-stock operation policies as user-facing actions.

This script is intentionally local and no-DeepSeek. It reuses the existing
walk-forward P0 decision stack, then evaluates the actions a user would
actually see: buy/add, hold, reduce/sell, wait, and data-review style outputs.

Future returns are used only for offline evaluation. Safe preview rows contain
decision-time scores and action thresholds, never GT labels or future returns.
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

from scripts.audit_p0_decision_stack_v1 import (  # noqa: E402
    ACTIVE_POSITION_THRESHOLD,
    BANK_20D_RETURN_PCT,
    FUTURE_OR_RESULT_FIELDS,
    FINAL_OOT,
    MIN_TARGET_ROWS,
    MIN_TRAIN_ROWS,
    MIN_VALID_ROWS,
    TARGET_BLOCKS,
    apply_frequency,
    apply_policy,
    build_scored_target,
    key_series,
    load_stack_frame,
    policy_names,
    safe_float,
    safe_prefix,
    stable_hash_int,
)
from scripts.audit_single_stock_review_quality import _rolling_split  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_PREFIX = "p0_operation_policy_v1"
DEFAULT_FREQUENCIES = "every_2_weeks,weekly_friday,weekly_tuesday"
DEFAULT_KLINE_GROUP = "kline_peer_chip_news_risk"
MAX_HGB_TRAIN_ROWS = 60000
PANEL_SIZE = 100
PANEL_SEEDS = 12

BUY_ACTIONS = {"buy_add", "small_buy_hold"}
REDUCE_ACTIONS = {"reduce_sell", "reduce_review"}
WAIT_ACTIONS = {"wait"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit P0 user-facing operation policy actions.")
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--frequencies", default=DEFAULT_FREQUENCIES)
    parser.add_argument("--kline-feature-group", default=DEFAULT_KLINE_GROUP)
    parser.add_argument("--max-hgb-train-rows", type=int, default=MAX_HGB_TRAIN_ROWS)
    parser.add_argument("--panel-size", type=int, default=PANEL_SIZE)
    parser.add_argument("--panel-seeds", type=int, default=PANEL_SEEDS)
    parser.add_argument("--preview-max-rows", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frequencies = [item.strip() for item in args.frequencies.split(",") if item.strip()]
    frame, feature_groups, notes = load_stack_frame()

    metrics_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
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
                policy_frame = with_operation_actions(apply_policy(scored, policy_name))
                metrics_rows.append(evaluate_operation_policy(policy_frame, frequency, target_block, policy_name))
                action_rows.extend(evaluate_action_slices(policy_frame, frequency, target_block, policy_name))
                if target_block == FINAL_OOT:
                    panel_rows.extend(
                        evaluate_h2026_panels(
                            policy_frame,
                            frequency=frequency,
                            policy_name=policy_name,
                            panel_size=args.panel_size,
                            panel_seeds=args.panel_seeds,
                        )
                    )
                    if policy_name in {"opp_kline_confirm_no_raise", "branch_stack_v1"}:
                        preview_rows.extend(build_safe_preview(policy_frame, frequency, policy_name, args.preview_max_rows))

    metrics = pd.DataFrame(metrics_rows)
    action_detail = pd.DataFrame(action_rows)
    panel = pd.DataFrame(panel_rows)
    preview = pd.DataFrame(preview_rows)
    hygiene = pd.DataFrame(hygiene_rows)
    summary = summarize_operation_metrics(metrics)
    panel_summary = summarize_panel_metrics(panel)

    prefix = safe_prefix(args.output_prefix)
    paths = {
        "metrics": REPORT_DIR / f"{prefix}_metrics.csv",
        "action_detail": REPORT_DIR / f"{prefix}_action_detail.csv",
        "summary": REPORT_DIR / f"{prefix}_summary.csv",
        "panel_detail": REPORT_DIR / f"{prefix}_h2026_panel_detail.csv",
        "panel_summary": REPORT_DIR / f"{prefix}_h2026_panel_summary.csv",
        "preview": REPORT_DIR / f"{prefix}_agent_preview_no_gt.jsonl",
        "hygiene": REPORT_DIR / f"{prefix}_hygiene.csv",
        "report": REPORT_DIR / f"{prefix}.md",
    }
    metrics.to_csv(paths["metrics"], index=False, encoding="utf-8-sig")
    action_detail.to_csv(paths["action_detail"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    panel.to_csv(paths["panel_detail"], index=False, encoding="utf-8-sig")
    panel_summary.to_csv(paths["panel_summary"], index=False, encoding="utf-8-sig")
    hygiene.to_csv(paths["hygiene"], index=False, encoding="utf-8-sig")
    write_jsonl(paths["preview"], preview)
    paths["report"].write_text(
        render_report(args, notes, summary, metrics, action_detail, panel_summary, hygiene, paths),
        encoding="utf-8",
    )
    print("A股研究Agent")
    print(f"metrics={len(metrics)} action_rows={len(action_detail)} panels={len(panel)} preview={len(preview)}")
    print(f"report={paths['report']}")


def with_operation_actions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["operation_action"] = out.apply(operation_action, axis=1)
    out["operation_action_cn"] = out["operation_action"].map(
        {
            "buy_add": "买入/加仓",
            "small_buy_hold": "小仓试探/持有",
            "reduce_review": "减仓/小仓复核",
            "reduce_sell": "卖出/回避",
            "wait": "等待/不操作",
        }
    )
    out["operation_threshold"] = out.apply(operation_threshold, axis=1)
    return out


def operation_action(row: pd.Series) -> str:
    pos = safe_float(row.get("target_position"))
    hard_risk = bool(row.get("risk_queue_high_hard_counter")) or bool(row.get("kline_hard_risk"))
    review = bool(row.get("risk_review_queue"))
    if hard_risk or pos <= 0:
        return "reduce_sell"
    if review:
        return "reduce_review"
    if pos >= 0.60:
        return "buy_add"
    if pos >= ACTIVE_POSITION_THRESHOLD:
        return "small_buy_hold"
    return "wait"


def operation_threshold(row: pd.Series) -> str:
    action = operation_action(row)
    opp = safe_float(row.get("opp_score"))
    opp_th = safe_float(row.get("opp_threshold"))
    kline = safe_float(row.get("kline_opp_score"))
    kline_th = safe_float(row.get("kline_opp_threshold"))
    risk = safe_float(row.get("kline_risk_score"))
    risk_th = safe_float(row.get("kline_risk_threshold"))
    if action == "buy_add":
        return f"opp_score>={opp_th:.4f} and kline_score>={kline_th:.4f} and kline_risk<{risk_th:.4f}"
    if action == "small_buy_hold":
        return f"target_position>={ACTIVE_POSITION_THRESHOLD:.2f}; require one more non-price confirmation before add"
    if action == "reduce_review":
        return "risk_review_queue=true; cap position <=5% until evidence improves"
    if action == "reduce_sell":
        return f"hard risk or kline_risk>={risk_th:.4f}; avoid new exposure"
    return f"wait until opp_score/kline_score exceed thresholds; current={opp:.4f}/{kline:.4f}"


def evaluate_operation_policy(frame: pd.DataFrame, frequency: str, target_block: str, policy_name: str) -> dict[str, Any]:
    ret = pd.to_numeric(frame["return_20d"], errors="coerce")
    cash = pd.to_numeric(frame["cash_adjusted_return_20d"], errors="coerce")
    pos = pd.to_numeric(frame["target_position"], errors="coerce").fillna(0.0)
    actions = frame["operation_action"].astype(str)
    buy = frame[actions.isin(BUY_ACTIONS)]
    buy_add = frame[actions.eq("buy_add")]
    small_buy = frame[actions.eq("small_buy_hold")]
    reduce = frame[actions.isin(REDUCE_ACTIONS)]
    wait = frame[actions.isin(WAIT_ACTIONS)]
    buy_ret = pd.to_numeric(buy["return_20d"], errors="coerce")
    buy_add_ret = pd.to_numeric(buy_add["return_20d"], errors="coerce")
    small_buy_ret = pd.to_numeric(small_buy["return_20d"], errors="coerce")
    reduce_ret = pd.to_numeric(reduce["return_20d"], errors="coerce")
    wait_ret = pd.to_numeric(wait["return_20d"], errors="coerce")
    base_pos = float((ret > 0).mean()) if len(ret) else np.nan
    return {
        "frequency": frequency,
        "target_block": target_block,
        "policy_name": policy_name,
        "rows": int(len(frame)),
        "base_pos20": round(base_pos, 6) if not math.isnan(base_pos) else np.nan,
        "base_avg20": round(float(ret.mean()), 6) if len(ret) else np.nan,
        "cash_pos20": round(float((cash > 0).mean()), 6) if len(cash) else np.nan,
        "cash_avg20": round(float(cash.mean()), 6) if len(cash) else np.nan,
        "cash_std20": round(float(cash.std(ddof=0)), 6) if len(cash) else np.nan,
        "avg_target_position": round(float(pos.mean()), 6),
        "buy_or_hold_rate": round(float(len(buy) / max(1, len(frame))), 6),
        "buy_or_hold_pos20": positive_rate(buy_ret),
        "buy_or_hold_avg20": mean_value(buy_ret),
        "buy_or_hold_loss_gt5": rate_le(buy_ret, -5),
        "buy_add_rate": round(float(len(buy_add) / max(1, len(frame))), 6),
        "buy_add_pos20": positive_rate(buy_add_ret),
        "buy_add_avg20": mean_value(buy_add_ret),
        "buy_add_loss_gt5": rate_le(buy_add_ret, -5),
        "small_buy_rate": round(float(len(small_buy) / max(1, len(frame))), 6),
        "small_buy_pos20": positive_rate(small_buy_ret),
        "small_buy_avg20": mean_value(small_buy_ret),
        "small_buy_loss_gt5": rate_le(small_buy_ret, -5),
        "small_buy_gain_gt5": rate_ge(small_buy_ret, 5),
        "reduce_rate": round(float(len(reduce) / max(1, len(frame))), 6),
        "reduce_correct_nonpositive": rate_le(reduce_ret, 0),
        "reduce_caught_loss_gt5": rate_le(reduce_ret, -5),
        "reduce_false_positive": rate_gt(reduce_ret, 0),
        "reduce_false_large_gain_gt5": rate_ge(reduce_ret, 5),
        "wait_rate": round(float(len(wait) / max(1, len(frame))), 6),
        "wait_missed_positive": rate_gt(wait_ret, 0),
        "wait_missed_large_gain_gt5": rate_ge(wait_ret, 5),
        "clear_direction_rate": round(float((actions.ne("wait")).mean()), 6) if len(actions) else np.nan,
        "active_exposure": round(float(pos.mean()), 6),
    }


def evaluate_action_slices(frame: pd.DataFrame, frequency: str, target_block: str, policy_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for action, group in frame.groupby("operation_action", sort=True):
        ret = pd.to_numeric(group["return_20d"], errors="coerce")
        rows.append(
            {
                "frequency": frequency,
                "target_block": target_block,
                "policy_name": policy_name,
                "operation_action": action,
                "operation_action_cn": str(group["operation_action_cn"].iloc[0]),
                "rows": int(len(group)),
                "row_rate": round(float(len(group) / max(1, len(frame))), 6),
                "positive_20d_rate": positive_rate(ret),
                "avg_return_20d": mean_value(ret),
                "loss_gt5_rate": rate_le(ret, -5),
                "gain_gt5_rate": rate_ge(ret, 5),
            }
        )
    return rows


def evaluate_h2026_panels(
    frame: pd.DataFrame,
    *,
    frequency: str,
    policy_name: str,
    panel_size: int,
    panel_seeds: int,
) -> list[dict[str, Any]]:
    codes = sorted(frame["code"].astype(str).str.zfill(6).unique())
    rows: list[dict[str, Any]] = []
    for seed in range(max(1, int(panel_seeds))):
        ordered = sorted(codes, key=lambda code: stable_hash_int("p0_operation_panel", seed, frequency, policy_name, code))
        selected = set(ordered[: min(panel_size, len(ordered))])
        panel = frame[frame["code"].astype(str).str.zfill(6).isin(selected)].copy()
        row = evaluate_operation_policy(panel, frequency, FINAL_OOT, policy_name)
        row["panel_seed"] = seed
        row["panel_size_codes"] = len(selected)
        rows.append(row)
    return rows


def summarize_operation_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in metrics.groupby(["frequency", "policy_name"], sort=True):
        h = group[group["target_block"].eq(FINAL_OOT)]
        prior = group[~group["target_block"].eq(FINAL_OOT)]
        hrow = h.iloc[0] if not h.empty else pd.Series(dtype=object)
        row = {
            "frequency": keys[0],
            "policy_name": keys[1],
            "prior_blocks": int(prior["target_block"].nunique()),
            "prior_buy_add_pos20_mean": mean_col(prior, "buy_add_pos20"),
            "prior_buy_add_avg20_mean": mean_col(prior, "buy_add_avg20"),
            "prior_small_buy_pos20_mean": mean_col(prior, "small_buy_pos20"),
            "prior_small_buy_avg20_mean": mean_col(prior, "small_buy_avg20"),
            "prior_reduce_correct_mean": mean_col(prior, "reduce_correct_nonpositive"),
            "prior_cash_avg20_mean": mean_col(prior, "cash_avg20"),
            "h2026_cash_pos20": get_val(hrow, "cash_pos20"),
            "h2026_cash_avg20": get_val(hrow, "cash_avg20"),
            "h2026_active_exposure": get_val(hrow, "active_exposure"),
            "h2026_buy_add_rate": get_val(hrow, "buy_add_rate"),
            "h2026_buy_add_pos20": get_val(hrow, "buy_add_pos20"),
            "h2026_buy_add_avg20": get_val(hrow, "buy_add_avg20"),
            "h2026_buy_add_loss_gt5": get_val(hrow, "buy_add_loss_gt5"),
            "h2026_small_buy_rate": get_val(hrow, "small_buy_rate"),
            "h2026_small_buy_pos20": get_val(hrow, "small_buy_pos20"),
            "h2026_small_buy_avg20": get_val(hrow, "small_buy_avg20"),
            "h2026_small_buy_loss_gt5": get_val(hrow, "small_buy_loss_gt5"),
            "h2026_small_buy_gain_gt5": get_val(hrow, "small_buy_gain_gt5"),
            "h2026_reduce_rate": get_val(hrow, "reduce_rate"),
            "h2026_reduce_correct_nonpositive": get_val(hrow, "reduce_correct_nonpositive"),
            "h2026_reduce_false_large_gain_gt5": get_val(hrow, "reduce_false_large_gain_gt5"),
            "h2026_wait_missed_large_gain_gt5": get_val(hrow, "wait_missed_large_gain_gt5"),
            "h2026_clear_direction_rate": get_val(hrow, "clear_direction_rate"),
        }
        row["promotion_status"] = promotion_status(row)
        row["rank_score"] = operation_rank_score(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["promotion_status", "rank_score"], ascending=[True, False])


def summarize_panel_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in panel.groupby(["frequency", "policy_name"], sort=True):
        rows.append(
            {
                "frequency": keys[0],
                "policy_name": keys[1],
                "panels": int(group["panel_seed"].nunique()),
                "cash_pos20_mean±std": fmt_mean_std(group, "cash_pos20"),
                "cash_avg20_mean±std": fmt_mean_std(group, "cash_avg20"),
                "buy_add_pos20_mean±std": fmt_mean_std(group, "buy_add_pos20"),
                "buy_add_avg20_mean±std": fmt_mean_std(group, "buy_add_avg20"),
                "small_buy_pos20_mean±std": fmt_mean_std(group, "small_buy_pos20"),
                "small_buy_avg20_mean±std": fmt_mean_std(group, "small_buy_avg20"),
                "reduce_correct_mean±std": fmt_mean_std(group, "reduce_correct_nonpositive"),
                "false_large_gain_mean±std": fmt_mean_std(group, "reduce_false_large_gain_gt5"),
                "buy_add_rate_mean±std": fmt_mean_std(group, "buy_add_rate"),
            }
        )
    return pd.DataFrame(rows)


def promotion_status(row: dict[str, Any]) -> str:
    policy_name = str(row.get("policy_name", ""))
    if policy_name.endswith("_baseline"):
        return "baseline_gray_reference"
    buy_pos = safe_float(row.get("h2026_buy_add_pos20"))
    buy_avg = safe_float(row.get("h2026_buy_add_avg20"))
    buy_rate = safe_float(row.get("h2026_buy_add_rate"))
    small_pos = safe_float(row.get("h2026_small_buy_pos20"))
    small_avg = safe_float(row.get("h2026_small_buy_avg20"))
    small_rate = safe_float(row.get("h2026_small_buy_rate"))
    small_loss = safe_float(row.get("h2026_small_buy_loss_gt5"))
    reduce_correct = safe_float(row.get("h2026_reduce_correct_nonpositive"))
    false_large = safe_float(row.get("h2026_reduce_false_large_gain_gt5"))
    prior_buy = safe_float(row.get("prior_buy_add_pos20_mean"))
    prior_small = safe_float(row.get("prior_small_buy_pos20_mean"))
    if (
        buy_pos >= 0.60
        and buy_avg > 0
        and 0.03 <= buy_rate <= 0.25
        and reduce_correct >= 0.55
        and false_large <= 0.25
        and prior_buy >= 0.60
    ):
        return "green_candidate_for_ds_action_ablation"
    if buy_pos >= 0.53 and buy_avg > 0 and 0.03 <= buy_rate <= 0.30 and prior_buy >= 0.55:
        return "yellow_action_candidate_needs_fresh_panel"
    if (
        small_pos >= 0.60
        and small_avg > 0
        and 0.01 <= small_rate <= 0.08
        and small_loss <= 0.22
        and prior_small >= 0.65
    ):
        return "yellow_small_entry_candidate_for_ds_confirmation"
    if buy_avg > 0 and buy_pos >= 0.50:
        return "observe_action_diagnostic_only"
    return "reject_or_reference_only"


def operation_rank_score(row: dict[str, Any]) -> float:
    return (
        25 * safe_float(row.get("h2026_buy_add_pos20"))
        + safe_float(row.get("h2026_buy_add_avg20"))
        + 8 * safe_float(row.get("h2026_small_buy_pos20"))
        + 0.5 * safe_float(row.get("h2026_small_buy_avg20"))
        + 5 * safe_float(row.get("h2026_reduce_correct_nonpositive"))
        - 8 * safe_float(row.get("h2026_reduce_false_large_gain_gt5"))
        - 2 * max(0.0, safe_float(row.get("h2026_buy_add_rate")) - 0.25)
    )


def build_safe_preview(frame: pd.DataFrame, frequency: str, policy_name: str, max_rows: int) -> list[dict[str, Any]]:
    ordered = frame.sort_values(["target_position", "opp_quantile_in_date", "kline_opp_score"], ascending=[False, False, False])
    rows: list[dict[str, Any]] = []
    for _, row in ordered.head(max_rows).iterrows():
        rows.append(
            {
                "date": str(row.get("date")),
                "code": str(row.get("code")).zfill(6),
                "name": str(row.get("name", "")),
                "time_block": str(row.get("time_block")),
                "tool_id": "p0_operation_policy_v1",
                "frequency": frequency,
                "policy_name": policy_name,
                "operation_action": str(row.get("operation_action")),
                "operation_action_cn": str(row.get("operation_action_cn")),
                "target_position": round(safe_float(row.get("target_position")), 4),
                "operation_threshold": str(row.get("operation_threshold")),
                "opp_score": round(safe_float(row.get("opp_score")), 6),
                "opp_threshold": round(safe_float(row.get("opp_threshold")), 6),
                "opp_quantile_in_date": round(safe_float(row.get("opp_quantile_in_date")), 6),
                "kline_opp_score": round(safe_float(row.get("kline_opp_score")), 6),
                "kline_opp_threshold": round(safe_float(row.get("kline_opp_threshold")), 6),
                "kline_risk_score": round(safe_float(row.get("kline_risk_score")), 6),
                "kline_risk_threshold": round(safe_float(row.get("kline_risk_threshold")), 6),
                "risk_review_queue": bool(row.get("risk_review_queue")),
                "risk_queue_high_hard_counter": bool(row.get("risk_queue_high_hard_counter")),
                "confirmation_count": int(row.get("confirmation_count", 0) or 0),
                "auto_trade": False,
                "evidence_needed_before_real_action": "news/financial/peer/bookskill confirmation and user position context",
            }
        )
    return rows


def render_report(
    args: argparse.Namespace,
    notes: list[str],
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    action_detail: pd.DataFrame,
    panel_summary: pd.DataFrame,
    hygiene: pd.DataFrame,
    paths: dict[str, Path],
) -> str:
    h2026 = metrics[metrics["target_block"].eq(FINAL_OOT)].copy()
    top_actions = action_detail[
        action_detail["target_block"].eq(FINAL_OOT)
        & action_detail["policy_name"].isin(["opp_kline_confirm_no_raise", "branch_stack_v1"])
    ].copy()
    lines = [
        "# P0 Operation Policy v1",
        "",
        "本报告把 P0 单支盯盘从“研究分级/观察权重”改成用户实际关心的动作审计：买入/加仓、持有、小仓复核、卖出/回避、等待。实验完全本地运行，不调用 DeepSeek，不读取 API key。",
        "",
        "## Setup",
        "",
        f"- frequencies: `{args.frequencies}`",
        f"- kline_feature_group: `{args.kline_feature_group}`",
        "- walk-forward: prior blocks train, previous block threshold validation, current block test; H2026_1 is final OOT.",
        "- `return_20d` 只用于离线评估；agent preview 不含 GT/future fields。",
        "- buy/add success uses future 20d positive rate and average return; reduce/sell success uses non-positive/loss capture and false large-gain veto cost.",
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
            "## H2026 Operation Detail",
            "",
            markdown_table(
                h2026[
                    [
                        "frequency",
                        "policy_name",
                        "cash_pos20",
                        "cash_avg20",
                        "active_exposure",
                        "buy_add_rate",
                        "buy_add_pos20",
                        "buy_add_avg20",
                        "buy_add_loss_gt5",
                        "small_buy_rate",
                        "small_buy_pos20",
                        "small_buy_avg20",
                        "small_buy_loss_gt5",
                        "small_buy_gain_gt5",
                        "reduce_rate",
                        "reduce_correct_nonpositive",
                        "reduce_false_large_gain_gt5",
                        "wait_missed_large_gain_gt5",
                    ]
                ].sort_values(["buy_add_pos20", "buy_add_avg20"], ascending=[False, False])
                if not h2026.empty
                else pd.DataFrame()
            ),
            "",
            "## H2026 Action Slices For Main Branches",
            "",
            markdown_table(top_actions) if not top_actions.empty else "_empty_",
            "",
            "## H2026 100-Stock Panel Stability",
            "",
            markdown_table(panel_summary),
            "",
            "## Hygiene",
            "",
            markdown_table(hygiene) if not hygiene.empty else "_empty_",
            "",
            "## Decision",
            "",
            "- `green_candidate_for_ds_action_ablation` 才能进入小规模 DS Flash/Pro 动作消融；否则不能宣称 P0 买入/卖出引擎完成。",
            "- `yellow_small_entry_candidate_for_ds_confirmation` 表示“确认但不过热”的小仓试探分叉值得进入 DS 语义确认；它不是全仓买入信号。",
            "- 若 buy/add 命中率高但 buy_add_rate 过低，说明只是极窄机会提示；若 reduce_false_large_gain_gt5 高，说明排雷会错过大涨。",
            "- P0 用户端可以输出明确动作，但必须同时返回阈值、反证、数据缺口、复查条件和 `auto_trade=false`。",
            "- 若 H2026 面板均值未达标，下一步应优化新闻/财报/BookSkill/同行确认，而不是继续叠 K 线模型。",
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
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def assert_no_future_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FUTURE_OR_RESULT_FIELDS or str(key).startswith("return_") or "future" in str(key).lower():
                raise ValueError(f"future/result field leaked: {key}")
            assert_no_future_fields(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_future_fields(item)


def positive_rate(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return round(float((values > 0).mean()), 6) if len(values) else np.nan


def mean_value(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return round(float(values.mean()), 6) if len(values) else np.nan


def rate_le(values: pd.Series, threshold: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return round(float((values <= threshold).mean()), 6) if len(values) else np.nan


def rate_gt(values: pd.Series, threshold: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return round(float((values > threshold).mean()), 6) if len(values) else np.nan


def rate_ge(values: pd.Series, threshold: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return round(float((values >= threshold).mean()), 6) if len(values) else np.nan


def mean_col(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return round(float(values.mean()), 6) if len(values) else np.nan


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
