"""Audit case-memory/RAG guards for the P0 small-entry branch.

This is a local, no-DeepSeek experiment. It rebuilds the walk-forward P0
operation rows, filters the user-facing `branch_stack_v1.small_buy_hold`
branch, and asks whether existing case memory/RAG conditions can improve that
branch without overfitting or false-vetoing too many future winners.

Future returns are used only for offline evaluation after retrieval. Evidence
packs and agent previews are decision-time only.
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
    FINAL_OOT,
    MIN_TARGET_ROWS,
    MIN_TRAIN_ROWS,
    MIN_VALID_ROWS,
    TARGET_BLOCKS,
    apply_frequency,
    apply_policy,
    build_scored_target,
    load_stack_frame,
    safe_prefix,
    stable_hash_int,
)
from scripts.audit_p0_friday_stack_case_memory_v1 import (  # noqa: E402
    GUARD_POLICIES,
    build_guard_detail,
    build_safe_preview,
    markdown_table,
    summarize_conditions,
    summarize_guard_policies,
    write_jsonl,
)
from scripts.audit_p0_operation_policy_v1 import with_operation_actions  # noqa: E402
from scripts.audit_single_stock_review_quality import _rolling_split  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_PREFIX = "p0_small_entry_case_memory_v1"
DEFAULT_FREQUENCIES = "weekly_friday,every_2_weeks,weekly_tuesday"
DEFAULT_KLINE_GROUP = "kline_peer_chip_news_risk"
DEFAULT_POLICY = "branch_stack_v1"
DEFAULT_ACTION = "small_buy_hold"
MAX_HGB_TRAIN_ROWS = 60000
PANEL_SIZE = 100
PANEL_SEEDS = 12
MIN_PROMOTION_PRIOR_BLOCKS = 2
MIN_PROMOTION_PRIOR_RETAINED_ROWS_MEAN = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit P0 small-entry case-memory/RAG guards locally.")
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--frequencies", default=DEFAULT_FREQUENCIES)
    parser.add_argument("--kline-feature-group", default=DEFAULT_KLINE_GROUP)
    parser.add_argument("--policy-name", default=DEFAULT_POLICY)
    parser.add_argument("--operation-action", default=DEFAULT_ACTION)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-applicable-conditions", type=int, default=2)
    parser.add_argument("--panel-size", type=int, default=PANEL_SIZE)
    parser.add_argument("--panel-seeds", type=int, default=PANEL_SEEDS)
    parser.add_argument("--preview-max-rows", type=int, default=300)
    parser.add_argument("--max-hgb-train-rows", type=int, default=MAX_HGB_TRAIN_ROWS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frequencies = [item.strip() for item in args.frequencies.split(",") if item.strip()]
    frame, feature_groups, notes = load_stack_frame()

    detail_frames: list[pd.DataFrame] = []
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
                        "status": "skip_stack_model_unavailable",
                        "train_rows": len(train),
                        "validation_rows": len(validation),
                        "target_rows": len(target),
                    }
                )
                continue
            branch = with_operation_actions(apply_policy(scored, args.policy_name))
            small = branch[
                branch["operation_action"].astype(str).eq(args.operation_action)
                & pd.to_numeric(branch["return_20d"], errors="coerce").notna()
            ].copy()
            if small.empty:
                hygiene_rows.append(
                    {
                        "frequency": frequency,
                        "target_block": target_block,
                        "status": "skip_no_target_action_rows",
                        "train_rows": len(train),
                        "validation_rows": len(validation),
                        "target_rows": len(target),
                    }
                )
                continue
            detail = build_guard_detail(
                small,
                top_k=args.top_k,
                min_applicable_conditions=args.min_applicable_conditions,
            )
            detail["frequency"] = frequency
            detail["target_block"] = target_block
            detail["operation_action"] = args.operation_action
            detail_frames.append(detail)

    detail_all = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    block_metrics = build_block_metrics(detail_all)
    summary = summarize_block_metrics(block_metrics)
    panel_detail = build_h2026_panel_metrics(detail_all, panel_size=args.panel_size, panel_seeds=args.panel_seeds)
    panel_summary = summarize_panel_metrics(panel_detail)
    condition_summary = summarize_conditions(detail_all)
    safe_preview = build_safe_preview(detail_all, max_rows=args.preview_max_rows) if not detail_all.empty else []
    hygiene = build_hygiene(args, detail_all, block_metrics, safe_preview, hygiene_rows)

    prefix = safe_prefix(args.output_prefix)
    paths = {
        "detail": REPORT_DIR / f"{prefix}_detail.csv",
        "block_metrics": REPORT_DIR / f"{prefix}_block_metrics.csv",
        "summary": REPORT_DIR / f"{prefix}_summary.csv",
        "panel_detail": REPORT_DIR / f"{prefix}_h2026_panel_detail.csv",
        "panel_summary": REPORT_DIR / f"{prefix}_h2026_panel_summary.csv",
        "conditions": REPORT_DIR / f"{prefix}_condition_summary.csv",
        "preview": REPORT_DIR / f"{prefix}_agent_preview_no_gt.jsonl",
        "hygiene": REPORT_DIR / f"{prefix}_hygiene.csv",
        "report": REPORT_DIR / f"{prefix}.md",
    }
    detail_all.to_csv(paths["detail"], index=False, encoding="utf-8-sig")
    block_metrics.to_csv(paths["block_metrics"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    panel_detail.to_csv(paths["panel_detail"], index=False, encoding="utf-8-sig")
    panel_summary.to_csv(paths["panel_summary"], index=False, encoding="utf-8-sig")
    condition_summary.to_csv(paths["conditions"], index=False, encoding="utf-8-sig")
    hygiene.to_csv(paths["hygiene"], index=False, encoding="utf-8-sig")
    write_jsonl(paths["preview"], safe_preview)
    paths["report"].write_text(
        render_report(args, notes, summary, block_metrics, panel_summary, condition_summary, hygiene, paths),
        encoding="utf-8",
    )
    print("A股研究Agent")
    print(f"detail_rows={len(detail_all)} block_metrics={len(block_metrics)} summary={len(summary)} preview={len(safe_preview)}")
    print(f"report={paths['report']}")


def build_block_metrics(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for keys, group in detail.groupby(["frequency", "target_block"], sort=True):
        metrics = summarize_guard_policies(group)
        metrics.insert(0, "target_block", keys[1])
        metrics.insert(0, "frequency", keys[0])
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_block_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in metrics.groupby(["frequency", "policy"], sort=True):
        h = group[group["target_block"].eq(FINAL_OOT)]
        prior = group[~group["target_block"].eq(FINAL_OOT)]
        hrow = h.iloc[0] if not h.empty else pd.Series(dtype=object)
        row = {
            "frequency": keys[0],
            "guard_policy": keys[1],
            "prior_blocks": int(prior["target_block"].nunique()),
            "prior_retained_rows_mean": mean_col(prior, "retained_rows"),
            "prior_retained_rate_mean": mean_col(prior, "retained_rate"),
            "prior_delta_pos_mean": mean_col(prior, "delta_active_pos_vs_no_guard"),
            "prior_delta_avg_mean": mean_col(prior, "delta_active_avg_vs_no_guard"),
            "prior_delta_pos_hit": hit_rate(prior, "delta_active_pos_vs_no_guard", 0),
            "prior_delta_avg_hit": hit_rate(prior, "delta_active_avg_vs_no_guard", 0),
            "h2026_total_rows": get_val(hrow, "total_active_rows"),
            "h2026_retained_rows": get_val(hrow, "retained_rows"),
            "h2026_retained_rate": get_val(hrow, "retained_rate"),
            "h2026_retained_pos20": get_val(hrow, "retained_pos20"),
            "h2026_retained_avg20_pp": get_val(hrow, "retained_avg20_pp"),
            "h2026_retained_loss_gt5": get_val(hrow, "retained_loss_gt5_rate"),
            "h2026_dropped_rows": get_val(hrow, "dropped_rows"),
            "h2026_dropped_pos20": get_val(hrow, "dropped_pos20"),
            "h2026_dropped_avg20_pp": get_val(hrow, "dropped_avg20_pp"),
            "h2026_captured_loss_gt5_rows": get_val(hrow, "captured_loss_gt5_rows"),
            "h2026_false_veto_positive_rows": get_val(hrow, "false_veto_positive_rows"),
            "h2026_delta_pos": get_val(hrow, "delta_active_pos_vs_no_guard"),
            "h2026_delta_avg": get_val(hrow, "delta_active_avg_vs_no_guard"),
            "h2026_delta_loss": get_val(hrow, "delta_loss_gt5_vs_no_guard"),
        }
        row["promotion_status"] = promotion_status(row)
        row["rank_score"] = rank_score(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["promotion_status", "rank_score"], ascending=[True, False])


def build_h2026_panel_metrics(detail: pd.DataFrame, *, panel_size: int, panel_seeds: int) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    h2026 = detail[detail["target_block"].eq(FINAL_OOT)].copy()
    for frequency, freq_group in h2026.groupby("frequency", sort=True):
        codes = sorted(freq_group["code"].astype(str).str.zfill(6).unique())
        for seed in range(max(1, int(panel_seeds))):
            ordered = sorted(codes, key=lambda code: stable_hash_int("p0_small_entry_case_memory", seed, frequency, code))
            selected = set(ordered[: min(panel_size, len(ordered))])
            panel = freq_group[freq_group["code"].astype(str).str.zfill(6).isin(selected)].copy()
            if panel.empty:
                continue
            metrics = summarize_guard_policies(panel)
            for _, row in metrics.iterrows():
                out = row.to_dict()
                out["frequency"] = frequency
                out["panel_seed"] = seed
                out["panel_size_codes"] = len(selected)
                rows.append(out)
    return pd.DataFrame(rows)


def summarize_panel_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in panel.groupby(["frequency", "policy"], sort=True):
        rows.append(
            {
                "frequency": keys[0],
                "guard_policy": keys[1],
                "panels": int(group["panel_seed"].nunique()),
                "retained_rows_mean": mean_col(group, "retained_rows"),
                "retained_rate_mean±std": fmt_mean_std(group, "retained_rate"),
                "retained_pos20_mean±std": fmt_mean_std(group, "retained_pos20"),
                "retained_avg20_mean±std": fmt_mean_std(group, "retained_avg20_pp"),
                "retained_loss_gt5_mean±std": fmt_mean_std(group, "retained_loss_gt5_rate"),
                "delta_pos_mean±std": fmt_mean_std(group, "delta_active_pos_vs_no_guard"),
                "delta_avg_mean±std": fmt_mean_std(group, "delta_active_avg_vs_no_guard"),
                "false_veto_positive_mean": mean_col(group, "false_veto_positive_rows"),
                "captured_loss_gt5_mean": mean_col(group, "captured_loss_gt5_rows"),
            }
        )
    return pd.DataFrame(rows)


def promotion_status(row: dict[str, Any]) -> str:
    policy = str(row.get("guard_policy", ""))
    if policy == "no_case_guard":
        return "branch_reference"
    retained_rate = as_float(row.get("h2026_retained_rate"))
    retained_pos = as_float(row.get("h2026_retained_pos20"))
    retained_avg = as_float(row.get("h2026_retained_avg20_pp"))
    loss = as_float(row.get("h2026_retained_loss_gt5"))
    delta_pos = as_float(row.get("h2026_delta_pos"))
    delta_avg = as_float(row.get("h2026_delta_avg"))
    delta_loss = as_float(row.get("h2026_delta_loss"))
    false_veto = as_float(row.get("h2026_false_veto_positive_rows"))
    captured_loss = as_float(row.get("h2026_captured_loss_gt5_rows"))
    prior_blocks = as_float(row.get("prior_blocks"))
    prior_rows = as_float(row.get("prior_retained_rows_mean"))
    prior_pos_hit = as_float(row.get("prior_delta_pos_hit"))
    prior_avg_hit = as_float(row.get("prior_delta_avg_hit"))
    false_veto_ok = false_veto <= captured_loss * 1.5 + 5
    if (
        retained_rate >= 0.50
        and retained_pos >= 0.65
        and delta_pos >= 0.03
        and delta_avg >= 0
        and delta_loss <= 0
        and loss <= 0.22
        and false_veto_ok
        and prior_blocks >= MIN_PROMOTION_PRIOR_BLOCKS
        and prior_rows >= MIN_PROMOTION_PRIOR_RETAINED_ROWS_MEAN
        and prior_pos_hit >= 0.67
        and prior_avg_hit >= 0.67
    ):
        return "green_candidate_for_small_ds_smoke"
    if (
        retained_rate >= 0.35
        and retained_pos >= 0.62
        and delta_pos >= 0.01
        and delta_avg >= 0
        and false_veto_ok
        and prior_blocks >= MIN_PROMOTION_PRIOR_BLOCKS
        and prior_rows >= MIN_PROMOTION_PRIOR_RETAINED_ROWS_MEAN
        and prior_pos_hit >= 0.50
    ):
        return "yellow_candidate_needs_fresh_panel"
    if retained_avg > 0 and retained_pos >= 0.58:
        return "observe_diagnostic_only"
    return "reject_or_reference_only"


def rank_score(row: dict[str, Any]) -> float:
    return (
        25 * as_float(row.get("h2026_retained_pos20"))
        + as_float(row.get("h2026_retained_avg20_pp"))
        + 100 * as_float(row.get("h2026_delta_pos"))
        + 0.5 * as_float(row.get("h2026_delta_avg"))
        - 6 * as_float(row.get("h2026_retained_loss_gt5"))
        - 0.05 * as_float(row.get("h2026_false_veto_positive_rows"))
        + as_float(row.get("prior_delta_pos_hit"))
    )


def build_hygiene(
    args: argparse.Namespace,
    detail: pd.DataFrame,
    block_metrics: pd.DataFrame,
    safe_preview: list[dict[str, Any]],
    hygiene_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    context = "\n".join(json.dumps(row, ensure_ascii=False) for row in safe_preview)
    future_hits = sorted(field for field in ["return_20d", "gt_status", "positive_20d", "loss_gt5", "future_return_20d"] if field in context)
    rows = [
        {
            "scope": "small_entry_case_memory_audit",
            "frequencies": args.frequencies,
            "policy_name": args.policy_name,
            "operation_action": args.operation_action,
            "detail_rows": len(detail),
            "block_metric_rows": len(block_metrics),
            "safe_preview_rows": len(safe_preview),
            "safe_preview_future_field_hits": ";".join(future_hits),
            "called_deepseek": False,
            "read_api_key": False,
        }
    ]
    rows.extend({"scope": "block_build", **row} for row in hygiene_rows)
    return pd.DataFrame(rows)


def render_report(
    args: argparse.Namespace,
    notes: list[str],
    summary: pd.DataFrame,
    block_metrics: pd.DataFrame,
    panel_summary: pd.DataFrame,
    condition_summary: pd.DataFrame,
    hygiene: pd.DataFrame,
    paths: dict[str, Path],
) -> str:
    h2026 = block_metrics[block_metrics["target_block"].eq(FINAL_OOT)].copy() if "target_block" in block_metrics else pd.DataFrame()
    lines = [
        f"# P0 Small-Entry Case-Memory Guard Audit ({safe_prefix(args.output_prefix)})",
        "",
        "本报告专门审计 `branch_stack_v1.small_buy_hold` 小仓试探/持有分叉。实验完全本地运行，不调用 DeepSeek、不读取 key；RAG/case-memory 只允许作为复核或降权 guard，不作为正向买入引擎。",
        "",
        "## Setup",
        "",
        f"- frequencies: `{args.frequencies}`",
        f"- policy_name: `{args.policy_name}`",
        f"- operation_action: `{args.operation_action}`",
        f"- kline_feature_group: `{args.kline_feature_group}`",
        f"- top_k/min_applicable_conditions: `{args.top_k}/{args.min_applicable_conditions}`",
        "- H2026_1 是最终 OOT；未来收益只用于离线评估，不进入 evidence 或 preview。",
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
            markdown_table(
                summary,
                [
                    "frequency",
                    "guard_policy",
                    "prior_blocks",
                    "prior_retained_rows_mean",
                    "prior_delta_pos_hit",
                    "prior_delta_avg_hit",
                    "h2026_total_rows",
                    "h2026_retained_rows",
                    "h2026_retained_rate",
                    "h2026_retained_pos20",
                    "h2026_retained_avg20_pp",
                    "h2026_retained_loss_gt5",
                    "h2026_false_veto_positive_rows",
                    "h2026_captured_loss_gt5_rows",
                    "h2026_delta_pos",
                    "h2026_delta_avg",
                    "promotion_status",
                ],
            ),
            "",
            "## H2026 Block Metrics",
            "",
            markdown_table(
                h2026,
                [
                    "frequency",
                    "policy",
                    "total_active_rows",
                    "retained_rows",
                    "retained_rate",
                    "retained_pos20",
                    "retained_avg20_pp",
                    "retained_loss_gt5_rate",
                    "dropped_pos20",
                    "dropped_avg20_pp",
                    "captured_loss_gt5_rows",
                    "false_veto_positive_rows",
                    "delta_active_pos_vs_no_guard",
                    "delta_active_avg_vs_no_guard",
                    "promotion_status",
                ],
            ),
            "",
            "## H2026 100-Stock Panel Stability",
            "",
            markdown_table(
                panel_summary,
                [
                    "frequency",
                    "guard_policy",
                    "panels",
                    "retained_rows_mean",
                    "retained_rate_mean±std",
                    "retained_pos20_mean±std",
                    "retained_avg20_mean±std",
                    "retained_loss_gt5_mean±std",
                    "delta_pos_mean±std",
                    "delta_avg_mean±std",
                    "false_veto_positive_mean",
                    "captured_loss_gt5_mean",
                ],
            ),
            "",
            "## Matched Condition Diagnostics",
            "",
            markdown_table(condition_summary.head(30), ["condition", "rows", "unique_codes", "pos20", "avg20_pp", "loss_gt5_rate"]),
            "",
            "## Hygiene",
            "",
            markdown_table(
                hygiene.head(20),
                [
                    "scope",
                    "frequencies",
                    "policy_name",
                    "operation_action",
                    "detail_rows",
                    "block_metric_rows",
                    "safe_preview_rows",
                    "safe_preview_future_field_hits",
                    "called_deepseek",
                    "read_api_key",
                    "frequency",
                    "target_block",
                    "status",
                ],
            ),
            "",
            "## Decision Rules",
            "",
            f"- 晋级 DS smoke 至少需要 H2026 retained_rate >= 0.50、retained_pos20 >= 0.65、delta_pos >= 0.03、loss 不升、false_veto 可控，并且 prior_blocks >= {MIN_PROMOTION_PRIOR_BLOCKS}、prior_retained_rows_mean >= {MIN_PROMOTION_PRIOR_RETAINED_ROWS_MEAN}。",
            "- 如果 guard 只靠砍掉大量样本提升 retained 指标，必须按 false_veto_positive_rows 和 missed large gain 处罚，不能作为默认模块。",
            "- ML confirmer 当前只能作为灰色上下文；本报告不把 ML 分数作为硬过滤。",
            "- 若本地未出现 green/yellow，下一步仍可把高频条件作为 Agent 解释 checklist，但不应恢复 DS 后直接扩大 Flash/Pro。",
            "",
            "## Artifacts",
            "",
        ]
    )
    lines.extend([f"- `{path}`" for path in paths.values()])
    return "\n".join(lines) + "\n"


def mean_col(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return round(float(values.mean()), 6) if len(values) else np.nan


def hit_rate(frame: pd.DataFrame, column: str, threshold: float) -> float:
    if frame.empty or column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return round(float((values > threshold).mean()), 6) if len(values) else 0.0


def get_val(row: pd.Series, column: str) -> Any:
    if row.empty:
        return np.nan
    return row.get(column, np.nan)


def as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(number) else number


def fmt_mean_std(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame:
        return ""
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return ""
    return f"{values.mean():.4f}±{values.std(ddof=0):.4f}"


if __name__ == "__main__":
    main()
