"""Audit a learned action/position layer for P0 single-stock watch.

This is a local, no-DeepSeek experiment. It reuses the existing opportunity,
K-line/peer/chip, and risk-review tools, then learns a small position mapping
from the validation block only. Forward returns are used only for offline
evaluation and never enter agent preview rows.
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
    FINAL_OOT,
    MIN_TARGET_ROWS,
    MIN_TRAIN_ROWS,
    MIN_VALID_ROWS,
    REPORT_DIR,
    TARGET_BLOCKS,
    apply_frequency,
    apply_policy,
    build_agent_preview_rows,
    build_scored_target,
    evaluate_policy,
    load_stack_frame,
    panel_stability,
    safe_float,
    safe_prefix,
    write_jsonl,
)
from scripts.run_lightweight_ml_channel_experiment import _rolling_split  # noqa: E402


DEFAULT_PREFIX = "p0_action_position_learner_v1"
DEFAULT_FREQUENCIES = "weekly_friday,every_2_weeks,weekly_tuesday"
DEFAULT_KLINE_GROUP = "kline_peer_chip_news_risk"
MAX_HGB_TRAIN_ROWS = 60000
PANEL_SIZE = 100
PANEL_SEEDS = 12
MIN_BIN_ROWS = 80

LEARNED_POLICIES = [
    "learned_precision_v1",
    "learned_balanced_v1",
    "learned_loss_guard_v1",
    "learned_delta_guard_v1",
]
BASELINE_POLICIES = [
    "opp_only",
    "opp_kline_confirm_no_raise",
    "branch_stack_v1",
    "bank_all_baseline",
    "hold_all_baseline",
]
PREVIEW_POLICIES = {"learned_balanced_v1", "learned_loss_guard_v1", "branch_stack_v1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit learned P0 action/position layer without DS/API calls.")
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--frequencies", default=DEFAULT_FREQUENCIES)
    parser.add_argument("--kline-feature-group", default=DEFAULT_KLINE_GROUP)
    parser.add_argument("--max-hgb-train-rows", type=int, default=MAX_HGB_TRAIN_ROWS)
    parser.add_argument("--panel-size", type=int, default=PANEL_SIZE)
    parser.add_argument("--panel-seeds", type=int, default=PANEL_SEEDS)
    parser.add_argument("--min-bin-rows", type=int, default=MIN_BIN_ROWS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frequencies = [item.strip() for item in args.frequencies.split(",") if item.strip()]
    frame, feature_groups, notes = load_stack_frame()
    metrics_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
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
            validation_scored = build_scored_target(
                train,
                validation,
                validation,
                feature_groups,
                kline_feature_group=args.kline_feature_group,
                max_hgb_train_rows=args.max_hgb_train_rows,
            )
            target_scored = build_scored_target(
                train,
                validation,
                target,
                feature_groups,
                kline_feature_group=args.kline_feature_group,
                max_hgb_train_rows=args.max_hgb_train_rows,
            )
            if validation_scored.empty or target_scored.empty:
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

            for policy_name in BASELINE_POLICIES:
                policy_frame = apply_policy(target_scored, policy_name)
                row = evaluate_policy(policy_frame, frequency=frequency, target_block=target_block, policy_name=policy_name)
                metrics_rows.append(row)
                if target_block == FINAL_OOT:
                    panel_rows.extend(panel_stability(policy_frame, row, panel_size=args.panel_size, panel_seeds=args.panel_seeds))
                    if policy_name in PREVIEW_POLICIES:
                        preview_rows.extend(build_agent_preview_rows(policy_frame, row, max_rows=400))

            for policy_name in LEARNED_POLICIES:
                profile = fit_position_profile(validation_scored, policy_name, min_bin_rows=args.min_bin_rows)
                learned = apply_position_profile(target_scored, profile, policy_name=policy_name)
                row = evaluate_policy(learned, frequency=frequency, target_block=target_block, policy_name=policy_name)
                metrics_rows.append(row)
                profile_rows.extend(profile_to_rows(profile, frequency=frequency, target_block=target_block, policy_name=policy_name))
                if target_block == FINAL_OOT:
                    panel_rows.extend(panel_stability(learned, row, panel_size=args.panel_size, panel_seeds=args.panel_seeds))
                    if policy_name in PREVIEW_POLICIES:
                        preview_rows.extend(build_agent_preview_rows(learned, row, max_rows=400))

    metrics = pd.DataFrame(metrics_rows)
    summary = summarize(metrics)
    panels = pd.DataFrame(panel_rows)
    panel_summary = summarize_panels(panels)
    profiles = pd.DataFrame(profile_rows)
    preview = pd.DataFrame(preview_rows)
    hygiene = pd.DataFrame(hygiene_rows)
    prefix = safe_prefix(args.output_prefix)
    paths = {
        "metrics": REPORT_DIR / f"{prefix}_metrics.csv",
        "summary": REPORT_DIR / f"{prefix}_summary.csv",
        "panels": REPORT_DIR / f"{prefix}_h2026_panel_stability.csv",
        "panel_summary": REPORT_DIR / f"{prefix}_h2026_panel_summary.csv",
        "profiles": REPORT_DIR / f"{prefix}_position_profiles.csv",
        "preview": REPORT_DIR / f"{prefix}_agent_preview.jsonl",
        "hygiene": REPORT_DIR / f"{prefix}_hygiene.csv",
        "report": REPORT_DIR / f"{prefix}.md",
    }
    metrics.to_csv(paths["metrics"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    panels.to_csv(paths["panels"], index=False, encoding="utf-8-sig")
    panel_summary.to_csv(paths["panel_summary"], index=False, encoding="utf-8-sig")
    profiles.to_csv(paths["profiles"], index=False, encoding="utf-8-sig")
    hygiene.to_csv(paths["hygiene"], index=False, encoding="utf-8-sig")
    write_jsonl(paths["preview"], preview)
    paths["report"].write_text(render_report(args, notes, summary, metrics, panel_summary, profiles, hygiene, paths), encoding="utf-8")
    print("A股研究Agent")
    print(f"rows={len(frame)} metrics={len(metrics)} summary={len(summary)} profiles={len(profiles)} preview={len(preview)}")
    print(f"report={paths['report']}")


def fit_position_profile(validation_scored: pd.DataFrame, policy_name: str, *, min_bin_rows: int = MIN_BIN_ROWS) -> dict[str, Any]:
    frame = add_action_bins(validation_scored)
    base_ret = pd.to_numeric(frame["return_20d"], errors="coerce")
    base = {
        "rows": int(len(frame)),
        "base_pos": float((base_ret > 0).mean()) if len(base_ret) else np.nan,
        "base_avg": float(base_ret.mean()) if len(base_ret) else np.nan,
        "base_loss": float((base_ret <= -5).mean()) if len(base_ret) else np.nan,
    }
    fallback = default_positions(policy_name)
    bins: dict[str, dict[str, Any]] = {}
    for action_bin, group in frame.groupby("action_bin", sort=True):
        stats = bin_stats(group, base)
        position = choose_position(stats, base, policy_name, min_bin_rows=min_bin_rows, fallback=fallback.get(action_bin, 0.05))
        bins[str(action_bin)] = {**stats, "position": position}
    return {
        "policy_name": policy_name,
        "min_bin_rows": int(min_bin_rows),
        "base": base,
        "fallback": fallback,
        "bins": bins,
    }


def apply_position_profile(scored: pd.DataFrame, profile: dict[str, Any], *, policy_name: str) -> pd.DataFrame:
    out = add_action_bins(scored).copy()
    fallback = profile.get("fallback") or {}
    bins = profile.get("bins") or {}
    positions = []
    for _, row in out.iterrows():
        action_bin = str(row.get("action_bin"))
        if action_bin in bins:
            pos = safe_float(bins[action_bin].get("position"))
        else:
            pos = safe_float(fallback.get(action_bin, 0.05))
        if bool(row.get("risk_queue_high_hard_counter")) or bool(row.get("kline_hard_risk")):
            pos = 0.0
        elif bool(row.get("risk_review_queue")):
            pos = min(pos, 0.20)
        positions.append(pos)
    out["target_position"] = pd.Series(positions, index=out.index).clip(lower=0.0, upper=0.70)
    out["policy_name"] = policy_name
    out["operation_hint"] = out.apply(learned_operation_hint, axis=1)
    ret = pd.to_numeric(out["return_20d"], errors="coerce")
    pos = pd.to_numeric(out["target_position"], errors="coerce").fillna(0.0)
    out["cash_adjusted_return_20d"] = pos * ret + (1.0 - pos) * 0.238095
    return out


def add_action_bins(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    hard = out.get("risk_queue_high_hard_counter", False).astype(bool) | out.get("kline_hard_risk", False).astype(bool)
    review = out.get("risk_review_queue", False).astype(bool)
    opp = out.get("opp_active", False).astype(bool)
    opp_strong = out.get("opp_strong", False).astype(bool)
    kline = out.get("kline_active", False).astype(bool)
    channel_hard = pd.to_numeric(out.get("channel_hard_counter_prob"), errors="coerce").fillna(0.0)
    channel_support = pd.to_numeric(out.get("channel_positive_support_prob"), errors="coerce").fillna(0.0)
    bins = np.select(
        [
            hard | channel_hard.ge(0.95),
            opp_strong & kline & ~review,
            opp & kline & ~review,
            opp & ~kline & channel_support.ge(0.45) & ~review,
            kline & ~opp & channel_support.ge(0.45) & ~review,
            opp & ~kline & ~review,
            kline & ~opp & ~review,
            (opp | kline) & review,
            review,
        ],
        [
            "hard_counter",
            "opp_strong_kline_clean",
            "opp_kline_clean",
            "opp_only_channel_support",
            "kline_only_channel_support",
            "opp_only_clean",
            "kline_only_clean",
            "confirm_with_review",
            "review_only",
        ],
        default="low_signal",
    )
    out["action_bin"] = pd.Series(bins, index=out.index).astype(str)
    return out


def bin_stats(group: pd.DataFrame, base: dict[str, float]) -> dict[str, Any]:
    ret = pd.to_numeric(group["return_20d"], errors="coerce").dropna()
    if ret.empty:
        return {
            "rows": 0,
            "pos_rate": np.nan,
            "avg_return": np.nan,
            "loss_rate": np.nan,
            "delta_pos": np.nan,
            "delta_avg": np.nan,
        }
    return {
        "rows": int(len(ret)),
        "pos_rate": round(float((ret > 0).mean()), 6),
        "avg_return": round(float(ret.mean()), 6),
        "loss_rate": round(float((ret <= -5).mean()), 6),
        "delta_pos": round(float((ret > 0).mean()) - safe_float(base.get("base_pos")), 6),
        "delta_avg": round(float(ret.mean()) - safe_float(base.get("base_avg")), 6),
    }


def choose_position(
    stats: dict[str, Any],
    base: dict[str, float],
    policy_name: str,
    *,
    min_bin_rows: int,
    fallback: float,
) -> float:
    rows = int(stats.get("rows") or 0)
    if rows < min_bin_rows:
        return float(fallback)
    pos = safe_float(stats.get("pos_rate"))
    avg = safe_float(stats.get("avg_return"))
    loss = safe_float(stats.get("loss_rate"))
    base_loss = safe_float(base.get("base_loss"))
    if policy_name == "learned_precision_v1":
        if pos >= 0.68 and avg >= 2.0 and loss <= max(0.22, base_loss):
            return 0.55
        if pos >= 0.60 and avg >= 1.0 and loss <= max(0.28, base_loss + 0.04):
            return 0.35
        if pos >= 0.53 and avg >= 0.0:
            return 0.20
        if avg < 0 or pos < 0.45:
            return 0.0
        return 0.10
    if policy_name == "learned_loss_guard_v1":
        if loss > max(0.30, base_loss + 0.06) or avg < -1.0:
            return 0.0
        if pos >= 0.65 and avg >= 1.5:
            return 0.50
        if pos >= 0.57 and avg >= 0.5:
            return 0.30
        if pos >= 0.50 and avg >= -0.25:
            return 0.15
        return 0.05
    if policy_name == "learned_delta_guard_v1":
        delta_pos = safe_float(stats.get("delta_pos"))
        delta_avg = safe_float(stats.get("delta_avg"))
        if delta_pos < 0 or delta_avg < 0:
            if loss > max(0.30, base_loss + 0.04) or avg < 0:
                return 0.0
            return 0.10
        if pos >= 0.72 and avg >= 2.0 and loss <= max(0.20, base_loss):
            return 0.55
        if pos >= 0.62 and avg >= 1.0 and loss <= max(0.25, base_loss + 0.03):
            return 0.40
        if pos >= 0.55 and avg >= 0.5:
            return 0.25
        return 0.10
    # learned_balanced_v1
    if pos >= 0.64 and avg >= 1.5 and loss <= max(0.30, base_loss + 0.05):
        return 0.55
    if pos >= 0.56 and avg >= 0.3:
        return 0.35
    if pos >= 0.48 and avg >= -0.5:
        return 0.20
    if avg < -1.0 or pos < 0.43:
        return 0.0
    return 0.10


def default_positions(policy_name: str) -> dict[str, float]:
    if policy_name == "learned_precision_v1":
        return {
            "hard_counter": 0.0,
            "opp_strong_kline_clean": 0.45,
            "opp_kline_clean": 0.35,
            "opp_only_channel_support": 0.20,
            "kline_only_channel_support": 0.20,
            "opp_only_clean": 0.15,
            "kline_only_clean": 0.15,
            "confirm_with_review": 0.05,
            "review_only": 0.0,
            "low_signal": 0.05,
        }
    if policy_name == "learned_loss_guard_v1":
        return {
            "hard_counter": 0.0,
            "opp_strong_kline_clean": 0.40,
            "opp_kline_clean": 0.30,
            "opp_only_channel_support": 0.20,
            "kline_only_channel_support": 0.20,
            "opp_only_clean": 0.10,
            "kline_only_clean": 0.10,
            "confirm_with_review": 0.05,
            "review_only": 0.0,
            "low_signal": 0.05,
        }
    if policy_name == "learned_delta_guard_v1":
        return {
            "hard_counter": 0.0,
            "opp_strong_kline_clean": 0.40,
            "opp_kline_clean": 0.35,
            "opp_only_channel_support": 0.20,
            "kline_only_channel_support": 0.10,
            "opp_only_clean": 0.25,
            "kline_only_clean": 0.10,
            "confirm_with_review": 0.05,
            "review_only": 0.0,
            "low_signal": 0.05,
        }
    return {
        "hard_counter": 0.0,
        "opp_strong_kline_clean": 0.55,
        "opp_kline_clean": 0.40,
        "opp_only_channel_support": 0.30,
        "kline_only_channel_support": 0.30,
        "opp_only_clean": 0.20,
        "kline_only_clean": 0.20,
        "confirm_with_review": 0.10,
        "review_only": 0.05,
        "low_signal": 0.10,
    }


def learned_operation_hint(row: pd.Series) -> str:
    pos = safe_float(row.get("target_position"))
    if pos <= 0:
        return "avoid_or_reduce_by_learned_position_tool"
    if pos >= 0.50:
        return "trial_buy_or_add_if_user_confirms_by_learned_tool"
    if pos >= ACTIVE_POSITION_THRESHOLD:
        return "small_buy_or_hold_by_learned_tool"
    if pos >= 0.15:
        return "hold_tiny_or_wait_for_confirmation_by_learned_tool"
    return "wait_or_hold_tiny_position_by_learned_tool"


def profile_to_rows(profile: dict[str, Any], *, frequency: str, target_block: str, policy_name: str) -> list[dict[str, Any]]:
    rows = []
    for action_bin, stats in sorted((profile.get("bins") or {}).items()):
        rows.append(
            {
                "frequency": frequency,
                "target_block": target_block,
                "policy_name": policy_name,
                "action_bin": action_bin,
                "rows": stats.get("rows"),
                "pos_rate": stats.get("pos_rate"),
                "avg_return": stats.get("avg_return"),
                "loss_rate": stats.get("loss_rate"),
                "delta_pos": stats.get("delta_pos"),
                "delta_avg": stats.get("delta_avg"),
                "learned_position": stats.get("position"),
            }
        )
    return rows


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in metrics.groupby(["frequency", "policy_name"], sort=True):
        h = group[group["target_block"].eq(FINAL_OOT)]
        prior = group[~group["target_block"].eq(FINAL_OOT)]
        hrow = h.iloc[0] if not h.empty else pd.Series(dtype=object)
        branch_ref = metrics[
            metrics["frequency"].eq(keys[0])
            & metrics["policy_name"].eq("branch_stack_v1")
            & metrics["target_block"].eq(FINAL_OOT)
        ]
        branch = branch_ref.iloc[0] if not branch_ref.empty else pd.Series(dtype=object)
        row = {
            "frequency": keys[0],
            "policy_name": keys[1],
            "prior_blocks": int(prior["target_block"].nunique()),
            "prior_strategy_avg_mean": mean(prior, "strategy_avg_return"),
            "prior_active_pos_mean": mean(prior, "active_pos_rate"),
            "prior_active_avg_mean": mean(prior, "active_avg_return"),
            "prior_active_avg_delta_hit_rate": hit_rate(prior, "active_delta_avg_vs_base", 0),
            "h2026_strategy_pos": val(hrow, "strategy_positive_rate"),
            "h2026_strategy_avg": val(hrow, "strategy_avg_return"),
            "h2026_excess_vs_hold": val(hrow, "excess_vs_hold_avg"),
            "h2026_avg_position": val(hrow, "avg_target_position"),
            "h2026_active_rate": val(hrow, "active_rate"),
            "h2026_active_pos": val(hrow, "active_pos_rate"),
            "h2026_active_avg": val(hrow, "active_avg_return"),
            "h2026_active_loss": val(hrow, "active_loss_gt5_rate"),
            "delta_strategy_avg_vs_branch": val(hrow, "strategy_avg_return") - val(branch, "strategy_avg_return"),
            "delta_active_pos_vs_branch": val(hrow, "active_pos_rate") - val(branch, "active_pos_rate"),
            "delta_active_avg_vs_branch": val(hrow, "active_avg_return") - val(branch, "active_avg_return"),
        }
        row["promotion_status"] = promotion_status(row)
        row["rank_score"] = rank_score(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["promotion_status", "rank_score"], ascending=[True, False])


def promotion_status(row: dict[str, Any]) -> str:
    policy = str(row.get("policy_name"))
    if policy.endswith("_baseline"):
        return "baseline_gray_reference"
    h_active = safe_float(row.get("h2026_active_pos"))
    h_avg = safe_float(row.get("h2026_active_avg"))
    h_rate = safe_float(row.get("h2026_active_rate"))
    prior_hit = safe_float(row.get("prior_active_avg_delta_hit_rate"))
    delta_branch_avg = safe_float(row.get("delta_active_avg_vs_branch"))
    delta_branch_strategy = safe_float(row.get("delta_strategy_avg_vs_branch"))
    if h_active >= 0.60 and h_avg > 0 and 0.03 <= h_rate <= 0.35 and prior_hit >= 0.75 and delta_branch_avg >= 0:
        return "green_candidate_for_flash_gate"
    if h_active >= 0.52 and h_avg > 0 and prior_hit >= 0.50 and (delta_branch_avg > 0 or delta_branch_strategy > 0):
        return "yellow_candidate_needs_fresh_panel"
    if h_avg > 0 and (delta_branch_avg > 0 or delta_branch_strategy > 0):
        return "observe_diagnostic_only"
    return "reject_or_reference_only"


def rank_score(row: dict[str, Any]) -> float:
    return (
        18 * safe_float(row.get("h2026_active_pos"))
        + safe_float(row.get("h2026_active_avg"))
        + 2 * safe_float(row.get("prior_active_avg_delta_hit_rate"))
        + safe_float(row.get("delta_active_avg_vs_branch"))
        + safe_float(row.get("delta_strategy_avg_vs_branch"))
        - 2 * max(0.0, safe_float(row.get("h2026_active_rate")) - 0.35)
    )


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
                "strategy_pos_mean": mean(group, "strategy_positive_rate"),
                "strategy_avg_mean": mean(group, "strategy_avg_return"),
                "active_pos_mean": mean(group, "active_pos_rate"),
                "active_pos_std": std(group, "active_pos_rate"),
                "active_avg_mean": mean(group, "active_avg_return"),
                "active_avg_std": std(group, "active_avg_return"),
                "active_loss_mean": mean(group, "active_loss_gt5_rate"),
            }
        )
    return pd.DataFrame(rows)


def render_report(
    args: argparse.Namespace,
    notes: list[str],
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    panel_summary: pd.DataFrame,
    profiles: pd.DataFrame,
    hygiene: pd.DataFrame,
    paths: dict[str, Path],
) -> str:
    h2026 = metrics[metrics["target_block"].eq(FINAL_OOT)].copy()
    lines = [
        "# P0 Action/Position Learner v1",
        "",
        "本实验是本地 walk-forward 审计，不调用 DeepSeek。目标是把 P0 工具层从“固定阈值 -> 固定仓位”推进到“validation 块学习动作/仓位映射”，再交给 Agent 审计。",
        "",
        "## Setup",
        "",
        f"- frequencies: `{args.frequencies}`",
        f"- kline_feature_group: `{args.kline_feature_group}`",
        f"- min_bin_rows: `{args.min_bin_rows}`",
        "- split: train = prior blocks before validation, validation = previous block learns position profile, target = current block.",
        "- future returns are used only for offline evaluation; agent preview is field-whitelisted.",
        "",
        "## Coverage Notes",
        "",
    ]
    lines.extend([f"- {note}" for note in notes[-10:]])
    lines.extend(
        [
            "",
            "## Summary",
            "",
            markdown_table(summary.head(30)),
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
                    ]
                ].sort_values(["active_pos_rate", "active_avg_return"], ascending=[False, False])
                if not h2026.empty
                else pd.DataFrame()
            ),
            "",
            "## H2026 Panel Summary",
            "",
            markdown_table(panel_summary),
            "",
            "## Learned Position Profiles",
            "",
            markdown_table(profiles.tail(80)),
            "",
            "## Hygiene",
            "",
            markdown_table(hygiene) if not hygiene.empty else "_empty_",
            "",
            "## Interpretation",
            "",
            "- `green_candidate_for_flash_gate` 才允许进入小规模 Flash Agent 行为验证。",
            "- `yellow_candidate_needs_fresh_panel` 只能说明本地工具层有候选方向，需要更多 fresh panel 或同样本 Agent 审计。",
            "- 如果 learned policy 只降低仓位并提高含现金指标，不能解释为主动买点能力。",
            "- 若 learned policy 低于 `branch_stack_v1`，应停止该仓位学习路线，转向更强输入特征或更好标签。",
            "",
            "## Artifacts",
            "",
        ]
    )
    lines.extend([f"- `{path}`" for path in paths.values()])
    return "\n".join(lines) + "\n"


def mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    return round(float(pd.to_numeric(frame[column], errors="coerce").mean()), 6)


def std(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    return round(float(pd.to_numeric(frame[column], errors="coerce").std(ddof=0)), 6)


def hit_rate(frame: pd.DataFrame, column: str, threshold: float) -> float:
    if frame.empty or column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return round(float((values > threshold).mean()), 6) if len(values) else 0.0


def val(row: pd.Series, column: str) -> float:
    if row.empty:
        return np.nan
    return round(safe_float(row.get(column)), 6)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    return frame.to_markdown(index=False)


if __name__ == "__main__":
    main()
