"""Calibrate user-path active entry thresholds without DeepSeek calls.

The current product path is good at reducing weak-market damage, but H2026
active buy/add/hold decisions are below the acceptance target. This script
tests whether simple prior-only threshold changes can improve the active-entry
slice without pretending cash-adjusted stability is buy precision.

Future returns are used only for offline scoring. The optional preview contains
decision-time fields only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_user_capability_backtest import (  # noqa: E402
    DEFAULT_EXCLUDE_GLOBS,
    DEFAULT_FREQUENCIES,
    BacktestConfig,
    add_policy_scores,
    apply_frequency,
    bank_return_20d,
    build_holdout_panels,
    load_excluded_codes,
    load_feature_frame,
    parse_csv,
    safe_prefix,
    select_candidate_shortlist,
)
from src.agent_training.dual_mode_round import DEFAULT_JOINED_GT_CACHE_PATH  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_PREFIX = "userpath_active_entry_calibration_v1"
BLOCK_GROUPS = {
    "Y2023H2": ["H2023_2"],
    "Y2024": ["H2024_1", "H2024_2"],
    "Y2025": ["H2025_1", "H2025_2"],
    "H2026": ["H2026_1"],
}
PRIOR_PERIODS = {"Y2023H2", "Y2024", "Y2025"}
FINAL_PERIOD = "H2026"
ACTIVE_THRESHOLD = 0.35
FUTURE_KEYS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "gt_status",
    "label",
    "target",
    "outcome",
}


@dataclass(frozen=True)
class EntryPolicy:
    policy_id: str
    small_q: float
    small_p: float
    buy_q: float
    buy_p: float
    require_support_for_buy: bool
    require_peer_for_new_entry: bool
    sparse_cap: float
    overheat_cap: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit P0/P1 active-entry threshold calibration.")
    parser.add_argument("--joined-cache", type=Path, default=DEFAULT_JOINED_GT_CACHE_PATH)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--single-stock-count", type=int, default=100)
    parser.add_argument("--candidate-pool-size", type=int, default=200)
    parser.add_argument("--panels", type=int, default=3)
    parser.add_argument("--holdout-seed", default="user-capability-holdout-v1")
    parser.add_argument("--frequencies", default="weekly_friday,every_2_weeks,twice_weekly")
    parser.add_argument("--exclude-glob", action="append", default=list(DEFAULT_EXCLUDE_GLOBS))
    parser.add_argument("--max-variants", type=int, default=192)
    parser.add_argument(
        "--full-replay-variants",
        type=int,
        default=0,
        help="Optional slow previous-position replay for top variants. Default 0 uses static active-entry metrics.",
    )
    parser.add_argument("--preview-max-rows", type=int, default=360)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = safe_prefix(args.output_prefix)
    frequencies = parse_csv(args.frequencies) or DEFAULT_FREQUENCIES
    frame = add_policy_scores(load_feature_frame(args.joined_cache))
    excluded_codes, exclusion_summary = load_excluded_codes(args.exclude_glob, output_prefix=prefix)
    eligible_codes = sorted(set(frame["code"].dropna().astype(str)) - excluded_codes)
    if len(eligible_codes) < max(args.single_stock_count, args.candidate_pool_size):
        eligible_codes = sorted(frame["code"].dropna().astype(str).unique())
        holdout_definition = "hash_stable_codes_prior_ds_exclusion_insufficient"
    else:
        holdout_definition = "exclude_codes_seen_in_prior_sample_or_ds_artifacts"
    panels = build_holdout_panels(
        eligible_codes,
        panels=args.panels,
        single_stock_count=args.single_stock_count,
        candidate_pool_size=args.candidate_pool_size,
        seed=args.holdout_seed,
    )
    cfg = BacktestConfig(
        output_prefix=prefix,
        single_stock_count=args.single_stock_count,
        candidate_pool_size=args.candidate_pool_size,
        panels=args.panels,
        holdout_seed=args.holdout_seed,
        frequencies=frequencies,
    )
    variants = build_policy_grid(max_variants=args.max_variants)
    source_rows = build_source_rows(frame, panels, cfg)
    screen_metrics = screen_variants_static(source_rows, variants)
    screen_summary = summarize_metrics(screen_metrics)
    screen_ranking = rank_variants(screen_summary)
    replay_variants = select_replay_variants(screen_ranking, variants, max_variants=args.full_replay_variants)
    if replay_variants:
        metrics = evaluate_variants(source_rows, replay_variants)
        metric_mode = "stateful_previous_position_replay"
    else:
        selected_ids = set(screen_ranking["variant"].astype(str).head(max(1, min(len(variants), 48))).tolist()) if not screen_ranking.empty else {"base_v4_like"}
        selected_ids.add("base_v4_like")
        metrics = screen_metrics[screen_metrics["variant"].astype(str).isin(selected_ids)].copy()
        metric_mode = "static_active_entry_screen"
    summary = summarize_metrics(metrics)
    ranking = rank_variants(summary)
    preview = build_preview(source_rows, ranking, variants, max_rows=args.preview_max_rows)
    hygiene = build_hygiene(source_rows, preview, exclusion_summary, holdout_definition, variants, metric_mode=metric_mode)

    paths = {
        "metrics": REPORT_DIR / f"{prefix}_metrics.csv",
        "summary": REPORT_DIR / f"{prefix}_summary.csv",
        "ranking": REPORT_DIR / f"{prefix}_ranking.csv",
        "screen_metrics": REPORT_DIR / f"{prefix}_screen_metrics.csv",
        "screen_summary": REPORT_DIR / f"{prefix}_screen_summary.csv",
        "screen_ranking": REPORT_DIR / f"{prefix}_screen_ranking.csv",
        "preview": REPORT_DIR / f"{prefix}_agent_preview_no_gt.jsonl",
        "hygiene": REPORT_DIR / f"{prefix}_hygiene.csv",
        "report": REPORT_DIR / f"{prefix}.md",
    }
    metrics.to_csv(paths["metrics"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    ranking.to_csv(paths["ranking"], index=False, encoding="utf-8-sig")
    screen_metrics.to_csv(paths["screen_metrics"], index=False, encoding="utf-8-sig")
    screen_summary.to_csv(paths["screen_summary"], index=False, encoding="utf-8-sig")
    screen_ranking.to_csv(paths["screen_ranking"], index=False, encoding="utf-8-sig")
    hygiene.to_csv(paths["hygiene"], index=False, encoding="utf-8-sig")
    write_jsonl(paths["preview"], preview)
    paths["report"].write_text(
        render_report(args, paths, metrics, summary, ranking, hygiene, holdout_definition, metric_mode),
        encoding="utf-8",
    )
    print("A股研究Agent")
    print(f"source_rows={len(source_rows)} screened_variants={len(variants)} replay_variants={len(replay_variants)} metric_mode={metric_mode} metrics={len(metrics)}")
    print(f"report={paths['report']}")


def build_policy_grid(*, max_variants: int) -> list[EntryPolicy]:
    policies = [
        EntryPolicy(
            policy_id="base_v4_like",
            small_q=0.70,
            small_p=0.45,
            buy_q=0.90,
            buy_p=0.62,
            require_support_for_buy=False,
            require_peer_for_new_entry=False,
            sparse_cap=0.60,
            overheat_cap=0.60,
        )
    ]
    for small_q in [0.70, 0.78, 0.82, 0.86]:
        for small_p in [0.45, 0.54, 0.62]:
            for buy_q in [0.90, 0.94]:
                for buy_p in [0.62, 0.72]:
                    for require_support in [False, True]:
                        for require_peer in [False, True]:
                            for sparse_cap in [0.60, 0.35]:
                                policy_id = (
                                    f"sq{small_q:.2f}_sp{small_p:.2f}_bq{buy_q:.2f}_bp{buy_p:.2f}"
                                    f"_sup{int(require_support)}_peer{int(require_peer)}_scap{sparse_cap:.2f}"
                                )
                                policies.append(
                                    EntryPolicy(
                                        policy_id=policy_id,
                                        small_q=small_q,
                                        small_p=small_p,
                                        buy_q=buy_q,
                                        buy_p=buy_p,
                                        require_support_for_buy=require_support,
                                        require_peer_for_new_entry=require_peer,
                                        sparse_cap=sparse_cap,
                                        overheat_cap=0.35 if require_peer and sparse_cap <= 0.35 else 0.60,
                                    )
                                )
    dedup: dict[str, EntryPolicy] = {}
    for policy in policies:
        dedup.setdefault(policy.policy_id, policy)
    return list(dedup.values())[: max(1, max_variants)]


def build_source_rows(frame: pd.DataFrame, panels: list[dict[str, Any]], cfg: BacktestConfig) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for panel in panels:
        single = frame[frame["code"].isin(set(panel["single_codes"]))].copy()
        candidate = frame[frame["code"].isin(set(panel["candidate_codes"]))].copy()
        for period_name, blocks in BLOCK_GROUPS.items():
            single_period = single[single["valid_block"].isin(blocks)].copy()
            candidate_period = candidate[candidate["valid_block"].isin(blocks)].copy()
            for frequency in cfg.frequencies:
                scheduled_single = apply_frequency(single_period, frequency)
                if not scheduled_single.empty:
                    keep = decision_columns(scheduled_single)
                    one = scheduled_single[keep].copy()
                    one["task_mode"] = "single_stock_watch"
                    one["panel_id"] = panel["panel_id"]
                    one["period"] = period_name
                    one["decision_frequency"] = frequency
                    rows.append(one)
                scheduled_candidate = apply_frequency(candidate_period, frequency)
                if not scheduled_candidate.empty:
                    selected_rows: list[pd.DataFrame] = []
                    for date, day in scheduled_candidate.groupby("date", sort=True):
                        selected = select_candidate_shortlist(day, pool_size=cfg.candidate_pool_size)
                        if selected.empty:
                            continue
                        selected = selected.copy()
                        selected["date"] = date
                        selected_rows.append(selected)
                    if selected_rows:
                        selected_all = pd.concat(selected_rows, ignore_index=True)
                        keep = decision_columns(selected_all)
                        two = selected_all[keep].copy()
                        two["task_mode"] = "candidate_select_then_single_watch"
                        two["panel_id"] = panel["panel_id"]
                        two["period"] = period_name
                        two["decision_frequency"] = frequency
                        rows.append(two)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype(str)
    out["code"] = out["code"].astype(str).str.zfill(6)
    return out.sort_values(["task_mode", "panel_id", "period", "decision_frequency", "code", "date"]).reset_index(drop=True)


def decision_columns(frame: pd.DataFrame) -> list[str]:
    cols = [
        "date",
        "code",
        "name",
        "return_20d",
        "rev_chip_score_quantile",
        "agent_policy_score",
        "news_warning_score",
        "news_warning_score_30d",
        "news_opportunity_score",
        "news_opportunity_event_score_30d",
        "news_missing_rate",
        "financial_quality_risk_score",
        "financial_report_event_count",
        "financial_surprise_score",
        "financial_report_join_status",
        "tushare_industry_positive_breadth_20d",
        "peer_group_positive_breadth_20d",
        "tushare_industry_relative_return_20d",
        "peer_relative_to_group_20d",
        "kline_return_20d",
        "prior_return_20d",
        "kline_return_60d",
        "kline_rsi14",
        "rsi14",
        "upper_overhang",
        "lower_support",
        "tushare_industry",
    ]
    return [col for col in cols if col in frame.columns]


def screen_variants_static(source_rows: pd.DataFrame, variants: list[EntryPolicy]) -> pd.DataFrame:
    """Fast no-position-state screen; used only to choose full replay candidates."""
    rows: list[dict[str, Any]] = []
    if source_rows.empty:
        return pd.DataFrame()
    features = static_feature_frame(source_rows)
    base = source_rows[
        ["task_mode", "panel_id", "period", "decision_frequency", "date", "code", "return_20d"]
    ].copy()
    ret = pd.to_numeric(base["return_20d"], errors="coerce")
    group_cols = ["task_mode", "variant", "panel_id", "period", "decision_frequency"]
    for policy in variants:
        target, reason = static_targets(features, policy)
        scored = base.copy()
        scored["variant"] = policy.policy_id
        scored["target_position"] = target
        scored["operation_action"] = np.where(scored["target_position"] >= ACTIVE_THRESHOLD, "买入/持有", "等待不买")
        scored["strategy_return_20d"] = scored["target_position"] * ret + (1 - scored["target_position"]) * bank_return_20d()
        scored["hard_risk_count"] = features["hard_risk_count"]
        scored["support_count"] = features["support_count"]
        scored["reason_code"] = reason
        for keys, group in scored.groupby(group_cols, sort=True):
            task_mode, variant, panel_id, period, frequency = keys
            rows.append(metric_row(group, task_mode, variant, panel_id, period, frequency, policy))
    return pd.DataFrame(rows)


def static_feature_frame(source_rows: pd.DataFrame) -> pd.DataFrame:
    score_q = num_series(source_rows, "rev_chip_score_quantile", 0.5)
    score = num_series(source_rows, "agent_policy_score", 0.0)
    news_warning = np.maximum(
        num_series(source_rows, "news_warning_score", 0.0),
        num_series(source_rows, "news_warning_score_30d", 0.0),
    )
    news_opp = np.maximum(
        num_series(source_rows, "news_opportunity_score", 0.0),
        num_series(source_rows, "news_opportunity_event_score_30d", 0.0),
    )
    missing_news = num_series(source_rows, "news_missing_rate", 1.0)
    fin_risk = num_series(source_rows, "financial_quality_risk_score", 0.0)
    fin_count = num_series(source_rows, "financial_report_event_count", 0.0)
    fin_surprise = num_series(source_rows, "financial_surprise_score", 0.0)
    fin_status = source_rows.get("financial_report_join_status", pd.Series("", index=source_rows.index)).fillna("").astype(str)
    peer_breadth = num_series(source_rows, "tushare_industry_positive_breadth_20d", np.nan)
    peer_breadth = peer_breadth.fillna(num_series(source_rows, "peer_group_positive_breadth_20d", 0.5))
    peer_rel = num_series(source_rows, "tushare_industry_relative_return_20d", np.nan)
    peer_rel = peer_rel.fillna(num_series(source_rows, "peer_relative_to_group_20d", 0.0))
    prior20 = num_series(source_rows, "kline_return_20d", np.nan)
    prior20 = prior20.fillna(num_series(source_rows, "prior_return_20d", 0.0))
    prior60 = num_series(source_rows, "kline_return_60d", 0.0)
    rsi = num_series(source_rows, "kline_rsi14", np.nan)
    rsi = rsi.fillna(num_series(source_rows, "rsi14", 50.0))
    overhang = num_series(source_rows, "upper_overhang", 0.0)
    lower_support = num_series(source_rows, "lower_support", 0.0)

    hard_news = news_warning >= 0.67
    hard_fin = (fin_risk >= 0.70) | ((fin_count >= 1) & (fin_surprise <= -0.35))
    hard_peer = (peer_rel <= -4.0) & (peer_breadth <= 0.40)
    hard_overheat = ((prior20 >= 25) & (rsi >= 75) & (overhang >= 0.25)) | ((prior60 >= 45) & (rsi >= 78))
    hard_falling = (prior20 <= -18) & (peer_rel <= -3.0)
    hard_count = hard_news.astype(int) + hard_fin.astype(int) + hard_peer.astype(int) + hard_overheat.astype(int) + hard_falling.astype(int)
    support_news = (news_opp >= 0.33) & (news_warning < 0.5)
    support_fin = (fin_count >= 1) & (fin_risk <= 0.4) & (fin_surprise >= 0)
    support_peer = (peer_breadth >= 0.58) & (peer_rel >= 0)
    support_chip = lower_support >= 0.20
    support_count = support_news.astype(int) + support_fin.astype(int) + support_peer.astype(int) + support_chip.astype(int)
    return pd.DataFrame(
        {
            "score_q": score_q,
            "score": score,
            "missing_news": missing_news,
            "fin_status": fin_status,
            "peer_ok": (peer_breadth >= 0.50) | (peer_rel >= 0.0),
            "prior20": prior20,
            "rsi": rsi,
            "overhang": overhang,
            "hard_risk_count": hard_count,
            "hard_block": (hard_count >= 2) | ((hard_count >= 1) & (score_q < 0.88)),
            "support_count": support_count,
        },
        index=source_rows.index,
    )


def static_targets(features: pd.DataFrame, policy: EntryPolicy) -> tuple[pd.Series, pd.Series]:
    buy = (
        (features["score_q"] >= policy.buy_q)
        & (features["score"] >= policy.buy_p)
        & (~features["hard_block"])
    )
    if policy.require_support_for_buy:
        buy &= features["support_count"] >= 1
    if policy.require_peer_for_new_entry:
        buy &= features["peer_ok"]
    small = (
        (features["score_q"] >= policy.small_q)
        & (features["score"] >= policy.small_p)
        & (~features["hard_block"])
    )
    if policy.require_peer_for_new_entry:
        small &= features["peer_ok"]
    target = pd.Series(np.where(buy, 0.80, np.where(small, 0.35, 0.0)), index=features.index, dtype=float)
    reason = pd.Series(np.where(buy, "buy_threshold_pass", np.where(small, "small_entry_threshold_pass", "below_threshold")), index=features.index)
    sparse = (features["missing_news"] >= 0.95) & (features["fin_status"].isin(["", "no_event_in_window"])) & (target > policy.sparse_cap)
    target.loc[sparse] = policy.sparse_cap
    reason.loc[sparse] = reason.loc[sparse] + ";sparse_news_fin_cap"
    overheat = (features["prior20"] >= 20) & (features["rsi"] >= 70) & (features["overhang"] >= 0.20) & (target > policy.overheat_cap)
    target.loc[overheat] = policy.overheat_cap
    reason.loc[overheat] = reason.loc[overheat] + ";overheat_cap"
    return target.round(4), reason


def select_replay_variants(
    screen_ranking: pd.DataFrame,
    variants: list[EntryPolicy],
    *,
    max_variants: int,
) -> list[EntryPolicy]:
    if max_variants <= 0:
        return []
    by_id = {policy.policy_id: policy for policy in variants}
    selected: list[str] = ["base_v4_like"]
    if not screen_ranking.empty:
        prior_ok = screen_ranking[
            (screen_ranking["prior_active_count_mean"] >= 50)
            & (screen_ranking["h2026_active_count_mean"] >= 25)
        ].copy()
        for _, row in prior_ok.iterrows():
            variant = str(row.get("variant"))
            if variant not in selected:
                selected.append(variant)
            if len(selected) >= max_variants:
                break
    return [by_id[item] for item in selected if item in by_id]


def evaluate_variants(source_rows: pd.DataFrame, variants: list[EntryPolicy]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["task_mode", "variant", "panel_id", "period", "decision_frequency"]
    for policy in variants:
        scored = simulate_policy(source_rows, policy)
        for keys, group in scored.groupby(group_cols, sort=True):
            task_mode, variant, panel_id, period, frequency = keys
            rows.append(metric_row(group, task_mode, variant, panel_id, period, frequency, policy))
    return pd.DataFrame(rows)


def simulate_policy(source_rows: pd.DataFrame, policy: EntryPolicy) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    state_keys = ["task_mode", "panel_id", "period", "decision_frequency", "code"]
    ordered = source_rows.sort_values([*state_keys, "date"]).copy()
    for _, group in ordered.groupby(state_keys, sort=False):
        previous_position = 0.0
        for _, row in group.iterrows():
            decision = calibrated_operation_decision(row, policy, previous_position=previous_position)
            ret = safe_number(row.get("return_20d"), np.nan)
            target = decision["target_position"]
            strategy_return = target * ret + (1 - target) * bank_return_20d() if not math.isnan(ret) else np.nan
            rows.append(
                {
                    "task_mode": row["task_mode"],
                    "panel_id": row["panel_id"],
                    "period": row["period"],
                    "decision_frequency": row["decision_frequency"],
                    "date": row["date"],
                    "code": row["code"],
                    "variant": policy.policy_id,
                    "return_20d": ret,
                    "target_position": target,
                    "operation_action": action_from_position_change(previous_position, target),
                    "strategy_return_20d": strategy_return,
                    "hard_risk_count": decision["hard_risk_count"],
                    "support_count": decision["support_count"],
                    "reason_code": decision["reason_code"],
                }
            )
            previous_position = target
    return pd.DataFrame(rows)


def calibrated_operation_decision(row: pd.Series, policy: EntryPolicy, *, previous_position: float) -> dict[str, Any]:
    score_q = safe_number(row.get("rev_chip_score_quantile"), 0.5)
    score = safe_number(row.get("agent_policy_score"), 0.0)
    news_warning = max(safe_number(row.get("news_warning_score"), 0.0), safe_number(row.get("news_warning_score_30d"), 0.0))
    news_opp = max(safe_number(row.get("news_opportunity_score"), 0.0), safe_number(row.get("news_opportunity_event_score_30d"), 0.0))
    missing_news = safe_number(row.get("news_missing_rate"), 1.0)
    fin_risk = safe_number(row.get("financial_quality_risk_score"), 0.0)
    fin_count = safe_number(row.get("financial_report_event_count"), 0.0)
    fin_surprise = safe_number(row.get("financial_surprise_score"), 0.0)
    fin_status = str(row.get("financial_report_join_status") or "")
    peer_breadth = safe_number(row.get("tushare_industry_positive_breadth_20d"), safe_number(row.get("peer_group_positive_breadth_20d"), 0.5))
    peer_rel = safe_number(row.get("tushare_industry_relative_return_20d"), safe_number(row.get("peer_relative_to_group_20d"), 0.0))
    prior20 = safe_number(row.get("kline_return_20d"), safe_number(row.get("prior_return_20d"), 0.0))
    prior60 = safe_number(row.get("kline_return_60d"), 0.0)
    rsi = safe_number(row.get("kline_rsi14"), safe_number(row.get("rsi14"), 50.0))
    overhang = safe_number(row.get("upper_overhang"), 0.0)
    lower_support = safe_number(row.get("lower_support"), 0.0)

    hard_risks: list[str] = []
    supports: list[str] = []
    if news_warning >= 0.67:
        hard_risks.append("news_warning")
    if fin_risk >= 0.70 or (fin_count >= 1 and fin_surprise <= -0.35):
        hard_risks.append("financial_risk")
    if peer_rel <= -4.0 and peer_breadth <= 0.40:
        hard_risks.append("peer_weak")
    if (prior20 >= 25 and rsi >= 75 and overhang >= 0.25) or (prior60 >= 45 and rsi >= 78):
        hard_risks.append("overheat")
    if prior20 <= -18 and peer_rel <= -3.0:
        hard_risks.append("falling_with_peer_weakness")

    if news_opp >= 0.33 and news_warning < 0.5:
        supports.append("news_opportunity")
    if fin_count >= 1 and fin_risk <= 0.4 and fin_surprise >= 0:
        supports.append("financial_low_risk_event")
    if peer_breadth >= 0.58 and peer_rel >= 0:
        supports.append("peer_support")
    if lower_support >= 0.20:
        supports.append("chip_support")

    peer_ok = peer_breadth >= 0.50 or peer_rel >= 0.0
    target = 0.0
    reason = "below_threshold"
    if len(hard_risks) >= 2 or (hard_risks and score_q < 0.88):
        reason = "hard_risk_blocks"
    elif (
        score_q >= policy.buy_q
        and score >= policy.buy_p
        and (not policy.require_support_for_buy or len(supports) >= 1)
        and (not policy.require_peer_for_new_entry or peer_ok)
    ):
        target = 0.80
        reason = "buy_threshold_pass"
    elif score_q >= policy.small_q and score >= policy.small_p and not hard_risks and (not policy.require_peer_for_new_entry or peer_ok):
        target = 0.35
        reason = "small_entry_threshold_pass"
    elif previous_position > 0 and score_q >= 0.55 and not hard_risks:
        target = min(previous_position, 0.30)
        reason = "hold_existing_small"

    if missing_news >= 0.95 and fin_status in {"", "no_event_in_window"} and target > policy.sparse_cap:
        target = policy.sparse_cap
        reason += ";sparse_news_fin_cap"
    if (prior20 >= 20 and rsi >= 70 and overhang >= 0.20) and target > policy.overheat_cap:
        target = policy.overheat_cap
        reason += ";overheat_cap"
    return {
        "target_position": round(float(max(0.0, min(1.0, target))), 4),
        "hard_risk_count": len(hard_risks),
        "support_count": len(supports),
        "reason_code": reason,
    }


def metric_row(
    group: pd.DataFrame,
    task_mode: str,
    variant: str,
    panel_id: str,
    period: str,
    frequency: str,
    policy: EntryPolicy,
) -> dict[str, Any]:
    ret = pd.to_numeric(group["return_20d"], errors="coerce")
    strategy = pd.to_numeric(group["strategy_return_20d"], errors="coerce")
    pos = pd.to_numeric(group["target_position"], errors="coerce").fillna(0.0)
    active = pos >= ACTIVE_THRESHOLD
    active_strategy = strategy[active]
    active_hold = ret[active]
    return {
        "task_mode": task_mode,
        "variant": variant,
        "panel_id": panel_id,
        "period": period,
        "decision_frequency": frequency,
        "decision_count": int(len(group)),
        "unique_stocks": int(group["code"].nunique()),
        "decision_dates": int(group["date"].nunique()),
        "active_rate": rate(active),
        "active_count": int(active.sum()),
        "active_pos20": positive_rate(active_strategy),
        "active_avg20": mean(active_strategy),
        "active_hold_pos20": positive_rate(active_hold),
        "active_hold_avg20": mean(active_hold),
        "active_excess_vs_hold": diff_mean(active_strategy, active_hold),
        "strategy_pos20": positive_rate(strategy),
        "strategy_avg20": mean(strategy),
        "strategy_std20": std(strategy),
        "strategy_loss_gt5": rate(strategy <= -5),
        "hold_pos20": positive_rate(ret),
        "hold_avg20": mean(ret),
        "excess_vs_hold": diff_mean(strategy, ret),
        "avg_target_position": mean(pos),
        "small_q": policy.small_q,
        "small_p": policy.small_p,
        "buy_q": policy.buy_q,
        "buy_p": policy.buy_p,
        "require_support_for_buy": policy.require_support_for_buy,
        "require_peer_for_new_entry": policy.require_peer_for_new_entry,
        "sparse_cap": policy.sparse_cap,
        "overheat_cap": policy.overheat_cap,
    }


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    out = (
        metrics.groupby(["task_mode", "variant", "period", "decision_frequency"], dropna=False)
        .agg(
            panels=("panel_id", "nunique"),
            decision_count_mean=("decision_count", "mean"),
            active_rate_mean=("active_rate", "mean"),
            active_rate_std=("active_rate", "std"),
            active_count_mean=("active_count", "mean"),
            active_pos20_mean=("active_pos20", "mean"),
            active_pos20_std=("active_pos20", "std"),
            active_avg20_mean=("active_avg20", "mean"),
            active_avg20_std=("active_avg20", "std"),
            active_excess_vs_hold_mean=("active_excess_vs_hold", "mean"),
            strategy_pos20_mean=("strategy_pos20", "mean"),
            strategy_avg20_mean=("strategy_avg20", "mean"),
            strategy_loss_gt5_mean=("strategy_loss_gt5", "mean"),
            hold_pos20_mean=("hold_pos20", "mean"),
            hold_avg20_mean=("hold_avg20", "mean"),
            excess_vs_hold_mean=("excess_vs_hold", "mean"),
            avg_target_position=("avg_target_position", "mean"),
            small_q=("small_q", "first"),
            small_p=("small_p", "first"),
            buy_q=("buy_q", "first"),
            buy_p=("buy_p", "first"),
            require_support_for_buy=("require_support_for_buy", "first"),
            require_peer_for_new_entry=("require_peer_for_new_entry", "first"),
            sparse_cap=("sparse_cap", "first"),
            overheat_cap=("overheat_cap", "first"),
        )
        .reset_index()
    )
    return out.round(6)


def rank_variants(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["task_mode", "variant", "decision_frequency"]
    for key_values, group in summary.groupby(keys, sort=True):
        task_mode, variant, frequency = key_values
        prior = group[group["period"].isin(PRIOR_PERIODS)].copy()
        h2026 = group[group["period"].eq(FINAL_PERIOD)].copy()
        if prior.empty or h2026.empty:
            continue
        hrow = h2026.iloc[0]
        prior_active_pos = pd.to_numeric(prior["active_pos20_mean"], errors="coerce")
        prior_active_avg = pd.to_numeric(prior["active_avg20_mean"], errors="coerce")
        prior_active_count = pd.to_numeric(prior["active_count_mean"], errors="coerce")
        h_active_count = float(pd.to_numeric(pd.Series([hrow.get("active_count_mean")]), errors="coerce").iloc[0])
        h_active_pos = float(pd.to_numeric(pd.Series([hrow.get("active_pos20_mean")]), errors="coerce").iloc[0])
        h_active_avg = float(pd.to_numeric(pd.Series([hrow.get("active_avg20_mean")]), errors="coerce").iloc[0])
        prior_mean_pos = float(prior_active_pos.mean())
        prior_min_pos = float(prior_active_pos.min())
        prior_mean_avg = float(prior_active_avg.mean())
        if h_active_count < 25 or float(prior_active_count.mean()) < 50:
            verdict = "reject_sparse"
        elif prior_mean_pos >= 0.58 and h_active_pos >= 0.60 and h_active_avg > 0:
            verdict = "green_candidate"
        elif prior_mean_pos >= 0.55 and h_active_pos >= 0.50 and h_active_avg >= 0:
            verdict = "yellow_candidate"
        elif prior_mean_pos >= 0.55 and h_active_pos < 0.50:
            verdict = "prior_overfit_h2026_fail"
        else:
            verdict = "reference_only"
        rows.append(
            {
                "task_mode": task_mode,
                "variant": variant,
                "decision_frequency": frequency,
                "prior_active_pos20_mean": round(prior_mean_pos, 6),
                "prior_active_pos20_min": round(prior_min_pos, 6),
                "prior_active_avg20_mean": round(prior_mean_avg, 6),
                "prior_active_count_mean": round(float(prior_active_count.mean()), 3),
                "h2026_active_pos20": round(h_active_pos, 6),
                "h2026_active_avg20": round(h_active_avg, 6),
                "h2026_active_count_mean": round(h_active_count, 3),
                "h2026_strategy_avg20": hrow.get("strategy_avg20_mean"),
                "h2026_hold_avg20": hrow.get("hold_avg20_mean"),
                "h2026_excess_vs_hold": hrow.get("excess_vs_hold_mean"),
                "h2026_active_rate": hrow.get("active_rate_mean"),
                "promotion_status": verdict,
                "small_q": hrow.get("small_q"),
                "small_p": hrow.get("small_p"),
                "buy_q": hrow.get("buy_q"),
                "buy_p": hrow.get("buy_p"),
                "require_support_for_buy": hrow.get("require_support_for_buy"),
                "require_peer_for_new_entry": hrow.get("require_peer_for_new_entry"),
                "sparse_cap": hrow.get("sparse_cap"),
                "overheat_cap": hrow.get("overheat_cap"),
            }
        )
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking
    order = ["green_candidate", "yellow_candidate", "prior_overfit_h2026_fail", "reference_only", "reject_sparse"]
    ranking["status_rank"] = ranking["promotion_status"].map({name: i for i, name in enumerate(order)}).fillna(99)
    return ranking.sort_values(
        ["status_rank", "h2026_active_pos20", "h2026_active_avg20", "prior_active_pos20_mean"],
        ascending=[True, False, False, False],
    ).drop(columns=["status_rank"]).reset_index(drop=True)


def build_preview(
    source_rows: pd.DataFrame,
    ranking: pd.DataFrame,
    variants: list[EntryPolicy],
    *,
    max_rows: int,
) -> list[dict[str, Any]]:
    if ranking.empty or source_rows.empty or max_rows <= 0:
        return []
    policy_by_id = {policy.policy_id: policy for policy in variants}
    keep_variants = ranking[ranking["task_mode"].eq("single_stock_watch")].head(3)["variant"].astype(str).tolist()
    if "base_v4_like" not in keep_variants:
        keep_variants.append("base_v4_like")
    h2026 = source_rows[source_rows["period"].eq(FINAL_PERIOD)].copy()
    out: list[dict[str, Any]] = []
    for variant in keep_variants:
        policy = policy_by_id.get(variant)
        if not policy:
            continue
        simulated = simulate_policy(h2026, policy)
        active = simulated[pd.to_numeric(simulated["target_position"], errors="coerce").fillna(0) >= ACTIVE_THRESHOLD].head(max_rows // max(1, len(keep_variants))).copy()
        original = h2026.set_index(["task_mode", "panel_id", "period", "decision_frequency", "date", "code"])
        for _, row in active.iterrows():
            key = (row["task_mode"], row["panel_id"], row["period"], row["decision_frequency"], row["date"], row["code"])
            src = original.loc[key]
            if isinstance(src, pd.DataFrame):
                src = src.iloc[0]
            payload = {
                "variant": variant,
                "task_mode": row["task_mode"],
                "panel_id": row["panel_id"],
                "period": row["period"],
                "decision_frequency": row["decision_frequency"],
                "date": row["date"],
                "code": row["code"],
                "name": safe_text(src.get("name")),
                "target_position": row["target_position"],
                "operation_action": row["operation_action"],
                "reason_code": row["reason_code"],
                "rev_chip_score_quantile": safe_round(src.get("rev_chip_score_quantile")),
                "agent_policy_score": safe_round(src.get("agent_policy_score")),
                "news_warning_score": safe_round(src.get("news_warning_score")),
                "news_opportunity_score": safe_round(src.get("news_opportunity_score")),
                "financial_quality_risk_score": safe_round(src.get("financial_quality_risk_score")),
                "financial_report_join_status": safe_text(src.get("financial_report_join_status")),
                "peer_breadth_20d": safe_round(src.get("tushare_industry_positive_breadth_20d")),
                "peer_relative_20d": safe_round(src.get("tushare_industry_relative_return_20d")),
                "kline_rsi14": safe_round(src.get("kline_rsi14")),
                "upper_overhang": safe_round(src.get("upper_overhang")),
                "lower_support": safe_round(src.get("lower_support")),
            }
            assert_no_future_keys(payload)
            out.append(payload)
    return out[:max_rows]


def build_hygiene(
    source_rows: pd.DataFrame,
    preview: list[dict[str, Any]],
    exclusion_summary: pd.DataFrame,
    holdout_definition: str,
    variants: list[EntryPolicy],
    *,
    metric_mode: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": "source_rows",
                "status": "ok" if len(source_rows) else "empty",
                "value": len(source_rows),
                "detail": holdout_definition,
            },
            {
                "check": "variants",
                "status": "ok",
                "value": len(variants),
                "detail": f"prior-only threshold grid; no DeepSeek call; metric_mode={metric_mode}",
            },
            {
                "check": "preview_future_key_scan",
                "status": "ok" if not preview_future_key_count(preview) else "fail",
                "value": preview_future_key_count(preview),
                "detail": "exact key scan on agent preview",
            },
            {
                "check": "excluded_artifact_rows",
                "status": "ok",
                "value": len(exclusion_summary),
                "detail": "code exclusion sources counted without printing secrets",
            },
        ]
    )


def render_report(
    args: argparse.Namespace,
    paths: dict[str, Path],
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    ranking: pd.DataFrame,
    hygiene: pd.DataFrame,
    holdout_definition: str,
    metric_mode: str,
) -> str:
    p0_top = compact_ranking(ranking, "single_stock_watch")
    p1_top = compact_ranking(ranking, "candidate_select_then_single_watch")
    status_counts = ranking.groupby(["task_mode", "promotion_status"]).size().reset_index(name="variants") if not ranking.empty else pd.DataFrame()
    lines = [
        "# User-Path Active Entry Calibration v1",
        "",
        "本实验只做本地阈值/通道校准，不调用 DeepSeek，不改生产默认策略。未来 20 日收益只用于离线评分。",
        "",
        "## Run",
        "",
        f"- metric rows: `{len(metrics)}` from `{args.single_stock_count}` single-stock holdout and `{args.candidate_pool_size}` candidate pools x `{args.panels}` panels; exact source row count is in Hygiene",
        f"- holdout_definition: `{holdout_definition}`",
        f"- frequencies: `{args.frequencies}`",
        f"- max_variants: `{args.max_variants}`",
        f"- metric_mode: `{metric_mode}`",
        "- `static_active_entry_screen` 不使用前一仓位状态，专门评估 target_position>=35% 的入场阈值；若要验证完整持仓路径，应对少数候选再跑 `--full-replay-variants`。",
        "",
        "## Verdict",
        "",
        verdict_text(ranking),
        "",
        "## P0 Single-Stock Top Variants",
        "",
        markdown_table(p0_top),
        "",
        "## P1 Candidate-Then-Watch Top Variants",
        "",
        markdown_table(p1_top),
        "",
        "## Promotion Status Counts",
        "",
        markdown_table(status_counts),
        "",
        "## Hygiene",
        "",
        markdown_table(hygiene),
        "",
        "## Artifacts",
        "",
        *(f"- `{path}`" for path in paths.values()),
        "",
    ]
    return "\n".join(lines)


def verdict_text(ranking: pd.DataFrame) -> str:
    if ranking.empty:
        return "_empty ranking_"
    green = ranking[ranking["promotion_status"].eq("green_candidate")]
    yellow = ranking[ranking["promotion_status"].eq("yellow_candidate")]
    if not green.empty:
        row = green.iloc[0]
        return (
            f"- Found green candidate `{row['variant']}` for `{row['task_mode']}/{row['decision_frequency']}`: "
            f"H2026 active_pos={row['h2026_active_pos20']}, active_avg={row['h2026_active_avg20']}pp. "
            "It still needs DS semantic confirmation and channel ablation before default promotion."
        )
    if not yellow.empty:
        row = yellow.iloc[0]
        return (
            f"- Found yellow candidate `{row['variant']}` for `{row['task_mode']}/{row['decision_frequency']}`: "
            f"H2026 active_pos={row['h2026_active_pos20']}, active_avg={row['h2026_active_avg20']}pp. "
            "This is a diagnostic candidate, not a default upgrade."
        )
    p0 = ranking[ranking["task_mode"].eq("single_stock_watch")].head(1)
    if p0.empty:
        return "- No promotable active-entry variant found."
    row = p0.iloc[0]
    return (
        f"- No green/yellow active-entry variant found. Best P0 row is `{row['variant']}` "
        f"at `{row['decision_frequency']}` with H2026 active_pos={row['h2026_active_pos20']}, "
        f"active_avg={row['h2026_active_avg20']}pp and status `{row['promotion_status']}`. "
        "Do not tighten thresholds mechanically; next work should add stronger positive news/announcement/financial confirmation."
    )


def compact_ranking(ranking: pd.DataFrame, task_mode: str, max_rows: int = 16) -> pd.DataFrame:
    if ranking.empty:
        return pd.DataFrame()
    cols = [
        "task_mode",
        "variant",
        "decision_frequency",
        "promotion_status",
        "prior_active_pos20_mean",
        "prior_active_avg20_mean",
        "h2026_active_pos20",
        "h2026_active_avg20",
        "h2026_active_count_mean",
        "h2026_active_rate",
        "h2026_strategy_avg20",
        "h2026_hold_avg20",
        "require_support_for_buy",
        "require_peer_for_new_entry",
        "sparse_cap",
    ]
    return ranking[ranking["task_mode"].eq(task_mode)][cols].head(max_rows)


def action_from_position_change(previous: float, target: float) -> str:
    if target <= 0.0:
        return "卖出/不买" if previous > 0 else "等待不买"
    if previous <= 0.0:
        return "买入" if target >= ACTIVE_THRESHOLD else "小仓试探"
    delta = target - previous
    if delta >= 0.15:
        return "加仓"
    if delta <= -0.15:
        return "减仓"
    return "持有"


def rate(mask: Iterable[Any]) -> float:
    series = pd.Series(mask)
    if len(series) == 0:
        return np.nan
    return round(float(series.fillna(False).astype(bool).mean()), 6)


def positive_rate(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return round(float((values > 0).mean()), 6) if len(values) else np.nan


def mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return round(float(values.mean()), 6) if len(values) else np.nan


def std(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return round(float(values.std(ddof=0)), 6) if len(values) else np.nan


def diff_mean(left: pd.Series, right: pd.Series) -> float:
    left = pd.to_numeric(left, errors="coerce").dropna()
    right = pd.to_numeric(right, errors="coerce").dropna()
    return round(float(left.mean() - right.mean()), 6) if len(left) and len(right) else np.nan


def safe_number(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(number) else number


def num_series(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce")
    if not math.isnan(default):
        values = values.fillna(default)
    return values


def safe_round(value: Any, digits: int = 6) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return round(number, digits)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def preview_future_key_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows for key in row if key in FUTURE_KEYS)


def assert_no_future_keys(row: dict[str, Any]) -> None:
    leaks = sorted(set(row) & FUTURE_KEYS)
    if leaks:
        raise ValueError(f"preview contains future/result keys: {leaks}")


def markdown_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame.empty:
        return "_empty_"
    try:
        return frame.head(max_rows).to_markdown(index=False)
    except Exception:
        return frame.head(max_rows).to_csv(index=False)


if __name__ == "__main__":
    main()
