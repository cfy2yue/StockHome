"""Audit non-quant positive confirmation rules without DS calls.

This is an offline research audit. Future 20-day returns are used only for
post-decision evaluation reports. Agent-facing rule previews contain rule
conditions and status only, not realized returns or GT labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.dual_mode_round import (  # noqa: E402
    DEFAULT_CHIP_CORE_FEATURES_PATH,
    DEFAULT_CORR_PEER_FEATURES_PATH,
    DEFAULT_JOINED_GT_CACHE_PATH,
    DEFAULT_KLINE_FEATURES_PATH,
    DEFAULT_PORTFOLIO_PRESET,
    DEFAULT_TUSHARE_PEER_FEATURES_PATH,
    TIME_BLOCKS,
    _portfolio_ranker_details,
    load_ground_truth,
)
from src.agent_training.quant_tool_context import sanitize_quant_tool_outcome  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
OUTPUT_PREFIX = "nonquant_positive_confirmation_audit_v1"
HIGH_RANKER_QUANTILE = 0.80
MIN_TOTAL_SELECTED = 30
MIN_H2026_SELECTED = 8
MAX_CONCENTRATION = 0.30
DEFAULT_GT_SOURCES = [
    ROOT / "reports" / "backtest_scale_500" / "epoch1" / "ground_truth.csv",
    ROOT / "reports" / "backtest_scale_500" / "test" / "ground_truth.csv",
]

BLOCK_ORDER = list(TIME_BLOCKS.keys())
VALID_BLOCKS = BLOCK_ORDER[1:]

FUTURE_FIELDS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "pool_excess_20d",
    "gt_status",
    "gt_pass",
    "rule_outcome_label",
}


RULE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "rule_id": "news_high_quality_positive_v1",
        "channel_group": "news",
        "required_flags": ["news_high_quality_positive"],
        "description": "target news is available, official/high quality, opportunity-tagged, and not high warning",
    },
    {
        "rule_id": "news_material_official_low_warning_v1",
        "channel_group": "news",
        "required_flags": ["news_material_official_low_warning"],
        "description": "official/material announcement/news with low warning score",
    },
    {
        "rule_id": "financial_event_quality_low_risk_v1",
        "channel_group": "financial",
        "required_flags": ["financial_event_quality_low_risk"],
        "description": "as-of financial event matched, material, good disclosure, low quality risk",
    },
    {
        "rule_id": "financial_surprise_low_risk_v1",
        "channel_group": "financial",
        "required_flags": ["financial_surprise_low_risk"],
        "description": "as-of financial event has positive surprise and low quality risk",
    },
    {
        "rule_id": "industry_peer_relative_support_v1",
        "channel_group": "peer",
        "required_flags": ["industry_peer_support"],
        "description": "target is not lagging industry and industry breadth is supportive",
    },
    {
        "rule_id": "area_peer_relative_support_v1",
        "channel_group": "peer",
        "required_flags": ["area_peer_support"],
        "description": "target is not lagging same-area peers and area breadth is supportive",
    },
    {
        "rule_id": "bookskill_support_low_counter_v1",
        "channel_group": "bookskill",
        "required_flags": ["bookskill_support_low_counter"],
        "description": "grounded BookSkill trigger has usable score and low counter-score",
    },
    {
        "rule_id": "news_plus_peer_support_v1",
        "channel_group": "news+peer",
        "required_flags": ["news_high_quality_positive", "industry_peer_support"],
        "description": "target-specific news confirmation plus industry-relative support",
    },
    {
        "rule_id": "financial_plus_peer_support_v1",
        "channel_group": "financial+peer",
        "required_flags": ["financial_event_quality_low_risk", "industry_peer_support"],
        "description": "as-of financial confirmation plus industry-relative support",
    },
    {
        "rule_id": "bookskill_plus_peer_support_v1",
        "channel_group": "bookskill+peer",
        "required_flags": ["bookskill_support_low_counter", "industry_peer_support"],
        "description": "BookSkill support plus industry-relative support",
    },
    {
        "rule_id": "news_financial_peer_support_v1",
        "channel_group": "news+financial+peer",
        "required_flags": ["news_material_official_low_warning", "financial_event_quality_low_risk", "industry_peer_support"],
        "description": "three-channel target/news, financial, and peer confirmation",
    },
    {
        "rule_id": "any_two_nonquant_confirmations_v1",
        "channel_group": "cross_channel",
        "min_confirmation_count": 2,
        "description": "at least two of news, financial, peer, BookSkill positive confirmations",
    },
    {
        "rule_id": "any_three_nonquant_confirmations_v1",
        "channel_group": "cross_channel",
        "min_confirmation_count": 3,
        "description": "at least three of news, financial, peer, BookSkill positive confirmations",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit non-quant positive confirmation rules.")
    parser.add_argument("--joined-cache", type=Path, default=DEFAULT_JOINED_GT_CACHE_PATH)
    parser.add_argument(
        "--ground-truth-sources",
        type=Path,
        nargs="*",
        default=DEFAULT_GT_SOURCES,
        help="Ground-truth source CSVs used to rebuild the default joined cache when needed.",
    )
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--high-ranker-quantile", type=float, default=HIGH_RANKER_QUANTILE)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_candidate_frame(
        args.joined_cache,
        ground_truth_sources=args.ground_truth_sources,
        high_ranker_quantile=args.high_ranker_quantile,
    )
    flagged = add_signal_flags(frame)
    rule_metrics = evaluate_rules(flagged)
    aggregate = aggregate_rule_metrics(rule_metrics)
    coverage = signal_coverage(flagged)
    report_ref = f"reports/date_generalization/{args.output_prefix}.md"
    previews = build_agent_rule_previews(aggregate, report_ref=report_ref)
    paths = write_outputs(
        prefix=args.output_prefix,
        flagged=flagged,
        rule_metrics=rule_metrics,
        aggregate=aggregate,
        coverage=coverage,
        previews=previews,
        high_ranker_quantile=args.high_ranker_quantile,
    )

    print("A股研究Agent")
    print(f"candidate_rows={len(flagged)}")
    print(f"rule_metrics={len(rule_metrics)}")
    print(f"report={paths['report']}")
    print(f"agent_rule_preview={paths['agent_rule_preview']}")


def load_candidate_frame(path: Path, *, ground_truth_sources: list[Path] | tuple[Path, ...] | None = None, high_ranker_quantile: float) -> pd.DataFrame:
    if Path(path).resolve() == DEFAULT_JOINED_GT_CACHE_PATH.resolve():
        frame = load_ground_truth(
            ground_truth_sources or DEFAULT_GT_SOURCES,
            kline_features_path=DEFAULT_KLINE_FEATURES_PATH,
            corr_peer_features_path=DEFAULT_CORR_PEER_FEATURES_PATH,
            tushare_peer_features_path=DEFAULT_TUSHARE_PEER_FEATURES_PATH,
            chip_core_features_path=DEFAULT_CHIP_CORE_FEATURES_PATH,
        )
    else:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    frame.columns = [col.lstrip("\ufeff") for col in frame.columns]
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    if "gt_status" in frame:
        frame = frame[frame["gt_status"].astype(str).eq("evaluated")].copy()
    frame["return_20d"] = pd.to_numeric(frame["return_20d"], errors="coerce")
    frame = frame.dropna(subset=["date", "code", "return_20d"]).copy()
    ranker = _portfolio_ranker_details(
        frame,
        preset=DEFAULT_PORTFOLIO_PRESET,
        valid_block="nonquant_positive_confirmation_audit",
        decision_frequency="every_2_weeks",
    )
    frame["rev_chip_score"] = pd.to_numeric(ranker["score"], errors="coerce")
    frame["rev_chip_score_quantile"] = pd.to_numeric(ranker["score_quantile"], errors="coerce")
    frame = frame[frame["rev_chip_score_quantile"].ge(high_ranker_quantile)].copy()
    frame["time_block"] = frame["date"].map(block_for_date)
    frame = frame[frame["time_block"].isin(BLOCK_ORDER)].copy()
    frame["date_pool_return_20d"] = frame.groupby(frame["date"].astype(str))["return_20d"].transform("mean")
    frame["pool_excess_20d"] = frame["return_20d"] - frame["date_pool_return_20d"]
    return frame.reset_index(drop=True)


def block_for_date(value: Any) -> str | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    for block, (start, end) in TIME_BLOCKS.items():
        if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
            return block
    return None


def add_signal_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    news_missing = num(out, "news_missing_rate", 1.0)
    news_available = news_missing < 0.75
    news_warning = num(out, "news_warning_score", 0.0)
    news_opp = num(out, "news_opportunity_score", 0.0)
    news_quality = num(out, "news_evidence_quality", 0.0)
    official = num(out, "official_confirmation_score", 0.0)
    announcement = num(out, "announcement_materiality_score", 0.0)

    out["news_high_quality_positive"] = news_available & news_opp.ge(0.30) & news_quality.ge(0.78) & official.ge(0.80) & news_warning.le(0.34)
    out["news_material_official_low_warning"] = news_available & announcement.ge(0.65) & official.ge(0.80) & news_warning.le(0.34)
    out["news_available_low_warning"] = news_available & news_warning.le(0.34)

    fin_status = out.get("financial_report_join_status", pd.Series("", index=out.index)).fillna("").astype(str)
    fin_matched = fin_status.eq("event_window_matched")
    fin_event = num(out, "financial_report_event_count", 0.0)
    fin_material = num(out, "financial_report_materiality_score", 0.0)
    fin_risk = num(out, "financial_quality_risk_score", 0.0)
    fin_surprise = num(out, "financial_surprise_score", 0.0)
    fin_disclosure = num(out, "financial_disclosure_quality_score", 0.0)
    out["financial_event_quality_low_risk"] = fin_matched & fin_event.ge(1.0) & fin_material.ge(0.65) & fin_risk.le(0.25) & fin_disclosure.ge(0.80)
    out["financial_surprise_low_risk"] = fin_matched & fin_surprise.gt(0.0) & fin_risk.le(0.25)

    ind_rel = num(out, "tushare_industry_relative_return_20d", np.nan)
    ind_breadth = num(out, "tushare_industry_positive_breadth_20d", np.nan)
    area_rel = num(out, "tushare_area_relative_return_20d", np.nan)
    area_breadth = num(out, "tushare_area_positive_breadth_20d", np.nan)
    out["industry_peer_support"] = ind_rel.ge(0.0) & ind_breadth.ge(0.60)
    out["area_peer_support"] = area_rel.ge(0.0) & area_breadth.ge(0.55)
    out["peer_support_any"] = out["industry_peer_support"] | out["area_peer_support"]

    triggered = out.get("triggered_skills", pd.Series("", index=out.index)).fillna("").astype(str)
    book_score = num(out, "book_score", 0.0)
    counter_score = num(out, "counter_score", 9.0)
    completeness = num(out, "completeness_score", 0.0)
    out["bookskill_support_low_counter"] = triggered.str.len().gt(0) & book_score.ge(6.0) & counter_score.le(6.0) & completeness.ge(7.5)
    out["bookskill_counter_high"] = counter_score.ge(8.0)

    confirmation_cols = [
        "news_high_quality_positive",
        "financial_event_quality_low_risk",
        "peer_support_any",
        "bookskill_support_low_counter",
    ]
    out["nonquant_confirmation_count"] = out[confirmation_cols].sum(axis=1).astype(int)
    out["soft_gap_bundle"] = (~out["news_available_low_warning"]) | (~fin_matched) | out["bookskill_counter_high"]
    return out


def num(frame: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col not in frame:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def evaluate_rules(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline_by_block = baseline_metrics(frame)
    for rule in RULE_DEFINITIONS:
        selected = select_rule_rows(frame, rule)
        for valid_block in VALID_BLOCKS:
            block_frame = frame[frame["time_block"].eq(valid_block)].copy()
            block_selected = selected[selected["time_block"].eq(valid_block)].copy()
            rows.append(evaluate_selection(rule, block_frame, block_selected, baseline_by_block.get(valid_block, {})))
    return pd.DataFrame(rows)


def baseline_metrics(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for block, group in frame.groupby("time_block"):
        returns = pd.to_numeric(group["return_20d"], errors="coerce")
        out[str(block)] = {
            "baseline_rows": float(len(group)),
            "baseline_positive_20d_rate": float((returns > 0).mean()) if len(returns) else np.nan,
            "baseline_loss_gt5_rate": float((returns <= -5.0).mean()) if len(returns) else np.nan,
            "baseline_avg_return_20d": float(returns.mean()) if len(returns) else np.nan,
        }
    return out


def select_rule_rows(frame: pd.DataFrame, rule: dict[str, Any]) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for flag in rule.get("required_flags", []):
        if flag not in frame:
            mask &= False
        else:
            mask &= frame[flag].fillna(False).astype(bool)
    min_count = int(rule.get("min_confirmation_count") or 0)
    if min_count:
        mask &= pd.to_numeric(frame.get("nonquant_confirmation_count", 0), errors="coerce").fillna(0).ge(min_count)
    return frame[mask].copy()


def evaluate_selection(rule: dict[str, Any], block_frame: pd.DataFrame, selected: pd.DataFrame, baseline: dict[str, float]) -> dict[str, Any]:
    returns = pd.to_numeric(selected.get("return_20d"), errors="coerce") if not selected.empty else pd.Series(dtype=float)
    excess = pd.to_numeric(selected.get("pool_excess_20d"), errors="coerce") if not selected.empty else pd.Series(dtype=float)
    concentration = selected["code"].astype(str).value_counts(normalize=True).max() if not selected.empty else np.nan
    base_pos = baseline.get("baseline_positive_20d_rate", np.nan)
    base_loss = baseline.get("baseline_loss_gt5_rate", np.nan)
    base_avg = baseline.get("baseline_avg_return_20d", np.nan)
    selected_pos = float((returns > 0).mean()) if len(returns) else np.nan
    selected_loss = float((returns <= -5.0).mean()) if len(returns) else np.nan
    selected_avg = float(returns.mean()) if len(returns) else np.nan
    return {
        "rule_id": rule["rule_id"],
        "channel_group": rule["channel_group"],
        "valid_block": str(block_frame["time_block"].iloc[0]) if not block_frame.empty else "",
        "candidate_rows": int(len(block_frame)),
        "selected_rows": int(len(selected)),
        "selected_rate": round(float(len(selected) / max(1, len(block_frame))), 6),
        "coverage_dates": int(selected["date"].nunique()) if not selected.empty else 0,
        "unique_stocks": int(selected["code"].nunique()) if not selected.empty else 0,
        "top_stock_concentration": round(float(concentration), 6) if not pd.isna(concentration) else np.nan,
        "baseline_positive_20d_rate": round(base_pos, 6) if not pd.isna(base_pos) else np.nan,
        "selected_positive_20d_rate": round(selected_pos, 6) if not pd.isna(selected_pos) else np.nan,
        "positive_rate_lift": round(float(selected_pos - base_pos), 6) if not pd.isna(selected_pos) and not pd.isna(base_pos) else np.nan,
        "baseline_loss_gt5_rate": round(base_loss, 6) if not pd.isna(base_loss) else np.nan,
        "selected_loss_gt5_rate": round(selected_loss, 6) if not pd.isna(selected_loss) else np.nan,
        "loss_gt5_lift": round(float(selected_loss - base_loss), 6) if not pd.isna(selected_loss) and not pd.isna(base_loss) else np.nan,
        "baseline_avg_return_20d": round(base_avg, 6) if not pd.isna(base_avg) else np.nan,
        "selected_avg_return_20d": round(selected_avg, 6) if not pd.isna(selected_avg) else np.nan,
        "selected_pool_excess_20d": round(float(excess.mean()), 6) if len(excess) else np.nan,
        "research_only": True,
        "not_investment_instruction": True,
    }


def aggregate_rule_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame()
    for rule_id, group in metrics.groupby("rule_id", sort=True):
        rule = next(item for item in RULE_DEFINITIONS if item["rule_id"] == rule_id)
        prior = group[~group["valid_block"].eq("H2026_1")]
        h2026 = group[group["valid_block"].eq("H2026_1")]
        rows.append(
            {
                "rule_id": rule_id,
                "channel_group": rule["channel_group"],
                "description": rule["description"],
                "blocks": int(group["valid_block"].nunique()),
                "prior_blocks": int(prior["valid_block"].nunique()),
                "total_selected_rows": int(group["selected_rows"].sum()),
                "h2026_selected_rows": int(h2026["selected_rows"].sum()) if not h2026.empty else 0,
                "mean_selected_rows": _mean(group, "selected_rows"),
                "prior_positive_rate_lift": _mean(prior, "positive_rate_lift"),
                "h2026_positive_rate_lift": _mean(h2026, "positive_rate_lift"),
                "prior_loss_gt5_lift": _mean(prior, "loss_gt5_lift"),
                "h2026_loss_gt5_lift": _mean(h2026, "loss_gt5_lift"),
                "prior_pool_excess_20d": _mean(prior, "selected_pool_excess_20d"),
                "h2026_pool_excess_20d": _mean(h2026, "selected_pool_excess_20d"),
                "prior_avg_return_20d": _mean(prior, "selected_avg_return_20d"),
                "h2026_avg_return_20d": _mean(h2026, "selected_avg_return_20d"),
                "max_top_stock_concentration": _max(group, "top_stock_concentration"),
                "min_coverage_dates": _min(group[group["selected_rows"].gt(0)], "coverage_dates"),
                "promotion_status": promotion_status(group),
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["promotion_status", "prior_pool_excess_20d", "h2026_pool_excess_20d"],
        ascending=[True, False, False],
    )


def _mean(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame:
        return np.nan
    values = pd.to_numeric(frame[col], errors="coerce")
    return round(float(values.mean()), 6) if values.notna().any() else np.nan


def _max(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame:
        return np.nan
    values = pd.to_numeric(frame[col], errors="coerce")
    return round(float(values.max()), 6) if values.notna().any() else np.nan


def _min(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame:
        return np.nan
    values = pd.to_numeric(frame[col], errors="coerce")
    return round(float(values.min()), 6) if values.notna().any() else np.nan


def promotion_status(group: pd.DataFrame) -> str:
    prior = group[~group["valid_block"].eq("H2026_1")]
    h2026 = group[group["valid_block"].eq("H2026_1")]
    total_rows = int(group["selected_rows"].sum())
    h_rows = int(h2026["selected_rows"].sum()) if not h2026.empty else 0
    concentration = _max(group, "top_stock_concentration")
    if total_rows < MIN_TOTAL_SELECTED or h_rows < MIN_H2026_SELECTED:
        return "reject_too_few_samples"
    if not pd.isna(concentration) and concentration > MAX_CONCENTRATION:
        return "reject_concentrated"
    prior_pos = _mean(prior, "positive_rate_lift")
    h_pos = _mean(h2026, "positive_rate_lift")
    prior_excess = _mean(prior, "selected_pool_excess_20d")
    h_excess = _mean(h2026, "selected_pool_excess_20d")
    prior_loss = _mean(prior, "loss_gt5_lift")
    h_loss = _mean(h2026, "loss_gt5_lift")
    if prior_pos >= 0.03 and h_pos >= 0.03 and prior_excess > 0 and h_excess > 0 and prior_loss <= 0.0 and h_loss <= 0.0:
        return "accepted_positive_confirmation_candidate"
    if prior_pos >= 0.03 and prior_excess > 0:
        return "observe_prior_positive_latest_weak"
    if prior_loss < 0 and h_loss < 0 and prior_excess >= -0.5 and h_excess >= -0.5:
        return "accepted_safety_hygiene_not_alpha"
    return "rejected_or_diagnostic_only"


def signal_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    flags = [
        "news_high_quality_positive",
        "news_material_official_low_warning",
        "news_available_low_warning",
        "financial_event_quality_low_risk",
        "financial_surprise_low_risk",
        "industry_peer_support",
        "area_peer_support",
        "peer_support_any",
        "bookskill_support_low_counter",
        "bookskill_counter_high",
    ]
    for flag in flags:
        if flag not in frame:
            continue
        values = frame[flag].fillna(False).astype(bool)
        rows.append(
            {
                "signal": flag,
                "coverage_rate": round(float(values.mean()), 6),
                "rows": int(values.sum()),
                "unique_stocks": int(frame.loc[values, "code"].nunique()) if values.any() else 0,
                "coverage_dates": int(frame.loc[values, "date"].nunique()) if values.any() else 0,
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return pd.DataFrame(rows).sort_values("coverage_rate", ascending=False)


def build_agent_rule_previews(
    aggregate: pd.DataFrame,
    *,
    report_ref: str = "reports/date_generalization/nonquant_positive_confirmation_audit_v1.md",
) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    if aggregate.empty:
        return previews
    rules_by_id = {rule["rule_id"]: rule for rule in RULE_DEFINITIONS}
    for _, row in aggregate.iterrows():
        rule = rules_by_id[str(row["rule_id"])]
        preview = sanitize_quant_tool_outcome(
            {
                "tool_id": f"nonquant_positive_confirmation:{rule['rule_id']}",
                "tool_version": "nonquant_positive_confirmation_audit_v1",
                "task_mode": "portfolio_pool",
                "feature_group": rule["channel_group"],
                "policy_status": str(row["promotion_status"]),
                "promotion_status": str(row["promotion_status"]),
                "usable_in_agent_default": str(row["promotion_status"]) == "accepted_positive_confirmation_candidate",
                "rule_conditions": rule.get("required_flags", []),
                "min_confirmation_count": int(rule.get("min_confirmation_count") or 0),
                "required_confirmation": [
                    "as_of_safe_current_or_prior_data_only",
                    "target_specific_nonquant_evidence",
                    "do_not_use_as_standalone_order_instruction",
                ],
                "action_hint": action_hint(str(row["promotion_status"])),
                "counter_evidence": counter_evidence(str(row["promotion_status"])),
                "description": rule["description"],
                "source_ref_ids": [
                    "data/date_generalization_cache/market_5000/joined_ground_truth_combined_news.csv",
                    report_ref,
                ],
                "train_valid_test_blocks": "walk-forward half-year blocks; metrics kept in offline report only",
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
        assert_no_future_fields(preview)
        previews.append(preview)
    return previews


def action_hint(status: str) -> str:
    if status == "accepted_positive_confirmation_candidate":
        return "may_act_as_one_positive_confirmation_but_requires_agent_cross_channel_audit"
    if status == "accepted_safety_hygiene_not_alpha":
        return "use_as_hygiene_or_uncertainty_context_not_positive_alpha"
    if status.startswith("observe"):
        return "observe_only_require_fresh_panel_or_ds_ablation_before_use"
    return "do_not_use_for_positive_confirmation"


def counter_evidence(status: str) -> list[str]:
    if status == "accepted_positive_confirmation_candidate":
        return ["not_standalone_alpha", "requires_bad_observe_and_missed_positive_monitoring"]
    if status == "accepted_safety_hygiene_not_alpha":
        return ["safety_hygiene_only", "not_positive_alpha"]
    return ["insufficient_or_unstable_lift", "do_not_raise_research_weight_from_this_rule"]


def assert_no_future_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FUTURE_FIELDS:
                raise ValueError(f"future field leaked to preview: {key}")
            assert_no_future_fields(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_future_fields(item)


def write_outputs(
    *,
    prefix: str,
    flagged: pd.DataFrame,
    rule_metrics: pd.DataFrame,
    aggregate: pd.DataFrame,
    coverage: pd.DataFrame,
    previews: list[dict[str, Any]],
    high_ranker_quantile: float,
) -> dict[str, Path]:
    aggregate_path = REPORT_DIR / f"{prefix}_aggregate.csv"
    metrics_path = REPORT_DIR / f"{prefix}_step_metrics.csv"
    coverage_path = REPORT_DIR / f"{prefix}_coverage.csv"
    preview_path = REPORT_DIR / f"{prefix}_agent_rule_preview.jsonl"
    detail_path = REPORT_DIR / f"{prefix}_safe_signal_detail.csv"
    report_path = REPORT_DIR / f"{prefix}.md"

    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    rule_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    with preview_path.open("w", encoding="utf-8") as handle:
        for item in previews:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    safe_cols = [
        "date",
        "code",
        "name",
        "time_block",
        "rev_chip_score_quantile",
        "news_high_quality_positive",
        "news_material_official_low_warning",
        "financial_event_quality_low_risk",
        "financial_surprise_low_risk",
        "industry_peer_support",
        "area_peer_support",
        "bookskill_support_low_counter",
        "nonquant_confirmation_count",
        "soft_gap_bundle",
        "research_only",
        "not_investment_instruction",
    ]
    detail = flagged.copy()
    detail["research_only"] = True
    detail["not_investment_instruction"] = True
    detail[[col for col in safe_cols if col in detail]].to_csv(detail_path, index=False, encoding="utf-8-sig")
    write_report(report_path, aggregate, rule_metrics, coverage, high_ranker_quantile=high_ranker_quantile)
    return {
        "aggregate": aggregate_path,
        "metrics": metrics_path,
        "coverage": coverage_path,
        "agent_rule_preview": preview_path,
        "safe_signal_detail": detail_path,
        "report": report_path,
    }


def write_report(path: Path, aggregate: pd.DataFrame, metrics: pd.DataFrame, coverage: pd.DataFrame, *, high_ranker_quantile: float) -> None:
    lines = [
        "# Non-Quant Positive Confirmation Audit v1",
        "",
        "本报告只用于 A 股研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Setup",
        "",
        f"- candidate pool: `rev_plus_chip_core score_quantile >= {high_ranker_quantile:.2f}`",
        "- split: half-year walk-forward evaluation; H2026_1 is latest-block validation only",
        "- evaluated channels: news/announcement, financial as-of event, peer/area relative strength, grounded BookSkill score",
        "- no DeepSeek call; future 20d results are used only in this offline report",
        "- Agent-facing preview contains no return/GT/pool-excess fields",
        "",
        "## Aggregate",
        "",
        table(aggregate),
        "",
        "## Step Metrics",
        "",
        table(metrics),
        "",
        "## Signal Coverage",
        "",
        table(coverage),
        "",
        "## Interpretation",
        "",
        "- `accepted_positive_confirmation_candidate` 才允许进入下一轮 DS 小面板，且仍只能作为一条正向确认，不是投资指令。",
        "- `observe_prior_positive_latest_weak` 说明早期块有信号但最新块弱，不能用于当前默认升权。",
        "- `accepted_safety_hygiene_not_alpha` 只能帮助区分软缺口/反证，不是正向 alpha。",
        "- 若新闻或财报规则因样本少被拒绝，下一步应优先补数据覆盖，而不是继续调 prompt。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.to_markdown(index=False)


if __name__ == "__main__":
    main()
