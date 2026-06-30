"""Audit non-price risk/support overlays for Agent decision policy.

This script is deliberately local and cheap: it uses existing cached features,
does not call DeepSeek, and does not request external data. Realized 20-day
returns are used only in offline reports. Agent-facing previews are prior-only
policy hints and must not expose returns, GT labels, or future outcomes.
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
OUTPUT_PREFIX = "nonprice_risk_overlay_v1"
DEFAULT_GT_SOURCES = [
    ROOT / "reports" / "backtest_scale_500" / "epoch1" / "ground_truth.csv",
    ROOT / "reports" / "backtest_scale_500" / "test" / "ground_truth.csv",
]

BLOCK_ORDER = list(TIME_BLOCKS.keys())
VALID_BLOCKS = BLOCK_ORDER[1:]
PULLBACK_20D_THRESHOLD = -10.1231
MIN_PRIOR_FLAGGED = 80
MIN_PRIOR_UNFLAGGED = 80
MIN_VALID_FLAGGED = 20

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
    "positive_20d_rate",
    "avg_return_20d",
    "loss_gt5_rate",
    "label",
    "outcome",
}


FLAG_DEFINITIONS: list[dict[str, Any]] = [
    {
        "flag_id": "news_high_warning_any",
        "flag_kind": "risk_or_friction",
        "description": "news is available and warning/risk score is high",
        "conditions": ["news_missing_rate < 0.80", "news_warning_score >= 0.65"],
    },
    {
        "flag_id": "news_high_warning_official",
        "flag_kind": "risk_or_friction",
        "description": "official/high-quality news warning is high",
        "conditions": ["news available", "news_warning_score >= 0.65", "official or evidence quality is usable"],
    },
    {
        "flag_id": "news_opportunity_with_warning",
        "flag_kind": "risk_or_friction",
        "description": "opportunity-tagged news also carries warning/conflict, a common false-veto zone",
        "conditions": ["news_opportunity_score >= 0.45", "news_warning_score >= 0.45"],
    },
    {
        "flag_id": "news_soft_gap_missing_or_low_quality",
        "flag_kind": "soft_gap",
        "description": "news coverage is missing or low-quality; confidence discount only unless validated as risk",
        "conditions": ["news_missing_rate >= 0.80 or evidence quality < 0.35"],
    },
    {
        "flag_id": "financial_high_risk_event",
        "flag_kind": "risk_or_friction",
        "description": "as-of financial event is matched and has high quality risk or negative surprise",
        "conditions": ["financial_report_join_status == event_window_matched", "risk high or surprise negative"],
    },
    {
        "flag_id": "financial_no_recent_event_soft_gap",
        "flag_kind": "soft_gap",
        "description": "no recent financial event in the as-of window; not equal to missing disclosure",
        "conditions": ["financial_report_join_status != event_window_matched"],
    },
    {
        "flag_id": "peer_industry_weak",
        "flag_kind": "risk_or_friction",
        "description": "industry-relative return and breadth are weak",
        "conditions": ["industry relative return < 0", "industry positive breadth < 0.45"],
    },
    {
        "flag_id": "peer_area_weak",
        "flag_kind": "risk_or_friction",
        "description": "same-area relative return and breadth are weak",
        "conditions": ["area relative return < 0", "area positive breadth < 0.45"],
    },
    {
        "flag_id": "bookskill_counter_high",
        "flag_kind": "risk_or_friction",
        "description": "BookSkill counter-score is high; use as review pressure, not a blind veto",
        "conditions": ["counter_score >= 8"],
    },
    {
        "flag_id": "bookskill_missing_soft_gap",
        "flag_kind": "soft_gap",
        "description": "no grounded BookSkill trigger is visible",
        "conditions": ["triggered_skills is empty"],
    },
    {
        "flag_id": "nonprice_hard_counter_min2",
        "flag_kind": "risk_or_friction",
        "description": "at least two hard non-price counters among news, financial, peer, and BookSkill",
        "conditions": ["sum(hard non-price counters) >= 2"],
    },
    {
        "flag_id": "nonprice_soft_gap_min2",
        "flag_kind": "soft_gap",
        "description": "at least two soft gaps among news, financial, peer, and BookSkill coverage",
        "conditions": ["sum(soft non-price gaps) >= 2"],
    },
    {
        "flag_id": "nonprice_support_min2",
        "flag_kind": "support",
        "description": "at least two target-specific support signals among news, financial, peer, and BookSkill",
        "conditions": ["sum(non-price support signals) >= 2"],
    },
]

SCOPE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "scope_id": "all_evaluated",
        "description": "all evaluated stock-date rows",
    },
    {
        "scope_id": "high_rev_chip",
        "description": "rev+chip_core score quantile >= 0.80",
    },
    {
        "scope_id": "pullback_high_rev_chip",
        "description": "high rev+chip_core plus 20d pullback",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit non-price risk/support overlays before DS spending.")
    parser.add_argument("--joined-cache", type=Path, default=DEFAULT_JOINED_GT_CACHE_PATH)
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--high-ranker-quantile", type=float, default=0.80)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_audit_frame(args.joined_cache, high_ranker_quantile=args.high_ranker_quantile)
    flagged = add_overlay_flags(frame)
    metrics = evaluate_policies(flagged)
    aggregate = aggregate_policies(metrics)
    coverage = coverage_table(flagged)
    previews = build_agent_previews(metrics)
    paths = write_outputs(
        prefix=args.output_prefix,
        flagged=flagged,
        metrics=metrics,
        aggregate=aggregate,
        coverage=coverage,
        previews=previews,
        high_ranker_quantile=args.high_ranker_quantile,
    )

    print("A股研究Agent")
    print(f"rows={len(flagged)}")
    print(f"policy_rows={len(metrics)}")
    print(f"preview_rows={len(previews)}")
    print(f"report={paths['report']}")


def load_audit_frame(path: Path, *, high_ranker_quantile: float) -> pd.DataFrame:
    if Path(path).resolve() == DEFAULT_JOINED_GT_CACHE_PATH.resolve():
        frame = load_ground_truth(
            DEFAULT_GT_SOURCES,
            kline_features_path=DEFAULT_KLINE_FEATURES_PATH,
            corr_peer_features_path=DEFAULT_CORR_PEER_FEATURES_PATH,
            tushare_peer_features_path=DEFAULT_TUSHARE_PEER_FEATURES_PATH,
            chip_core_features_path=DEFAULT_CHIP_CORE_FEATURES_PATH,
        )
    else:
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
        valid_block="nonprice_risk_overlay_audit",
        decision_frequency="every_2_weeks",
    )
    frame["rev_chip_score"] = pd.to_numeric(ranker["score"], errors="coerce")
    frame["rev_chip_score_quantile"] = pd.to_numeric(ranker["score_quantile"], errors="coerce")
    frame["high_ranker_cutoff"] = float(high_ranker_quantile)
    frame["time_block"] = frame["date"].map(block_for_date)
    frame = frame[frame["time_block"].isin(BLOCK_ORDER)].copy()
    frame["date_pool_return_20d"] = frame.groupby("date")["return_20d"].transform("mean")
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


def add_overlay_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    news_missing = num(out, "news_missing_rate", 1.0)
    news_warning = coalesce(out, ["news_warning_score", "news_risk_event_score_30d", "news_negative_materiality_30d"], 0.0)
    news_opportunity = coalesce(out, ["news_opportunity_score", "news_opportunity_event_score_30d", "news_positive_materiality_30d"], 0.0)
    news_quality = coalesce(out, ["news_evidence_quality", "news_evidence_quality_score_30d"], 0.0)
    official = num(out, "official_confirmation_score", 0.0)

    out["news_high_warning_any"] = news_missing.lt(0.80) & news_warning.ge(0.65)
    out["news_high_warning_official"] = out["news_high_warning_any"] & (official.ge(0.50) | news_quality.ge(0.60))
    out["news_opportunity_with_warning"] = news_missing.lt(0.80) & news_opportunity.ge(0.45) & news_warning.ge(0.45)
    out["news_soft_gap_missing_or_low_quality"] = news_missing.ge(0.80) | (news_missing.lt(0.80) & news_quality.lt(0.35))

    fin_status = out.get("financial_report_join_status", pd.Series("", index=out.index)).fillna("").astype(str)
    fin_matched = fin_status.eq("event_window_matched")
    fin_risk = num(out, "financial_quality_risk_score", 0.0)
    fin_surprise = num(out, "financial_surprise_score", 0.0)
    fin_disclosure = num(out, "financial_disclosure_quality_score", 0.0)
    fin_material = num(out, "financial_report_materiality_score", 0.0)
    out["financial_high_risk_event"] = fin_matched & (fin_risk.ge(0.60) | fin_surprise.le(-0.50))
    out["financial_no_recent_event_soft_gap"] = ~fin_matched
    out["financial_low_risk_support"] = fin_matched & fin_risk.le(0.25) & fin_disclosure.ge(0.80) & fin_material.ge(0.50)

    ind_rel = num(out, "tushare_industry_relative_return_20d", np.nan)
    ind_breadth = num(out, "tushare_industry_positive_breadth_20d", np.nan)
    area_rel = num(out, "tushare_area_relative_return_20d", np.nan)
    area_breadth = num(out, "tushare_area_positive_breadth_20d", np.nan)
    out["peer_industry_weak"] = ind_rel.lt(0.0) & ind_breadth.lt(0.45)
    out["peer_area_weak"] = area_rel.lt(0.0) & area_breadth.lt(0.45)
    out["peer_missing_soft_gap"] = ind_rel.isna() & area_rel.isna()
    out["peer_support"] = (ind_rel.ge(0.0) & ind_breadth.ge(0.60)) | (area_rel.ge(0.0) & area_breadth.ge(0.60))

    triggered = out.get("triggered_skills", pd.Series("", index=out.index)).fillna("").astype(str).str.strip()
    has_skill = triggered.ne("") & ~triggered.str.lower().isin({"nan", "none", "[]"})
    book_score = num(out, "book_score", 0.0)
    counter_score = num(out, "counter_score", 0.0)
    completeness = num(out, "completeness_score", 0.0)
    out["bookskill_counter_high"] = counter_score.ge(8.0)
    out["bookskill_missing_soft_gap"] = ~has_skill
    out["bookskill_support"] = has_skill & book_score.ge(6.0) & counter_score.le(6.0) & completeness.ge(7.5)

    hard_cols = ["news_high_warning_official", "financial_high_risk_event", "peer_industry_weak", "bookskill_counter_high"]
    soft_cols = ["news_soft_gap_missing_or_low_quality", "financial_no_recent_event_soft_gap", "peer_missing_soft_gap", "bookskill_missing_soft_gap"]
    support_cols = ["financial_low_risk_support", "peer_support", "bookskill_support"]
    out["nonprice_hard_counter_count"] = out[hard_cols].sum(axis=1).astype(int)
    out["nonprice_soft_gap_count"] = out[soft_cols].sum(axis=1).astype(int)
    out["nonprice_support_count"] = out[support_cols].sum(axis=1).astype(int)
    out["nonprice_hard_counter_min2"] = out["nonprice_hard_counter_count"].ge(2)
    out["nonprice_soft_gap_min2"] = out["nonprice_soft_gap_count"].ge(2)
    out["nonprice_support_min2"] = out["nonprice_support_count"].ge(2)

    out["scope_all_evaluated"] = True
    out["scope_high_rev_chip"] = num(out, "rev_chip_score_quantile", 0.0).ge(out["high_ranker_cutoff"])
    out["scope_pullback_high_rev_chip"] = out["scope_high_rev_chip"] & num(out, "kline_return_20d", 0.0).le(PULLBACK_20D_THRESHOLD)
    return out


def num(frame: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col not in frame:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def coalesce(frame: pd.DataFrame, cols: list[str], default: float) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    for col in cols:
        if col in frame:
            out = out.fillna(pd.to_numeric(frame[col], errors="coerce"))
    return out.fillna(default)


def evaluate_policies(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope in SCOPE_DEFINITIONS:
        scope_id = str(scope["scope_id"])
        scope_mask = frame[f"scope_{scope_id}"].fillna(False).astype(bool)
        scoped = frame[scope_mask].copy()
        for flag in FLAG_DEFINITIONS:
            flag_id = str(flag["flag_id"])
            for valid_block in VALID_BLOCKS:
                current_order = BLOCK_ORDER.index(valid_block)
                prior_blocks = BLOCK_ORDER[:current_order]
                prior = scoped[scoped["time_block"].isin(prior_blocks)].copy()
                current = scoped[scoped["time_block"].eq(valid_block)].copy()
                prior_metrics = compare_flag(prior, flag_id)
                current_metrics = compare_flag(current, flag_id)
                prior_policy = classify_prior_policy(prior_metrics, str(flag["flag_kind"]))
                rows.append(
                    {
                        "scope_id": scope_id,
                        "scope_description": scope["description"],
                        "flag_id": flag_id,
                        "flag_kind": flag["flag_kind"],
                        "flag_description": flag["description"],
                        "valid_block": valid_block,
                        "train_blocks": "+".join(prior_blocks),
                        "prior_policy_status": prior_policy,
                        "validation_agrees_with_prior": validation_agrees(prior_policy, current_metrics),
                        **{f"prior_{key}": value for key, value in prior_metrics.items()},
                        **{f"valid_{key}": value for key, value in current_metrics.items()},
                        "research_only": True,
                        "not_investment_instruction": True,
                    }
                )
    return pd.DataFrame(rows)


def compare_flag(frame: pd.DataFrame, flag_id: str) -> dict[str, Any]:
    if frame.empty or flag_id not in frame:
        return empty_metrics()
    flag = frame[flag_id].fillna(False).astype(bool)
    flagged = frame[flag].copy()
    unflagged = frame[~flag].copy()
    base = metric_block(frame)
    f = metric_block(flagged)
    u = metric_block(unflagged)
    return {
        "candidate_rows": int(len(frame)),
        "flagged_rows": int(len(flagged)),
        "unflagged_rows": int(len(unflagged)),
        "flagged_rate": safe_round(len(flagged) / max(1, len(frame))),
        "flagged_unique_stocks": int(flagged["code"].nunique()) if not flagged.empty else 0,
        "flagged_top_stock_concentration": concentration(flagged),
        "base_avg_return_20d": base["avg"],
        "base_positive_20d_rate": base["pos"],
        "base_loss_gt5_rate": base["loss"],
        "flagged_avg_return_20d": f["avg"],
        "flagged_positive_20d_rate": f["pos"],
        "flagged_loss_gt5_rate": f["loss"],
        "flagged_pool_excess_20d": f["excess"],
        "unflagged_avg_return_20d": u["avg"],
        "unflagged_positive_20d_rate": u["pos"],
        "unflagged_loss_gt5_rate": u["loss"],
        "unflagged_pool_excess_20d": u["excess"],
        "flag_vs_unflag_avg_delta": delta(f["avg"], u["avg"]),
        "flag_vs_unflag_pos_delta": delta(f["pos"], u["pos"]),
        "flag_vs_unflag_loss_delta": delta(f["loss"], u["loss"]),
        "unflagged_vs_base_avg_delta": delta(u["avg"], base["avg"]),
        "unflagged_vs_base_pos_delta": delta(u["pos"], base["pos"]),
        "unflagged_vs_base_loss_delta": delta(u["loss"], base["loss"]),
    }


def empty_metrics() -> dict[str, Any]:
    keys = [
        "candidate_rows",
        "flagged_rows",
        "unflagged_rows",
        "flagged_rate",
        "flagged_unique_stocks",
        "flagged_top_stock_concentration",
        "base_avg_return_20d",
        "base_positive_20d_rate",
        "base_loss_gt5_rate",
        "flagged_avg_return_20d",
        "flagged_positive_20d_rate",
        "flagged_loss_gt5_rate",
        "flagged_pool_excess_20d",
        "unflagged_avg_return_20d",
        "unflagged_positive_20d_rate",
        "unflagged_loss_gt5_rate",
        "unflagged_pool_excess_20d",
        "flag_vs_unflag_avg_delta",
        "flag_vs_unflag_pos_delta",
        "flag_vs_unflag_loss_delta",
        "unflagged_vs_base_avg_delta",
        "unflagged_vs_base_pos_delta",
        "unflagged_vs_base_loss_delta",
    ]
    return {key: np.nan for key in keys}


def metric_block(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {"avg": np.nan, "pos": np.nan, "loss": np.nan, "excess": np.nan}
    returns = pd.to_numeric(frame["return_20d"], errors="coerce")
    excess = pd.to_numeric(frame.get("pool_excess_20d"), errors="coerce")
    return {
        "avg": safe_round(returns.mean()),
        "pos": safe_round((returns > 0).mean()),
        "loss": safe_round((returns <= -5.0).mean()),
        "excess": safe_round(excess.mean()),
    }


def concentration(frame: pd.DataFrame) -> float:
    if frame.empty:
        return np.nan
    return safe_round(frame["code"].astype(str).value_counts(normalize=True).iloc[0])


def safe_round(value: Any, digits: int = 6) -> float:
    if pd.isna(value):
        return np.nan
    return round(float(value), digits)


def delta(left: Any, right: Any) -> float:
    if pd.isna(left) or pd.isna(right):
        return np.nan
    return safe_round(float(left) - float(right))


def classify_prior_policy(metrics: dict[str, Any], flag_kind: str) -> str:
    flagged = int(metrics.get("flagged_rows") or 0)
    unflagged = int(metrics.get("unflagged_rows") or 0)
    if flagged < MIN_PRIOR_FLAGGED or unflagged < MIN_PRIOR_UNFLAGGED:
        return "insufficient_prior_samples"
    avg_delta = metrics.get("flag_vs_unflag_avg_delta")
    pos_delta = metrics.get("flag_vs_unflag_pos_delta")
    loss_delta = metrics.get("flag_vs_unflag_loss_delta")
    if pd.isna(avg_delta) or pd.isna(pos_delta) or pd.isna(loss_delta):
        return "insufficient_prior_samples"
    avg_delta = float(avg_delta)
    pos_delta = float(pos_delta)
    loss_delta = float(loss_delta)
    if flag_kind == "support":
        if avg_delta >= 1.0 and pos_delta >= 0.03 and loss_delta <= 0.0:
            return "prior_positive_support_candidate"
        if avg_delta <= -1.0 or pos_delta <= -0.03:
            return "prior_support_failed_or_counter"
        return "prior_support_mixed_observe_only"
    if flag_kind == "soft_gap":
        if avg_delta <= -1.0 and loss_delta >= 0.03:
            return "prior_soft_gap_risk_discount"
        if avg_delta >= 1.0 and pos_delta >= 0.03:
            return "prior_soft_gap_not_negative_false_veto_risk"
        return "prior_soft_gap_neutral_discount_only"
    if avg_delta <= -1.0 and loss_delta >= 0.03:
        return "prior_risk_downweight_candidate"
    if avg_delta >= 1.0 and pos_delta >= 0.03:
        return "prior_false_veto_guard_candidate"
    return "prior_mixed_agent_judgment"


def validation_agrees(policy: str, current: dict[str, Any]) -> bool | None:
    if int(current.get("flagged_rows") or 0) < MIN_VALID_FLAGGED:
        return None
    avg_delta = current.get("flag_vs_unflag_avg_delta")
    pos_delta = current.get("flag_vs_unflag_pos_delta")
    loss_delta = current.get("flag_vs_unflag_loss_delta")
    if pd.isna(avg_delta) or pd.isna(pos_delta) or pd.isna(loss_delta):
        return None
    avg_delta = float(avg_delta)
    pos_delta = float(pos_delta)
    loss_delta = float(loss_delta)
    if policy in {"prior_risk_downweight_candidate", "prior_soft_gap_risk_discount", "prior_support_failed_or_counter"}:
        return avg_delta <= 0.0 and loss_delta >= 0.0
    if policy in {"prior_false_veto_guard_candidate", "prior_soft_gap_not_negative_false_veto_risk", "prior_positive_support_candidate"}:
        return avg_delta >= 0.0 or pos_delta >= 0.0
    return None


def aggregate_policies(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame()
    for (scope_id, flag_id), group in metrics.groupby(["scope_id", "flag_id"], sort=True):
        h2026 = group[group["valid_block"].eq("H2026_1")]
        latest = h2026.iloc[0] if not h2026.empty else group.iloc[-1]
        policy_counts = group["prior_policy_status"].value_counts().to_dict()
        agree = group["validation_agrees_with_prior"].dropna()
        rows.append(
            {
                "scope_id": scope_id,
                "flag_id": flag_id,
                "flag_kind": latest["flag_kind"],
                "flag_description": latest["flag_description"],
                "latest_prior_policy_status": latest["prior_policy_status"],
                "latest_validation_agrees": latest["validation_agrees_with_prior"],
                "policy_counts": json.dumps(policy_counts, ensure_ascii=False, sort_keys=True),
                "agreement_rate_when_checkable": safe_round(agree.astype(bool).mean()) if not agree.empty else np.nan,
                "h2026_flagged_rows": int(latest["valid_flagged_rows"]) if not pd.isna(latest["valid_flagged_rows"]) else 0,
                "h2026_flag_vs_unflag_avg_delta": latest["valid_flag_vs_unflag_avg_delta"],
                "h2026_flag_vs_unflag_pos_delta": latest["valid_flag_vs_unflag_pos_delta"],
                "h2026_flag_vs_unflag_loss_delta": latest["valid_flag_vs_unflag_loss_delta"],
                "agent_policy_recommendation": agent_policy_recommendation(str(latest["prior_policy_status"])),
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return pd.DataFrame(rows).sort_values(["scope_id", "agent_policy_recommendation", "flag_id"])


def agent_policy_recommendation(policy: str) -> str:
    if policy in {"prior_risk_downweight_candidate", "prior_soft_gap_risk_discount", "prior_support_failed_or_counter"}:
        return "downweight_or_request_confirmation"
    if policy in {"prior_false_veto_guard_candidate", "prior_soft_gap_not_negative_false_veto_risk"}:
        return "do_not_mechanically_veto"
    if policy == "prior_positive_support_candidate":
        return "positive_support_but_requires_cross_channel_audit"
    if policy == "insufficient_prior_samples":
        return "insufficient_history"
    return "agent_judgment_only"


def coverage_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for flag in FLAG_DEFINITIONS:
        flag_id = str(flag["flag_id"])
        values = frame[flag_id].fillna(False).astype(bool)
        rows.append(
            {
                "flag_id": flag_id,
                "flag_kind": flag["flag_kind"],
                "coverage_rate": safe_round(values.mean()),
                "rows": int(values.sum()),
                "unique_stocks": int(frame.loc[values, "code"].nunique()) if values.any() else 0,
                "coverage_dates": int(frame.loc[values, "date"].nunique()) if values.any() else 0,
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return pd.DataFrame(rows).sort_values("coverage_rate", ascending=False)


def build_agent_previews(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    rule_lookup = {str(item["flag_id"]): item for item in FLAG_DEFINITIONS}
    for _, row in metrics.iterrows():
        flag = rule_lookup[str(row["flag_id"])]
        policy = str(row["prior_policy_status"])
        action = agent_policy_recommendation(policy)
        preview = sanitize_quant_tool_outcome(
            {
                "tool_id": f"nonprice_risk_overlay:{row['scope_id']}:{row['flag_id']}:{row['valid_block']}",
                "tool_version": "nonprice_risk_overlay_v1",
                "task_mode": "single_stock",
                "policy_profile": "nonprice_risk_overlay_prior_only",
                "policy_status": policy,
                "feature_group": str(row["scope_id"]),
                "selection_mode": str(row["flag_id"]),
                "risk_tier": action,
                "primary_risk_branch": str(row["flag_kind"]),
                "risk_branch_labels": [
                    str(row["flag_id"]),
                    str(row["flag_kind"]),
                    f"valid_block={row['valid_block']}",
                    f"scope={row['scope_id']}",
                ],
                "branch_policy": f"{policy}; agent_use={action}",
                "promotion_status": policy,
                "usable_in_agent_default": action in {"do_not_mechanically_veto", "downweight_or_request_confirmation"},
                "top_features": flag["conditions"],
                "description": flag["description"],
                "required_confirmation": [
                    "use_current_or_prior_data_only",
                    "treat_as_research_grade_context_not_trade_instruction",
                    "do_not_override_hard_source_or_data_missing_boundary",
                ],
                "known_false_veto_risk": failure_mode_for_policy(policy),
                "calibration_policy": (
                    f"prior_only_policy_for_{row['valid_block']}; "
                    f"train_blocks={row['train_blocks']}; "
                    "offline returns stay only in report"
                ),
                "action_hint": action,
                "counter_evidence": counter_evidence_for_policy(policy),
                "missing_flags": [] if policy != "insufficient_prior_samples" else ["insufficient_prior_samples"],
                "source_ref_ids": [
                    "data/date_generalization_cache/market_5000/joined_ground_truth_combined_news.csv",
                    "reports/date_generalization/nonprice_risk_overlay_v1.md",
                ],
                "train_valid_test_blocks": f"train={row['train_blocks']}; validation_block={row['valid_block']}; no future metrics exposed",
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
        assert_no_future_fields(preview)
        previews.append(preview)
    return previews


def failure_mode_for_policy(policy: str) -> str:
    if "false_veto" in policy:
        return "flag historically did not imply poor outcome; do not mechanically downgrade without explicit hard event"
    if "risk" in policy or "counter" in policy:
        return "flag historically carried downside pressure; still check reversible-friction exceptions"
    if "support" in policy:
        return "support context is not standalone alpha and must be checked against hard counters"
    return "mixed or sparse history; Agent must use judgment and report uncertainty"


def counter_evidence_for_policy(policy: str) -> list[str]:
    if "false_veto" in policy:
        return ["mechanical_veto_can_miss_reversal", "require_explicit_hard_event_before_downgrade"]
    if "risk" in policy or "counter" in policy:
        return ["risk_flag_needs_confirmation", "check_reversible_reversal_friction_exception"]
    if "support" in policy:
        return ["support_not_standalone_alpha", "cross_channel_audit_required"]
    if policy == "insufficient_prior_samples":
        return ["insufficient_prior_samples"]
    return ["mixed_prior_outcome", "agent_judgment_required"]


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
    metrics: pd.DataFrame,
    aggregate: pd.DataFrame,
    coverage: pd.DataFrame,
    previews: list[dict[str, Any]],
    high_ranker_quantile: float,
) -> dict[str, Path]:
    metrics_path = REPORT_DIR / f"{prefix}_policy_metrics.csv"
    aggregate_path = REPORT_DIR / f"{prefix}_aggregate.csv"
    coverage_path = REPORT_DIR / f"{prefix}_coverage.csv"
    preview_path = REPORT_DIR / f"{prefix}_agent_preview.jsonl"
    detail_path = REPORT_DIR / f"{prefix}_safe_flag_detail.csv"
    report_path = REPORT_DIR / f"{prefix}.md"

    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
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
        "scope_high_rev_chip",
        "scope_pullback_high_rev_chip",
        *[str(item["flag_id"]) for item in FLAG_DEFINITIONS],
        "nonprice_hard_counter_count",
        "nonprice_soft_gap_count",
        "nonprice_support_count",
    ]
    detail = flagged[[col for col in safe_cols if col in flagged]].copy()
    detail["research_only"] = True
    detail["not_investment_instruction"] = True
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    write_report(
        report_path,
        metrics=metrics,
        aggregate=aggregate,
        coverage=coverage,
        high_ranker_quantile=high_ranker_quantile,
    )
    return {
        "metrics": metrics_path,
        "aggregate": aggregate_path,
        "coverage": coverage_path,
        "agent_preview": preview_path,
        "safe_detail": detail_path,
        "report": report_path,
    }


def write_report(path: Path, *, metrics: pd.DataFrame, aggregate: pd.DataFrame, coverage: pd.DataFrame, high_ranker_quantile: float) -> None:
    h2026 = aggregate[aggregate["h2026_flagged_rows"].ge(MIN_VALID_FLAGGED)].copy() if not aggregate.empty else pd.DataFrame()
    lines = [
        "# Non-Price Risk Overlay Audit v1",
        "",
        "本报告只用于 A 股研究辅助，不构成投资建议，不自动交易，不接券商接口。本实验不调用 DeepSeek，不读取 API key/token。",
        "",
        "## Purpose",
        "",
        "上一轮非价格正向确认审计说明：新闻、财报、同行、BookSkill 还不能单独做正向 alpha。本轮转向 Agent 更擅长的任务：判断这些非价格信号到底应该降权、避免机械 veto，还是仅作为信息缺口。",
        "",
        "## Setup",
        "",
        f"- rows evaluated through existing joined cache; high-ranker cutoff: `rev+chip_core score_quantile >= {high_ranker_quantile:.2f}`",
        "- scopes: all evaluated rows, high rev+chip rows, high rev+chip + 20d pullback rows",
        "- split: each valid block uses only prior blocks to assign `prior_policy_status`; H2026_1 is validation only",
        "- Agent preview contains only prior policy status and rule conditions; return/GT/pool-excess fields are excluded",
        "",
        "## Aggregate",
        "",
        table(aggregate),
        "",
        "## H2026 Checkable Rows",
        "",
        table(h2026) if not h2026.empty else "H2026 可检查样本不足。",
        "",
        "## Coverage",
        "",
        table(coverage),
        "",
        "## Interpretation",
        "",
        "- `do_not_mechanically_veto` 是本轮最重要的 Agent 规则：历史上该类信号不一定差，不能因为出现风险/缺口就直接下调研究优先级。",
        "- `downweight_or_request_confirmation` 也不是交易指令，只表示需要更多确认或降低研究置信度。",
        "- 若同一规则在 H2026 与 prior 方向相反，只能作为 mixed judgment，不得升权。",
        "- 该审计补充的是决策工作流的冲突处理，不是新的独立 alpha。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无数据。"
    return frame.to_markdown(index=False)


if __name__ == "__main__":
    main()
