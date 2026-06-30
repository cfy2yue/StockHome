"""Audit DeepSeek news-questionnaire branch labels and prior-only case context.

This script is intentionally offline and cheap: it does not call DeepSeek and
does not request external data. Realized future returns are used only in the
audit report/CSV metrics. Agent-facing previews contain branch labels, branch
policy, and prior-case identifiers only; they must not contain return/GT fields.
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

from src.agent_training.deepseek_runner import write_jsonl  # noqa: E402
from src.agent_training.dual_mode_round import (  # noqa: E402
    DEFAULT_PORTFOLIO_PRESET,
    TIME_BLOCKS,
    _portfolio_ranker_details,
)


REPORT_DIR = ROOT / "reports" / "date_generalization"
MARKET_CACHE = ROOT / "data" / "date_generalization_cache" / "market_5000"
DEFAULT_QUESTIONNAIRE = MARKET_CACHE / "news_questionnaire_features.csv.gz"
DEFAULT_JOINED_GT = MARKET_CACHE / "joined_ground_truth_combined_news.csv"
OUTPUT_PREFIX = "news_questionnaire_branch_case_audit_v1"

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
    "target",
    "label",
    "outcome",
}

JOINED_COLUMNS = [
    "date",
    "code",
    "name",
    "gt_status",
    "return_20d",
    "prior_return_20d",
    "rsi14",
    "relative_strength_rank",
    "counter_score",
    "book_score",
    "completeness_score",
    "triggered_skills",
    "news_count_30d",
    "news_negative_materiality_30d",
    "news_warning_score",
    "news_opportunity_score",
    "official_confirmation_score",
    "announcement_materiality_score",
    "news_evidence_quality",
    "news_missing_rate",
    "event_count",
    "self_news_intensity",
    "news_event_table_join_status",
    "financial_report_event_count",
    "financial_quality_risk_score",
    "financial_surprise_score",
    "financial_disclosure_quality_score",
    "financial_report_join_status",
    "kline_return_20d",
    "kline_return_60d",
    "kline_drawdown_20d",
    "kline_drawdown_60d",
    "kline_rsi14",
    "corr_peer_avg_return_20d",
    "corr_peer_relative_return_20d",
    "corr_peer_positive_breadth_20d",
    "peer_group_positive_breadth_20d",
    "peer_relative_to_group_20d",
    "tushare_industry_relative_return_20d",
    "tushare_industry_positive_breadth_20d",
    "tushare_area_relative_return_20d",
    "tushare_area_positive_breadth_20d",
    "lower_support",
    "chip_concentration",
    "cost_band_width",
    "upper_overhang",
    "winner_rate_pct",
    "neg_winner_rate",
]

VECTOR_FIELDS = [
    "ds_news_risk_score",
    "ds_news_opportunity_score",
    "ds_news_uncertainty_score",
    "ds_news_quality_score",
    "ds_news_net_score",
    "ds_news_conflict_intensity",
    "ds_news_peer_risk_diffusion",
    "ds_news_peer_opportunity_diffusion",
    "ds_news_policy_tailwind",
    "ds_news_policy_headwind",
    "ds_news_region_policy_support",
    "ds_news_region_risk",
    "rev_chip_score_quantile",
    "prior_return_20d",
    "kline_return_20d",
    "kline_return_60d",
    "kline_drawdown_60d",
    "kline_rsi14",
    "lower_support",
    "upper_overhang",
    "peer_relative_to_group_20d",
    "tushare_industry_relative_return_20d",
]

BRANCH_ORDER = [
    "explicit_negative_event",
    "reversible_reversal_friction",
    "routine_official_low_signal",
    "soft_gap",
    "peer_diffusion",
    "policy_region_direct_support",
    "target_specific_positive_event",
    "unclassified_news_context",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit news questionnaire branch labels and prior-only case context.")
    parser.add_argument("--questionnaire-cache", type=Path, default=DEFAULT_QUESTIONNAIRE)
    parser.add_argument("--joined-gt", type=Path, default=DEFAULT_JOINED_GT)
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--max-preview-rows", type=int, default=300)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_audit_frame(args.questionnaire_cache, args.joined_gt)
    branched = add_branch_labels(frame)
    branch_metrics = evaluate_branch_metrics(branched)
    branch_aggregate = aggregate_branch_metrics(branch_metrics)
    prior_metrics = build_prior_policy_metrics(branched)
    similar_cases = build_prior_similar_cases(branched)
    previews = build_agent_previews(branched, prior_metrics, similar_cases, max_rows=args.max_preview_rows)
    paths = write_outputs(
        prefix=args.output_prefix,
        branched=branched,
        branch_metrics=branch_metrics,
        branch_aggregate=branch_aggregate,
        prior_metrics=prior_metrics,
        similar_cases=similar_cases,
        previews=previews,
    )

    print("A股研究Agent")
    print(f"questionnaire_rows={len(frame)}")
    print(f"branch_rows={len(branched)}")
    print(f"agent_preview_rows={len(previews)}")
    print(f"report={paths['report']}")
    print(f"agent_preview={paths['agent_preview']}")


def load_audit_frame(questionnaire_path: Path, joined_gt_path: Path) -> pd.DataFrame:
    if not questionnaire_path.exists():
        raise FileNotFoundError(questionnaire_path)
    if not joined_gt_path.exists():
        raise FileNotFoundError(joined_gt_path)

    q = pd.read_csv(questionnaire_path, dtype={"code": str}, low_memory=False)
    if q.empty:
        return q
    q["date"] = pd.to_datetime(q.get("decision_date", q.get("date")), errors="coerce").dt.date.astype(str)
    q["decision_date"] = q["date"]
    q["code"] = q["code"].astype(str).str.zfill(6)
    q = q.drop_duplicates(["date", "code"], keep="last").copy()

    header = pd.read_csv(joined_gt_path, nrows=0).columns.tolist()
    usecols = [col for col in JOINED_COLUMNS if col in header]
    joined = pd.read_csv(joined_gt_path, usecols=usecols, dtype={"code": str}, low_memory=False)
    joined["date"] = pd.to_datetime(joined["date"], errors="coerce").dt.date.astype(str)
    joined["code"] = joined["code"].astype(str).str.zfill(6)
    if "gt_status" in joined:
        joined = joined[joined["gt_status"].astype(str).eq("evaluated")].copy()

    ranker = _portfolio_ranker_details(
        joined,
        preset=DEFAULT_PORTFOLIO_PRESET,
        valid_block="news_questionnaire_branch_case_audit",
        decision_frequency="every_2_weeks",
    )
    joined["rev_chip_score"] = pd.to_numeric(ranker["score"], errors="coerce")
    joined["rev_chip_score_quantile"] = pd.to_numeric(ranker["score_quantile"], errors="coerce")
    joined["time_block"] = joined["date"].map(block_for_date)
    joined["date_pool_return_20d"] = joined.groupby("date")["return_20d"].transform(lambda item: pd.to_numeric(item, errors="coerce").mean())
    joined["pool_excess_20d"] = pd.to_numeric(joined["return_20d"], errors="coerce") - joined["date_pool_return_20d"]

    merged = q.merge(joined, on=["date", "code"], how="left", suffixes=("_questionnaire", ""))
    if "name" not in merged and "name_questionnaire" in merged:
        merged["name"] = merged["name_questionnaire"]
    merged["time_block"] = merged["time_block"].fillna(merged["date"].map(block_for_date))
    return merged.reset_index(drop=True)


def add_branch_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    rows = [assign_branch_tags(row) for _, row in out.iterrows()]
    out["news_branch_tags"] = [";".join(item["tags"]) for item in rows]
    out["primary_news_branch"] = [item["primary_branch"] for item in rows]
    out["branch_policy"] = [item["branch_policy"] for item in rows]
    out["branch_rationale"] = [item["branch_rationale"] for item in rows]
    out["research_only"] = True
    out["not_investment_instruction"] = True
    return out


def assign_branch_tags(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    tags: list[str] = []
    reasons: list[str] = []

    risk = f(row, "ds_news_risk_score")
    uncertainty = f(row, "ds_news_uncertainty_score")
    opportunity = f(row, "ds_news_opportunity_score")
    quality = f(row, "ds_news_quality_score")
    conflict = f(row, "ds_news_conflict_intensity")
    official = max_present(f(row, "ds_news_official_support"), f(row, "official_confirmation_score"))
    material = max_present(f(row, "ds_news_self_material_event"), f(row, "announcement_materiality_score"))
    negative_self = min_present(
        f(row, "ds_news_self_regulatory_legal"),
        f(row, "ds_news_self_holder_change"),
        f(row, "ds_news_self_capital_financing"),
        f(row, "ds_news_policy_headwind"),
    )
    warning = max_present(f(row, "news_warning_score"), f(row, "news_negative_materiality_30d"), f(row, "ds_news_region_risk"))
    rev_q = f(row, "rev_chip_score_quantile")
    prior_ret = max_present_abs_negative(f(row, "prior_return_20d"), f(row, "kline_return_20d"))
    drawdown = min_present(f(row, "kline_drawdown_60d"), f(row, "drawdown60"))
    rsi = max_present(f(row, "kline_rsi14"), f(row, "rsi14"))
    lower_support = f(row, "lower_support")
    upper_overhang = f(row, "upper_overhang")
    mainline = f(row, "ds_news_mainline_clarity")
    relevance = f(row, "ds_news_decision_relevance")
    repetition = f(row, "ds_news_repetition_lag")
    novelty = f(row, "ds_news_novelty")
    source_coverage = f(row, "ds_news_source_coverage")
    missing = max_present(f(row, "news_missing_rate"), 1.0 - source_coverage if not math.isnan(source_coverage) else np.nan)

    text = " ".join(
        str(row.get(key, "") if isinstance(row, dict) else row.get(key, ""))
        for key in ["ds_news_mainline_summary", "ds_news_missing_or_conflict_notes"]
    )

    explicit_negative = (
        risk >= 0.75
        and (
            negative_self <= -1.0
            or warning >= 0.65
            or contains_any(text, ["监管", "处罚", "立案", "诉讼", "质押", "退市", "st", "问询"])
        )
    ) or (warning >= 0.85 and risk >= 0.55)
    if explicit_negative:
        tags.append("explicit_negative_event")
        reasons.append("risk/warning and explicit negative event evidence are high")

    reversal_friction = (
        rev_q >= 0.80
        and (risk >= 0.50 or uncertainty >= 0.60 or conflict >= 0.45 or contains_any(text, ["异常波动", "问询", "风险提示", "下跌"]))
        and (
            prior_ret <= -5.0
            or drawdown <= -12.0
            or (not math.isnan(rsi) and rsi <= 42.0)
            or lower_support >= 0.55
        )
    )
    if reversal_friction:
        tags.append("reversible_reversal_friction")
        reasons.append("high rev+chip context with risk/uncertainty that may be reversal friction")

    routine_official = (
        official >= 0.75
        and opportunity <= 0.30
        and risk <= 0.35
        and (mainline <= 0.50 or relevance <= 0.50 or repetition >= 0.55 or novelty <= 0.30)
    )
    if routine_official:
        tags.append("routine_official_low_signal")
        reasons.append("official/routine information is visible but decision signal is low")

    soft_gap = (
        missing >= 0.75
        or source_coverage <= 0.35
        or (uncertainty >= 0.75 and risk < 0.75 and opportunity < 0.60)
    )
    if soft_gap and not explicit_negative:
        tags.append("soft_gap")
        reasons.append("news source coverage or confidence is weak without a confirmed hard negative")

    peer_diffusion = (
        abs_non_nan(f(row, "ds_news_peer_risk_diffusion")) >= 0.40
        or abs_non_nan(f(row, "ds_news_peer_opportunity_diffusion")) >= 0.40
        or f(row, "ds_news_peer_industry_heat") >= 0.55
        or f(row, "ds_news_cross_stock_confirmation") >= 0.45
    )
    if peer_diffusion:
        tags.append("peer_diffusion")
        reasons.append("news signal is materially tied to peer or cross-stock diffusion")

    policy_region_support = (
        f(row, "ds_news_policy_tailwind") >= 0.40
        or f(row, "ds_news_region_policy_support") >= 0.40
        or f(row, "ds_news_policy_support_score") >= 0.40
        or f(row, "ds_news_region_support_score") >= 0.40
    ) and f(row, "ds_news_policy_headwind") <= 0.30 and f(row, "ds_news_region_risk") <= 0.40
    if policy_region_support:
        tags.append("policy_region_direct_support")
        reasons.append("policy or region support is visible and not offset by headwind/risk")

    target_positive = (
        material >= 1.0
        and opportunity >= 0.60
        and quality >= 0.60
        and risk < 0.70
        and max_present(
            f(row, "ds_news_self_earnings_change"),
            f(row, "ds_news_self_order_product"),
            f(row, "ds_news_self_supply_chain_position"),
        )
        >= 0.30
    )
    if target_positive:
        tags.append("target_specific_positive_event")
        reasons.append("target-specific material positive event is visible")

    if not tags:
        tags.append("unclassified_news_context")
        reasons.append("questionnaire evidence does not meet any promoted branch condition")

    tags = [tag for tag in BRANCH_ORDER if tag in set(tags)]
    primary = tags[0]
    return {
        "tags": tags,
        "primary_branch": primary,
        "branch_policy": branch_policy(primary, tags, upper_overhang=upper_overhang),
        "branch_rationale": "; ".join(reasons)[:360],
    }


def branch_policy(primary: str, tags: list[str], *, upper_overhang: float) -> str:
    if primary == "explicit_negative_event":
        return "hard_review_counterevidence; do not raise research grade without independent remediation evidence"
    if primary == "reversible_reversal_friction":
        if "explicit_negative_event" in tags:
            return "conflicted_reversal_review; explicit negative must be resolved before any positive use"
        if upper_overhang >= 0.65:
            return "reversal_friction_observe; chip overhang requires extra confirmation"
        return "do_not_hard_veto_from_news_risk_alone; require cross-channel confirmation"
    if primary == "routine_official_low_signal":
        return "coverage_only_not_alpha; cap opportunity interpretation"
    if primary == "soft_gap":
        return "soft_gap_confidence_discount; not directional by itself"
    if primary == "peer_diffusion":
        return "peer_relative_check_required; separate target evidence from peer spillover"
    if primary == "policy_region_direct_support":
        return "policy_region_support_observe; require target and peer confirmation before upgrade"
    if primary == "target_specific_positive_event":
        return "positive_event_candidate_observe; still not standalone alpha"
    return "observe_only; insufficient structured news signal"


def evaluate_branch_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    data = frame.copy()
    data["return_20d"] = pd.to_numeric(data.get("return_20d"), errors="coerce")
    data["pool_excess_20d"] = pd.to_numeric(data.get("pool_excess_20d"), errors="coerce")
    for block, block_frame in data.groupby("time_block", dropna=False):
        base_ret = pd.to_numeric(block_frame["return_20d"], errors="coerce")
        base_pos = float((base_ret > 0).mean()) if base_ret.notna().any() else np.nan
        for branch in BRANCH_ORDER:
            selected = block_frame[block_frame["news_branch_tags"].fillna("").str.contains(branch, regex=False)].copy()
            rows.append(branch_metric_row(str(block), branch, selected, base_pos, scope="tag"))
        for branch in BRANCH_ORDER:
            selected = block_frame[block_frame["primary_news_branch"].eq(branch)].copy()
            rows.append(branch_metric_row(str(block), branch, selected, base_pos, scope="primary"))
    return pd.DataFrame(rows)


def branch_metric_row(block: str, branch: str, selected: pd.DataFrame, base_pos: float, *, scope: str) -> dict[str, Any]:
    returns = pd.to_numeric(selected.get("return_20d"), errors="coerce") if not selected.empty else pd.Series(dtype=float)
    excess = pd.to_numeric(selected.get("pool_excess_20d"), errors="coerce") if not selected.empty else pd.Series(dtype=float)
    pos = float((returns > 0).mean()) if returns.notna().any() else np.nan
    return {
        "scope": scope,
        "time_block": block,
        "branch": branch,
        "rows": int(len(selected)),
        "unique_stocks": int(selected["code"].nunique()) if not selected.empty and "code" in selected else 0,
        "positive_20d_rate": round(pos, 6) if not math.isnan(pos) else np.nan,
        "block_baseline_positive_20d_rate": round(base_pos, 6) if not math.isnan(base_pos) else np.nan,
        "positive_rate_lift": round(float(pos - base_pos), 6) if not math.isnan(pos) and not math.isnan(base_pos) else np.nan,
        "avg_return_20d": round(float(returns.mean()), 6) if returns.notna().any() else np.nan,
        "avg_pool_excess_20d": round(float(excess.mean()), 6) if excess.notna().any() else np.nan,
        "research_only": True,
        "not_investment_instruction": True,
    }


def aggregate_branch_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (scope, branch), group in metrics[metrics["rows"].gt(0)].groupby(["scope", "branch"], sort=True):
        for split_name, split in [
            ("prior_blocks", group[~group["time_block"].eq("H2026_1")]),
            ("H2026_1", group[group["time_block"].eq("H2026_1")]),
            ("all_blocks", group),
        ]:
            if split.empty:
                rows.append(
                    {
                        "scope": scope,
                        "branch": branch,
                        "split": split_name,
                        "rows": 0,
                        "positive_20d_rate": np.nan,
                        "positive_rate_lift": np.nan,
                        "avg_return_20d": np.nan,
                        "avg_pool_excess_20d": np.nan,
                        "research_only": True,
                        "not_investment_instruction": True,
                    }
                )
                continue
            weight = pd.to_numeric(split["rows"], errors="coerce").fillna(0)
            rows.append(
                {
                    "scope": scope,
                    "branch": branch,
                    "split": split_name,
                    "rows": int(weight.sum()),
                    "positive_20d_rate": weighted_mean(split, "positive_20d_rate", weight),
                    "positive_rate_lift": weighted_mean(split, "positive_rate_lift", weight),
                    "avg_return_20d": weighted_mean(split, "avg_return_20d", weight),
                    "avg_pool_excess_20d": weighted_mean(split, "avg_pool_excess_20d", weight),
                    "research_only": True,
                    "not_investment_instruction": True,
                }
            )
    return pd.DataFrame(rows).sort_values(["scope", "branch", "split"]).reset_index(drop=True)


def weighted_mean(frame: pd.DataFrame, col: str, weight: pd.Series) -> float:
    if frame.empty or col not in frame or weight.sum() <= 0:
        return np.nan
    values = pd.to_numeric(frame[col], errors="coerce")
    mask = values.notna() & weight.gt(0)
    if not mask.any():
        return np.nan
    return round(float((values[mask] * weight[mask]).sum() / weight[mask].sum()), 6)


def build_prior_policy_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    data = frame.copy()
    data["block_order"] = data["time_block"].map(block_index)
    for _, row in data.iterrows():
        current_order = int(row.get("block_order") if not pd.isna(row.get("block_order")) else 999)
        branch = str(row.get("primary_news_branch") or "unclassified_news_context")
        prior = data[data["block_order"].lt(current_order) & data["news_branch_tags"].fillna("").str.contains(branch, regex=False)]
        policy = derive_prior_policy(prior)
        rows.append(
            {
                "date": row.get("date"),
                "code": row.get("code"),
                "primary_news_branch": branch,
                "prior_case_count": int(len(prior)),
                "prior_case_count_bucket": count_bucket(len(prior)),
                "prior_branch_policy_status": policy["status"],
                "prior_branch_policy_hint": policy["hint"],
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return pd.DataFrame(rows)


def derive_prior_policy(prior: pd.DataFrame) -> dict[str, str]:
    if len(prior) < 8:
        return {"status": "insufficient_prior_cases", "hint": "use branch checklist only; do not change grade from prior cases"}
    returns = pd.to_numeric(prior.get("return_20d"), errors="coerce")
    excess = pd.to_numeric(prior.get("pool_excess_20d"), errors="coerce")
    pos = float((returns > 0).mean()) if returns.notna().any() else np.nan
    avg_excess = float(excess.mean()) if excess.notna().any() else np.nan
    if not math.isnan(pos) and not math.isnan(avg_excess) and pos >= 0.62 and avg_excess > 0:
        return {"status": "prior_support_observe_only", "hint": "prior branch was supportive, but still require current cross-channel confirmation"}
    if (not math.isnan(pos) and pos <= 0.45) or (not math.isnan(avg_excess) and avg_excess < -0.5):
        return {"status": "prior_counterevidence_review", "hint": "prior branch was weak; require stronger remediation/current evidence"}
    return {"status": "mixed_prior_cases", "hint": "prior cases are mixed; use branch as explanation and risk checklist"}


def build_prior_similar_cases(frame: pd.DataFrame, *, top_k: int = 3) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    data["block_order"] = data["time_block"].map(block_index)
    vectors = normalized_vectors(data, VECTOR_FIELDS)
    rows: list[dict[str, Any]] = []
    for idx, row in data.iterrows():
        current_order = int(row.get("block_order") if not pd.isna(row.get("block_order")) else 999)
        branch_tags = set(str(row.get("news_branch_tags") or "").split(";"))
        prior_index = data.index[data["block_order"].lt(current_order)].tolist()
        candidates: list[tuple[float, int, str]] = []
        for prior_idx in prior_index:
            prior_tags = set(str(data.at[prior_idx, "news_branch_tags"] or "").split(";"))
            overlap = branch_tags & prior_tags
            if not overlap:
                continue
            distance = vector_distance(vectors.loc[idx], vectors.loc[prior_idx])
            if math.isnan(distance):
                continue
            candidates.append((distance, int(prior_idx), ";".join(tag for tag in BRANCH_ORDER if tag in overlap)))
        for rank, (distance, prior_idx, matched_tags) in enumerate(sorted(candidates, key=lambda item: item[0])[:top_k], start=1):
            rows.append(
                {
                    "date": row.get("date"),
                    "code": row.get("code"),
                    "primary_news_branch": row.get("primary_news_branch"),
                    "similar_rank": rank,
                    "prior_date": data.at[prior_idx, "date"],
                    "prior_code": data.at[prior_idx, "code"],
                    "prior_name": data.at[prior_idx, "name"] if "name" in data else "",
                    "prior_time_block": data.at[prior_idx, "time_block"],
                    "prior_primary_news_branch": data.at[prior_idx, "primary_news_branch"],
                    "matched_branch_tags": matched_tags,
                    "similarity_band": similarity_band(distance),
                    "research_only": True,
                    "not_investment_instruction": True,
                }
            )
    return pd.DataFrame(rows)


def build_agent_previews(
    frame: pd.DataFrame,
    prior_metrics: pd.DataFrame,
    similar_cases: pd.DataFrame,
    *,
    max_rows: int,
) -> list[dict[str, Any]]:
    prior_by_key = {
        (str(row["date"]), str(row["code"]).zfill(6)): row.to_dict()
        for _, row in prior_metrics.iterrows()
    }
    similar_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if not similar_cases.empty:
        for _, row in similar_cases.iterrows():
            key = (str(row["date"]), str(row["code"]).zfill(6))
            similar_by_key.setdefault(key, []).append(
                {
                    "prior_date": str(row["prior_date"]),
                    "prior_code": str(row["prior_code"]).zfill(6),
                    "prior_name": str(row.get("prior_name") or ""),
                    "prior_time_block": str(row.get("prior_time_block") or ""),
                    "matched_branch_tags": str(row.get("matched_branch_tags") or ""),
                    "similarity_band": str(row.get("similarity_band") or ""),
                }
            )

    previews: list[dict[str, Any]] = []
    safe_frame = frame.sort_values(["date", "code"]).head(max_rows)
    for _, row in safe_frame.iterrows():
        key = (str(row["date"]), str(row["code"]).zfill(6))
        prior = prior_by_key.get(key, {})
        item = {
            "tool_id": "news_questionnaire_branch_case_auditor",
            "tool_version": "v1",
            "date": key[0],
            "code": key[1],
            "name": str(row.get("name") or ""),
            "time_block": str(row.get("time_block") or ""),
            "primary_news_branch": str(row.get("primary_news_branch") or ""),
            "news_branch_tags": str(row.get("news_branch_tags") or ""),
            "branch_policy": str(row.get("branch_policy") or ""),
            "branch_rationale": str(row.get("branch_rationale") or ""),
            "prior_case_count_bucket": str(prior.get("prior_case_count_bucket") or "none"),
            "prior_branch_policy_status": str(prior.get("prior_branch_policy_status") or "insufficient_prior_cases"),
            "prior_branch_policy_hint": str(prior.get("prior_branch_policy_hint") or "use branch checklist only"),
            "similar_prior_cases": similar_by_key.get(key, [])[:3],
            "agent_use": "checklist_and_counterevidence_only_not_alpha",
            "forbidden_use": "do_not_use_branch_or_prior_cases_as_order_instruction_or_standalone_positive_alpha",
            "source_ref_ids": [
                "data/date_generalization_cache/market_5000/news_questionnaire_features.csv.gz",
                "reports/date_generalization/news_questionnaire_branch_case_audit_v1.md",
            ],
            "research_only": True,
            "not_investment_instruction": True,
        }
        assert_no_future_fields(item)
        previews.append(item)
    return previews


def write_outputs(
    *,
    prefix: str,
    branched: pd.DataFrame,
    branch_metrics: pd.DataFrame,
    branch_aggregate: pd.DataFrame,
    prior_metrics: pd.DataFrame,
    similar_cases: pd.DataFrame,
    previews: list[dict[str, Any]],
) -> dict[str, Path]:
    paths = {
        "branch_detail": REPORT_DIR / f"{prefix}_branch_detail.csv",
        "branch_metrics": REPORT_DIR / f"{prefix}_branch_metrics.csv",
        "branch_aggregate": REPORT_DIR / f"{prefix}_branch_aggregate.csv",
        "prior_policy": REPORT_DIR / f"{prefix}_prior_policy.csv",
        "similar_cases": REPORT_DIR / f"{prefix}_similar_cases.csv",
        "agent_preview": REPORT_DIR / f"{prefix}_agent_preview.jsonl",
        "report": REPORT_DIR / f"{prefix}.md",
    }
    safe_detail_cols = [
        "date",
        "code",
        "name",
        "time_block",
        "primary_news_branch",
        "news_branch_tags",
        "branch_policy",
        "branch_rationale",
        "ds_news_risk_score",
        "ds_news_opportunity_score",
        "ds_news_uncertainty_score",
        "ds_news_quality_score",
        "ds_news_net_score",
        "rev_chip_score_quantile",
        "research_only",
        "not_investment_instruction",
    ]
    branched[[col for col in safe_detail_cols if col in branched]].to_csv(paths["branch_detail"], index=False, encoding="utf-8-sig")
    branch_metrics.to_csv(paths["branch_metrics"], index=False, encoding="utf-8-sig")
    branch_aggregate.to_csv(paths["branch_aggregate"], index=False, encoding="utf-8-sig")
    prior_metrics.to_csv(paths["prior_policy"], index=False, encoding="utf-8-sig")
    similar_cases.to_csv(paths["similar_cases"], index=False, encoding="utf-8-sig")
    write_jsonl(str(paths["agent_preview"]), previews)
    paths["report"].write_text(render_report(branched, branch_metrics, branch_aggregate, prior_metrics, similar_cases, paths), encoding="utf-8")
    return paths


def render_report(
    branched: pd.DataFrame,
    branch_metrics: pd.DataFrame,
    branch_aggregate: pd.DataFrame,
    prior_metrics: pd.DataFrame,
    similar_cases: pd.DataFrame,
    paths: dict[str, Path],
) -> str:
    primary_summary = (
        branched["primary_news_branch"].value_counts().rename_axis("primary_news_branch").reset_index(name="rows")
        if not branched.empty
        else pd.DataFrame(columns=["primary_news_branch", "rows"])
    )
    preview_policy = (
        prior_metrics["prior_branch_policy_status"].value_counts().rename_axis("prior_branch_policy_status").reset_index(name="rows")
        if not prior_metrics.empty
        else pd.DataFrame(columns=["prior_branch_policy_status", "rows"])
    )
    lines = [
        "# News Questionnaire Branch Case Audit v1",
        "",
        "本报告只用于 A 股研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Summary",
        "",
        f"- questionnaire_rows: `{len(branched)}`",
        f"- unique_stocks: `{nunique(branched, 'code')}`",
        f"- min_date: `{min_text(branched, 'date')}`",
        f"- max_date: `{max_text(branched, 'date')}`",
        f"- branch_detail: `{paths['branch_detail']}`",
        f"- branch_metrics: `{paths['branch_metrics']}`",
        f"- branch_aggregate: `{paths['branch_aggregate']}`",
        f"- prior_policy: `{paths['prior_policy']}`",
        f"- similar_cases: `{paths['similar_cases']}`",
        f"- agent_preview: `{paths['agent_preview']}`",
        "",
        "## Branch Counts",
        "",
        table(primary_summary),
        "",
        "## Offline Outcome Check",
        "",
        "未来 20 日收益只在本节离线评估中使用，不能进入 Agent evidence。小样本分叉只能证明需要复核，不能证明 alpha。",
        "",
        "### Aggregate",
        "",
        table(branch_aggregate[(branch_aggregate["scope"].eq("primary")) & branch_aggregate["split"].isin(["prior_blocks", "H2026_1"])] if not branch_aggregate.empty else branch_aggregate),
        "",
        "### By Block",
        "",
        table(branch_metrics[(branch_metrics["scope"].eq("primary")) & branch_metrics["rows"].gt(0)] if not branch_metrics.empty else branch_metrics),
        "",
        "## Prior-Only Policy Mix",
        "",
        table(preview_policy),
        "",
        "## Similar Case Coverage",
        "",
        f"- similar_case_rows: `{len(similar_cases)}`",
        f"- target_stockdates_with_prior_cases: `{nunique(similar_cases, ['date', 'code'])}`",
        "",
        "## Interpretation",
        "",
        "- `explicit_negative_event` 是复核/反证分叉，不是机械输出结论；需要看是否有独立修复证据。",
        "- `reversible_reversal_friction` 是本轮新增重点：高 ranker 反转候选遇到新闻风险/不确定性时，不得直接按风险分硬否决。",
        "- `routine_official_low_signal` 和 `soft_gap` 主要是 coverage / confidence / prompt hygiene，不是正向 alpha。",
        "- `peer_diffusion` 与 `policy_region_direct_support` 必须继续区分目标自身证据、同行扩散和政策背景，不能把同业热度自动迁移到个股。",
        "- Agent preview 只含分叉、prior policy 状态和相似 prior case 元信息；不含 return/GT/pool-excess 字段。",
    ]
    return "\n".join(lines) + "\n"


def assert_no_future_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            if key_text in FUTURE_FIELDS or key_text.startswith("future_"):
                raise ValueError(f"future/result field leaked to agent preview: {key_text}")
            assert_no_future_fields(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_future_fields(item)


def normalized_vectors(frame: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    values = pd.DataFrame(index=frame.index)
    for field in fields:
        if field in frame:
            values[field] = pd.to_numeric(frame[field], errors="coerce")
        else:
            values[field] = np.nan
    for field in values.columns:
        col = values[field]
        median = float(col.median()) if col.notna().any() else 0.0
        std = float(col.std()) if col.notna().sum() >= 3 else 1.0
        if std <= 0 or math.isnan(std):
            std = 1.0
        values[field] = (col.fillna(median) - median) / std
    return values


def vector_distance(a: pd.Series, b: pd.Series) -> float:
    diff = pd.to_numeric(a, errors="coerce") - pd.to_numeric(b, errors="coerce")
    if diff.isna().all():
        return float("nan")
    return float(np.sqrt(np.nanmean(np.square(diff))))


def block_for_date(value: Any) -> str | None:
    text = str(pd.to_datetime(value, errors="coerce").date()) if not pd.isna(pd.to_datetime(value, errors="coerce")) else ""
    for block, (start, end) in TIME_BLOCKS.items():
        if start <= text <= end:
            return block
    return None


def block_index(block: Any) -> int:
    order = {name: idx for idx, name in enumerate(TIME_BLOCKS)}
    return int(order.get(str(block), 999))


def count_bucket(count: int) -> str:
    if count <= 0:
        return "none"
    if count < 8:
        return "low"
    if count < 25:
        return "medium"
    return "high"


def similarity_band(distance: float) -> str:
    if distance <= 0.55:
        return "high"
    if distance <= 0.90:
        return "medium"
    return "low"


def f(row: pd.Series | dict[str, Any], key: str, default: float = np.nan) -> float:
    try:
        value = row.get(key, default)  # type: ignore[attr-defined]
    except AttributeError:
        return default
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def max_present(*values: float) -> float:
    present = [value for value in values if not math.isnan(value)]
    return max(present) if present else float("nan")


def min_present(*values: float) -> float:
    present = [value for value in values if not math.isnan(value)]
    return min(present) if present else float("nan")


def max_present_abs_negative(*values: float) -> float:
    present = [value for value in values if not math.isnan(value)]
    if not present:
        return float("nan")
    return min(present)


def abs_non_nan(value: float) -> float:
    return abs(value) if not math.isnan(value) else float("nan")


def contains_any(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.to_markdown(index=False)


def nunique(frame: pd.DataFrame, field: str | list[str]) -> int:
    if frame.empty:
        return 0
    if isinstance(field, list):
        missing = [item for item in field if item not in frame]
        if missing:
            return 0
        return int(frame[field].drop_duplicates().shape[0])
    if field not in frame:
        return 0
    return int(frame[field].dropna().astype(str).nunique())


def min_text(frame: pd.DataFrame, field: str) -> str:
    if field not in frame or frame.empty:
        return ""
    values = frame[field].dropna().astype(str)
    return values.min() if not values.empty else ""


def max_text(frame: pd.DataFrame, field: str) -> str:
    if field not in frame or frame.empty:
        return ""
    values = frame[field].dropna().astype(str)
    return values.max() if not values.empty else ""


if __name__ == "__main__":
    main()
