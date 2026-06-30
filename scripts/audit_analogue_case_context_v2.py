"""Audit time-safe analogue-case context as an Agent support tool.

This is a local, no-network, no-DeepSeek experiment. Future returns are used
only for offline evaluation. Agent preview outputs are sanitized and contain no
query future/result fields.
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

from scripts.run_kline_peer_chip_regime_scorer import (  # noqa: E402
    HIGH_RANKER_QUANTILE,
    TOP_PCTS,
    VALID_BLOCKS,
    apply_frequency,
    load_frame,
)
from src.agent_training.analogue_case_retriever import (  # noqa: E402
    DEFAULT_K,
    DEFAULT_RECENT_WINDOW_TD,
    assert_context_columns_safe,
    build_case_library,
    run_leakage_self_check,
    score_analogue_features,
)
from src.agent_training.dual_mode_round import DEFAULT_JOINED_GT_CACHE_PATH  # noqa: E402
from src.agent_training.quant_tool_context import sanitize_quant_tool_outcome  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
OUTPUT_PREFIX = "analogue_case_context_v2"
ROUND_TRIP_COST_PCT = 1.5
DECISION_FREQUENCIES = ["every_2_weeks", "weekly_friday", "weekly_tuesday"]
VARIANTS = [
    "baseline_rev_chip_score",
    "analogue_support_score",
    "rev_chip_plus_analogue_support",
    "rev_chip_analogue_guard",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit analogue-case context as an Agent tool.")
    parser.add_argument("--joined-cache", default=str(DEFAULT_JOINED_GT_CACHE_PATH))
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--recent-window-td", type=int, default=DEFAULT_RECENT_WINDOW_TD)
    parser.add_argument("--decision-frequencies", default=",".join(DECISION_FREQUENCIES))
    parser.add_argument("--top-pcts", default="0.05,0.10")
    parser.add_argument("--high-ranker-quantile", type=float, default=HIGH_RANKER_QUANTILE)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    decision_frequencies = [x.strip() for x in str(args.decision_frequencies).split(",") if x.strip()]
    top_pcts = [float(x) for x in str(args.top_pcts).split(",") if x.strip()]

    frame = load_audit_frame(Path(args.joined_cache), high_ranker_quantile=args.high_ranker_quantile)
    library = build_case_library(frame)
    leakage = run_leakage_self_check(library)
    scored = build_scored_frame(frame, library, k=args.k, recent_window_td=args.recent_window_td)
    scored = add_analogue_scores(scored)
    step_metrics = evaluate_all(scored, decision_frequencies=decision_frequencies, top_pcts=top_pcts)
    aggregate = aggregate_metrics(step_metrics)
    coverage = feature_coverage(scored)
    preview_rows = build_agent_preview_rows(aggregate, coverage)
    paths = write_outputs(
        prefix=args.output_prefix,
        scored=scored,
        step_metrics=step_metrics,
        aggregate=aggregate,
        coverage=coverage,
        preview_rows=preview_rows,
        leakage=leakage,
        high_ranker_quantile=args.high_ranker_quantile,
        top_pcts=top_pcts,
        decision_frequencies=decision_frequencies,
    )

    print("A股研究Agent")
    print(f"rows={len(scored)}")
    print(f"metrics={len(step_metrics)}")
    print(f"preview_rows={len(preview_rows)}")
    print(f"report={paths['report']}")
    print(f"agent_preview={paths['agent_preview']}")


def load_audit_frame(path: Path, *, high_ranker_quantile: float) -> pd.DataFrame:
    frame = load_frame(path, high_ranker_quantile=high_ranker_quantile)
    frame = frame.copy()
    frame["fwd_ret_20d"] = pd.to_numeric(frame["return_20d"], errors="coerce")
    required = {"date", "code", "time_block", "return_20d", "fwd_ret_20d", "rev_chip_score", "rev_chip_score_quantile"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required audit columns: {missing}")
    return frame.dropna(subset=["date", "code", "time_block", "fwd_ret_20d"]).reset_index(drop=True)


def build_scored_frame(
    frame: pd.DataFrame,
    library: Any,
    *,
    k: int,
    recent_window_td: int,
) -> pd.DataFrame:
    analogue = score_analogue_features(library, k=k, recent_window_td=recent_window_td)
    keep_cols = [
        "date",
        "code",
        "name",
        "time_block",
        "return_20d",
        "rev_chip_score",
        "rev_chip_score_quantile",
        "portfolio_candidate_pool",
        "tushare_industry",
        "kline_return_20d",
        "kline_return_60d",
        "kline_drawdown_60d",
        "corr_peer_relative_return_20d",
        "corr_peer_avg_return_20d",
        "tushare_industry_relative_return_20d",
        "lower_support",
        "upper_overhang",
    ]
    base = frame[[col for col in keep_cols if col in frame.columns]].copy()
    for col in ["date", "code"]:
        base[col] = base[col].astype(str)
        analogue[col] = analogue[col].astype(str)
    merged = base.merge(
        analogue.drop(columns=["time_block", "fwd_ret_20d"], errors="ignore"),
        on=["date", "code"],
        how="left",
    )
    return merged


def add_analogue_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["baseline_rev_chip_score"] = pd.to_numeric(out["rev_chip_score"], errors="coerce").fillna(0.0)
    out["analogue_branch"] = [
        analogue_branch(pos, decay, n)
        for pos, decay, n in zip(out.get("analogue_pos_rate"), out.get("regime_decay_signal"), out.get("n_candidates"))
    ]
    out["analogue_support_score"] = (
        0.45 * per_date_z(out, "analogue_pos_rate")
        + 0.30 * per_date_z(out, "analogue_base_rate")
        + 0.15 * per_date_z(out, "regime_decay_signal")
        - 0.10 * per_date_z(out, "analogue_std")
    )
    rev_rank = per_date_rank(out, "baseline_rev_chip_score")
    analogue_rank = per_date_rank(out, "analogue_support_score")
    out["rev_chip_plus_analogue_support"] = 0.80 * rev_rank + 0.20 * analogue_rank
    penalty = out["analogue_branch"].astype(str).eq("decay_warning_low_support").astype(float)
    low_count = out["analogue_branch"].astype(str).eq("insufficient_case_pool").astype(float)
    bonus = out["analogue_branch"].astype(str).eq("historical_supportive").astype(float)
    out["rev_chip_analogue_guard"] = rev_rank - 0.18 * penalty - 0.04 * low_count + 0.06 * bonus
    return out


def analogue_branch(pos_rate: Any, decay: Any, n_candidates: Any) -> str:
    pos = to_float(pos_rate)
    dec = to_float(decay)
    n = to_float(n_candidates)
    if math.isnan(n) or n < DEFAULT_K:
        return "insufficient_case_pool"
    if not math.isnan(pos) and not math.isnan(dec) and pos < 0.45 and dec < -2.0:
        return "decay_warning_low_support"
    if not math.isnan(pos) and not math.isnan(dec) and pos >= 0.60 and dec >= 0.0:
        return "historical_supportive"
    if not math.isnan(dec) and dec < -2.0:
        return "regime_decay_warning"
    if not math.isnan(pos) and pos < 0.45:
        return "low_historical_support"
    return "neutral_or_mixed"


def evaluate_all(
    scored: pd.DataFrame,
    *,
    decision_frequencies: list[str],
    top_pcts: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for decision_frequency in decision_frequencies:
        freq_frame = apply_frequency(scored, decision_frequency)
        for task_mode in ["portfolio_pool", "single_stock"]:
            task = task_frame(freq_frame, task_mode)
            if task.empty:
                continue
            for valid_block in VALID_BLOCKS:
                valid = task[task["time_block"].astype(str).eq(valid_block)].copy()
                if len(valid) < 50:
                    continue
                for top_pct in top_pcts:
                    for variant in VARIANTS:
                        if variant not in valid.columns:
                            continue
                        rows.append(
                            evaluate_variant(
                                valid,
                                variant=variant,
                                task_mode=task_mode,
                                valid_block=valid_block,
                                decision_frequency=decision_frequency,
                                top_pct=top_pct,
                            )
                        )
    return pd.DataFrame(rows)


def task_frame(frame: pd.DataFrame, task_mode: str) -> pd.DataFrame:
    if task_mode == "portfolio_pool":
        return frame[frame["portfolio_candidate_pool"].astype(bool)].copy()
    if task_mode == "single_stock":
        return frame.copy()
    raise ValueError(task_mode)


def evaluate_variant(
    frame: pd.DataFrame,
    *,
    variant: str,
    task_mode: str,
    valid_block: str,
    decision_frequency: str,
    top_pct: float,
) -> dict[str, Any]:
    work = frame.dropna(subset=[variant, "return_20d", "date", "code"]).copy()
    selected = select_top_by_date(work, score_col=variant, top_pct=top_pct)
    returns = pd.to_numeric(work["return_20d"], errors="coerce")
    selected_returns = pd.to_numeric(selected["return_20d"], errors="coerce") if not selected.empty else pd.Series(dtype=float)
    base_by_date = returns.groupby(work["date"].astype(str), sort=False).transform("mean")
    selected_base = base_by_date.loc[selected.index] if not selected.empty else pd.Series(dtype=float)
    selected_excess = selected_returns - selected_base
    daily_ic = daily_rank_ic(work, variant)
    turnover = mean_turnover(selected)
    cost = turnover * ROUND_TRIP_COST_PCT if task_mode == "portfolio_pool" else np.nan
    net_excess = float(selected_excess.mean() - cost) if selected_excess.size and not pd.isna(cost) else np.nan
    branch_mix = selected["analogue_branch"].astype(str).value_counts(normalize=True).to_dict() if "analogue_branch" in selected else {}
    concentration = selected["code"].astype(str).value_counts(normalize=True).max() if not selected.empty else np.nan
    return {
        "task_mode": task_mode,
        "variant": variant,
        "decision_frequency": decision_frequency,
        "top_pct": top_pct,
        "valid_block": valid_block,
        "candidate_rows": int(len(work)),
        "selected_rows": int(len(selected)),
        "coverage_dates": int(selected["date"].nunique()) if not selected.empty else 0,
        "unique_stocks": int(selected["code"].nunique()) if not selected.empty else 0,
        "top_stock_concentration": round(float(concentration), 6) if not pd.isna(concentration) else np.nan,
        "rank_ic": round(float(np.nanmean(daily_ic)), 6) if daily_ic else np.nan,
        "ic_positive_rate": round(float(np.mean([x > 0 for x in daily_ic])), 6) if daily_ic else np.nan,
        "avg_return_20d": round(float(selected_returns.mean()), 6) if selected_returns.size else np.nan,
        "positive_20d_rate": round(float((selected_returns > 0).mean()), 6) if selected_returns.size else np.nan,
        "base_avg_return_20d": round(float(selected_base.mean()), 6) if selected_base.size else np.nan,
        "pool_excess_20d": round(float(selected_excess.mean()), 6) if selected_excess.size else np.nan,
        "mean_turnover_one_way": round(float(turnover), 6) if not pd.isna(turnover) else np.nan,
        "net_pool_excess_after_turnover_cost": round(float(net_excess), 6) if not pd.isna(net_excess) else np.nan,
        "selected_decay_warning_rate": round(float(branch_mix.get("decay_warning_low_support", 0.0) + branch_mix.get("regime_decay_warning", 0.0)), 6),
        "selected_historical_supportive_rate": round(float(branch_mix.get("historical_supportive", 0.0)), 6),
        "research_only": True,
        "not_investment_instruction": True,
    }


def select_top_by_date(frame: pd.DataFrame, *, score_col: str, top_pct: float) -> pd.DataFrame:
    rows = []
    for _, group in frame.groupby(frame["date"].astype(str), sort=True):
        if group.empty:
            continue
        k = max(1, int(math.ceil(len(group) * top_pct)))
        rows.append(group.sort_values([score_col, "code"], ascending=[False, True]).head(k))
    if not rows:
        return frame.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=False)


def daily_rank_ic(frame: pd.DataFrame, score_col: str) -> list[float]:
    values: list[float] = []
    for _, group in frame.groupby(frame["date"].astype(str), sort=True):
        if len(group) < 5:
            continue
        scores = pd.to_numeric(group[score_col], errors="coerce")
        returns = pd.to_numeric(group["return_20d"], errors="coerce")
        valid = scores.notna() & returns.notna()
        if int(valid.sum()) < 5:
            continue
        corr = scores.loc[valid].corr(returns.loc[valid], method="spearman")
        if not pd.isna(corr):
            values.append(float(corr))
    return values


def mean_turnover(selected: pd.DataFrame) -> float:
    if selected.empty:
        return np.nan
    prev: set[str] = set()
    turnovers: list[float] = []
    for _, group in selected.groupby(selected["date"].astype(str), sort=True):
        cur = set(group["code"].astype(str))
        if not prev:
            turnovers.append(1.0 if cur else 0.0)
        else:
            overlap = len(prev & cur)
            turnovers.append(1.0 - overlap / max(len(prev), len(cur), 1))
        prev = cur
    return float(np.mean(turnovers)) if turnovers else np.nan


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["task_mode", "variant", "decision_frequency", "top_pct"]
    baseline = metrics[metrics["variant"].eq("baseline_rev_chip_score")]
    for values, group in metrics.groupby(keys, sort=True):
        task_mode, variant, decision_frequency, top_pct = values
        prior = group[~group["valid_block"].astype(str).eq("H2026_1")]
        h2026 = group[group["valid_block"].astype(str).eq("H2026_1")]
        base_group = baseline[
            baseline["task_mode"].eq(task_mode)
            & baseline["decision_frequency"].eq(decision_frequency)
            & baseline["top_pct"].eq(top_pct)
        ]
        base_prior = base_group[~base_group["valid_block"].astype(str).eq("H2026_1")]
        base_h2026 = base_group[base_group["valid_block"].astype(str).eq("H2026_1")]
        row = {key: value for key, value in zip(keys, values)}
        row.update(
            {
                "blocks": int(group["valid_block"].nunique()),
                "prior_blocks": int(prior["valid_block"].nunique()),
                "prior_rank_ic": mean(prior, "rank_ic"),
                "h2026_rank_ic": mean(h2026, "rank_ic"),
                "prior_avg_return_20d": mean(prior, "avg_return_20d"),
                "h2026_avg_return_20d": mean(h2026, "avg_return_20d"),
                "prior_positive_20d_rate": mean(prior, "positive_20d_rate"),
                "h2026_positive_20d_rate": mean(h2026, "positive_20d_rate"),
                "prior_pool_excess_20d": mean(prior, "pool_excess_20d"),
                "h2026_pool_excess_20d": mean(h2026, "pool_excess_20d"),
                "prior_net_pool_excess_after_turnover_cost": mean(prior, "net_pool_excess_after_turnover_cost"),
                "h2026_net_pool_excess_after_turnover_cost": mean(h2026, "net_pool_excess_after_turnover_cost"),
                "prior_decay_warning_rate": mean(prior, "selected_decay_warning_rate"),
                "h2026_decay_warning_rate": mean(h2026, "selected_decay_warning_rate"),
                "prior_supportive_rate": mean(prior, "selected_historical_supportive_rate"),
                "h2026_supportive_rate": mean(h2026, "selected_historical_supportive_rate"),
                "baseline_prior_avg_return_20d": mean(base_prior, "avg_return_20d"),
                "baseline_h2026_avg_return_20d": mean(base_h2026, "avg_return_20d"),
                "baseline_prior_positive_20d_rate": mean(base_prior, "positive_20d_rate"),
                "baseline_h2026_positive_20d_rate": mean(base_h2026, "positive_20d_rate"),
                "delta_prior_avg_return_20d": delta_mean(prior, base_prior, "avg_return_20d"),
                "delta_h2026_avg_return_20d": delta_mean(h2026, base_h2026, "avg_return_20d"),
                "delta_prior_positive_20d_rate": delta_mean(prior, base_prior, "positive_20d_rate"),
                "delta_h2026_positive_20d_rate": delta_mean(h2026, base_h2026, "positive_20d_rate"),
                "max_top_stock_concentration": max_or_nan(group, "top_stock_concentration"),
                "promotion_status": "baseline_reference" if variant == "baseline_rev_chip_score" else "",
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
        if variant != "baseline_rev_chip_score":
            row["promotion_status"] = promotion_status(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["promotion_status", "task_mode", "decision_frequency", "top_pct", "delta_h2026_avg_return_20d"],
        ascending=[True, True, True, True, False],
    )


def promotion_status(row: dict[str, Any]) -> str:
    prior_delta = to_float(row.get("delta_prior_avg_return_20d"))
    h_delta = to_float(row.get("delta_h2026_avg_return_20d"))
    prior_pos_delta = to_float(row.get("delta_prior_positive_20d_rate"))
    h_pos_delta = to_float(row.get("delta_h2026_positive_20d_rate"))
    prior_ic = to_float(row.get("prior_rank_ic"))
    h_ic = to_float(row.get("h2026_rank_ic"))
    conc = to_float(row.get("max_top_stock_concentration"))
    conc_ok = math.isnan(conc) or conc <= 0.35
    if prior_delta > 0 and h_delta > 0 and prior_pos_delta >= 0 and h_pos_delta >= 0 and prior_ic > 0 and h_ic > 0 and conc_ok:
        return "observe_relative_improvement_context_candidate"
    if h_delta > 0 and h_pos_delta >= 0 and conc_ok:
        return "observe_latest_positive_prior_weak"
    if prior_delta > 0 and prior_pos_delta >= 0 and conc_ok:
        return "observe_prior_positive_latest_weak"
    return "diagnostic_or_do_not_promote"


def build_agent_preview_rows(aggregate: pd.DataFrame, coverage: pd.DataFrame) -> list[dict[str, Any]]:
    if aggregate.empty:
        return []
    rows: list[dict[str, Any]] = []
    candidate = aggregate[
        aggregate["variant"].isin(["rev_chip_analogue_guard", "rev_chip_plus_analogue_support", "analogue_support_score"])
    ].copy()
    candidate = candidate[candidate["promotion_status"].astype(str).ne("diagnostic_or_do_not_promote")]
    if not candidate.empty:
        candidate = (
            candidate.assign(_status_rank=candidate["promotion_status"].astype(str).map(status_rank).fillna(99))
            .sort_values(["_status_rank", "task_mode", "decision_frequency", "top_pct"], ascending=[True, True, True, True])
            .drop(columns=["_status_rank"], errors="ignore")
        )
    if candidate.empty:
        candidate = aggregate[aggregate["variant"].eq("rev_chip_analogue_guard")].head(2).copy()
    for _, row in candidate.head(8).iterrows():
        status = str(row.get("promotion_status") or "diagnostic_or_do_not_promote")
        usable = status == "observe_relative_improvement_context_candidate"
        branch = "analogue_case_context_checklist"
        item = {
            "tool_id": f"analogue_case_context:{row.get('task_mode')}:{row.get('variant')}:{row.get('decision_frequency')}:top{int(float(row.get('top_pct', 0.1)) * 100)}",
            "tool_version": "v2",
            "task_mode": row.get("task_mode"),
            "policy_profile": "time_safe_analogue_case_context_v2",
            "policy_status": "context_only" if not usable else "relative_improvement_context_only",
            "decision_frequency": row.get("decision_frequency"),
            "feature_group": "kline_peer_industry_case_memory",
            "selection_mode": "prior_matured_case_knn_context",
            "score": status_score(status),
            "confidence": status_confidence(status),
            "risk_tier": "medium" if "latest_positive" in status else "review",
            "primary_risk_branch": branch,
            "risk_branch_labels": ["case_decay_warning", "historical_support_check", "do_not_use_as_standalone_alpha"],
            "branch_policy": "Agent may use this as base-rate and failure-case context; do not override hard news/financial/BookSkill counter-evidence.",
            "required_confirmation": [
                "current_news_or_announcement_check",
                "financial_asof_check",
                "peer_relative_strength_check",
                "grounded_bookskill_applicability_check",
            ],
            "known_false_veto_risk": "analogue decay can miss reversal candidates; use as checklist, not hard veto",
            "calibration_policy": "walk_forward_prior_blocks_only; H2026 is validation evidence, not tuning data",
            "action_hint": "observe_checklist",
            "usable_in_agent_default": usable,
            "top_features": [
                "kline_20d_momentum_context",
                "kline_60d_momentum_context",
                "kline_60d_drawdown_context",
                "peer_relative_momentum_context",
                "industry_relative_momentum_context",
            ],
            "missing_flags": missing_flags_from_coverage(coverage),
            "counter_evidence": counter_evidence_for_status(status),
            "source_ref_ids": ["analogue_case_context_v2", "analogue_case_retriever", "rag_case_retrieval_experiment_v1"],
            "train_valid_test_blocks": "walk_forward_H2023_2_to_H2026_1",
            "promotion_status": status,
            "research_only": True,
            "not_investment_instruction": True,
        }
        rows.append(sanitize_quant_tool_outcome(item))
    return rows


def write_outputs(
    *,
    prefix: str,
    scored: pd.DataFrame,
    step_metrics: pd.DataFrame,
    aggregate: pd.DataFrame,
    coverage: pd.DataFrame,
    preview_rows: list[dict[str, Any]],
    leakage: dict[str, Any],
    high_ranker_quantile: float,
    top_pcts: list[float],
    decision_frequencies: list[str],
) -> dict[str, Path]:
    scored_path = REPORT_DIR / f"{prefix}_safe_scored_detail.csv.gz"
    step_path = REPORT_DIR / f"{prefix}_step_metrics.csv"
    aggregate_path = REPORT_DIR / f"{prefix}_aggregate.csv"
    coverage_path = REPORT_DIR / f"{prefix}_feature_coverage.csv"
    preview_path = REPORT_DIR / f"{prefix}_agent_preview.jsonl"
    report_path = REPORT_DIR / f"{prefix}_findings.md"
    detail_cols = [
        "date",
        "code",
        "name",
        "time_block",
        "rev_chip_score_quantile",
        "portfolio_candidate_pool",
        "analogue_base_rate",
        "analogue_pos_rate",
        "analogue_std",
        "regime_decay_signal",
        "n_candidates",
        "dominant_skill_tag",
        "analogue_branch",
        "baseline_rev_chip_score",
        "analogue_support_score",
        "rev_chip_plus_analogue_support",
        "rev_chip_analogue_guard",
    ]
    scored[[col for col in detail_cols if col in scored.columns]].to_csv(scored_path, index=False, encoding="utf-8-sig")
    step_metrics.to_csv(step_path, index=False, encoding="utf-8-sig")
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    with preview_path.open("w", encoding="utf-8") as handle:
        for row in preview_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report_path.write_text(
        render_report(
            aggregate=aggregate,
            step_metrics=step_metrics,
            coverage=coverage,
            preview_rows=preview_rows,
            leakage=leakage,
            high_ranker_quantile=high_ranker_quantile,
            top_pcts=top_pcts,
            decision_frequencies=decision_frequencies,
        ),
        encoding="utf-8",
    )
    return {
        "scored": scored_path,
        "step_metrics": step_path,
        "aggregate": aggregate_path,
        "coverage": coverage_path,
        "agent_preview": preview_path,
        "report": report_path,
    }


def render_report(
    *,
    aggregate: pd.DataFrame,
    step_metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    preview_rows: list[dict[str, Any]],
    leakage: dict[str, Any],
    high_ranker_quantile: float,
    top_pcts: list[float],
    decision_frequencies: list[str],
) -> str:
    best = aggregate[
        aggregate["promotion_status"].astype(str).isin(
            [
                "observe_relative_improvement_context_candidate",
                "observe_latest_positive_prior_weak",
                "observe_prior_positive_latest_weak",
            ]
        )
    ].copy()
    if not best.empty:
        best = (
            best.assign(_status_rank=best["promotion_status"].astype(str).map(status_rank).fillna(99))
            .sort_values(["_status_rank", "task_mode", "decision_frequency", "top_pct"], ascending=[True, True, True, True])
            .head(12)
            .drop(columns=["_status_rank"], errors="ignore")
        )
    guard = aggregate[aggregate["variant"].eq("rev_chip_analogue_guard")].copy()
    baseline = aggregate[aggregate["variant"].eq("baseline_rev_chip_score")].copy()
    lines = [
        "# Analogue Case Context Audit v2",
        "",
        "本报告用于 A 股研究辅助型操作建议的上下文审计；不自动交易，不接券商接口，不承诺收益。零网络、零 DeepSeek。",
        "",
        "## 目的",
        "",
        "复查时间安全的历史相似案例检索是否应进入 Agent 默认上下文。重点不是让 RAG 直接预测收益，而是评估它能否作为 base-rate、失败案例和 regime 衰减提醒，帮助单支盯盘和候选对比减少误判。",
        "",
        "## 安全边界",
        "",
        f"- leakage 抽检条数: `{leakage.get('n_checked')}`",
        f"- 上下文特征数: `{leakage.get('context_feature_count')}`",
        f"- 未来字段进入上下文: `{leakage.get('forbidden_fields_in_context') or 'none'}`",
        f"- 时间安全断言: `{leakage.get('time_safe_assertions_passed')}`",
        f"- portfolio pool 默认阈值: `rev_chip_score_quantile >= {high_ranker_quantile}`",
        f"- 决策频率: `{', '.join(decision_frequencies)}`",
        f"- top_pct: `{', '.join(str(x) for x in top_pcts)}`",
        "",
        "## 覆盖",
        "",
        table(coverage),
        "",
        "## 结论摘要",
        "",
    ]
    if best.empty:
        lines.extend(
            [
                "- 没有 analogue 变体同时改善 prior 和 H2026；不应升为 alpha/ranker。",
                "- 继续保留为 Agent 审计上下文：重点提醒 `decay_warning_low_support`、`regime_decay_warning` 和相似失败案例，而不是直接提高研究等级。",
            ]
        )
    else:
        lines.extend(
            [
                "- 有 analogue 变体进入观察候选，但仍必须通过 Agent 多通道确认，不能单独升权。",
                "- 若与新闻/财报/同行/BookSkill 冲突，以当前多通道证据为准，历史相似案例只做 base-rate 和反证提醒。",
            ]
        )
    lines.extend(
        [
            "",
            "## 观察候选/失败候选",
            "",
            table(best if not best.empty else guard.head(12)),
            "",
            "## 默认 baseline 参考",
            "",
            table(baseline.head(12)),
            "",
            "## Agent preview",
            "",
        ]
    )
    if preview_rows:
        for row in preview_rows[:6]:
            lines.append(
                f"- `{row.get('tool_id')}` status=`{row.get('promotion_status')}` usable_default=`{row.get('usable_in_agent_default')}`"
            )
    else:
        lines.append("- no preview rows")
    lines.extend(
        [
            "",
            "## 使用建议",
            "",
            "- 默认接入方式：作为 `context_only` 检查清单，放在 Agent 决策前，与 BookSkill、新闻问卷、财报 as-of、同行/地域、筹码/K线工具一起审阅。",
            "- 严禁用相似案例 base-rate 单独输出 `继续深挖`；只有多通道确认同时支持时，才可作为辅助加分。",
            "- 当出现 `decay_warning_low_support` 且当前新闻/财报/同行没有强正向确认时，优先要求 Agent 明确写出反证和复核条件。",
            "",
            "## 复现",
            "",
            "```bash",
            "/data/cyx/1030/stock/.conda/stock-agent/bin/python scripts/audit_analogue_case_context_v2.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def feature_coverage(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [
        "analogue_base_rate",
        "analogue_pos_rate",
        "regime_decay_signal",
        "n_candidates",
        "dominant_skill_tag",
        "analogue_branch",
    ]:
        if col not in scored:
            continue
        values = scored[col]
        rows.append(
            {
                "feature": col,
                "non_null_rate": round(float(values.notna().mean()), 6),
                "non_empty_rate": round(float(values.astype(str).str.len().gt(0).mean()), 6),
                "unique_values": int(values.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def per_date_z(frame: pd.DataFrame, col: str) -> pd.Series:
    values = pd.to_numeric(frame.get(col, pd.Series(np.nan, index=frame.index)), errors="coerce")

    def _z(group: pd.Series) -> pd.Series:
        std = float(group.std())
        if std <= 0 or math.isnan(std):
            return pd.Series(0.0, index=group.index)
        return ((group - float(group.mean())) / std).fillna(0.0)

    return values.groupby(frame["date"].astype(str), sort=False).transform(_z).fillna(0.0)


def per_date_rank(frame: pd.DataFrame, col: str) -> pd.Series:
    values = pd.to_numeric(frame.get(col, pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    return values.groupby(frame["date"].astype(str), sort=False).rank(pct=True, method="average").fillna(0.5)


def mean(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame:
        return np.nan
    values = pd.to_numeric(frame[col], errors="coerce")
    return round(float(values.mean()), 6) if values.notna().any() else np.nan


def delta_mean(frame: pd.DataFrame, base: pd.DataFrame, col: str) -> float:
    left = mean(frame, col)
    right = mean(base, col)
    if pd.isna(left) or pd.isna(right):
        return np.nan
    return round(float(left - right), 6)


def max_or_nan(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame:
        return np.nan
    values = pd.to_numeric(frame[col], errors="coerce")
    return round(float(values.max()), 6) if values.notna().any() else np.nan


def to_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def status_score(status: str) -> float:
    if status == "observe_relative_improvement_context_candidate":
        return 0.55
    if status == "observe_latest_positive_prior_weak":
        return 0.40
    if status == "observe_prior_positive_latest_weak":
        return 0.35
    return 0.15


def status_confidence(status: str) -> float:
    if status == "observe_relative_improvement_context_candidate":
        return 0.50
    if status.startswith("observe_"):
        return 0.30
    return 0.20


def counter_evidence_for_status(status: str) -> list[str]:
    if status == "observe_relative_improvement_context_candidate":
        return [
            "context_only_not_standalone_alpha",
            "relative_improvement_not_absolute_profit_proof",
            "requires_news_financial_peer_bookskill_confirmation",
            "watch_false_veto_on_reversal_candidates",
        ]
    if status == "observe_latest_positive_prior_weak":
        return ["latest_block_positive_but_prior_weak", "do_not_promote_without_more_panels"]
    if status == "observe_prior_positive_latest_weak":
        return ["prior_positive_but_latest_block_weak", "do_not_promote_without_latest_confirmation"]
    return ["diagnostic_only", "do_not_promote", "analogue_context_not_enough_for_ranker"]


def status_rank(status: str) -> int:
    order = {
        "observe_relative_improvement_context_candidate": 0,
        "observe_latest_positive_prior_weak": 1,
        "observe_prior_positive_latest_weak": 2,
        "diagnostic_or_do_not_promote": 3,
        "baseline_reference": 4,
    }
    return order.get(status, 99)


def missing_flags_from_coverage(coverage: pd.DataFrame) -> list[str]:
    flags: list[str] = []
    if coverage.empty:
        return ["coverage_audit_missing"]
    lookup = {str(row["feature"]): float(row["non_null_rate"]) for _, row in coverage.iterrows()}
    if lookup.get("analogue_pos_rate", 0.0) < 0.90:
        flags.append("analogue_case_coverage_below_90pct")
    if lookup.get("regime_decay_signal", 0.0) < 0.80:
        flags.append("regime_decay_coverage_below_80pct")
    return flags


def table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return frame.to_csv(index=False)


if __name__ == "__main__":
    assert_context_columns_safe([])
    main()
