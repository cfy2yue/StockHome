"""Audit realized turnover cost for kline/peer/chip portfolio scorer outputs.

This is an offline evaluation helper. Future returns are used only to audit
already generated scorer details and are never written into Agent evidence.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.quant_tool_context import sanitize_quant_tool_outcome  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
OUTPUT_PREFIX = "kline_peer_chip_turnover_cost_audit_v1"
DEFAULT_ROUND_TRIP_COST_PCT = 1.5
DEFAULT_TOP_PCTS = [0.05, 0.10, 0.20]
VARIANTS = [
    "baseline_rev_chip_score",
    "manual_regime_reversal_score",
    "logistic_kline_peer_chip",
    "logistic_kline_peer_chip_regime",
]
DEFAULT_DETAIL_FILES = [
    REPORT_DIR / "kline_peer_chip_regime_scorer_v1_scored_detail.csv",
    REPORT_DIR / "kline_peer_chip_regime_scorer_v1_biweekly_scored_detail.csv",
    REPORT_DIR / "kline_peer_chip_regime_scorer_v1_weekly_friday_scored_detail.csv",
    REPORT_DIR / "kline_peer_chip_regime_scorer_v1_weekly_tuesday_scored_detail.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit realized turnover/cost for kline peer chip scorer.")
    parser.add_argument("--detail-files", nargs="*", default=[str(p) for p in DEFAULT_DETAIL_FILES])
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--top-pcts", default=",".join(str(x) for x in DEFAULT_TOP_PCTS))
    parser.add_argument("--round-trip-cost-pct", type=float, default=DEFAULT_ROUND_TRIP_COST_PCT)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    top_pcts = [float(x) for x in str(args.top_pcts).split(",") if str(x).strip()]
    details, missing = load_detail_files([Path(x) for x in args.detail_files])
    daily_rows: list[pd.DataFrame] = []
    for run_name, frame in details:
        for variant in VARIANTS:
            if variant not in frame.columns:
                continue
            for top_pct in top_pcts:
                daily = portfolio_daily_metrics(
                    frame,
                    variant=variant,
                    top_pct=top_pct,
                    source_run=run_name,
                    round_trip_cost_pct=args.round_trip_cost_pct,
                )
                if not daily.empty:
                    daily_rows.append(daily)
    daily_metrics = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    aggregate = aggregate_turnover_metrics(daily_metrics)

    daily_path = REPORT_DIR / f"{args.output_prefix}_daily.csv"
    aggregate_path = REPORT_DIR / f"{args.output_prefix}_aggregate.csv"
    report_path = REPORT_DIR / f"{args.output_prefix}.md"
    rule_outcomes_path = REPORT_DIR / f"{args.output_prefix}_rule_outcomes.jsonl"
    daily_metrics.to_csv(daily_path, index=False, encoding="utf-8-sig")
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    write_rule_outcomes(rule_outcomes_path, build_rule_outcomes(aggregate))
    write_report(report_path, aggregate, missing, args.round_trip_cost_pct)

    print("A股研究Agent")
    print(f"detail_files={len(details)}")
    print(f"daily_rows={len(daily_metrics)}")
    print(f"aggregate_rows={len(aggregate)}")
    print(f"report={report_path}")
    print(f"rule_outcomes={rule_outcomes_path}")


def load_detail_files(paths: list[Path]) -> tuple[list[tuple[str, pd.DataFrame]], list[str]]:
    details: list[tuple[str, pd.DataFrame]] = []
    missing: list[str] = []
    for path in paths:
        if not path.exists():
            missing.append(str(path))
            continue
        frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
        frame.columns = [col.lstrip("\ufeff") for col in frame.columns]
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
        frame["return_20d"] = pd.to_numeric(frame["return_20d"], errors="coerce")
        frame = frame[
            frame["task_mode"].astype(str).eq("portfolio_pool")
            & frame["date"].notna()
            & frame["return_20d"].notna()
        ].copy()
        details.append((infer_source_run(path), frame))
    return details, missing


def infer_source_run(path: Path) -> str:
    name = path.name
    if "biweekly" in name:
        return "every_2_weeks"
    if "weekly_friday" in name:
        return "weekly_friday"
    if "weekly_tuesday" in name:
        return "weekly_tuesday"
    return "all_dates"


def portfolio_daily_metrics(
    frame: pd.DataFrame,
    *,
    variant: str,
    top_pct: float,
    source_run: str,
    round_trip_cost_pct: float,
) -> pd.DataFrame:
    work = frame.copy()
    work[variant] = pd.to_numeric(work[variant], errors="coerce")
    work = work.dropna(subset=[variant, "return_20d", "date", "code"])
    rows: list[dict[str, Any]] = []
    prev_holdings: set[str] = set()

    for date, group in work.groupby(work["date"].astype(str), sort=True):
        if group.empty:
            continue
        selected = select_top_by_date(group, variant=variant, top_pct=top_pct)
        holdings = set(selected["code"].astype(str))
        turnover = 1.0 if not prev_holdings else turnover_one_way(prev_holdings, holdings)
        pool_return = float(pd.to_numeric(group["return_20d"], errors="coerce").mean())
        selected_return = float(pd.to_numeric(selected["return_20d"], errors="coerce").mean())
        gross_excess = selected_return - pool_return
        rank_ic = daily_rank_ic(group, variant)
        cost = turnover * round_trip_cost_pct
        concentration = selected["code"].astype(str).value_counts(normalize=True).max() if not selected.empty else np.nan
        rows.append(
            {
                "source_run": source_run,
                "decision_frequency": source_run,
                "variant": variant,
                "top_pct": top_pct,
                "date": date,
                "valid_block": str(group["valid_block"].iloc[0]) if "valid_block" in group else "",
                "pool_rows": int(len(group)),
                "selected_rows": int(len(selected)),
                "unique_stocks": int(selected["code"].nunique()),
                "top_stock_concentration_day": round(float(concentration), 6) if not pd.isna(concentration) else np.nan,
                "portfolio_return_20d": round(selected_return, 6),
                "pool_return_20d": round(pool_return, 6),
                "gross_pool_excess_20d": round(gross_excess, 6),
                "rank_ic": round(float(rank_ic), 6) if not pd.isna(rank_ic) else np.nan,
                "turnover_one_way": round(float(turnover), 6),
                "estimated_cost_pct": round(float(cost), 6),
                "net_pool_excess_after_turnover_cost": round(float(gross_excess - cost), 6),
                "portfolio_positive_20d": bool(selected_return > 0),
                "selected_codes": ";".join(sorted(holdings)[:30]),
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
        prev_holdings = holdings
    return pd.DataFrame(rows)


def select_top_by_date(group: pd.DataFrame, *, variant: str, top_pct: float) -> pd.DataFrame:
    k = max(1, int(math.ceil(len(group) * top_pct)))
    return group.sort_values([variant, "code"], ascending=[False, True]).head(k)


def turnover_one_way(previous: set[str], current: set[str]) -> float:
    if not previous and not current:
        return 0.0
    if not previous or not current:
        return 1.0
    overlap = len(previous & current)
    denom = max(len(previous), len(current), 1)
    return 1.0 - overlap / denom


def daily_rank_ic(group: pd.DataFrame, variant: str) -> float:
    scores = pd.to_numeric(group[variant], errors="coerce")
    returns = pd.to_numeric(group["return_20d"], errors="coerce")
    valid = scores.notna() & returns.notna()
    if int(valid.sum()) < 5:
        return np.nan
    corr = scores.loc[valid].corr(returns.loc[valid], method="spearman")
    return float(corr) if not pd.isna(corr) else np.nan


def aggregate_turnover_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["source_run", "decision_frequency", "variant", "top_pct"]
    for values, group in daily.groupby(keys, sort=True):
        prior = group[~group["valid_block"].astype(str).eq("H2026_1")]
        h2026 = group[group["valid_block"].astype(str).eq("H2026_1")]
        all_selected = (
            group.assign(_codes=group["selected_codes"].astype(str).str.split(";"))
            .explode("_codes")["_codes"]
            .replace("", np.nan)
            .dropna()
        )
        max_conc = float(all_selected.value_counts(normalize=True).max()) if not all_selected.empty else np.nan
        row = {key: value for key, value in zip(keys, values)}
        row.update(
            {
                "days": int(len(group)),
                "prior_days": int(len(prior)),
                "h2026_days": int(len(h2026)),
                "prior_gross_pool_excess_20d": mean(prior, "gross_pool_excess_20d"),
                "h2026_gross_pool_excess_20d": mean(h2026, "gross_pool_excess_20d"),
                "prior_mean_rank_ic": mean(prior, "rank_ic"),
                "h2026_rank_ic": mean(h2026, "rank_ic"),
                "prior_avg_turnover_one_way": mean(prior, "turnover_one_way"),
                "h2026_avg_turnover_one_way": mean(h2026, "turnover_one_way"),
                "prior_estimated_cost_pct": mean(prior, "estimated_cost_pct"),
                "h2026_estimated_cost_pct": mean(h2026, "estimated_cost_pct"),
                "prior_net_pool_excess_after_turnover_cost": mean(prior, "net_pool_excess_after_turnover_cost"),
                "h2026_net_pool_excess_after_turnover_cost": mean(h2026, "net_pool_excess_after_turnover_cost"),
                "prior_positive_20d_rate": bool_mean(prior, "portfolio_positive_20d"),
                "h2026_positive_20d_rate": bool_mean(h2026, "portfolio_positive_20d"),
                "prior_avg_return_20d": mean(prior, "portfolio_return_20d"),
                "h2026_avg_return_20d": mean(h2026, "portfolio_return_20d"),
                "max_top_stock_concentration": round(max_conc, 6) if not pd.isna(max_conc) else np.nan,
                "promotion_status": turnover_promotion_status(prior, h2026, max_conc),
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["promotion_status", "source_run", "top_pct", "prior_net_pool_excess_after_turnover_cost"],
        ascending=[True, True, True, False],
    )


def mean(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame:
        return np.nan
    values = pd.to_numeric(frame[col], errors="coerce")
    return round(float(values.mean()), 6) if values.notna().any() else np.nan


def bool_mean(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame:
        return np.nan
    return round(float(frame[col].astype(bool).mean()), 6)


def turnover_promotion_status(prior: pd.DataFrame, h2026: pd.DataFrame, concentration: float) -> str:
    if prior.empty or h2026.empty:
        return "observe_insufficient_blocks"
    prior_net = float(pd.to_numeric(prior["net_pool_excess_after_turnover_cost"], errors="coerce").mean())
    h_net = float(pd.to_numeric(h2026["net_pool_excess_after_turnover_cost"], errors="coerce").mean())
    prior_pos = float(prior["portfolio_positive_20d"].astype(bool).mean())
    h_pos = float(h2026["portfolio_positive_20d"].astype(bool).mean())
    prior_ic = float(pd.to_numeric(prior["rank_ic"], errors="coerce").mean())
    h_ic = float(pd.to_numeric(h2026["rank_ic"], errors="coerce").mean())
    conc_ok = pd.isna(concentration) or concentration <= 0.25
    if prior_net > 0 and h_net > 0 and prior_pos >= 0.60 and h_pos >= 0.60 and prior_ic >= 0.03 and h_ic >= 0.03 and conc_ok:
        return "accepted_cost_recheck_candidate"
    if h_net > 0 and h_pos >= 0.60 and h_ic >= 0.03:
        return "observe_h2026_positive_prior_weak"
    if prior_net > 0 and prior_pos >= 0.60 and prior_ic >= 0.03:
        return "observe_prior_positive_latest_weak"
    return "rejected_or_diagnostic_only"


def write_report(path: Path, aggregate: pd.DataFrame, missing: list[str], round_trip_cost_pct: float) -> None:
    lines = [
        "# Kline Peer Chip Turnover Cost Audit v1",
        "",
        "本报告只用于 A 股研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Purpose",
        "",
        "上一轮 `kline_peer_chip_regime_scorer` 使用固定 1.5% 成本扣减，可能对低频组合过严。本审计按相邻决策日组合重叠度估算真实换手成本：",
        "",
        "- `turnover_one_way = 1 - overlap / max(previous_count, current_count)`；",
        f"- 初始建仓日按 `turnover_one_way=1.0`；成本 = `turnover_one_way * {round_trip_cost_pct:.2f}%`；",
        "- 只重估 `portfolio_pool`，不改 Agent evidence，不写全局 rule outcome。",
        "",
    ]
    if missing:
        lines.extend(["## Missing Inputs", "", *[f"- {item}" for item in missing], ""])
    lines.extend(["## Aggregate Results", "", table(aggregate), ""])
    if not aggregate.empty:
        accepted = aggregate[aggregate["promotion_status"].astype(str).eq("accepted_cost_recheck_candidate")]
        best = aggregate.sort_values(
            ["h2026_net_pool_excess_after_turnover_cost", "prior_net_pool_excess_after_turnover_cost"],
            ascending=[False, False],
        ).head(8)
        lines.extend(["## Best H2026 Rows", "", table(best), ""])
        if accepted.empty:
            lines.extend(
                [
                    "## Decision",
                    "",
                    "真实换手成本重估后仍未出现可直接默认升权的组合。若 H2026 为正但 prior 不稳，只能进入小样本 Agent/DS 复核或继续作为观察型工具。",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "## Decision",
                    "",
                    "`accepted_cost_recheck_candidate` 只表示通过成本重估进入下一轮小样本 DS ablation，不等于最终默认策略。下一步仍需验证 Agent 是否能正确使用该工具且不增加坏暴露。",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_rule_outcomes(aggregate: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if aggregate.empty:
        return rows
    for _, row in aggregate.iterrows():
        variant = str(row.get("variant") or "")
        if variant == "baseline_rev_chip_score":
            continue
        status = str(row.get("promotion_status") or "")
        usable = status == "accepted_cost_recheck_candidate"
        item = {
            "tool_id": f"kline_peer_chip_turnover_cost:{variant}:top{int(float(row['top_pct']) * 100)}:{row['decision_frequency']}",
            "tool_version": "v1",
            "task_mode": "portfolio_pool",
            "policy_profile": "turnover_cost_rechecked_kline_peer_chip_v1",
            "decision_frequency": row.get("decision_frequency", "every_2_weeks"),
            "feature_group": "kline_peer_chip",
            "selection_mode": "walk_forward_top_pct_rerank_with_realized_turnover_cost_audit",
            "score": {"accepted_cost_recheck_candidate": 0.72, "observe_h2026_positive_prior_weak": 0.45, "observe_prior_positive_latest_weak": 0.35}.get(status, 0.1),
            "score_quantile": None,
            "confidence": {"accepted_cost_recheck_candidate": 0.68, "observe_h2026_positive_prior_weak": 0.45, "observe_prior_positive_latest_weak": 0.35}.get(status, 0.2),
            "action_hint": "continue_research" if usable else "observe",
            "usable_in_agent_default": usable,
            "top_features": [
                "kline_return_20d",
                "kline_return_60d",
                "corr_peer_avg_return_20d",
                "lower_support",
                "chip_concentration",
            ],
            "missing_flags": [],
            "counter_evidence": status_counter_evidence(status),
            "source_ref_ids": ["kline_peer_chip_turnover_cost_audit_v1", "kline_peer_chip_regime_scorer_v1"],
            "train_valid_test_blocks": "walk_forward_H2023_2_to_H2026_1",
            "promotion_status": status,
            "research_only": True,
            "not_investment_instruction": True,
        }
        rows.append(sanitize_quant_tool_outcome(item))
    return rows


def status_counter_evidence(status: str) -> list[str]:
    if status == "accepted_cost_recheck_candidate":
        return [
            "research_only",
            "requires_current_news_financial_bookskill_peer_review",
            "not_final_default_strategy",
        ]
    if status == "observe_h2026_positive_prior_weak":
        return ["latest_block_positive_but_prior_or_stability_gate_failed", "observe_only"]
    if status == "observe_prior_positive_latest_weak":
        return ["prior_blocks_positive_but_latest_gate_failed", "observe_only"]
    return ["turnover_cost_or_rank_ic_gate_failed", "diagnostic_only", "do_not_promote"]


def write_rule_outcomes(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(__import__("json").dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return frame.to_csv(index=False)


if __name__ == "__main__":
    main()
