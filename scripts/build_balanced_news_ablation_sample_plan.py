from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_news_financial_interactions import (  # noqa: E402
    DEFAULT_JOINED_GT,
    DEFAULT_REPORT_DIR,
    PLAN_COLUMNS,
    add_interaction_columns,
    assert_no_future_plan_columns,
    load_rows,
)


OUTPUT_PREFIX = "balanced_news_ablation_sample_plan_v1"
TARGET_BLOCKS = ["H2023_2", "H2024_2", "H2025_1", "H2026_1"]


@dataclass(frozen=True)
class BalancedRule:
    rule_id: str
    rule_type: str
    reason: str
    selector: pd.Series
    preferred_sort: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced next-round news ablation sample plan without future columns.")
    parser.add_argument("--joined-gt", default=str(DEFAULT_JOINED_GT))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--max-rows", type=int, default=24)
    parser.add_argument("--max-per-rule", type=int, default=4)
    parser.add_argument("--max-per-block", type=int, default=8)
    parser.add_argument("--min-plan-rows", type=int, default=8)
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    frame = add_balanced_columns(add_interaction_columns(load_rows(Path(args.joined_gt))))
    rule_metrics = build_rule_metrics(frame)
    plan = build_balanced_sample_plan(
        frame,
        rule_metrics,
        max_rows=args.max_rows,
        max_per_rule=args.max_per_rule,
        max_per_block=args.max_per_block,
    )
    if len(plan) < args.min_plan_rows:
        raise RuntimeError(f"balanced sample plan too small: {len(plan)} < {args.min_plan_rows}")
    assert_no_future_plan_columns(plan)

    metrics_path = report_dir / f"{OUTPUT_PREFIX}_rule_metrics.csv"
    plan_path = report_dir / f"{OUTPUT_PREFIX}.csv"
    micro_path = report_dir / f"{OUTPUT_PREFIX}_micro48.csv"
    report_path = report_dir / f"{OUTPUT_PREFIX}.md"
    micro = build_micro48_plan(plan)
    assert_no_future_plan_columns(micro)
    rule_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    plan.to_csv(plan_path, index=False, encoding="utf-8-sig")
    micro.to_csv(micro_path, index=False, encoding="utf-8-sig")
    report_path.write_text(render_report(frame, rule_metrics, plan, micro, metrics_path, plan_path, micro_path), encoding="utf-8")

    print("A股研究Agent")
    print(f"rows={len(frame)}")
    print(f"plan_rows={len(plan)}")
    print(f"unique_codes={plan['code'].nunique() if not plan.empty else 0}")
    print(f"metrics={metrics_path}")
    print(f"sample_plan={plan_path}")
    print(f"micro48_plan={micro_path}")
    print(f"report={report_path}")


def add_balanced_columns(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["moderate_rsi"] = data["rsi14"].between(40, 70, inclusive="both")
    data["moderate_prior_return"] = data["prior_return_20d"].between(-12, 25, inclusive="both")
    data["high_prior_return"] = data["prior_return_20d"].ge(35) | data["rsi14"].ge(80)
    data["low_news_risk"] = data["news_warning_score"].fillna(0).le(0.1) & data["news_negative_materiality_30d"].fillna(0).le(0.2)
    data["news_quality_available"] = data["news_available"] & data["news_evidence_quality"].fillna(0).ge(0.6)
    data["financial_low_risk_or_missing"] = (~data["financial_available"]) | data["financial_quality_risk_score"].fillna(0).le(0.35)
    data["financial_low_risk_available"] = data["financial_available"] & data["financial_quality_risk_score"].fillna(0).le(0.35)
    data["book_unresolved_or_gap"] = data["triggered_skills"].fillna("").astype(str).str.len().gt(0) | data["data_gaps"].fillna("").astype(str).str.contains("missing|unavailable", case=False, regex=True)
    data["potential_active_clean_context"] = (
        data["moderate_prior_return"]
        & data["moderate_rsi"]
        & data["relative_strength_rank"].between(0.55, 0.9, inclusive="both")
        & data["peer_confirmed"]
        & data["low_news_risk"]
        & data["financial_low_risk_or_missing"]
    )
    data["potential_active_with_news_financial"] = (
        data["potential_active_clean_context"]
        & data["news_quality_available"]
        & data["financial_low_risk_available"]
    )
    data["counter_high_momentum_with_gaps"] = (
        data["high_prior_return"]
        & (data["news_missing_rate"].ge(0.8) | ~data["financial_available"] | data["book_unresolved_or_gap"])
    )
    data["quality_control_news_available"] = (
        data["news_quality_available"]
        & data["low_news_risk"]
        & data["official_confirmation_score"].fillna(0).ge(0.6)
        & data["news_opportunity_score"].fillna(0).le(0.2)
    )
    return data


def balanced_rules(frame: pd.DataFrame) -> list[BalancedRule]:
    return [
        BalancedRule(
            "counter_news_opportunity_peer_weak_or_fin_missing_v1",
            "counterevidence",
            "新闻机会或公告可见但同行/财报确认弱，测试机会分是否仍应被压低。",
            frame["news_high_opportunity"] & (frame["peer_weak"] | ~frame["financial_available"]),
            "counter_score",
        ),
        BalancedRule(
            "counter_news_risk_peer_weak_v1",
            "counterevidence",
            "新闻风险高且同行弱，测试风险扩散/同行弱确认。",
            frame["news_high_risk"] & frame["peer_weak"],
            "counter_score",
        ),
        BalancedRule(
            "control_routine_official_low_signal_v1",
            "control",
            "官方/公告很多但方向性弱，测试常规公告是否应只作为 control。",
            frame["routine_official_low_signal"],
            "event_count",
        ),
        BalancedRule(
            "control_quality_news_available_low_risk_v1",
            "quality_control",
            "新闻质量可用、风险低但机会分弱，测试 quality/uncertainty 问卷是否能避免过度防守。",
            frame["quality_control_news_available"],
            "news_evidence_quality",
        ),
        BalancedRule(
            "potential_active_clean_context_v1",
            "potential_active",
            "量价不过热、同行确认、新闻风险低，作为可能产生非零研究暴露的均衡对照。",
            frame["potential_active_clean_context"],
            "relative_strength_rank",
        ),
        BalancedRule(
            "potential_active_news_financial_confirmed_v1",
            "potential_active",
            "普通新闻与财报低风险同时可见，测试新闻+财报是否只会防守还是能支持合理观察。",
            frame["potential_active_with_news_financial"],
            "relative_strength_rank",
        ),
        BalancedRule(
            "counter_high_momentum_with_info_gaps_v1",
            "shortcut_risk",
            "高动量叠加新闻/财报/Book Skill 缺口，测试 keyword_only 是否追高。",
            frame["counter_high_momentum_with_gaps"],
            "prior_return_20d",
        ),
    ]


def build_rule_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = _metric_row(frame)
    rows: list[dict[str, Any]] = []
    for rule in balanced_rules(frame):
        subset = frame[rule.selector.fillna(False)].copy()
        row = {
            "candidate_rule": rule.rule_id,
            "rule_type": rule.rule_type,
            "reason_to_test": rule.reason,
            **_metric_row(subset),
            "baseline_positive_20d_rate": baseline["positive_20d_rate"],
            "baseline_avg_return_20d": baseline["avg_return_20d"],
            "delta_positive_20d_rate": _round(_metric_row(subset)["positive_20d_rate"] - baseline["positive_20d_rate"]) if not subset.empty else None,
            "delta_avg_return_20d": _round(_metric_row(subset)["avg_return_20d"] - baseline["avg_return_20d"]) if not subset.empty else None,
            "block_count": int(subset["time_block"].nunique()) if not subset.empty else 0,
            "avg_news_missing_rate": _round(subset["news_missing_rate"].mean()) if not subset.empty else None,
            "next_action": _next_action(rule, subset),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_balanced_sample_plan(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    max_rows: int,
    max_per_rule: int,
    max_per_block: int,
) -> pd.DataFrame:
    metric_by_rule = metrics.set_index("candidate_rule") if not metrics.empty else pd.DataFrame()
    selected: list[pd.DataFrame] = []
    used_keys: set[tuple[str, str]] = set()
    block_counts: dict[str, int] = {}
    for rule in balanced_rules(frame):
        subset = frame[rule.selector.fillna(False)].copy()
        if subset.empty:
            continue
        subset = subset[subset["time_block"].isin(TARGET_BLOCKS)].copy()
        if subset.empty:
            continue
        ascending = rule.rule_type in {"counterevidence", "shortcut_risk"}
        subset["_sort_value"] = pd.to_numeric(subset.get(rule.preferred_sort), errors="coerce")
        subset = subset.sort_values(["time_block", "_sort_value"], ascending=[True, ascending])
        per_rule = []
        for block in TARGET_BLOCKS:
            block_subset = subset[subset["time_block"].eq(block)]
            if block_subset.empty:
                continue
            for _, row in block_subset.iterrows():
                key = (str(row["date"]), str(row["code"]).zfill(6))
                if key in used_keys:
                    continue
                if block_counts.get(block, 0) >= max_per_block:
                    continue
                per_rule.append(row)
                used_keys.add(key)
                block_counts[block] = block_counts.get(block, 0) + 1
                break
            if len(per_rule) >= max_per_rule:
                break
        if not per_rule:
            continue
        sample = pd.DataFrame(per_rule)
        sample["candidate_rule"] = rule.rule_id
        sample["reason_to_test"] = rule.reason
        if rule.rule_id in metric_by_rule.index:
            metric = metric_by_rule.loc[rule.rule_id]
            sample["sample_stock_concentration_note"] = (
                "rows="
                + str(int(metric.get("rows", 0) or 0))
                + "; stocks="
                + str(int(metric.get("unique_stocks", 0) or 0))
                + "; top_share="
                + str(metric.get("top_stock_share", ""))
            )
        else:
            sample["sample_stock_concentration_note"] = ""
        selected.append(sample)
        if sum(len(item) for item in selected) >= max_rows:
            break
    if not selected:
        return pd.DataFrame(columns=PLAN_COLUMNS)
    plan = pd.concat(selected, ignore_index=True).head(max_rows)
    for col in PLAN_COLUMNS:
        if col not in plan:
            plan[col] = ""
    plan["code"] = plan["code"].astype(str).str.zfill(6)
    return plan.reindex(columns=PLAN_COLUMNS)


def build_micro48_plan(plan: pd.DataFrame) -> pd.DataFrame:
    """Select 4 rows, one per target block, for a 4 rows x 2 modes x 6 variants = 48-card shard."""
    if plan.empty:
        return pd.DataFrame(columns=PLAN_COLUMNS)
    priority_by_block = {
        "H2023_2": [
            "counter_news_opportunity_peer_weak_or_fin_missing_v1",
            "counter_news_risk_peer_weak_v1",
            "control_quality_news_available_low_risk_v1",
        ],
        "H2024_2": [
            "potential_active_clean_context_v1",
            "control_quality_news_available_low_risk_v1",
            "counter_news_risk_peer_weak_v1",
        ],
        "H2025_1": [
            "potential_active_news_financial_confirmed_v1",
            "potential_active_clean_context_v1",
            "control_routine_official_low_signal_v1",
        ],
        "H2026_1": [
            "potential_active_news_financial_confirmed_v1",
            "potential_active_clean_context_v1",
            "counter_high_momentum_with_info_gaps_v1",
        ],
    }
    rows = []
    used_codes: set[str] = set()
    for block in TARGET_BLOCKS:
        block_rows = plan[plan["time_block"].astype(str).eq(block)].copy()
        if block_rows.empty:
            continue
        chosen = None
        for rule in priority_by_block.get(block, []):
            candidates = block_rows[block_rows["candidate_rule"].astype(str).eq(rule)]
            candidates = candidates[~candidates["code"].astype(str).isin(used_codes)]
            if not candidates.empty:
                chosen = candidates.iloc[0]
                break
        if chosen is None:
            candidates = block_rows[~block_rows["code"].astype(str).isin(used_codes)]
            chosen = candidates.iloc[0] if not candidates.empty else block_rows.iloc[0]
        rows.append(chosen)
        used_codes.add(str(chosen["code"]))
    if not rows:
        return pd.DataFrame(columns=PLAN_COLUMNS)
    micro = pd.DataFrame(rows).reindex(columns=PLAN_COLUMNS)
    micro["code"] = micro["code"].astype(str).str.zfill(6)
    return micro


def render_report(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    plan: pd.DataFrame,
    micro: pd.DataFrame,
    metrics_path: Path,
    plan_path: Path,
    micro_path: Path,
) -> str:
    variants_4 = "no_news,keyword_only,no_financial_report_channel,news_plus_financial_report_guarded"
    variants_6 = "no_news,keyword_only,uncertainty_only_questionnaire,quality_only_questionnaire,no_financial_report_channel,news_plus_financial_report_guarded"
    lines = [
        "# Balanced News Ablation Sample Plan V1",
        "",
        "本报告只做研究辅助，不自动交易，不接券商接口，不输出投资指令。",
        "",
        "## Purpose",
        "",
        "上一轮 backfill Flash micro 全部 `exposure_cards=0`，主要比较防守路径。此计划刻意混入 potential_active / quality_control / counterevidence 三类样本，避免下一轮继续只测全员防守。",
        "",
        "## Outputs",
        "",
        f"- metrics: `{metrics_path}`",
        f"- sample_plan: `{plan_path}`",
        f"- micro48_plan: `{micro_path}`",
        f"- evaluated_rows: `{len(frame)}`",
        f"- plan_rows: `{len(plan)}`",
        f"- unique_codes: `{plan['code'].nunique() if not plan.empty else 0}`",
        f"- time_blocks: `{','.join(sorted(plan['time_block'].dropna().astype(str).unique())) if not plan.empty else ''}`",
        "",
        "## Rule Metrics",
        "",
        _table(metrics),
        "",
        "## Sample Plan Block Counts",
        "",
        _table(plan.groupby(["candidate_rule", "time_block"]).size().reset_index(name="rows")) if not plan.empty else "_empty_",
        "",
        "## Micro48 Plan",
        "",
        "该子计划固定 4 行、4 个时间块各 1 行；使用 6 个变体、双任务时正好生成 48 张卡，避免 `--sample-plan-per-rule 1` 偏向最早时间块。",
        "",
        _table(micro[["candidate_rule", "time_block", "date", "code", "name", "reason_to_test"]]) if not micro.empty else "_empty_",
        "",
        "## Suggested Low-Cost DS Shards",
        "",
        f"- Preferred micro shard: use `{micro_path}` with `--variants {variants_6}` -> `4 rows * 2 modes * 6 variants = 48` cards.",
        f"- 4-variant wider shard: use the full sample plan with `--sample-plan-per-rule 1 --variants {variants_4}`, but check block balance before DS.",
        "- Always run dry-run and leakage audit before `--call-deepseek`.",
        "- Stop rule: if the next shard again has `increase_cards=0`, do not scale DS; revisit sample construction and portfolio replay first.",
        "",
        "## Boundary",
        "",
        "- Sample plan excludes `return_20d`, `gt_status`, `gt_pass` and other future/result fields.",
        "- Rule metrics use future returns only for offline validation and are not sent to DeepSeek.",
        "- Potential-active samples are not recommendations; they are controlled probes for Agent behavior.",
    ]
    return "\n".join(lines) + "\n"


def _metric_row(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "unique_stocks": 0,
            "top_stock_share": None,
            "avg_return_20d": None,
            "median_return_20d": None,
            "positive_20d_rate": None,
            "loss_20d_over_5_rate": None,
        }
    returns = pd.to_numeric(frame["return_20d"], errors="coerce").dropna()
    code_counts = frame["code"].astype(str).value_counts()
    return {
        "rows": int(len(frame)),
        "unique_stocks": int(frame["code"].astype(str).nunique()),
        "top_stock_share": _round(code_counts.iloc[0] / len(frame)) if not code_counts.empty else None,
        "avg_return_20d": _round(returns.mean()),
        "median_return_20d": _round(returns.median()),
        "positive_20d_rate": _round((returns > 0).mean()),
        "loss_20d_over_5_rate": _round((returns <= -5).mean()),
    }


def _next_action(rule: BalancedRule, subset: pd.DataFrame) -> str:
    if subset.empty or subset["code"].nunique() < 10:
        return "insufficient; use only if needed for dry-run shape, not DS scaling"
    if rule.rule_type == "potential_active":
        return "include as balanced behavior probe; not a positive rule"
    if rule.rule_type == "quality_control":
        return "include to test uncertainty/quality questionnaire without positive alpha"
    return "include as counter-evidence/control; not positive alpha"


def _round(value: Any, digits: int = 4) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return round(number, digits)


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    return frame.to_markdown(index=False)


if __name__ == "__main__":
    main()
