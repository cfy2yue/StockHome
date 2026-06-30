"""Build ICML-style capability tables for the user-facing stock agent.

This report is intentionally evaluation-facing: it compares the actual product
paths users asked for rather than only internal research labels.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_STRICT_PREFIX = "user_capability_backtest_strict_unseen180_v1"
DEFAULT_HASH200_PREFIX = "user_capability_backtest_v3"
DEFAULT_DS_FLASH_PREFIX = "user_capability_ds_audit_flash_v1"
DEFAULT_DS_PRO_PREFIX = "user_capability_ds_audit_pro_v1"
DEFAULT_CANDIDATE_FLASH_PREFIX = "candidate_comparison_rankavg_operation_protocol_flash_v1"
DEFAULT_CANDIDATE_PRO_PREFIX = "candidate_comparison_rankavg_pro_v2_operation_confirm_merged_v1"
DEFAULT_CANDIDATE_ABLATION_FLASH_PREFIX = "candidate_comparison_rankavg_operation_protocol_ablation_flash_v1"
DEFAULT_CANDIDATE_ABLATION_PRO_PREFIX = "candidate_comparison_rankavg_operation_protocol_ablation_pro_merged_v1"
DEFAULT_WORKFLOW_P1_AGG = REPORT_DIR / "candidate_comparison_workflow_v2_rankavg_cross_aggregate.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final user capability comparison tables.")
    parser.add_argument("--strict-prefix", default=DEFAULT_STRICT_PREFIX)
    parser.add_argument("--hash200-prefix", default=DEFAULT_HASH200_PREFIX)
    parser.add_argument("--p0-task-label", default="P0 单支盯盘 100股 x 3 panels")
    parser.add_argument("--p0-holdout-label", default="strict_unseen_eligible183")
    parser.add_argument("--p1-primary-task-label", default="P1 180候选池 -> 行业分散Top12 -> 单支逻辑")
    parser.add_argument("--p1-primary-holdout-label", default="strict_unseen_eligible183")
    parser.add_argument("--p1-secondary-task-label", default="P1 200候选池 -> 行业分散Top12 -> 单支逻辑")
    parser.add_argument("--p1-secondary-holdout-label", default="hash_holdout_all495")
    parser.add_argument("--ds-flash-prefix", default=DEFAULT_DS_FLASH_PREFIX)
    parser.add_argument("--ds-pro-prefix", default=DEFAULT_DS_PRO_PREFIX)
    parser.add_argument("--candidate-flash-prefix", default=DEFAULT_CANDIDATE_FLASH_PREFIX)
    parser.add_argument("--candidate-pro-prefix", default=DEFAULT_CANDIDATE_PRO_PREFIX)
    parser.add_argument("--candidate-ablation-flash-prefix", default=DEFAULT_CANDIDATE_ABLATION_FLASH_PREFIX)
    parser.add_argument("--candidate-ablation-pro-prefix", default=DEFAULT_CANDIDATE_ABLATION_PRO_PREFIX)
    parser.add_argument("--candidate-workflow-aggregate", type=Path, default=DEFAULT_WORKFLOW_P1_AGG)
    parser.add_argument("--output-prefix", default="user_capability_final_tables_v1")
    parser.add_argument(
        "--deepseek-status-note",
        default="Flash/Pro rows reuse completed real DeepSeek runs; no fresh status check was recorded in this table build.",
    )
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    p0_table = build_p0_table(args.strict_prefix, task_label=args.p0_task_label, holdout=args.p0_holdout_label)
    p1_table = build_p1_table(
        args.strict_prefix,
        args.hash200_prefix,
        primary_task_label=args.p1_primary_task_label,
        primary_holdout=args.p1_primary_holdout_label,
        secondary_task_label=args.p1_secondary_task_label,
        secondary_holdout=args.p1_secondary_holdout_label,
    )
    selection_table = build_candidate_selection_table(
        args.strict_prefix,
        args.hash200_prefix,
        primary_task_label=args.p1_primary_task_label.replace(" -> 单支逻辑", "筛选前置质量"),
        primary_holdout=args.p1_primary_holdout_label,
        secondary_task_label=args.p1_secondary_task_label.replace(" -> 单支逻辑", "筛选前置质量"),
        secondary_holdout=args.p1_secondary_holdout_label,
    )
    ds_single_table = build_ds_single_table(args.ds_flash_prefix, args.ds_pro_prefix)
    candidate_agent_table = build_candidate_agent_table(args.candidate_flash_prefix, args.candidate_pro_prefix)
    candidate_ablation_table = build_candidate_ablation_table(
        [
            ("DS V4 Flash", args.candidate_ablation_flash_prefix),
            ("DS V4 Pro", args.candidate_ablation_pro_prefix),
        ]
    )
    candidate_workflow_table = build_candidate_workflow_baseline_table(args.candidate_workflow_aggregate)

    prefix = safe_prefix(args.output_prefix)
    paths = {
        "p0": REPORT_DIR / f"{prefix}_p0_single_stock.csv",
        "p1": REPORT_DIR / f"{prefix}_p1_candidate_then_watch.csv",
        "selection": REPORT_DIR / f"{prefix}_candidate_selection.csv",
        "ds_single": REPORT_DIR / f"{prefix}_ds_single_ablation.csv",
        "candidate_agent": REPORT_DIR / f"{prefix}_candidate_agent_flash_pro.csv",
        "candidate_ablation": REPORT_DIR / f"{prefix}_candidate_agent_ablation_flash_pro.csv",
        "candidate_workflow": REPORT_DIR / f"{prefix}_candidate_workflow_baselines.csv",
        "report": REPORT_DIR / f"{prefix}.md",
        "latex": REPORT_DIR / f"{prefix}_latex_tables.tex",
    }
    for key, frame in [
        ("p0", p0_table),
        ("p1", p1_table),
        ("selection", selection_table),
        ("ds_single", ds_single_table),
        ("candidate_agent", candidate_agent_table),
        ("candidate_ablation", candidate_ablation_table),
        ("candidate_workflow", candidate_workflow_table),
    ]:
        frame.to_csv(paths[key], index=False, encoding="utf-8-sig")
    paths["report"].write_text(
        render_report(
            p0_table=p0_table,
            p1_table=p1_table,
            selection_table=selection_table,
            ds_single_table=ds_single_table,
            candidate_agent_table=candidate_agent_table,
            candidate_ablation_table=candidate_ablation_table,
            candidate_workflow_table=candidate_workflow_table,
            paths=paths,
            report_name=prefix,
            deepseek_status_note=args.deepseek_status_note,
        ),
        encoding="utf-8",
    )
    paths["latex"].write_text(
        render_latex_tables(
            p0_table=p0_table,
            p1_table=p1_table,
            ds_single_table=ds_single_table,
            candidate_agent_table=candidate_agent_table,
            candidate_ablation_table=candidate_ablation_table,
            candidate_workflow_table=candidate_workflow_table,
        ),
        encoding="utf-8",
    )
    print("A股研究Agent")
    print(f"wrote: {paths['report']}")


def build_p0_table(strict_prefix: str, *, task_label: str, holdout: str) -> pd.DataFrame:
    summary = read_csv(REPORT_DIR / f"{strict_prefix}_single_stock_summary.csv")
    if summary.empty:
        return pd.DataFrame()
    out = aggregate_user_summary(
        summary,
        task_label=task_label,
        holdout=holdout,
    )
    return out


def build_p1_table(
    strict_prefix: str,
    hash200_prefix: str,
    *,
    primary_task_label: str,
    primary_holdout: str,
    secondary_task_label: str,
    secondary_holdout: str,
) -> pd.DataFrame:
    rows = []
    seen_prefixes: set[str] = set()
    for prefix, task_label, holdout in [
        (strict_prefix, primary_task_label, primary_holdout),
        (hash200_prefix, secondary_task_label, secondary_holdout),
    ]:
        if not prefix or prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        summary = read_csv(REPORT_DIR / f"{prefix}_candidate_then_watch_summary.csv")
        if not summary.empty:
            rows.append(aggregate_user_summary(summary, task_label=task_label, holdout=holdout))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_candidate_selection_table(
    strict_prefix: str,
    hash200_prefix: str,
    *,
    primary_task_label: str,
    primary_holdout: str,
    secondary_task_label: str,
    secondary_holdout: str,
) -> pd.DataFrame:
    rows = []
    seen_prefixes: set[str] = set()
    for prefix, task_label, holdout in [
        (strict_prefix, primary_task_label, primary_holdout),
        (hash200_prefix, secondary_task_label, secondary_holdout),
    ]:
        if not prefix or prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        detail = read_csv(REPORT_DIR / f"{prefix}_candidate_selection_detail.csv")
        if detail.empty:
            continue
        grouped = (
            detail.groupby(["period", "decision_frequency"], dropna=False)
            .agg(
                panels=("panel_id", "nunique"),
                decision_dates=("date", "nunique"),
                selected_count=("selected_count", "mean"),
                selected_industries=("selected_industries", "mean"),
                pool_pos=("pool_positive_20d_rate", "mean"),
                selected_pos=("selected_positive_20d_rate", "mean"),
                selected_pos_std=("selected_positive_20d_rate", "std"),
                pool_avg=("pool_avg_return_20d", "mean"),
                selected_avg=("selected_avg_return_20d", "mean"),
                selected_avg_std=("selected_avg_return_20d", "std"),
                selected_excess=("selected_excess_vs_pool", "mean"),
                selected_excess_std=("selected_excess_vs_pool", "std"),
            )
            .reset_index()
        )
        grouped.insert(0, "task", task_label)
        grouped.insert(1, "holdout", holdout)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True).round(6) if rows else pd.DataFrame()


def aggregate_user_summary(summary: pd.DataFrame, *, task_label: str, holdout: str) -> pd.DataFrame:
    grouped = (
        summary.groupby(["period", "decision_frequency"], dropna=False)
        .agg(
            panels=("panel_id", "nunique"),
            decision_count=("decision_count", "mean"),
            active_rate=("active_decision_rate", "mean"),
            active_pos=("active_strategy_positive_20d_rate", "mean"),
            active_pos_std=("active_strategy_positive_20d_rate", "std"),
            active_avg=("active_strategy_avg_return_20d", "mean"),
            active_avg_std=("active_strategy_avg_return_20d", "std"),
            active_excess=("active_excess_avg_return_vs_hold", "mean"),
            active_excess_std=("active_excess_avg_return_vs_hold", "std"),
            strategy_pos=("strategy_positive_20d_rate", "mean"),
            strategy_pos_std=("strategy_positive_20d_rate", "std"),
            strategy_avg=("strategy_avg_return_20d", "mean"),
            strategy_avg_std=("strategy_avg_return_20d", "std"),
            hold_pos=("hold_positive_20d_rate", "mean"),
            hold_avg=("hold_avg_return_20d", "mean"),
            excess_vs_hold=("excess_avg_return_vs_hold", "mean"),
            excess_vs_hold_std=("excess_avg_return_vs_hold", "std"),
            loss_gt5=("strategy_loss_gt5_rate", "mean"),
            avg_target_position=("avg_target_position", "mean"),
        )
        .reset_index()
    )
    grouped.insert(0, "task", task_label)
    grouped.insert(1, "holdout", holdout)
    return grouped.round(6)


def build_ds_single_table(flash_prefix: str, pro_prefix: str) -> pd.DataFrame:
    rows = []
    for model_label, prefix in [("DS V4 Flash", flash_prefix), ("DS V4 Pro", pro_prefix)]:
        metrics = read_csv(REPORT_DIR / f"{prefix}_metrics.csv")
        if metrics.empty:
            continue
        metrics = metrics.copy()
        metrics.insert(0, "model", model_label)
        metrics["source_prefix"] = prefix
        full_cash = _variant_value(metrics, "full_agent", "cash_adjusted_avg_return_20d")
        full_pos = _variant_value(metrics, "full_agent", "cash_adjusted_positive_20d_rate")
        metrics["delta_cash_avg_vs_full"] = pd.to_numeric(metrics["cash_adjusted_avg_return_20d"], errors="coerce") - full_cash
        metrics["delta_pos_rate_vs_full"] = pd.to_numeric(metrics["cash_adjusted_positive_20d_rate"], errors="coerce") - full_pos
        keep = [
            "model",
            "variant",
            "task_mode",
            "decision_cards",
            "invalid_outputs",
            "schema_pass_rate",
            "exposure_cards",
            "active_exposure",
            "cash_adjusted_avg_return_20d",
            "cash_adjusted_positive_20d_rate",
            "cash_adjusted_std_return_20d",
            "delta_cash_avg_vs_full",
            "delta_pos_rate_vs_full",
            "data_missing_flag_cards",
            "source_prefix",
        ]
        rows.append(metrics[[col for col in keep if col in metrics]].copy())
    return pd.concat(rows, ignore_index=True).round(6) if rows else pd.DataFrame()


def build_candidate_agent_table(flash_prefix: str, pro_prefix: str) -> pd.DataFrame:
    rows = []
    for model_label, prefix in [("DS V4 Flash", flash_prefix), ("DS V4 Pro", pro_prefix)]:
        aggregate = read_csv(REPORT_DIR / f"{prefix}_aggregate.csv")
        if aggregate.empty:
            continue
        aggregate = aggregate.copy()
        aggregate.insert(0, "model", model_label)
        aggregate["source_prefix"] = prefix
        rows.append(aggregate)
    return pd.concat(rows, ignore_index=True).round(6) if rows else pd.DataFrame()


def build_candidate_ablation_table(model_prefixes: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for model_label, prefix in model_prefixes:
        aggregate = read_csv(REPORT_DIR / f"{prefix}_aggregate.csv")
        if aggregate.empty:
            continue
        aggregate = aggregate.copy()
        aggregate.insert(0, "model", model_label)
        aggregate["source_prefix"] = prefix
        rows.append(aggregate)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    for (model, scenario), group in out.groupby(["model", "comparison_scenario"], dropna=False):
        full = group[group["variant"].astype(str).eq("ranker_anchor_agent")]
        if full.empty:
            continue
        full_top1 = float(pd.to_numeric(full["top1_excess_mean"], errors="coerce").iloc[0])
        full_top2 = float(pd.to_numeric(full["top2_excess_mean"], errors="coerce").iloc[0])
        mask = out["model"].astype(str).eq(str(model)) & out["comparison_scenario"].astype(str).eq(str(scenario))
        out.loc[mask, "delta_top1_excess_vs_anchor"] = pd.to_numeric(out.loc[mask, "top1_excess_mean"], errors="coerce") - full_top1
        out.loc[mask, "delta_top2_excess_vs_anchor"] = pd.to_numeric(out.loc[mask, "top2_excess_mean"], errors="coerce") - full_top2
    return out.round(6)


def build_candidate_workflow_baseline_table(path: Path) -> pd.DataFrame:
    frame = read_csv(path)
    if frame.empty:
        return frame
    keep_scores = {"equal_or_random_baseline", "p1_default_selector_v1", "rev_chip_core", "single_watch_proxy"}
    out = frame[frame["time_block"].astype(str).eq("ALL") & frame["score_name"].astype(str).isin(keep_scores)].copy()
    return out.round(6)


def render_report(
    *,
    p0_table: pd.DataFrame,
    p1_table: pd.DataFrame,
    selection_table: pd.DataFrame,
    ds_single_table: pd.DataFrame,
    candidate_agent_table: pd.DataFrame,
    candidate_ablation_table: pd.DataFrame,
    candidate_workflow_table: pd.DataFrame,
    paths: dict[str, Path],
    report_name: str,
    deepseek_status_note: str,
) -> str:
    p0_primary = compact_user_table(p0_table, task_contains="P0", frequency="every_2_weeks")
    p1_primary = compact_user_table(p1_table, frequency="every_2_weeks")
    selection_primary = compact_selection_table(selection_table, frequency="every_2_weeks")
    ds_primary = compact_ds_single_table(ds_single_table)
    candidate_primary = compact_candidate_agent_table(candidate_agent_table)
    candidate_ablation_primary = compact_candidate_ablation_table(candidate_ablation_table)
    workflow_primary = compact_workflow_table(candidate_workflow_table)
    p0_h2026 = user_metric_sentence(p0_table, task_contains="P0", period="H2026", frequency="every_2_weeks")
    p1_h2026 = user_metric_sentence(p1_table, period="H2026", frequency="every_2_weeks")
    p1_selection_h2026 = selection_metric_sentence(selection_table, period="H2026", frequency="every_2_weeks")
    lines = [
        f"# User Capability Evaluation Tables ({report_name})",
        "",
        "本报告按用户真实能力路径汇总：P0 单支盯盘、P1 候选池筛选后盯盘、P1 2-20 支候选对比，以及 DS Flash/Pro 小样本审计。输出允许给买入/卖出/加减仓/持有/等待等研究辅助型操作建议，但不自动执行、不承诺收益。",
        "",
        "## Main Findings",
        "",
        f"- P0 单支盯盘：{p0_h2026}。当前收益主要来自仓位控制和弱市减损，主动介入胜率仍低于 0.60/0.65 验收线。",
        f"- P1 候选池先筛选再盯盘：{p1_h2026}；前置筛选质量为 {p1_selection_h2026}。它在弱市能改善池均值，但 2024/2025 强市场中低仓位会跑输长期持有。",
        "- DS Flash 单股小样本没有把研究暴露打起来，full_agent 也没有稳定跑赢 ablation；这意味着当前 Agent 层更像审计器/防守器，不是独立 alpha。",
        "- P1 2-20 支候选对比的小面板中，Flash/Pro 主协议 Top1/Top2 超额为正；但 Pro ablation 独立重跑的 anchor 与早先 Pro 主协议不完全一致，说明小样本 LLM 排序存在路径敏感，不能只凭单次 Pro 结果宣称稳健。",
        "- Flash/Pro candidate ablation 已覆盖 no_quant/no_news/no_peer/no_bookskill/no_financial；通道贡献按场景混合，尤其 Pro 下 same-sector 去新闻/去同行明显变差，cross-sector 却有若干 ablation 高于 anchor，下一轮必须做 fresh panels 与固定采样重复。",
        f"- DS 状态：{deepseek_status_note}",
        "",
        "## Table 1. P0 Single-Stock Watch, 100 Stocks x 3 Panels",
        "",
        markdown_table(p0_primary),
        "",
        "## Table 2. P1 Candidate Pool Then Watch",
        "",
        markdown_table(p1_primary),
        "",
        "## Table 3. Candidate Selection Quality Before Watch Logic",
        "",
        markdown_table(selection_primary),
        "",
        "## Table 4. DS Flash/Pro Single-Stock Ablation Audit",
        "",
        markdown_table(ds_primary),
        "",
        "## Table 5. P1 Candidate Comparison Agent Audit",
        "",
        markdown_table(candidate_primary),
        "",
        "## Table 6. P1 Candidate Workflow Large-Sample Baselines",
        "",
        markdown_table(workflow_primary),
        "",
        "## Table 7. P1 Candidate Comparison Flash/Pro Ablation",
        "",
        markdown_table(candidate_ablation_primary),
        "",
        "## Metric Notes",
        "",
        "- `active_pos` / `active_avg` 只统计目标仓位 >=35% 的有效介入，优先看这个判断买/加/持能力。",
        "- `strategy_pos` / `strategy_avg` 是含现金仓位后的用户操作路径结果；低仓位会提高防守稳定性，但不能直接解释为高 alpha。",
        "- baseline rows are marked by method names such as `equal_or_random_baseline`, `hold_pos`, and `hold_avg`.",
        "- holdout 标签来自输入表：若历史 DS/sample artifact 已覆盖过多股票，报告会诚实标为 hash holdout 或 prior-DS-exclusion-insufficient，而不是宣称严格未见。",
        "- Table 5 的 Pro 主协议与 Table 7 的 Pro ablation anchor 是两次独立调用；若二者差异较大，应解释为模型输出路径敏感，后续需要固定采样、多 seed/多 panel 和确定性 ranker 约束。",
        "- Candidate ablation 中 `delta_*_vs_anchor` 为同一模型、同一场景下移除对应通道后相对 anchor full protocol 的变化；正值表示该 ablation 在小面板中更高，不代表应删除通道，需结合 fresh panel 和通道解释价值。",
        "",
        "## Artifacts",
        "",
        *(f"- `{path}`" for path in paths.values()),
        "",
    ]
    return "\n".join(lines)


def user_metric_sentence(
    frame: pd.DataFrame,
    *,
    period: str,
    frequency: str,
    task_contains: str | None = None,
) -> str:
    if frame.empty:
        return "no table rows"
    subset = frame[
        frame["period"].astype(str).eq(period)
        & frame["decision_frequency"].astype(str).eq(frequency)
    ].copy()
    if task_contains:
        subset = subset[subset["task"].astype(str).str.contains(task_contains, regex=False)].copy()
    if subset.empty:
        return f"{period}/{frequency} no rows"
    row = subset.iloc[0]
    return (
        f"{period}/{frequency} active_pos={fmt_pm(row.get('active_pos'), row.get('active_pos_std'))}, "
        f"active_avg={fmt_pm(row.get('active_avg'), row.get('active_avg_std'))}pp, "
        f"strategy_avg={num_text(row.get('strategy_avg'))}pp vs hold_avg={num_text(row.get('hold_avg'))}pp, "
        f"excess_vs_hold={num_text(row.get('excess_vs_hold'))}pp"
    )


def selection_metric_sentence(frame: pd.DataFrame, *, period: str, frequency: str) -> str:
    if frame.empty:
        return "no selection rows"
    subset = frame[
        frame["period"].astype(str).eq(period)
        & frame["decision_frequency"].astype(str).eq(frequency)
    ].copy()
    if subset.empty:
        return f"{period}/{frequency} no rows"
    row = subset.iloc[0]
    return (
        f"selected_pos={fmt_pm(row.get('selected_pos'), row.get('selected_pos_std'))}, "
        f"selected_excess={fmt_pm(row.get('selected_excess'), row.get('selected_excess_std'))}pp"
    )


def render_latex_tables(
    *,
    p0_table: pd.DataFrame,
    p1_table: pd.DataFrame,
    ds_single_table: pd.DataFrame,
    candidate_agent_table: pd.DataFrame,
    candidate_ablation_table: pd.DataFrame,
    candidate_workflow_table: pd.DataFrame,
) -> str:
    """Render compact booktabs-style tables for paper/report paste-in."""
    blocks = [
        "% Auto-generated by scripts/build_user_capability_final_tables.py",
        "% Requires \\usepackage{booktabs}. Chinese labels may require XeLaTeX/CJK support.",
        "",
    ]
    specs = [
        ("P0 single-stock strict-unseen key results", "tab:p0_single_stock", compact_user_table(p0_table, task_contains="P0", frequency="every_2_weeks")),
        ("P1 candidate pool then watch key results", "tab:p1_candidate_watch", compact_user_table(p1_table, frequency="every_2_weeks")),
        ("DeepSeek Flash/Pro single-stock ablation", "tab:ds_single_ablation", compact_ds_single_table(ds_single_table)),
        ("P1 candidate comparison Flash/Pro main protocol", "tab:candidate_agent_main", compact_candidate_agent_table(candidate_agent_table)),
        ("P1 candidate comparison Flash/Pro ablation", "tab:candidate_agent_ablation", compact_candidate_ablation_table(candidate_ablation_table)),
        ("P1 large-sample deterministic baselines", "tab:candidate_workflow_baselines", compact_workflow_table(candidate_workflow_table)),
    ]
    for caption, label, frame in specs:
        if frame.empty:
            continue
        blocks.append(latex_table(frame, caption=caption, label=label))
        blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"


def latex_table(frame: pd.DataFrame, *, caption: str, label: str, max_rows: int = 40) -> str:
    shown = frame.head(max_rows).copy()
    try:
        body = shown.to_latex(index=False, escape=False, caption=caption, label=label)
    except Exception:
        body = shown.to_csv(index=False)
    return body


def compact_user_table(frame: pd.DataFrame, *, frequency: str, task_contains: str | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame[frame["decision_frequency"].astype(str).eq(frequency)].copy()
    if task_contains:
        out = out[out["task"].astype(str).str.contains(task_contains, regex=False)].copy()
    out = out[
        [
            "task",
            "holdout",
            "period",
            "decision_frequency",
            "panels",
            "active_rate",
            "active_pos",
            "active_pos_std",
            "active_avg",
            "active_avg_std",
            "active_excess",
            "strategy_pos",
            "strategy_avg",
            "hold_pos",
            "hold_avg",
            "excess_vs_hold",
            "loss_gt5",
            "avg_target_position",
        ]
    ].copy()
    for mean_col, std_col in [
        ("active_pos", "active_pos_std"),
        ("active_avg", "active_avg_std"),
    ]:
        out[mean_col + "_mean±std"] = out.apply(lambda row: fmt_pm(row.get(mean_col), row.get(std_col)), axis=1)
    return out.drop(columns=["active_pos_std", "active_avg_std"]).round(4)


def compact_selection_table(frame: pd.DataFrame, *, frequency: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame[frame["decision_frequency"].astype(str).eq(frequency)].copy()
    cols = [
        "task",
        "holdout",
        "period",
        "decision_dates",
        "selected_count",
        "selected_industries",
        "pool_pos",
        "selected_pos",
        "selected_avg",
        "pool_avg",
        "selected_excess",
        "selected_excess_std",
    ]
    out = out[[col for col in cols if col in out]].copy()
    if "selected_excess_std" in out:
        out["selected_excess_mean±std"] = out.apply(lambda row: fmt_pm(row.get("selected_excess"), row.get("selected_excess_std")), axis=1)
        out = out.drop(columns=["selected_excess_std"])
    return out.round(4)


def compact_ds_single_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    cols = [
        "model",
        "variant",
        "decision_cards",
        "invalid_outputs",
        "schema_pass_rate",
        "exposure_cards",
        "active_exposure",
        "cash_adjusted_avg_return_20d",
        "cash_adjusted_positive_20d_rate",
        "delta_cash_avg_vs_full",
        "delta_pos_rate_vs_full",
        "data_missing_flag_cards",
    ]
    return frame[[col for col in cols if col in frame]].round(4)


def compact_candidate_agent_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    cols = [
        "model",
        "variant",
        "comparison_scenario",
        "cards",
        "top1_excess_mean",
        "top2_excess_mean",
        "top1_positive_rate",
        "top2_positive_rate",
        "top1_worst_rate",
        "regret_mean",
        "avg_confidence",
        "source_prefix",
    ]
    return frame[[col for col in cols if col in frame]].round(4)


def compact_workflow_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    cols = [
        "comparison_scenario",
        "score_name",
        "n_groups",
        "mean_rank_ic",
        "rank_ic_positive_rate",
        "top1_excess_mean",
        "top2_excess_mean",
        "top1_positive_rate",
        "top2_positive_rate",
        "top1_worst_rate",
        "regret_mean",
    ]
    return frame[[col for col in cols if col in frame]].round(4)


def compact_candidate_ablation_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    cols = [
        "model",
        "variant",
        "comparison_scenario",
        "cards",
        "top1_excess_mean",
        "top2_excess_mean",
        "top1_positive_rate",
        "top2_positive_rate",
        "top1_worst_rate",
        "delta_top1_excess_vs_anchor",
        "delta_top2_excess_vs_anchor",
        "avg_confidence",
    ]
    return frame[[col for col in cols if col in frame]].round(4)


def _variant_value(frame: pd.DataFrame, variant: str, column: str) -> float:
    subset = frame[frame["variant"].astype(str).eq(variant)]
    if subset.empty or column not in subset:
        return float("nan")
    return float(pd.to_numeric(subset[column], errors="coerce").iloc[0])


def fmt_pm(mean: Any, std: Any, digits: int = 4) -> str:
    mean_num = pd.to_numeric(pd.Series([mean]), errors="coerce").iloc[0]
    std_num = pd.to_numeric(pd.Series([std]), errors="coerce").iloc[0]
    if pd.isna(mean_num):
        return "NA"
    if pd.isna(std_num):
        return f"{mean_num:.{digits}f}"
    return f"{mean_num:.{digits}f}±{std_num:.{digits}f}"


def num_text(value: Any, digits: int = 4) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "NA"
    return f"{number:.{digits}f}"


def markdown_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame.empty:
        return "_empty_"
    try:
        return frame.head(max_rows).to_markdown(index=False)
    except Exception:
        return frame.head(max_rows).to_csv(index=False)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")


def safe_prefix(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value)).strip("_")
    return safe or "user_capability_final_tables_v1"


if __name__ == "__main__":
    main()
