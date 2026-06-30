"""Stability audit for P1 candidate-comparison rankers.

This is a local/offline audit. It uses forward returns only for evaluation and
never writes a DeepSeek evidence pack. The goal is to decide whether the P1
ranker is stable enough to justify more DS Flash/Pro panels.
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

from scripts.audit_candidate_comparison_workflow_v1 import (  # noqa: E402
    P1_DEFAULT_SCORE,
    SCORE_COLUMNS,
    AuditConfig,
    aggregate_metrics,
    build_candidate_groups,
    evaluate_groups,
    ensure_task_default_score,
    load_candidate_frame,
)


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_PREFIX = "candidate_comparison_stability_v1"
DEFAULT_FREQUENCIES = ["every_2_weeks", "weekly_friday", "weekly_tuesday"]
PRIMARY_SCORES = [
    P1_DEFAULT_SCORE,
    "rank_avg_rev_watch",
    "rev_chip_core",
    "single_watch_proxy",
    "candidate_context_blend_v1",
]


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = safe_prefix(args.output_prefix)
    frequencies = [item.strip() for item in args.frequencies.split(",") if item.strip()]
    score_names = [score for score in PRIMARY_SCORES if score in SCORE_COLUMNS]

    base_frame = load_candidate_frame()
    all_candidate_rows: list[pd.DataFrame] = []
    all_detail: list[pd.DataFrame] = []
    all_aggregate: list[pd.DataFrame] = []
    all_gap: list[pd.DataFrame] = []

    for frequency in frequencies:
        cfg = AuditConfig(
            candidate_size=args.candidate_size,
            repeats=args.panels * args.repeats_per_panel,
            industries_per_date=args.industries_per_date,
            decision_frequency=frequency,
            min_industry_size=args.min_industry_size,
            max_dates_per_block=args.max_dates_per_block,
            output_prefix=prefix,
        )
        candidate_rows = build_candidate_groups(base_frame, cfg)
        if candidate_rows.empty:
            continue
        candidate_rows = candidate_rows.copy()
        candidate_rows["decision_frequency"] = frequency
        candidate_rows["sample_panel_id"] = candidate_rows["repeat_seed"].map(
            lambda seed: f"panel_{(int(seed) % args.panels) + 1:02d}"
        )
        detail, aggregate = evaluate_groups(candidate_rows)
        if detail.empty:
            continue
        panel_lookup = candidate_rows[["comparison_group_id", "sample_panel_id", "decision_frequency"]].drop_duplicates()
        detail = detail.merge(panel_lookup, on="comparison_group_id", how="left")
        detail["decision_frequency"] = detail["decision_frequency"].fillna(frequency)
        aggregate.insert(0, "decision_frequency", frequency)
        candidate_rows = ensure_task_default_score(candidate_rows)
        gap = score_gap_detail(candidate_rows, score_names=score_names)
        all_candidate_rows.append(candidate_rows)
        all_detail.append(detail)
        all_aggregate.append(aggregate)
        all_gap.append(gap)

    if not all_detail:
        raise SystemExit("no candidate comparison detail generated")

    candidate_rows_all = pd.concat(all_candidate_rows, ignore_index=True)
    detail_all = pd.concat(all_detail, ignore_index=True)
    aggregate_all = pd.concat(all_aggregate, ignore_index=True)
    gap_all = pd.concat(all_gap, ignore_index=True) if all_gap else pd.DataFrame()

    panel_metrics = panel_stability_metrics(detail_all)
    score_contrasts = paired_score_contrasts(detail_all, baseline=P1_DEFAULT_SCORE)
    gate_summary = build_gate_summary(panel_metrics, score_contrasts, gap_all)

    paths = {
        "candidate_rows_eval": REPORT_DIR / f"{prefix}_candidate_rows_eval.csv",
        "detail": REPORT_DIR / f"{prefix}_detail.csv",
        "aggregate": REPORT_DIR / f"{prefix}_aggregate.csv",
        "panel_metrics": REPORT_DIR / f"{prefix}_panel_metrics.csv",
        "score_contrasts": REPORT_DIR / f"{prefix}_score_contrasts.csv",
        "score_gap": REPORT_DIR / f"{prefix}_score_gap.csv",
        "gate_summary": REPORT_DIR / f"{prefix}_gate_summary.csv",
        "report": REPORT_DIR / f"{prefix}.md",
    }
    candidate_rows_all.to_csv(paths["candidate_rows_eval"], index=False, encoding="utf-8-sig")
    detail_all.to_csv(paths["detail"], index=False, encoding="utf-8-sig")
    aggregate_all.to_csv(paths["aggregate"], index=False, encoding="utf-8-sig")
    panel_metrics.to_csv(paths["panel_metrics"], index=False, encoding="utf-8-sig")
    score_contrasts.to_csv(paths["score_contrasts"], index=False, encoding="utf-8-sig")
    gap_all.to_csv(paths["score_gap"], index=False, encoding="utf-8-sig")
    gate_summary.to_csv(paths["gate_summary"], index=False, encoding="utf-8-sig")
    paths["report"].write_text(
        render_report(
            frequencies=frequencies,
            args=args,
            aggregate=aggregate_all,
            panel_metrics=panel_metrics,
            score_contrasts=score_contrasts,
            gate_summary=gate_summary,
            paths=paths,
        ),
        encoding="utf-8",
    )
    print(f"wrote: {paths['report']}")


def panel_stability_metrics(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["decision_frequency", "comparison_scenario", "score_name", "time_block"]
    for key_values, group in detail.groupby(keys, sort=True):
        rows.append(_panel_row(key_values, group))
    for key_values, group in detail.groupby(["decision_frequency", "comparison_scenario", "score_name"], sort=True):
        rows.append(_panel_row((key_values[0], key_values[1], key_values[2], "ALL"), group))
    return pd.DataFrame(rows).sort_values(["decision_frequency", "comparison_scenario", "time_block", "score_name"]).reset_index(drop=True)


def _panel_row(key_values: tuple[Any, ...], group: pd.DataFrame) -> dict[str, Any]:
    frequency, scenario, score_name, block = key_values
    panel_rows = []
    for panel_id, panel in group.groupby("sample_panel_id", sort=True):
        rank_ic = pd.to_numeric(panel["rank_ic"], errors="coerce")
        panel_rows.append(
            {
                "sample_panel_id": panel_id,
                "n_groups": int(panel["comparison_group_id"].nunique()),
                "mean_rank_ic": float(rank_ic.mean()) if rank_ic.notna().any() else np.nan,
                "top1_excess_mean": float(pd.to_numeric(panel["top1_excess_20d"], errors="coerce").mean()),
                "top2_excess_mean": float(pd.to_numeric(panel["top2_excess_20d"], errors="coerce").mean()),
                "top1_positive_rate": float(panel["top1_positive"].astype(bool).mean()),
                "top2_positive_rate": float(pd.to_numeric(panel["top2_positive_rate"], errors="coerce").mean()),
                "top1_worst_rate": float(panel["top1_is_worst"].astype(bool).mean()),
                "regret_mean": float(pd.to_numeric(panel["top1_regret_vs_best"], errors="coerce").mean()),
            }
        )
    panels = pd.DataFrame(panel_rows)
    rank_ic = pd.to_numeric(group["rank_ic"], errors="coerce")
    return {
        "decision_frequency": frequency,
        "comparison_scenario": scenario,
        "score_name": score_name,
        "time_block": block,
        "panels": int(panels["sample_panel_id"].nunique()) if not panels.empty else 0,
        "n_groups": int(group["comparison_group_id"].nunique()),
        "mean_rank_ic": round(float(rank_ic.mean()), 6) if rank_ic.notna().any() else np.nan,
        "rank_ic_positive_rate": round(float((rank_ic.dropna() > 0).mean()), 6) if rank_ic.notna().any() else np.nan,
        "top1_excess_mean": round(float(pd.to_numeric(group["top1_excess_20d"], errors="coerce").mean()), 6),
        "top1_excess_panel_std": round(float(panels["top1_excess_mean"].std()), 6) if len(panels) > 1 else np.nan,
        "top1_excess_panel_min": round(float(panels["top1_excess_mean"].min()), 6) if not panels.empty else np.nan,
        "top2_excess_mean": round(float(pd.to_numeric(group["top2_excess_20d"], errors="coerce").mean()), 6),
        "top2_excess_panel_std": round(float(panels["top2_excess_mean"].std()), 6) if len(panels) > 1 else np.nan,
        "top2_excess_panel_min": round(float(panels["top2_excess_mean"].min()), 6) if not panels.empty else np.nan,
        "top1_positive_rate": round(float(group["top1_positive"].astype(bool).mean()), 6),
        "top2_positive_rate": round(float(pd.to_numeric(group["top2_positive_rate"], errors="coerce").mean()), 6),
        "top1_worst_rate": round(float(group["top1_is_worst"].astype(bool).mean()), 6),
        "regret_mean": round(float(pd.to_numeric(group["top1_regret_vs_best"], errors="coerce").mean()), 6),
    }


def paired_score_contrasts(detail: pd.DataFrame, *, baseline: str) -> pd.DataFrame:
    metric_cols = ["rank_ic", "top1_excess_20d", "top2_excess_20d", "top1_positive", "top1_is_worst", "top1_regret_vs_best"]
    keys = ["decision_frequency", "comparison_group_id"]
    base = detail[detail["score_name"].eq(baseline)][keys + metric_cols + ["comparison_scenario", "time_block", "sample_panel_id"]].copy()
    base = base.rename(columns={col: f"baseline_{col}" for col in metric_cols})
    others = detail[~detail["score_name"].eq(baseline)][keys + ["score_name"] + metric_cols + ["comparison_scenario", "time_block", "sample_panel_id"]].copy()
    merged = others.merge(
        base,
        on=keys,
        how="inner",
        suffixes=("", "_base_meta"),
    )
    rows = []
    for key_values, group in merged.groupby(["decision_frequency", "comparison_scenario", "score_name", "time_block"], sort=True):
        rows.append(_contrast_row(key_values, group))
    for key_values, group in merged.groupby(["decision_frequency", "comparison_scenario", "score_name"], sort=True):
        rows.append(_contrast_row((key_values[0], key_values[1], key_values[2], "ALL"), group))
    return pd.DataFrame(rows).sort_values(["decision_frequency", "comparison_scenario", "time_block", "score_name"]).reset_index(drop=True)


def _contrast_row(key_values: tuple[Any, ...], group: pd.DataFrame) -> dict[str, Any]:
    frequency, scenario, score_name, block = key_values
    top1_delta = pd.to_numeric(group["top1_excess_20d"], errors="coerce") - pd.to_numeric(group["baseline_top1_excess_20d"], errors="coerce")
    top2_delta = pd.to_numeric(group["top2_excess_20d"], errors="coerce") - pd.to_numeric(group["baseline_top2_excess_20d"], errors="coerce")
    regret_delta = pd.to_numeric(group["top1_regret_vs_best"], errors="coerce") - pd.to_numeric(group["baseline_top1_regret_vs_best"], errors="coerce")
    top1_pos_delta = group["top1_positive"].astype(float) - group["baseline_top1_positive"].astype(float)
    worst_delta = group["top1_is_worst"].astype(float) - group["baseline_top1_is_worst"].astype(float)
    return {
        "decision_frequency": frequency,
        "comparison_scenario": scenario,
        "score_name": score_name,
        "baseline_score": P1_DEFAULT_SCORE,
        "time_block": block,
        "n_groups": int(group["comparison_group_id"].nunique()),
        "delta_top1_excess_mean": round(float(top1_delta.mean()), 6),
        "delta_top2_excess_mean": round(float(top2_delta.mean()), 6),
        "delta_top1_positive_rate": round(float(top1_pos_delta.mean()), 6),
        "delta_top1_worst_rate": round(float(worst_delta.mean()), 6),
        "delta_regret_mean": round(float(regret_delta.mean()), 6),
        "beats_baseline_top1_rate": round(float((top1_delta > 0).mean()), 6),
        "beats_baseline_top2_rate": round(float((top2_delta > 0).mean()), 6),
    }


def score_gap_detail(candidate_rows: pd.DataFrame, *, score_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_id, group in candidate_rows.groupby("comparison_group_id", sort=True):
        meta = group.iloc[0]
        for score_name in score_names:
            if score_name not in group:
                continue
            ranked = group.assign(_score=pd.to_numeric(group[score_name], errors="coerce")).dropna(subset=["_score"])
            if ranked.empty:
                continue
            ranked = ranked.sort_values(["_score", "code"], ascending=[False, True]).reset_index(drop=True)
            top1 = ranked.iloc[0]
            top2 = ranked.iloc[1] if len(ranked) > 1 else ranked.iloc[0]
            gap = float(top1["_score"] - top2["_score"]) if len(ranked) > 1 else np.nan
            rows.append(
                {
                    "decision_frequency": meta["decision_frequency"],
                    "sample_panel_id": meta["sample_panel_id"],
                    "comparison_group_id": group_id,
                    "comparison_scenario": meta["comparison_scenario"],
                    "time_block": meta["time_block"],
                    "date": meta["date"],
                    "score_name": score_name,
                    "top1_code": str(top1["code"]).zfill(6),
                    "top2_code": str(top2["code"]).zfill(6),
                    "score_gap_top1_top2": gap,
                    "score_gap_abs_lt_0_25": bool(abs(gap) < 0.25) if not pd.isna(gap) else False,
                    "top1_return_20d": float(pd.to_numeric(top1["return_20d"], errors="coerce")),
                    "top2_return_20d": float(pd.to_numeric(top2["return_20d"], errors="coerce")),
                }
            )
    return pd.DataFrame(rows)


def build_gate_summary(panel_metrics: pd.DataFrame, score_contrasts: pd.DataFrame, gap: pd.DataFrame) -> pd.DataFrame:
    primary = panel_metrics[
        panel_metrics["score_name"].isin(PRIMARY_SCORES)
        & panel_metrics["time_block"].eq("ALL")
    ].copy()
    if primary.empty:
        return primary
    gap_summary = (
        gap.groupby(["decision_frequency", "comparison_scenario", "score_name"], dropna=False)
        .agg(
            score_gap_mean=("score_gap_top1_top2", "mean"),
            ambiguous_gap_rate=("score_gap_abs_lt_0_25", "mean"),
        )
        .reset_index()
    )
    out = primary.merge(gap_summary, on=["decision_frequency", "comparison_scenario", "score_name"], how="left")
    out["candidate_for_ds_panel"] = (
        pd.to_numeric(out["mean_rank_ic"], errors="coerce").ge(0.03)
        & pd.to_numeric(out["rank_ic_positive_rate"], errors="coerce").ge(0.55)
        & pd.to_numeric(out["top2_excess_mean"], errors="coerce").gt(0.0)
        & pd.to_numeric(out["top2_excess_panel_min"], errors="coerce").gt(-1.0)
        & pd.to_numeric(out["top1_worst_rate"], errors="coerce").le(0.16)
        & pd.to_numeric(out["ambiguous_gap_rate"], errors="coerce").fillna(0.0).le(0.35)
    )
    out["stability_note"] = out.apply(_stability_note, axis=1)
    contrast_all = score_contrasts[score_contrasts["time_block"].eq("ALL")].copy()
    if not contrast_all.empty:
        contrast_keep = contrast_all[
            [
                "decision_frequency",
                "comparison_scenario",
                "score_name",
                "delta_top1_excess_mean",
                "delta_top2_excess_mean",
                "beats_baseline_top1_rate",
                "beats_baseline_top2_rate",
            ]
        ]
        out = out.merge(contrast_keep, on=["decision_frequency", "comparison_scenario", "score_name"], how="left")
    return out.sort_values(["candidate_for_ds_panel", "decision_frequency", "comparison_scenario", "score_name"], ascending=[False, True, True, True]).reset_index(drop=True)


def _stability_note(row: pd.Series) -> str:
    problems = []
    if pd.to_numeric(pd.Series([row.get("mean_rank_ic")]), errors="coerce").iloc[0] < 0.03:
        problems.append("rank_ic_weak")
    if pd.to_numeric(pd.Series([row.get("rank_ic_positive_rate")]), errors="coerce").iloc[0] < 0.55:
        problems.append("rank_ic_hit_rate_weak")
    if pd.to_numeric(pd.Series([row.get("top2_excess_mean")]), errors="coerce").iloc[0] <= 0:
        problems.append("top2_no_excess")
    if pd.to_numeric(pd.Series([row.get("top2_excess_panel_min")]), errors="coerce").iloc[0] <= -1.0:
        problems.append("panel_min_negative")
    if pd.to_numeric(pd.Series([row.get("top1_worst_rate")]), errors="coerce").iloc[0] > 0.16:
        problems.append("worst_rate_high")
    if pd.to_numeric(pd.Series([row.get("ambiguous_gap_rate")]), errors="coerce").iloc[0] > 0.35:
        problems.append("rank_gap_ambiguous")
    return "pass_candidate" if not problems else ";".join(problems)


def render_report(
    *,
    frequencies: list[str],
    args: argparse.Namespace,
    aggregate: pd.DataFrame,
    panel_metrics: pd.DataFrame,
    score_contrasts: pd.DataFrame,
    gate_summary: pd.DataFrame,
    paths: dict[str, Path],
) -> str:
    primary_gate = gate_summary[gate_summary["score_name"].isin([P1_DEFAULT_SCORE, "rank_avg_rev_watch", "rev_chip_core", "single_watch_proxy"])].copy()
    all_primary = panel_metrics[
        panel_metrics["time_block"].eq("ALL")
        & panel_metrics["score_name"].isin(PRIMARY_SCORES)
    ].copy()
    h2026 = panel_metrics[
        panel_metrics["time_block"].eq("H2026_1")
        & panel_metrics["score_name"].isin(PRIMARY_SCORES)
    ].copy()
    contrast_all = score_contrasts[
        score_contrasts["time_block"].eq("ALL")
        & score_contrasts["score_name"].isin(["candidate_context_blend_v1", "rev_chip_core", "single_watch_proxy", "rank_avg_rev_watch"])
    ].copy()
    findings = key_findings(gate_summary, panel_metrics, score_contrasts)
    lines = [
        "# P1 Candidate Comparison Stability Audit v1",
        "",
        "本审计只做本地离线评估：未来 20 日收益只用于评估，不进入 DeepSeek evidence，不生成用户决策卡。",
        "",
        "## Setup",
        "",
        f"- frequencies: `{','.join(frequencies)}`",
        f"- panels: `{args.panels}`",
        f"- repeats_per_panel: `{args.repeats_per_panel}`",
        f"- candidate_size: `{args.candidate_size}`",
        f"- industries_per_date: `{args.industries_per_date}`",
        f"- max_dates_per_block: `{args.max_dates_per_block}` (`0` means all dates after frequency filter)",
        "",
        "## Key Findings",
        "",
        *findings,
        "",
        "## Gate Summary",
        "",
        markdown_table(primary_gate[gate_columns(primary_gate)]),
        "",
        "## ALL Primary Score Stability",
        "",
        markdown_table(all_primary[primary_columns(all_primary)]),
        "",
        "## H2026_1 Primary Score Stability",
        "",
        markdown_table(h2026[primary_columns(h2026)]),
        "",
        "## Paired Score Contrast vs P1 Default",
        "",
        markdown_table(contrast_all[contrast_columns(contrast_all)]),
        "",
        "## Interpretation",
        "",
        "- `candidate_for_ds_panel=True` 只表示底层排序值得进入下一轮 DS fresh-panel 验证，不表示可直接升默认。",
        "- 若 `candidate_context_blend_v1` 相对默认分数为负，说明新闻/财报/非价格 blend 暂时不应作为正向排序升权，只能作为解释或反证。",
        "- 若 `ambiguous_gap_rate` 高，说明 Top1/Top2 分数间距太小，LLM 更容易路径敏感；下一轮应强化 ranker anchor 或只让 Agent 在硬反证下调整。",
        "- DS 下一轮优先挑 `candidate_for_ds_panel=True` 且 H2026 不崩的场景；否则先修本地 ranker/采样，不烧 Pro。",
        "",
        "## Artifacts",
        "",
        *(f"- `{path}`" for path in paths.values()),
        "",
    ]
    return "\n".join(lines)


def key_findings(gate_summary: pd.DataFrame, panel_metrics: pd.DataFrame, score_contrasts: pd.DataFrame) -> list[str]:
    findings: list[str] = []
    candidates = gate_summary[gate_summary.get("candidate_for_ds_panel", pd.Series(dtype=bool)).astype(bool)].copy()
    if candidates.empty:
        findings.append("- 没有场景通过本地 DS panel 候选 gate；下一步应先修 deterministic ranker，不建议烧 DS。")
    else:
        labels = [
            f"{row.decision_frequency}/{row.comparison_scenario}/{row.score_name}"
            for row in candidates.itertuples(index=False)
        ]
        findings.append("- 通过本地 DS panel 候选 gate 的场景：" + "；".join(labels) + "。")
    blend = score_contrasts[
        score_contrasts["time_block"].eq("ALL")
        & score_contrasts["score_name"].eq("candidate_context_blend_v1")
    ].copy()
    if not blend.empty:
        bad = blend[pd.to_numeric(blend["delta_top2_excess_mean"], errors="coerce").lt(0)]
        if len(bad) == len(blend):
            findings.append("- `candidate_context_blend_v1` 在所有主场景相对 P1 default 的 Top2 paired delta 均为负；新闻/财报/非价格 blend 继续只做解释/反证，不做正向排序升权。")
    h2026 = panel_metrics[
        panel_metrics["time_block"].eq("H2026_1")
        & panel_metrics["score_name"].isin([P1_DEFAULT_SCORE, "rank_avg_rev_watch", "rev_chip_core", "single_watch_proxy"])
    ].copy()
    if not h2026.empty:
        best = h2026.sort_values(["top2_excess_mean", "mean_rank_ic"], ascending=[False, False]).head(3)
        best_labels = [
            f"{row.decision_frequency}/{row.comparison_scenario}/{row.score_name}: top2_excess={row.top2_excess_mean:.3f}, RankIC={row.mean_rank_ic:.3f}"
            for row in best.itertuples(index=False)
        ]
        findings.append("- H2026_1 相对较好的 Top2 底层排序：" + "；".join(best_labels) + "。")
    ambiguous = gate_summary[pd.to_numeric(gate_summary.get("ambiguous_gap_rate"), errors="coerce").gt(0.35)]
    if not ambiguous.empty:
        findings.append("- 多数同领域 `rev_chip_core/default` 的 Top1/Top2 分数差过小，容易诱发 LLM 路径敏感；下一轮 DS 应固定 ranker anchor，只允许硬反证 override。")
    findings.append("- 推荐下一步：只对 every_2_weeks/cross_sector/default(rank_avg) 与 every_2_weeks/same_sector/rank_avg 做 Flash fresh paired 小面板；Pro 等 Flash paired 通过后再最小确认。")
    return findings


def gate_columns(frame: pd.DataFrame) -> list[str]:
    cols = [
        "decision_frequency",
        "comparison_scenario",
        "score_name",
        "n_groups",
        "mean_rank_ic",
        "rank_ic_positive_rate",
        "top1_excess_mean",
        "top1_excess_panel_std",
        "top2_excess_mean",
        "top2_excess_panel_min",
        "top1_positive_rate",
        "top1_worst_rate",
        "ambiguous_gap_rate",
        "candidate_for_ds_panel",
        "stability_note",
    ]
    return [col for col in cols if col in frame]


def primary_columns(frame: pd.DataFrame) -> list[str]:
    cols = [
        "decision_frequency",
        "comparison_scenario",
        "score_name",
        "panels",
        "n_groups",
        "mean_rank_ic",
        "rank_ic_positive_rate",
        "top1_excess_mean",
        "top1_excess_panel_std",
        "top2_excess_mean",
        "top2_excess_panel_min",
        "top1_positive_rate",
        "top1_worst_rate",
        "regret_mean",
    ]
    return [col for col in cols if col in frame]


def contrast_columns(frame: pd.DataFrame) -> list[str]:
    cols = [
        "decision_frequency",
        "comparison_scenario",
        "score_name",
        "n_groups",
        "delta_top1_excess_mean",
        "delta_top2_excess_mean",
        "delta_top1_positive_rate",
        "delta_top1_worst_rate",
        "delta_regret_mean",
        "beats_baseline_top1_rate",
        "beats_baseline_top2_rate",
    ]
    return [col for col in cols if col in frame]


def markdown_table(frame: pd.DataFrame, max_rows: int = 60) -> str:
    if frame.empty:
        return "_empty_"
    try:
        return frame.head(max_rows).to_markdown(index=False)
    except Exception:
        return frame.head(max_rows).to_csv(index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit P1 candidate-comparison stability across panels and frequencies.")
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--frequencies", default=",".join(DEFAULT_FREQUENCIES))
    parser.add_argument("--panels", type=int, default=3)
    parser.add_argument("--repeats-per-panel", type=int, default=2)
    parser.add_argument("--candidate-size", type=int, default=8)
    parser.add_argument("--industries-per-date", type=int, default=3)
    parser.add_argument("--min-industry-size", type=int, default=12)
    parser.add_argument("--max-dates-per-block", type=int, default=0)
    return parser.parse_args()


def safe_prefix(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value)).strip("_")
    return safe or DEFAULT_PREFIX


if __name__ == "__main__":
    main()
