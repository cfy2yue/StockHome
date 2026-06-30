"""Build data-driven conflict-quality labels for Agent Auditor training.

The goal is to learn which apparent conflicts are acceptable reversal friction
and which conflicts should remain hard veto candidates. This script is fully
local and does not call DeepSeek or any online data source.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.dual_mode_round import (  # noqa: E402
    DEFAULT_CHIP_CORE_FEATURES_PATH,
    DEFAULT_CORR_PEER_FEATURES_PATH,
    DEFAULT_KLINE_FEATURES_PATH,
    DEFAULT_TUSHARE_PEER_FEATURES_PATH,
    TIME_BLOCKS,
    _hard_conflict_count,
    _numeric,
    _portfolio_ranker_details,
    _positive_confirmation_count,
    load_ground_truth,
)


OUTPUT_DIR = ROOT / "reports" / "date_generalization"
GT_SOURCES = [
    ROOT / "reports" / "backtest_scale_500" / "epoch1" / "ground_truth.csv",
    ROOT / "reports" / "backtest_scale_500" / "test" / "ground_truth.csv",
]
DEFAULT_BLOCKS = ["H2023_2", "H2024_1", "H2024_2", "H2025_1", "H2025_2", "H2026_1"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze conflict quality labels for rev+chip candidates.")
    parser.add_argument("--score-quantile-min", type=float, default=0.80)
    parser.add_argument("--blocks", nargs="*", default=DEFAULT_BLOCKS)
    parser.add_argument("--output-prefix", default="conflict_quality_labels_v1")
    args = parser.parse_args()
    output_prefix = _safe_prefix(args.output_prefix)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_ground_truth(
        GT_SOURCES,
        kline_features_path=DEFAULT_KLINE_FEATURES_PATH,
        corr_peer_features_path=DEFAULT_CORR_PEER_FEATURES_PATH,
        tushare_peer_features_path=DEFAULT_TUSHARE_PEER_FEATURES_PATH,
        chip_core_features_path=DEFAULT_CHIP_CORE_FEATURES_PATH,
    )
    labeled = _build_labeled_candidates(frame, blocks=args.blocks, score_quantile_min=args.score_quantile_min)
    detail_path = OUTPUT_DIR / f"{output_prefix}_detail.csv"
    label_path = OUTPUT_DIR / f"{output_prefix}_label_summary.csv"
    combo_path = OUTPUT_DIR / f"{output_prefix}_combo_summary.csv"
    rule_path = OUTPUT_DIR / f"{output_prefix}_agent_rules.json"
    report_path = OUTPUT_DIR / f"{output_prefix}.md"

    label_summary = _summarize_by_conflict(labeled)
    combo_summary = _summarize_by_combo(labeled)
    agent_rules = _build_agent_rules(label_summary, combo_summary)
    labeled.to_csv(detail_path, index=False, encoding="utf-8-sig")
    label_summary.to_csv(label_path, index=False, encoding="utf-8-sig")
    combo_summary.to_csv(combo_path, index=False, encoding="utf-8-sig")
    rule_path.write_text(json.dumps(agent_rules, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(
        report_path,
        labeled=labeled,
        label_summary=label_summary,
        combo_summary=combo_summary,
        agent_rules=agent_rules,
        score_quantile_min=args.score_quantile_min,
    )
    print("A股研究Agent")
    print(f"candidate_rows: {len(labeled)}")
    print(f"wrote: {detail_path}")
    print(f"wrote: {label_path}")
    print(f"wrote: {combo_path}")
    print(f"wrote: {rule_path}")
    print(f"wrote: {report_path}")


def _build_labeled_candidates(frame: pd.DataFrame, *, blocks: list[str], score_quantile_min: float) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    source = frame.copy()
    source["code"] = source["code"].astype(str).str.zfill(6)
    source["date"] = pd.to_datetime(source["date"], errors="coerce").dt.date.astype(str)
    for block in blocks:
        scoped = _window(source, block)
        if scoped.empty:
            continue
        if "gt_status" in scoped and scoped["gt_status"].notna().any():
            scoped = scoped[scoped["gt_status"].astype(str).eq("evaluated")].copy()
        if scoped.empty:
            continue
        details = _portfolio_ranker_details(scoped, preset="rev_plus_chip_core", valid_block=block, decision_frequency="every_2_weeks")
        scoped = scoped.copy()
        scoped["rev_chip_score"] = details["score"]
        scoped["rev_chip_score_quantile"] = details["score_quantile"]
        selected = scoped[_numeric(scoped["rev_chip_score_quantile"]) >= score_quantile_min].copy()
        if selected.empty:
            continue
        selected["valid_block"] = block
        selected["pool_mean_return_20d"] = selected.groupby("date")["return_20d"].transform(lambda s: _numeric(s).mean())
        selected["pool_excess_20d"] = _numeric(selected["return_20d"]) - _numeric(selected["pool_mean_return_20d"])
        selected["positive_confirmation_count"] = _positive_confirmation_count(selected)
        selected["hard_conflict_count"] = _hard_conflict_count(selected)
        conflict_flags = _conflict_flags(selected)
        for name, values in conflict_flags.items():
            selected[name] = values
        selected["conflict_combo"] = selected.apply(_conflict_combo, axis=1)
        selected["conflict_quality_label"] = selected.apply(_row_quality_label, axis=1)
        rows.append(selected)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    keep = [
        "valid_block",
        "date",
        "code",
        "name",
        "industry",
        "rev_chip_score",
        "rev_chip_score_quantile",
        "return_20d",
        "pool_mean_return_20d",
        "pool_excess_20d",
        "positive_confirmation_count",
        "hard_conflict_count",
        "conflict_combo",
        "conflict_quality_label",
        "peer_weak_conflict",
        "chip_overhang_conflict",
        "kline_risk_conflict",
        "news_risk_conflict",
        "financial_risk_conflict",
        "financial_true_missing_conflict",
        "bookskill_missing_or_weak_conflict",
        "news_missing_conflict",
        "financial_no_recent_event",
        "lower_support",
        "upper_overhang",
        "cost_band_width",
        "tushare_industry_positive_breadth_20d",
        "tushare_industry_relative_return_20d",
        "news_missing_rate",
        "news_warning_score",
        "news_opportunity_score",
        "financial_report_join_status",
        "financial_quality_risk_score",
        "financial_surprise_score",
        "kline_return_20d",
        "kline_return_60d",
        "kline_atr20_pct",
        "triggered_skills",
    ]
    return out[[col for col in keep if col in out.columns]].copy()


def _conflict_flags(frame: pd.DataFrame) -> dict[str, pd.Series]:
    news_warning = _num_col(frame, "news_warning_score", 0.0)
    news_risk_legacy = _num_col(frame, "news_risk_event_score_30d", 0.0)
    news_missing = _num_col(frame, "news_missing_rate", 1.0)
    news_opportunity = _num_col(frame, "news_opportunity_score", 0.0)
    financial_status = frame["financial_report_join_status"].fillna("").astype(str) if "financial_report_join_status" in frame else pd.Series("", index=frame.index)
    financial_risk = _num_col(frame, "financial_quality_risk_score", 0.0)
    financial_surprise = _num_col(frame, "financial_surprise_score", 0.0)
    peer_breadth = _num_col(frame, "tushare_industry_positive_breadth_20d", 0.5)
    peer_rel = _num_col(frame, "tushare_industry_relative_return_20d", 0.0)
    upper_overhang = _num_col(frame, "upper_overhang", 0.0)
    cost_band = _num_col(frame, "cost_band_width", 0.0)
    kline20 = _num_col(frame, "kline_return_20d", 0.0)
    kline60 = _num_col(frame, "kline_return_60d", 0.0)
    atr20 = _num_col(frame, "kline_atr20_pct", 0.0)
    skills = frame["triggered_skills"].fillna("").astype(str) if "triggered_skills" in frame else pd.Series("", index=frame.index)
    return {
        "peer_weak_conflict": (peer_breadth <= 0.40) & (peer_rel < 0.0),
        "chip_overhang_conflict": (upper_overhang >= 1.50) | (cost_band >= 1.50),
        "kline_risk_conflict": (kline20 <= -20.0) | (kline60 <= -35.0) | (atr20 >= 12.0),
        "news_risk_conflict": (news_warning >= 0.55) | (news_risk_legacy > 0) | ((news_warning >= 0.35) & (news_warning > news_opportunity)),
        "financial_risk_conflict": (financial_risk >= 0.55) | (financial_surprise <= -0.35),
        "financial_true_missing_conflict": financial_status.isin(["feature_table_missing", "code_not_in_feature_table", "decision_date_invalid"]),
        "bookskill_missing_or_weak_conflict": skills.str.len().eq(0) | skills.str.contains("UNKNOWN", case=False, regex=False),
        "news_missing_conflict": news_missing >= 0.80,
        "financial_no_recent_event": financial_status.eq("no_event_in_window"),
    }


def _conflict_combo(row: pd.Series) -> str:
    names = [
        "peer_weak_conflict",
        "chip_overhang_conflict",
        "kline_risk_conflict",
        "news_risk_conflict",
        "financial_risk_conflict",
        "financial_true_missing_conflict",
        "bookskill_missing_or_weak_conflict",
        "news_missing_conflict",
        "financial_no_recent_event",
    ]
    active = [name.replace("_conflict", "").replace("financial_no_recent_event", "financial_no_recent_event") for name in names if bool(row.get(name))]
    return "+".join(active) if active else "no_conflict"


def _row_quality_label(row: pd.Series) -> str:
    ret = _safe(row.get("return_20d"))
    excess = _safe(row.get("pool_excess_20d"))
    if math.isnan(ret) or math.isnan(excess):
        return "unknown"
    if ret > 0 and excess > 0:
        return "acceptable_conflict_or_alpha"
    if ret <= -5 or excess <= -5:
        return "veto_risk"
    if ret > 0:
        return "market_beta_only"
    return "weak_or_negative"


def _summarize_by_conflict(labeled: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    conflict_cols = [
        "peer_weak_conflict",
        "chip_overhang_conflict",
        "kline_risk_conflict",
        "news_risk_conflict",
        "financial_risk_conflict",
        "financial_true_missing_conflict",
        "bookskill_missing_or_weak_conflict",
        "news_missing_conflict",
        "financial_no_recent_event",
    ]
    for name in conflict_cols:
        subset = labeled[labeled[name].astype(bool)] if name in labeled else pd.DataFrame()
        rows.append(_summary_row(subset, conflict=name))
    rows.append(_summary_row(labeled[labeled["hard_conflict_count"].eq(0)], conflict="no_hard_conflict"))
    rows.append(_summary_row(labeled[labeled["positive_confirmation_count"].ge(2)], conflict="positive_confirmation_ge2"))
    out = pd.DataFrame(rows)
    out["rule_status"] = out.apply(_rule_status, axis=1)
    return out.sort_values(["rule_status", "conflict"]).reset_index(drop=True)


def _summarize_by_combo(labeled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for combo, subset in labeled.groupby("conflict_combo", dropna=False):
        if len(subset) < 20:
            continue
        rows.append(_summary_row(subset, conflict=str(combo)))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rule_status"] = out.apply(_rule_status, axis=1)
    return out.sort_values(["rule_status", "rows"], ascending=[True, False]).reset_index(drop=True)


def _summary_row(subset: pd.DataFrame, *, conflict: str) -> dict[str, Any]:
    returns = _numeric(subset["return_20d"]) if not subset.empty and "return_20d" in subset else pd.Series(dtype=float)
    excess = _numeric(subset["pool_excess_20d"]) if not subset.empty and "pool_excess_20d" in subset else pd.Series(dtype=float)
    labels = subset["conflict_quality_label"].astype(str) if not subset.empty and "conflict_quality_label" in subset else pd.Series(dtype=str)
    block_pos = {}
    if not subset.empty:
        for block, block_df in subset.groupby("valid_block"):
            block_pos[f"{block}_pos20"] = _positive_rate(_numeric(block_df["return_20d"]))
    return {
        "conflict": conflict,
        "rows": int(len(subset)),
        "unique_stocks": int(subset["code"].astype(str).nunique()) if not subset.empty and "code" in subset else 0,
        "avg20": _mean(returns),
        "pos20": _positive_rate(returns),
        "pool_excess20": _mean(excess),
        "loss_gt5_rate": _rate(returns <= -5) if not returns.empty else None,
        "acceptable_label_rate": _rate(labels.eq("acceptable_conflict_or_alpha")) if not labels.empty else None,
        "veto_label_rate": _rate(labels.eq("veto_risk")) if not labels.empty else None,
        "min_block_pos20": min((value for value in block_pos.values() if value is not None), default=None),
        **block_pos,
    }


def _rule_status(row: pd.Series) -> str:
    rows = int(row.get("rows") or 0)
    avg20 = _safe(row.get("avg20"))
    pos20 = _safe(row.get("pos20"))
    excess = _safe(row.get("pool_excess20"))
    loss = _safe(row.get("loss_gt5_rate"))
    min_block = _safe(row.get("min_block_pos20"))
    if rows < 30:
        return "insufficient_sample"
    if avg20 > 0 and pos20 >= 0.55 and excess > 0 and (math.isnan(loss) or loss <= 0.25) and (math.isnan(min_block) or min_block >= 0.40):
        return "acceptable_reversal_friction"
    if avg20 < 0 or pos20 <= 0.45 or excess < 0 or (not math.isnan(loss) and loss >= 0.30):
        return "veto_or_downweight"
    return "mixed_needs_agent_judgment"


def _build_agent_rules(summary: pd.DataFrame, combo_summary: pd.DataFrame) -> dict[str, Any]:
    rules = []
    for _, row in summary.iterrows():
        if int(row.get("rows") or 0) < 30:
            continue
        rules.append(
            {
                "conflict": row["conflict"],
                "rule_status": row["rule_status"],
                "rows": int(row["rows"]),
                "avg20": _round(row.get("avg20")),
                "pos20": _round(row.get("pos20")),
                "pool_excess20": _round(row.get("pool_excess20")),
                "loss_gt5_rate": _round(row.get("loss_gt5_rate")),
                "agent_use": _agent_use_text(str(row["rule_status"])),
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    combo_rules = []
    if not combo_summary.empty:
        for _, row in combo_summary.iterrows():
            if int(row.get("rows") or 0) < 30:
                continue
            status = str(row.get("rule_status") or "")
            if status == "insufficient_sample":
                continue
            combo_rules.append(
                {
                    "conflict_combo": row["conflict"],
                    "rule_status": status,
                    "rows": int(row["rows"]),
                    "avg20": _round(row.get("avg20")),
                    "pos20": _round(row.get("pos20")),
                    "pool_excess20": _round(row.get("pool_excess20")),
                    "loss_gt5_rate": _round(row.get("loss_gt5_rate")),
                    "agent_use": _agent_use_text(status),
                    "research_only": True,
                    "not_investment_instruction": True,
                }
            )
    return {
        "rule_version": "conflict_quality_labels_v1",
        "scope": "rev_plus_chip_core score_quantile>=0.80; future labels are offline only",
        "single_conflict_rules": rules,
        "combo_rules": combo_rules[:40],
        "research_only": True,
        "not_investment_instruction": True,
    }


def _agent_use_text(status: str) -> str:
    if status == "acceptable_reversal_friction":
        return "可作为反转候选的可接受摩擦，但仍需检查新闻/财报/同行/BookSkill是否有新增硬反证。"
    if status == "veto_or_downweight":
        return "应作为降权或否决候选，除非有强新闻/公告/财报催化和多通道确认。"
    return "不可机械放行或否决；交给Agent结合上下文判断冲突质量。"


def _write_report(
    path: Path,
    *,
    labeled: pd.DataFrame,
    label_summary: pd.DataFrame,
    combo_summary: pd.DataFrame,
    agent_rules: dict[str, Any],
    score_quantile_min: float,
) -> None:
    status_counts = label_summary["rule_status"].value_counts().rename_axis("rule_status").reset_index(name="conflicts") if not label_summary.empty else pd.DataFrame()
    lines = [
        "# Conflict Quality Labels v1",
        "",
        "本报告只用于研究辅助与回测训练诊断，不构成投资建议，不自动交易，不接券商接口。",
        "",
        f"- universe: rev_plus_chip_core score_quantile >= {score_quantile_min}",
        f"- candidate_rows: {len(labeled)}",
        "- label target: 20日收益与同池超额，未来结果只用于离线训练/反思，不进入决策 evidence。",
        "",
        "## Rule Status Counts",
        "",
        status_counts.to_markdown(index=False) if not status_counts.empty else "_无数据_",
        "",
        "## Conflict Summary",
        "",
        label_summary.to_markdown(index=False) if not label_summary.empty else "_无数据_",
        "",
        "## Frequent Conflict Combos",
        "",
        combo_summary.head(20).to_markdown(index=False) if not combo_summary.empty else "_无足量组合_",
        "",
        "## Agent Rule Draft",
        "",
        pd.DataFrame(agent_rules.get("single_conflict_rules") or []).to_markdown(index=False) if agent_rules.get("single_conflict_rules") else "_无足量规则_",
        "",
        "## Agent Combo Rule Draft",
        "",
        pd.DataFrame(agent_rules.get("combo_rules") or []).head(20).to_markdown(index=False) if agent_rules.get("combo_rules") else "_无足量组合规则_",
        "",
        "## Interpretation",
        "",
        "- `acceptable_reversal_friction` 表示该冲突在高 ranker 候选中并非自动否决，后续 Agent 可继续看上下文。",
        "- `veto_or_downweight` 表示该冲突在当前数据中更像风险，应优先作为反证或降权因素。",
        "- `mixed_needs_agent_judgment` 是 Agent 模式的主战场：不要机械过滤，要结合新闻、公告/财报、同行、筹码、BookSkill 与记忆判断冲突质量。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _window(frame: pd.DataFrame, block: str) -> pd.DataFrame:
    if block not in TIME_BLOCKS:
        return frame.iloc[0:0].copy()
    start, end = TIME_BLOCKS[block]
    dates = pd.to_datetime(frame["date"], errors="coerce")
    return frame[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))].copy()


def _num_col(frame: pd.DataFrame, field: str, default: float) -> pd.Series:
    if field not in frame:
        return pd.Series(default, index=frame.index, dtype="float64")
    return _numeric(frame[field]).fillna(default)


def _positive_rate(values: pd.Series) -> float | None:
    values = values.dropna()
    if values.empty:
        return None
    return round(float((values > 0).mean()), 4)


def _mean(values: pd.Series) -> float | None:
    values = values.dropna()
    if values.empty:
        return None
    return round(float(values.mean()), 4)


def _rate(mask: pd.Series) -> float | None:
    if mask.empty:
        return None
    return round(float(mask.mean()), 4)


def _safe(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _round(value: Any) -> float | None:
    number = _safe(value)
    return None if math.isnan(number) else round(number, 4)


def _safe_prefix(value: str) -> str:
    chars = [char if char.isalnum() or char in {"_", "-"} else "_" for char in value]
    return "".join(chars).strip("_") or "conflict_quality_labels"


if __name__ == "__main__":
    main()
