"""Build a safe hard-counter stratified sample plan.

The sample plan is selected only from time-available classifier probabilities
and current-channel conflict flags. Future labels/returns are written only to
the audit detail, not to the safe plan consumed by DeepSeek runners.
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

from src.agent_training.dual_mode_round import DEFAULT_JOINED_GT_CACHE_PATH, TIME_BLOCKS  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_SCORED_DETAIL = REPORT_DIR / "channel_rule_outcome_classifier_v1_scored_detail.csv"
OUTPUT_PREFIX = "hard_counter_stratified_sample_v1"
SAFE_COLUMNS = [
    "date",
    "code",
    "name",
    "valid_block",
    "task_mode",
    "stratum",
    "sample_panel_id",
    "sample_rank_in_panel",
    "sampler_context",
    "hard_counter_probability",
    "soft_gap_probability",
    "positive_support_probability",
    "conflict_count",
    "news_available_flag",
    "financial_available_flag",
    "bookskill_available_flag",
    "peer_weak_flag",
    "chip_overhang_flag",
]
FUTURE_COLUMNS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "gt_status",
    "gt_pass",
    "rule_outcome_label",
    "pool_excess_20d",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hard-counter stratified sample plan.")
    parser.add_argument("--scored-detail", type=Path, default=DEFAULT_SCORED_DETAIL)
    parser.add_argument("--joined-cache", type=Path, default=DEFAULT_JOINED_GT_CACHE_PATH)
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--valid-blocks", default="H2025_2,H2026_1")
    parser.add_argument("--rows-per-stratum", type=int, default=1)
    parser.add_argument("--task-modes", default="portfolio_pool,single_stock")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    blocks = [item.strip() for item in args.valid_blocks.split(",") if item.strip()]
    task_modes = [item.strip() for item in args.task_modes.split(",") if item.strip()]
    frame = load_and_enrich(args.scored_detail, args.joined_cache)
    selected = select_stratified_rows(frame, valid_blocks=blocks, rows_per_stratum=args.rows_per_stratum)
    safe_plan = expand_task_modes(selected, task_modes=task_modes)
    safe_plan = safe_plan[[col for col in SAFE_COLUMNS if col in safe_plan.columns]]
    audit_detail = selected.copy()
    paths = write_outputs(args.output_prefix, safe_plan, audit_detail, frame, blocks=blocks, task_modes=task_modes, rows_per_stratum=args.rows_per_stratum)

    print("A股研究Agent")
    print(f"candidate_rows={len(frame)}")
    print(f"selected_stock_dates={len(selected)}")
    print(f"safe_plan_rows={len(safe_plan)}")
    print(f"safe_plan={paths['safe_plan']}")
    print(f"audit_detail={paths['audit_detail']}")
    print(f"report={paths['report']}")


def load_and_enrich(scored_path: Path, joined_path: Path) -> pd.DataFrame:
    if not scored_path.exists():
        raise FileNotFoundError(scored_path)
    scored = pd.read_csv(scored_path, dtype={"code": str}, low_memory=False)
    scored.columns = [col.lstrip("\ufeff") for col in scored.columns]
    scored["date"] = pd.to_datetime(scored["date"], errors="coerce").dt.date.astype(str)
    scored["code"] = scored["code"].astype(str).str.zfill(6)
    joined_cols = [
        "date",
        "code",
        "name",
        "news_missing_rate",
        "news_warning_score",
        "financial_report_missing_rate",
        "triggered_skills",
        "tushare_industry_positive_breadth_20d",
        "tushare_industry_relative_return_20d",
        "upper_overhang",
        "return_20d",
        "pool_excess_20d",
        "gt_status",
    ]
    joined = pd.read_csv(joined_path, dtype={"code": str}, usecols=lambda col: col in joined_cols, low_memory=False)
    joined.columns = [col.lstrip("\ufeff") for col in joined.columns]
    joined["date"] = pd.to_datetime(joined["date"], errors="coerce").dt.date.astype(str)
    joined["code"] = joined["code"].astype(str).str.zfill(6)
    frame = scored.merge(joined, on=["date", "code"], how="left", suffixes=("", "_joined"))
    if "name_joined" in frame:
        frame["name"] = frame.get("name").fillna(frame["name_joined"]) if "name" in frame else frame["name_joined"]
    frame["valid_block"] = frame.get("valid_block", frame["date"].map(block_for_date)).fillna(frame["date"].map(block_for_date))
    frame["hard_counter_probability"] = _num(frame.get("logistic_channel_outcome__prob_hard_counter"))
    frame["soft_gap_probability"] = _num(frame.get("logistic_channel_outcome__prob_soft_gap"))
    frame["positive_support_probability"] = _num(frame.get("logistic_channel_outcome__prob_positive_support"))

    news_missing = _col(frame, "news_missing_rate", 1.0).fillna(1.0) >= 0.75
    news_warning = _col(frame, "news_warning_score", 0.0).fillna(0.0) >= 0.6
    fin_missing = _col(frame, "financial_report_missing_rate", 1.0).fillna(1.0) >= 0.75
    book_missing = frame.get("triggered_skills", pd.Series("", index=frame.index)).fillna("").astype(str).str.len().eq(0)
    peer_weak = (_col(frame, "tushare_industry_positive_breadth_20d", 0.5).fillna(0.5) < 0.4) | (
        _col(frame, "tushare_industry_relative_return_20d", 0.0).fillna(0.0) < -2.0
    )
    overhang = _col(frame, "upper_overhang", 0.0).fillna(0.0)
    overhang_threshold = overhang.quantile(0.70)
    if pd.isna(overhang_threshold):
        overhang_threshold = 0.6
    chip_overhang = overhang >= float(overhang_threshold)

    frame["news_available_flag"] = (~news_missing).astype(int)
    frame["financial_available_flag"] = (~fin_missing).astype(int)
    frame["bookskill_available_flag"] = (~book_missing).astype(int)
    frame["peer_weak_flag"] = peer_weak.astype(int)
    frame["chip_overhang_flag"] = chip_overhang.astype(int)
    frame["conflict_count"] = (
        news_missing.astype(int)
        + news_warning.astype(int)
        + fin_missing.astype(int)
        + book_missing.astype(int)
        + peer_weak.astype(int)
        + chip_overhang.astype(int)
    )
    return frame


def select_stratified_rows(frame: pd.DataFrame, *, valid_blocks: list[str], rows_per_stratum: int) -> pd.DataFrame:
    scoped = frame[frame["valid_block"].astype(str).isin(valid_blocks)].copy()
    if scoped.empty:
        return scoped
    high_q = scoped["hard_counter_probability"].quantile(0.75)
    low_q = scoped["hard_counter_probability"].quantile(0.25)
    scoped["hard_bucket"] = np.where(
        scoped["hard_counter_probability"] >= high_q,
        "high",
        np.where(scoped["hard_counter_probability"] <= low_q, "low", "mid"),
    )
    scoped["conflict_bucket"] = np.where(scoped["conflict_count"] >= 3, "high", "low")
    scoped = scoped[scoped["hard_bucket"].isin(["high", "low"])].copy()
    scoped["stratum"] = scoped["hard_bucket"] + "_hard_" + scoped["conflict_bucket"] + "_conflict"
    strata = ["high_hard_high_conflict", "high_hard_low_conflict", "low_hard_high_conflict", "low_hard_low_conflict"]
    selected = []
    used: set[tuple[str, str]] = set()
    used_codes: set[str] = set()
    for block in valid_blocks:
        block_frame = scoped[scoped["valid_block"].astype(str).eq(block)].copy()
        for stratum in strata:
            group = block_frame[block_frame["stratum"].eq(stratum)].copy()
            if group.empty:
                continue
            ascending = [True, True, True]
            if stratum.startswith("high_hard"):
                group = group.sort_values(["hard_counter_probability", "conflict_count", "date", "code"], ascending=[False, False, True, True])
            else:
                group = group.sort_values(["hard_counter_probability", "conflict_count", "date", "code"], ascending=[True, False, True, True])
            rows = _take_diverse_rows(group, used=used, used_codes=used_codes, rows_per_stratum=max(1, rows_per_stratum))
            selected.extend(rows)
    if not selected:
        return scoped.iloc[0:0].copy()
    out = pd.DataFrame(selected).reset_index(drop=True)
    out["sample_panel_id"] = out["valid_block"].astype(str) + "_" + out["stratum"].astype(str)
    out["sample_rank_in_panel"] = out.groupby(["valid_block", "stratum"]).cumcount() + 1
    out["sampler_context"] = (
        "hard_counter_stratified_v1;"
        + "hard_prob="
        + out["hard_counter_probability"].round(4).astype(str)
        + ";conflict_count="
        + out["conflict_count"].astype(int).astype(str)
        + ";stratum="
        + out["stratum"].astype(str)
    )
    return out


def _take_diverse_rows(
    group: pd.DataFrame,
    *,
    used: set[tuple[str, str]],
    used_codes: set[str],
    rows_per_stratum: int,
) -> list[pd.Series]:
    rows: list[pd.Series] = []
    deferred: list[pd.Series] = []
    for _, row in group.iterrows():
        code = str(row["code"]).zfill(6)
        key = (str(row["date"]), code)
        if key in used:
            continue
        if code in used_codes:
            deferred.append(row)
            continue
        used.add(key)
        used_codes.add(code)
        rows.append(row)
        if len(rows) >= rows_per_stratum:
            return rows
    for row in deferred:
        code = str(row["code"]).zfill(6)
        key = (str(row["date"]), code)
        if key in used:
            continue
        used.add(key)
        used_codes.add(code)
        rows.append(row)
        if len(rows) >= rows_per_stratum:
            break
    return rows


def expand_task_modes(selected: pd.DataFrame, *, task_modes: list[str]) -> pd.DataFrame:
    if selected.empty:
        return selected.copy()
    rows = []
    for _, row in selected.iterrows():
        for task_mode in task_modes:
            item = row.to_dict()
            item["task_mode"] = task_mode
            rows.append(item)
    return pd.DataFrame(rows)


def write_outputs(prefix: str, safe_plan: pd.DataFrame, audit_detail: pd.DataFrame, frame: pd.DataFrame, *, blocks: list[str], task_modes: list[str], rows_per_stratum: int) -> dict[str, Path]:
    safe_path = REPORT_DIR / f"{prefix}_sample_plan.csv"
    audit_path = REPORT_DIR / f"{prefix}_audit_detail.csv"
    summary_path = REPORT_DIR / f"{prefix}_summary.csv"
    report_path = REPORT_DIR / f"{prefix}.md"
    safe_plan[[col for col in SAFE_COLUMNS if col in safe_plan]].to_csv(safe_path, index=False, encoding="utf-8-sig")
    audit_cols = [col for col in [*SAFE_COLUMNS, "rule_outcome_label", "return_20d", "pool_excess_20d", "gt_status"] if col in audit_detail.columns]
    audit_detail[audit_cols].to_csv(audit_path, index=False, encoding="utf-8-sig")
    summary = summarize(audit_detail)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_report(report_path, safe_plan, audit_detail, summary, frame, blocks=blocks, task_modes=task_modes, rows_per_stratum=rows_per_stratum)
    return {"safe_plan": safe_path, "audit_detail": audit_path, "summary": summary_path, "report": report_path}


def summarize(audit_detail: pd.DataFrame) -> pd.DataFrame:
    if audit_detail.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in audit_detail.groupby(["valid_block", "stratum"], sort=True):
        block, stratum = keys
        returns = _num(group.get("return_20d"))
        excess = _num(group.get("pool_excess_20d"))
        rows.append(
            {
                "valid_block": block,
                "stratum": stratum,
                "stock_dates": int(len(group)),
                "unique_stocks": int(group["code"].nunique()),
                "avg_hard_counter_probability": round(float(group["hard_counter_probability"].mean()), 6),
                "avg_conflict_count": round(float(group["conflict_count"].mean()), 6),
                "posterior_positive_20d_rate": round(float((returns > 0).mean()), 6) if returns.notna().any() else np.nan,
                "posterior_loss_gt5_rate": round(float((returns <= -5).mean()), 6) if returns.notna().any() else np.nan,
                "posterior_avg_return_20d": round(float(returns.mean()), 6) if returns.notna().any() else np.nan,
                "posterior_pool_excess_20d": round(float(excess.mean()), 6) if excess.notna().any() else np.nan,
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return pd.DataFrame(rows)


def write_report(path: Path, safe_plan: pd.DataFrame, audit_detail: pd.DataFrame, summary: pd.DataFrame, frame: pd.DataFrame, *, blocks: list[str], task_modes: list[str], rows_per_stratum: int) -> None:
    future_in_safe = sorted(set(safe_plan.columns) & FUTURE_COLUMNS)
    lines = [
        "# Hard-Counter Stratified Sample Plan v1",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Setup",
        "",
        f"- valid_blocks: `{','.join(blocks)}`",
        f"- rows_per_stratum: `{rows_per_stratum}`",
        f"- task_modes: `{','.join(task_modes)}`",
        f"- candidate_rows: `{len(frame)}`",
        f"- selected_stock_dates: `{len(audit_detail)}`",
        f"- safe_plan_rows: `{len(safe_plan)}`",
        f"- future_columns_in_safe_plan: `{future_in_safe or 'none'}`",
        "",
        "## Strata",
        "",
        "- `high_hard_high_conflict`: classifier thinks hard-counter risk is high and current-channel conflicts are high.",
        "- `high_hard_low_conflict`: high hard-counter probability but fewer visible conflicts; useful for over-defense checks.",
        "- `low_hard_high_conflict`: visible conflicts but classifier does not think hard-counter is high; useful for false-negative checks.",
        "- `low_hard_low_conflict`: low-risk control.",
        "",
        "## Summary",
        "",
        _table(summary),
        "",
        "## Safe Plan Preview",
        "",
        _table(safe_plan.head(20)),
        "",
        "## Boundary",
        "",
        "- Safe sample plan excludes `return_20d`, `gt_status`, `rule_outcome_label`, and pool-excess fields.",
        "- Audit detail may contain posterior outcomes for offline interpretation only.",
        "- This sample is diagnostic and stratified; performance metrics from it are not population-level proof.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def block_for_date(value: Any) -> str | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    for block, (start, end) in TIME_BLOCKS.items():
        if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
            return block
    return None


def _num(value: Any) -> pd.Series:
    if isinstance(value, pd.Series):
        return pd.to_numeric(value, errors="coerce")
    return pd.to_numeric(pd.Series(value), errors="coerce")


def _col(frame: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col in frame:
        return pd.to_numeric(frame[col], errors="coerce")
    return pd.Series(default, index=frame.index, dtype="float64")


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_无数据_"
    return frame.to_markdown(index=False)


if __name__ == "__main__":
    main()
