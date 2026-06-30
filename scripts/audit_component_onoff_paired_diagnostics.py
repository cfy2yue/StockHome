from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_full_channel_ablation import build_joined_cards, _read_jsonl, _load_gt
from src.agent_training.dual_mode_round import BANK_ANNUAL_RATE


OUTPUT = ROOT / "reports" / "date_generalization"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build paired treatment-vs-control diagnostics for component on/off DeepSeek rounds."
    )
    parser.add_argument("--prefix", required=True, help="Input run prefix under reports/date_generalization.")
    parser.add_argument("--treatment", default="full_agent")
    parser.add_argument("--control", required=True)
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--output-dir", default=str(OUTPUT))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    input_prefix = _safe_prefix(args.prefix)
    output_prefix = _safe_prefix(args.output_prefix or f"{input_prefix}_{args.treatment}_vs_{args.control}_paired")

    evidence = _read_jsonl(output_dir / f"{input_prefix}_evidence_pack.jsonl")
    cards = _read_jsonl(output_dir / f"{input_prefix}_decision_ledger.jsonl")
    joined = build_joined_cards(cards, evidence, _load_gt())
    detail = build_paired_detail(joined, treatment=args.treatment, control=args.control)
    summary = summarize_pairs(detail, ["task_mode"])
    by_block = summarize_pairs(detail, ["task_mode", "valid_block"])
    by_panel = summarize_pairs(detail, ["task_mode", "sample_panel_id"])

    detail.to_csv(output_dir / f"{output_prefix}_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / f"{output_prefix}_summary.csv", index=False, encoding="utf-8-sig")
    by_block.to_csv(output_dir / f"{output_prefix}_by_block.csv", index=False, encoding="utf-8-sig")
    by_panel.to_csv(output_dir / f"{output_prefix}_by_panel.csv", index=False, encoding="utf-8-sig")
    write_report(
        output_dir / f"{output_prefix}.md",
        input_prefix=input_prefix,
        treatment=args.treatment,
        control=args.control,
        detail=detail,
        summary=summary,
        by_block=by_block,
        by_panel=by_panel,
    )

    print("A股研究Agent")
    print(f"paired_rows={len(detail)}")
    print(f"wrote: {output_dir / f'{output_prefix}.md'}")


def build_paired_detail(joined: pd.DataFrame, *, treatment: str, control: str) -> pd.DataFrame:
    if joined.empty:
        return pd.DataFrame()
    required = {"variant", "task_mode", "valid_block", "decision_date", "code"}
    missing = sorted(required - set(joined.columns))
    if missing:
        raise ValueError(f"joined decisions missing required columns: {missing}")
    work = joined[joined["variant"].isin([treatment, control])].copy()
    if work.empty:
        return pd.DataFrame()
    if "sample_panel_id" not in work:
        work["sample_panel_id"] = "panel_unknown"
    if "sample_rank_in_panel" not in work:
        work["sample_rank_in_panel"] = 1
    key_cols = ["task_mode", "valid_block", "decision_date", "code", "sample_panel_id", "sample_rank_in_panel"]
    value_cols = [
        "variant",
        "name",
        "research_grade",
        "simulated_action",
        "simulated_weight_change_num",
        "cash_adjusted_return_20d",
        "return_20d",
        "final_agent_reasoning_summary",
        "counter_evidence",
        "data_missing_flags",
        "quant_tool_adoption_decision",
        "quant_tool_override_reasons",
    ]
    available = [col for col in value_cols if col in work]
    slim = work[key_cols + available].copy()
    wide = slim.pivot_table(
        index=key_cols,
        columns="variant",
        values=[col for col in available if col != "variant"],
        aggfunc="first",
    )
    wide.columns = [f"{field}__{variant}" for field, variant in wide.columns]
    wide = wide.reset_index()
    t_weight = pd.to_numeric(wide.get(f"simulated_weight_change_num__{treatment}"), errors="coerce")
    c_weight = pd.to_numeric(wide.get(f"simulated_weight_change_num__{control}"), errors="coerce")
    detail = wide[t_weight.notna() & c_weight.notna()].copy()
    if detail.empty:
        return detail
    t_weight = pd.to_numeric(detail.get(f"simulated_weight_change_num__{treatment}"), errors="coerce").fillna(0.0)
    c_weight = pd.to_numeric(detail.get(f"simulated_weight_change_num__{control}"), errors="coerce").fillna(0.0)
    returns = _first_numeric(
        detail,
        [f"return_20d__{treatment}", f"return_20d__{control}", "return_20d"],
    )
    detail["return_20d"] = returns
    detail["positive_20d"] = returns.gt(0)
    detail["negative_20d"] = returns.lt(0)
    detail["weight_delta_treatment_minus_control"] = t_weight - c_weight
    cash = _bank_return_20d()
    detail["delta_raw_return_20d"] = detail["weight_delta_treatment_minus_control"] * returns
    detail["delta_cash_adjusted_return_20d"] = detail["weight_delta_treatment_minus_control"] * (returns - cash)
    detail["pair_direction"] = [
        classify_pair(delta, ret) for delta, ret in zip(detail["weight_delta_treatment_minus_control"], returns, strict=False)
    ]
    detail["research_only"] = True
    detail["not_investment_instruction"] = True
    return detail.sort_values(["task_mode", "valid_block", "sample_panel_id", "decision_date", "code"])


def summarize_pairs(detail: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in detail.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = {col: key for col, key in zip(group_cols, keys, strict=False)}
        directions = group["pair_direction"].value_counts()
        rows.append(
            {
                **base,
                "paired_rows": int(len(group)),
                "changed_rows": int(group["pair_direction"].ne("unchanged").sum()),
                "raised_positive": int(directions.get("raised_positive", 0)),
                "raised_negative": int(directions.get("raised_negative", 0)),
                "lowered_positive": int(directions.get("lowered_positive", 0)),
                "lowered_negative": int(directions.get("lowered_negative", 0)),
                "unchanged": int(directions.get("unchanged", 0)),
                "sum_delta_cash_adjusted_return_20d": _round_sum(group["delta_cash_adjusted_return_20d"]),
                "mean_delta_cash_adjusted_return_20d": _round_mean(group["delta_cash_adjusted_return_20d"]),
                "sum_delta_raw_return_20d": _round_sum(group["delta_raw_return_20d"]),
                "mean_weight_delta": _round_mean(group["weight_delta_treatment_minus_control"]),
                "lowered_positive_cost": _round_sum(
                    group.loc[group["pair_direction"].eq("lowered_positive"), "delta_cash_adjusted_return_20d"]
                ),
                "raised_negative_cost": _round_sum(
                    group.loc[group["pair_direction"].eq("raised_negative"), "delta_cash_adjusted_return_20d"]
                ),
                "useful_delta": _round_sum(
                    group.loc[group["pair_direction"].isin(["raised_positive", "lowered_negative"]), "delta_cash_adjusted_return_20d"]
                ),
                "harmful_delta": _round_sum(
                    group.loc[group["pair_direction"].isin(["lowered_positive", "raised_negative"]), "delta_cash_adjusted_return_20d"]
                ),
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return pd.DataFrame(rows)


def classify_pair(delta: Any, ret: Any, eps: float = 1e-9) -> str:
    try:
        delta_num = float(delta)
        ret_num = float(ret)
    except (TypeError, ValueError):
        return "unpaired_or_missing_return"
    if math.isnan(delta_num) or math.isnan(ret_num):
        return "unpaired_or_missing_return"
    if abs(delta_num) <= eps:
        return "unchanged"
    if delta_num > 0 and ret_num > 0:
        return "raised_positive"
    if delta_num > 0 and ret_num <= 0:
        return "raised_negative"
    if delta_num < 0 and ret_num > 0:
        return "lowered_positive"
    return "lowered_negative"


def write_report(
    path: Path,
    *,
    input_prefix: str,
    treatment: str,
    control: str,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    by_block: pd.DataFrame,
    by_panel: pd.DataFrame,
) -> None:
    lines = [
        f"# {input_prefix} Paired Component Diagnostics",
        "",
        "本报告只用于 A 股研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Scope",
        "",
        f"- treatment: `{treatment}`",
        f"- control: `{control}`",
        f"- paired_rows: `{len(detail)}`",
        "- future_label_boundary: 本报告使用 `return_20d` 做离线后验评估；这些字段不得进入 Agent evidence。",
        "- interpretation: `raised_positive/lowered_negative` 是有用方向；`lowered_positive/raised_negative` 是主要代价。",
        "",
        "## Summary By Task",
        "",
        _table(summary),
        "",
        "## By Block",
        "",
        _table(by_block),
        "",
        "## By Panel",
        "",
        _table(by_panel),
        "",
        "## Decision",
        "",
        *_decision_lines(summary, treatment=treatment, control=control),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _decision_lines(summary: pd.DataFrame, *, treatment: str, control: str) -> list[str]:
    if summary.empty:
        return ["- 没有足够配对行，不能判断组件贡献。"]
    lines = []
    for _, row in summary.iterrows():
        task = row.get("task_mode", "unknown")
        paired = int(row.get("paired_rows", 0) or 0)
        changed = int(row.get("changed_rows", 0) or 0)
        useful = float(row.get("useful_delta", 0) or 0)
        harmful = float(row.get("harmful_delta", 0) or 0)
        total = float(row.get("sum_delta_cash_adjusted_return_20d", 0) or 0)
        lp = int(row.get("lowered_positive", 0) or 0)
        rn = int(row.get("raised_negative", 0) or 0)
        if paired <= 0:
            verdict = "样本不足"
        elif changed == 0:
            verdict = "组件没有改变决策路径"
        elif total > 0 and harmful >= 0:
            verdict = "观察为正，但仍需 fresh panel"
        elif total > 0:
            verdict = "总量为正但存在错杀/错升成本"
        else:
            verdict = "不支持升权"
        lines.append(
            f"- `{task}`: `{treatment}` vs `{control}` paired={paired}, changed={changed}, "
            f"delta_sum={total:.4f}, useful={useful:.4f}, harmful={harmful:.4f}, "
            f"lowered_positive={lp}, raised_negative={rn}。判定：{verdict}。"
        )
    lines.append("- 若 `lowered_positive` 或 `raised_negative` 持续出现，组件只能保留为 checklist/反证材料，不能升为独立 alpha。")
    return lines


def _first_numeric(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    result = pd.Series([math.nan] * len(frame), index=frame.index, dtype="float64")
    for col in cols:
        if col not in frame:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        result = result.where(result.notna(), values)
    return result


def _round_mean(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    return round(float(clean.mean()), 6)


def _round_sum(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    return round(float(clean.sum()), 6)


def _bank_return_20d() -> float:
    return ((1 + BANK_ANNUAL_RATE) ** (20 / 252) - 1) * 100


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return frame.to_csv(index=False)


def _safe_prefix(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value).strip("_")
    return safe or "component_onoff_paired"


if __name__ == "__main__":
    main()

