"""Replay portfolio quant-tool adoption guards on completed decisions.

This script is an offline post-decision audit. It joins future returns only
after DeepSeek decision cards have already been produced. The replayed guard
metrics are for policy design and must not be inserted into same-block
evidence packs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "date_generalization"
BANK_CASH_RETURN_20D_PCT = ((1.0 + 0.03) ** (20 / 252) - 1.0) * 100
REASON_KEYS = [
    "news_gap",
    "financial_gap",
    "peer_gap",
    "bookskill_gap",
    "chip_overhang",
    "data_missing",
    "overheat_or_volatility",
    "memory_or_rag_counter",
]


GuardFn = Callable[[pd.DataFrame], pd.Series]


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay portfolio quant-tool adoption guard candidates.")
    parser.add_argument("--detail", type=Path, required=True, help="Detail CSV from analyze_portfolio_keypoint_flash.py.")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--treatment-variant", default="full_agent_with_quant_tools")
    parser.add_argument("--control-variant", default="full_agent_without_quant_tools")
    args = parser.parse_args()

    detail = load_detail(args.detail)
    pair = make_pair_frame(detail, treatment=args.treatment_variant, control=args.control_variant)
    summary, replay_detail = evaluate_policies(pair)
    paths = write_outputs(args.output_prefix, summary, replay_detail, args)
    print("A股研究Agent")
    print(f"pairs={len(pair)}")
    print(f"summary={paths['summary']}")
    print(f"report={paths['report']}")


def load_detail(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    frame.columns = [col.lstrip("\ufeff") for col in frame.columns]
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    return frame


def make_pair_frame(detail: pd.DataFrame, *, treatment: str, control: str) -> pd.DataFrame:
    needed = {"variant", "decision_date", "code", "simulated_weight_change", "return_20d"}
    missing = needed - set(detail.columns)
    if missing:
        raise ValueError(f"detail missing columns: {sorted(missing)}")
    work = detail[detail["variant"].isin([treatment, control])].copy()
    attrs = [
        "decision_date",
        "code",
        "stratum",
        "return_20d",
        "quant_tool_adoption_decision",
        "quant_tool_override_reasons",
    ]
    attrs = [col for col in attrs if col in work.columns]
    treatment_frame = work[work["variant"].eq(treatment)][attrs + ["simulated_weight_change"]].rename(
        columns={"simulated_weight_change": "treatment_weight"}
    )
    control_frame = work[work["variant"].eq(control)][["decision_date", "code", "simulated_weight_change"]].rename(
        columns={"simulated_weight_change": "control_weight"}
    )
    pair = treatment_frame.merge(control_frame, on=["decision_date", "code"], how="inner")
    pair["treatment_weight"] = pd.to_numeric(pair["treatment_weight"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    pair["control_weight"] = pd.to_numeric(pair["control_weight"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    pair["return_20d"] = pd.to_numeric(pair["return_20d"], errors="coerce")
    reasons = pair.get("quant_tool_override_reasons", pd.Series("", index=pair.index)).fillna("").astype(str)
    for key in REASON_KEYS:
        pair[key] = reasons.str.contains(key, regex=False)
    pair["gap_count"] = pair[REASON_KEYS].sum(axis=1)
    pair["raise_from_quant"] = pair["treatment_weight"] > pair["control_weight"]
    return pair


def guard_policies() -> dict[str, GuardFn]:
    return {
        "no_guard_treatment": lambda frame: pd.Series(False, index=frame.index),
        "cap_all_raises": lambda frame: pd.Series(True, index=frame.index),
        "cap_ordinary_raises": lambda frame: frame["stratum"].astype(str).eq("ordinary_control_midkey"),
        "cap_gap_ge6_raises": lambda frame: frame["gap_count"] >= 6,
        "cap_gap_ge7_raises": lambda frame: frame["gap_count"] >= 7,
        "cap_partially_adopted_gap_ge6_raises": lambda frame: frame["quant_tool_adoption_decision"].astype(str).eq(
            "partially_adopted"
        )
        & (frame["gap_count"] >= 6),
        "cap_ordinary_or_partial_gap_ge6_raises": lambda frame: frame["stratum"].astype(str).eq("ordinary_control_midkey")
        | (
            frame["quant_tool_adoption_decision"].astype(str).eq("partially_adopted")
            & (frame["gap_count"] >= 6)
        ),
        "cap_chip_and_overheat_raises": lambda frame: frame["chip_overhang"] & frame["overheat_or_volatility"],
    }


def evaluate_policies(pair: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    replay_parts = []
    for policy, guard_fn in guard_policies().items():
        guarded = pair.copy()
        cap_mask = guard_fn(guarded).fillna(False) & guarded["raise_from_quant"]
        guarded["policy"] = policy
        guarded["guard_applied"] = cap_mask
        guarded["replay_weight"] = guarded["treatment_weight"]
        guarded.loc[cap_mask, "replay_weight"] = guarded.loc[cap_mask, "control_weight"]
        guarded["delta_weight_vs_control"] = guarded["replay_weight"] - guarded["control_weight"]
        guarded["delta_cash_vs_control"] = guarded["delta_weight_vs_control"] * guarded["return_20d"]
        guarded["cash_adjusted_return_20d"] = (
            guarded["replay_weight"] * guarded["return_20d"]
            + (1.0 - guarded["replay_weight"]) * BANK_CASH_RETURN_20D_PCT
        )
        guarded["direction"] = guarded.apply(classify_direction, axis=1)
        rows.append(summarize_policy(guarded, policy))
        replay_parts.append(guarded)
    summary = pd.DataFrame(rows)
    baseline = summary[summary["policy"].eq("no_guard_treatment")].iloc[0]
    summary["delta_sum_cash_vs_no_guard"] = summary["sum_delta_cash_vs_control"] - baseline["sum_delta_cash_vs_control"]
    summary["delta_avg20_vs_no_guard"] = summary["cash_adjusted_avg20"] - baseline["cash_adjusted_avg20"]
    return summary, pd.concat(replay_parts, ignore_index=True)


def classify_direction(row: pd.Series) -> str:
    delta = float(row.get("delta_weight_vs_control") or 0.0)
    ret = float(row.get("return_20d") or 0.0)
    if abs(delta) < 1e-12:
        return "unchanged"
    if delta > 0 and ret > 0:
        return "raised_positive"
    if delta > 0 and ret < 0:
        return "raised_negative"
    if delta < 0 and ret > 0:
        return "lowered_positive"
    if delta < 0 and ret < 0:
        return "lowered_negative"
    return "changed_zero_return"


def summarize_policy(frame: pd.DataFrame, policy: str) -> dict[str, float | int | str]:
    counts = frame["direction"].value_counts().to_dict()
    values = pd.to_numeric(frame["cash_adjusted_return_20d"], errors="coerce")
    return {
        "policy": policy,
        "rows": int(len(frame)),
        "guard_applied_rows": int(frame["guard_applied"].sum()),
        "avg_replay_weight": round(float(frame["replay_weight"].mean()), 6),
        "cash_adjusted_avg20": round(float(values.mean()), 6) if values.notna().any() else np.nan,
        "cash_adjusted_pos20": round(float((values > 0).mean()), 6) if values.notna().any() else np.nan,
        "sum_delta_cash_vs_control": round(float(frame["delta_cash_vs_control"].sum()), 6),
        "avg_delta_cash_vs_control": round(float(frame["delta_cash_vs_control"].mean()), 6),
        "raised_positive": int(counts.get("raised_positive", 0)),
        "raised_negative": int(counts.get("raised_negative", 0)),
        "lowered_positive": int(counts.get("lowered_positive", 0)),
        "lowered_negative": int(counts.get("lowered_negative", 0)),
        "unchanged": int(counts.get("unchanged", 0)),
    }


def write_outputs(
    output_prefix: str,
    summary: pd.DataFrame,
    replay_detail: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Path]:
    paths = {
        "summary": REPORT_DIR / f"{output_prefix}_summary.csv",
        "detail": REPORT_DIR / f"{output_prefix}_detail.csv",
        "report": REPORT_DIR / f"{output_prefix}.md",
    }
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    replay_detail.to_csv(paths["detail"], index=False, encoding="utf-8-sig")
    write_report(paths["report"], summary, args)
    return paths


def write_report(path: Path, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    best = summary.sort_values(["sum_delta_cash_vs_control", "raised_negative"], ascending=[False, True]).iloc[0]
    lines = [
        f"# {args.output_prefix} Quant Adoption Guard Replay",
        "",
        "本报告只用于 A 股研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Scope",
        "",
        f"- detail: `{args.detail}`",
        f"- treatment: `{args.treatment_variant}`",
        f"- control: `{args.control_variant}`",
        "- 规则只使用决策卡中已存在的 stratum、adoption decision 和 override reasons；未来收益只用于 replay 后验评估。",
        "",
        "## Summary",
        "",
        markdown_table(
            summary,
            [
                "policy",
                "guard_applied_rows",
                "cash_adjusted_avg20",
                "sum_delta_cash_vs_control",
                "delta_sum_cash_vs_no_guard",
                "raised_positive",
                "raised_negative",
                "lowered_positive",
                "lowered_negative",
                "unchanged",
            ],
        ),
        "",
        "## Decision",
        "",
        f"- Best replay by sum_delta is `{best['policy']}`, sum_delta_cash_vs_control={best['sum_delta_cash_vs_control']}.",
        "- 若简单 cap 规则降低了 `raised_negative` 但同时扩大 `lowered_positive` 或降低总 delta，不得升为默认护栏。",
        "- 下一步需要学习型 adoption guard：把 ordinary/keypoint、soft gap、hard counter-evidence、acceptable reversal friction 分开，而不是一刀切压回 control。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    show = frame[[col for col in columns if col in frame.columns]].copy()
    for col in show.columns:
        if pd.api.types.is_numeric_dtype(show[col]):
            show[col] = show[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    header = "| " + " | ".join(show.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(show.columns)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in show.astype(str).to_numpy()]
    return "\n".join([header, sep, *body])


if __name__ == "__main__":
    main()
