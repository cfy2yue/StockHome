"""Analyze portfolio keypoint Flash results.

This is a post-decision offline audit. Future returns may be joined from an
audit detail file only after decision cards already exist. Outputs are reports
and diagnostics; they must not be fed back into same-block Agent evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "date_generalization"
BANK_CASH_RETURN_20D_PCT = ((1.0 + 0.03) ** (20 / 252) - 1.0) * 100
FUTURE_KEYS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "gt_status",
    "gt_pass",
    "pool_excess_20d",
    "offline_high_impact_label",
}
INSTRUCTION_PHRASES = ["买入", "卖出", "强烈推荐", "目标价必达"]
SECRET_RE = re.compile(r"sk-[A-Za-z0-9]{20,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze portfolio keypoint Flash result.")
    parser.add_argument("--run-prefix", required=True, help="Prefix used by run_full_channel_ablation_round.py.")
    parser.add_argument("--audit-detail", type=Path, required=True, help="Sample-plan audit detail with offline returns.")
    parser.add_argument("--output-prefix", default="", help="Defaults to <run-prefix>_keypoint_analysis.")
    parser.add_argument("--control-variant", default="full_agent_without_quant_tools")
    parser.add_argument("--treatment-variant", default="full_agent_with_quant_tools")
    args = parser.parse_args()

    output_prefix = args.output_prefix or f"{args.run_prefix}_keypoint_analysis"
    decision_path = REPORT_DIR / f"{args.run_prefix}_decision_ledger.jsonl"
    evidence_path = REPORT_DIR / f"{args.run_prefix}_evidence_pack.jsonl"
    cards = load_jsonl(decision_path)
    evidence = load_jsonl(evidence_path)
    audit = load_audit(args.audit_detail)
    detail = join_detail(cards, audit)
    detail = add_return_metrics(detail)
    variant_summary = summarize_variant(detail)
    stratum_summary = summarize_stratum(detail)
    adoption_summary = summarize_adoption(detail)
    pair_detail, pair_summary = summarize_pairs(detail, args.treatment_variant, args.control_variant)
    safety = safety_audit(evidence=evidence, cards=cards)

    paths = write_outputs(
        output_prefix,
        detail,
        variant_summary,
        stratum_summary,
        adoption_summary,
        pair_detail,
        pair_summary,
        safety,
        args,
    )

    print("A股研究Agent")
    print(f"cards={len(cards)}")
    print(f"detail={paths['detail']}")
    print(f"report={paths['report']}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_audit(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    frame.columns = [col.lstrip("\ufeff") for col in frame.columns]
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    return frame


def join_detail(cards: list[dict[str, Any]], audit: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(cards)
    if frame.empty:
        return frame
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="coerce").dt.date.astype(str)
    keep = [
        "date",
        "code",
        "stratum",
        "offline_high_impact_label",
        "return_20d",
        "gt_status",
        "ml_keypoint_score",
        "heuristic_key_score_pct",
        "rev_chip_score_quantile",
    ]
    keep = [col for col in keep if col in audit.columns]
    return frame.merge(audit[keep], left_on=["decision_date", "code"], right_on=["date", "code"], how="left")


def add_return_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["simulated_weight_change"] = pd.to_numeric(out.get("simulated_weight_change"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["return_20d"] = pd.to_numeric(out.get("return_20d"), errors="coerce")
    out["cash_adjusted_return_20d"] = (
        out["simulated_weight_change"] * out["return_20d"]
        + (1.0 - out["simulated_weight_change"]) * BANK_CASH_RETURN_20D_PCT
    )
    out["posterior_positive_20d"] = out["return_20d"] > 0
    out["posterior_negative_20d"] = out["return_20d"] < 0
    out["exposure_card"] = out.get("simulated_action", "").astype(str).eq("增加研究暴露")
    out["active_observe_card"] = out["simulated_weight_change"] > 0
    out["bad_observe_weight"] = out["active_observe_card"] & out["posterior_negative_20d"]
    out["missed_positive_cash"] = out["simulated_weight_change"].eq(0) & out["posterior_positive_20d"]
    reasons = out["quant_tool_override_reasons"] if "quant_tool_override_reasons" in out else pd.Series("", index=out.index)
    out["override_reasons_list"] = reasons.fillna("").astype(str).str.split(";")
    return out


def summarize_variant(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for variant, group in detail.groupby("variant", sort=True):
        rows.append(summary_row(group, {"variant": variant}))
    return pd.DataFrame(rows)


def summarize_stratum(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for values, group in detail.groupby(["variant", "stratum"], sort=True):
        rows.append(summary_row(group, {"variant": values[0], "stratum": values[1]}))
    return pd.DataFrame(rows)


def summarize_adoption(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for values, group in detail.groupby(["variant", "quant_tool_adoption_decision"], dropna=False, sort=True):
        row = summary_row(group, {"variant": values[0], "quant_tool_adoption_decision": values[1]})
        reasons = []
        for items in group["override_reasons_list"]:
            reasons.extend([str(item).strip() for item in items if str(item).strip() and str(item).strip() != "none"])
        reason_counts = pd.Series(reasons).value_counts().head(8).to_dict() if reasons else {}
        row["top_override_reasons"] = ";".join(f"{key}:{value}" for key, value in reason_counts.items())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_pairs(detail: pd.DataFrame, treatment: str, control: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if detail.empty:
        return pd.DataFrame(), pd.DataFrame()
    key_cols = ["decision_date", "code", "stratum", "return_20d", "offline_high_impact_label"]
    work = detail[detail["variant"].isin([treatment, control])].copy()
    pivot = work.pivot_table(index=key_cols, columns="variant", values="simulated_weight_change", aggfunc="first").reset_index()
    if treatment not in pivot or control not in pivot:
        return pivot, pd.DataFrame()
    pivot["delta_weight"] = pivot[treatment] - pivot[control]
    pivot["delta_cash"] = pivot["delta_weight"] * pd.to_numeric(pivot["return_20d"], errors="coerce")
    pivot["direction"] = pivot.apply(classify_pair_direction, axis=1)
    rows = []
    for values, group in pivot.groupby(["stratum", "direction"], dropna=False, sort=True):
        rows.append(
            {
                "stratum": values[0],
                "direction": values[1],
                "rows": int(len(group)),
                "sum_delta_cash": round(float(group["delta_cash"].sum()), 6),
                "avg_delta_cash": round(float(group["delta_cash"].mean()), 6),
                "avg_delta_weight": round(float(group["delta_weight"].mean()), 6),
            }
        )
    rows.append(
        {
            "stratum": "ALL",
            "direction": "ALL",
            "rows": int(len(pivot)),
            "sum_delta_cash": round(float(pivot["delta_cash"].sum()), 6),
            "avg_delta_cash": round(float(pivot["delta_cash"].mean()), 6),
            "avg_delta_weight": round(float(pivot["delta_weight"].mean()), 6),
        }
    )
    return pivot, pd.DataFrame(rows)


def classify_pair_direction(row: pd.Series) -> str:
    delta = float(row.get("delta_weight") or 0.0)
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


def summary_row(group: pd.DataFrame, base: dict[str, Any]) -> dict[str, Any]:
    row = dict(base)
    values = pd.to_numeric(group["cash_adjusted_return_20d"], errors="coerce")
    weights = pd.to_numeric(group["simulated_weight_change"], errors="coerce")
    returns = pd.to_numeric(group["return_20d"], errors="coerce")
    row.update(
        {
            "cards": int(len(group)),
            "exposure_cards": int(group["exposure_card"].sum()),
            "active_observe_cards": int(group["active_observe_card"].sum()),
            "avg_weight": round(float(weights.mean()), 6) if weights.notna().any() else np.nan,
            "cash_adjusted_avg20": round(float(values.mean()), 6) if values.notna().any() else np.nan,
            "cash_adjusted_pos20": round(float((values > 0).mean()), 6) if values.notna().any() else np.nan,
            "raw_return_avg20": round(float(returns.mean()), 6) if returns.notna().any() else np.nan,
            "raw_positive_rate20": round(float((returns > 0).mean()), 6) if returns.notna().any() else np.nan,
            "bad_observe_weight_cards": int(group["bad_observe_weight"].sum()),
            "missed_positive_cash_cards": int(group["missed_positive_cash"].sum()),
            "offline_high_impact_rate": round(float(pd.to_numeric(group.get("offline_high_impact_label"), errors="coerce").mean()), 6)
            if "offline_high_impact_label" in group
            else np.nan,
            "data_missing_flag_cards": int(group.get("data_missing_flags", pd.Series("", index=group.index)).fillna("").astype(str).ne("").sum()),
            "research_only": True,
            "not_investment_instruction": True,
        }
    )
    return row


def safety_audit(*, evidence: list[dict[str, Any]], cards: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for name, rows in [("evidence", evidence), ("decision", cards)]:
        future_hits = 0
        secret_hits = 0
        instruction_hits = 0
        sampler_context = 0
        quant_counts: dict[str, set[int]] = {}
        for row in rows:
            text = json.dumps(row, ensure_ascii=False)
            secret_hits += int(bool(SECRET_RE.search(text)))
            instruction_hits += int(any(phrase in text for phrase in INSTRUCTION_PHRASES))
            future_hits += count_future_keys(row)
            if row.get("sampler_context"):
                sampler_context += 1
            variant = str(row.get("variant") or row.get("component_ablation_variant") or "unknown")
            quant_counts.setdefault(variant, set()).add(len(row.get("quant_tool_summaries") or []))
        result[f"{name}_rows"] = len(rows)
        result[f"{name}_future_key_hits"] = future_hits
        result[f"{name}_secret_hits"] = secret_hits
        result[f"{name}_instruction_hits"] = instruction_hits
        result[f"{name}_sampler_context_rows"] = sampler_context
        result[f"{name}_quant_counts"] = {key: sorted(value) for key, value in quant_counts.items()}
    return result


def count_future_keys(value: Any) -> int:
    if isinstance(value, dict):
        return sum(1 for key in value if str(key) in FUTURE_KEYS) + sum(count_future_keys(child) for child in value.values())
    if isinstance(value, list):
        return sum(count_future_keys(child) for child in value)
    return 0


def write_outputs(
    output_prefix: str,
    detail: pd.DataFrame,
    variant_summary: pd.DataFrame,
    stratum_summary: pd.DataFrame,
    adoption_summary: pd.DataFrame,
    pair_detail: pd.DataFrame,
    pair_summary: pd.DataFrame,
    safety: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Path]:
    paths = {
        "detail": REPORT_DIR / f"{output_prefix}_detail.csv",
        "variant_summary": REPORT_DIR / f"{output_prefix}_variant_summary.csv",
        "stratum_summary": REPORT_DIR / f"{output_prefix}_stratum_summary.csv",
        "adoption_summary": REPORT_DIR / f"{output_prefix}_adoption_summary.csv",
        "pair_detail": REPORT_DIR / f"{output_prefix}_pair_detail.csv",
        "pair_summary": REPORT_DIR / f"{output_prefix}_pair_summary.csv",
        "safety": REPORT_DIR / f"{output_prefix}_safety.json",
        "report": REPORT_DIR / f"{output_prefix}.md",
    }
    detail.to_csv(paths["detail"], index=False, encoding="utf-8-sig")
    variant_summary.to_csv(paths["variant_summary"], index=False, encoding="utf-8-sig")
    stratum_summary.to_csv(paths["stratum_summary"], index=False, encoding="utf-8-sig")
    adoption_summary.to_csv(paths["adoption_summary"], index=False, encoding="utf-8-sig")
    pair_detail.to_csv(paths["pair_detail"], index=False, encoding="utf-8-sig")
    pair_summary.to_csv(paths["pair_summary"], index=False, encoding="utf-8-sig")
    paths["safety"].write_text(json.dumps(safety, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(paths["report"], variant_summary, stratum_summary, adoption_summary, pair_summary, safety, args)
    return paths


def write_report(
    path: Path,
    variant_summary: pd.DataFrame,
    stratum_summary: pd.DataFrame,
    adoption_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    safety: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    lines = [
        f"# {args.run_prefix} Keypoint Analysis",
        "",
        "本报告只用于 A 股研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Safety",
        "",
        f"- evidence future-key hits: `{safety.get('evidence_future_key_hits', 0)}`",
        f"- decision future-key hits: `{safety.get('decision_future_key_hits', 0)}`",
        f"- evidence secret hits: `{safety.get('evidence_secret_hits', 0)}`",
        f"- decision secret hits: `{safety.get('decision_secret_hits', 0)}`",
        f"- evidence instruction hits: `{safety.get('evidence_instruction_hits', 0)}`",
        f"- decision instruction hits: `{safety.get('decision_instruction_hits', 0)}`",
        f"- sampler_context evidence rows: `{safety.get('evidence_sampler_context_rows', 0)}`",
        "",
        "## Variant Summary",
        "",
        markdown_table(
            variant_summary,
            [
                "variant",
                "cards",
                "exposure_cards",
                "active_observe_cards",
                "avg_weight",
                "cash_adjusted_avg20",
                "cash_adjusted_pos20",
                "bad_observe_weight_cards",
                "missed_positive_cash_cards",
            ],
        ),
        "",
        "## Stratum Summary",
        "",
        markdown_table(
            stratum_summary,
            [
                "variant",
                "stratum",
                "cards",
                "avg_weight",
                "cash_adjusted_avg20",
                "cash_adjusted_pos20",
                "bad_observe_weight_cards",
                "offline_high_impact_rate",
            ],
        ),
        "",
        "## Pair Summary",
        "",
        markdown_table(pair_summary, ["stratum", "direction", "rows", "sum_delta_cash", "avg_delta_cash", "avg_delta_weight"]),
        "",
        "## Adoption Summary",
        "",
        markdown_table(
            adoption_summary,
            [
                "variant",
                "quant_tool_adoption_decision",
                "cards",
                "cash_adjusted_avg20",
                "bad_observe_weight_cards",
                "top_override_reasons",
            ],
        ),
        "",
        "## Decision",
        "",
        "- 该脚本只做 post-decision 离线归因；未来收益不得回灌同块 evidence。",
        "- 若 `raised_negative` 或 `bad_observe_weight_cards` 增加，下一轮必须先收紧 adoption/override 规则，而不是只扩大样本。",
        "- 若所有 `exposure_cards=0`，结果只能说明观察权重路径，不可宣称主动 alpha。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    show = frame[[col for col in columns if col in frame]].copy()
    for col in show.columns:
        if pd.api.types.is_numeric_dtype(show[col]):
            show[col] = show[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    header = "| " + " | ".join(show.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(show.columns)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in show.astype(str).to_numpy()]
    return "\n".join([header, sep, *body])


if __name__ == "__main__":
    main()
