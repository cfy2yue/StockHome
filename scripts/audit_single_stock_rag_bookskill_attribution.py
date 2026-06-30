from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.dual_mode_round import _bank_return_20d


OUTPUT = ROOT / "reports" / "date_generalization"
GT_SOURCES = [
    ROOT / "reports" / "backtest_scale_500" / "epoch1" / "ground_truth.csv",
    ROOT / "reports" / "backtest_scale_500" / "test" / "ground_truth.csv",
]
JOIN_KEYS = ["variant", "task_mode", "valid_block", "decision_date", "code", "sample_panel_id"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit single-stock RAG and BookSkill attribution with optional no-RAG paired control."
    )
    parser.add_argument("--rag-prefix", required=True)
    parser.add_argument("--control-prefix", default=None)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    rag_prefix = _safe_prefix(args.rag_prefix)
    control_prefix = _safe_prefix(args.control_prefix) if args.control_prefix else None
    output_prefix = _safe_prefix(args.output_prefix)

    rag_packs = _read_jsonl(OUTPUT / f"{rag_prefix}_evidence_pack.jsonl")
    rag_cards = _read_jsonl(OUTPUT / f"{rag_prefix}_decision_ledger.jsonl")
    control_cards = _read_jsonl(OUTPUT / f"{control_prefix}_decision_ledger.jsonl") if control_prefix else []
    gt = _load_ground_truth()

    detail = build_attribution_detail(rag_packs, rag_cards, gt, control_cards=control_cards)
    variant_summary = summarize_by_variant(detail)
    rag_case_summary = summarize_exploded(detail, "rag_case_ids")
    bookskill_summary = summarize_exploded(detail, "bookskill_strategy_ids")
    branch_bookskill = summarize_branch_bookskill(detail)
    harmful_examples = detail[detail["delta_cash"].fillna(0) < 0].sort_values("delta_cash").head(40)

    detail_path = OUTPUT / f"{output_prefix}_detail.csv"
    variant_path = OUTPUT / f"{output_prefix}_variant_summary.csv"
    rag_case_path = OUTPUT / f"{output_prefix}_rag_case_summary.csv"
    bookskill_path = OUTPUT / f"{output_prefix}_bookskill_summary.csv"
    branch_bookskill_path = OUTPUT / f"{output_prefix}_branch_bookskill_summary.csv"
    harmful_path = OUTPUT / f"{output_prefix}_harmful_examples.csv"
    report_path = OUTPUT / f"{output_prefix}.md"

    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    variant_summary.to_csv(variant_path, index=False, encoding="utf-8-sig")
    rag_case_summary.to_csv(rag_case_path, index=False, encoding="utf-8-sig")
    bookskill_summary.to_csv(bookskill_path, index=False, encoding="utf-8-sig")
    branch_bookskill.to_csv(branch_bookskill_path, index=False, encoding="utf-8-sig")
    harmful_examples.to_csv(harmful_path, index=False, encoding="utf-8-sig")
    write_report(
        report_path,
        rag_prefix=rag_prefix,
        control_prefix=control_prefix,
        detail=detail,
        variant_summary=variant_summary,
        rag_case_summary=rag_case_summary,
        bookskill_summary=bookskill_summary,
        branch_bookskill=branch_bookskill,
        harmful_examples=harmful_examples,
        output_paths=[
            detail_path,
            variant_path,
            rag_case_path,
            bookskill_path,
            branch_bookskill_path,
            harmful_path,
        ],
    )

    print("A股研究Agent")
    print(
        f"rag_bookskill_attribution=True rag_prefix={rag_prefix} "
        f"control_prefix={control_prefix or 'none'} rows={len(detail)}"
    )
    print(f"wrote: {report_path}")


def build_attribution_detail(
    rag_packs: list[dict[str, Any]],
    rag_cards: list[dict[str, Any]],
    gt: pd.DataFrame,
    *,
    control_cards: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    pack_by_key = {_join_key(_normalize(row)): _normalize(row) for row in rag_packs}
    control_by_key = {_join_key(_normalize(row)): _normalize(row) for row in control_cards or []}
    returns = _return_lookup(gt)
    cash = _bank_return_20d()
    rows: list[dict[str, Any]] = []
    for card_raw in rag_cards:
        card = _normalize(card_raw)
        if str(card.get("task_mode") or "") not in {"single_stock", "single_stock_watch"}:
            continue
        key = _join_key(card)
        pack = pack_by_key.get(key, {})
        control = control_by_key.get(key, {})
        ret = returns.get((str(card.get("decision_date") or ""), str(card.get("code") or "").zfill(6)))
        weight = _safe_float(card.get("simulated_weight_change"), default=0.0)
        control_weight = _safe_float(control.get("simulated_weight_change"), default=float("nan"))
        cash_value = _cash_adjusted(weight, ret, cash)
        control_cash = _cash_adjusted(control_weight, ret, cash) if control else float("nan")
        delta_weight = weight - control_weight if control and not pd.isna(control_weight) else float("nan")
        delta_cash = cash_value - control_cash if control and not pd.isna(control_cash) else float("nan")
        context = str(pack.get("retrieved_cases_context") or "none")
        risk_tool = _tool_row(pack, "single_stock_risk_calibration_v2_review_queue")
        opportunity_tool = _tool_row(pack, "single_stock_opportunity_scorer_v2")
        row = {
            "variant": card.get("variant", ""),
            "task_mode": card.get("task_mode", ""),
            "valid_block": card.get("valid_block", ""),
            "decision_date": card.get("decision_date", ""),
            "code": str(card.get("code") or "").zfill(6),
            "name": card.get("name", ""),
            "sample_panel_id": card.get("sample_panel_id", "panel_01"),
            "research_grade": card.get("research_grade", ""),
            "simulated_action": card.get("simulated_action", ""),
            "simulated_weight_change": weight,
            "control_research_grade": control.get("research_grade", ""),
            "control_simulated_action": control.get("simulated_action", ""),
            "control_simulated_weight_change": control_weight if control else None,
            "return_20d": ret,
            "cash_adjusted_return_20d": cash_value,
            "control_cash_adjusted_return_20d": control_cash if control else None,
            "delta_weight": delta_weight if control else None,
            "delta_cash": delta_cash if control else None,
            "paired_change_type": _paired_change_type(delta_weight, ret),
            "case_memory_mode": pack.get("case_memory_mode") or card.get("case_memory_mode") or "",
            "rag_useful_context": _has_useful_rag_context(context),
            "rag_applicable_count": context.count("applicability=applicable"),
            "rag_partial_count": context.count("applicability=partial"),
            "rag_case_ids": ";".join(_rag_case_ids(context)),
            "bookskill_strategy_ids": ";".join(_bookskill_ids(pack)),
            "bookskill_count": len(_bookskill_ids(pack)),
            "primary_risk_branch": risk_tool.get("primary_risk_branch", "") if risk_tool else "",
            "risk_tier": risk_tool.get("risk_tier", "") if risk_tool else "",
            "opportunity_status": opportunity_tool.get("policy_status", "") if opportunity_tool else "",
            "opportunity_score": opportunity_tool.get("score", "") if opportunity_tool else "",
            "memory_experience_used": str(card.get("memory_experience_used") or "")[:400],
            "counter_evidence": _render_listish(card.get("counter_evidence"))[:500],
            "research_only": True,
            "not_investment_instruction": True,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_variant(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for variant, group in detail.groupby("variant", sort=True):
        delta = pd.to_numeric(group.get("delta_cash"), errors="coerce")
        weight_delta = pd.to_numeric(group.get("delta_weight"), errors="coerce")
        ret = pd.to_numeric(group.get("return_20d"), errors="coerce")
        rows.append(
            {
                "variant": variant,
                "rows": int(len(group)),
                "rag_useful_context_rate": _mean_bool(group["rag_useful_context"]),
                "changed_weight_rows": int((weight_delta.abs() > 1e-9).sum()),
                "avg_cash_adjusted_return_20d": _mean(group["cash_adjusted_return_20d"]),
                "avg_control_cash_adjusted_return_20d": _mean(group["control_cash_adjusted_return_20d"]),
                "avg_delta_cash": _mean(delta),
                "sum_delta_cash": _sum(delta),
                "lowered_positive": int(((weight_delta < 0) & (ret > 0)).sum()),
                "lowered_negative": int(((weight_delta < 0) & (ret < 0)).sum()),
                "raised_positive": int(((weight_delta > 0) & (ret > 0)).sum()),
                "raised_negative": int(((weight_delta > 0) & (ret < 0)).sum()),
                "unique_stocks": int(group["code"].nunique()),
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return pd.DataFrame(rows)


def summarize_exploded(detail: pd.DataFrame, column: str) -> pd.DataFrame:
    if detail.empty or column not in detail:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in detail.iterrows():
        ids = [item for item in str(row.get(column) or "").split(";") if item and item != "__truncated__"]
        for item_id in ids:
            copied = row.to_dict()
            copied["item_id"] = item_id
            rows.append(copied)
    if not rows:
        return pd.DataFrame()
    exploded = pd.DataFrame(rows)
    out: list[dict[str, Any]] = []
    for item_id, group in exploded.groupby("item_id", sort=True):
        delta = pd.to_numeric(group.get("delta_cash"), errors="coerce")
        ret = pd.to_numeric(group.get("return_20d"), errors="coerce")
        out.append(
            {
                "item_id": item_id,
                "rows": int(len(group)),
                "unique_stocks": int(group["code"].nunique()),
                "variants": ";".join(sorted(set(group["variant"].astype(str)))),
                "positive_20d_rate": round(float((ret > 0).mean()), 4) if ret.notna().any() else None,
                "avg_return_20d": _mean(ret),
                "avg_delta_cash": _mean(delta),
                "sum_delta_cash": _sum(delta),
                "changed_weight_rows": int((pd.to_numeric(group.get("delta_weight"), errors="coerce").abs() > 1e-9).sum()),
            }
        )
    return pd.DataFrame(out).sort_values(["rows", "sum_delta_cash"], ascending=[False, False])


def summarize_branch_bookskill(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in detail.iterrows():
        ids = [item for item in str(row.get("bookskill_strategy_ids") or "").split(";") if item and item != "__truncated__"]
        branch = str(row.get("primary_risk_branch") or "no_risk_branch")
        for strategy_id in ids:
            copied = row.to_dict()
            copied["strategy_id"] = strategy_id
            copied["branch"] = branch
            rows.append(copied)
    if not rows:
        return pd.DataFrame()
    exploded = pd.DataFrame(rows)
    out: list[dict[str, Any]] = []
    for values, group in exploded.groupby(["branch", "strategy_id"], sort=True):
        delta = pd.to_numeric(group.get("delta_cash"), errors="coerce")
        ret = pd.to_numeric(group.get("return_20d"), errors="coerce")
        out.append(
            {
                "branch": values[0],
                "strategy_id": values[1],
                "rows": int(len(group)),
                "unique_stocks": int(group["code"].nunique()),
                "positive_20d_rate": round(float((ret > 0).mean()), 4) if ret.notna().any() else None,
                "avg_return_20d": _mean(ret),
                "avg_delta_cash": _mean(delta),
                "sum_delta_cash": _sum(delta),
            }
        )
    return pd.DataFrame(out).sort_values(["rows", "sum_delta_cash"], ascending=[False, False])


def write_report(
    path: Path,
    *,
    rag_prefix: str,
    control_prefix: str | None,
    detail: pd.DataFrame,
    variant_summary: pd.DataFrame,
    rag_case_summary: pd.DataFrame,
    bookskill_summary: pd.DataFrame,
    branch_bookskill: pd.DataFrame,
    harmful_examples: pd.DataFrame,
    output_paths: list[Path],
) -> None:
    lines = [
        "# Single-Stock RAG / BookSkill Attribution Audit",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Setup",
        "",
        f"- rag_prefix: `{rag_prefix}`",
        f"- control_prefix: `{control_prefix or 'none'}`",
        f"- rows: `{len(detail)}`",
        "- future returns are joined only after decision cards are written; they are not evidence-pack inputs.",
        "",
        "## Variant Summary",
        "",
        _table(variant_summary),
        "",
        "## RAG Case Attribution",
        "",
        _table(rag_case_summary.head(20)),
        "",
        "## BookSkill Attribution",
        "",
        _table(bookskill_summary.head(20)),
        "",
        "## Branch x BookSkill Attribution",
        "",
        _table(branch_bookskill.head(30)),
        "",
        "## Harmful / Missed-Positive Examples",
        "",
        _table(
            harmful_examples[
                [
                    "variant",
                    "decision_date",
                    "code",
                    "name",
                    "return_20d",
                    "delta_weight",
                    "delta_cash",
                    "primary_risk_branch",
                    "bookskill_strategy_ids",
                    "rag_case_ids",
                ]
            ]
            if not harmful_examples.empty
            else harmful_examples
        ),
        "",
        "## Reading Rules",
        "",
        "- `avg_delta_cash` measures RAG-vs-control contribution only when a control prefix is supplied.",
        "- RAG and BookSkill attribution is associative, not causal proof. A strategy_id or case_id can be promoted only after a larger on/off panel and leakage-safe repeated validation.",
        "- RAG is allowed as a failure-condition checklist/counter-evidence. It must not be used as a standalone upgrade rule.",
        "- BookSkill remains mandatory pre-decision material, but no strategy_id should be strengthened unless branch-level paired outcomes repeatedly support it.",
        "",
        "## Outputs",
        "",
        *[f"- `{path.relative_to(ROOT)}`" for path in output_paths],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_ground_truth() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in GT_SOURCES:
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
        if {"date", "code", "return_20d"} <= set(frame.columns):
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
            frame["code"] = frame["code"].astype(str).str.zfill(6)
            frames.append(frame[["date", "code", "return_20d"]])
    if not frames:
        return pd.DataFrame(columns=["date", "code", "return_20d"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["date", "code"], keep="last")


def _return_lookup(gt: pd.DataFrame) -> dict[tuple[str, str], float]:
    if gt.empty:
        return {}
    frame = gt.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["return_20d"] = pd.to_numeric(frame["return_20d"], errors="coerce")
    return {
        (str(row["date"]), str(row["code"]).zfill(6)): float(row["return_20d"])
        for _, row in frame.dropna(subset=["return_20d"]).iterrows()
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"line {line_number} in {path} is not a JSON object")
            rows.append(obj)
    return rows


def _normalize(row: dict[str, Any]) -> dict[str, Any]:
    copied = dict(row)
    copied["code"] = str(copied.get("code", "")).zfill(6)
    copied["sample_panel_id"] = str(copied.get("sample_panel_id") or "panel_01")
    return copied


def _join_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(key, "")) for key in JOIN_KEYS)


def _rag_case_ids(context: str) -> list[str]:
    ids: list[str] = []
    for line in context.splitlines():
        match = re.match(r"\s*-\s*([^|]+)\s*\|", line)
        if match:
            ids.append(match.group(1).strip())
    return ids


def _bookskill_ids(pack: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for row in pack.get("book_skill_candidates") or []:
        if not isinstance(row, dict):
            continue
        strategy_id = str(row.get("strategy_id") or "").strip()
        if strategy_id and strategy_id != "__truncated__":
            ids.append(strategy_id)
    return ids


def _tool_row(pack: dict[str, Any], tool_id: str) -> dict[str, Any]:
    for row in pack.get("quant_tool_summaries") or []:
        if isinstance(row, dict) and row.get("tool_id") == tool_id:
            return row
    return {}


def _has_useful_rag_context(context: str) -> bool:
    return "applicability=" in context or context.startswith("retrieved_cases:\n-")


def _paired_change_type(delta_weight: float, ret: float | None) -> str:
    if pd.isna(delta_weight) or abs(delta_weight) <= 1e-9 or ret is None or pd.isna(ret):
        return "unchanged_or_unpaired"
    if delta_weight < 0 and ret > 0:
        return "lowered_positive"
    if delta_weight < 0 and ret < 0:
        return "lowered_negative"
    if delta_weight > 0 and ret > 0:
        return "raised_positive"
    if delta_weight > 0 and ret < 0:
        return "raised_negative"
    return "changed_flat_return"


def _cash_adjusted(weight: float, ret: float | None, cash_return_20d: float) -> float:
    if ret is None or pd.isna(ret) or pd.isna(weight):
        return float("nan")
    weight = max(0.0, min(1.0, float(weight)))
    return weight * float(ret) + (1 - weight) * cash_return_20d


def _safe_float(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(number):
        return default
    return number


def _render_listish(value: Any) -> str:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return str(value or "")


def _mean(values: Any) -> float | None:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    if not series.notna().any():
        return None
    return round(float(series.mean()), 4)


def _sum(values: Any) -> float | None:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    if not series.notna().any():
        return None
    return round(float(series.sum()), 4)


def _mean_bool(values: Any) -> float:
    series = pd.Series(values).astype(bool)
    if series.empty:
        return 0.0
    return round(float(series.mean()), 4)


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return frame.to_csv(index=False)


def _safe_prefix(value: str | None) -> str:
    if not value:
        return ""
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value)).strip("_")
    return safe or "rag_bookskill_attribution"


if __name__ == "__main__":
    main()
