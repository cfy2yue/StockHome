"""Audit why Agent decisions do or do not adopt accepted quant tools."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


JOIN_KEYS = ["variant", "task_mode", "valid_block", "decision_date", "code", "sample_panel_id"]
REASON_ORDER = [
    "news_gap",
    "financial_gap",
    "peer_gap",
    "bookskill_gap",
    "chip_overhang",
    "overheat_or_volatility",
    "data_missing",
    "memory_or_rag_counter",
]


def audit_agent_input_gap(evidence_pack: Path, decision_ledger: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    packs = [_normalize_key(row) for row in _read_jsonl(evidence_pack)]
    cards = [_normalize_key(row) for row in _read_jsonl(decision_ledger)]
    pack_by_key = {_join_key(row): row for row in packs}
    detail_rows: list[dict[str, Any]] = []
    for card in cards:
        pack = pack_by_key.get(_join_key(card), {})
        reasons = _split_reasons(card.get("quant_tool_override_reasons"))
        row = {
            "variant": card.get("variant", ""),
            "task_mode": card.get("task_mode", ""),
            "valid_block": card.get("valid_block", ""),
            "decision_date": card.get("decision_date", ""),
            "code": card.get("code", ""),
            "name": card.get("name", ""),
            "research_grade": card.get("research_grade", ""),
            "simulated_action": card.get("simulated_action", ""),
            "quant_tool_adoption_decision": card.get("quant_tool_adoption_decision", "missing"),
            "quant_tool_override_reasons": ";".join(reasons) if reasons else "none",
            "accepted_tool_available": _accepted_tool_available(pack),
            "actionable_news_available": _actionable_news_available(pack),
            "semantic_news_available": _semantic_news_available(pack),
            "financial_event_asof_available": _financial_event_asof_available(pack),
            "grounded_bookskill_available": _grounded_bookskill_available(pack),
            "peer_positive_confirmation": _peer_positive_confirmation(pack),
            "chip_overhang_flag": _chip_overhang_flag(pack),
            "retrieved_cases_available": _retrieved_cases_available(pack),
            "data_missing_flag": _data_missing_flag(pack, card),
            "research_only": True,
            "not_investment_instruction": True,
        }
        for reason in REASON_ORDER:
            row[f"reason_{reason}"] = reason in reasons
        detail_rows.append(row)
    detail = pd.DataFrame(detail_rows)
    return detail, summarize_agent_input_gap(detail)


def summarize_agent_input_gap(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["variant", "task_mode", "valid_block"]
    for values, group in detail.groupby(group_cols, dropna=False, sort=True):
        row: dict[str, Any] = {
            "variant": values[0],
            "task_mode": values[1],
            "valid_block": values[2],
            "cards": int(len(group)),
            "exposure_cards": int(group["simulated_action"].astype(str).eq("增加研究暴露").sum()),
            "accepted_tool_available_rate": _mean_bool(group, "accepted_tool_available"),
            "actionable_news_rate": _mean_bool(group, "actionable_news_available"),
            "semantic_news_rate": _mean_bool(group, "semantic_news_available"),
            "financial_event_asof_rate": _mean_bool(group, "financial_event_asof_available"),
            "grounded_bookskill_rate": _mean_bool(group, "grounded_bookskill_available"),
            "peer_positive_rate": _mean_bool(group, "peer_positive_confirmation"),
            "chip_overhang_rate": _mean_bool(group, "chip_overhang_flag"),
            "retrieved_cases_rate": _mean_bool(group, "retrieved_cases_available"),
            "data_missing_rate": _mean_bool(group, "data_missing_flag"),
            "research_only": True,
            "not_investment_instruction": True,
        }
        adoption_counts = Counter(group["quant_tool_adoption_decision"].astype(str))
        for status in ["adopted", "partially_adopted", "not_adopted_counter_evidence", "not_applicable"]:
            row[f"adoption_{status}"] = int(adoption_counts.get(status, 0))
        for reason in REASON_ORDER:
            row[f"reason_{reason}_rate"] = _mean_bool(group, f"reason_{reason}")
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(path: Path, *, evidence_pack: Path, decision_ledger: Path, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    priorities = _priority_rows(detail)
    lines = [
        "# Agent Input Gap Audit",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        f"- evidence_pack: `{evidence_pack}`",
        f"- decision_ledger: `{decision_ledger}`",
        f"- cards: `{len(detail)}`",
        "",
        "## Summary",
        "",
        _table(summary),
        "",
        "## Channel Priorities",
        "",
        _table(priorities),
        "",
        "## Detail Sample",
        "",
        _table(detail.head(40)),
        "",
        "## Reading Rules",
        "",
        "- `*_rate` 只描述 evidence/decision card 中的可见输入，不读取未来收益。",
        "- `actionable_news_rate` 要求目标自身近期新闻或结构化新闻问卷有实质内容；单纯 `news_missing_rate=1` 不算正向覆盖。",
        "- `financial_event_asof_rate` 要求有 as-of 财报/公告事件；缺披露日或 code 未入表不算可用。",
        "- accepted 量化工具被反证覆盖时，优先补高频出现且低覆盖的通道，而不是强行调高研究暴露。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _priority_rows(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    mapping = [
        ("news_gap", "actionable_news_available", "补目标/行业/政策/地域新闻与公告问卷覆盖"),
        ("financial_gap", "financial_event_asof_available", "补真实披露日财报与公告事件 as-of 通道"),
        ("peer_gap", "peer_positive_confirmation", "补同行/地域/产业链相对强度确认"),
        ("bookskill_gap", "grounded_bookskill_available", "补 BookSkill 适用/失效条件 grounding"),
        ("memory_or_rag_counter", "retrieved_cases_available", "补时间安全 RAG 案例检索或关闭误导性记忆"),
    ]
    rows = []
    for reason, coverage_col, action in mapping:
        reason_col = f"reason_{reason}"
        rows.append(
            {
                "gap": reason,
                "reason_cards": int(detail[reason_col].sum()) if reason_col in detail else 0,
                "reason_rate": round(float(detail[reason_col].mean()), 4) if reason_col in detail else 0.0,
                "coverage_rate": _mean_bool(detail, coverage_col) if coverage_col in detail else 0.0,
                "next_action": action,
            }
        )
    return pd.DataFrame(rows).sort_values(["reason_cards", "coverage_rate"], ascending=[False, True])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _normalize_key(row: dict[str, Any]) -> dict[str, Any]:
    copied = dict(row)
    copied["code"] = str(copied.get("code", "")).zfill(6)
    copied["sample_panel_id"] = str(copied.get("sample_panel_id") or "panel_01")
    return copied


def _join_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(key, "")) for key in JOIN_KEYS)


def _split_reasons(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "nan", "null", "not_applicable"}:
        return []
    return [part.strip() for part in text.replace(",", ";").split(";") if part.strip()]


def _accepted_tool_available(pack: dict[str, Any]) -> bool:
    summaries = pack.get("quant_tool_summaries")
    if not isinstance(summaries, list):
        return False
    return any(isinstance(item, dict) and item.get("promotion_status") == "accepted_cost_recheck_candidate" for item in summaries)


def _actionable_news_available(pack: dict[str, Any]) -> bool:
    features = pack.get("news_features") if isinstance(pack.get("news_features"), dict) else {}
    count = _to_float(features.get("news_count_30d"))
    if count and count > 0:
        return True
    return _semantic_news_available(pack)


def _semantic_news_available(pack: dict[str, Any]) -> bool:
    questionnaire = pack.get("news_semantic_questionnaire")
    if not isinstance(questionnaire, dict):
        return False
    for key, value in questionnaire.items():
        if key == "news_semantic_questionnaire_version":
            continue
        if _meaningful(value):
            return True
    return False


def _financial_event_asof_available(pack: dict[str, Any]) -> bool:
    features = pack.get("financial_report_features") if isinstance(pack.get("financial_report_features"), dict) else {}
    event_count = _to_float(features.get("financial_report_event_count"))
    available_at = str(features.get("financial_report_available_at") or "").strip().lower()
    join_status = str(features.get("financial_report_join_status") or "").strip().lower()
    missing_rate = _to_float(features.get("financial_report_missing_rate"))
    if event_count and event_count > 0 and available_at not in {"", "none", "nan", "null"}:
        return True
    return bool(join_status and join_status not in {"code_not_in_feature_table", "none", "nan", "null"} and missing_rate is not None and missing_rate < 1)


def _grounded_bookskill_available(pack: dict[str, Any]) -> bool:
    candidates = pack.get("book_skill_candidates")
    if not isinstance(candidates, list):
        return False
    return any(
        isinstance(item, dict)
        and item.get("source_status") == "grounded"
        and bool(item.get("strategy_id"))
        and bool(item.get("source_book"))
        for item in candidates
    )


def _peer_positive_confirmation(pack: dict[str, Any]) -> bool:
    features = pack.get("peer_context_features") if isinstance(pack.get("peer_context_features"), dict) else {}
    breadth = _to_float(features.get("tushare_industry_positive_breadth_20d"))
    relative = _to_float(features.get("tushare_industry_relative_return_20d"))
    area_breadth = _to_float(features.get("tushare_area_positive_breadth_20d"))
    area_relative = _to_float(features.get("tushare_area_relative_return_20d"))
    industry_ok = breadth is not None and breadth >= 0.5 and relative is not None and relative >= 0
    area_ok = area_breadth is not None and area_breadth >= 0.5 and area_relative is not None and area_relative >= 0
    return industry_ok or area_ok


def _chip_overhang_flag(pack: dict[str, Any]) -> bool:
    features = pack.get("chip_features") if isinstance(pack.get("chip_features"), dict) else {}
    upper = _to_float(features.get("upper_overhang"))
    width = _to_float(features.get("cost_band_width"))
    return bool((upper is not None and upper >= 1.0) or (width is not None and width >= 1.5))


def _retrieved_cases_available(pack: dict[str, Any]) -> bool:
    text = str(pack.get("retrieved_cases_context") or "").strip().lower()
    return bool(text and text not in {"none", "nan", "null"})


def _data_missing_flag(pack: dict[str, Any], card: dict[str, Any]) -> bool:
    text = f"{pack.get('data_missing_flags') or ''};{card.get('data_missing_flags') or ''}".strip().lower()
    return bool(text and text not in {"none", "nan", "null", "无"})


def _mean_bool(frame: pd.DataFrame, column: str) -> float:
    if column not in frame or frame.empty:
        return 0.0
    return round(float(frame[column].fillna(False).astype(bool).mean()), 4)


def _meaningful(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(value, (int, float, bool)):
        return True
    text = str(value).strip().lower()
    return bool(text and text not in {"none", "nan", "null", "na"})


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
        if pd.isna(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    return frame.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Agent input gaps behind quant-tool adoption decisions.")
    parser.add_argument("--evidence-pack", type=Path, required=True)
    parser.add_argument("--decision-ledger", type=Path, required=True)
    parser.add_argument("--output-prefix", default="")
    args = parser.parse_args()

    prefix = args.output_prefix or args.decision_ledger.name.replace("_decision_ledger.jsonl", "")
    out_dir = args.decision_ledger.parent
    detail, summary = audit_agent_input_gap(args.evidence_pack, args.decision_ledger)
    detail_path = out_dir / f"{prefix}_agent_input_gap_detail.csv"
    summary_path = out_dir / f"{prefix}_agent_input_gap_summary.csv"
    report_path = out_dir / f"{prefix}_agent_input_gap_audit.md"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_report(report_path, evidence_pack=args.evidence_pack, decision_ledger=args.decision_ledger, detail=detail, summary=summary)
    print("A股研究Agent")
    print(f"cards={len(detail)}")
    print(f"wrote: {detail_path}")
    print(f"wrote: {summary_path}")
    print(f"wrote: {report_path}")


if __name__ == "__main__":
    main()
