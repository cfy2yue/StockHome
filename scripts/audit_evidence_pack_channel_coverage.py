from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


CHANNEL_COLUMNS = [
    "python_features",
    "quant_tool_summaries",
    "quant_tool_signal_summary",
    "kline_features",
    "peer_context_features",
    "news_features",
    "news_semantic_questionnaire",
    "news_branch_case_context",
    "analogue_case_context",
    "nonprice_risk_overlay_context",
    "financial_report_features",
    "book_skill_candidates",
    "memory_context",
    "retrieved_cases_context",
    "counter_evidence",
    "data_missing_flags",
]


def audit_evidence_pack_channel_coverage(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            pack = json.loads(line)
            row = {
                "line_number": line_number,
                "code": str(pack.get("code", "")),
                "decision_date": str(pack.get("decision_date", "")),
                "task_mode": str(pack.get("task_mode", "")),
                "variant": str(pack.get("variant", "")),
            }
            for channel in CHANNEL_COLUMNS:
                row[f"{channel}_nonempty"] = _has_signal(pack.get(channel))
            candidates = pack.get("book_skill_candidates") if isinstance(pack.get("book_skill_candidates"), list) else []
            row["book_skill_count"] = len(candidates)
            row["book_skill_grounded_count"] = sum(1 for item in candidates if isinstance(item, dict) and item.get("source_status") == "grounded")
            row["book_skill_missing_grounded_count"] = sum(
                1 for item in candidates if isinstance(item, dict) and item.get("source_status") == "missing_grounded_card"
            )
            row["book_skill_has_source_detail"] = any(
                isinstance(item, dict) and bool(item.get("source_book")) and bool(item.get("page_range")) for item in candidates
            )
            detail_rows.append(row)
    detail = pd.DataFrame(detail_rows)
    summary_rows: list[dict[str, Any]] = []
    total = len(detail)
    for channel in CHANNEL_COLUMNS:
        col = f"{channel}_nonempty"
        nonempty = int(detail[col].sum()) if col in detail else 0
        summary_rows.append(
            {
                "channel": channel,
                "records": total,
                "nonempty": nonempty,
                "coverage_rate": round(nonempty / total, 4) if total else 0.0,
            }
        )
    if total:
        summary_rows.append(
            {
                "channel": "book_skill_grounded_source_detail",
                "records": total,
                "nonempty": int(detail["book_skill_has_source_detail"].sum()),
                "coverage_rate": round(float(detail["book_skill_has_source_detail"].mean()), 4),
            }
        )
    return detail, pd.DataFrame(summary_rows)


def write_report(report_path: Path, *, evidence_path: Path, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Evidence Pack Channel Coverage Audit",
        "",
        "本报告检查 DeepSeek evidence pack 中各输入通道是否真的非空；它不评估收益，也不读取 API key/token。",
        "",
        f"- evidence_pack: `{evidence_path}`",
        f"- records: `{len(detail)}`",
        "",
        "## Channel Coverage",
        "",
        _table(summary),
        "",
        "## Book Skill Source Detail",
        "",
    ]
    if detail.empty:
        lines.append("- no records")
    else:
        grounded = int(detail["book_skill_grounded_count"].sum())
        missing = int(detail["book_skill_missing_grounded_count"].sum())
        source_detail = int(detail["book_skill_has_source_detail"].sum())
        lines.extend(
            [
                f"- grounded_skill_references: `{grounded}`",
                f"- missing_grounded_skill_references: `{missing}`",
                f"- packs_with_book_skill_source_detail: `{source_detail}`",
                "",
                "## Detail Sample",
                "",
                _table(detail.head(30)),
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _has_signal(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return any(_has_signal(child) for child in value.values())
    if isinstance(value, list):
        return any(_has_signal(child) for child in value)
    if isinstance(value, (int, float, bool)):
        return True
    text = str(value).strip()
    if not text:
        return False
    return text.lower() not in {"none", "nan", "null", "no python signal", "无强反证"}


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    return frame.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit channel-level nonempty coverage for DeepSeek evidence-pack JSONL.")
    parser.add_argument("evidence_pack", type=Path)
    parser.add_argument("--detail-csv", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    detail, summary = audit_evidence_pack_channel_coverage(args.evidence_pack)
    if args.detail_csv:
        args.detail_csv.parent.mkdir(parents=True, exist_ok=True)
        detail.to_csv(args.detail_csv, index=False, encoding="utf-8-sig")
    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_csv, index=False, encoding="utf-8-sig")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_report(args.report, evidence_path=args.evidence_pack, detail=detail, summary=summary)
    print("A股研究Agent")
    print(f"evidence_pack={args.evidence_pack}")
    print(f"records={len(detail)}")
    if not summary.empty:
        low = summary[summary["coverage_rate"] < 0.5]["channel"].tolist()
        print(f"low_coverage_channels={','.join(low) if low else 'none'}")


if __name__ == "__main__":
    main()
