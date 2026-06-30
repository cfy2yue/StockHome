"""Audit P0 transfer analog/RAG on/off dry-run evidence packs.

The audit is structural only: channel visibility, row-level analogue matching,
and future-result field leakage. It does not use realized returns.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


REPORT_DIR = Path(__file__).resolve().parents[1] / "reports" / "date_generalization"
DEFAULT_EVIDENCE = REPORT_DIR / "p0_transfer_analog_rag_onoff_dryrun_v1_evidence_pack.jsonl"
DEFAULT_PREFIX = "p0_transfer_analog_rag_onoff_dryrun_v1"
FUTURE_RESULT_FIELDS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "fwd_ret_20d",
    "positive_20d",
    "loss_gt5",
    "gt_status",
    "gt_pass",
    "pool_excess_20d",
    "rule_outcome_label",
    "label",
    "target_label",
    "outcome",
}
TOKEN_PATTERN = re.compile(
    r"\b(return_5d|return_10d|return_20d|future_return_5d|future_return_10d|future_return_20d|fwd_ret_20d|positive_20d|loss_gt5|gt_status|gt_pass|pool_excess_20d|rule_outcome_label|target_label)\b"
)
ALLOWED_STRING_TOKENS = {"prior_return_20d"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze P0 transfer analog/RAG on/off dry-run evidence packs.")
    parser.add_argument("--evidence-pack", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packs = load_jsonl(args.evidence_pack)
    detail = build_visibility_detail(packs)
    summary = build_visibility_summary(detail)
    status = build_status(summary)
    paths = write_outputs(args.output_prefix, args.evidence_pack, detail, summary, status)
    print("A股研究Agent")
    print(f"evidence_packs={len(packs)}")
    print(f"status={status}")
    print(f"report={paths['report']}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing evidence pack: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_visibility_detail(packs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pack in packs:
        analogue = pack.get("analogue_case_context") if isinstance(pack.get("analogue_case_context"), list) else []
        decision_date = str(pack.get("decision_date") or "")
        code = str(pack.get("code") or "").zfill(6)
        row_level_analogue = [
            item
            for item in analogue
            if isinstance(item, dict)
            and str(item.get("date") or "") == decision_date
            and str(item.get("code") or "").zfill(6) == code
        ]
        rows.append(
            {
                "variant": str(pack.get("variant") or ""),
                "task_mode": str(pack.get("task_mode") or ""),
                "valid_block": str(pack.get("valid_block") or ""),
                "decision_date": decision_date,
                "code": code,
                "analogue_visible": int(bool(analogue)),
                "row_level_analogue_visible": int(bool(row_level_analogue)),
                "analogue_context_count": int(len(analogue)),
                "chip_visible": int(nonempty(pack.get("chip_features"))),
                "financial_visible": int(nonempty(pack.get("financial_report_features"))),
                "news_visible": int(nonempty(pack.get("news_features")) or nonempty(pack.get("news_semantic_questionnaire"))),
                "peer_visible": int(nonempty(pack.get("peer_context_features"))),
                "bookskill_visible": int(nonempty(pack.get("book_skill_candidates"))),
                "quant_tool_visible": int(nonempty(pack.get("quant_tool_summaries"))),
                "python_visible": int(nonempty(pack.get("python_features"))),
                "kline_visible": int(nonempty(pack.get("kline_features"))),
                "future_key_leak_count": int(len(find_future_result_tokens(pack))),
            }
        )
    return pd.DataFrame(rows)


def build_visibility_summary(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    metric_cols = [
        "analogue_visible",
        "row_level_analogue_visible",
        "analogue_context_count",
        "chip_visible",
        "financial_visible",
        "news_visible",
        "peer_visible",
        "bookskill_visible",
        "quant_tool_visible",
        "python_visible",
        "kline_visible",
        "future_key_leak_count",
    ]
    rows: list[dict[str, Any]] = []
    for (variant, task_mode, valid_block), group in detail.groupby(["variant", "task_mode", "valid_block"], sort=True):
        row: dict[str, Any] = {
            "variant": variant,
            "task_mode": task_mode,
            "valid_block": valid_block,
            "evidence_packs": int(len(group)),
        }
        for col in metric_cols:
            row[f"{col}_sum"] = int(pd.to_numeric(group[col], errors="coerce").fillna(0).sum())
            if col.endswith("_count"):
                row[f"{col}_mean"] = float(pd.to_numeric(group[col], errors="coerce").fillna(0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def build_status(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "fail_empty"
    checks = [
        variant_sum(summary, "full_agent", "row_level_analogue_visible_sum") > 0,
        variant_sum(summary, "no_analogue_case_context", "analogue_visible_sum") == 0,
        variant_sum(summary, "no_chip_context", "chip_visible_sum") == 0,
        variant_sum(summary, "no_financial_report", "financial_visible_sum") == 0,
        variant_sum(summary, "no_news", "news_visible_sum") == 0,
        variant_sum(summary, "no_peer", "peer_visible_sum") == 0,
        variant_sum(summary, "no_bookskill", "bookskill_visible_sum") == 0,
        variant_sum(summary, "no_quant_tools", "quant_tool_visible_sum") == 0,
        int(pd.to_numeric(summary.get("future_key_leak_count_sum", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) == 0,
    ]
    return "pass" if all(checks) else "fail"


def write_outputs(
    prefix: str,
    evidence_path: Path,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    status: str,
) -> dict[str, Path]:
    safe = safe_prefix(prefix)
    paths = {
        "detail": REPORT_DIR / f"{safe}_onoff_visibility_detail.csv",
        "summary": REPORT_DIR / f"{safe}_onoff_visibility_summary.csv",
        "report": REPORT_DIR / f"{safe}_onoff_visibility_audit.md",
    }
    detail.to_csv(paths["detail"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    paths["report"].write_text(render_report(evidence_path, summary, status), encoding="utf-8")
    return paths


def render_report(evidence_path: Path, summary: pd.DataFrame, status: str) -> str:
    return "\n".join(
        [
            "# P0 Transfer Analog/RAG On/Off Visibility Audit",
            "",
            "本报告只检查 dry-run evidence pack 的输入通道隔离，不使用未来收益。",
            "",
            f"- evidence_pack: `{evidence_path}`",
            f"- status: `{status}`",
            "",
            "## Summary",
            "",
            markdown_table(
                summary,
                [
                    "variant",
                    "task_mode",
                    "valid_block",
                    "evidence_packs",
                    "row_level_analogue_visible_sum",
                    "analogue_visible_sum",
                    "chip_visible_sum",
                    "financial_visible_sum",
                    "news_visible_sum",
                    "peer_visible_sum",
                    "bookskill_visible_sum",
                    "quant_tool_visible_sum",
                    "future_key_leak_count_sum",
                ],
            ),
            "",
        ]
    )


def variant_sum(summary: pd.DataFrame, variant: str, column: str) -> int:
    if column not in summary:
        return 0
    subset = summary[summary["variant"].astype(str).eq(variant)]
    if subset.empty:
        return 0
    return int(pd.to_numeric(subset[column], errors="coerce").fillna(0).sum())


def nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return bool(value)
    if isinstance(value, list):
        return bool(value)
    if isinstance(value, str):
        text = value.strip()
        return bool(text and text.lower() not in {"none", "null", "component ablation: hidden"})
    return bool(value)


def find_future_result_tokens(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text in FUTURE_RESULT_FIELDS:
                found.append(key_text)
            elif key_text not in ALLOWED_STRING_TOKENS:
                for match in TOKEN_PATTERN.finditer(key_text):
                    token = match.group(1)
                    if token not in ALLOWED_STRING_TOKENS:
                        found.append(token)
            found.extend(find_future_result_tokens(child))
    elif isinstance(value, list):
        for item in value:
            found.extend(find_future_result_tokens(item))
    elif isinstance(value, str):
        text = value.replace("prior_return_20d", "")
        for match in TOKEN_PATTERN.finditer(text):
            found.append(match.group(1))
    return found


def safe_prefix(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value)).strip("_")
    return safe or DEFAULT_PREFIX


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows available._"
    cols = [col for col in columns if col in frame]
    table = frame[cols].fillna("").astype(str)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in table.values.tolist()]
    return "\n".join([header, sep, *body])


if __name__ == "__main__":
    main()
