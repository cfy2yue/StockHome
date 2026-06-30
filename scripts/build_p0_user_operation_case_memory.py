"""Build safe P0 user-operation case memory from offline error examples.

The input examples are post-hoc audit artifacts that may contain future 20d
returns. This builder writes only condition tags, case patterns, source refs,
and countermeasures into memory so retrieved cases can be used as prior
checklists without leaking numeric future outcomes.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "date_generalization"
MEMORY_DIR = ROOT / "memory"
DEFAULT_EXAMPLES = REPORT_DIR / "p0_user_operation_error_pattern_audit_v1_examples.csv"
DEFAULT_OUTPUT = MEMORY_DIR / "p0_user_operation_case_memory_ledger.csv"
DEFAULT_PREFIX = "p0_user_operation_case_memory_v1"
FORBIDDEN_COLUMNS = {
    "return_20d",
    "target_cash20",
    "sim_cash20",
    "raw_positive_20d",
    "future_return",
    "gt_status",
    "pool_excess_20d",
}
FORBIDDEN_TEXT_RE = re.compile(r"\breturn_20d\b|target_cash20|future_return|gt_status|pool_excess|sk-[A-Za-z0-9_-]{16,}", re.I)


FLAG_COLUMNS = [
    "news_missing_or_empty",
    "financial_missing_or_no_event",
    "peer_weak_or_lagging",
    "bookskill_weak_or_missing",
    "chip_overhang_or_trapped",
    "rag_failure_or_case_risk",
    "explicit_or_financial_hard_risk",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build safe P0 user-operation case memory.")
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--max-per-bucket", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = pd.read_csv(args.examples, low_memory=False)
    ledger = build_case_memory(examples, max_per_bucket=args.max_per_bucket)
    validate_safe_ledger(ledger)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(output, index=False, encoding="utf-8-sig")
    report = write_report(args.report_prefix, output, ledger)
    print(f"ledger={output}")
    print(f"report={report}")
    print(f"rows={len(ledger)}")


def build_case_memory(examples: pd.DataFrame, *, max_per_bucket: int = 8) -> pd.DataFrame:
    if examples.empty:
        return pd.DataFrame(columns=output_columns())
    safe = examples.copy()
    for column in FORBIDDEN_COLUMNS:
        if column in safe.columns:
            safe = safe.drop(columns=[column])
    rows: list[dict[str, object]] = []
    grouped = safe.groupby("example_bucket", sort=True)
    for bucket, group in grouped:
        for _, row in group.head(max_per_bucket).iterrows():
            conditions = visible_conditions(row)
            rows.append(
                {
                    "case_id": f"P0CASE-20260630-{len(rows) + 1:03d}",
                    "source_round": "p0_user_operation_error_pattern_audit_v1",
                    "task_mode": "single_stock_watch",
                    "case_bucket": bucket,
                    "case_pattern": case_pattern(bucket, row, conditions),
                    "visible_conditions": ";".join(conditions) if conditions else "no_explicit_condition_tag",
                    "countermeasure": countermeasure(bucket, conditions),
                    "status": status_for_bucket(bucket),
                    "source_ref": source_ref(row),
                }
            )
    return pd.DataFrame(rows, columns=output_columns())


def output_columns() -> list[str]:
    return [
        "case_id",
        "source_round",
        "task_mode",
        "case_bucket",
        "case_pattern",
        "visible_conditions",
        "countermeasure",
        "status",
        "source_ref",
    ]


def visible_conditions(row: pd.Series) -> list[str]:
    conditions: list[str] = []
    for column in FLAG_COLUMNS:
        if boolish(row.get(column)):
            conditions.append(column)
    suggestion = str(row.get("user_operation_suggestion") or "")
    if "试探" in suggestion or "买入" in suggestion or "持有" in suggestion:
        conditions.append("small_entry_branch")
    if "减仓" in suggestion or "卖出" in suggestion or "等待" in suggestion:
        conditions.append("risk_review_or_wait_branch")
    return sorted(dict.fromkeys(conditions))


def case_pattern(bucket: str, row: pd.Series, conditions: list[str]) -> str:
    name = str(row.get("name") or "")
    suggestion = str(row.get("user_operation_suggestion") or "")
    error_type = str(row.get("error_type") or bucket)
    condition_text = " ".join(conditions)
    if bucket == "successful_large_gain_buy":
        return f"single_stock small_entry_branch prior success case {condition_text} user_suggestion={suggestion} stock={name}"
    if bucket == "false_positive_buy":
        return f"single_stock small_entry_branch false_positive_or_large_loss case {condition_text} user_suggestion={suggestion} stock={name}"
    if bucket in {"risk_false_veto_large_gain", "missed_large_gain"}:
        return f"single_stock false_veto_or_missed_positive case {condition_text} user_suggestion={suggestion} stock={name}"
    if bucket == "avoided_large_loss":
        return f"single_stock avoided_large_loss risk_review case {condition_text} user_suggestion={suggestion} stock={name}"
    return f"single_stock {error_type} case {condition_text} user_suggestion={suggestion} stock={name}"


def countermeasure(bucket: str, conditions: list[str]) -> str:
    condition_set = set(conditions)
    if bucket == "successful_large_gain_buy":
        return "If small-entry branch has no hard counter, keep a small trial/hold floor; soft gaps only downsize confidence."
    if bucket == "false_positive_buy":
        if {"news_missing_or_empty", "financial_missing_or_no_event", "peer_weak_or_lagging"} <= condition_set:
            return "When news, financial and peer are all weak/missing, cap position and require one confirming channel before sizing above a small trial."
        return "Keep small-entry floor conservative and require second check before increasing position."
    if bucket in {"risk_false_veto_large_gain", "missed_large_gain"}:
        if "explicit_or_financial_hard_risk" in condition_set:
            return "Hard risk requires second check and no raise, but avoid blind zero unless deterioration persists across price/chip/news/financial channels."
        if "rag_failure_or_case_risk" in condition_set:
            return "RAG failure/case-risk should shrink or trigger review only; do not use it as a standalone veto."
        return "Do not mechanically veto soft gaps; keep low observation position or explicit re-entry trigger."
    if bucket == "avoided_large_loss":
        return "Risk review can prevent losses; keep data-missing and hard-risk checks as review gates, but monitor false veto cost."
    return "Use as checklist only; require fresh-panel validation before promotion."


def status_for_bucket(bucket: str) -> str:
    if bucket == "successful_large_gain_buy":
        return "accepted_prior_positive_case_checklist_only"
    if bucket == "false_positive_buy":
        return "open_failure_case_requires_confirmation"
    if bucket in {"risk_false_veto_large_gain", "missed_large_gain"}:
        return "open_false_veto_case_do_not_blind_zero"
    if bucket == "avoided_large_loss":
        return "accepted_risk_review_case_monitor_false_veto"
    return "observe_case"


def source_ref(row: pd.Series) -> str:
    parts = [
        str(row.get("source_label") or ""),
        str(row.get("valid_block") or ""),
        str(row.get("decision_date") or ""),
        str(row.get("code") or "").zfill(6),
        str(row.get("name") or ""),
    ]
    return "|".join(part for part in parts if part)


def validate_safe_ledger(ledger: pd.DataFrame) -> None:
    forbidden_cols = FORBIDDEN_COLUMNS & set(ledger.columns)
    if forbidden_cols:
        raise ValueError(f"Forbidden columns in case memory: {sorted(forbidden_cols)}")
    text = ledger.to_csv(index=False)
    if FORBIDDEN_TEXT_RE.search(text):
        raise ValueError("Forbidden future/result/key text found in safe case memory ledger")


def write_report(prefix: str, output: Path, ledger: pd.DataFrame) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = REPORT_DIR / f"{safe_name(prefix)}.md"
    summary = (
        ledger.groupby(["case_bucket", "status"], dropna=False)
        .agg(rows=("case_id", "count"), unique_refs=("source_ref", "nunique"))
        .reset_index()
    )
    lines = [
        "# P0 User-Operation Case Memory v1",
        "",
        "本文件把已成熟的 P0 用户操作错误反思转成 RAG/Memory 案例卡。输出不包含未来收益、GT、target_cash 或 API key。",
        "",
        f"- ledger: `{output.relative_to(ROOT)}`",
        f"- rows: `{len(ledger)}`",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Use Boundary",
        "",
        "- 这些案例只能作为 prior checklist / counter-evidence / false-veto reminder。",
        "- 不得把案例当作独立 alpha 或买入公式。",
        "- 新一轮 evidence pack 只能检索案例的模式和对策，不能携带原始未来收益。",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


if __name__ == "__main__":
    main()
