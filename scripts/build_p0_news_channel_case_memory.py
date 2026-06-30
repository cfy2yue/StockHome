"""Build safe P0 news-channel case memory from offline news audit examples.

The input examples contain posterior returns for audit only. This builder
writes only visible news/soft-gap conditions and countermeasures so future
agents can retrieve a checklist without seeing future outcomes.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "date_generalization"
MEMORY_DIR = ROOT / "memory"
DEFAULT_EXAMPLES = REPORT_DIR / "p0_news_channel_policy_audit_v1_examples.csv"
DEFAULT_OUTPUT = MEMORY_DIR / "p0_news_channel_case_memory_ledger.csv"
DEFAULT_PREFIX = "p0_news_channel_case_memory_v1"

FORBIDDEN_COLUMNS = {
    "return_20d",
    "target_cash20",
    "sim_cash20",
    "raw_positive_20d",
    "future_return",
    "gt_status",
    "pool_excess_20d",
}
FORBIDDEN_TEXT_RE = re.compile(
    r"\breturn_20d\b|target_cash20|future_return|gt_status|pool_excess|sk-[A-Za-z0-9_-]{16,}|[-+]?\d+(\.\d+)?%",
    re.I,
)

FLAG_COLUMNS = [
    "news_missing_questionnaire",
    "news_neutral_no_catalyst",
    "news_opportunity_or_catalyst",
    "news_hard_warning",
    "financial_missing_or_no_event",
    "peer_weak_or_lagging",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--max-per-bucket", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples_path = args.examples if args.examples.is_absolute() else ROOT / args.examples
    examples = pd.read_csv(examples_path, low_memory=False)
    ledger = build_news_case_memory(examples, max_per_bucket=args.max_per_bucket)
    validate_safe_ledger(ledger)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(output, index=False, encoding="utf-8-sig")
    report = write_report(args.report_prefix, output, ledger)
    print(f"ledger={output}")
    print(f"report={report}")
    print(f"rows={len(ledger)}")


def build_news_case_memory(examples: pd.DataFrame, *, max_per_bucket: int = 6) -> pd.DataFrame:
    if examples.empty:
        return pd.DataFrame(columns=output_columns())
    safe = examples.copy()
    for column in FORBIDDEN_COLUMNS:
        if column in safe.columns:
            safe = safe.drop(columns=[column])
    rows: list[dict[str, str]] = []
    for bucket, group in safe.groupby("example_bucket", sort=True):
        for _, row in group.head(max_per_bucket).iterrows():
            conditions = visible_conditions(row)
            rows.append(
                {
                    "case_id": f"P0NEWS-20260630-{len(rows) + 1:03d}",
                    "source_round": "p0_news_channel_policy_audit_v1",
                    "task_mode": "single_stock_watch",
                    "case_bucket": str(bucket),
                    "case_pattern": case_pattern(str(bucket), row, conditions),
                    "visible_conditions": ";".join(conditions) if conditions else "news_context_no_explicit_flag",
                    "countermeasure": countermeasure(str(bucket), conditions),
                    "status": status_for_bucket(str(bucket)),
                    "source_ref": source_ref(row),
                }
            )
    ledger = pd.DataFrame(rows, columns=output_columns())
    return ledger


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
    bucket = str(row.get("example_bucket") or "")
    if "missing_news" in bucket:
        conditions.append("news_missing_no_hard_warning")
    if "opportunity" in bucket:
        conditions.append("news_opportunity_context")
    if "hard_warning" in bucket or boolish(row.get("news_hard_warning")):
        conditions.append("news_hard_warning")
    suggestion = str(row.get("user_operation_suggestion") or "")
    if any(token in suggestion for token in ["试探", "买入", "持有"]):
        conditions.append("small_entry_branch")
    if any(token in suggestion for token in ["等待", "卖出", "减仓", "补数据"]):
        conditions.append("risk_review_or_wait_branch")
    return sorted(dict.fromkeys(conditions))


def case_pattern(bucket: str, row: pd.Series, conditions: list[str]) -> str:
    name = str(row.get("name") or "")
    suggestion = str(row.get("user_operation_suggestion") or "")
    condition_text = " ".join(conditions)
    if "missing_news_success" in bucket:
        theme = "news missing no hard warning but small-entry branch later worked"
    elif "missing_news_false_positive" in bucket:
        theme = "news missing plus soft-gap cluster can still create false-positive small entry"
    elif "missing_news_risk_false_veto" in bucket:
        theme = "news missing plus soft gaps can create false veto or missed positive"
    elif "hard_warning" in bucket:
        theme = "news hard warning requires verification before any add or raise"
    elif "opportunity" in bucket:
        theme = "news opportunity context needs quant peer chip bookskill confirmation"
    else:
        theme = "news channel branch checklist case"
    return f"single_stock news_channel {theme} {condition_text} user_suggestion={suggestion} stock={name}"


def countermeasure(bucket: str, conditions: list[str]) -> str:
    condition_set = set(conditions)
    if "hard_warning" in bucket or "news_hard_warning" in condition_set:
        return "Verify hard-warning news before raising exposure; no add/raise until official context and price/peer/chip evidence are checked."
    if "opportunity" in bucket or "news_opportunity_context" in condition_set:
        return "Treat opportunity news as supportive context only; require quant, peer/chip, BookSkill, or financial confirmation before increasing size."
    if "missing_news_false_positive" in bucket:
        return "News missing plus financial/peer soft gaps should cap position and require one confirming channel before sizing above a small trial."
    if "missing_news_risk_false_veto" in bucket:
        return "Do not blindly zero because news is missing; keep a low observation position or explicit re-entry trigger when no hard counter exists."
    if "missing_news_success" in bucket:
        return "News absence is an uncertainty cap, not a sell signal; if no hard warning exists, retain small-entry floor and define review triggers."
    return "Use as news-channel checklist only; never as standalone alpha."


def status_for_bucket(bucket: str) -> str:
    if "hard_warning" in bucket:
        return "accepted_hard_warning_second_check_no_raise"
    if "opportunity" in bucket:
        return "accepted_opportunity_support_only_not_alpha"
    if "false_positive" in bucket:
        return "open_news_soft_gap_false_positive_case"
    if "risk_false_veto" in bucket:
        return "open_news_missing_false_veto_case"
    if "success" in bucket:
        return "accepted_news_missing_uncertainty_cap_case"
    return "observe_news_case"


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
        raise ValueError(f"Forbidden columns in news case memory: {sorted(forbidden_cols)}")
    text = ledger.to_csv(index=False)
    if FORBIDDEN_TEXT_RE.search(text):
        raise ValueError("Forbidden future/result/key text found in news case memory ledger")


def write_report(prefix: str, output: Path, ledger: pd.DataFrame) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = REPORT_DIR / f"{safe_name(prefix)}.md"
    summary = (
        ledger.groupby(["case_bucket", "status"], dropna=False)
        .agg(rows=("case_id", "count"), unique_refs=("source_ref", "nunique"))
        .reset_index()
    )
    lines = [
        "# P0 News Channel Case Memory v1",
        "",
        "本文件把 P0 新闻通道后验审计转成安全 RAG/Memory 案例卡。输出不包含未来收益、GT、target_cash 或 API key。",
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
        "- 新闻案例只能作为 semantic questionnaire / checklist / false-veto reminder。",
        "- 新闻缺失、新闻机会和硬风险新闻都不得作为单独买卖公式。",
        "- 新一轮 evidence pack 只能检索案例模式和对策，不能携带原始后验收益。",
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
