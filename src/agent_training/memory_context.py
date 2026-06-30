from __future__ import annotations

import csv
import re
from pathlib import Path


DEFAULT_LEDGER_FILES = [
    "memory/strategy_experience_ledger.csv",
    "memory/book_skill_adaptation_ledger.csv",
    "memory/news_world_model_ledger.csv",
    "memory/ablation_findings_ledger.csv",
    "memory/failure_case_ledger.csv",
]

PREFERRED_COLUMNS = {
    "memory/strategy_experience_ledger.csv": [
        "experience_id",
        "rule_or_observation",
        "accepted_or_rejected",
        "failure_condition",
        "next_action",
    ],
    "memory/book_skill_adaptation_ledger.csv": [
        "experience_id",
        "strategy_id",
        "rule_or_observation",
        "accepted_or_rejected",
        "failure_condition",
        "next_action",
    ],
    "memory/news_world_model_ledger.csv": [
        "experience_id",
        "rule_or_observation",
        "accepted_or_rejected",
        "failure_condition",
        "next_action",
    ],
    "memory/ablation_findings_ledger.csv": [
        "experience_id",
        "rule_or_observation",
        "accepted_or_rejected",
        "failure_condition",
        "next_action",
    ],
    "memory/failure_case_ledger.csv": [
        "failure_id",
        "failure_pattern",
        "countermeasure",
        "status",
    ],
}

FORBIDDEN_PROMPT_COLUMNS = {
    "metric_before",
    "metric_after",
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "gt_status",
    "gt_pass",
}

FORBIDDEN_PROMPT_TOKEN_RE = re.compile(
    r"(return_5d|return_10d|return_20d|future_return_5d|future_return_10d|future_return_20d|gt_status|gt_pass|metric_before|metric_after)",
    re.IGNORECASE,
)


def load_compact_memory_context(
    root: Path,
    *,
    ledger_files: list[str] | None = None,
    rows_per_file: int = 2,
    max_chars: int = 3600,
    row_char_limit: int = 260,
) -> str:
    files = ledger_files or DEFAULT_LEDGER_FILES
    sections: list[str] = []
    for rel in files:
        path = root / rel
        if not path.exists():
            continue
        rows = _read_csv_rows(path)
        if not rows:
            continue
        selected = rows[-rows_per_file:]
        lines = [f"## {rel}"]
        for row in selected:
            lines.append(_format_row(rel, row, row_char_limit=row_char_limit))
        sections.append("\n".join(lines))
    text = "\n\n".join(sections).strip()
    if not text:
        return "none"
    if len(text) <= max_chars:
        return text
    return _tail_by_lines(text, max_chars=max_chars)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _format_row(rel: str, row: dict[str, str], *, row_char_limit: int) -> str:
    columns = [col for col in PREFERRED_COLUMNS.get(rel, []) if col in row and col not in FORBIDDEN_PROMPT_COLUMNS]
    if not columns:
        columns = [col for col in row if col not in FORBIDDEN_PROMPT_COLUMNS][:6]
    first = columns[0] if columns else "row"
    parts = [_sanitize_prompt_text(str(row.get(first, "")).strip()) or "row"]
    for col in columns[1:]:
        value = _sanitize_prompt_text(str(row.get(col, "")).strip())
        if value:
            parts.append(f"{col}={value}")
    text = " - " + "; ".join(parts)
    return text[: row_char_limit - 3] + "..." if len(text) > row_char_limit else text


def _sanitize_prompt_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"sk-(?=[A-Za-z0-9_-])", "sk_", text)
    parts = re.split(r"([;，。])", text)
    kept: list[str] = []
    for index in range(0, len(parts), 2):
        segment = parts[index]
        delimiter = parts[index + 1] if index + 1 < len(parts) else ""
        if FORBIDDEN_PROMPT_TOKEN_RE.search(segment):
            kept.append("[redacted_hindsight_metric]")
            if delimiter:
                kept.append(delimiter)
            continue
        kept.append(segment)
        if delimiter:
            kept.append(delimiter)
    return "".join(kept).strip()


def _tail_by_lines(text: str, *, max_chars: int) -> str:
    kept: list[str] = []
    total = 0
    for line in reversed(text.splitlines()):
        addition = len(line) + (1 if kept else 0)
        if kept and total + addition > max_chars:
            break
        if not kept and len(line) > max_chars:
            return line[: max_chars - 3] + "..."
        kept.append(line)
        total += addition
    return "\n".join(reversed(kept)).strip()
