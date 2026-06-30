from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FUTURE_RESULT_FIELDS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "gt_status",
}

ALLOWED_HISTORICAL_FIELDS = {
    "prior_return_20d",
}

TOKEN_PATTERN = re.compile(r"\b(return_5d|return_10d|return_20d|future_return_5d|future_return_10d|future_return_20d|gt_status)\b")


@dataclass(frozen=True)
class LeakFinding:
    line_number: int
    path: str
    token: str
    finding_type: str


def _is_allowed_token(text: str, token: str) -> bool:
    if token == "return_20d" and "prior_return_20d" in text:
        stripped = text.replace("prior_return_20d", "")
        return not TOKEN_PATTERN.search(stripped)
    return token in ALLOWED_HISTORICAL_FIELDS


def _scan_value(value: Any, *, line_number: int, path: str) -> list[LeakFinding]:
    findings: list[LeakFinding] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if key_text in FUTURE_RESULT_FIELDS:
                findings.append(LeakFinding(line_number, child_path, key_text, "future_result_key"))
            elif key_text not in ALLOWED_HISTORICAL_FIELDS:
                for match in TOKEN_PATTERN.finditer(key_text):
                    token = match.group(1)
                    if not _is_allowed_token(key_text, token):
                        findings.append(LeakFinding(line_number, child_path, token, "future_result_key_token"))
            findings.extend(_scan_value(child, line_number=line_number, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_scan_value(child, line_number=line_number, path=f"{path}[{index}]"))
    elif isinstance(value, str):
        for match in TOKEN_PATTERN.finditer(value):
            token = match.group(1)
            if not _is_allowed_token(value, token):
                findings.append(LeakFinding(line_number, path, token, "future_result_string_token"))
    return findings


def audit_jsonl_file(path: Path) -> tuple[int, list[LeakFinding]]:
    line_count = 0
    findings: list[LeakFinding] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            line_count += 1
            payload = json.loads(line)
            findings.extend(_scan_value(payload, line_number=line_number, path=""))
    return line_count, findings


def _write_report(path: Path, *, evidence_path: Path, line_count: int, findings: list[LeakFinding]) -> None:
    status = "pass" if not findings else "fail"
    lines = [
        "# Evidence Pack Leakage Audit",
        "",
        "本报告只检查 DeepSeek evidence pack 是否包含未来收益/GT 字段；`prior_return_20d` 是历史特征，默认允许。",
        "",
        f"- evidence_pack: `{evidence_path}`",
        f"- jsonl_records: `{line_count}`",
        f"- status: `{status}`",
        f"- finding_count: `{len(findings)}`",
        "",
    ]
    if findings:
        lines.extend(
            [
                "| line | path | token | type |",
                "|---:|---|---|---|",
            ]
        )
        for finding in findings:
            lines.append(f"| {finding.line_number} | `{finding.path}` | `{finding.token}` | `{finding.finding_type}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit generated DeepSeek evidence-pack JSONL for future result leakage.")
    parser.add_argument("evidence_pack", type=Path)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    line_count, findings = audit_jsonl_file(args.evidence_pack)
    if args.report:
        _write_report(args.report, evidence_path=args.evidence_pack, line_count=line_count, findings=findings)
    print(f"A股研究Agent")
    print(f"evidence_pack={args.evidence_pack}")
    print(f"jsonl_records={line_count}")
    print(f"future_leak_findings={len(findings)}")
    if findings:
        for finding in findings[:20]:
            print(
                "finding "
                f"line={finding.line_number} path={finding.path} token={finding.token} type={finding.finding_type}"
            )
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
