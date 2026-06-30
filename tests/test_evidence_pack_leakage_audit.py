from __future__ import annotations

import json

from scripts.audit_evidence_pack_leakage import audit_jsonl_file


def test_audit_allows_prior_return_20d(tmp_path) -> None:
    path = tmp_path / "pack.jsonl"
    path.write_text(
        json.dumps(
            {
                "code": "000001",
                "python_features": {"prior_return_20d": -3.2},
                "python_signal_summary": "prior_return_20d=-3.2",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    line_count, findings = audit_jsonl_file(path)

    assert line_count == 1
    assert findings == []


def test_audit_flags_future_return_field(tmp_path) -> None:
    path = tmp_path / "pack.jsonl"
    path.write_text(
        json.dumps(
            {
                "code": "000001",
                "return_20d": 8.8,
                "notes": "future return_20d should not be visible",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    line_count, findings = audit_jsonl_file(path)

    assert line_count == 1
    assert {finding.token for finding in findings} == {"return_20d"}
    assert {finding.finding_type for finding in findings} == {"future_result_key", "future_result_string_token"}
