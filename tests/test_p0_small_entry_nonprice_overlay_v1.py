from __future__ import annotations

import pandas as pd
import pytest

from scripts.audit_p0_small_entry_nonprice_overlay_v1 import (
    assert_no_future_fields,
    confirmation_count,
    overlay_rules,
    preview_num,
    write_jsonl,
)


def test_confirmation_count_uses_multiple_nonprice_channels() -> None:
    frame = pd.DataFrame(
        [
            {
                "news_warning_score": 0.1,
                "financial_report_missing_rate": 0.0,
                "financial_quality_risk_score": 0.2,
                "corr_peer_relative_return_20d": 1.0,
                "tushare_industry_positive_breadth_20d": 0.6,
                "lower_support": 0.2,
                "upper_overhang": 0.1,
                "official_confirmation_score": 1.0,
            },
            {
                "news_warning_score": 0.7,
                "financial_report_missing_rate": 1.0,
                "financial_quality_risk_score": 0.9,
                "corr_peer_relative_return_20d": -2.0,
                "tushare_industry_positive_breadth_20d": 0.3,
                "lower_support": 0.01,
                "upper_overhang": 0.5,
                "official_confirmation_score": 0.0,
            },
        ]
    )
    counts = confirmation_count(frame)
    assert counts.iloc[0] >= 5
    assert counts.iloc[1] == 0


def test_overlay_rules_include_baseline_and_clean_confirmed() -> None:
    rule_ids = {rule.rule_id for rule in overlay_rules()}
    assert "small_entry_all" in rule_ids
    assert "small_entry_clean_confirmed" in rule_ids
    assert "news_available_low_warning" in rule_ids


def test_preview_future_key_guard() -> None:
    assert_no_future_fields({"rule_id": "small_entry_all", "score": 1})
    with pytest.raises(ValueError):
        assert_no_future_fields({"gt_status": "evaluated"})


def test_preview_jsonl_writes_missing_values_as_null(tmp_path) -> None:
    assert preview_num(float("nan")) is None
    assert preview_num("nan") is None
    assert preview_num(None) is None
    assert preview_num("0.1234567") == 0.123457

    path = tmp_path / "preview.jsonl"
    write_jsonl(path, pd.DataFrame([{"score": float("nan"), "nested": {"risk": pd.NA}}]))
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert '"score": null' in text
    assert '"risk": null' in text
