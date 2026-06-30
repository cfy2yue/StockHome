from __future__ import annotations

import pandas as pd
import pytest

from scripts.audit_p0_small_entry_bookskill_attribution_v1 import (
    assert_no_future_fields,
    build_agent_preview,
    enrich_bookskill,
    rule_promotion_status,
    strategy_verdict,
)


def test_bookskill_gap_is_diagnostic_not_promotion() -> None:
    row = {
        "rule_id": "bookskill_gap_any",
        "selected_rows": 120,
        "selected_rate": 0.8,
        "selected_pos20": 0.7,
        "selected_avg20_pp": 5.0,
        "selected_loss_gt5": 0.1,
        "delta_pos_vs_all": 0.08,
        "delta_avg_vs_all": 1.0,
        "false_veto_positive_rows": 1,
        "captured_loss_gt5_rows": 2,
    }

    assert rule_promotion_status(row) == "risk_or_gap_diagnostic_only"


def test_source_card_history_rule_is_diagnostic_not_promotion() -> None:
    row = {
        "rule_id": "positive_historical_skill_any",
        "selected_rows": 120,
        "selected_rate": 0.8,
        "selected_pos20": 0.7,
        "selected_avg20_pp": 5.0,
        "selected_loss_gt5": 0.1,
        "delta_pos_vs_all": 0.08,
        "delta_avg_vs_all": 1.0,
        "false_veto_positive_rows": 1,
        "captured_loss_gt5_rows": 2,
    }

    assert rule_promotion_status(row) == "source_card_stat_diagnostic_only"


def test_weak_source_strategy_cannot_promote() -> None:
    prior = pd.DataFrame({"return_20d": [2.0] * 50})
    h2026 = pd.DataFrame({"return_20d": [2.0] * 20})

    assert strategy_verdict(prior, h2026, source_status="needs_grounding") == "weak_until_grounded_do_not_promote"


def test_enrich_bookskill_separates_grounded_and_weak_ids() -> None:
    detail = pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "code": "000001",
                "name": "测试",
                "frequency": "weekly_friday",
                "target_block": "H2026_1",
                "operation_action": "small_buy_hold",
                "return_20d": 1.2,
                "positive_20d": True,
                "loss_gt5": False,
            }
        ]
    )
    joined = pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "code": "000001",
                "book_score": 0.6,
                "triggered_skills": "PPS-Q-017;MISSING-X",
                "triggered_formulas": "",
            }
        ]
    )
    cards = {"PPS-Q-017": {"source_status": "grounded", "raw_positive_20d_rate": 0.6}}
    grounding = {"PPS-Q-017": {"policy_status": "mandatory_checklist_not_alpha"}}

    enriched = enrich_bookskill(detail, joined, cards, grounding)
    row = enriched.iloc[0]

    assert row["grounded_skill_count"] == 1
    assert row["weak_skill_count"] == 1
    assert bool(row["bookskill_gap"]) is True
    assert row["positive_historical_skill_count"] == 1


def test_agent_preview_excludes_future_fields() -> None:
    frame = pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "code": "000001",
                "name": "测试",
                "frequency": "weekly_friday",
                "target_block": "H2026_1",
                "operation_action": "small_buy_hold",
                "book_score": 0.5,
                "bookskill_gap": False,
                "triggered_skill_count": 1,
                "grounded_skill_count": 1,
                "weak_skill_count": 0,
                "triggered_skills": "PPS-Q-017",
            }
        ]
    )
    cards = {
        "PPS-Q-017": {
            "source_book": "专业投机原理",
            "source_status": "grounded",
            "page_range": "OCR_PAGE 0098-0118",
            "applicable_condition": "需要多通道确认",
            "failure_condition": "强反证冲突时降权",
        }
    }
    grounding = {"PPS-Q-017": {"policy_status": "mandatory_checklist_not_alpha", "agent_use": "review_only"}}

    previews = build_agent_preview(frame, cards, grounding, max_rows=5)

    assert previews
    assert_no_future_fields(previews)
    with pytest.raises(ValueError):
        assert_no_future_fields({"return_20d": 1.0})
