from __future__ import annotations

import pandas as pd

from scripts.analyze_p0_small_entry_bookskill_onoff_results import (
    build_pair_summary,
    build_variant_summary,
)
from scripts.audit_p0_small_entry_pps_q017_channel_interactions import (
    build_interaction_metrics,
    local_verdict,
)


def test_pair_summary_detects_useful_full_agent_raise() -> None:
    detail = pd.DataFrame(
        [
            {
                "variant": "full_agent",
                "date": "2026-01-02",
                "code": "000001",
                "task_mode": "single_stock",
                "valid_block": "H2026_1",
                "sample_panel_id": "p1",
                "research_grade": "继续深挖",
                "simulated_action": "增加研究暴露",
                "simulated_weight_change": 0.6,
                "return_20d": 5.0,
                "cash_adjusted_return_20d": 3.0,
                "active_exposure": True,
                "positive_20d": True,
                "loss_gt5": False,
            },
            {
                "variant": "no_pps_q017",
                "date": "2026-01-02",
                "code": "000001",
                "task_mode": "single_stock",
                "valid_block": "H2026_1",
                "sample_panel_id": "p1",
                "research_grade": "放入观察",
                "simulated_action": "保持观察",
                "simulated_weight_change": 0.1,
                "return_20d": 5.0,
                "cash_adjusted_return_20d": 0.7,
                "active_exposure": False,
                "positive_20d": True,
                "loss_gt5": False,
            },
        ]
    )

    pair = build_pair_summary(detail, ["no_pps_q017"])
    assert pair.iloc[0]["changed_rows"] == 1
    assert pair.iloc[0]["raised_positive"] == 1
    assert pair.iloc[0]["sum_delta_cash20_pp"] > 0


def test_variant_summary_handles_empty_cards_as_not_run() -> None:
    summary = build_variant_summary(pd.DataFrame(), [{"evidence_pack": {"variant": "full_agent"}}])
    assert summary.iloc[0]["status"] == "not_run_or_no_valid_cards"


def test_interaction_metrics_marks_positive_condition_candidate() -> None:
    assert local_verdict(60, 0.66, 5.0, 0.60, 4.0) == "candidate_condition_for_ds_prompt_check"
    assert local_verdict(60, 0.52, 3.0, 0.60, 4.0) == "negative_or_false_filter_risk"
    assert local_verdict(10, 0.90, 10.0, 0.60, 4.0) == "too_sparse_do_not_promote"


def test_interaction_metrics_include_pps_q017_scope() -> None:
    frame = pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "code": "000001",
                "triggered_skill_ids": "PPS-Q-017",
                "return_20d": 3.0,
                "positive_20d": True,
                "loss_gt5": False,
                "has_pps_q017": True,
                "news_available": True,
                "news_low_warning": True,
                "news_positive_or_official": False,
                "financial_event_matched": False,
                "financial_no_recent_event": True,
                "financial_missing": False,
                "peer_breadth_ok": True,
                "peer_relative_positive": True,
                "kline_deep_pullback": False,
                "kline_not_overheated": True,
                "kline_above_ma200": True,
                "chip_low_overhang": True,
                "chip_support_visible": True,
                "all_triggered_grounded_bool": True,
                "weak_skill_present": False,
                "news_or_financial_available": True,
                "peer_and_kline_confirm": True,
                "soft_gap_bundle": False,
            }
        ]
    )
    metrics = build_interaction_metrics(frame)
    assert "pps_q017_all" in set(metrics["scope"])
    assert "news_available" in set(metrics["rule_id"])
