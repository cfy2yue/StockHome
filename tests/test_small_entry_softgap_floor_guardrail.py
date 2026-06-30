from __future__ import annotations

from src.agent_training.evidence_pack import apply_decision_guardrails


def _card() -> dict:
    return {
        "research_grade": "放入观察",
        "simulated_action": "转入现金",
        "simulated_weight_change": 0.0,
        "user_operation_suggestion": "卖出/不买",
        "target_position": 0.0,
        "counter_evidence": "真实行业peer弱且目标落后",
        "final_agent_reasoning_summary": "同行弱，等待。",
        "error_reflection": "",
    }


def _pack(**overrides) -> dict:
    pack = {
        "operation_plan_context": {
            "operation_action": "small_buy_hold",
            "target_position": 0.2,
            "default_position_floor_if_no_hard_counter": 0.1,
        },
        "counter_evidence": "反证分偏低;真实行业peer弱且目标落后",
        "chip_features": {"lower_support": 0.2, "upper_overhang": 0.2},
        "news_features": {"news_warning_score": 0.0},
        "news_semantic_questionnaire": {"ds_news_risk_score": None},
        "financial_report_features": {
            "financial_quality_risk_score": None,
            "financial_surprise_score": None,
            "financial_report_join_status": "no_event_in_window",
        },
        "nonprice_risk_overlay_context": "none",
    }
    pack.update(overrides)
    return pack


def test_small_entry_floor_keeps_clean_chip_peer_weak_as_softgap() -> None:
    card = _card()

    apply_decision_guardrails(card, _pack())

    assert card["user_operation_suggestion"] == "试探买入/持有"
    assert card["target_position"] == 0.2
    assert card["simulated_action"] == "保持观察"
    assert "small_entry_softgap_floor_v1" in card["error_reflection"]


def test_small_entry_floor_does_not_override_financial_hard_counter() -> None:
    card = _card()

    apply_decision_guardrails(
        card,
        _pack(
            counter_evidence="财报质量风险",
            financial_report_features={"financial_quality_risk_score": 0.8, "financial_surprise_score": -0.8},
        ),
    )

    assert card["target_position"] == 0.0
    assert card["user_operation_suggestion"] == "卖出/不买"


def test_small_entry_floor_keeps_peer_hard_when_chip_support_missing() -> None:
    card = _card()

    apply_decision_guardrails(card, _pack(chip_features={"lower_support": 0.0, "upper_overhang": 0.6}))

    assert card["target_position"] == 0.0
    assert card["user_operation_suggestion"] == "卖出/不买"


def test_small_entry_floor_treats_pps_m003_tuesday_peer_weak_as_frequency_softgap() -> None:
    card = _card()
    pack = _pack(chip_features={"lower_support": 0.0, "upper_overhang": 0.6})
    pack["operation_plan_context"]["reason_code"] = "pps_m003_tuesday"
    pack["operation_plan_context"]["target_position"] = 0.25

    apply_decision_guardrails(card, pack)

    assert card["target_position"] == 0.25
    assert card["user_operation_suggestion"] == "试探买入/持有"


def test_action_label_buy_add_floor_keeps_soft_gaps_as_low_position() -> None:
    card = _card()
    pack = _pack(
        operation_plan_context={
            "operation_action": "buy_add",
            "target_position": 0.6,
            "reason_code": "p0_action_label_scorer_v1",
            "default_position_floor_if_no_hard_counter": 0.10,
        },
        counter_evidence="news_missing;financial_no_event_in_window;bookskill_observe_only",
        chip_features={"lower_support": 0.2, "upper_overhang": 0.3},
    )

    apply_decision_guardrails(card, pack)

    assert card["target_position"] == 0.10
    assert card["user_operation_suggestion"] == "试探买入/持有复核"
    assert card["simulated_action"] == "保持观察"
    assert "action_label_buy_add_softgap_floor_v1" in card["error_reflection"]


def test_action_label_buy_add_floor_does_not_override_hard_news_risk() -> None:
    card = _card()
    pack = _pack(
        operation_plan_context={
            "operation_action": "buy_add",
            "target_position": 0.6,
            "reason_code": "p0_action_label_scorer_v1",
            "default_position_floor_if_no_hard_counter": 0.10,
        },
        news_features={"news_warning_score": 0.85},
    )

    apply_decision_guardrails(card, pack)

    assert card["target_position"] == 0.0
    assert card["user_operation_suggestion"] == "卖出/不买"


def test_action_label_wait_branch_caps_position_and_internal_weight() -> None:
    card = {
        **_card(),
        "research_grade": "放入观察",
        "simulated_action": "保持观察",
        "simulated_weight_change": 0.1,
        "user_operation_suggestion": "试探买入/持有复核",
        "target_position": 0.1,
    }
    pack = _pack(
        operation_plan_context={
            "operation_action": "wait",
            "target_position": 0.0,
            "reason_code": "p0_action_label_scorer_v1",
        }
    )

    apply_decision_guardrails(card, pack)

    assert card["target_position"] == 0.0
    assert card["simulated_weight_change"] == 0.0
    assert card["simulated_action"] == "转入现金"
    assert card["user_operation_suggestion"] == "等待不买"
    assert "action_label_wait_reduce_cap_v1" in card["error_reflection"]


def test_action_label_reduce_branch_allows_only_tiny_review_weight() -> None:
    card = {
        **_card(),
        "research_grade": "继续深挖",
        "simulated_action": "增加研究暴露",
        "simulated_weight_change": 0.25,
        "user_operation_suggestion": "试探买入",
        "target_position": 0.35,
    }
    pack = _pack(
        operation_plan_context={
            "operation_action": "reduce_review",
            "target_position": 0.1,
            "reason_code": "p0_action_label_scorer_v1",
        }
    )

    apply_decision_guardrails(card, pack)

    assert card["target_position"] == 0.10
    assert card["simulated_weight_change"] == 0.05
    assert card["simulated_action"] == "降低研究暴露"
    assert card["user_operation_suggestion"] == "减仓/卖出复核"
