from __future__ import annotations

import scripts.audit_tool_adoption_contract as audit


def test_p0_operation_plan_override_requires_hard_counter() -> None:
    pack = {
        "operation_plan_context": {
            "tool_id": "local_user_operation_plan_context_v1",
            "operation_action": "small_buy_hold",
            "target_position": 0.2,
            "default_position_floor_if_no_hard_counter": 0.1,
        },
        "counter_evidence": "新闻空窗，财报近窗口无事件",
    }
    card = {
        "target_position": 0,
        "user_operation_suggestion": "等待不买",
        "counter_evidence": "新闻空窗，财报近窗口无事件",
        "final_agent_reasoning_summary": "软缺口导致等待",
    }

    assert audit.p0_operation_status(card, pack["operation_plan_context"]) == "overridden_to_zero_or_wait"
    assert not audit.contains_hard_counter(card, pack)

    pack["counter_evidence"] = "财务质量风险高且同行显著弱"
    card["counter_evidence"] = "财务质量风险高且同行显著弱"
    assert audit.contains_hard_counter(card, pack)


def test_p1_anchor_change_without_hard_counter_is_detectable() -> None:
    pack = {
        "default_ranked_candidates": [
            {"default_rank": 1, "code": "000001"},
            {"default_rank": 2, "code": "000002"},
            {"default_rank": 3, "code": "000003"},
        ]
    }
    card = {
        "top_research_codes": ["000003", "000001"],
        "rank_override_audit": "none",
    }

    assert audit.default_top_codes(pack, n=2) == ["000001", "000002"]
    assert audit.agent_top_codes(card, n=2) == ["000003", "000001"]
    assert not audit.contains_hard_counter(card, pack)

    card["rank_override_audit"] = "000002 新闻预警>=0.6，构成明确硬反证"
    assert audit.contains_hard_counter(card, pack)


def test_build_summary_requires_no_failures() -> None:
    detail = audit.pd.DataFrame(
        [
            {
                "task": "P0",
                "status": "pass",
                "operation_tool_id": "local_user_operation_plan_context_v1",
                "issues": "",
                "forbidden_eval_hits": "",
            },
            {
                "task": "P1",
                "status": "pass",
                "default_top2": "000001;000002",
                "issues": "",
                "forbidden_eval_hits": "",
            },
        ]
    )

    summary = audit.build_summary(detail)

    assert summary.loc[0, "status"] == "pass"
    assert int(summary.loc[0, "p0_operation_tool_rows"]) == 1
    assert int(summary.loc[0, "p1_anchor_rows"]) == 1

    detail.loc[1, "status"] = "fail"
    detail.loc[1, "issues"] = "ranker_anchor_changed_without_structured_hard_counter"
    summary = audit.build_summary(detail)
    assert summary.loc[0, "status"] == "incomplete"
