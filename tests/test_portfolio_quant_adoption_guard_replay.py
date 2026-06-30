import pandas as pd

from scripts.replay_portfolio_quant_adoption_guard import evaluate_policies, make_pair_frame


def test_quant_adoption_guard_replay_caps_only_raised_rows() -> None:
    detail = pd.DataFrame(
        [
            {
                "variant": "full_agent_with_quant_tools",
                "decision_date": "2026-01-01",
                "code": "000001",
                "stratum": "ordinary_control_midkey",
                "return_20d": -10.0,
                "simulated_weight_change": 0.10,
                "quant_tool_adoption_decision": "partially_adopted",
                "quant_tool_override_reasons": "news_gap;financial_gap;bookskill_gap;chip_overhang;data_missing;peer_gap",
            },
            {
                "variant": "full_agent_without_quant_tools",
                "decision_date": "2026-01-01",
                "code": "000001",
                "return_20d": -10.0,
                "simulated_weight_change": 0.05,
            },
            {
                "variant": "full_agent_with_quant_tools",
                "decision_date": "2026-01-02",
                "code": "000002",
                "stratum": "ml_keypoint_top20",
                "return_20d": 20.0,
                "simulated_weight_change": 0.05,
                "quant_tool_adoption_decision": "not_adopted_counter_evidence",
                "quant_tool_override_reasons": "news_gap;financial_gap",
            },
            {
                "variant": "full_agent_without_quant_tools",
                "decision_date": "2026-01-02",
                "code": "000002",
                "return_20d": 20.0,
                "simulated_weight_change": 0.10,
            },
        ]
    )
    pair = make_pair_frame(
        detail,
        treatment="full_agent_with_quant_tools",
        control="full_agent_without_quant_tools",
    )
    summary, replay_detail = evaluate_policies(pair)
    no_guard = summary[summary["policy"].eq("no_guard_treatment")].iloc[0]
    ordinary_cap = summary[summary["policy"].eq("cap_ordinary_raises")].iloc[0]

    assert no_guard["raised_negative"] == 1
    assert no_guard["lowered_positive"] == 1
    assert ordinary_cap["raised_negative"] == 0
    assert ordinary_cap["lowered_positive"] == 1
    assert ordinary_cap["guard_applied_rows"] == 1
    capped = replay_detail[
        replay_detail["policy"].eq("cap_ordinary_raises")
        & replay_detail["code"].astype(str).str.zfill(6).eq("000001")
    ].iloc[0]
    assert capped["replay_weight"] == capped["control_weight"]
