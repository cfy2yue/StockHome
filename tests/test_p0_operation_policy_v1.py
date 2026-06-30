from __future__ import annotations

import pandas as pd
import pytest

from scripts.audit_p0_operation_policy_v1 import (
    assert_no_future_fields,
    operation_action,
    operation_threshold,
    with_operation_actions,
)


def test_operation_action_mapping() -> None:
    buy = pd.Series(
        {
            "target_position": 0.6,
            "risk_review_queue": False,
            "risk_queue_high_hard_counter": False,
            "kline_hard_risk": False,
        }
    )
    assert operation_action(buy) == "buy_add"

    review = buy.copy()
    review["target_position"] = 0.05
    review["risk_review_queue"] = True
    assert operation_action(review) == "reduce_review"

    hard = buy.copy()
    hard["target_position"] = 0.6
    hard["risk_queue_high_hard_counter"] = True
    assert operation_action(hard) == "reduce_sell"

    wait = buy.copy()
    wait["target_position"] = 0.05
    assert operation_action(wait) == "wait"


def test_with_operation_actions_adds_clear_cn_and_threshold() -> None:
    frame = pd.DataFrame(
        [
            {
                "target_position": 0.6,
                "risk_review_queue": False,
                "risk_queue_high_hard_counter": False,
                "kline_hard_risk": False,
                "opp_score": 0.8,
                "opp_threshold": 0.3,
                "kline_opp_score": 0.7,
                "kline_opp_threshold": 0.4,
                "kline_risk_score": 0.1,
                "kline_risk_threshold": 0.5,
            }
        ]
    )
    out = with_operation_actions(frame)
    assert out.loc[0, "operation_action"] == "buy_add"
    assert out.loc[0, "operation_action_cn"] == "买入/加仓"
    assert "opp_score>=0.3000" in out.loc[0, "operation_threshold"]
    assert "kline_score>=0.4000" in operation_threshold(out.iloc[0])


def test_safe_preview_rejects_future_keys() -> None:
    assert_no_future_fields({"operation_action": "buy_add", "score": 0.1})
    with pytest.raises(ValueError):
        assert_no_future_fields({"future_return_20d": 1.2})
