from __future__ import annotations

import pandas as pd

from scripts.audit_p0_action_position_learner_v1 import (
    add_action_bins,
    apply_position_profile,
    fit_position_profile,
)


def _rows() -> pd.DataFrame:
    rows = []
    for i in range(120):
        rows.append(
            {
                "date": "2026-01-02",
                "code": f"{i:06d}",
                "time_block": "H2026_1",
                "return_20d": 8.0 if i < 80 else -3.0,
                "opp_active": True,
                "opp_strong": i < 60,
                "kline_active": True,
                "kline_hard_risk": False,
                "risk_review_queue": False,
                "risk_queue_high_hard_counter": False,
                "channel_hard_counter_prob": 0.0,
                "channel_positive_support_prob": 0.5,
            }
        )
    for i in range(120, 180):
        rows.append(
            {
                "date": "2026-01-02",
                "code": f"{i:06d}",
                "time_block": "H2026_1",
                "return_20d": -8.0,
                "opp_active": False,
                "opp_strong": False,
                "kline_active": False,
                "kline_hard_risk": True,
                "risk_review_queue": True,
                "risk_queue_high_hard_counter": True,
                "channel_hard_counter_prob": 0.98,
                "channel_positive_support_prob": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_add_action_bins_separates_confirmed_and_hard_counter() -> None:
    out = add_action_bins(_rows())

    assert "opp_strong_kline_clean" in set(out["action_bin"])
    assert "hard_counter" in set(out["action_bin"])


def test_fit_position_profile_assigns_high_confirmed_and_zero_hard_counter() -> None:
    profile = fit_position_profile(_rows(), "learned_balanced_v1", min_bin_rows=10)

    assert profile["bins"]["opp_strong_kline_clean"]["position"] >= 0.35
    assert profile["bins"]["hard_counter"]["position"] == 0.0


def test_apply_position_profile_caps_review_and_hard_risk() -> None:
    profile = fit_position_profile(_rows(), "learned_balanced_v1", min_bin_rows=10)
    target = _rows().copy()
    target.loc[0, "risk_review_queue"] = True
    target.loc[1, "kline_hard_risk"] = True

    out = apply_position_profile(target, profile, policy_name="learned_balanced_v1")

    assert out.loc[0, "target_position"] <= 0.20
    assert out.loc[1, "target_position"] == 0.0
    assert "cash_adjusted_return_20d" in out.columns


def test_delta_guard_does_not_raise_negative_delta_bin() -> None:
    rows = _rows()
    rows.loc[rows["opp_strong"], "return_20d"] = -10.0

    profile = fit_position_profile(rows, "learned_delta_guard_v1", min_bin_rows=10)

    assert profile["bins"]["opp_strong_kline_clean"]["delta_avg"] < 0
    assert profile["bins"]["opp_strong_kline_clean"]["position"] <= 0.10
