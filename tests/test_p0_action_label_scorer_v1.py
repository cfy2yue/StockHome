from __future__ import annotations

import pandas as pd

from scripts.audit_p0_action_label_scorer_v1 import (
    apply_action_policy,
    balanced_preview_sample,
    build_agent_preview_rows,
    choose_threshold_profile,
    make_action_labels,
)


def _rows() -> pd.DataFrame:
    rows = []
    for i in range(120):
        rows.append(
            {
                "date": "2026-01-02",
                "code": f"{i:06d}",
                "time_block": "H2026_1",
                "return_20d": 8.0 if i < 70 else (-7.0 if i < 95 else 0.5),
                "entry_prob": 0.90 if i < 70 else 0.25,
                "strong_entry_prob": 0.80 if i < 50 else 0.20,
                "reduce_prob": 0.15 if i < 70 else (0.92 if i < 95 else 0.35),
            }
        )
    frame = pd.DataFrame(rows)
    frame["action_edge_score"] = frame["entry_prob"] + 0.55 * frame["strong_entry_prob"] - 0.90 * frame["reduce_prob"]
    return frame


def test_make_action_labels_separates_entry_and_reduce() -> None:
    labels = make_action_labels(_rows())

    assert labels["entry_label"].sum() > 0
    assert labels["strong_entry_label"].sum() > 0
    assert labels["reduce_label"].sum() > 0
    assert labels.loc[0, "entry_label"] == 1
    assert labels.loc[80, "reduce_label"] == 1


def test_action_policy_reduces_high_risk_and_activates_entry() -> None:
    frame = _rows()
    profile = choose_threshold_profile(frame, "balanced_action_v1")
    out = apply_action_policy(frame, profile)

    assert out.loc[0, "target_position"] >= 0.35
    assert out.loc[80, "target_position"] == 0.0
    assert out.loc[0, "operation_hint"] in {"trial_buy_or_add_review", "small_buy_or_hold_review"}
    assert out.loc[80, "operation_hint"] == "reduce_or_avoid_review"


def test_balanced_preview_includes_high_and_low_signal_without_future_fields() -> None:
    frame = _rows()
    profile = choose_threshold_profile(frame, "balanced_action_v1")
    out = apply_action_policy(frame, profile)
    out["policy_name"] = "balanced_action_v1"
    sample = balanced_preview_sample(out, max_rows=40)

    assert len(sample) <= 40
    assert sample["target_position"].max() >= 0.35
    assert sample["target_position"].min() <= 0.05

    preview = build_agent_preview_rows(out, {"frequency": "every_2_weeks", "feature_group": "wide_safe", "model": "hgb", "policy_name": "balanced_action_v1"}, max_rows=40, mode="balanced")
    rendered = str(preview)
    assert "return_20d" not in rendered
    assert "entry_label" not in rendered
