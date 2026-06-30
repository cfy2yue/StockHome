from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.audit_latest_rolling_product_risk import (
    build_gate_table,
    summarize_p0_latest,
    summarize_p1_latest,
    summarize_preflight,
)


def test_p0_zero_exposure_is_not_latest_confirmation(tmp_path: Path) -> None:
    metrics_path = tmp_path / "p0_metrics.csv"
    pd.DataFrame(
        [
            {
                "decision_cards": 6,
                "invalid_outputs": 0,
                "exposure_cards": 0,
                "cash_adjusted_positive_20d_rate": 0.5,
                "cash_adjusted_avg_return_20d": -0.2,
                "active_exposure": 0.0667,
                "data_missing_flag_cards": 4,
            }
        ]
    ).to_csv(metrics_path, index=False)

    summary = summarize_p0_latest(metrics_path)

    assert summary["status"] == "not_confirmed_zero_exposure"
    assert summary["exposure_cards"] == 0


def test_p1_tiny_positive_smoke_is_partial_not_confirmation(tmp_path: Path) -> None:
    metrics_path = tmp_path / "p1_metrics.csv"
    pd.DataFrame(
        [
            {
                "top1_excess_20d": 1.0,
                "top2_excess_20d": 2.0,
                "top1_positive": True,
                "top1_is_worst": False,
                "agent_top1_matches_default_top1": True,
                "agent_top2_overlap_default_top2": 1.0,
            },
            {
                "top1_excess_20d": 2.0,
                "top2_excess_20d": 3.0,
                "top1_positive": False,
                "top1_is_worst": False,
                "agent_top1_matches_default_top1": True,
                "agent_top2_overlap_default_top2": 1.0,
            },
            {
                "top1_excess_20d": 3.0,
                "top2_excess_20d": 4.0,
                "top1_positive": False,
                "top1_is_worst": False,
                "agent_top1_matches_default_top1": True,
                "agent_top2_overlap_default_top2": 1.0,
            },
        ]
    ).to_csv(metrics_path, index=False)

    summary = summarize_p1_latest(metrics_path)

    assert summary["status"] == "partial_sorting_smoke_not_confirmation"
    assert summary["cards"] == 3


def test_risk_register_overall_logs_not_complete(tmp_path: Path) -> None:
    preflight_path = tmp_path / "gates.csv"
    pd.DataFrame(
        [
            {"gate": "P0_latest_sample_plan", "status": "pass"},
            {"gate": "P0_latest_dryrun_evidence", "status": "pass"},
            {"gate": "P1_rolling_newdata_preflight", "status": "pass_cross_sector_only"},
            {"gate": "rolling_confirmation_next_step", "status": "ready_for_bounded_flash"},
        ]
    ).to_csv(preflight_path, index=False)
    p0 = {
        "status": "not_confirmed_zero_exposure",
        "cards": 6,
        "invalid_outputs": 0,
        "exposure_cards": 0,
        "cash_pos20": 0.5,
        "cash_avg20": -0.2,
        "active_exposure": 0.0667,
        "data_missing_cards": 4,
        "reason": "zero exposure",
    }
    p1 = {
        "status": "partial_sorting_smoke_not_confirmation",
        "cards": 3,
        "top1_excess": 6.0,
        "top2_excess": 5.0,
        "top1_positive": 0.333,
        "top1_worst": 0.0,
        "anchor_match": 1.0,
        "top2_overlap": 1.0,
        "reason": "tiny sample",
    }
    preflight = summarize_preflight(preflight_path)

    gates = build_gate_table(p0, p1, preflight)

    overall = gates[gates["gate"].eq("latest_rolling_product_risk_register")].iloc[0]
    assert overall["status"] == "logged_not_complete"
    assert "p0_confirmed=False" in overall["evidence"]
