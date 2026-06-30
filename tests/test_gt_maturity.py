from __future__ import annotations

import pandas as pd

from src.agent_training.gt_maturity import MaturityConfig, annotate_gt_maturity, maturity_report


def test_gt_maturity_marks_h2026_ytd_as_provisional() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2025-12-15", "code": "000001", "gt_status": "evaluated", "return_20d": 1.2},
            {"date": "2026-06-18", "code": "000001", "gt_status": "insufficient_future_data", "return_20d": None},
        ]
    )
    result = annotate_gt_maturity(frame, MaturityConfig(current_date="2026-06-25"))
    old = result[result["date"] == "2025-12-15"].iloc[0]
    ytd = result[result["date"] == "2026-06-18"].iloc[0]
    assert old["gt_status_phase2"] == "evaluated"
    assert not bool(old["is_provisional"])
    assert ytd["time_block"] == "H2026_1_YTD"
    assert ytd["gt_status_phase2"] == "gt_pending"
    assert bool(ytd["is_provisional"])
    assert not bool(ytd["is_final_metric_eligible"])


def test_maturity_report_counts_pending_and_evaluated() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2024-01-15", "code": "000001", "gt_status": "evaluated", "return_20d": 1.2},
            {"date": "2024-01-16", "code": "000002", "gt_status": "insufficient_future_data", "return_20d": None},
        ]
    )
    report = maturity_report(frame, MaturityConfig(current_date="2026-06-25"))
    row = report[report["time_block"] == "H2024_1"].iloc[0]
    assert row["evaluated_count"] == 1
    assert row["gt_pending_count"] == 1
    assert row["final_metric_eligible_count"] == 1
