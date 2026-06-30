from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.audit_rolling_confirmation_preflight import (
    audit_p0_dryrun_evidence,
    audit_p0_sample_plan,
    audit_p1_preflight,
    forbidden_key_paths,
)


def test_forbidden_key_scan_allows_prior_and_kline_returns_but_rejects_future_result() -> None:
    safe = {
        "python_features": {"prior_return_20d": -3.2},
        "kline_features": {"kline_return_20d": 1.5},
    }
    unsafe = {"nested": {"return_20d": 1.0}}

    assert forbidden_key_paths(safe) == []
    assert forbidden_key_paths(unsafe) == ["nested.return_20d"]


def test_p0_latest_sample_plan_gate_passes_safe_plan(tmp_path: Path) -> None:
    plan = pd.DataFrame(
        [
            {
                "date": "2026-05-22",
                "code": f"{index:06d}",
                "valid_block": "H2026_1",
                "research_only": True,
                "not_investment_instruction": True,
            }
            for index in range(20)
        ]
    )
    coverage = pd.DataFrame(
        [
            {"stratum": f"s{index}", "selected_rows": 4}
            for index in range(5)
        ]
    )
    plan_path = tmp_path / "plan.csv"
    coverage_path = tmp_path / "coverage.csv"
    plan.to_csv(plan_path, index=False)
    coverage.to_csv(coverage_path, index=False)

    row = audit_p0_sample_plan(plan_path, coverage_path, min_stockdates=20)

    assert row["status"] == "pass"
    assert "future_cols=[]" in row["evidence"]


def test_p0_dryrun_evidence_gate_detects_exact_future_keys(tmp_path: Path) -> None:
    plan = pd.DataFrame(
        [{"date": "2026-05-22", "code": f"{index:06d}"} for index in range(2)]
    )
    plan_path = tmp_path / "plan.csv"
    evidence_path = tmp_path / "evidence.jsonl"
    invalid_path = tmp_path / "invalid.jsonl"
    plan.to_csv(plan_path, index=False)
    evidence_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "variant": variant,
                    "research_only": True,
                    "not_investment_instruction": True,
                    "python_features": {"prior_return_20d": -1.0},
                    "kline_features": {"kline_return_20d": 2.0},
                },
                ensure_ascii=False,
            )
            for variant in ["full", "no_news"]
            for _ in range(2)
        ),
        encoding="utf-8",
    )
    invalid_path.write_text("", encoding="utf-8")

    row = audit_p0_dryrun_evidence(plan_path, evidence_path, invalid_path)

    assert row["status"] == "pass"
    assert "future_hits=0" in row["evidence"]

    evidence_path.write_text(
        json.dumps(
            {
                "variant": "full",
                "research_only": True,
                "not_investment_instruction": True,
                "return_20d": 1.0,
            }
        ),
        encoding="utf-8",
    )

    failed = audit_p0_dryrun_evidence(plan_path, evidence_path, invalid_path)
    assert failed["status"] == "incomplete"
    assert "future_hits=1" in failed["evidence"]


def test_p1_preflight_allows_cross_sector_only(tmp_path: Path) -> None:
    gate_summary = pd.DataFrame(
        [
            {
                "decision_frequency": "every_2_weeks",
                "comparison_scenario": "cross_sector",
                "score_name": "p1_default_selector_v1",
                "candidate_for_ds_panel": True,
            },
            {
                "decision_frequency": "weekly_friday",
                "comparison_scenario": "cross_sector",
                "score_name": "rank_avg_rev_watch",
                "candidate_for_ds_panel": True,
            },
            {
                "decision_frequency": "weekly_friday",
                "comparison_scenario": "same_sector",
                "score_name": "rank_avg_rev_watch",
                "candidate_for_ds_panel": False,
            },
        ]
    )
    panel_metrics = pd.DataFrame(
        [
            {
                "time_block": "H2026_1",
                "comparison_scenario": "cross_sector",
                "score_name": "p1_default_selector_v1",
                "decision_frequency": "every_2_weeks",
                "top2_excess_mean": 1.2,
                "mean_rank_ic": 0.08,
            }
        ]
    )
    gate_path = tmp_path / "gate.csv"
    panel_path = tmp_path / "panel.csv"
    gate_summary.to_csv(gate_path, index=False)
    panel_metrics.to_csv(panel_path, index=False)

    row = audit_p1_preflight(gate_path, panel_path, min_cross_sector_candidates=2)

    assert row["status"] == "pass_cross_sector_only"
    assert "cross_sector_candidates=2" in row["evidence"]
