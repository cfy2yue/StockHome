from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "audit_candidate_comparison_stability_v1.py"
    spec = importlib.util.spec_from_file_location("audit_candidate_comparison_stability_v1", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_paired_score_contrasts_compare_same_groups_against_default() -> None:
    module = _load_script()
    detail = pd.DataFrame(
        [
            {
                "decision_frequency": "every_2_weeks",
                "comparison_group_id": "G1",
                "comparison_scenario": "same_sector",
                "time_block": "H2026_1",
                "sample_panel_id": "panel_01",
                "score_name": "p1_default_selector_v1",
                "rank_ic": 0.1,
                "top1_excess_20d": 1.0,
                "top2_excess_20d": 0.5,
                "top1_positive": True,
                "top1_is_worst": False,
                "top1_regret_vs_best": 2.0,
            },
            {
                "decision_frequency": "every_2_weeks",
                "comparison_group_id": "G1",
                "comparison_scenario": "same_sector",
                "time_block": "H2026_1",
                "sample_panel_id": "panel_01",
                "score_name": "rev_chip_core",
                "rank_ic": 0.2,
                "top1_excess_20d": 3.0,
                "top2_excess_20d": 1.5,
                "top1_positive": True,
                "top1_is_worst": False,
                "top1_regret_vs_best": 0.5,
            },
        ]
    )

    out = module.paired_score_contrasts(detail, baseline="p1_default_selector_v1")

    row = out[out["time_block"].eq("ALL")].iloc[0]
    assert row["score_name"] == "rev_chip_core"
    assert row["delta_top1_excess_mean"] == 2.0
    assert row["delta_top2_excess_mean"] == 1.0
    assert row["delta_regret_mean"] == -1.5
    assert row["beats_baseline_top1_rate"] == 1.0


def test_gate_summary_flags_stable_candidate_and_ambiguous_gap() -> None:
    module = _load_script()
    panel_metrics = pd.DataFrame(
        [
            {
                "decision_frequency": "weekly_friday",
                "comparison_scenario": "same_sector",
                "score_name": "rev_chip_core",
                "time_block": "ALL",
                "panels": 3,
                "n_groups": 30,
                "mean_rank_ic": 0.05,
                "rank_ic_positive_rate": 0.6,
                "top1_excess_mean": 1.2,
                "top1_excess_panel_std": 0.2,
                "top1_excess_panel_min": 0.8,
                "top2_excess_mean": 0.7,
                "top2_excess_panel_std": 0.1,
                "top2_excess_panel_min": 0.2,
                "top1_positive_rate": 0.55,
                "top2_positive_rate": 0.54,
                "top1_worst_rate": 0.1,
                "regret_mean": 9.0,
            }
        ]
    )
    score_contrasts = pd.DataFrame(
        [
            {
                "decision_frequency": "weekly_friday",
                "comparison_scenario": "same_sector",
                "score_name": "rev_chip_core",
                "time_block": "ALL",
                "delta_top1_excess_mean": 0.0,
                "delta_top2_excess_mean": 0.0,
                "beats_baseline_top1_rate": 0.5,
                "beats_baseline_top2_rate": 0.5,
            }
        ]
    )
    gap = pd.DataFrame(
        [
            {
                "decision_frequency": "weekly_friday",
                "comparison_scenario": "same_sector",
                "score_name": "rev_chip_core",
                "score_gap_top1_top2": 0.1,
                "score_gap_abs_lt_0_25": True,
            }
        ]
    )

    out = module.build_gate_summary(panel_metrics, score_contrasts, gap)

    assert not bool(out.iloc[0]["candidate_for_ds_panel"])
    assert "rank_gap_ambiguous" in out.iloc[0]["stability_note"]

