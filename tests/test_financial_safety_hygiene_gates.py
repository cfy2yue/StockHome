from __future__ import annotations

import pandas as pd

from scripts.audit_financial_safety_hygiene_gates import (
    VALID_BLOCKS,
    aggregate_gate_metrics,
    evaluate_gate_variants,
    load_detail_cache,
)


def _detail_frame() -> pd.DataFrame:
    rows = []
    for block_index, block in enumerate(VALID_BLOCKS):
        for date_index in range(10):
            date = f"2026-01-{date_index + 1:02d}" if block == "H2026_1" else f"2025-01-{date_index + 1:02d}"
            rows.extend(
                [
                    {
                        "date": date,
                        "code": f"{block_index}{date_index}01",
                        "time_block": block,
                        "window_days": 90,
                        "scope": "high_ranker_q0.80",
                        "return_20d": 4.0,
                        "financial_high_risk_guard": False,
                        "financial_quality_low_risk": True,
                        "financial_positive_surprise_low_risk": False,
                    },
                    {
                        "date": date,
                        "code": f"{block_index}{date_index}02",
                        "time_block": block,
                        "window_days": 90,
                        "scope": "high_ranker_q0.80",
                        "return_20d": 6.0,
                        "financial_high_risk_guard": False,
                        "financial_quality_low_risk": False,
                        "financial_positive_surprise_low_risk": False,
                    },
                    {
                        "date": date,
                        "code": f"{block_index}{date_index}03",
                        "time_block": block,
                        "window_days": 90,
                        "scope": "high_ranker_q0.80",
                        "return_20d": -10.0,
                        "financial_high_risk_guard": True,
                        "financial_quality_low_risk": False,
                        "financial_positive_surprise_low_risk": False,
                    },
                ]
            )
    return pd.DataFrame(rows)


def test_high_risk_subset_is_review_no_raise_candidate() -> None:
    metrics = evaluate_gate_variants(_detail_frame())
    aggregate = aggregate_gate_metrics(metrics)

    high_risk = aggregate[(aggregate["gate_id"] == "high_risk_subset") & (aggregate["window_days"] == 90)].iloc[0]
    baseline = aggregate[(aggregate["gate_id"] == "baseline") & (aggregate["window_days"] == 90)].iloc[0]

    assert baseline["policy_status"] == "baseline_reference"
    assert high_risk["policy_status"] == "accepted_review_no_raise_candidate"
    assert high_risk["prior_avg_return_lift"] < 0
    assert high_risk["h2026_avg_return_lift"] < 0
    assert high_risk["prior_loss_gt5_lift"] > 0
    assert high_risk["h2026_loss_gt5_lift"] > 0


def test_empty_selection_preserves_time_block_label() -> None:
    metrics = evaluate_gate_variants(_detail_frame())
    empty_surprise = metrics[
        (metrics["gate_id"] == "require_positive_surprise")
        & (metrics["valid_block"] == "H2026_1")
        & (metrics["selected_rows"] == 0)
    ]

    assert len(empty_surprise) == 1


def test_load_detail_cache_filters_scope_and_window_and_normalizes_code(tmp_path) -> None:
    cache_path = tmp_path / "detail.csv.gz"
    pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "code": "1",
                "time_block": "H2026_1",
                "window_days": 90,
                "scope": "high_ranker_q0.80",
                "return_20d": 1.0,
            },
            {
                "date": "2026-01-02",
                "code": "2",
                "time_block": "H2026_1",
                "window_days": 30,
                "scope": "high_ranker_q0.80",
                "return_20d": 2.0,
            },
            {
                "date": "2026-01-02",
                "code": "3",
                "time_block": "H2026_1",
                "window_days": 90,
                "scope": "all_pool",
                "return_20d": 3.0,
            },
        ]
    ).to_csv(cache_path, index=False)

    loaded = load_detail_cache(cache_path, windows=[90], scope="high_ranker_q0.80")

    assert loaded["code"].tolist() == ["000001"]
    assert loaded["window_days"].tolist() == [90]
    assert loaded["scope"].tolist() == ["high_ranker_q0.80"]
