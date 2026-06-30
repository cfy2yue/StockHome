from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.build_news_hq_positive_fresh_sample_plan import (
    assert_no_future_plan_columns,
    build_fresh_sample_plan,
    load_excluded_stockdates,
)


def _candidate(date: str, code: str, block: str, score: float, quality: float = 0.9) -> dict[str, object]:
    return {
        "date": date,
        "code": code,
        "name": f"S{code}",
        "time_block": block,
        "rev_chip_score_quantile": score,
        "news_high_quality_positive": True,
        "news_missing_rate": 0.0,
        "news_opportunity_score": 0.6,
        "news_evidence_quality": quality,
        "official_confirmation_score": 0.95,
        "news_warning_score": 0.0,
        "prior_return_20d": 8.0,
        "rsi14": 55.0,
        "relative_strength_rank": 0.75,
        "peer_relative_to_group_20d": 1.0,
        "peer_group_positive_breadth_20d": 0.6,
        "financial_report_event_count": 1,
        "financial_quality_risk_score": 0.1,
        "financial_surprise_score": 0.2,
        "triggered_skills": "PPS-Q-017",
        "return_20d": 99.0,
        "gt_status": "evaluated",
    }


def test_build_fresh_sample_plan_excludes_prior_stockdates_and_future_columns() -> None:
    frame = pd.DataFrame(
        [
            _candidate("2023-07-14", "000001", "H2023_2", 0.99),
            _candidate("2023-07-21", "000002", "H2023_2", 0.95),
            _candidate("2024-03-19", "000003", "H2024_1", 0.98),
            _candidate("2024-03-26", "000004", "H2024_1", 0.97),
        ]
    )

    plan, audit = build_fresh_sample_plan(
        frame,
        excluded_stockdates={("2023-07-14", "000001")},
        stockdates_per_block=1,
        blocks=["H2023_2", "H2024_1"],
        task_modes=["portfolio_pool", "single_stock"],
        sample_panel_id="unit_fresh_v1",
    )

    assert len(plan) == 4
    assert set(plan["task_mode"]) == {"portfolio_pool", "single_stock"}
    assert ("2023-07-14", "000001") not in set(zip(plan["date"], plan["code"]))
    assert ("2023-07-21", "000002") in set(zip(plan["date"], plan["code"]))
    ds_visible_text = " ".join(
        plan[["sample_panel_id", "stratum", "sampler_context"]].astype(str).agg(" ".join, axis=1).tolist()
    )
    assert "news_high_quality_positive" not in ds_visible_text
    assert "return_20d" not in plan.columns
    assert "gt_status" not in plan.columns
    assert "return_20d" not in audit.columns
    assert "gt_status" not in audit.columns
    assert_no_future_plan_columns(plan)
    assert_no_future_plan_columns(audit)


def test_load_excluded_stockdates_reads_date_code_once(tmp_path: Path) -> None:
    path = tmp_path / "sample_plan.csv"
    pd.DataFrame(
        [
            {"date": "2025-03-28", "code": "631", "task_mode": "portfolio_pool"},
            {"date": "2025-03-28", "code": "000631", "task_mode": "single_stock"},
        ]
    ).to_csv(path, index=False)

    keys = load_excluded_stockdates([path])

    assert keys == {("2025-03-28", "000631")}


def test_assert_no_future_plan_columns_rejects_result_columns() -> None:
    frame = pd.DataFrame([{"date": "2026-01-01", "code": "000001", "return_20d": 1.0}])

    try:
        assert_no_future_plan_columns(frame)
    except ValueError as exc:
        assert "return_20d" in str(exc)
    else:
        raise AssertionError("future/result column was not rejected")
