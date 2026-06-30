from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_top_pct_per_date_caps_each_date() -> None:
    module = _load_script("audit_single_stock_risk_calibration_v2")
    frame = pd.DataFrame(
        [
            {"date": "2026-01-06", "code": f"{idx:06d}", "score": idx}
            for idx in range(10)
        ]
        + [
            {"date": "2026-01-07", "code": f"{idx + 20:06d}", "score": idx}
            for idx in range(5)
        ]
    )

    selected = module.select_top_pct_per_date(frame, "score", 0.20)

    counts = selected.groupby("date")["code"].count().to_dict()
    assert counts == {"2026-01-06": 2, "2026-01-07": 1}
    assert set(selected[selected["date"] == "2026-01-06"]["score"]) == {8, 9}
    assert set(selected[selected["date"] == "2026-01-07"]["score"]) == {4}


def test_review_queue_rejects_future_fields_and_uses_allowed_grades() -> None:
    module = _load_script("audit_single_stock_risk_calibration_v2")
    selected = pd.DataFrame(
        [
            {
                "date": "2026-01-06",
                "code": "000001",
                "time_block": "H2026_1",
                "risk_score": 0.8,
                "review_priority_score": 1.1,
                "channel_hard_counter_prob": 0.97,
                "channel_soft_gap_prob": 0.01,
                "channel_positive_support_prob": 0.01,
                "channel_neutral_prob": 0.01,
                "channel_score_coverage": 1.0,
                "return_20d": -9.0,
                "single_stock_label": "reduce_or_exclude",
            },
            {
                "date": "2026-01-07",
                "code": "000002",
                "time_block": "H2026_1",
                "risk_score": 0.4,
                "review_priority_score": 0.6,
                "channel_hard_counter_prob": 0.85,
                "channel_soft_gap_prob": 0.04,
                "channel_positive_support_prob": 0.02,
                "channel_neutral_prob": 0.09,
                "channel_score_coverage": 1.0,
            },
        ]
    )

    queue = module.build_review_queue(selected, cap_pct=0.10, policy_status="review_only")

    assert set(queue["research_grade"]) == {"暂时剔除", "放入观察"}
    assert "return_20d" not in queue.columns
    assert "single_stock_label" not in queue.columns
    assert "买入" not in queue.to_string()
    assert "卖出" not in queue.to_string()


def test_review_queue_leak_guard_raises_on_result_field() -> None:
    module = _load_script("audit_single_stock_risk_calibration_v2")
    bad = pd.DataFrame([{"research_grade": "放入观察", "return_20d": 1.2}])

    try:
        module._reject_queue_leak(bad)
    except ValueError as exc:
        assert "future/result" in str(exc)
    else:
        raise AssertionError("expected future/result field rejection")
