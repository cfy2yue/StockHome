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


def test_build_sample_plan_prefers_unique_codes_per_tier() -> None:
    module = _load_script("build_single_stock_risk_queue_sample_plan")
    queue = pd.DataFrame(
        [
            {"date": "2026-01-01", "code": "000001", "risk_tier": "high", "review_priority_score": 0.99, "risk_score": 0.9, "cap_pct": 0.15},
            {"date": "2026-01-02", "code": "000001", "risk_tier": "high", "review_priority_score": 0.98, "risk_score": 0.8, "cap_pct": 0.15},
            {"date": "2026-01-03", "code": "000002", "risk_tier": "high", "review_priority_score": 0.90, "risk_score": 0.7, "cap_pct": 0.15},
            {"date": "2026-01-04", "code": "000003", "risk_tier": "high", "review_priority_score": 0.80, "risk_score": 0.6, "cap_pct": 0.15},
        ]
    )

    plan = module.build_sample_plan(queue, max_per_tier=3)

    assert len(plan) == 3
    assert plan["code"].nunique() == 3
    assert list(plan["code"]) == ["000001", "000002", "000003"]


def test_exclude_queue_rows_removes_exact_date_code_only() -> None:
    module = _load_script("build_single_stock_risk_queue_sample_plan")
    queue = pd.DataFrame(
        [
            {"date": "2026-01-01", "code": "000001", "risk_tier": "high"},
            {"date": "2026-01-02", "code": "000001", "risk_tier": "high"},
            {"date": "2026-01-01", "code": "000002", "risk_tier": "high"},
        ]
    )
    exclude = pd.DataFrame(
        [
            {"date": "2026-01-01", "code": "000001"},
        ]
    )

    filtered = module.exclude_queue_rows(queue, exclude)

    assert list(zip(filtered["date"], filtered["code"])) == [
        ("2026-01-02", "000001"),
        ("2026-01-01", "000002"),
    ]
