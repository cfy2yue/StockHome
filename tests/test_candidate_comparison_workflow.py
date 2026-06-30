from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "audit_candidate_comparison_workflow_v1.py"
    spec = importlib.util.spec_from_file_location("audit_candidate_comparison_workflow_v1", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_evaluate_candidate_group_quantifies_ranking_edges() -> None:
    module = _load_script()
    frame = pd.DataFrame(
        [
            {
                "comparison_group_id": "G1",
                "comparison_scenario": "same_sector",
                "repeat_seed": 0,
                "time_block": "H2025_1",
                "date": "2025-01-07",
                "candidate_count": 3,
                "candidate_codes": "000001;000002;000003",
                "candidate_names": "A;B;C",
                "industry_context": "测试行业",
                "code": "000001",
                "name": "A",
                "tushare_industry": "测试行业",
                "tushare_area": "深圳",
                "return_20d": 6.0,
                "candidate_context_blend_v1": 0.9,
                "rev_chip_core": 0.8,
                "original_total_score": 0.1,
                "single_watch_proxy": 0.7,
            },
            {
                "comparison_group_id": "G1",
                "comparison_scenario": "same_sector",
                "repeat_seed": 0,
                "time_block": "H2025_1",
                "date": "2025-01-07",
                "candidate_count": 3,
                "candidate_codes": "000001;000002;000003",
                "candidate_names": "A;B;C",
                "industry_context": "测试行业",
                "code": "000002",
                "name": "B",
                "tushare_industry": "测试行业",
                "tushare_area": "深圳",
                "return_20d": -2.0,
                "candidate_context_blend_v1": 0.1,
                "rev_chip_core": 0.2,
                "original_total_score": 0.9,
                "single_watch_proxy": 0.1,
            },
            {
                "comparison_group_id": "G1",
                "comparison_scenario": "same_sector",
                "repeat_seed": 0,
                "time_block": "H2025_1",
                "date": "2025-01-07",
                "candidate_count": 3,
                "candidate_codes": "000001;000002;000003",
                "candidate_names": "A;B;C",
                "industry_context": "测试行业",
                "code": "000003",
                "name": "C",
                "tushare_industry": "测试行业",
                "tushare_area": "北京",
                "return_20d": -8.0,
                "candidate_context_blend_v1": -0.2,
                "rev_chip_core": -0.1,
                "original_total_score": 0.3,
                "single_watch_proxy": -0.4,
            },
        ]
    )

    detail, aggregate = module.evaluate_groups(frame)

    best = detail[
        detail["score_name"].eq("candidate_context_blend_v1")
        & detail["comparison_group_id"].eq("G1")
    ].iloc[0]
    assert best["top1_code"] == "000001"
    assert best["top1_is_best"]
    assert best["top1_excess_20d"] > 0

    weak = detail[detail["score_name"].eq("original_total_score")].iloc[0]
    assert weak["top1_code"] == "000002"
    assert not weak["top1_is_best"]
    assert set(aggregate["score_name"]) >= {"candidate_context_blend_v1", "equal_or_random_baseline"}


def test_agent_sample_plan_excludes_future_result_fields() -> None:
    module = _load_script()
    frame = pd.DataFrame(
        [
            {
                "comparison_group_id": "G1",
                "comparison_scenario": "cross_sector",
                "repeat_seed": 1,
                "time_block": "H2026_1",
                "date": "2026-01-06",
                "candidate_count": 3,
                "candidate_codes": "000001;000002;000003",
                "candidate_names": "A;B;C",
                "industry_context": "银行;化工;电力",
                "code": "000001",
                "name": "A",
                "return_20d": 5.0,
                "candidate_context_blend_v1": 0.5,
                "rev_chip_core": 0.4,
                "original_total_score": 0.3,
                "single_watch_proxy": 0.2,
            }
        ]
    )

    plan = module.build_agent_sample_plan(frame, max_groups=1)

    forbidden = module.FUTURE_RESULT_COLUMNS & set(plan.columns)
    assert not forbidden
    assert plan.iloc[0]["task_mode"] == "candidate_comparison"
    assert plan.iloc[0]["research_only"]
    assert plan.iloc[0]["not_investment_instruction"]


def test_build_candidate_groups_uses_stable_hash_sampling() -> None:
    module = _load_script()
    rows = []
    industries = ["行业A", "行业B", "行业C", "行业D", "行业E", "行业F"]
    for i in range(18):
        rows.append(
            {
                "date": "2025-01-07",
                "code": str(i + 1).zfill(6),
                "name": f"S{i}",
                "time_block": "H2025_1",
                "tushare_industry": industries[i // 3],
                "tushare_area": "深圳",
                "return_20d": float(i - 5),
                "candidate_context_blend_v1": float(i),
                "rev_chip_core": float(i),
                "original_total_score": float(i),
                "single_watch_proxy": float(i),
            }
        )
    frame = pd.DataFrame(rows)
    cfg = module.AuditConfig(candidate_size=3, repeats=2, industries_per_date=1, decision_frequency="all_dates", min_industry_size=3)

    first = module.build_candidate_groups(frame, cfg)
    second = module.build_candidate_groups(frame, cfg)

    assert not first.empty
    assert first[["comparison_group_id", "code"]].equals(second[["comparison_group_id", "code"]])
    assert set(first["comparison_scenario"]) >= {"same_sector", "cross_sector"}
