from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.backtest.adaptation import build_adaptation_skills
from src.backtest.case_memory import build_case_memory, explain_similarity, find_similar_cases


def test_build_adaptation_skills_from_candidate_rules(tmp_path: Path):
    rules = {
        "rules": [
            {
                "status": "candidate",
                "derived_from": {"strategy_id": "PPS-Q-019", "book": "专业投机原理"},
                "formula": "relative_strength_rank >= 0.67",
                "thresholds": {"top_percentile_min": 0.67},
                "applies_to": {"sector_group": ["star_technology"], "cadence": "daily"},
                "evidence": {"test": {"trigger_count": 10}},
                "anti_leakage_checks": {"ground_truth_excluded_from_scoring": True},
                "reuse_instruction": "候选研究规则",
            },
            {"status": "do_not_reuse", "derived_from": {"strategy_id": "PPS-Q-023"}},
        ]
    }
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(yaml.safe_dump(rules, allow_unicode=True), encoding="utf-8")
    out = tmp_path / "adaptation.yaml"
    adaptations = build_adaptation_skills(rules_path, out)
    assert out.exists()
    assert len(adaptations) == 1
    assert adaptations[0]["source_skill"] == "PPS-Q-019"
    assert "500股" in adaptations[0]["next_validation"]


def test_case_memory_similarity(tmp_path: Path):
    gt = pd.DataFrame(
        [
            {
                "code": "600888",
                "name": "A",
                "date": "2025-09-01",
                "sector_group": "nonferrous_materials",
                "rating": "放入观察",
                "triggered_skills": "PPS-Q-017;PPS-Q-019",
                "prior_return_20d": 12,
                "relative_strength_rank": 0.8,
                "total_score": 6,
                "trend_score": 7,
                "book_score": 6,
                "counter_score": 8,
                "completeness_score": 5,
                "return_5d": 2,
                "return_10d": 4,
                "return_20d": 9,
                "gt_pass": True,
            },
            {
                "code": "688001",
                "name": "B",
                "date": "2025-09-01",
                "sector_group": "star_technology",
                "rating": "暂时剔除",
                "triggered_skills": "DOW-B-004",
                "prior_return_20d": -10,
                "relative_strength_rank": 0.2,
                "total_score": 3,
                "trend_score": 2,
                "book_score": 3,
                "counter_score": 4,
                "completeness_score": 5,
                "return_5d": -3,
                "return_10d": -4,
                "return_20d": -8,
                "gt_pass": True,
            },
        ]
    )
    path = tmp_path / "gt.csv"
    gt.to_csv(path, index=False)
    cases = build_case_memory(path, tmp_path / "cases.csv")
    similar = find_similar_cases(cases, gt.iloc[0].to_dict(), top_n=1)
    explanation = explain_similarity(similar.iloc[0], gt.iloc[0].to_dict())
    assert similar.iloc[0]["code"] == "600888"
    assert "PPS-Q-019" in explanation["shared_skills"]
