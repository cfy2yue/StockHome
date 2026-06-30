from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.run_news_questionnaire_smoke import _build_questionnaire_pack, _select_rows_from_sample_plan
from src.world_model.news_questionnaire import build_news_questionnaire_messages, load_news_questionnaire


def test_select_rows_from_sample_plan_matches_date_code_and_preserves_order(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {"date": "2025-01-02", "code": "000002", "name": "B"},
            {"date": "2025-01-01", "code": "000001", "name": "A"},
        ]
    )
    plan_path = tmp_path / "plan.csv"
    pd.DataFrame(
        [
            {"date": "2025-01-01", "code": "1"},
            {"date": "2025-01-02", "code": "000002"},
        ]
    ).to_csv(plan_path, index=False)

    selected = _select_rows_from_sample_plan(frame, plan_path)

    assert selected["code"].tolist() == ["000001", "000002"]
    assert selected["name"].tolist() == ["A", "B"]


def test_select_rows_from_sample_plan_rejects_future_result_fields(tmp_path) -> None:
    frame = pd.DataFrame([{"date": "2025-01-01", "code": "000001"}])
    plan_path = tmp_path / "plan.csv"
    pd.DataFrame([{"date": "2025-01-01", "code": "000001", "return_20d": 1.2}]).to_csv(plan_path, index=False)

    with pytest.raises(ValueError, match="future/result"):
        _select_rows_from_sample_plan(frame, plan_path)


def test_questionnaire_pack_prompt_excludes_future_result_fields() -> None:
    config = load_news_questionnaire()
    row = pd.Series(
        {
            "date": "2026-01-06",
            "code": "000001",
            "name": "测试股票",
            "sector_group": "测试行业",
            "prior_return_20d": -6.5,
            "return_20d": 8.8,
            "future_return_20d": 8.8,
            "gt_status": "evaluated",
            "news_count_30d": 1,
            "news_missing_rate": 0,
        }
    )
    events = pd.DataFrame(
        [
            {
                "code": "000001",
                "_available_at_ts": pd.Timestamp("2026-01-05 09:30:00"),
                "available_at": "2026-01-05 09:30:00",
                "source_type": "official_disclosure",
                "source_name": "unit_test",
                "event_type": "announcement",
                "title": "测试公告",
                "content_excerpt": "仅含决策日前可得材料。",
            }
        ]
    )

    pack = _build_questionnaire_pack(
        row,
        events,
        related={},
        master={"000001": {"region": "北京"}},
        config=config,
        window_days=30,
        decision_time="15:00:00",
        max_self_events=2,
        max_peer_events=2,
        max_policy_events=2,
        max_region_events=2,
        max_event_chars=80,
        include_event_url=False,
    )
    messages = build_news_questionnaire_messages(questionnaire_config=config, evidence=pack["evidence"])
    payload = json.loads(messages[1]["content"])
    evidence_keys = _nested_keys(payload["evidence"])

    assert "prior_return_20d" in evidence_keys
    assert "return_20d" not in evidence_keys
    assert "future_return_20d" not in evidence_keys
    assert "gt_status" not in evidence_keys


def _nested_keys(value) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for item in value.values():
            keys.update(_nested_keys(item))
        return keys
    if isinstance(value, list):
        keys = set()
        for item in value:
            keys.update(_nested_keys(item))
        return keys
    return set()
