from __future__ import annotations

import json

import pandas as pd

from scripts.build_p0_action_label_tool_sample_plan_v1 import build_sample_plan, load_preview


def _row(index: int, *, hint: str = "trial_buy_or_add_review", date: str = "2026-03-17") -> dict:
    return {
        "date": date,
        "code": f"{index:06d}",
        "name": f"stock{index}",
        "time_block": "H2026_1",
        "tool_id": "p0_action_label_scorer_v1",
        "frequency": "every_2_weeks",
        "feature_group": "wide_safe",
        "model": "hgb",
        "policy_name": "precision_entry_v1",
        "entry_prob": 0.7 - index * 0.001,
        "strong_entry_prob": 0.6,
        "reduce_prob": 0.1,
        "action_edge_score": 0.8 - index * 0.001,
        "entry_threshold": 0.3,
        "reduce_threshold": 0.6,
        "target_position": 0.6 if hint == "trial_buy_or_add_review" else 0.4,
        "operation_hint": hint,
        "tool_interpretation": "multi_action_label_tool",
        "source_ref_ids": "joined_ground_truth_combined_news_asof_cache;p0_action_label_scorer_v1",
        "research_only": True,
        "not_investment_instruction": True,
    }


def test_build_action_label_sample_plan_is_safe_and_balanced(tmp_path) -> None:
    path = tmp_path / "preview.jsonl"
    high_dates = ["2026-03-17", "2026-03-20", "2026-03-31"]
    small_dates = ["2026-04-03", "2026-04-14"]
    rows = [_row(i, date=high_dates[i % len(high_dates)]) for i in range(6)]
    rows.extend(_row(i + 10, hint="small_buy_or_hold_review", date=small_dates[i % len(small_dates)]) for i in range(6))
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    preview = load_preview(path)
    plan, audit = build_sample_plan(
        preview,
        policy_name="precision_entry_v1",
        max_rows=6,
        max_per_date=2,
        include_small_hold=2,
    )

    assert len(plan) == 6
    assert len(audit) == 6
    assert set(plan["task_mode"]) == {"single_stock"}
    assert "return_20d" not in plan.columns
    assert plan["date"].value_counts().max() <= 2
    assert {"试探买入/加仓", "试探买入/持有"} <= set(plan["operation_action_cn"])


def test_action_label_sample_plan_rejects_future_fields(tmp_path) -> None:
    path = tmp_path / "bad_preview.jsonl"
    bad = _row(1)
    bad["return_20d"] = 9.0
    path.write_text(json.dumps(bad, ensure_ascii=False) + "\n", encoding="utf-8")

    try:
        load_preview(path)
    except ValueError as exc:
        assert "future/result" in str(exc)
    else:
        raise AssertionError("expected future-field rejection")
