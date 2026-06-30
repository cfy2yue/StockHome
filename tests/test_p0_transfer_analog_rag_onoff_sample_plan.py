from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_p0_transfer_analog_rag_onoff_sample_plan import (
    GREEN_STATUS,
    build_leakage_audit,
    build_sample_plan,
    green_rule_keys,
    load_preview,
    load_summary,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_build_transfer_analog_rag_onoff_plan_filters_green_rules_without_future_fields(tmp_path: Path) -> None:
    preview_path = tmp_path / "preview.jsonl"
    summary_path = tmp_path / "summary.csv"
    green_key = {
        "frequency": "every_2_weeks",
        "variant": "opportunity_context_rows__stack_plus_all_channels__logistic_l1_c005__top30",
        "analog_id": "analog_k15_min10",
        "gate_id": "chip_support_plus_analog065",
    }
    rows = []
    for idx in range(5):
        rows.append(
            {
                "date": f"2026-04-{idx + 1:02d}",
                "code": f"00000{idx}",
                "time_block": "H2026_1",
                "tool_id": "p0_transfer_analog_rag_v1",
                **green_key,
                "base_branch": "branch_stack_v1.small_buy_hold",
                "operation_action_cn": "小仓试探/持有",
                "position_cap_hint": 0.1,
                "transfer_score": 0.7,
                "transfer_threshold": 0.6,
                "analog_neighbor_count": 15,
                "analog_pos_rate": 0.8,
                "analog_avg_return": 6.0,
                "analog_historical_tail_risk_rate": 0.05,
                "analog_top_case_refs": "2025-01-01:000001:0.1",
                "channel_support_count": 3,
                "channel_hard_counter_count": 0,
                "news_low_warning": True,
                "financial_no_recent_event": True,
                "chip_support_visible": True,
                "agent_instruction": "use as checklist only",
                "auto_trade": False,
            }
        )
    rows.append({**rows[0], "code": "300001", "gate_id": "analog_pos_ge060"})
    _write_jsonl(preview_path, rows)
    pd.DataFrame(
        [
            {**green_key, "promotion_status": GREEN_STATUS, "h2026_selected_pos20": 0.82},
            {**green_key, "gate_id": "analog_pos_ge060", "promotion_status": "observe_diagnostic_only"},
        ]
    ).to_csv(summary_path, index=False)

    preview = load_preview(preview_path)
    summary = load_summary(summary_path)
    green = green_rule_keys(summary, promotion_status=GREEN_STATUS)
    plan, pool, filtered_preview = build_sample_plan(preview, green, max_rows=3, max_per_green_rule=3, seed="unit")
    leakage = build_leakage_audit(plan, filtered_preview)

    assert len(plan) == 3
    assert set(plan["task_mode"]) == {"single_stock"}
    assert set(plan["gate_id"]) == {"chip_support_plus_analog065"}
    assert len(pool) == 5
    assert len(filtered_preview) == 3
    assert leakage["passes"].all()
    assert "return_20d" not in plan.to_json(force_ascii=False)
    assert "return_20d" not in filtered_preview.to_json(force_ascii=False)


def test_transfer_analog_rag_preview_rejects_future_fields(tmp_path: Path) -> None:
    preview_path = tmp_path / "bad_preview.jsonl"
    _write_jsonl(preview_path, [{"date": "2026-04-01", "code": "000001", "return_20d": 5.0}])

    with pytest.raises(ValueError, match="return_20d"):
        load_preview(preview_path)
