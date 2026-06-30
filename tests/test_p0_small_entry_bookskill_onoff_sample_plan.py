from __future__ import annotations

import pandas as pd

from scripts.build_p0_small_entry_bookskill_onoff_sample_plan import (
    FUTURE_OR_RESULT_FIELDS,
    build_leakage_audit,
    build_sample_plan,
    has_strategy,
    load_exclude_keys,
)


def _detail() -> pd.DataFrame:
    rows = []
    for block in ["H2024_1", "H2025_1", "H2026_1"]:
        for frequency in ["weekly_friday", "every_2_weeks", "weekly_tuesday"]:
            for index in range(4):
                rows.append(
                    {
                        "date": f"2026-0{(index % 3) + 1}-0{(index % 6) + 1}",
                        "code": f"00000{index}",
                        "name": f"测试{index}",
                        "target_block": block,
                        "frequency": frequency,
                        "operation_action": "small_buy_hold",
                        "triggered_skill_ids": "PPS-Q-017;DOW-B-004",
                        "grounded_skill_ids": "PPS-Q-017;DOW-B-004",
                        "weak_skill_ids": "",
                        "book_score": 2.0,
                        "triggered_skill_count": 2,
                        "grounded_skill_count": 2,
                        "weak_skill_count": 0,
                        "return_20d": float(index - 1),
                    }
                )
    return pd.DataFrame(rows)


def test_build_sample_plan_excludes_future_result_columns() -> None:
    plan, pool = build_sample_plan(
        _detail(),
        focus_strategy_id="PPS-Q-017",
        blocks={"H2024_1", "H2025_1", "H2026_1"},
        frequencies={"weekly_friday", "every_2_weeks", "weekly_tuesday"},
        max_per_block_frequency=2,
        max_rows=18,
    )

    assert len(plan) == 18
    assert not (set(plan.columns) & FUTURE_OR_RESULT_FIELDS)
    assert set(plan["task_mode"]) == {"single_stock"}
    assert pool["triggered_skill_ids"].map(lambda value: has_strategy(value, "PPS-Q-017")).all()
    leakage = build_leakage_audit(plan)
    assert bool(leakage.iloc[0]["passes"]) is True


def test_sampling_does_not_depend_on_future_returns() -> None:
    detail = _detail()
    plan_a, _ = build_sample_plan(
        detail,
        focus_strategy_id="PPS-Q-017",
        blocks={"H2024_1", "H2025_1", "H2026_1"},
        frequencies={"weekly_friday", "every_2_weeks", "weekly_tuesday"},
        max_per_block_frequency=1,
        max_rows=9,
    )
    shuffled_outcomes = detail.copy()
    shuffled_outcomes["return_20d"] = list(reversed(shuffled_outcomes["return_20d"].tolist()))
    plan_b, _ = build_sample_plan(
        shuffled_outcomes,
        focus_strategy_id="PPS-Q-017",
        blocks={"H2024_1", "H2025_1", "H2026_1"},
        frequencies={"weekly_friday", "every_2_weeks", "weekly_tuesday"},
        max_per_block_frequency=1,
        max_rows=9,
    )

    keys = ["date", "code", "target_block", "frequency"]
    assert plan_a[keys].to_dict("records") == plan_b[keys].to_dict("records")


def test_has_strategy_uses_exact_semicolon_tokens() -> None:
    assert has_strategy("PPS-Q-017;DOW-B-004", "PPS-Q-017")
    assert not has_strategy("PPS-Q-0179;DOW-B-004", "PPS-Q-017")


def test_build_sample_plan_can_exclude_prior_date_code(tmp_path) -> None:
    detail = _detail()
    exclude_path = tmp_path / "prior_plan.csv"
    pd.DataFrame([{"date": detail.iloc[0]["date"], "code": detail.iloc[0]["code"]}]).to_csv(exclude_path, index=False)
    exclude_keys = load_exclude_keys([exclude_path])

    plan, _ = build_sample_plan(
        detail,
        focus_strategy_id="PPS-Q-017",
        blocks={"H2024_1", "H2025_1", "H2026_1"},
        frequencies={"weekly_friday", "every_2_weeks", "weekly_tuesday"},
        max_per_block_frequency=4,
        max_rows=18,
        exclude_keys=exclude_keys,
    )

    excluded = (pd.to_datetime(detail.iloc[0]["date"]).date().isoformat(), str(detail.iloc[0]["code"]).zfill(6))
    keys = set(zip(plan["date"].astype(str), plan["code"].astype(str).str.zfill(6)))
    assert excluded not in keys
