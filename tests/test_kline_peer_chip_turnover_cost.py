import json

import pandas as pd

from scripts.audit_kline_peer_chip_turnover_cost import (
    build_rule_outcomes,
    portfolio_daily_metrics,
    turnover_one_way,
    turnover_promotion_status,
    write_rule_outcomes,
)


def test_turnover_one_way_overlap_cases():
    assert turnover_one_way(set(), set()) == 0.0
    assert turnover_one_way(set(), {"000001"}) == 1.0
    assert turnover_one_way({"000001", "000002"}, {"000001", "000002"}) == 0.0
    assert turnover_one_way({"000001", "000002"}, {"000002", "000003"}) == 0.5
    assert turnover_one_way({"000001", "000002"}, {"000003", "000004"}) == 1.0


def test_portfolio_daily_metrics_uses_actual_rebalance_overlap_cost():
    frame = pd.DataFrame(
        [
            {"date": "2026-01-06", "code": "000001", "valid_block": "H2026_1", "task_mode": "portfolio_pool", "return_20d": 4.0, "score": 0.9},
            {"date": "2026-01-06", "code": "000002", "valid_block": "H2026_1", "task_mode": "portfolio_pool", "return_20d": 2.0, "score": 0.8},
            {"date": "2026-01-06", "code": "000003", "valid_block": "H2026_1", "task_mode": "portfolio_pool", "return_20d": -2.0, "score": 0.1},
            {"date": "2026-01-13", "code": "000001", "valid_block": "H2026_1", "task_mode": "portfolio_pool", "return_20d": 3.0, "score": 0.7},
            {"date": "2026-01-13", "code": "000002", "valid_block": "H2026_1", "task_mode": "portfolio_pool", "return_20d": 1.0, "score": 0.2},
            {"date": "2026-01-13", "code": "000003", "valid_block": "H2026_1", "task_mode": "portfolio_pool", "return_20d": 5.0, "score": 0.95},
        ]
    )

    daily = portfolio_daily_metrics(
        frame,
        variant="score",
        top_pct=2 / 3,
        source_run="weekly_tuesday",
        round_trip_cost_pct=1.5,
    )

    assert list(daily["selected_codes"]) == ["000001;000002", "000001;000003"]
    assert list(daily["turnover_one_way"]) == [1.0, 0.5]
    assert list(daily["estimated_cost_pct"]) == [1.5, 0.75]


def test_promotion_status_requires_prior_and_h2026_net_positive():
    prior = pd.DataFrame(
        {
            "net_pool_excess_after_turnover_cost": [0.2, 0.4],
            "portfolio_positive_20d": [True, True],
            "rank_ic": [0.04, 0.05],
        }
    )
    h2026 = pd.DataFrame(
        {
            "net_pool_excess_after_turnover_cost": [0.3],
            "portfolio_positive_20d": [True],
            "rank_ic": [0.04],
        }
    )

    assert turnover_promotion_status(prior, h2026, 0.2) == "accepted_cost_recheck_candidate"
    assert turnover_promotion_status(prior, h2026, 0.4) == "observe_h2026_positive_prior_weak"


def test_turnover_cost_rule_outcomes_are_agent_safe(tmp_path):
    aggregate = pd.DataFrame(
        [
            {
                "variant": "logistic_kline_peer_chip",
                "top_pct": 0.05,
                "decision_frequency": "every_2_weeks",
                "promotion_status": "accepted_cost_recheck_candidate",
            },
            {
                "variant": "baseline_rev_chip_score",
                "top_pct": 0.05,
                "decision_frequency": "every_2_weeks",
                "promotion_status": "rejected_or_diagnostic_only",
            },
        ]
    )

    rows = build_rule_outcomes(aggregate)

    assert len(rows) == 1
    assert rows[0]["usable_in_agent_default"] is True
    assert rows[0]["task_mode"] == "portfolio_pool"
    payload = json.loads(json.dumps(rows[0], ensure_ascii=False))
    assert "return_20d" not in payload
    assert "rank_ic" not in payload
    assert "return_20d" not in payload["top_features"]
    assert "rank_ic" not in payload["top_features"]

    output = tmp_path / "outcomes.jsonl"
    write_rule_outcomes(output, rows)
    assert "accepted_cost_recheck_candidate" in output.read_text(encoding="utf-8")
