import pandas as pd

from scripts.analyze_portfolio_keypoint_flash import add_return_metrics, summarize_pairs


def test_keypoint_pair_summary_counts_negative_and_positive_changes():
    detail = pd.DataFrame(
        [
            {
                "variant": "with",
                "decision_date": "2026-01-01",
                "code": "000001",
                "stratum": "key",
                "return_20d": 10.0,
                "offline_high_impact_label": 1,
                "simulated_weight_change": 0.10,
                "simulated_action": "保持观察",
            },
            {
                "variant": "without",
                "decision_date": "2026-01-01",
                "code": "000001",
                "stratum": "key",
                "return_20d": 10.0,
                "offline_high_impact_label": 1,
                "simulated_weight_change": 0.05,
                "simulated_action": "保持观察",
            },
            {
                "variant": "with",
                "decision_date": "2026-01-02",
                "code": "000002",
                "stratum": "key",
                "return_20d": -8.0,
                "offline_high_impact_label": 1,
                "simulated_weight_change": 0.10,
                "simulated_action": "保持观察",
            },
            {
                "variant": "without",
                "decision_date": "2026-01-02",
                "code": "000002",
                "stratum": "key",
                "return_20d": -8.0,
                "offline_high_impact_label": 1,
                "simulated_weight_change": 0.05,
                "simulated_action": "保持观察",
            },
        ]
    )
    detail = add_return_metrics(detail)
    _, summary = summarize_pairs(detail, "with", "without")
    directions = dict(zip(summary["direction"], summary["rows"]))
    assert directions["raised_positive"] == 1
    assert directions["raised_negative"] == 1
    assert float(summary[summary["direction"].eq("ALL")]["sum_delta_cash"].iloc[0]) == 0.1
