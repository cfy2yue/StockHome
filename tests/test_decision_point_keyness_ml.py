import pandas as pd

from scripts.audit_decision_point_keyness_ml import build_rule_outcomes, high_impact_label, label_thresholds
from src.agent_training.quant_tool_context import FUTURE_RESULT_FIELDS


def test_high_impact_label_uses_train_thresholds():
    train = pd.DataFrame(
        {
            "future_return_std_20d": [1.0, 2.0, 3.0, 10.0],
            "rev_chip_top5_pool_excess_20d": [0.1, 0.2, 0.3, 4.0],
        }
    )
    valid = pd.DataFrame(
        {
            "future_return_std_20d": [1.0, 9.0],
            "rev_chip_top5_pool_excess_20d": [0.1, 0.2],
        }
    )
    threshold = label_thresholds(train, 20)
    labels = high_impact_label(valid, 20, threshold)
    assert labels.tolist() == [0, 1]


def test_rule_outcomes_are_sanitized_for_agent_use():
    aggregate = pd.DataFrame(
        [
            {
                "task_mode": "portfolio_pool",
                "horizon": "20d",
                "h2026_ml_capture_rate": 0.2,
                "h2026_ml_precision": 0.8,
                "promotion_status": "observe_training_sampler_candidate",
            }
        ]
    )
    importance = pd.DataFrame(
        [
            {
                "task_mode": "portfolio_pool",
                "horizon": "20d",
                "feature": "news_conflict_pressure",
                "mean_abs_coef": 0.7,
            }
        ]
    )
    outcomes = build_rule_outcomes(aggregate, importance, 0.2)
    assert outcomes
    assert set(outcomes[0]).isdisjoint(FUTURE_RESULT_FIELDS)
    assert outcomes[0]["research_only"] is True
    assert outcomes[0]["not_investment_instruction"] is True
