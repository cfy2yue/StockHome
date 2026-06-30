import json

import pandas as pd

from scripts.run_positive_evidence_scorer_experiment import build_rule_outcomes, write_rule_outcomes
from src.agent_training.quant_tool_context import FUTURE_RESULT_FIELDS


def test_positive_evidence_rule_outcomes_do_not_expose_future_fields(tmp_path):
    aggregate = pd.DataFrame(
        [
            {
                "variant": "logistic_kline_peer_only",
                "promotion_status": "observe_prior_positive_latest_weak",
            },
            {
                "variant": "baseline_rev_chip_score",
                "promotion_status": "rejected_or_diagnostic_only",
            },
        ]
    )

    rows = build_rule_outcomes(aggregate)

    assert len(rows) == 1
    assert not _future_keys(rows[0])
    assert rows[0]["usable_in_agent_default"] is False
    assert rows[0]["research_only"] is True
    assert rows[0]["not_investment_instruction"] is True

    output = tmp_path / "positive_rule_outcomes.jsonl"
    write_rule_outcomes(output, rows, append=False)
    loaded = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert loaded == rows


def _future_keys(value):
    if isinstance(value, dict):
        leaked = {str(key) for key in value if str(key) in FUTURE_RESULT_FIELDS}
        for child in value.values():
            leaked.update(_future_keys(child))
        return leaked
    if isinstance(value, list):
        leaked = set()
        for child in value:
            leaked.update(_future_keys(child))
        return leaked
    return set()
