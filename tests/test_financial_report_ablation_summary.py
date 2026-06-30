from __future__ import annotations

import pandas as pd

from scripts.summarize_financial_report_ablation import build_rule_variant_summary, build_variant_delta


def test_rule_summary_uses_evidence_pack_candidate_rule() -> None:
    evidence_rows = [
        {
            "variant": "financial_report_only",
            "task_mode": "portfolio_pool",
            "valid_block": "H2026_1",
            "decision_date": "2026-01-09",
            "code": "1309",
            "python_signal_summary": "candidate=sample_plan_rule=financial_nonpositive_surprise_news_available_v1; task_mode=portfolio_pool",
        }
    ]
    decision_rows = [
        {
            "variant": "financial_report_only",
            "task_mode": "portfolio_pool",
            "valid_block": "H2026_1",
            "decision_date": "2026-01-09",
            "code": "001309",
            "research_grade": "放入观察",
            "simulated_weight_change": 0.1,
        }
    ]

    summary = build_rule_variant_summary(evidence_rows, decision_rows)

    assert summary.loc[0, "candidate_rule"] == "financial_nonpositive_surprise_news_available_v1"
    assert summary.loc[0, "watch_cards"] == 1


def test_variant_delta_uses_no_financial_baseline() -> None:
    metrics = pd.DataFrame(
        [
            {
                "variant": "no_financial_report_channel",
                "task_mode": "portfolio_pool",
                "cash_adjusted_avg_return_20d": 0.2,
            },
            {
                "variant": "news_plus_financial_report_guarded",
                "task_mode": "portfolio_pool",
                "cash_adjusted_avg_return_20d": 0.45,
            },
        ]
    )

    delta = build_variant_delta(metrics)

    assert delta.loc[0, "variant"] == "news_plus_financial_report_guarded"
    assert round(delta.loc[0, "delta_vs_no_financial_report_channel"], 4) == 0.25
