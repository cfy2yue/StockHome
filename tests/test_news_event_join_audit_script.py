from __future__ import annotations

import pandas as pd

from scripts.audit_news_event_feature_join import evidence_pack_smoke, status_count_frame, summarize_join


def test_join_audit_summary_and_smoke_use_joined_news_features() -> None:
    frame = pd.DataFrame(
        [
            {
                "date": "2025-04-25",
                "code": "000089",
                "name": "测试A",
                "news_event_table_join_status": "event_window_matched",
                "event_count": 4,
                "self_news_intensity": 0.8,
                "news_warning_score": 0.3333,
                "news_opportunity_score": 0.0,
                "announcement_materiality_score": 0.7,
                "news_missing_rate": 0.0,
            },
            {
                "date": "2025-04-25",
                "code": "000090",
                "name": "测试B",
                "news_event_table_join_status": "no_event_in_window",
                "news_missing_rate": 1.0,
            },
        ]
    )

    summary = summarize_join(frame, event_table_rows=10, event_feature_rows=2)
    statuses = status_count_frame(frame)
    smoke = evidence_pack_smoke(frame)

    assert summary["gt_rows"] == 2
    assert summary["matched_rows"] == 1
    assert summary["event_table_rows"] == 10
    assert summary["news_missing_rate_mean"] == 0.5
    assert set(statuses["news_event_table_join_status"]) == {"event_window_matched", "no_event_in_window"}
    assert smoke["code"] == "000089"
    assert smoke["news_warning_score"] == 0.3333
    assert smoke["news_missing_rate"] == 0.0
