from src.world_model.single_stock_news_vector import (
    SINGLE_STOCK_NEWS_VECTOR_DIMENSIONS,
    build_single_stock_news_query_plan,
    vectorize_single_stock_news,
)


def test_single_stock_news_vector_filters_by_available_at():
    events = [
        {
            "event_title": "样本公司收到监管处罚",
            "available_at": "2026-01-10 14:30:00",
            "entity_scope": "self",
            "event_type": "regulation",
            "direction_hint": "negative",
            "materiality_score": 8,
            "evidence_level": "official_disclosure",
            "actionability": "verify",
        },
        {
            "event_title": "同行公司签订重大合同",
            "available_at": "2026-01-10 16:30:00",
            "entity_scope": "peer",
            "event_type": "contract",
            "direction_hint": "positive",
            "materiality_score": 6,
        },
        {
            "event_title": "行业政策补贴发布",
            "available_at": "2026-01-09",
            "entity_scope": "sector",
            "event_type": "policy",
            "direction_hint": "positive",
            "materiality_score": 6,
            "time_safety": "approximate",
        },
    ]

    vector = vectorize_single_stock_news(events, "2026-01-10 15:00:00")

    assert len(SINGLE_STOCK_NEWS_VECTOR_DIMENSIONS) >= 48
    assert vector["single_news_count_30d"] == 2
    assert vector["single_news_self_count_30d"] == 1
    assert vector["single_news_peer_count_30d"] == 0
    assert vector["single_news_sector_count_30d"] == 1
    assert vector["single_news_negative_materiality_30d"] == 8
    assert vector["single_news_regulation_score_30d"] == -8
    assert vector["single_news_action_verify_count_30d"] == 1
    assert vector["single_news_time_approx_count_30d"] == 1


def test_single_stock_news_query_plan_is_time_safe_and_scoped():
    plan = build_single_stock_news_query_plan(
        "600000",
        "浦发银行",
        peers=["招商银行", "平安银行"],
        sector_keywords=["银行", "净息差"],
        as_of_time="2026-01-10 15:00:00",
    )

    scopes = {item["entity_scope"] for item in plan}
    assert {"self", "peer", "sector", "upstream", "downstream", "macro"}.issubset(scopes)
    assert all(item["policy"] == "filter available_at <= as_of_time before vectorization" for item in plan)
    assert any("招商银行" in item["query"] for item in plan)
